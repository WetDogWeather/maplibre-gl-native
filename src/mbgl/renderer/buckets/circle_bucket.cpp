#include <mbgl/renderer/buckets/circle_bucket.hpp>
#include <mbgl/renderer/bucket_parameters.hpp>
#include <mbgl/style/layers/circle_layer_impl.hpp>
#include <mbgl/renderer/layers/render_circle_layer.hpp>
#include <mbgl/util/constants.hpp>
#include <mbgl/util/math.hpp>

namespace mbgl {

using namespace style;

CircleBucket::CircleBucket(const std::map<std::string, Immutable<LayerProperties>>& layerPaintProperties,
                           const MapMode mode_,
                           const float zoom)
    : mode(mode_) {
    for (const auto& pair : layerPaintProperties) {
        paintPropertyBinders.emplace(std::piecewise_construct,
                                     std::forward_as_tuple(pair.first),
                                     std::forward_as_tuple(getEvaluated<CircleLayerProperties>(pair.second), zoom));
    }
}

CircleBucket::~CircleBucket() {
    sharedVertices->release();
}

void CircleBucket::upload([[maybe_unused]] gfx::UploadPass& uploadPass) {
    uploaded = true;
}

bool CircleBucket::hasData() const {
    return !segments.empty();
}

namespace {
template <class Property>
float get(const CirclePaintProperties::PossiblyEvaluated& evaluated,
          const std::string& id,
          const std::map<std::string, CircleBinders>& paintPropertyBinders) {
    const auto it = paintPropertyBinders.find(id);
    if (it == paintPropertyBinders.end() || !it->second.statistics<Property>().max()) {
        return evaluated.get<Property>().constantOr(Property::defaultValue());
    } else {
        return *it->second.statistics<Property>().max();
    }
}
} // namespace

float CircleBucket::getQueryRadius(const RenderLayer& layer) const {
    const auto& evaluated = getEvaluated<CircleLayerProperties>(layer.evaluatedProperties);
    float radius = get<CircleRadius>(evaluated, layer.getID(), paintPropertyBinders);
    float stroke = get<CircleStrokeWidth>(evaluated, layer.getID(), paintPropertyBinders);
    auto translate = evaluated.get<CircleTranslate>();
    return radius + stroke + util::length(translate[0], translate[1]);
}

void CircleBucket::update(const FeatureStates& states,
                          const GeometryTileLayer& layer,
                          const std::string& layerID,
                          const ImagePositions& imagePositions) {
    auto it = paintPropertyBinders.find(layerID);
    if (it != paintPropertyBinders.end()) {
        it->second.updateVertexVectors(states, layer, imagePositions);
        uploaded = false;

        sharedVertices->updateModified();
    }
}

namespace {
bool isWithinExtent(const GeometryCoordinate& p) noexcept {
    return 0 <= p.x && p.x < util::EXTENT && 0 <= p.y && p.y < util::EXTENT;
}

// Exclude points outside the tile extent in Continuous mode.  In other modes, include
// all points so that circles from neighbouring tiles are not clipped at tile boundaries.
std::size_t countValidPoints(const GeometryCollection& geoms, MapMode mode) {
    auto countPoints = (mode == MapMode::Continuous) ?
        [](const GeometryCoordinates& geom) { return static_cast<std::size_t>(std::ranges::count_if(geom, isWithinExtent)); } :
        [](const GeometryCoordinates& geom){ return geom.size(); };
    auto addGeom = [f = std::move(countPoints)](std::size_t acc, const auto& geoms) {
        return acc + f(geoms);
    };
    return std::accumulate(geoms.begin(), geoms.end(), std::size_t{0}, std::move(addGeom));
}
} // namespace

void CircleBucket::addCircle(const GeometryTileFeature& feature,
                             const GeometryCollection& geometry,
                             std::size_t featureIndex,
                             float sortKey,
                             const CanonicalTileID& canonical) {
    const auto validPointCount = countValidPoints(geometry, mode);
    if (validPointCount == 0) {
        return;
    }

    // We only need to build the instanced geometry once
    // TODO: Don't we need a separate segment for each `sortKey`?  But it wasn't before...
    if (segments.empty()) {
        constexpr const uint16_t vertexLength = 4;
        constexpr const uint16_t indexLength = 6;

        VertexVector& vertices = *sharedVertices;
        TriangleIndexVector& triangles = *sharedTriangles;

        // this geometry will be of the Point type, and we'll derive two triangles from it.
        // 3┌─┐2
        //  │/│
        // 0└─┘1
        vertices.emplace_back(CircleBucket::vertex({0, 0}, -1, -1));
        vertices.emplace_back(CircleBucket::vertex({0, 0}, 1, -1));
        vertices.emplace_back(CircleBucket::vertex({0, 0}, 1, 1));
        vertices.emplace_back(CircleBucket::vertex({0, 0}, -1, 1));

        triangles.emplace_back(0, 1, 2);
        triangles.emplace_back(0, 3, 2);

        segments.emplace_back(0, 0, vertexLength, indexLength, sortKey);
    }

    // Generate one set of vertex attributes for each point in the feature.
    // The shader will apply it to each vertex.
    for (auto& pair : paintPropertyBinders) {
        pair.second.populateVertexVectors(feature, validPointCount, featureIndex, {}, {}, canonical);
    }
}

} // namespace mbgl
