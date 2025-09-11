#pragma once
#include <mbgl/geometry/feature_index.hpp>
#include <mbgl/layout/layout.hpp>
#include <mbgl/renderer/bucket_parameters.hpp>
#include <mbgl/renderer/buckets/circle_bucket.hpp>
#include <mbgl/renderer/render_layer.hpp>
#include <mbgl/style/layers/circle_layer_impl.hpp>
#include <mbgl/util/containers.hpp>

namespace mbgl {

class CircleLayout final : public Layout {
public:
    CircleLayout(const BucketParameters& parameters,
                 const std::vector<Immutable<style::LayerProperties>>& group,
                 std::unique_ptr<GeometryTileLayer> sourceLayer_)
        : sourceLayer(std::move(sourceLayer_)),
          zoom(parameters.tileID.overscaledZ),
          mode(parameters.mode) {
        assert(!group.empty());
        auto leaderLayerProperties = staticImmutableCast<style::CircleLayerProperties>(group.front());
        const auto& unevaluatedLayout = leaderLayerProperties->layerImpl().layout;
        const bool sortFeaturesByKey = !unevaluatedLayout.get<style::CircleSortKey>().isUndefined();
        const auto& layout = unevaluatedLayout.evaluate(PropertyEvaluationParameters(zoom));
        sourceLayerID = leaderLayerProperties->layerImpl().sourceLayer;
        bucketLeaderID = leaderLayerProperties->layerImpl().id;

        for (const auto& layerProperties : group) {
            const std::string& layerId = layerProperties->baseImpl->id;
            layerPropertiesMap.emplace(layerId, layerProperties);
        }

        const size_t featureCount = sourceLayer->featureCount();
        for (size_t i = 0; i < featureCount; ++i) {
            auto feature = sourceLayer->getFeature(i);
            if (!leaderLayerProperties->layerImpl().filter(style::expression::EvaluationContext(zoom, feature.get())
                                                               .withCanonicalTileID(&parameters.tileID.canonical))) {
                continue;
            }

            if (!sortFeaturesByKey) {
                features.push_back({i, std::move(feature), style::CircleSortKey::defaultValue()});
                continue;
            }

            const auto& sortKeyProperty = layout.template get<style::CircleSortKey>();
            float sortKey = sortKeyProperty.evaluate(*feature, zoom, style::CircleSortKey::defaultValue());
            CircleFeature circleFeature{.i = i, .feature = std::move(feature), .sortKey = sortKey};
            const auto sortPosition = std::ranges::lower_bound(features, circleFeature, std::less<>());
            features.insert(sortPosition, std::move(circleFeature));
        }
    }

    bool hasDependencies() const override { return false; }

    void createBucket(const ImagePositions&,
                      std::unique_ptr<FeatureIndex>& featureIndex,
                      mbgl::unordered_map<std::string, LayerRenderData>& renderData,
                      const bool,
                      const bool,
                      const CanonicalTileID& canonical) override {
        auto bucket = std::make_shared<CircleBucket>(layerPropertiesMap, mode, zoom);

        for (auto& circleFeature : features) {
            const auto i = circleFeature.i;
            const auto& feature = *circleFeature.feature;
            const GeometryCollection& geometries = feature.getGeometries();

            bucket->addCircle(feature, geometries, i, circleFeature.sortKey, canonical);
            featureIndex->insert(geometries, i, sourceLayerID, bucketLeaderID);
        }

        if (bucket->hasData()) {
            for (const auto& pair : layerPropertiesMap) {
                renderData.emplace(pair.first, LayerRenderData{.bucket = bucket, .layerProperties = pair.second});
            }
        }
    }

private:
    struct CircleFeature {
        friend bool operator<(const CircleFeature& lhs, const CircleFeature& rhs) noexcept {
            return lhs.sortKey < rhs.sortKey;
        }

        size_t i;
        std::unique_ptr<GeometryTileFeature> feature;
        float sortKey;
    };

    std::map<std::string, Immutable<style::LayerProperties>> layerPropertiesMap;
    std::string bucketLeaderID;

    const std::unique_ptr<GeometryTileLayer> sourceLayer;
    std::list<CircleFeature> features;

    const float zoom;
    const MapMode mode;
    std::string sourceLayerID;
};

} // namespace mbgl
