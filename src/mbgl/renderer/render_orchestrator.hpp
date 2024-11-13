#pragma once
#if MLN_DRAWABLE_RENDERER
#include <mbgl/renderer/layer_group.hpp>
#endif
#include <mbgl/actor/scheduler.hpp>
#include <mbgl/renderer/renderer.hpp>
#include <mbgl/renderer/render_source_observer.hpp>
#include <mbgl/renderer/render_light.hpp>
#include <mbgl/style/image.hpp>
#include <mbgl/style/source.hpp>
#include <mbgl/style/layer.hpp>
#include <mbgl/map/transform_state.hpp>
#include <mbgl/map/zoom_history.hpp>
#include <mbgl/text/cross_tile_symbol_index.hpp>
#include <mbgl/text/glyph_manager_observer.hpp>
#include <mbgl/renderer/image_manager_observer.hpp>
#include <mbgl/text/placement.hpp>
#include <mbgl/renderer/render_tree.hpp>
#include <mbgl/util/std.hpp>

#include <map>
#include <memory>
#include <ranges>
#include <string>
#include <unordered_map>
#include <vector>

namespace mbgl {
#if MLN_DRAWABLE_RENDERER
class ChangeRequest;
#endif
class RendererObserver;
class RenderSource;
class UpdateParameters;
class RenderStaticData;
class RenderedQueryOptions;
class SourceQueryOptions;
class GlyphManager;
class ImageManager;
class LineAtlas;
class PatternAtlas;
class CrossTileSymbolIndex;
class RenderTree;

namespace gfx {
class ShaderRegistry;
#if MLN_DRAWABLE_RENDERER
class Drawable;
using DrawablePtr = std::shared_ptr<Drawable>;
#endif
} // namespace gfx

namespace style {
class LayerProperties;
} // namespace style

using ImmutableLayer = Immutable<style::Layer::Impl>;

class RenderOrchestrator final : public GlyphManagerObserver, public ImageManagerObserver, public RenderSourceObserver {
public:
    RenderOrchestrator(bool backgroundLayerAsColor_,
                       TaggedScheduler& threadPool_,
                       Scheduler* renderThreadPool_,
                       const std::optional<std::string>& localFontFamily_);
    ~RenderOrchestrator() override;

    void markContextLost() { contextLost = true; };
    // TODO: Introduce RenderOrchestratorObserver.
    void setObserver(RendererObserver*);

    std::unique_ptr<RenderTree> createRenderTree(const std::shared_ptr<UpdateParameters>&);

    std::vector<Feature> queryRenderedFeatures(const ScreenLineString&, const RenderedQueryOptions&) const;
    std::vector<Feature> querySourceFeatures(const std::string& sourceID, const SourceQueryOptions&) const;
    std::vector<Feature> queryShapeAnnotations(const ScreenLineString&) const;

    FeatureExtensionValue queryFeatureExtensions(const std::string& sourceID,
                                                 const Feature& feature,
                                                 const std::string& extension,
                                                 const std::string& extensionField,
                                                 const std::optional<std::map<std::string, Value>>& args) const;

    void setFeatureState(const std::string& sourceID,
                         const std::optional<std::string>& layerID,
                         const std::string& featureID,
                         const FeatureState& state);

    void getFeatureState(FeatureState& state,
                         const std::string& sourceID,
                         const std::optional<std::string>& layerID,
                         const std::string& featureID) const;

    void removeFeatureState(const std::string& sourceID,
                            const std::optional<std::string>& sourceLayerID,
                            const std::optional<std::string>& featureID,
                            const std::optional<std::string>& stateKey);

    void setTileCacheEnabled(bool);
    bool getTileCacheEnabled() const;
    void reduceMemoryUse();
    void dumpDebugLogs();
    void collectPlacedSymbolData(bool);
    const std::vector<PlacedSymbolData>& getPlacedSymbolsData() const;
    void clearData();

    void update(const std::shared_ptr<UpdateParameters>&);

#if MLN_DRAWABLE_RENDERER
    bool addLayerGroup(LayerGroupBasePtr);
    bool removeLayerGroup(const LayerGroupBasePtr&);
    size_t numLayerGroups() const noexcept;
    void updateLayerIndex(LayerGroupBasePtr, int32_t newIndex);

    template <typename Func>
        requires requires(Func f, LayerGroupBase& layerGroup, std::size_t layerIndex) {
            { f(layerGroup, layerIndex) } -> std::same_as<void>;
        }
    void visitLayerGroups(bool reversed, Func f) {
        eachGroup(reversed, [&, i = 0_uz](auto& group) mutable { f(group, i++); });
    }

    template <typename Func>
    void visitLayerGroups(Func f) {
        visitLayerGroups(/*reversed=*/false, f);
    }

    template <typename Func>
    void visitLayerGroupsReversed(Func f) {
        visitLayerGroups(/*reversed=*/true, f);
    }

    /// @brief Run a function for each layer, using multiple threads
    /// @param scheduler The scheduler on which to schedule tasks
    /// @param reversed True to run the tasks in order of decreasing layer index
    /// @param f The function to execute
    ///
    /// The layer index passed to the given function is sequential, not the index
    /// used as a sort key internally, and does not match `getLayerIndex()`.
    ///
    /// Each I of N threads handles the Ith 1/N of the items (as opposed to I%N) so that when the results
    /// are concatenated, all the encoded commands appear in the same order as the layer indexes.
    template <typename Func>
        requires requires(Func f,
                          LayerGroupBase& layerGroup,
                          std::optional<std::size_t> threadIndex,
                          std::size_t layerIndex) {
            { f(layerGroup, threadIndex, layerIndex) } -> std::same_as<void>;
        }
    void visitLayerGroups(Scheduler* scheduler, bool reversed, Func f) {
        const auto layerCount = numLayerGroups();
        if (!layerCount) {
            return;
        }
        if (!scheduler) {
            const std::optional<std::size_t> threadIndex{};
            if (reversed) {
                visitLayerGroupsReversed([&](auto& group, auto layerIndex) { f(group, threadIndex, layerIndex); });
            } else {
                visitLayerGroups([&](auto& group, auto layerIndex) { f(group, threadIndex, layerIndex); });
            }
            return;
        }

        // We use sequential indexes, not the sort key used in the group map, so put everything in a vector
        std::vector<std::reference_wrapper<LayerGroupBase>> groups;
        groups.reserve(layerCount);
        eachGroup(reversed, [&](auto& group) { groups.push_back(group); });
        assert(groups.size() == layerCount);

        // Use different threads to render a given layer in subsequent frames, to ensure that
        // we don't have problems due to objects cached from previous frames being used on a
        // different thread.
#ifndef NDEBUG
        const bool reverseThreads = rand() & 1;
#endif

        // Submit one task to each available thread, running the function
        // on the corresponding items in each group in the specified order.
        const auto threadCount = scheduler->getThreadCount();
        scheduler->eachThread([&, threadCount, layerCount, reversed](const auto threadIndex_) {
            if (threadIndex_) {
#ifndef NDEBUG
                const auto threadIndex = reverseThreads ? threadCount - *threadIndex_ - 1 : *threadIndex_;
#else
                const auto threadIndex = *threadIndex_;
#endif
                const auto minIndex = threadIndex * layerCount / threadCount;
                const auto maxIndex = (threadIndex + 1) * layerCount / threadCount;
                for (auto i = minIndex; i != maxIndex; ++i) {
                    f(groups[i].get(), threadIndex, reversed ? layerCount - i - 1 : i);
                }
            }
        });
    }

    template <typename Func>
    void visitLayerGroups(Scheduler* scheduler, Func f) {
        visitLayerGroups(scheduler, /*reversed=*/false, f);
    }

    void updateLayers(gfx::ShaderRegistry&,
                      gfx::Context&,
                      const TransformState&,
                      const std::shared_ptr<UpdateParameters>&,
                      const RenderTree&);

    void processChanges();

    bool addRenderTarget(RenderTargetPtr);
    bool removeRenderTarget(const RenderTargetPtr&);

    template <typename Func /* void(RenderTarget&) */>
    void visitRenderTargets(Func f) {
        for (auto& renderTarget : renderTargets) {
            f(*renderTarget);
        }
    }

    void updateDebugLayerGroups(const RenderTree& renderTree, PaintParameters& parameters);

    template <typename Func /* void(LayerGroupBase&) */>
    void visitDebugLayerGroups(Func f) {
        for (auto& pair : debugLayerGroups) {
            if (pair.second) {
                f(*pair.second);
            }
        }
    }
#endif

    const ZoomHistory& getZoomHistory() const { return zoomHistory; }

private:
    bool isLoaded() const;
    bool hasTransitions(TimePoint) const;

    RenderSource* getRenderSource(const std::string& id) const;

    RenderLayer* getRenderLayer(const std::string& id);
    const RenderLayer* getRenderLayer(const std::string& id) const;

    void queryRenderedSymbols(std::unordered_map<std::string, std::vector<Feature>>& resultsByLayer,
                              const ScreenLineString& geometry,
                              const std::unordered_map<std::string, const RenderLayer*>& layers,
                              const RenderedQueryOptions& options) const;

    std::vector<Feature> queryRenderedFeatures(const ScreenLineString&,
                                               const RenderedQueryOptions&,
                                               const std::unordered_map<std::string, const RenderLayer*>&) const;

    // GlyphManagerObserver implementation.
    void onGlyphsLoaded(const FontStack&, const GlyphRange&) override;
    void onGlyphsError(const FontStack&, const GlyphRange&, std::exception_ptr) override;
    void onGlyphsRequested(const FontStack&, const GlyphRange&) override;
    // RenderSourceObserver implementation.
    void onTileChanged(RenderSource&, const OverscaledTileID&) override;
    void onTileError(RenderSource&, const OverscaledTileID&, std::exception_ptr) override;
    void onTileAction(RenderSource&, TileOperation, const OverscaledTileID&, const std::string&) override;

    // ImageManagerObserver implementation
    void onStyleImageMissing(const std::string&, const std::function<void()>&) override;
    void onRemoveUnusedStyleImages(const std::vector<std::string>&) override;

#if MLN_DRAWABLE_RENDERER
    /// Move changes into the pending set, clearing the provided collection
    void addChanges(UniqueChangeRequestVec&);

    template <typename Func>
        requires requires(Func f, LayerGroupBase& layerGroup) {
            { f(layerGroup) } -> std::same_as<void>;
        }
    void eachGroup(bool reversed, Func f) {
        using namespace std::ranges;
        auto wrap = [&](auto& pair) {
            assert(pair.second);
            f(*pair.second);
        };
        if (reversed) {
            for_each(layerGroupsByLayerIndex | std::views::reverse, std::move(wrap));
        } else {
            for_each(layerGroupsByLayerIndex, std::move(wrap));
        }
    }
#endif

    RendererObserver* observer;

    ZoomHistory zoomHistory;
    TransformState transformState;

    std::shared_ptr<GlyphManager> glyphManager;
    std::shared_ptr<ImageManager> imageManager;
    std::unique_ptr<LineAtlas> lineAtlas;
    std::unique_ptr<PatternAtlas> patternAtlas;

    Immutable<std::vector<Immutable<style::Image::Impl>>> imageImpls;
    Immutable<std::vector<Immutable<style::Source::Impl>>> sourceImpls;
    Immutable<std::vector<Immutable<style::Layer::Impl>>> layerImpls;

    std::unordered_map<std::string, std::unique_ptr<RenderSource>> renderSources;
    std::unordered_map<std::string, std::unique_ptr<RenderLayer>> renderLayers;
    RenderLight renderLight;

    CrossTileSymbolIndex crossTileSymbolIndex;
    PlacementController placementController;

    const bool backgroundLayerAsColor;
    bool contextLost = false;
    bool placedSymbolDataCollected = false;
    bool tileCacheEnabled = true;

    // Vectors with reserved capacity of layerImpls->size() to avoid
    // reallocation on each frame.
    std::vector<Immutable<style::LayerProperties>> filteredLayersForSource;
    RenderLayerReferences orderedLayers;
    RenderLayerReferences layersNeedPlacement;

    TaggedScheduler threadPool;
    Scheduler* renderThreadPool;

#if MLN_DRAWABLE_RENDERER
    std::vector<std::unique_ptr<ChangeRequest>> pendingChanges;

    using LayerGroupMap = std::multimap<int32_t, LayerGroupBasePtr>;
    LayerGroupMap layerGroupsByLayerIndex;

    std::vector<RenderTargetPtr> renderTargets;
    RenderItem::DebugLayerGroupMap debugLayerGroups;
#endif
};

} // namespace mbgl
