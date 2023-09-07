#include <mbgl/util/string_indexer.hpp>

#include <cassert>

namespace mbgl {

namespace {
const std::string empty;
}

StringIdentity StringIndexer::get(const std::string& string) {
    MapType& stringToIdentity = getMap();
    VectorType& identityToString = getVector();
    assert(stringToIdentity.size() == identityToString.size());

    if (auto it = stringToIdentity.find(string); it != stringToIdentity.end()) {
        return it->second;
    } else {
        StringIdentity id = identityToString.size();
        identityToString.push_back(string);
        stringToIdentity[string] = id;
        return id;
    }
}

const std::string& StringIndexer::get(const StringIdentity id) {
    const VectorType& identityToString = getVector();
    assert(id < identityToString.size());

    return id < identityToString.size() ? identityToString[id] : empty;
}

void StringIndexer::clear() {
    StringIndexer::getMap().clear();
    StringIndexer::getVector().clear();
}

size_t StringIndexer::size() {
    [[maybe_unused]] const MapType& stringToIdentity = getMap();
    const VectorType& identityToString = getVector();
    assert(stringToIdentity.size() == identityToString.size());

    return identityToString.size();
}

} // namespace mbgl