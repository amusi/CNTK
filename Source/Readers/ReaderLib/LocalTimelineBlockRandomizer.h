//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include "SequenceEnumerator.h"
#include "DataDeserializer.h"
#include "ReaderUtil.h"
#include "LocalTimelineRandomizerBase.h"
#include <tuple>

namespace CNTK {

// A randomizer that firstly randomizes chunks and then sequences inside a tumbling window of chunks.
class LocalTimelineBlockRandomizer : public LocalTimelineRandomizerBase
{
    typedef LocalTimelineRandomizerBase Base;

public:
    LocalTimelineBlockRandomizer(
        DataDeserializerPtr deserializer,
        size_t randomizationRange,
        size_t seedOffset = 0,
        bool multithreadedGetNextSequences = false,
        size_t maxNumberOfInvalidSequences= 0); // per worker

    Dictionary GetInnerState() override;
    void SetInnerState(const Dictionary& state) override;
    void RefillSequenceWindow() override;

private:
    void PrefetchChunks();

    const size_t m_randomizationRange;
    const size_t m_seedOffset;

    // Current chunk position that the randomizer works with.
    ChunkIdType m_globalChunkPosition;

    std::mt19937_64 m_rng;

    // Randomized chunk descriptions.
    ChunkDescriptions m_prefetchedChunkDescriptions;

    std::future<void> m_prefetch;
    std::vector<std::tuple<ChunkDescription, ChunkPtr, std::vector<SequenceDescription>>> m_prefetchedChunks;
};

}
