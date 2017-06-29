//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include <random>

#include "LocalTimelineBlockRandomizer.h"
#include "RandomOrdering.h"

namespace CNTK {

LocalTimelineBlockRandomizer::LocalTimelineBlockRandomizer(
    DataDeserializerPtr deserializer,
    size_t randomizationRange,
    size_t seedOffset,
    bool multithreadedGetNextSequences,
    size_t maxNumberOfInvalidSequences)
: Base(deserializer, multithreadedGetNextSequences, maxNumberOfInvalidSequences),
  m_randomizationRange(randomizationRange),
  m_seedOffset(seedOffset),
  m_globalChunkPosition(0)
{
    m_chunkDescriptions = m_originalChunkDescriptions;
    m_rng.seed((unsigned long)m_sweepIndex + m_seedOffset);
    Microsoft::MSR::CNTK::RandomShuffleMT(m_chunkDescriptions, m_rng);
}

void LocalTimelineBlockRandomizer::PrefetchChunks()
{
    size_t position = m_globalChunkPosition % m_originalChunkDescriptions.size();
    size_t sweepIndex = m_sweepIndex;

    m_prefetch = std::async(std::launch::async, [&, this]()
    {
        // Prefetch does not change any state that cannot be recalculated,
        // only prefetches data.
        size_t range = m_randomizationRange;
        while (range > 0)
        {
            if (position == 0)
            {
                sweepIndex++;
                m_prefetchedChunkDescriptions = m_originalChunkDescriptions;
                m_rng.seed((unsigned long)sweepIndex);
                Microsoft::MSR::CNTK::RandomShuffleMT(m_prefetchedChunkDescriptions, m_rng);
            }

            auto desc = m_prefetchedChunkDescriptions[position];
            if (position % m_config.m_numberOfWorkers == m_config.m_workerRank &&
                m_chunks.find(desc.m_id) == m_chunks.end())
            {
                ChunkPtr data = m_deserializer->GetChunk(desc.m_id);
                m_prefetchedChunks.push_back(std::make_pair(desc, data));
                --range;
            }

            position = (position + 1) % m_originalChunkDescriptions.size();
        }
    });
}

void LocalTimelineBlockRandomizer::RefillSequenceWindow()
{
    if (!m_prefetch.valid())
        PrefetchChunks();

    m_prefetch.wait();

    auto sweepIndex = m_sweepIndex;

    // Actually moving cursors forward, if the data is small
    // the sweep boundary can be crossed several times.
    size_t range = m_randomizationRange;
    while(range > 0)
    {
        ++m_globalChunkPosition;

        auto sweepPosition = m_globalChunkPosition % m_originalChunkDescriptions.size();
        if (sweepPosition % m_config.m_numberOfWorkers == m_config.m_workerRank)
        {
            auto desc = m_chunkDescriptions[sweepPosition];
            m_deserializer->GetSequencesForChunk(desc.m_id, m_sequenceWindow);
            --range;
        }

        // Last chunk
        if (sweepPosition == m_chunkDescriptions.size() - 1)
        {
            m_sequenceWindow.push_back(s_endOfSweep);
            sweepIndex++;
            m_chunkDescriptions = m_originalChunkDescriptions;
            m_rng.seed((unsigned long)sweepIndex + m_seedOffset);
            Microsoft::MSR::CNTK::RandomShuffleMT(m_chunkDescriptions, m_rng);
        }
    }

    // Chunks are updated only on the main thread.
    for (const auto& c : m_prefetchedChunks)
        m_chunks.insert(std::make_pair(c.first.m_id, c.second));

    // Prefetch new data chunks.
    PrefetchChunks();
}


Dictionary LocalTimelineBlockRandomizer::GetInnerState()
{
    Dictionary state;
    state[L"globalChunkPosition"] = (size_t)m_globalChunkPosition;
    return state;
}

void LocalTimelineBlockRandomizer::SetInnerState(const Dictionary& state)
{
    m_chunkDescriptions = m_originalChunkDescriptions;
    m_rng.seed((unsigned long)m_sweepIndex + m_seedOffset);
    Microsoft::MSR::CNTK::RandomShuffleMT(m_chunkDescriptions, m_rng);

    m_globalChunkPosition = (ChunkIdType)state[L"globalChunkPosition"].Value<size_t>();
    if (m_prefetch.valid())
        m_prefetch.wait();
    m_prefetchedChunkDescriptions = m_chunkDescriptions;
}

}
