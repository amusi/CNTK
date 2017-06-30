//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include <random>

#include "LocalTimelineBlockRandomizer.h"
#include "RandomOrdering.h"
#include <tuple>

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
    m_prefetchedChunkDescriptions = m_originalChunkDescriptions;
    m_rng.seed((unsigned long)m_sweepIndex + m_seedOffset);
    Microsoft::MSR::CNTK::RandomShuffleMT(m_prefetchedChunkDescriptions, m_rng);
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
            if (position % m_config.m_numberOfWorkers == m_config.m_workerRank) // Need to add to the window
            {
                std::vector<SequenceDescription> sequences;
                ChunkPtr data;
                if (m_window.m_dataChunks.find(desc.m_id) == m_window.m_dataChunks.end())
                {
                    // Query deserializer.
                    data = m_deserializer->GetChunk(desc.m_id);
                    m_deserializer->GetSequencesForChunk(desc.m_id, sequences);
                }
                else // Simple copy
                {
                    for (size_t i = 0; i < m_window.m_sequences.size(); ++i)
                        if (m_window.m_sequences[i].m_chunkId == desc.m_id)
                            sequences.push_back(m_window.m_sequences[i]);
                    data = m_window.m_dataChunks[desc.m_id];
                }

                m_prefetchedChunks.push_back(std::make_tuple(desc, data, sequences));
                --range;
            }
            else
            {
                // Empty, we do not need this except for tracking the current 
                m_prefetchedChunks.push_back(std::make_tuple(ChunkDescription{}, nullptr, std::vector<SequenceDescription>{}));
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

    m_window.m_sequences.clear();
    m_window.m_dataChunks.clear();

    for (const auto& c : m_prefetchedChunks)
    {
        ++m_globalChunkPosition;

        auto sweepPosition = m_globalChunkPosition % m_originalChunkDescriptions.size();
        if (sweepPosition % m_config.m_numberOfWorkers == m_config.m_workerRank)
        {
            m_window.m_sequences.insert(m_window.m_sequences.end(), std::get<2>(c).begin(), std::get<2>(c).end());
            m_window.m_dataChunks.insert(std::make_pair(std::get<0>(c).m_id, std::get<1>(c)));
        }

        // Last chunk
        if (sweepPosition == m_originalChunkDescriptions.size() - 1)
            m_window.m_sequences.push_back(s_endOfSweep);
    }

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
    if (m_prefetch.valid())
        m_prefetch.wait();
    m_rng.seed((unsigned long)m_sweepIndex + m_seedOffset);
    Microsoft::MSR::CNTK::RandomShuffleMT(m_prefetchedChunkDescriptions, m_rng);
    m_globalChunkPosition = (ChunkIdType)state[L"globalChunkPosition"].Value<size_t>();
}

}
