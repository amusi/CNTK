//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>

#include "LocalTimelineNoRandomizer.h"
#include "DataReader.h"
#include "ExceptionCapture.h"

namespace CNTK {

LocalTimelineNoRandomizer::LocalTimelineNoRandomizer(DataDeserializerPtr deserializer, bool multithreadedGetNextSequences, size_t maxNumberOfInvalidSequences)
: m_deserializer(deserializer),
  m_currentChunkPosition(ChunkIdMax),
  m_currentSequencePositionInChunk(0),
  m_sweepSizeInSamples(0),
  m_multithreadedGetNextSequences(multithreadedGetNextSequences),
  m_cleaner(maxNumberOfInvalidSequences),
  m_sweepIndex(0),
  m_samplePositionInSweep(0),
  m_sequencePositionInSweep(0)
{
    assert(deserializer != nullptr);
    m_streams = m_deserializer->GetStreamDescriptions();
    m_chunkDescriptions = m_deserializer->GetChunkDescriptions();

    if (m_chunkDescriptions.empty())
        RuntimeError("LocalTimelineNoRandomizer: Expected input to contain samples, but the number of successfully read samples was 0.");
}

void LocalTimelineNoRandomizer::StartEpoch(const EpochConfiguration& config)
{
    if(config.m_epochIndex != 0)
        RuntimeError("LocalTimelineNoRandomizer not supported for old configs.");

    m_config = config;
    if (config.m_totalEpochSizeInSweeps == g_infinity && m_config.m_totalEpochSizeInSamples == Microsoft::MSR::CNTK::requestDataSize)
        m_config.m_totalEpochSizeInSweeps = 1;

    // Make local sample count.
    int shouldAddOneSample = (int)m_config.m_totalEpochSizeInSamples % m_config.m_numberOfWorkers > m_config.m_workerRank;
    m_config.m_totalEpochSizeInSamples = m_config.m_totalEpochSizeInSamples / m_config.m_numberOfWorkers + shouldAddOneSample;

    Dictionary zeroPosition;
    zeroPosition[L"sweepSizeInSamples"] = 0;
    zeroPosition[L"sweepIndex"] = 0;
    zeroPosition[L"samplePositionInSweep"] = 0;
    zeroPosition[L"sequencePositionInSweep"] = 0;
    zeroPosition[L"currentChunkPosition"] = 0;
    zeroPosition[L"currentSequencePositionInChunk"] = 0;
    SetState(zeroPosition);
}

void LocalTimelineNoRandomizer::MoveToNextSequence()
{
    if (m_currentSequencePositionInChunk + 1 >= m_chunkDescriptions[m_currentChunkPosition].m_numberOfSequences)
    {
        // Reached the end of the sweep, update the numbers.
        if (m_currentChunkPosition == m_chunkDescriptions.size() - 1)
        {
            m_sweepSizeInSamples = m_samplePositionInSweep;
            m_sweepIndex++;
            m_samplePositionInSweep = 0;
            m_sequencePositionInSweep = 0;
        }

        // Moving to the next chunk.
        m_currentChunkPosition = (m_currentChunkPosition + 1) % m_chunkDescriptions.size();
        m_currentSequencePositionInChunk = 0;
        m_sequenceWindow.clear();
        m_deserializer->GetSequencesForChunk(m_currentChunkPosition, m_sequenceWindow);
    }
    else
    {
        m_currentSequencePositionInChunk++;
        m_sequencePositionInSweep++;
        m_samplePositionInSweep += m_sequenceWindow[m_currentSequencePositionInChunk].m_numberOfSamples;
    }
}

// Gets next sequences not exceeding local and global samples.
void LocalTimelineNoRandomizer::GetNextSequenceDescriptions(size_t sampleCount, Sequences& result)
{
    assert(sampleCount != 0);

    if (sampleCount > std::numeric_limits<int>::max())
        RuntimeError("Local size of the minibatch cannot exceed max int.");

    assert(m_sequenceWindow.size() != 0);
    assert(m_chunkDescriptions[m_currentChunkPosition].m_numberOfSequences > m_currentSequencePositionInChunk);

    size_t numLocalSamplesLoaded = 0;
    bool atLeastOneSequenceNeeded = true;

    m_sequenceBuffer.clear();

    while (numLocalSamplesLoaded < sampleCount)
    {
        const SequenceDescription& sequence = m_sequenceWindow[m_currentSequencePositionInChunk];
        auto sequenceLength = sequence.m_numberOfSamples;

        // Let's check whether we need to return this sequence or skip it.
        bool isLocal = m_sequencePositionInSweep % m_config.m_numberOfWorkers == m_config.m_workerRank;
        if (!atLeastOneSequenceNeeded)
        {
            // Break if we're exceeding the local requested sample count.
            if (isLocal && numLocalSamplesLoaded + sequenceLength > sampleCount)
                break;
        }

        if (IsEndReached())
            break;

        if (isLocal) // Ok good to add it to the result.
        {
            m_sequenceBuffer.push_back(sequence);
            numLocalSamplesLoaded += sequenceLength;
            atLeastOneSequenceNeeded = false;
        }

        MoveToNextSequence();
    }

    // Set the end-of-epoch flag (true when the current batch is last in an epoch).
    result.m_endOfEpoch = IsEndReached();
    result.m_endOfSweep = IsBeginningOfSweep() && m_sweepIndex != 1;
}

Sequences LocalTimelineNoRandomizer::GetNextSequences(size_t, size_t sampleCount)
{
    if (sampleCount == 0)
        LogicError("Sample count must not be zero.");

    Sequences result;
    if (IsEndReached())
    {
        result.m_endOfEpoch = true;
        result.m_endOfSweep = IsBeginningOfSweep() && m_sweepIndex != 1;
        return result;
    }

    GetNextSequenceDescriptions(sampleCount, result);

    if (m_sequenceBuffer.size() == 0)
        return result;

    result.m_data.resize(m_streams.size(), std::vector<SequenceDataPtr>(m_sequenceBuffer.size()));

    // Collect all the chunks that we need
    std::map<ChunkIdType, ChunkPtr> chunks;
    for (const auto& s : m_sequenceBuffer)
    {
        if (chunks.find(s.m_chunkId) != chunks.end())
            continue;

        auto old = m_chunks.find(s.m_chunkId);
        if (old != m_chunks.end())
            chunks.insert(std::make_pair(s.m_chunkId, old->second));
        else
            chunks[s.m_chunkId] = m_deserializer->GetChunk(s.m_chunkId);
    }

    // swap current chunks with new ones:
    m_chunks.swap(chunks);

    auto process = [&](int i) -> void {
        std::vector<SequenceDataPtr> sequence;
        const auto& sequenceDescription = m_sequenceBuffer[i];

        auto it = m_chunks.find(sequenceDescription.m_chunkId);
        if (it == m_chunks.end())
        {
            LogicError("Invalid chunk requested.");
        }

        it->second->GetSequence(sequenceDescription.m_indexInChunk, sequence);
        for (int j = 0; j < m_streams.size(); ++j)
        {
            result.m_data[j][i] = sequence[j];
        }
    };

    if (m_multithreadedGetNextSequences)
    {
        ExceptionCapture capture;
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < m_sequenceBuffer.size(); ++i)
            capture.SafeRun(process, i);
        capture.RethrowIfHappened();
    }
    else
    {
        for (int i = 0; i < m_sequenceBuffer.size(); ++i)
            process(i);
    }

    m_cleaner.Clean(result);
    return result;
}

Dictionary LocalTimelineNoRandomizer::GetState()
{
    Dictionary result;
    result[L"sweepSizeInSamples"] = m_sweepSizeInSamples;
    result[L"sweepIndex"] = m_sweepIndex;
    result[L"samplePositionInSweep"] = m_samplePositionInSweep;
    result[L"sequencePositionInSweep"] = m_sequencePositionInSweep;
    result[L"currentChunkPosition"] = (size_t)m_currentChunkPosition;
    result[L"currentSequencePositionInChunk"] = m_currentSequencePositionInChunk;
    return result;
}

void LocalTimelineNoRandomizer::SetState(const Dictionary& state)
{
    m_sweepSizeInSamples = state[L"sweepSizeInSamples"].Value<size_t>();
    m_sweepIndex = state[L"sweepIndex"].Value<size_t>();
    m_samplePositionInSweep = state[L"samplePositionInSweep"].Value<size_t>();
    m_sequencePositionInSweep = state[L"sequencePositionInSweep"].Value<size_t>();

    auto oldChunkPosition = m_currentChunkPosition;

    m_currentChunkPosition = (ChunkIdType)state[L"currentChunkPosition"].Value<size_t>();
    m_currentSequencePositionInChunk = state[L"currentSequencePositionInChunk"].Value<size_t>();

    if (oldChunkPosition != m_currentChunkPosition)
    {
        m_sequenceWindow.clear();
        m_deserializer->GetSequencesForChunk(m_currentChunkPosition, m_sequenceWindow);
    }
}

void LocalTimelineNoRandomizer::SetConfiguration(const ReaderConfiguration& config)
{
    *((ReaderConfiguration*)&m_config) = config;
}

}
