//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include "SequenceEnumerator.h"
#include "DataDeserializer.h"
#include "ReaderUtil.h"

namespace CNTK {

// The class is a base randomizer for the local timeline.
class LocalTimelineRandomizerBase : public SequenceEnumerator
{
public:
    LocalTimelineRandomizerBase(
        DataDeserializerPtr deserializer,
        bool multithreadedGetNextSequences = false,
        size_t maxNumberOfInvalidSequences= 0); // per worker

    virtual void StartEpoch(const EpochConfiguration& config) override;

    void SetConfiguration(const ReaderConfiguration& config) override;

    virtual Sequences GetNextSequences(size_t globalSampleCount, size_t localSampleCount) override;

    virtual std::vector<StreamInformation> GetStreamDescriptions() const override
    {
        static auto streams = m_deserializer->GetStreamDescriptions();
        return streams;
    }

    Dictionary GetState() override;
    void SetState(const Dictionary& state) override;

private:
    // Should preserve/retrieve the state in the derived classes.
    virtual Dictionary GetInnerState() = 0;
    virtual void SetInnerState(const Dictionary& state) = 0;

    // The function should fill m_sequenceWindow with new data.
    virtual void RefillSequenceWindow() = 0;

    // Gets next sequences not exceeding localSampleCount for this worker and globalSampleCount across workers.
    void GetNextSequenceDescriptions(size_t maxSampleCount, Sequences& result);

    // Moves the cursor to the sequence possibly updating the chunk.
    void MoveToNextSequence();

    // Checks if the end of the data has been reached.
    inline bool IsEndReached() const
    {
        if (m_config.m_totalEpochSizeInSweeps != g_infinity)
            return m_config.m_totalEpochSizeInSweeps == m_sweepIndex;

        // Limit in global samples, make local sample limit.
        int shouldAddOneSample = (int)m_config.m_totalEpochSizeInSamples % m_config.m_numberOfWorkers > m_config.m_workerRank;
        return m_numberOfSamplesSeenSoFar >= m_config.m_totalEpochSizeInSamples / m_config.m_numberOfWorkers + shouldAddOneSample;
    }

    // Checks if a sequence descriptor is a special marker for the end of the sweep.
    inline bool IsEndOfSweep(const SequenceDescription& sequence)
    {
        return sequence.m_indexInChunk == s_endOfSweep.m_indexInChunk &&
               sequence.m_chunkId == s_endOfSweep.m_chunkId &&
               sequence.m_numberOfSamples == s_endOfSweep.m_numberOfSamples;
    }

protected:
    const static SequenceDescription s_endOfSweep; // Sequence indicating end of the sweep.

    DataDeserializerPtr m_deserializer;

    // Whether to get sequences using multiple thread.
    // Useful in case deserializer performs CPU intensive deserialization (e.g. decompression)
    bool m_multithreadedGetNextSequences;

    // Epoch configuration
    EpochConfiguration m_config;

    // Original chunk descriptions.
    ChunkDescriptions m_originalChunkDescriptions;

    // Current chunk data, should always contain
    // chunks that are returned by the GetNextSequenceDescription function.
    std::map<ChunkIdType, ChunkPtr> m_chunks;

    // Current window of sequence descriptions.
    std::vector<SequenceDescription> m_sequenceWindow;

    // Current sequence position the randomizer works with.
    size_t m_currentSequencePositionInWindow;
    size_t m_sweepIndex;
    size_t m_numberOfSamplesSeenSoFar;

    // Temp buffer to avoid allocations.
    std::vector<SequenceDescription> m_sequenceBuffer;

    // Helper class for removing invalid sequences.
    SequenceCleaner m_cleaner;

    Dictionary m_currentState;
};

}
