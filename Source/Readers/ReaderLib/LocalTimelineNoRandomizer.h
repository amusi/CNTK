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

// The class represents a randomizer that does not randomize input (identity function over the original timeline).
// Used training where the training data has already been pre - randomized.
// TODO: currently this code moved from the old block randomizer.
// TODO: The class will be further refactored and common based will be extracted with BlockRandomizer.
class LocalTimelineNoRandomizer : public SequenceEnumerator
{
public:
    LocalTimelineNoRandomizer(
        DataDeserializerPtr deserializer, 
        bool multithreadedGetNextSequences = false,
        size_t maxNumberOfInvalidSequences = 0); // per worker

    virtual void StartEpoch(const EpochConfiguration& config) override;
    virtual Sequences GetNextSequences(size_t globalSampleCount, size_t localSampleCount) override;
    virtual std::vector<StreamInformation> GetStreamDescriptions() const override
    {
        return m_deserializer->GetStreamDescriptions();
    }

    Dictionary GetCurrentSamplePosition() override;
    void SetCurrentSamplePosition(const Dictionary& currentSamplePosition) override;

    void SetConfiguration(const ReaderConfiguration& config) override;

private:
    // Gets next sequences not exceeding localSampleCount for this worker and globalSampleCount across workers.
    void GetNextSequenceDescriptions(size_t localSampleCount, Sequences& result);

    // Moves the cursor to the sequence possibly updating the chunk.
    void MoveToNextSequence();

    inline bool IsEndReached() const
    {
        if (m_config.m_totalEpochSizeInSweeps != g_infinity)
            return m_config.m_totalEpochSizeInSweeps == m_sweepIndex + 1;
        return m_config.m_totalEpochSizeInSamples <= m_sequencePositionInSweep + (m_sweepIndex + 1) * m_sweepSizeInSamples;
    }

    inline bool IsBeginningOfSweep() const
    {
        return m_sequencePositionInSweep == 0;
    }

    DataDeserializerPtr m_deserializer;

    // Whether to get sequences using multiple thread.
    // Useful in case deserializer performs CPU intensive deserialization (e.g. decompression)
    bool m_multithreadedGetNextSequences;

    // Stream descriptions
    std::vector<StreamInformation> m_streams;

    // Epoch configuration
    EpochConfiguration m_config;

    // Chunk descriptions.
    ChunkDescriptions m_chunkDescriptions;

    // Current chunk data.
    std::map<ChunkIdType, ChunkPtr> m_chunks;

    // Current chunk position that the randomizer works with.
    // An index inside the m_chunkDescriptions.
    ChunkIdType m_currentChunkPosition;

    // Current sequence position the randomizer works with.
    size_t m_currentSequencePositionInChunk;

    // Current window of sequence descriptions.
    std::vector<SequenceDescription> m_sequenceWindow;

    // Total number of samples in the sweep.
    size_t m_sweepSizeInSamples;
    size_t m_sweepIndex;
    size_t m_samplePositionInSweep;
    size_t m_sequencePositionInSweep;

    // Temp buffer to avoid allocations.
    std::vector<SequenceDescription> m_sequenceBuffer;

    // Helper class for removing invalid sequences.
    SequenceCleaner m_cleaner;
};

}
