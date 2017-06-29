//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Swig specific utility classes, the file should be used only from cntk_py.i
//

#pragma once

#include <memory>

namespace CNTK
{
    // Swig deserializer is used to expose user defined deserializers
    // to Python.
    class SwigDataDeserializer final : public CNTK::DataDeserializer
    {
        mutable std::vector<StreamInformation> m_streamInfos;
        mutable std::once_flag m_streamInfosInitFlag;

        mutable ChunkDescriptions m_chunkInfos;
        mutable std::once_flag m_chunkInfosInitFlag;

    public:
        SwigDataDeserializer() { }

        // Interface implemented in Python.
        virtual void _GetStreamInfos(std::vector<CNTK::StreamInformation>&) { NOT_IMPLEMENTED; }
        virtual void _GetChunkInfos(std::vector<CNTK::ChunkDescription>&) { NOT_IMPLEMENTED; }
        virtual void _GetSequencesForChunk(size_t id, std::vector<CNTK::SequenceDescription>&) { NOT_IMPLEMENTED; }
        virtual ChunkPtr _GetChunk(ChunkIdType chunkId) { NOT_IMPLEMENTED; }

        // Simple 2Py redirectors.
        std::vector<StreamInformation> GetStreamDescriptions() override
        {
            std::call_once(m_streamInfosInitFlag, [this]() {
                GilStateGuard guard;
                _GetStreamInfos(m_streamInfos);
            });
            return m_streamInfos;
        }

        ChunkDescriptions GetChunkDescriptions() override
        {
            std::call_once(m_chunkInfosInitFlag, [this]() {
                GilStateGuard guard;
                _GetChunkInfos(m_chunkInfos);
            });
            return m_chunkInfos;
        }

        void GetSequencesForChunk(ChunkIdType chunkId, std::vector<CNTK::SequenceDescription>& descriptions) override
        {
            GilStateGuard guard;
            _GetSequencesForChunk(chunkId, descriptions);
        }

        ChunkPtr GetChunk(ChunkIdType chunkId)
        {
            GilStateGuard guard;
            return _GetChunk(chunkId);
        }

        bool GetSequenceDescription(const SequenceDescription& primary, SequenceDescription& description) override
        {
            NOT_IMPLEMENTED;
        }
    };

    class SwigChunk final : public CNTK::Chunk
    {
        std::vector<CNTK::StreamInformation> m_streamInfos;

        struct SwigDenseData final : public CNTK::DenseSequenceData
        {
            SwigDenseData(PyArrayObject* object) : m_object(object)
            {
                Py_INCREF(m_object);
            }

            ~SwigDenseData()
            {
                GilStateGuard guard;
                Py_DECREF(m_object);
            };

            virtual const void* GetDataBuffer()
            {
                GilStateGuard guard;
                return PyArray_DATA(m_object);
            }

            virtual const CNTK::NDShape& GetSampleShape()
            {
                RuntimeError("Sample shape should be specified on the stream.");
            }

        private:
            PyArrayObject* m_object;

            SwigDenseData(const SwigDenseData&) = delete; SwigDenseData& operator=(const SwigDenseData&) = delete;
            SwigDenseData& operator=(SwigDenseData&&) = delete; SwigDenseData(SwigDenseData&& other) = delete;
        };

        struct SwigSparseData final : public CNTK::SparseSequenceData
        {
            SwigSparseData(PyObject* object, PyArrayObject* data, PyArrayObject* indices, PyArrayObject* indptr)
                : m_object(object), m_pyData(data), m_pyIndptr(indptr), m_pyIndices(indices)
            {
                Py_INCREF(m_object);
                Py_INCREF(m_pyData);
                Py_INCREF(m_pyIndptr);
                Py_INCREF(m_pyIndices);

                m_indices = (SparseIndexType*)PyArray_DATA(m_pyIndices);
                m_totalNnzCount = static_cast<SparseIndexType>(PyArray_SIZE(m_pyData));
                auto nnzCountsSize = PyArray_SIZE(m_pyIndptr);
                m_nnzCounts.resize(nnzCountsSize);
                auto type = PyArray_TYPE(m_pyIndptr);
                auto indPtr = PyArray_DATA(m_pyIndptr);

                size_t elementSize = 0;
                switch (type)
                {
                case NPY_LONG:
                    elementSize = NPY_SIZEOF_LONG;
                    break;
                case NPY_INT:
                    elementSize = NPY_SIZEOF_INT;
                    break;
                default:
                    RuntimeError("Unsupported index type '%d'", type);
                }

                if(elementSize != sizeof(SparseIndexType))
                    RuntimeError("Number of bits for index is unsupported for type '%d'", type);

                memcpy(&m_nnzCounts[0], indPtr, nnzCountsSize * elementSize);
                for (size_t i = 0; i < m_nnzCounts.size() - 1; ++i)
                    m_nnzCounts[i] = m_nnzCounts[i + 1] - m_nnzCounts[i];
                m_nnzCounts.resize(m_nnzCounts.size() - 1);
            }

            virtual ~SwigSparseData()
            {
                GilStateGuard guard;
                Py_DECREF(m_object);
                Py_DECREF(m_pyData);
                Py_DECREF(m_pyIndptr);
                Py_DECREF(m_pyIndices);
            };

            virtual const void* GetDataBuffer()
            {
                GilStateGuard guard;
                return PyArray_DATA(m_pyData);
            }

            virtual const CNTK::NDShape& GetSampleShape()
            {
                RuntimeError("Sample shape should be specified on the stream.");
            }

        private:
            PyObject* m_object;
            PyArrayObject* m_pyData;
            PyArrayObject* m_pyIndptr;
            PyArrayObject* m_pyIndices;
        };

        SequenceDataPtr FromNumPy(PyObject* object, size_t index)
        {
            if (!PyArray_Check((PyArrayObject*)object))
                throw std::logic_error("NumPy array expected");

            PyArrayObject* array = (PyArrayObject*)object;
            int rank = PyArray_NDIM(array);
            npy_intp* np_shape = PyArray_SHAPE(array);

            const auto& info = m_streamInfos[index];
            uint32_t numSamples = info.m_sampleLayout.Rank() == rank ? 1 : static_cast<uint32_t>(np_shape[0]);
            int typecode = PyArray_TYPE(array);

            SequenceDataPtr result = std::make_shared<SwigDenseData>(array);
            result->m_numberOfSamples = numSamples;
            return result;
        }

        SequenceDataPtr FromCSR(PyObject* object, size_t index)
        {
            auto data = (PyArrayObject*)PyObject_GetAttrString(object, "data");
            auto indptr = (PyArrayObject*)PyObject_GetAttrString(object, "indptr");
            auto indices = (PyArrayObject*)PyObject_GetAttrString(object, "indices");

            auto shape = PyObject_GetAttrString(object, "shape");
            auto numElements = PyTuple_GET_ITEM(shape, 0);

            SequenceDataPtr result = std::make_shared<SwigSparseData>(object, data, indices, indptr);
            result->m_numberOfSamples = static_cast<uint32_t>(PyLong_AsSize_t(numElements));
            return result;
        }

    public:
        SwigChunk(const std::vector<CNTK::StreamInformation>& streamInfos) : m_streamInfos(streamInfos)
        {}

        void GetSequence(size_t sequenceIndex, std::vector<CNTK::SequenceDataPtr>& result) override
        {
            GilStateGuard guard;
            PyObject *pylist = PyList_New(0);
            Py_INCREF(pylist);

            _GetSequence(sequenceIndex, pylist);

            PyObject *item = nullptr;
            PyObject *iterator = PyObject_GetIter(pylist);
            if (!iterator)
                SWIG_exception_fail(SWIG_ValueError, "cannot convert list element to CNTK::StreamInformation");

            size_t i = 0;
            while ((item = PyIter_Next(iterator)))
            {
                if (PyArray_Check(item))
                {
                    auto sequence = FromNumPy(item, i++);
                    result.push_back(sequence);
                }
                else if (item->ob_type->tp_name == std::string("csr_matrix"))
                {
                    auto sequence = FromCSR(item, i++);
                    result.push_back(sequence);
                }
                Py_DECREF(item);
            }

        fail:
            Py_DECREF(iterator);
            Py_DECREF(pylist);
        }

        virtual void _GetSequence(size_t index, PyObject*) { NOT_IMPLEMENTED; }
    };
}