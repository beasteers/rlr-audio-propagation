#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "RLRAudioPropagation.h"

namespace py = pybind11;

RLRAudioPropagation::VertexData as_vertex_data(py::array_t<float, py::array::c_style | py::array::forcecast> array) {
    py::buffer_info info = array.request();
    if (info.ndim != 2 || info.shape[1] != 3 || info.format != py::format_descriptor<float>::format()) {
        std::ostringstream oss;
        oss << "Invalid buffer format! Got array with shape (";
        for (size_t i = 0; i < info.ndim; ++i) {
            if (i > 0) oss << ", ";
            oss << info.shape[i];
        }
        oss << ") and format " << info.format << ". Expected a 2D array with shape (N, 3) and float data type.";
        throw std::runtime_error(oss.str());
    }
    RLRAudioPropagation::VertexData vertex_data;
    vertex_data.vertices = info.ptr;
    vertex_data.vertexCount = info.shape[0];
    vertex_data.vertexStride = sizeof(float) * 3;
    vertex_data.byteOffset = 0; // Assuming no byte offset for simplicity
    return vertex_data;
}

void bind_RLRAudioPropagation(py::module &m) {
    // Bind the RLRAudioPropagation::ErrorCodes enum
    py::enum_<RLRAudioPropagation::ErrorCodes>(m, "ErrorCodes")
        .value("Success", RLRAudioPropagation::ErrorCodes::Success)
        .value("Unknown", RLRAudioPropagation::ErrorCodes::Unknown)
        .value("InvalidParam", RLRAudioPropagation::ErrorCodes::InvalidParam)
        .value("BadSampleRate", RLRAudioPropagation::ErrorCodes::BadSampleRate)
        .value("MissingDLL", RLRAudioPropagation::ErrorCodes::MissingDLL)
        .value("BadAlignment", RLRAudioPropagation::ErrorCodes::BadAlignment)
        .value("Uninitialized", RLRAudioPropagation::ErrorCodes::Uninitialized)
        .value("HRTFInitFailure", RLRAudioPropagation::ErrorCodes::HRTFInitFailure)
        .value("BadVersion", RLRAudioPropagation::ErrorCodes::BadVersion)
        .value("SymbolNotFound", RLRAudioPropagation::ErrorCodes::SymbolNotFound)
        .value("SharedReverbDisabled", RLRAudioPropagation::ErrorCodes::SharedReverbDisabled)
        .value("NoAvailableAmbisonicInstance", RLRAudioPropagation::ErrorCodes::NoAvailableAmbisonicInstance)
        .value("MemoryAllocFailure", RLRAudioPropagation::ErrorCodes::MemoryAllocFailure)
        .value("UnsupportedFeature", RLRAudioPropagation::ErrorCodes::UnsupportedFeature)
        .value("InternalEnd", RLRAudioPropagation::ErrorCodes::InternalEnd);


    // Bind the RLRAudioPropagation::Simulator class
    py::class_<RLRAudioPropagation::Simulator>(m, "Simulator")
        .def(py::init<>())
        .def("Configure", &RLRAudioPropagation::Simulator::Configure)
        .def("GetRayEfficiency", &RLRAudioPropagation::Simulator::GetRayEfficiency)
        .def("GetChannelCount", &RLRAudioPropagation::Simulator::GetChannelCount)
        .def("GetSampleCount", &RLRAudioPropagation::Simulator::GetSampleCount)
        .def("LoadAudioMaterialJSON", &RLRAudioPropagation::Simulator::LoadAudioMaterialJSON)
        // .def("LoadMesh", static_cast<RLRAudioPropagation::ErrorCodes (RLRAudioPropagation::Simulator::*)(const std::string&)>(&RLRAudioPropagation::Simulator::LoadMesh))
        // .def("LoadMesh", static_cast<RLRAudioPropagation::ErrorCodes (RLRAudioPropagation::Simulator::*)(const std::string&, const std::string&)>(&RLRAudioPropagation::Simulator::LoadMesh))
        .def("LoadMeshData", &RLRAudioPropagation::Simulator::LoadMeshData)
        // .def("LoadMeshDataWithMaterial", &RLRAudioPropagation::Simulator::LoadMeshDataWithMaterial)
        .def("LoadMeshVertices", &RLRAudioPropagation::Simulator::LoadMeshVertices)
        .def("LoadMeshIndices", &RLRAudioPropagation::Simulator::LoadMeshIndices)
        .def("UploadMesh", &RLRAudioPropagation::Simulator::UploadMesh)
        .def("AddListener", &RLRAudioPropagation::Simulator::AddListener)
        .def("AddSource", &RLRAudioPropagation::Simulator::AddSource)
        .def("RunSimulation", &RLRAudioPropagation::Simulator::RunSimulation)
        .def("GetImpulseResponse", &RLRAudioPropagation::Simulator::GetImpulseResponse)
        .def("GetImpulseResponseForChannel", &RLRAudioPropagation::Simulator::GetImpulseResponseForChannel);

    // Bind the RLRAudioPropagation::Configuration struct
    py::class_<RLRAudioPropagation::Configuration>(m, "Configuration")
        .def(py::init<>())
        .def_readwrite("sampleRate", &RLRAudioPropagation::Configuration::sampleRate)
        .def_readwrite("frequencyBands", &RLRAudioPropagation::Configuration::frequencyBands)
        .def_readwrite("directSHOrder", &RLRAudioPropagation::Configuration::directSHOrder)
        .def_readwrite("indirectSHOrder", &RLRAudioPropagation::Configuration::indirectSHOrder)
        .def_readwrite("threadCount", &RLRAudioPropagation::Configuration::threadCount)
        .def_readwrite("updateDt", &RLRAudioPropagation::Configuration::updateDt)
        .def_readwrite("irTime", &RLRAudioPropagation::Configuration::irTime)
        .def_readwrite("unitScale", &RLRAudioPropagation::Configuration::unitScale)
        .def_readwrite("globalVolume", &RLRAudioPropagation::Configuration::globalVolume)
        .def_readwrite("listenerRadius", &RLRAudioPropagation::Configuration::listenerRadius)
        .def_readwrite("indirectRayCount", &RLRAudioPropagation::Configuration::indirectRayCount)
        .def_readwrite("indirectRayDepth", &RLRAudioPropagation::Configuration::indirectRayDepth)
        .def_readwrite("sourceRayCount", &RLRAudioPropagation::Configuration::sourceRayCount)
        .def_readwrite("sourceRayDepth", &RLRAudioPropagation::Configuration::sourceRayDepth)
        .def_readwrite("maxDiffractionOrder", &RLRAudioPropagation::Configuration::maxDiffractionOrder)
        .def_readwrite("direct", &RLRAudioPropagation::Configuration::direct)
        .def_readwrite("indirect", &RLRAudioPropagation::Configuration::indirect)
        .def_readwrite("diffraction", &RLRAudioPropagation::Configuration::diffraction)
        .def_readwrite("transmission", &RLRAudioPropagation::Configuration::transmission)
        .def_readwrite("meshSimplification", &RLRAudioPropagation::Configuration::meshSimplification)
        .def_readwrite("temporalCoherence", &RLRAudioPropagation::Configuration::temporalCoherence)
        .def_readwrite("dumpWaveFiles", &RLRAudioPropagation::Configuration::dumpWaveFiles)
        .def_readwrite("enableMaterials", &RLRAudioPropagation::Configuration::enableMaterials)
        .def_readwrite("writeIrToFile", &RLRAudioPropagation::Configuration::writeIrToFile);

    // Bind other necessary RLRAudioPropagation classes and structs if any
    py::class_<RLRAudioPropagation::Vector3f>(m, "Vector3f")
        .def(py::init<float, float, float>())
        .def_property("x", [](const RLRAudioPropagation::Vector3f &v) { return v.x; }, [](RLRAudioPropagation::Vector3f &v, float x) { v.x = x; })
        .def_property("y", [](const RLRAudioPropagation::Vector3f &v) { return v.y; }, [](RLRAudioPropagation::Vector3f &v, float y) { v.y = y; })
        .def_property("z", [](const RLRAudioPropagation::Vector3f &v) { return v.z; }, [](RLRAudioPropagation::Vector3f &v, float z) { v.z = z; })
        .def("__getitem__", [](const RLRAudioPropagation::Vector3f &v, int i) {
            if (i == 0) return v.x;
            else if (i == 1) return v.y;
            else if (i == 2) return v.z;
            else throw py::index_error();
        });

    py::class_<RLRAudioPropagation::Quaternion>(m, "Quaternion")
        .def(py::init<float, float, float, float>())
        .def_property("w", [](const RLRAudioPropagation::Quaternion &q) { return q.s; }, [](RLRAudioPropagation::Quaternion &q, float w) { q.s = w; })
        .def_property("x", [](const RLRAudioPropagation::Quaternion &q) { return q.x; }, [](RLRAudioPropagation::Quaternion &q, float x) { q.x = x; })
        .def_property("y", [](const RLRAudioPropagation::Quaternion &q) { return q.y; }, [](RLRAudioPropagation::Quaternion &q, float y) { q.y = y; })
        .def_property("z", [](const RLRAudioPropagation::Quaternion &q) { return q.z; }, [](RLRAudioPropagation::Quaternion &q, float z) { q.z = z; })
        .def("__getitem__", [](const RLRAudioPropagation::Quaternion &q, int i) {
            if (i == 0) return q.s;
            else if (i == 1) return q.x;
            else if (i == 2) return q.y;
            else if (i == 3) return q.z;
            else throw py::index_error();
        });

    // Bind the RLRAudioPropagation::ChannelLayout struct
    py::class_<RLRAudioPropagation::ChannelLayout>(m, "ChannelLayout")
        .def(py::init<>())
        .def_readwrite("channelType", &RLRAudioPropagation::ChannelLayout::channelType)
        .def_readwrite("channelCount", &RLRAudioPropagation::ChannelLayout::channelCount);



    // Bind the RLRAudioPropagation::VertexData struct
    py::class_<RLRAudioPropagation::VertexData>(m, "VertexData")
        .def(py::init<>())
        // .def_readwrite("vertices", &RLRAudioPropagation::VertexData::vertices)
        .def_readwrite("byteOffset", &RLRAudioPropagation::VertexData::byteOffset)
        .def_readwrite("vertexCount", &RLRAudioPropagation::VertexData::vertexCount)
        .def_readwrite("vertexStride", &RLRAudioPropagation::VertexData::vertexStride);

    // Add the numpy_to_vertex_data function
    m.def("as_vertex_data", &as_vertex_data, "Convert a NumPy array to VertexData");

    // Bind the RLRAudioPropagation::IndexData struct
    py::class_<RLRAudioPropagation::IndexData>(m, "IndexData")
        .def(py::init<>())
        .def_readwrite("indices", &RLRAudioPropagation::IndexData::indices)
        .def_readwrite("byteOffset", &RLRAudioPropagation::IndexData::byteOffset)
        .def_readwrite("indexCount", &RLRAudioPropagation::IndexData::indexCount);
// DLTensor dltensor;
//     auto pybind_capsule= py::capsule(&dltensor,"dltensor",nullptr);
//     return pybind_capsule; 
    // // Bind the RLRAudioPropagation::VertexData struct
    // py::class_<RLRAudioPropagation::VertexData>(m, "VertexData")
    //     .def(py::init<>())
    //     .def_property("vertices", [](RLRAudioPropagation::VertexData &v) { 
    //         return py::memoryview::from_buffer(static_cast<float*>(v.vertices), {v.vertexCount, 3}, {sizeof(float) * 3, sizeof(float)});
    //     }, [](RLRAudioPropagation::VertexData &v, py::buffer b) {
    //         py::buffer_info info = b.request();
    //         if (info.ndim != 2 || info.shape[1] != 3 || info.format != py::format_descriptor<float>::format()) {
    //             throw std::runtime_error("Invalid buffer format!");
    //         }
    //         v.vertices = info.ptr;
    //         v.vertexCount = info.shape[0];
    //         v.vertexStride = sizeof(float) * 3;
    //     })
    //     .def_readwrite("byteOffset", &RLRAudioPropagation::VertexData::byteOffset)
    //     .def_readwrite("vertexCount", &RLRAudioPropagation::VertexData::vertexCount)
    //     .def_readwrite("vertexStride", &RLRAudioPropagation::VertexData::vertexStride);

    // // Bind the RLRAudioPropagation::IndexData struct
    // py::class_<RLRAudioPropagation::IndexData>(m, "IndexData")
    //     .def(py::init<>())
    //     .def_property("indices", [](RLRAudioPropagation::IndexData &i) { 
    //         return py::memoryview::from_buffer(static_cast<uint32_t*>(i.indices), {i.indexCount}, {sizeof(uint32_t)});
    //     }, [](RLRAudioPropagation::IndexData &i, py::buffer b) {
    //         py::buffer_info info = b.request();
    //         if (info.ndim != 1 || info.format != py::format_descriptor<uint32_t>::format()) {
    //             throw std::runtime_error("Invalid buffer format!");
    //         }
    //         i.indices = info.ptr;
    //         i.indexCount = info.shape[0];
    //     })
    //     .def_readwrite("byteOffset", &RLRAudioPropagation::IndexData::byteOffset)
    //     .def_readwrite("indexCount", &RLRAudioPropagation::IndexData::indexCount);


}

PYBIND11_MODULE(RLRAudioPropagation, m) {
    m.doc() = "Python bindings for RLRAudioPropagation";
    bind_RLRAudioPropagation(m);
}
