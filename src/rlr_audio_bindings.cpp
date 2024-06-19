#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "RLRAudioPropagation.h"

namespace py = pybind11;

std::vector<float> get_array(const float* arr) {
    return std::vector<float>(arr, arr + 3);
}

void set_array(float* arr, const std::vector<float>& value) {
    if (value.size() != 3) throw std::runtime_error("Array must have exactly 3 elements");
    std::copy(value.begin(), value.end(), arr);
}

void get_default_config(RLRA_ContextConfiguration& config) {
    config.thisSize = sizeof(RLRA_ContextConfiguration);
    RLRA_Error err = RLRA_ContextConfigurationDefault(&config);
    if (err != RLRA_Success) {
        throw std::runtime_error("Failed to create context");
    }
}

class Context {
public:
    Context() : context(nullptr) {}

    void create(const RLRA_ContextConfiguration& config) {
        RLRA_Error err = RLRA_CreateContext(&context, &config);
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to create context");
        }
    }

    void destroy() {
        if (context) {
            RLRA_Error err = RLRA_DestroyContext(context);
            if (err != RLRA_Success) {
                throw std::runtime_error("Failed to destroy context");
            }
            context = nullptr;
        }
    }

    void reset(const RLRA_ContextConfiguration& config) {
        RLRA_Error err = RLRA_ResetContext(context, &config);
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to reset context");
        }
    }

    void add_source() {
        RLRA_Error err = RLRA_AddSource(context);
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to add source");
        }
    }

    void clear_sources() {
        RLRA_Error err = RLRA_ClearSources(context);
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to clear sources");
        }
    }

    size_t get_source_count() const {
        return RLRA_GetSourceCount(context);
    }

    void set_source_position(size_t source_index, const std::array<float, 3>& position) {
        RLRA_Error err = RLRA_SetSourcePosition(context, source_index, position.data());
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to set source position");
        }
    }

    void set_source_radius(size_t source_index, float radius) {
        RLRA_Error err = RLRA_SetSourceRadius(context, source_index, radius);
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to set source radius");
        }
    }

    void add_listener(const RLRA_ChannelLayout& channel_layout) {
        RLRA_Error err = RLRA_AddListener(context, &channel_layout);
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to add listener");
        }
    }

    void clear_listeners() {
        RLRA_Error err = RLRA_ClearListeners(context);
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to clear listeners");
        }
    }

    size_t get_listener_count() const {
        return RLRA_GetListenerCount(context);
    }

    void set_listener_position(size_t listener_index, const std::array<float, 3>& position) {
        RLRA_Error err = RLRA_SetListenerPosition(context, listener_index, position.data());
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to set listener position");
        }
    }

    void set_listener_orientation_quaternion(size_t listener_index, const std::array<float, 4>& orientation) {
        RLRA_Error err = RLRA_SetListenerOrientationQuaternion(context, listener_index, orientation.data());
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to set listener orientation quaternion");
        }
    }

    void set_listener_radius(size_t listener_index, float radius) {
        RLRA_Error err = RLRA_SetListenerRadius(context, listener_index, radius);
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to set listener radius");
        }
    }

    void set_listener_hrtf(size_t listener_index, const std::string& hrtf_file_path) {
        RLRA_Error err = RLRA_SetListenerHRTF(context, listener_index, hrtf_file_path.c_str());
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to set listener HRTF");
        }
    }

    void add_object() {
        RLRA_Error err = RLRA_AddObject(context);
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to add object");
        }
    }

    void clear_objects() {
        RLRA_Error err = RLRA_ClearObjects(context);
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to clear objects");
        }
    }

    size_t get_object_count() const {
        return RLRA_GetObjectCount(context);
    }

    void set_object_position(size_t object_index, const std::array<float, 3>& position) {
        RLRA_Error err = RLRA_SetObjectPosition(context, object_index, position.data());
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to set object position");
        }
    }

    void set_object_orientation_quaternion(size_t object_index, const std::array<float, 4>& orientation) {
        RLRA_Error err = RLRA_SetObjectOrientationQuaternion(context, object_index, orientation.data());
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to set object orientation quaternion");
        }
    }

    void set_object_mesh_obj(size_t object_index, const std::string& obj_file_path, const std::string& material_category_name) {
        RLRA_Error err = RLRA_SetObjectMeshOBJ(context, object_index, obj_file_path.c_str(), material_category_name.c_str());
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to set object mesh OBJ");
        }
    }

    void set_object_mesh_ply(size_t object_index, const std::string& ply_file_path, const std::string& material_category_name) {
        RLRA_Error err = RLRA_SetObjectMeshPLY(context, object_index, ply_file_path.c_str(), material_category_name.c_str());
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to set object mesh PLY");
        }
    }

    void set_object_box(size_t object_index, const std::array<float, 3>& box_min, const std::array<float, 3>& box_max, const RLRA_BoxMaterialCategories* materials) {
        RLRA_Error err = RLRA_SetObjectBox(context, object_index, box_min.data(), box_max.data(), materials);
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to set object box");
        }
    }

    void add_mesh_vertices(const std::vector<float>& vertex_data) {
        RLRA_Error err = RLRA_AddMeshVertices(context, vertex_data.data(), vertex_data.size() / 3);
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to add mesh vertices");
        }
    }

    void add_mesh_indices(const std::vector<uint32_t>& index_data, size_t vertices_per_face, const std::string& material_category_name) {
        RLRA_Error err = RLRA_AddMeshIndices(context, index_data.data(), index_data.size(), vertices_per_face, material_category_name.c_str());
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to add mesh indices");
        }
    }

    void finalize_object_mesh(size_t object_index) {
        RLRA_Error err = RLRA_FinalizeObjectMesh(context, object_index);
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to finalize object mesh");
        }
    }

    void set_material_database_json(const std::string& json_path) {
        RLRA_Error err = RLRA_SetMaterialDatabaseJSON(context, json_path.c_str());
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to set material database JSON");
        }
    }

    void write_scene_mesh_obj(const std::string& output_path) const {
        RLRA_Error err = RLRA_WriteSceneMeshOBJ(context, output_path.c_str());
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to write scene mesh OBJ");
        }
    }

    void simulate() {
        RLRA_Error err = RLRA_Simulate(context);
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to simulate");
        }
    }

    size_t get_ir_count() const {
        return RLRA_GetIRCount(context);
    }

    size_t get_ir_channel_count(size_t listener_index, size_t source_index) const {
        return RLRA_GetIRChannelCount(context, listener_index, source_index);
    }

    size_t get_ir_sample_count(size_t listener_index, size_t source_index) const {
        return RLRA_GetIRSampleCount(context, listener_index, source_index);
    }

    std::vector<float> get_ir_channel(size_t listener_index, size_t source_index, size_t channel_index) const {
        size_t sample_count = get_ir_sample_count(listener_index, source_index);
        const float* data = RLRA_GetIRChannel(context, listener_index, source_index, channel_index);
        if (data == nullptr) {
            throw std::runtime_error("Failed to get IR channel");
        }
        return std::vector<float>(data, data + sample_count);
    }

    void write_ir_wave(size_t listener_index, size_t source_index, const std::string& output_file_path) const {
        RLRA_Error err = RLRA_WriteIRWave(context, listener_index, source_index, output_file_path.c_str());
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to write IR wave");
        }
    }

    void write_ir_metrics(size_t listener_index, size_t source_index, const std::string& output_file_path) const {
        RLRA_Error err = RLRA_WriteIRMetrics(context, listener_index, source_index, output_file_path.c_str());
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to write IR metrics");
        }
    }

    float get_indirect_ray_efficiency() const {
        return RLRA_GetIndirectRayEfficiency(context);
    }

    void trace_ray_any_hit(RLRA_Ray& ray) {
        RLRA_Error err = RLRA_TraceRayAnyHit(context, &ray);
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to trace ray any hit");
        }
    }

    void trace_ray_first_hit(RLRA_Ray& ray) {
        RLRA_Error err = RLRA_TraceRayFirstHit(context, &ray);
        if (err != RLRA_Success) {
            throw std::runtime_error("Failed to trace ray first hit");
        }
    }

private:
    RLRA_Context context;
};

std::string config_repr(const RLRA_ContextConfiguration& config) {
    std::ostringstream oss;
    oss << "Config("
        << "  \n  frequency_bands=" << config.frequencyBands
        << ", \n  direct_sh_order=" << config.directSHOrder
        << ", \n  indirect_sh_order=" << config.indirectSHOrder
        << ", \n  direct_ray_count=" << config.directRayCount
        << ", \n  indirect_ray_count=" << config.indirectRayCount
        << ", \n  indirect_ray_depth=" << config.indirectRayDepth
        << ", \n  source_ray_count=" << config.sourceRayCount
        << ", \n  source_ray_depth=" << config.sourceRayDepth
        << ", \n  max_diffraction_order=" << config.maxDiffractionOrder
        << ", \n  thread_count=" << config.threadCount
        << ", \n  sample_rate=" << config.sampleRate
        << ", \n  max_ir_length=" << config.maxIRLength
        << ", \n  unit_scale=" << config.unitScale
        << ", \n  global_volume=" << config.globalVolume
        << ", \n  hrtf_right=[" << config.hrtfRight[0] << ", " << config.hrtfRight[1] << ", " << config.hrtfRight[2] << "]"
        << ", \n  hrtf_up=[" << config.hrtfUp[0] << ", " << config.hrtfUp[1] << ", " << config.hrtfUp[2] << "]"
        << ", \n  hrtf_back=[" << config.hrtfBack[0] << ", " << config.hrtfBack[1] << ", " << config.hrtfBack[2] << "]"
        << ", \n  direct=" << (config.direct ? "True" : "False")
        << ", \n  indirect=" << (config.indirect ? "True" : "False")
        << ", \n  diffraction=" << (config.diffraction ? "True" : "False")
        << ", \n  transmission=" << (config.transmission ? "True" : "False")
        << ", \n  mesh_simplification=" << (config.meshSimplification ? "True" : "False")
        << ", \n  temporal_coherence=" << (config.temporalCoherence ? "True" : "False")
        << "\n)";
    return oss.str();
}

PYBIND11_MODULE(_rlr_audio_propagation, m) {
    m.doc() = "Python bindings for RLRAudioPropagation module";

    // // Boolean type
    // py::enum_<RLR_Bool>(m, "Bool")
    //     .value("False", RLR_Bool(0))
    //     .value("True", RLR_Bool(1))
    //     .export_values();

    // Error codes
    py::enum_<RLRA_Error>(m, "Error")
        .value("Success", RLRA_Success)
        .value("Unknown", RLRA_Error_Unknown)
        .value("InvalidParam", RLRA_Error_InvalidParam)
        .value("BadSampleRate", RLRA_Error_BadSampleRate)
        .value("MissingDLL", RLRA_Error_MissingDLL)
        .value("BadAlignment", RLRA_Error_BadAlignment)
        .value("Uninitialized", RLRA_Error_Uninitialized)
        .value("BadAlloc", RLRA_Error_BadAlloc)
        .value("UnsupportedFeature", RLRA_Error_UnsupportedFeature)
        .value("InternalEnd", RLRA_Error_InternalEnd)
        .export_values();

    // Channel layout type
    py::enum_<RLRA_ChannelLayoutType>(m, "ChannelLayoutType")
        .value("Unknown", RLRA_ChannelLayoutType_Unknown)
        .value("Mono", RLRA_ChannelLayoutType_Mono)
        .value("Binaural", RLRA_ChannelLayoutType_Binaural)
        .value("Ambisonics", RLRA_ChannelLayoutType_Ambisonics)
        .export_values();

    // Channel layout
    py::class_<RLRA_ChannelLayout>(m, "ChannelLayout")
        .def(py::init([](RLRA_ChannelLayoutType &type, int &channelCount) {
            RLRA_ChannelLayout layout;
            layout.channelCount = channelCount;
            layout.type = type;
            return layout;
        }))
        .def_readwrite("channel_count", &RLRA_ChannelLayout::channelCount)
        .def_readwrite("type", &RLRA_ChannelLayout::type);

    // Context configuration
    py::class_<RLRA_ContextConfiguration>(m, "Config")
        .def(py::init([]() {
            RLRA_ContextConfiguration config;
            get_default_config(config);
            return config;
        }))
        .def_readwrite("size", &RLRA_ContextConfiguration::thisSize)
        .def_readwrite("frequency_bands", &RLRA_ContextConfiguration::frequencyBands)
        .def_readwrite("direct_sh_order", &RLRA_ContextConfiguration::directSHOrder)
        .def_readwrite("indirect_sh_order", &RLRA_ContextConfiguration::indirectSHOrder)
        .def_readwrite("direct_ray_count", &RLRA_ContextConfiguration::directRayCount)
        .def_readwrite("indirect_ray_count", &RLRA_ContextConfiguration::indirectRayCount)
        .def_readwrite("indirect_ray_depth", &RLRA_ContextConfiguration::indirectRayDepth)
        .def_readwrite("source_ray_count", &RLRA_ContextConfiguration::sourceRayCount)
        .def_readwrite("source_ray_depth", &RLRA_ContextConfiguration::sourceRayDepth)
        .def_readwrite("max_diffraction_order", &RLRA_ContextConfiguration::maxDiffractionOrder)
        .def_readwrite("thread_count", &RLRA_ContextConfiguration::threadCount)
        .def_readwrite("sample_rate", &RLRA_ContextConfiguration::sampleRate)
        .def_readwrite("max_ir_length", &RLRA_ContextConfiguration::maxIRLength)
        .def_readwrite("unit_scale", &RLRA_ContextConfiguration::unitScale)
        .def_readwrite("global_volume", &RLRA_ContextConfiguration::globalVolume)
        .def_property("hrtf_right",
                      [](const RLRA_ContextConfiguration &c) { return get_array(c.hrtfRight); },
                      [](RLRA_ContextConfiguration &c, const std::vector<float> &value) { set_array(c.hrtfRight, value); })
        .def_property("hrtf_up",
                      [](const RLRA_ContextConfiguration &c) { return get_array(c.hrtfUp); },
                      [](RLRA_ContextConfiguration &c, const std::vector<float> &value) { set_array(c.hrtfUp, value); })
        .def_property("hrtf_back",
                      [](const RLRA_ContextConfiguration &c) { return get_array(c.hrtfBack); },
                      [](RLRA_ContextConfiguration &c, const std::vector<float> &value) { set_array(c.hrtfBack, value); })
        .def_readwrite("direct", &RLRA_ContextConfiguration::direct)
        .def_readwrite("indirect", &RLRA_ContextConfiguration::indirect)
        .def_readwrite("diffraction", &RLRA_ContextConfiguration::diffraction)
        .def_readwrite("transmission", &RLRA_ContextConfiguration::transmission)
        .def_readwrite("mesh_simplification", &RLRA_ContextConfiguration::meshSimplification)
        .def_readwrite("temporal_coherence", &RLRA_ContextConfiguration::temporalCoherence)
        .def("__repr__", &config_repr);

    // Box material categories
    py::class_<RLRA_BoxMaterialCategories>(m, "BoxMaterialCategories")
        .def(py::init<>())
        .def_readwrite("x_min", &RLRA_BoxMaterialCategories::xMin)
        .def_readwrite("x_max", &RLRA_BoxMaterialCategories::xMax)
        .def_readwrite("y_min", &RLRA_BoxMaterialCategories::yMin)
        .def_readwrite("y_max", &RLRA_BoxMaterialCategories::yMax)
        .def_readwrite("z_min", &RLRA_BoxMaterialCategories::zMin)
        .def_readwrite("z_max", &RLRA_BoxMaterialCategories::zMax);

    // Ray
    py::class_<RLRA_Ray>(m, "Ray")
        .def(py::init<>())
        .def_property("origin",
                      [](const RLRA_Ray &r) { return get_array(r.origin); },
                      [](RLRA_Ray &r, const std::vector<float> &value) { set_array(r.origin, value); })
        .def_property("direction",
                      [](const RLRA_Ray &r) { return get_array(r.direction); },
                      [](RLRA_Ray &r, const std::vector<float> &value) { set_array(r.direction, value); })
        .def_readwrite("t_min", &RLRA_Ray::tMin)
        .def_readwrite("t_max", &RLRA_Ray::tMax)
        .def_property("hit",
                  [](const RLRA_Ray &r) { return r.hit == RLRA_RayHit_True; },
                  [](RLRA_Ray &r, bool value) { r.hit = value ? RLRA_RayHit_True : RLRA_RayHit_False; })
        .def_property("normal",
                      [](const RLRA_Ray &r) { return get_array(r.normal); },
                      [](RLRA_Ray &r, const std::vector<float> &value) { set_array(r.normal, value); });



    m.def("get_default_config", &RLRA_ContextConfigurationDefault,
          py::arg("config"),
          "Initialize a context configuration with the default parameters.");

    py::class_<Context>(m, "Context")
        .def(py::init<>())
        .def("__init__", [](Context &self, const RLRA_ContextConfiguration& config) {
            new (&self) Context();
            self.create(config);
        })
        .def("__del__", &Context::destroy)
        .def("reset", &Context::reset, py::arg("config"))
        // sources
        .def("add_source", &Context::add_source)
        .def("clear_sources", &Context::clear_sources)
        .def("get_source_count", &Context::get_source_count)
        .def("set_source_position", &Context::set_source_position, py::arg("source_index"), py::arg("position"))
        .def("set_source_radius", &Context::set_source_radius, py::arg("source_index"), py::arg("radius"))
        // listeners
        .def("add_listener", &Context::add_listener, py::arg("channel_layout"))
        .def("clear_listeners", &Context::clear_listeners)
        .def("get_listener_count", &Context::get_listener_count)
        .def("set_listener_position", &Context::set_listener_position, py::arg("listener_index"), py::arg("position"))
        .def("set_listener_orientation_quaternion", &Context::set_listener_orientation_quaternion, py::arg("listener_index"), py::arg("orientation"))
        .def("set_listener_radius", &Context::set_listener_radius, py::arg("listener_index"), py::arg("radius"))
        .def("set_listener_hrtf", &Context::set_listener_hrtf, py::arg("listener_index"), py::arg("hrtf_file_path"))
        // objects
        .def("add_object", &Context::add_object)
        .def("clear_objects", &Context::clear_objects)
        .def("get_object_count", &Context::get_object_count)
        .def("set_object_position", &Context::set_object_position, py::arg("object_index"), py::arg("position"))
        .def("set_object_orientation_quaternion", &Context::set_object_orientation_quaternion, py::arg("object_index"), py::arg("orientation"))
        .def("set_object_mesh_obj", &Context::set_object_mesh_obj, py::arg("object_index"), py::arg("obj_file_path"), py::arg("material_category_name"))
        .def("set_object_mesh_ply", &Context::set_object_mesh_ply, py::arg("object_index"), py::arg("ply_file_path"), py::arg("material_category_name"))
        .def("set_object_box", &Context::set_object_box, py::arg("object_index"), py::arg("box_min"), py::arg("box_max"), py::arg("materials"))
        .def("finalize_object_mesh", &Context::finalize_object_mesh, py::arg("object_index"))
        // mesh
        .def("add_mesh_vertices", &Context::add_mesh_vertices, py::arg("vertex_data"))
        .def("add_mesh_indices", &Context::add_mesh_indices, py::arg("index_data"), py::arg("vertices_per_face"), py::arg("material_category_name"))
        .def("set_material_database_json", &Context::set_material_database_json, py::arg("json_path"))
        .def("write_scene_mesh_obj", &Context::write_scene_mesh_obj, py::arg("output_path"))
        // simulation
        .def("simulate", &Context::simulate)
        // results
        .def("get_ir_count", &Context::get_ir_count)
        .def("get_ir_channel_count", &Context::get_ir_channel_count, py::arg("listener_index"), py::arg("source_index"))
        .def("get_ir_sample_count", &Context::get_ir_sample_count, py::arg("listener_index"), py::arg("source_index"))
        .def("get_ir_channel", &Context::get_ir_channel, py::arg("listener_index"), py::arg("source_index"), py::arg("channel_index"))
        .def("write_ir_wave", &Context::write_ir_wave, py::arg("listener_index"), py::arg("source_index"), py::arg("output_file_path"))
        .def("write_ir_metrics", &Context::write_ir_metrics, py::arg("listener_index"), py::arg("source_index"), py::arg("output_file_path"))
        .def("get_indirect_ray_efficiency", &Context::get_indirect_ray_efficiency)
        .def("trace_ray_any_hit", &Context::trace_ray_any_hit, py::arg("ray"))
        .def("trace_ray_first_hit", &Context::trace_ray_first_hit, py::arg("ray"));

        // m.def("add_source", [](RLRA_Context& context) {
        //     RLRA_Error err = RLRA_AddSource(context);
        //     if (err != RLRA_Success) {
        //         throw std::runtime_error("Failed to add source");
        //     }
        // }, py::arg("context"));

        // m.def("clear_sources", [](RLRA_Context& context) {
        //     RLRA_Error err = RLRA_ClearSources(context);
        //     if (err != RLRA_Success) {
        //         throw std::runtime_error("Failed to clear sources");
        //     }
        // }, py::arg("context"));

        // m.def("get_source_count", [](const RLRA_Context& context) {
        //     return RLRA_GetSourceCount(context);
        // }, py::arg("context"));

        // m.def("set_source_position", [](RLRA_Context& context, size_t sourceIndex, const std::array<float, 3>& position) {
        //     RLRA_Error err = RLRA_SetSourcePosition(context, sourceIndex, position.data());
        //     if (err != RLRA_Success) {
        //         throw std::runtime_error("Failed to set source position");
        //     }
        // }, py::arg("context"), py::arg("source_index"), py::arg("position"));

        // m.def("set_source_radius", [](RLRA_Context& context, size_t sourceIndex, float radius) {
        //     RLRA_Error err = RLRA_SetSourceRadius(context, sourceIndex, radius);
        //     if (err != RLRA_Success) {
        //         throw std::runtime_error("Failed to set source radius");
        //     }
        // }, py::arg("context"), py::arg("source_index"), py::arg("radius"));
}
