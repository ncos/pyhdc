#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include <iostream>
#include <random>

#include <Python.h>
#include "structmember.h"


#define QUOTEME(M)  #M
#define INCLUDE_FILE(M) QUOTEME(M)

#include INCLUDE_FILE(HNAME)


// LBV structure
typedef struct {
    PyObject_HEAD
    uint32_t *data;
} LBV;

static PyObject *LBV_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    LBV *v = (LBV *)type->tp_alloc(type, 0);
    v->data = (uint32_t *)calloc(VECTOR_WIDTH / BIT_WIDTH, sizeof(uint32_t));   
    return (PyObject *)v;
}

static void LBV_dealloc(LBV *v) {
    free(v->data);
    Py_TYPE(v)->tp_free((PyObject*)v);
}

static int LBV_init(LBV *v, PyObject *args, PyObject *kwds) {
    return 0;
}

static PyObject *LBV_repr(LBV *v) {
    std::string ret = "";
    for (uint32_t j = 0; j < VECTOR_WIDTH / BIT_WIDTH; ++j) {
        const uint32_t chunk = v->data[j];
        for (uint8_t i = 0; i < BIT_WIDTH; ++i) {
            const uint32_t bit_src = (chunk >> (BIT_WIDTH - i - 1)) & 0x1;
            ret += std::to_string(bit_src);
        }
        ret += '_';
    }

    return PyUnicode_FromFormat(ret.c_str());
}

static PyObject *LBV_randomize(LBV *v, PyObject *Py_UNUSED(ignored)) {
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> gen(0, 255);

    uint8_t *data = (uint8_t *)v->data;
    for (uint32_t i = 0; i < VECTOR_WIDTH / 8; ++i) {
        data[i] = gen(rng);
    }

    Py_RETURN_NONE;
}

static PyObject *LBV_is_zero(LBV *v, PyObject *Py_UNUSED(ignored)) {
    for (uint32_t i = 0; i < VECTOR_WIDTH / BIT_WIDTH; ++i) {
        if (v->data[i] != 0) {
            Py_RETURN_FALSE;
        }
    }

    Py_RETURN_TRUE;
}


static PyMemberDef LBV_members[] = {
    {NULL}  /* Sentinel */
};


static PyObject *LBV_xor(LBV *v1, PyObject *args);
static PyObject *LBV_permute(LBV* v, PyObject* args, PyObject *kwds);
static PyObject* LBV_inv_permute(LBV *v, PyObject* args, PyObject *kwds);
static PyMethodDef LBV_methods[] = {
    {"rand", (PyCFunction) LBV_randomize, METH_NOARGS,
     "Set vector elements to random values"
    },
    {"is_zero", (PyCFunction) LBV_is_zero, METH_NOARGS,
     "Check if all bits are zero"
    },
    {"xor", (PyCFunction) LBV_xor, METH_VARARGS,
     "xor this vector with another vector and store the result in the current vector"
    },
    {"permute", (PyCFunction) LBV_permute, METH_VARARGS,
     "permute a given vector; the permutation is specified by its set ('x', 'y', etc) and order"
    },
    {"inv_permute", (PyCFunction) LBV_inv_permute, METH_VARARGS,
     "permute a given vector but inverse the permutation; the permutation is specified by its set ('x', 'y', etc) and order"
    },
    {NULL}  /* Sentinel */
};


static PyTypeObject LBVType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "pyhdc.LBV",               /* tp_name */
    sizeof(LBV),               /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)LBV_dealloc,   /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_compare */
    (reprfunc)LBV_repr,        /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash */
    0,                         /* tp_call */
    (reprfunc)LBV_repr,        /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,   /* tp_flags */
    "Long Binary Vector",      /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    LBV_methods,               /* tp_methods */
    LBV_members,               /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)LBV_init,        /* tp_init */
    0,                         /* tp_alloc */
    LBV_new,                   /* tp_new */
};

// =================


// XOR
static PyObject *LBV_xor(LBV *v1, PyObject *args) {
    LBV *v2;
    if (!PyArg_ParseTuple(args, "O!", &LBVType, &v2))
        return NULL;

    for (uint32_t i = 0; i < VECTOR_WIDTH / BIT_WIDTH; ++i) {
        v1->data[i] = v1->data[i] ^ v2->data[i];
    }

    Py_RETURN_NONE;
}

// Permutation
void permute_chunk(LBV *v, const uint8_t p[][2], uint32_t id, LBV *ref) {
    for (uint8_t i = 0; i < BIT_WIDTH; ++i) {
        const uint8_t chunk_id = p[id * BIT_WIDTH + i][0];
        const uint8_t bit_id   = p[id * BIT_WIDTH + i][1];
        const uint32_t source  = ref->data[id];
        const uint32_t target  = ref->data[chunk_id];


        const uint32_t mask = (((source >> (BIT_WIDTH - i - 1)) ^ (target >> (BIT_WIDTH - bit_id - 1))) & 0x1) << (BIT_WIDTH - i - 1);
        v->data[id] ^= mask;


        //const uint32_t bit_src = (source >> (BIT_WIDTH - i - 1)) & 0x1;
        //const uint32_t bit_dst = (target >> (BIT_WIDTH - bit_id - 1)) & 0x1;
        //std::cout << (uint32_t)id << ", " << (uint32_t)i << " (" << bit_src << ") -> " 
        //          << (uint32_t)chunk_id << " " << (uint32_t)bit_id << " ("<< bit_dst << ") " << mask << "\n";
    }
}


void permute(LBV *v, const uint8_t p[][2]) {
    LBV ref;
    ref.data = (uint32_t *)malloc(VECTOR_WIDTH / 8);   
    memcpy(ref.data, v->data, VECTOR_WIDTH / 8);

    for (uint32_t i = 0; i < VECTOR_WIDTH / BIT_WIDTH; ++i) {
        permute_chunk(v, p, i, &ref);
    }

    free(ref.data);
}

static PyObject* LBV_permute(LBV *v, PyObject* args, PyObject *kwds) {
    static char *kwlist[] = {"axis", "order", NULL};
    const char *axis;
    int order = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "si", kwlist,
                                     &axis, &order))
        return NULL;
    
    if (order < 0)
        return NULL;

    if (axis[0] == 'x') {
        if (order >= DEPTH_X) return NULL;
        permute(v, Px[order]);
    } else if (axis[0] == 'y') {
        if (order >= DEPTH_Y) return NULL;
        permute(v, Py[order]);
    } else {
        return NULL;
    }

    Py_RETURN_NONE;
}


void inv_permute(LBV *v, const uint8_t p[][2]) {
    uint8_t p_inv[VECTOR_WIDTH][2];
    for (uint32_t i = 0; i < VECTOR_WIDTH; ++i) {
        uint32_t id = p[i][0] * BIT_WIDTH + p[i][1];
        p_inv[id][0] = i / BIT_WIDTH;
        p_inv[id][1] = i % BIT_WIDTH;
    }

    LBV ref;
    ref.data = (uint32_t *)malloc(VECTOR_WIDTH / 8);   
    memcpy(ref.data, v->data, VECTOR_WIDTH / 8);

    for (uint32_t i = 0; i < VECTOR_WIDTH / BIT_WIDTH; ++i) {
        permute_chunk(v, p_inv, i, &ref);
    }

    free(ref.data);
}


static PyObject* LBV_inv_permute(LBV *v, PyObject* args, PyObject *kwds) {
    static char *kwlist[] = {"axis", "order", NULL};
    const char *axis;
    int order = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "si", kwlist,
                                     &axis, &order))
        return NULL;
    
    if (order < 0)
        return NULL;

    if (axis[0] == 'x') {
        if (order >= DEPTH_X) return NULL;
        inv_permute(v, Px[order]);
    } else if (axis[0] == 'y') {
        if (order >= DEPTH_Y) return NULL;
        inv_permute(v, Py[order]);
    } else {
        return NULL;
    }

    Py_RETURN_NONE;
}

// Boilerplate
int main() {

    return 0;
}


inline std::string perm_to_str(const uint8_t P[][2]) {
    std::string ret = "";
    for (uint32_t i = 0; i < VECTOR_WIDTH; ++i) {
        ret += std::to_string(P[i][0] * BIT_WIDTH + P[i][1]) + " ";
    }
    return ret;
}


static PyObject* permutation_to_str(PyObject* self, PyObject* args, PyObject *kwds) {
    static char *kwlist[] = {"axis", "order", NULL};
    const char *axis;
    int order = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "si", kwlist,
                                     &axis, &order))
        return NULL;
    
    if (order < 0)
        return NULL;

    std::string ret = "";
    if (axis[0] == 'x') {
        if (order >= DEPTH_X) return NULL;
        ret = perm_to_str(Px[order]);
    } else if (axis[0] == 'y') {
        if (order >= DEPTH_Y) return NULL;
        ret = perm_to_str(Py[order]);
    } else {
        return NULL;
    }

    return PyUnicode_FromFormat(ret.c_str());
}


static PyObject* get_max_order(PyObject* self, PyObject* args, PyObject *kwds) {
    static char *kwlist[] = {"axis", NULL};
    const char *axis;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist,
                                     &axis))
        return NULL;
    
    int max_order = 0;
    if (axis[0] == 'x') {
        max_order = DEPTH_X - 1;
    } else if (axis[0] == 'y') {
        max_order = DEPTH_Y - 1;
    } else {
        return NULL;
    }

    return PyLong_FromLong(max_order); 
}


static PyObject* get_vector_width(PyObject* self, PyObject* args, PyObject *kwds) {
    return PyLong_FromLong(VECTOR_WIDTH); 
}


static PyMethodDef module_methods[] = {
    {"permutation_to_str", (PyCFunction)permutation_to_str, METH_VARARGS,
     "convert permutation to string, for printing"},
    {"get_max_order", (PyCFunction)get_max_order, METH_VARARGS,
     "return the maximum order for a given set of permutations"},
    {"get_vector_width", (PyCFunction)get_vector_width, METH_NOARGS,
     "retutn vector with the library was compiled for"},
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif


static struct PyModuleDef cModPyDem = {
    PyModuleDef_HEAD_INIT,
    "pyhdc",    /* name of module */
    "",         /* module documentation, may be NULL */
    -1,         /* size of per-interpreter state of the module, or -1 if the module keeps state in
                   global variables. */
    module_methods
};


/* module initialization */
PyMODINIT_FUNC PyInit_pyhdc(void) {
    PyObject *m;
    if (PyType_Ready(&LBVType) < 0)
        return NULL;
        
    m = PyModule_Create(&cModPyDem);
    if (m == NULL)
        return NULL;

    Py_INCREF(&LBVType);
    PyModule_AddObject(m, "LBV", (PyObject *) &LBVType);

    return m;
};
