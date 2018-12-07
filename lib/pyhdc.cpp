#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include <iostream>
#include <random>

#include <Python.h>
#include "structmember.h"


#include "permutations.h" 


void print_vector(const uint8_t vec[][2]) {
    for (unsigned long int j = 0; j < VECTOR_WIDTH; ++j) {
        printf("(%d | %d)\n", vec[j][0], vec[j][1]);
    }
}

void addup_vector(const uint8_t vec[][2]) {
    uint64_t lo = 0, hi = 0;
    for (unsigned long int j = 0; j < VECTOR_WIDTH; ++j) {
        lo += vec[j][0];
        hi += vec[j][1];
    }

    printf("(%lu | %lu)\n", lo, hi);
}


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
    uint8_t *data = (uint8_t *)v->data;
    for (uint32_t i = 0; i < VECTOR_WIDTH / 8; ++i) {
        uint8_t byte = data[i];
        ret += (byte & 0x80 ? '1' : '0');
        ret += (byte & 0x40 ? '1' : '0');
        ret += (byte & 0x20 ? '1' : '0');
        ret += (byte & 0x10 ? '1' : '0');
        ret += (byte & 0x08 ? '1' : '0');
        ret += (byte & 0x04 ? '1' : '0');
        ret += (byte & 0x02 ? '1' : '0');
        ret += (byte & 0x01 ? '1' : '0');
    }

    return PyUnicode_FromFormat(ret.c_str());
}

static PyObject *LBV_randomize(LBV *v, PyObject *Py_UNUSED(ignored)) {
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> gen(0, 255); // distribution in range [1, 6]

    uint8_t *data = (uint8_t *)v->data;
    for (uint32_t i = 0; i < VECTOR_WIDTH / 8; ++i) {
        data[i] = gen(rng);
    }

    Py_RETURN_NONE;
}


static PyMemberDef LBV_members[] = {
    {NULL}  /* Sentinel */
};


static PyObject *LBV_xor(LBV *v1, PyObject *args);
static PyMethodDef LBV_methods[] = {
    {"rand", (PyCFunction) LBV_randomize, METH_NOARGS,
     "Set vector elements to random values"
    },
    {"xor", (PyCFunction) LBV_xor, METH_VARARGS,
     "xor this vector with another vector and store the result in the current vector"
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


// Lowlevel math
static PyObject *LBV_xor(LBV *v1, PyObject *args) {
    LBV *v2;
    if (!PyArg_ParseTuple(args, "O!", &LBVType, &v2))
        return NULL;

    for (uint32_t i = 0; i < VECTOR_WIDTH / BIT_WIDTH; ++i) {
        v1->data[i] = v1->data[i] ^ v2->data[i];
    }

    Py_RETURN_NONE;
}


void permute_chunk(LBV v, const uint8_t p[][2], uint32_t id) {
    const uint32_t ref = v.data[id];
    uint32_t mask = 0;

    for (uint8_t i = 0; i < BIT_WIDTH; ++i) {
        const uint8_t chunk_id = p[id * BIT_WIDTH + i][0];
        const uint8_t bit_id   = p[id * BIT_WIDTH + i][1];
        const uint32_t swp = v.data[chunk_id];

        //ref & (1 << (BIT_WIDTH - i))
    }
}


// Permutation
void permute(LBV v, const uint8_t p[][2]) {
    for (uint32_t i = 0; i < VECTOR_WIDTH / BIT_WIDTH; ++i) {
        permute_chunk(v, p, i);
    }
}



int main() {
    for (unsigned long int i = 0; i < DEPTH_X; ++i) {
        addup_vector(Px[i]);
    }



    return 0;
}



static PyMethodDef module_methods[] = {
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
/*
PyMODINIT_FUNC
initpyhdc(void)
{
    PyObject* m;

    if (PyType_Ready(&LBVType) < 0)
        return;

    m = Py_InitModule3("pyhdc", module_methods,
                       "Module to work with long binary vectors");

    if (m == NULL)
        return;

    Py_INCREF(&LBVType);
    PyModule_AddObject(m, "LBV", (PyObject *)&LBVType);
}
*/


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
