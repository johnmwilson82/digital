#include "gf2n.h"
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <structmember.h>

typedef struct {
    PyObject_HEAD
    uint32_t degree;
    uint32_t value;
    uint32_t generator;
} gf2n;

static void
gf2n_dealloc(gf2n* self)
{
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
gf2n_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    gf2n *self;

    self = (gf2n *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->degree = 1;
        self->value = 0x0;
        self->generator = 0x3;
    }

    return (PyObject *)self;
}

uint32_t
highest_set_bit(uint32_t v)
{
    // From https://graphics.stanford.edu/~seander/bithacks.html#IntegerLogLookup
    union { uint32_t u[2]; double d; } t; // temp

    t.u[__FLOAT_WORD_ORDER==LITTLE_ENDIAN] = 0x43300000;
    t.u[__FLOAT_WORD_ORDER!=LITTLE_ENDIAN] = v;
    t.d -= 4503599627370496.0;
    return (t.u[__FLOAT_WORD_ORDER==LITTLE_ENDIAN] >> 20) - 0x3FF;
}

static int
gf2n_init(gf2n *self, PyObject *args, PyObject *kwds)
{
    int generator = 0;
    int value = 0;

    static char *kwlist[] = {"exponent", "value", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "I|I", kwlist,
                                    &generator,
                                    &value))
        return -1;

    self->generator = generator;
    self->value = value;
    self->degree = highest_set_bit(generator) - 1;

    return 0;
}

static PyMemberDef gf2n_members[] = {
    {"degree", T_UINT, offsetof(gf2n, degree), 0, "exponent"},
    {"generator", T_UINT, offsetof(gf2n, generator), 0, "generator"},
    {"value", T_UINT, offsetof(gf2n, value), 0, "value"},
    {NULL}
};

static PyMethodDef gf2n_methods[] = {
    {NULL},
};

static PyTypeObject gf2nType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "gf2n.gf2n",               /*tp_name*/
    sizeof(gf2n),              /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)gf2n_dealloc,  /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,        /*tp_flags*/
    "GF(2^n) objects",         /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    gf2n_methods,              /* tp_methods */
    gf2n_members,              /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)gf2n_init,       /* tp_init */
    0,                         /* tp_alloc */
    gf2n_new,                  /* tp_new */
};

static PyMethodDef module_methods[] = {
    {NULL}
};

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initgf2n(void)
{
    PyObject* m;

    if (PyType_Ready(&gf2nType) < 0)
        return;

    m = Py_InitModule3("gf2n", module_methods,
            "GF(2^n) extension type");

    if (m == NULL)
        return;

    import_array();

    Py_INCREF(&gf2nType);
    PyModule_AddObject(m, "gf2n", (PyObject *)&gf2nType);
}

