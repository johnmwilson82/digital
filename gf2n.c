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
gf2n_NEW(PyTypeObject *type, uint32_t degree, uint32_t value, uint32_t generator)
{
    gf2n *self;

    self = (gf2n *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->degree = degree;
        self->value = value;
        self->generator = generator;
    }

    return (PyObject *)self;
}

static PyObject *
gf2n_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    return gf2n_NEW(type, 1, 0x0, 0x3);
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
    self->degree = highest_set_bit(generator);

    return 0;
}

static PyObject *
gf2n_mul(PyObject *obj_v, PyObject *obj_w)
{
    gf2n *v = (gf2n*) obj_v;
    gf2n *w = (gf2n*) obj_w;

    if (v->generator != w->generator)
        return (PyObject*) NULL;

    // From an introduction to Galois Fields and Reed-Solomon coding by
    // James Westall and James Martin
    // http://people.cs.clemson.edu/~westall/851/rs-code.pdf

    uint32_t v1 = v->value;
    uint32_t v2 = w->value;
    uint32_t poly = v->generator & ((1 << v->degree) - 1);

    int prod = 0, k, mask;
    int m = v->degree;

    for (k = 0; k < m; k++)
    {
        if(v1 & 1)
        {
            prod ^= (v2 << k);
        }
        v1 >>= 1;
        if (v1 == 0)
            break;
    }
    printf("%x\n", prod);
    mask = 1 << m;
    mask <<= m - 2;

    for (k = m - 2; k >= 0; k--)
    {
        if (prod & mask)
        {
            prod &= ~mask;
            prod ^= (poly << k);
        }
        mask >>= 1;
    }

    return gf2n_NEW(v->ob_type, v->degree, prod, v->generator);
}

static PyObject *
gf2n_xor(PyObject *obj_v, PyObject *obj_w)
{
    gf2n *v = (gf2n*) obj_v;
    gf2n *w = (gf2n*) obj_w;

    if (v->generator != w->generator)
        return (PyObject*) NULL;

    uint32_t val = w->value ^ v->value;

    return gf2n_NEW(v->ob_type, v->degree, val, v->generator);
}

static PyObject *
gf2n_or(PyObject *obj_v, PyObject *obj_w)
{
    gf2n *v = (gf2n*) obj_v;
    gf2n *w = (gf2n*) obj_w;

    if (v->generator != w->generator)
        return (PyObject*) NULL;

    uint32_t val = w->value | v->value;

    return gf2n_NEW(v->ob_type, v->degree, val, v->generator);
}

static PyObject *
gf2n_and(PyObject *obj_v, PyObject *obj_w)
{
    gf2n *v = (gf2n*) obj_v;
    gf2n *w = (gf2n*) obj_w;

    if (v->generator != w->generator)
        return (PyObject*) NULL;

    uint32_t val = w->value & v->value;

    return gf2n_NEW(v->ob_type, v->degree, val, v->generator);
}

static PyObject *
gf2n_mod(PyObject *obj_v, PyObject *obj_w)
{
    if (!PyInt_Check(obj_w))
        return (PyObject*) NULL;

    gf2n *v = (gf2n*) obj_v;
    long modval = PyInt_AsLong(obj_w);

    uint32_t val = v->value % modval;

    return gf2n_NEW(v->ob_type, v->degree, val, v->generator);
}

static PyObject *
gf2n_lshift(PyObject *obj_v, PyObject *obj_w)
{
    if (!PyInt_Check(obj_w))
        return (PyObject*) NULL;

    gf2n *v = (gf2n*) obj_v;

    long shift = PyInt_AsLong(obj_w);

    uint32_t val = (v->value << shift) & ((1 << v->degree) - 1);

    return gf2n_NEW(v->ob_type, v->degree, val, v->generator);
}

static PyObject *
gf2n_rshift(PyObject *obj_v, PyObject *obj_w)
{
    if (!PyInt_Check(obj_w))
        return (PyObject*) NULL;

    gf2n *v = (gf2n*) obj_v;

    long shift = PyInt_AsLong(obj_w);

    uint32_t val = (v->value >> shift);

    return gf2n_NEW(v->ob_type, v->degree, val, v->generator);
}

static PyObject *
gf2n_nochange(PyObject * obj_v)
{
    return obj_v;
}

static int
gf2n_nonzero(PyObject *obj_v)
{
    gf2n *v = (gf2n*) obj_v;
    return (v->value == 0) ? 0 : 1;
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

static PyNumberMethods gf2n_as_number = {
    gf2n_xor,               /* __add__ */
    gf2n_xor,               /* __sub__ */
    gf2n_mul,               /* __mul__ */
    0,                      /* __div__ */
    gf2n_mod,               /* __mod__ */
    0,                      /* __divmod__ */
    0, //gf2n_pow,             /* __pow__ */
    0,                      /* __neg__ */
    gf2n_nochange,          /* __pos__ */
    gf2n_nochange,          /* __abs__ */
    gf2n_nonzero,           /* __nonzero__ */
    0,                      /* __invert__ */
    gf2n_lshift,            /* __lshift__ */
    gf2n_rshift,            /* __rshift__ */
    gf2n_and,               /* __and__ */
    gf2n_xor,               /* __xor__ */
    gf2n_or,                /* __or__ */
    0,                      /* __coerce__ */
    0,                      /* __int__ */
    0,                      /* __long__ */
    0,                      /* __float__ */
    0,                      /* __oct__ */
    0,                      /* __hex__ */
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
    &gf2n_as_number,           /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES,        /*tp_flags*/
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

