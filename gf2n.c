#include "gf2n.h"
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <structmember.h>

typedef struct {
    PyObject_HEAD
    int exponent;
    PyObject* value;
} gf2n;

static void
gf2n_dealloc(gf2n* self)
{
    Py_XDECREF(self->value);
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
gf2n_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    gf2n *self;

    self = (gf2n *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->exponent = 1;
        npy_intp length[1];
        length[0] = 1;
        self->value = PyArray_ZEROS(1, length, NPY_DOUBLE, 0);
        if (self->value == NULL)
        {
            Py_DECREF(self);
            return NULL;
        }
    }

    return (PyObject *)self;
}

static int
gf2n_init(gf2n *self, PyObject *args, PyObject *kwds)
{
    PyObject *value=NULL, *tmp;

    static char *kwlist[] = {"exponent", "value", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "I|O", kwlist,
                                    &self->exponent,
                                    &value))
        return -1;

    if (value) {
        if(PyString_Check(value)) {
            Py_ssize_t len = PyString_Size(value);
            if(len == 0) return 0;

            char *charstr = malloc(sizeof(char) * len);
            uint8_t *valbytes = malloc(sizeof(uint8_t) * (len+1)/2);

            charstr = PyString_AsString(value);
            int i = 0;

            for (char* ptr = &charstr[len-2]; ptr >= charstr-1; ptr -= 2)
            {
                char substr[3];
                substr[0] = (ptr < charstr) ? '0' : ptr[0];
                substr[1] = ptr[1];
                substr[2] = 0; // Null char - ends string
                uint32_t val;
                sscanf(substr, "%x", &val);
                valbytes[i++] = val;
            }

            self->value = PyArray_SimpleNewFromData(1, (npy_intp*) &i, NPY_UBYTE, valbytes);
        }
        else if(PyArray_Check(value)) {
            tmp = self->value;
            Py_INCREF(value);
            self->value = value;
            Py_XDECREF(tmp);
        }
        else
            return -1;
    }

    return 0;
}

static PyMemberDef gf2n_members[] = {
    {"exponent", T_UINT, offsetof(gf2n, exponent), 0, "exponent"},
    {"value", T_OBJECT_EX, offsetof(gf2n, value), 0, "value"},
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

