==============================
fht : Fast Hadamard Transform
==============================

What is fht ?
==============

This is a tested implementation of the `Hadamard transform`_ .  It
should be quite fast since it uses the `fast Hadamard transform`_
algorithm and is implemented using python C api. You can apply the
Hadamard transform to ndarray of dimensions 1, 2 or 3.  It requires
that input shapes are power of two since otherwise the Hadamard
transform is not defined.

Requirements
=============

You need numpy for fht to run.

.. _`Hadamard transform`:  http://en.wikipedia.org/wiki/Hadamard_transform
.. _`fast Hadamard transform`: http://en.wikipedia.org/wiki/Fast_Hadamard_transform
