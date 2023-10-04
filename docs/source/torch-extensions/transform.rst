Transformation
______________

An extension module which aims to simplify the transformation between Polar and Cartesian coordinates within the ATLAS detector. 
This is particularly useful in the context of Neural Networks, since all functions within the module are written in both native C++ and CUDA, resulting in faster training and inference performance.
Some of the module member functions have also been ported to natively support Python




.. py:module:: pyc.Transform

        .. py:function:: Pt(px, py) -> Pt
            :no-index:

            :params torch.tensor, double px:
