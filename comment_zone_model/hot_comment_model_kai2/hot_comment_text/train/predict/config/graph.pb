
/
ConstConst*
value	B : *
dtype0
8
is_train/inputConst*
value	B : *
dtype0
L
is_trainPlaceholderWithDefaultis_train/input*
shape: *
dtype0
K
MIO_TABLE_ADDRESSConst"/device:CPU:0*
value
B  *
dtype0
Ľ
2mio_compress_indices/COMPRESS_INDEX__USER/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerCOMPRESS_INDEX__USER*
shape:˙˙˙˙˙˙˙˙˙
Ľ
2mio_compress_indices/COMPRESS_INDEX__USER/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerCOMPRESS_INDEX__USER*
shape:˙˙˙˙˙˙˙˙˙
h
CastCast2mio_compress_indices/COMPRESS_INDEX__USER/variable*

SrcT0*
Truncate( *

DstT0

&mio_embeddings/user_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containeruser_embedding*
shape:˙˙˙˙˙˙˙˙˙

&mio_embeddings/user_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containeruser_embedding*
shape:˙˙˙˙˙˙˙˙˙

'mio_embeddings/photo_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerphoto_embedding*
shape:˙˙˙˙˙˙˙˙˙

'mio_embeddings/photo_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerphoto_embedding*
shape:˙˙˙˙˙˙˙˙˙

&mio_embeddings/c_id_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerc_id_embedding*
shape:˙˙˙˙˙˙˙˙˙

&mio_embeddings/c_id_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:˙˙˙˙˙˙˙˙˙*
	containerc_id_embedding

(mio_embeddings/c_info_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerc_info_embedding*
shape:˙˙˙˙˙˙˙˙˙Ŕ

(mio_embeddings/c_info_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerc_info_embedding*
shape:˙˙˙˙˙˙˙˙˙Ŕ

"mio_extra_param/mask_pack/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	container	mask_pack*
shape:˙˙˙˙˙˙˙˙˙

"mio_extra_param/mask_pack/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	container	mask_pack*
shape:˙˙˙˙˙˙˙˙˙
B
Reshape/shapeConst*
valueB"   ˙˙˙˙*
dtype0
\
ReshapeReshape"mio_extra_param/mask_pack/variableReshape/shape*
T0*
Tshape0
>
concat/values_0/axisConst*
value	B : *
dtype0

concat/values_0GatherV2&mio_embeddings/user_embedding/variableCastconcat/values_0/axis*
Taxis0*
Tindices0*
Tparams0
>
concat/values_1/axisConst*
dtype0*
value	B : 

concat/values_1GatherV2'mio_embeddings/photo_embedding/variableCastconcat/values_1/axis*
Tparams0*
Taxis0*
Tindices0
>
concat/axisConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
ą
concatConcatV2concat/values_0concat/values_1&mio_embeddings/c_id_embedding/variable(mio_embeddings/c_info_embedding/variableconcat/axis*

Tidx0*
T0*
N
8
ExpandDims/dimConst*
value	B : *
dtype0
E

ExpandDims
ExpandDimsconcatExpandDims/dim*
T0*

Tdim0
H
multi_head_attention/ShapeShape
ExpandDims*
T0*
out_type0
V
(multi_head_attention/strided_slice/stackConst*
dtype0*
valueB:
X
*multi_head_attention/strided_slice/stack_1Const*
valueB:*
dtype0
X
*multi_head_attention/strided_slice/stack_2Const*
valueB:*
dtype0
Ę
"multi_head_attention/strided_sliceStridedSlicemulti_head_attention/Shape(multi_head_attention/strided_slice/stack*multi_head_attention/strided_slice/stack_1*multi_head_attention/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
J
multi_head_attention/Shape_1Shape
ExpandDims*
T0*
out_type0
X
*multi_head_attention/strided_slice_1/stackConst*
valueB:*
dtype0
Z
,multi_head_attention/strided_slice_1/stack_1Const*
dtype0*
valueB:
Z
,multi_head_attention/strided_slice_1/stack_2Const*
valueB:*
dtype0
Ô
$multi_head_attention/strided_slice_1StridedSlicemulti_head_attention/Shape_1*multi_head_attention/strided_slice_1/stack,multi_head_attention/strided_slice_1/stack_1,multi_head_attention/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
´
7mio_variable/multi_head_attention/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!multi_head_attention/dense/kernel*
shape:
Č
´
7mio_variable/multi_head_attention/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!multi_head_attention/dense/kernel*
shape:
Č
U
 Initializer/random_uniform/shapeConst*
dtype0*
valueB"Č     
K
Initializer/random_uniform/minConst*
valueB
 *&Ď˝*
dtype0
K
Initializer/random_uniform/maxConst*
dtype0*
valueB
 *&Ď=

(Initializer/random_uniform/RandomUniformRandomUniform Initializer/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
n
Initializer/random_uniform/subSubInitializer/random_uniform/maxInitializer/random_uniform/min*
T0
x
Initializer/random_uniform/mulMul(Initializer/random_uniform/RandomUniformInitializer/random_uniform/sub*
T0
j
Initializer/random_uniformAddInitializer/random_uniform/mulInitializer/random_uniform/min*
T0
ă
AssignAssign7mio_variable/multi_head_attention/dense/kernel/gradientInitializer/random_uniform*
use_locking(*
T0*J
_class@
><loc:@mio_variable/multi_head_attention/dense/kernel/gradient*
validate_shape(
Ť
5mio_variable/multi_head_attention/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!multi_head_attention/dense/bias*
shape:
Ť
5mio_variable/multi_head_attention/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!multi_head_attention/dense/bias*
shape:
E
Initializer_1/zerosConst*
valueB*    *
dtype0
Ú
Assign_1Assign5mio_variable/multi_head_attention/dense/bias/gradientInitializer_1/zeros*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/multi_head_attention/dense/bias/gradient*
validate_shape(
W
)multi_head_attention/dense/Tensordot/axesConst*
dtype0*
valueB:
^
)multi_head_attention/dense/Tensordot/freeConst*
valueB"       *
dtype0
X
*multi_head_attention/dense/Tensordot/ShapeShape
ExpandDims*
T0*
out_type0
\
2multi_head_attention/dense/Tensordot/GatherV2/axisConst*
dtype0*
value	B : 
đ
-multi_head_attention/dense/Tensordot/GatherV2GatherV2*multi_head_attention/dense/Tensordot/Shape)multi_head_attention/dense/Tensordot/free2multi_head_attention/dense/Tensordot/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0
^
4multi_head_attention/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0
ô
/multi_head_attention/dense/Tensordot/GatherV2_1GatherV2*multi_head_attention/dense/Tensordot/Shape)multi_head_attention/dense/Tensordot/axes4multi_head_attention/dense/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0
X
*multi_head_attention/dense/Tensordot/ConstConst*
valueB: *
dtype0
˛
)multi_head_attention/dense/Tensordot/ProdProd-multi_head_attention/dense/Tensordot/GatherV2*multi_head_attention/dense/Tensordot/Const*
T0*

Tidx0*
	keep_dims( 
Z
,multi_head_attention/dense/Tensordot/Const_1Const*
dtype0*
valueB: 
¸
+multi_head_attention/dense/Tensordot/Prod_1Prod/multi_head_attention/dense/Tensordot/GatherV2_1,multi_head_attention/dense/Tensordot/Const_1*
T0*

Tidx0*
	keep_dims( 
Z
0multi_head_attention/dense/Tensordot/concat/axisConst*
value	B : *
dtype0
Ý
+multi_head_attention/dense/Tensordot/concatConcatV2)multi_head_attention/dense/Tensordot/free)multi_head_attention/dense/Tensordot/axes0multi_head_attention/dense/Tensordot/concat/axis*

Tidx0*
T0*
N
¨
*multi_head_attention/dense/Tensordot/stackPack)multi_head_attention/dense/Tensordot/Prod+multi_head_attention/dense/Tensordot/Prod_1*
T0*

axis *
N

.multi_head_attention/dense/Tensordot/transpose	Transpose
ExpandDims+multi_head_attention/dense/Tensordot/concat*
T0*
Tperm0
Ş
,multi_head_attention/dense/Tensordot/ReshapeReshape.multi_head_attention/dense/Tensordot/transpose*multi_head_attention/dense/Tensordot/stack*
T0*
Tshape0
j
5multi_head_attention/dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
Ă
0multi_head_attention/dense/Tensordot/transpose_1	Transpose7mio_variable/multi_head_attention/dense/kernel/variable5multi_head_attention/dense/Tensordot/transpose_1/perm*
Tperm0*
T0
i
4multi_head_attention/dense/Tensordot/Reshape_1/shapeConst*
dtype0*
valueB"Č     
¸
.multi_head_attention/dense/Tensordot/Reshape_1Reshape0multi_head_attention/dense/Tensordot/transpose_14multi_head_attention/dense/Tensordot/Reshape_1/shape*
T0*
Tshape0
Â
+multi_head_attention/dense/Tensordot/MatMulMatMul,multi_head_attention/dense/Tensordot/Reshape.multi_head_attention/dense/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
[
,multi_head_attention/dense/Tensordot/Const_2Const*
valueB:*
dtype0
\
2multi_head_attention/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0
č
-multi_head_attention/dense/Tensordot/concat_1ConcatV2-multi_head_attention/dense/Tensordot/GatherV2,multi_head_attention/dense/Tensordot/Const_22multi_head_attention/dense/Tensordot/concat_1/axis*
T0*
N*

Tidx0
˘
$multi_head_attention/dense/TensordotReshape+multi_head_attention/dense/Tensordot/MatMul-multi_head_attention/dense/Tensordot/concat_1*
T0*
Tshape0
Ş
"multi_head_attention/dense/BiasAddBiasAdd$multi_head_attention/dense/Tensordot5mio_variable/multi_head_attention/dense/bias/variable*
T0*
data_formatNHWC
¸
9mio_variable/multi_head_attention/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#multi_head_attention/dense_1/kernel*
shape:
Č
¸
9mio_variable/multi_head_attention/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#multi_head_attention/dense_1/kernel*
shape:
Č
W
"Initializer_2/random_uniform/shapeConst*
valueB"Č     *
dtype0
M
 Initializer_2/random_uniform/minConst*
valueB
 *&Ď˝*
dtype0
M
 Initializer_2/random_uniform/maxConst*
dtype0*
valueB
 *&Ď=

*Initializer_2/random_uniform/RandomUniformRandomUniform"Initializer_2/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
t
 Initializer_2/random_uniform/subSub Initializer_2/random_uniform/max Initializer_2/random_uniform/min*
T0
~
 Initializer_2/random_uniform/mulMul*Initializer_2/random_uniform/RandomUniform Initializer_2/random_uniform/sub*
T0
p
Initializer_2/random_uniformAdd Initializer_2/random_uniform/mul Initializer_2/random_uniform/min*
T0
ë
Assign_2Assign9mio_variable/multi_head_attention/dense_1/kernel/gradientInitializer_2/random_uniform*
use_locking(*
T0*L
_classB
@>loc:@mio_variable/multi_head_attention/dense_1/kernel/gradient*
validate_shape(
Ż
7mio_variable/multi_head_attention/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!multi_head_attention/dense_1/bias*
shape:
Ż
7mio_variable/multi_head_attention/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!multi_head_attention/dense_1/bias*
shape:
E
Initializer_3/zerosConst*
valueB*    *
dtype0
Ţ
Assign_3Assign7mio_variable/multi_head_attention/dense_1/bias/gradientInitializer_3/zeros*
validate_shape(*
use_locking(*
T0*J
_class@
><loc:@mio_variable/multi_head_attention/dense_1/bias/gradient
Y
+multi_head_attention/dense_1/Tensordot/axesConst*
valueB:*
dtype0
`
+multi_head_attention/dense_1/Tensordot/freeConst*
valueB"       *
dtype0
Z
,multi_head_attention/dense_1/Tensordot/ShapeShape
ExpandDims*
T0*
out_type0
^
4multi_head_attention/dense_1/Tensordot/GatherV2/axisConst*
value	B : *
dtype0
ř
/multi_head_attention/dense_1/Tensordot/GatherV2GatherV2,multi_head_attention/dense_1/Tensordot/Shape+multi_head_attention/dense_1/Tensordot/free4multi_head_attention/dense_1/Tensordot/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0
`
6multi_head_attention/dense_1/Tensordot/GatherV2_1/axisConst*
dtype0*
value	B : 
ü
1multi_head_attention/dense_1/Tensordot/GatherV2_1GatherV2,multi_head_attention/dense_1/Tensordot/Shape+multi_head_attention/dense_1/Tensordot/axes6multi_head_attention/dense_1/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0
Z
,multi_head_attention/dense_1/Tensordot/ConstConst*
valueB: *
dtype0
¸
+multi_head_attention/dense_1/Tensordot/ProdProd/multi_head_attention/dense_1/Tensordot/GatherV2,multi_head_attention/dense_1/Tensordot/Const*
T0*

Tidx0*
	keep_dims( 
\
.multi_head_attention/dense_1/Tensordot/Const_1Const*
dtype0*
valueB: 
ž
-multi_head_attention/dense_1/Tensordot/Prod_1Prod1multi_head_attention/dense_1/Tensordot/GatherV2_1.multi_head_attention/dense_1/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0
\
2multi_head_attention/dense_1/Tensordot/concat/axisConst*
value	B : *
dtype0
ĺ
-multi_head_attention/dense_1/Tensordot/concatConcatV2+multi_head_attention/dense_1/Tensordot/free+multi_head_attention/dense_1/Tensordot/axes2multi_head_attention/dense_1/Tensordot/concat/axis*
N*

Tidx0*
T0
Ž
,multi_head_attention/dense_1/Tensordot/stackPack+multi_head_attention/dense_1/Tensordot/Prod-multi_head_attention/dense_1/Tensordot/Prod_1*
N*
T0*

axis 

0multi_head_attention/dense_1/Tensordot/transpose	Transpose
ExpandDims-multi_head_attention/dense_1/Tensordot/concat*
T0*
Tperm0
°
.multi_head_attention/dense_1/Tensordot/ReshapeReshape0multi_head_attention/dense_1/Tensordot/transpose,multi_head_attention/dense_1/Tensordot/stack*
T0*
Tshape0
l
7multi_head_attention/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
É
2multi_head_attention/dense_1/Tensordot/transpose_1	Transpose9mio_variable/multi_head_attention/dense_1/kernel/variable7multi_head_attention/dense_1/Tensordot/transpose_1/perm*
Tperm0*
T0
k
6multi_head_attention/dense_1/Tensordot/Reshape_1/shapeConst*
valueB"Č     *
dtype0
ž
0multi_head_attention/dense_1/Tensordot/Reshape_1Reshape2multi_head_attention/dense_1/Tensordot/transpose_16multi_head_attention/dense_1/Tensordot/Reshape_1/shape*
T0*
Tshape0
Č
-multi_head_attention/dense_1/Tensordot/MatMulMatMul.multi_head_attention/dense_1/Tensordot/Reshape0multi_head_attention/dense_1/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
]
.multi_head_attention/dense_1/Tensordot/Const_2Const*
valueB:*
dtype0
^
4multi_head_attention/dense_1/Tensordot/concat_1/axisConst*
value	B : *
dtype0
đ
/multi_head_attention/dense_1/Tensordot/concat_1ConcatV2/multi_head_attention/dense_1/Tensordot/GatherV2.multi_head_attention/dense_1/Tensordot/Const_24multi_head_attention/dense_1/Tensordot/concat_1/axis*
T0*
N*

Tidx0
¨
&multi_head_attention/dense_1/TensordotReshape-multi_head_attention/dense_1/Tensordot/MatMul/multi_head_attention/dense_1/Tensordot/concat_1*
T0*
Tshape0
°
$multi_head_attention/dense_1/BiasAddBiasAdd&multi_head_attention/dense_1/Tensordot7mio_variable/multi_head_attention/dense_1/bias/variable*
T0*
data_formatNHWC
¸
9mio_variable/multi_head_attention/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
Č*2
	container%#multi_head_attention/dense_2/kernel
¸
9mio_variable/multi_head_attention/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#multi_head_attention/dense_2/kernel*
shape:
Č
W
"Initializer_4/random_uniform/shapeConst*
dtype0*
valueB"Č     
M
 Initializer_4/random_uniform/minConst*
valueB
 *&Ď˝*
dtype0
M
 Initializer_4/random_uniform/maxConst*
valueB
 *&Ď=*
dtype0

*Initializer_4/random_uniform/RandomUniformRandomUniform"Initializer_4/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
t
 Initializer_4/random_uniform/subSub Initializer_4/random_uniform/max Initializer_4/random_uniform/min*
T0
~
 Initializer_4/random_uniform/mulMul*Initializer_4/random_uniform/RandomUniform Initializer_4/random_uniform/sub*
T0
p
Initializer_4/random_uniformAdd Initializer_4/random_uniform/mul Initializer_4/random_uniform/min*
T0
ë
Assign_4Assign9mio_variable/multi_head_attention/dense_2/kernel/gradientInitializer_4/random_uniform*
use_locking(*
T0*L
_classB
@>loc:@mio_variable/multi_head_attention/dense_2/kernel/gradient*
validate_shape(
Ż
7mio_variable/multi_head_attention/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!multi_head_attention/dense_2/bias*
shape:
Ż
7mio_variable/multi_head_attention/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!multi_head_attention/dense_2/bias*
shape:
E
Initializer_5/zerosConst*
valueB*    *
dtype0
Ţ
Assign_5Assign7mio_variable/multi_head_attention/dense_2/bias/gradientInitializer_5/zeros*
validate_shape(*
use_locking(*
T0*J
_class@
><loc:@mio_variable/multi_head_attention/dense_2/bias/gradient
Y
+multi_head_attention/dense_2/Tensordot/axesConst*
dtype0*
valueB:
`
+multi_head_attention/dense_2/Tensordot/freeConst*
valueB"       *
dtype0
Z
,multi_head_attention/dense_2/Tensordot/ShapeShape
ExpandDims*
T0*
out_type0
^
4multi_head_attention/dense_2/Tensordot/GatherV2/axisConst*
value	B : *
dtype0
ř
/multi_head_attention/dense_2/Tensordot/GatherV2GatherV2,multi_head_attention/dense_2/Tensordot/Shape+multi_head_attention/dense_2/Tensordot/free4multi_head_attention/dense_2/Tensordot/GatherV2/axis*
Tparams0*
Taxis0*
Tindices0
`
6multi_head_attention/dense_2/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0
ü
1multi_head_attention/dense_2/Tensordot/GatherV2_1GatherV2,multi_head_attention/dense_2/Tensordot/Shape+multi_head_attention/dense_2/Tensordot/axes6multi_head_attention/dense_2/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0
Z
,multi_head_attention/dense_2/Tensordot/ConstConst*
valueB: *
dtype0
¸
+multi_head_attention/dense_2/Tensordot/ProdProd/multi_head_attention/dense_2/Tensordot/GatherV2,multi_head_attention/dense_2/Tensordot/Const*

Tidx0*
	keep_dims( *
T0
\
.multi_head_attention/dense_2/Tensordot/Const_1Const*
dtype0*
valueB: 
ž
-multi_head_attention/dense_2/Tensordot/Prod_1Prod1multi_head_attention/dense_2/Tensordot/GatherV2_1.multi_head_attention/dense_2/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0
\
2multi_head_attention/dense_2/Tensordot/concat/axisConst*
value	B : *
dtype0
ĺ
-multi_head_attention/dense_2/Tensordot/concatConcatV2+multi_head_attention/dense_2/Tensordot/free+multi_head_attention/dense_2/Tensordot/axes2multi_head_attention/dense_2/Tensordot/concat/axis*
N*

Tidx0*
T0
Ž
,multi_head_attention/dense_2/Tensordot/stackPack+multi_head_attention/dense_2/Tensordot/Prod-multi_head_attention/dense_2/Tensordot/Prod_1*
T0*

axis *
N

0multi_head_attention/dense_2/Tensordot/transpose	Transpose
ExpandDims-multi_head_attention/dense_2/Tensordot/concat*
T0*
Tperm0
°
.multi_head_attention/dense_2/Tensordot/ReshapeReshape0multi_head_attention/dense_2/Tensordot/transpose,multi_head_attention/dense_2/Tensordot/stack*
T0*
Tshape0
l
7multi_head_attention/dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
É
2multi_head_attention/dense_2/Tensordot/transpose_1	Transpose9mio_variable/multi_head_attention/dense_2/kernel/variable7multi_head_attention/dense_2/Tensordot/transpose_1/perm*
T0*
Tperm0
k
6multi_head_attention/dense_2/Tensordot/Reshape_1/shapeConst*
valueB"Č     *
dtype0
ž
0multi_head_attention/dense_2/Tensordot/Reshape_1Reshape2multi_head_attention/dense_2/Tensordot/transpose_16multi_head_attention/dense_2/Tensordot/Reshape_1/shape*
T0*
Tshape0
Č
-multi_head_attention/dense_2/Tensordot/MatMulMatMul.multi_head_attention/dense_2/Tensordot/Reshape0multi_head_attention/dense_2/Tensordot/Reshape_1*
transpose_b( *
T0*
transpose_a( 
]
.multi_head_attention/dense_2/Tensordot/Const_2Const*
valueB:*
dtype0
^
4multi_head_attention/dense_2/Tensordot/concat_1/axisConst*
value	B : *
dtype0
đ
/multi_head_attention/dense_2/Tensordot/concat_1ConcatV2/multi_head_attention/dense_2/Tensordot/GatherV2.multi_head_attention/dense_2/Tensordot/Const_24multi_head_attention/dense_2/Tensordot/concat_1/axis*
T0*
N*

Tidx0
¨
&multi_head_attention/dense_2/TensordotReshape-multi_head_attention/dense_2/Tensordot/MatMul/multi_head_attention/dense_2/Tensordot/concat_1*
T0*
Tshape0
°
$multi_head_attention/dense_2/BiasAddBiasAdd&multi_head_attention/dense_2/Tensordot7mio_variable/multi_head_attention/dense_2/bias/variable*
T0*
data_formatNHWC
W
$multi_head_attention/Reshape/shape/0Const*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
N
$multi_head_attention/Reshape/shape/2Const*
dtype0*
value	B :
N
$multi_head_attention/Reshape/shape/3Const*
value	B :*
dtype0
Ţ
"multi_head_attention/Reshape/shapePack$multi_head_attention/Reshape/shape/0"multi_head_attention/strided_slice$multi_head_attention/Reshape/shape/2$multi_head_attention/Reshape/shape/3*
T0*

axis *
N

multi_head_attention/ReshapeReshape"multi_head_attention/dense/BiasAdd"multi_head_attention/Reshape/shape*
T0*
Tshape0
`
#multi_head_attention/transpose/permConst*%
valueB"             *
dtype0

multi_head_attention/transpose	Transposemulti_head_attention/Reshape#multi_head_attention/transpose/perm*
Tperm0*
T0
Y
&multi_head_attention/Reshape_1/shape/0Const*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
P
&multi_head_attention/Reshape_1/shape/2Const*
dtype0*
value	B :
P
&multi_head_attention/Reshape_1/shape/3Const*
value	B :*
dtype0
č
$multi_head_attention/Reshape_1/shapePack&multi_head_attention/Reshape_1/shape/0$multi_head_attention/strided_slice_1&multi_head_attention/Reshape_1/shape/2&multi_head_attention/Reshape_1/shape/3*
T0*

axis *
N

multi_head_attention/Reshape_1Reshape$multi_head_attention/dense_1/BiasAdd$multi_head_attention/Reshape_1/shape*
T0*
Tshape0
b
%multi_head_attention/transpose_1/permConst*%
valueB"             *
dtype0

 multi_head_attention/transpose_1	Transposemulti_head_attention/Reshape_1%multi_head_attention/transpose_1/perm*
T0*
Tperm0
Y
&multi_head_attention/Reshape_2/shape/0Const*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
P
&multi_head_attention/Reshape_2/shape/2Const*
dtype0*
value	B :
P
&multi_head_attention/Reshape_2/shape/3Const*
dtype0*
value	B :
č
$multi_head_attention/Reshape_2/shapePack&multi_head_attention/Reshape_2/shape/0$multi_head_attention/strided_slice_1&multi_head_attention/Reshape_2/shape/2&multi_head_attention/Reshape_2/shape/3*
N*
T0*

axis 

multi_head_attention/Reshape_2Reshape$multi_head_attention/dense_2/BiasAdd$multi_head_attention/Reshape_2/shape*
T0*
Tshape0
b
%multi_head_attention/transpose_2/permConst*%
valueB"             *
dtype0

 multi_head_attention/transpose_2	Transposemulti_head_attention/Reshape_2%multi_head_attention/transpose_2/perm*
Tperm0*
T0

multi_head_attention/MatMulBatchMatMulmulti_head_attention/transpose multi_head_attention/transpose_1*
adj_x( *
adj_y(*
T0
E
multi_head_attention/Cast/xConst*
value	B :*
dtype0
f
multi_head_attention/CastCastmulti_head_attention/Cast/x*
Truncate( *

DstT0*

SrcT0
E
multi_head_attention/SqrtSqrtmulti_head_attention/Cast*
T0
h
multi_head_attention/truedivRealDivmulti_head_attention/MatMulmulti_head_attention/Sqrt*
T0
g
*multi_head_attention/strided_slice_2/stackConst*%
valueB"                *
dtype0
i
,multi_head_attention/strided_slice_2/stack_1Const*
dtype0*%
valueB"                
i
,multi_head_attention/strided_slice_2/stack_2Const*%
valueB"            *
dtype0
ż
$multi_head_attention/strided_slice_2StridedSliceReshape*multi_head_attention/strided_slice_2/stack,multi_head_attention/strided_slice_2/stack_1,multi_head_attention/strided_slice_2/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask
*
end_mask*
Index0*
T0
O
%multi_head_attention/Tile/multiples/0Const*
value	B :*
dtype0
O
%multi_head_attention/Tile/multiples/1Const*
dtype0*
value	B :
O
%multi_head_attention/Tile/multiples/2Const*
value	B :*
dtype0
ä
#multi_head_attention/Tile/multiplesPack%multi_head_attention/Tile/multiples/0%multi_head_attention/Tile/multiples/1%multi_head_attention/Tile/multiples/2$multi_head_attention/strided_slice_1*
N*
T0*

axis 

multi_head_attention/TileTile$multi_head_attention/strided_slice_2#multi_head_attention/Tile/multiples*

Tmultiples0*
T0
G
multi_head_attention/sub/xConst*
valueB
 *  ?*
dtype0
_
multi_head_attention/subSubmulti_head_attention/sub/xmulti_head_attention/Tile*
T0
G
multi_head_attention/mul/yConst*
dtype0*
valueB
 *(knÎ
^
multi_head_attention/mulMulmulti_head_attention/submulti_head_attention/mul/y*
T0
`
multi_head_attention/addAddmulti_head_attention/truedivmulti_head_attention/mul*
T0
J
multi_head_attention/SoftmaxSoftmaxmulti_head_attention/add*
T0

multi_head_attention/MatMul_1BatchMatMulmulti_head_attention/Softmax multi_head_attention/transpose_2*
adj_x( *
adj_y( *
T0
b
%multi_head_attention/transpose_3/permConst*%
valueB"             *
dtype0

 multi_head_attention/transpose_3	Transposemulti_head_attention/MatMul_1%multi_head_attention/transpose_3/perm*
T0*
Tperm0
Y
&multi_head_attention/Reshape_3/shape/0Const*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Q
&multi_head_attention/Reshape_3/shape/2Const*
value
B :*
dtype0
Ŕ
$multi_head_attention/Reshape_3/shapePack&multi_head_attention/Reshape_3/shape/0$multi_head_attention/strided_slice_1&multi_head_attention/Reshape_3/shape/2*
T0*

axis *
N

multi_head_attention/Reshape_3Reshape multi_head_attention/transpose_3$multi_head_attention/Reshape_3/shape*
T0*
Tshape0
n
multi_head_attention/add_1Addmulti_head_attention/Reshape_3"multi_head_attention/dense/BiasAdd*
T0

Mmulti_head_attention/multi_head_attention_norm/moments/mean/reduction_indicesConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
Ô
;multi_head_attention/multi_head_attention_norm/moments/meanMeanmulti_head_attention/add_1Mmulti_head_attention/multi_head_attention_norm/moments/mean/reduction_indices*
T0*

Tidx0*
	keep_dims(

Cmulti_head_attention/multi_head_attention_norm/moments/StopGradientStopGradient;multi_head_attention/multi_head_attention_norm/moments/mean*
T0
Ç
Hmulti_head_attention/multi_head_attention_norm/moments/SquaredDifferenceSquaredDifferencemulti_head_attention/add_1Cmulti_head_attention/multi_head_attention_norm/moments/StopGradient*
T0

Qmulti_head_attention/multi_head_attention_norm/moments/variance/reduction_indicesConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0

?multi_head_attention/multi_head_attention_norm/moments/varianceMeanHmulti_head_attention/multi_head_attention_norm/moments/SquaredDifferenceQmulti_head_attention/multi_head_attention_norm/moments/variance/reduction_indices*
T0*

Tidx0*
	keep_dims(

2multi_head_attention/multi_head_attention_norm/subSubmulti_head_attention/add_1;multi_head_attention/multi_head_attention_norm/moments/mean*
T0
a
4multi_head_attention/multi_head_attention_norm/add/yConst*
valueB
 *ŹĹ'7*
dtype0
š
2multi_head_attention/multi_head_attention_norm/addAdd?multi_head_attention/multi_head_attention_norm/moments/variance4multi_head_attention/multi_head_attention_norm/add/y*
T0
z
4multi_head_attention/multi_head_attention_norm/RsqrtRsqrt2multi_head_attention/multi_head_attention_norm/add*
T0
Ź
2multi_head_attention/multi_head_attention_norm/mulMul2multi_head_attention/multi_head_attention_norm/sub4multi_head_attention/multi_head_attention_norm/Rsqrt*
T0

dmio_variable/multi_head_attention/multi_head_attention_norm/gamma_multi_head_attention_norm/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*]
	containerPNmulti_head_attention/multi_head_attention_norm/gamma_multi_head_attention_norm*
shape:

dmio_variable/multi_head_attention/multi_head_attention_norm/gamma_multi_head_attention_norm/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*]
	containerPNmulti_head_attention/multi_head_attention_norm/gamma_multi_head_attention_norm*
shape:
D
Initializer_6/onesConst*
valueB*  ?*
dtype0
ˇ
Assign_6Assigndmio_variable/multi_head_attention/multi_head_attention_norm/gamma_multi_head_attention_norm/gradientInitializer_6/ones*
validate_shape(*
use_locking(*
T0*w
_classm
kiloc:@mio_variable/multi_head_attention/multi_head_attention_norm/gamma_multi_head_attention_norm/gradient

cmio_variable/multi_head_attention/multi_head_attention_norm/beta_multi_head_attention_norm/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*\
	containerOMmulti_head_attention/multi_head_attention_norm/beta_multi_head_attention_norm*
shape:

cmio_variable/multi_head_attention/multi_head_attention_norm/beta_multi_head_attention_norm/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*\
	containerOMmulti_head_attention/multi_head_attention_norm/beta_multi_head_attention_norm
D
Initializer_7/onesConst*
dtype0*
valueB*  ?
ľ
Assign_7Assigncmio_variable/multi_head_attention/multi_head_attention_norm/beta_multi_head_attention_norm/gradientInitializer_7/ones*
use_locking(*
T0*v
_classl
jhloc:@mio_variable/multi_head_attention/multi_head_attention_norm/beta_multi_head_attention_norm/gradient*
validate_shape(
Ţ
4multi_head_attention/multi_head_attention_norm/mul_1Muldmio_variable/multi_head_attention/multi_head_attention_norm/gamma_multi_head_attention_norm/variable2multi_head_attention/multi_head_attention_norm/mul*
T0
ß
4multi_head_attention/multi_head_attention_norm/add_1Add4multi_head_attention/multi_head_attention_norm/mul_1cmio_variable/multi_head_attention/multi_head_attention_norm/beta_multi_head_attention_norm/variable*
T0

-mio_variable/expand_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerexpand_xtr/dense/kernel*
shape:	

-mio_variable/expand_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	*&
	containerexpand_xtr/dense/kernel
W
"Initializer_8/random_uniform/shapeConst*
valueB"      *
dtype0
M
 Initializer_8/random_uniform/minConst*
dtype0*
valueB
 *n×\ž
M
 Initializer_8/random_uniform/maxConst*
valueB
 *n×\>*
dtype0

*Initializer_8/random_uniform/RandomUniformRandomUniform"Initializer_8/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
t
 Initializer_8/random_uniform/subSub Initializer_8/random_uniform/max Initializer_8/random_uniform/min*
T0
~
 Initializer_8/random_uniform/mulMul*Initializer_8/random_uniform/RandomUniform Initializer_8/random_uniform/sub*
T0
p
Initializer_8/random_uniformAdd Initializer_8/random_uniform/mul Initializer_8/random_uniform/min*
T0
Ó
Assign_8Assign-mio_variable/expand_xtr/dense/kernel/gradientInitializer_8/random_uniform*
use_locking(*
T0*@
_class6
42loc:@mio_variable/expand_xtr/dense/kernel/gradient*
validate_shape(

+mio_variable/expand_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containerexpand_xtr/dense/bias

+mio_variable/expand_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containerexpand_xtr/dense/bias
D
Initializer_9/zerosConst*
dtype0*
valueB*    
Ć
Assign_9Assign+mio_variable/expand_xtr/dense/bias/gradientInitializer_9/zeros*
T0*>
_class4
20loc:@mio_variable/expand_xtr/dense/bias/gradient*
validate_shape(*
use_locking(
M
expand_xtr/dense/Tensordot/axesConst*
valueB:*
dtype0
T
expand_xtr/dense/Tensordot/freeConst*
valueB"       *
dtype0
x
 expand_xtr/dense/Tensordot/ShapeShape4multi_head_attention/multi_head_attention_norm/add_1*
T0*
out_type0
R
(expand_xtr/dense/Tensordot/GatherV2/axisConst*
dtype0*
value	B : 
Č
#expand_xtr/dense/Tensordot/GatherV2GatherV2 expand_xtr/dense/Tensordot/Shapeexpand_xtr/dense/Tensordot/free(expand_xtr/dense/Tensordot/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0
T
*expand_xtr/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0
Ě
%expand_xtr/dense/Tensordot/GatherV2_1GatherV2 expand_xtr/dense/Tensordot/Shapeexpand_xtr/dense/Tensordot/axes*expand_xtr/dense/Tensordot/GatherV2_1/axis*
Tparams0*
Taxis0*
Tindices0
N
 expand_xtr/dense/Tensordot/ConstConst*
valueB: *
dtype0

expand_xtr/dense/Tensordot/ProdProd#expand_xtr/dense/Tensordot/GatherV2 expand_xtr/dense/Tensordot/Const*

Tidx0*
	keep_dims( *
T0
P
"expand_xtr/dense/Tensordot/Const_1Const*
valueB: *
dtype0

!expand_xtr/dense/Tensordot/Prod_1Prod%expand_xtr/dense/Tensordot/GatherV2_1"expand_xtr/dense/Tensordot/Const_1*
T0*

Tidx0*
	keep_dims( 
P
&expand_xtr/dense/Tensordot/concat/axisConst*
value	B : *
dtype0
ľ
!expand_xtr/dense/Tensordot/concatConcatV2expand_xtr/dense/Tensordot/freeexpand_xtr/dense/Tensordot/axes&expand_xtr/dense/Tensordot/concat/axis*

Tidx0*
T0*
N

 expand_xtr/dense/Tensordot/stackPackexpand_xtr/dense/Tensordot/Prod!expand_xtr/dense/Tensordot/Prod_1*
T0*

axis *
N
 
$expand_xtr/dense/Tensordot/transpose	Transpose4multi_head_attention/multi_head_attention_norm/add_1!expand_xtr/dense/Tensordot/concat*
Tperm0*
T0

"expand_xtr/dense/Tensordot/ReshapeReshape$expand_xtr/dense/Tensordot/transpose expand_xtr/dense/Tensordot/stack*
T0*
Tshape0
`
+expand_xtr/dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
Ľ
&expand_xtr/dense/Tensordot/transpose_1	Transpose-mio_variable/expand_xtr/dense/kernel/variable+expand_xtr/dense/Tensordot/transpose_1/perm*
Tperm0*
T0
_
*expand_xtr/dense/Tensordot/Reshape_1/shapeConst*
dtype0*
valueB"      

$expand_xtr/dense/Tensordot/Reshape_1Reshape&expand_xtr/dense/Tensordot/transpose_1*expand_xtr/dense/Tensordot/Reshape_1/shape*
T0*
Tshape0
¤
!expand_xtr/dense/Tensordot/MatMulMatMul"expand_xtr/dense/Tensordot/Reshape$expand_xtr/dense/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
P
"expand_xtr/dense/Tensordot/Const_2Const*
valueB:*
dtype0
R
(expand_xtr/dense/Tensordot/concat_1/axisConst*
dtype0*
value	B : 
Ŕ
#expand_xtr/dense/Tensordot/concat_1ConcatV2#expand_xtr/dense/Tensordot/GatherV2"expand_xtr/dense/Tensordot/Const_2(expand_xtr/dense/Tensordot/concat_1/axis*
T0*
N*

Tidx0

expand_xtr/dense/TensordotReshape!expand_xtr/dense/Tensordot/MatMul#expand_xtr/dense/Tensordot/concat_1*
T0*
Tshape0

expand_xtr/dense/BiasAddBiasAddexpand_xtr/dense/Tensordot+mio_variable/expand_xtr/dense/bias/variable*
T0*
data_formatNHWC
F
expand_xtr/dense/SigmoidSigmoidexpand_xtr/dense/BiasAdd*
T0
D
Reshape_1/shapeConst*
valueB"˙˙˙˙   *
dtype0
V
	Reshape_1Reshapeexpand_xtr/dense/SigmoidReshape_1/shape*
T0*
Tshape0

+mio_variable/like_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerlike_xtr/dense/kernel*
shape:	

+mio_variable/like_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerlike_xtr/dense/kernel*
shape:	
X
#Initializer_10/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_10/random_uniform/minConst*
valueB
 *n×\ž*
dtype0
N
!Initializer_10/random_uniform/maxConst*
valueB
 *n×\>*
dtype0

+Initializer_10/random_uniform/RandomUniformRandomUniform#Initializer_10/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_10/random_uniform/subSub!Initializer_10/random_uniform/max!Initializer_10/random_uniform/min*
T0

!Initializer_10/random_uniform/mulMul+Initializer_10/random_uniform/RandomUniform!Initializer_10/random_uniform/sub*
T0
s
Initializer_10/random_uniformAdd!Initializer_10/random_uniform/mul!Initializer_10/random_uniform/min*
T0
Ń
	Assign_10Assign+mio_variable/like_xtr/dense/kernel/gradientInitializer_10/random_uniform*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@mio_variable/like_xtr/dense/kernel/gradient

)mio_variable/like_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*"
	containerlike_xtr/dense/bias

)mio_variable/like_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*"
	containerlike_xtr/dense/bias*
shape:
E
Initializer_11/zerosConst*
dtype0*
valueB*    
Ä
	Assign_11Assign)mio_variable/like_xtr/dense/bias/gradientInitializer_11/zeros*
T0*<
_class2
0.loc:@mio_variable/like_xtr/dense/bias/gradient*
validate_shape(*
use_locking(
K
like_xtr/dense/Tensordot/axesConst*
valueB:*
dtype0
R
like_xtr/dense/Tensordot/freeConst*
valueB"       *
dtype0
v
like_xtr/dense/Tensordot/ShapeShape4multi_head_attention/multi_head_attention_norm/add_1*
T0*
out_type0
P
&like_xtr/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0
Ŕ
!like_xtr/dense/Tensordot/GatherV2GatherV2like_xtr/dense/Tensordot/Shapelike_xtr/dense/Tensordot/free&like_xtr/dense/Tensordot/GatherV2/axis*
Tindices0*
Tparams0*
Taxis0
R
(like_xtr/dense/Tensordot/GatherV2_1/axisConst*
dtype0*
value	B : 
Ä
#like_xtr/dense/Tensordot/GatherV2_1GatherV2like_xtr/dense/Tensordot/Shapelike_xtr/dense/Tensordot/axes(like_xtr/dense/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0
L
like_xtr/dense/Tensordot/ConstConst*
dtype0*
valueB: 

like_xtr/dense/Tensordot/ProdProd!like_xtr/dense/Tensordot/GatherV2like_xtr/dense/Tensordot/Const*
T0*

Tidx0*
	keep_dims( 
N
 like_xtr/dense/Tensordot/Const_1Const*
valueB: *
dtype0

like_xtr/dense/Tensordot/Prod_1Prod#like_xtr/dense/Tensordot/GatherV2_1 like_xtr/dense/Tensordot/Const_1*
T0*

Tidx0*
	keep_dims( 
N
$like_xtr/dense/Tensordot/concat/axisConst*
value	B : *
dtype0
­
like_xtr/dense/Tensordot/concatConcatV2like_xtr/dense/Tensordot/freelike_xtr/dense/Tensordot/axes$like_xtr/dense/Tensordot/concat/axis*
N*

Tidx0*
T0

like_xtr/dense/Tensordot/stackPacklike_xtr/dense/Tensordot/Prodlike_xtr/dense/Tensordot/Prod_1*
T0*

axis *
N

"like_xtr/dense/Tensordot/transpose	Transpose4multi_head_attention/multi_head_attention_norm/add_1like_xtr/dense/Tensordot/concat*
T0*
Tperm0

 like_xtr/dense/Tensordot/ReshapeReshape"like_xtr/dense/Tensordot/transposelike_xtr/dense/Tensordot/stack*
T0*
Tshape0
^
)like_xtr/dense/Tensordot/transpose_1/permConst*
dtype0*
valueB"       

$like_xtr/dense/Tensordot/transpose_1	Transpose+mio_variable/like_xtr/dense/kernel/variable)like_xtr/dense/Tensordot/transpose_1/perm*
Tperm0*
T0
]
(like_xtr/dense/Tensordot/Reshape_1/shapeConst*
dtype0*
valueB"      

"like_xtr/dense/Tensordot/Reshape_1Reshape$like_xtr/dense/Tensordot/transpose_1(like_xtr/dense/Tensordot/Reshape_1/shape*
T0*
Tshape0

like_xtr/dense/Tensordot/MatMulMatMul like_xtr/dense/Tensordot/Reshape"like_xtr/dense/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
N
 like_xtr/dense/Tensordot/Const_2Const*
valueB:*
dtype0
P
&like_xtr/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0
¸
!like_xtr/dense/Tensordot/concat_1ConcatV2!like_xtr/dense/Tensordot/GatherV2 like_xtr/dense/Tensordot/Const_2&like_xtr/dense/Tensordot/concat_1/axis*
T0*
N*

Tidx0
~
like_xtr/dense/TensordotReshapelike_xtr/dense/Tensordot/MatMul!like_xtr/dense/Tensordot/concat_1*
T0*
Tshape0

like_xtr/dense/BiasAddBiasAddlike_xtr/dense/Tensordot)mio_variable/like_xtr/dense/bias/variable*
T0*
data_formatNHWC
B
like_xtr/dense/SigmoidSigmoidlike_xtr/dense/BiasAdd*
T0
D
Reshape_2/shapeConst*
valueB"˙˙˙˙   *
dtype0
T
	Reshape_2Reshapelike_xtr/dense/SigmoidReshape_2/shape*
T0*
Tshape0

,mio_variable/reply_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense/kernel*
shape:	

,mio_variable/reply_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense/kernel*
shape:	
X
#Initializer_12/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_12/random_uniform/minConst*
valueB
 *n×\ž*
dtype0
N
!Initializer_12/random_uniform/maxConst*
valueB
 *n×\>*
dtype0

+Initializer_12/random_uniform/RandomUniformRandomUniform#Initializer_12/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_12/random_uniform/subSub!Initializer_12/random_uniform/max!Initializer_12/random_uniform/min*
T0

!Initializer_12/random_uniform/mulMul+Initializer_12/random_uniform/RandomUniform!Initializer_12/random_uniform/sub*
T0
s
Initializer_12/random_uniformAdd!Initializer_12/random_uniform/mul!Initializer_12/random_uniform/min*
T0
Ó
	Assign_12Assign,mio_variable/reply_xtr/dense/kernel/gradientInitializer_12/random_uniform*
T0*?
_class5
31loc:@mio_variable/reply_xtr/dense/kernel/gradient*
validate_shape(*
use_locking(

*mio_variable/reply_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerreply_xtr/dense/bias*
shape:

*mio_variable/reply_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*#
	containerreply_xtr/dense/bias
E
Initializer_13/zerosConst*
valueB*    *
dtype0
Ć
	Assign_13Assign*mio_variable/reply_xtr/dense/bias/gradientInitializer_13/zeros*
T0*=
_class3
1/loc:@mio_variable/reply_xtr/dense/bias/gradient*
validate_shape(*
use_locking(
L
reply_xtr/dense/Tensordot/axesConst*
valueB:*
dtype0
S
reply_xtr/dense/Tensordot/freeConst*
dtype0*
valueB"       
w
reply_xtr/dense/Tensordot/ShapeShape4multi_head_attention/multi_head_attention_norm/add_1*
T0*
out_type0
Q
'reply_xtr/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0
Ä
"reply_xtr/dense/Tensordot/GatherV2GatherV2reply_xtr/dense/Tensordot/Shapereply_xtr/dense/Tensordot/free'reply_xtr/dense/Tensordot/GatherV2/axis*
Tparams0*
Taxis0*
Tindices0
S
)reply_xtr/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0
Č
$reply_xtr/dense/Tensordot/GatherV2_1GatherV2reply_xtr/dense/Tensordot/Shapereply_xtr/dense/Tensordot/axes)reply_xtr/dense/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0
M
reply_xtr/dense/Tensordot/ConstConst*
valueB: *
dtype0

reply_xtr/dense/Tensordot/ProdProd"reply_xtr/dense/Tensordot/GatherV2reply_xtr/dense/Tensordot/Const*

Tidx0*
	keep_dims( *
T0
O
!reply_xtr/dense/Tensordot/Const_1Const*
dtype0*
valueB: 

 reply_xtr/dense/Tensordot/Prod_1Prod$reply_xtr/dense/Tensordot/GatherV2_1!reply_xtr/dense/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0
O
%reply_xtr/dense/Tensordot/concat/axisConst*
dtype0*
value	B : 
ą
 reply_xtr/dense/Tensordot/concatConcatV2reply_xtr/dense/Tensordot/freereply_xtr/dense/Tensordot/axes%reply_xtr/dense/Tensordot/concat/axis*
T0*
N*

Tidx0

reply_xtr/dense/Tensordot/stackPackreply_xtr/dense/Tensordot/Prod reply_xtr/dense/Tensordot/Prod_1*
T0*

axis *
N

#reply_xtr/dense/Tensordot/transpose	Transpose4multi_head_attention/multi_head_attention_norm/add_1 reply_xtr/dense/Tensordot/concat*
Tperm0*
T0

!reply_xtr/dense/Tensordot/ReshapeReshape#reply_xtr/dense/Tensordot/transposereply_xtr/dense/Tensordot/stack*
T0*
Tshape0
_
*reply_xtr/dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
˘
%reply_xtr/dense/Tensordot/transpose_1	Transpose,mio_variable/reply_xtr/dense/kernel/variable*reply_xtr/dense/Tensordot/transpose_1/perm*
T0*
Tperm0
^
)reply_xtr/dense/Tensordot/Reshape_1/shapeConst*
valueB"      *
dtype0

#reply_xtr/dense/Tensordot/Reshape_1Reshape%reply_xtr/dense/Tensordot/transpose_1)reply_xtr/dense/Tensordot/Reshape_1/shape*
T0*
Tshape0
Ą
 reply_xtr/dense/Tensordot/MatMulMatMul!reply_xtr/dense/Tensordot/Reshape#reply_xtr/dense/Tensordot/Reshape_1*
transpose_b( *
T0*
transpose_a( 
O
!reply_xtr/dense/Tensordot/Const_2Const*
dtype0*
valueB:
Q
'reply_xtr/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0
ź
"reply_xtr/dense/Tensordot/concat_1ConcatV2"reply_xtr/dense/Tensordot/GatherV2!reply_xtr/dense/Tensordot/Const_2'reply_xtr/dense/Tensordot/concat_1/axis*
T0*
N*

Tidx0

reply_xtr/dense/TensordotReshape reply_xtr/dense/Tensordot/MatMul"reply_xtr/dense/Tensordot/concat_1*
T0*
Tshape0

reply_xtr/dense/BiasAddBiasAddreply_xtr/dense/Tensordot*mio_variable/reply_xtr/dense/bias/variable*
T0*
data_formatNHWC
D
reply_xtr/dense/SigmoidSigmoidreply_xtr/dense/BiasAdd*
T0
D
Reshape_3/shapeConst*
dtype0*
valueB"˙˙˙˙   
U
	Reshape_3Reshapereply_xtr/dense/SigmoidReshape_3/shape*
T0*
Tshape0"