
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
dtype0*
shape: 
K
MIO_TABLE_ADDRESSConst"/device:CPU:0*
value
B  *
dtype0
Ĩ
2mio_compress_indices/COMPRESS_INDEX__USER/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerCOMPRESS_INDEX__USER*
shape:˙˙˙˙˙˙˙˙˙
Ĩ
2mio_compress_indices/COMPRESS_INDEX__USER/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerCOMPRESS_INDEX__USER*
shape:˙˙˙˙˙˙˙˙˙
h
CastCast2mio_compress_indices/COMPRESS_INDEX__USER/variable*
Truncate( *

DstT0*

SrcT0

&mio_embeddings/user_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containeruser_embedding*
shape:˙˙˙˙˙˙˙˙˙

&mio_embeddings/user_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containeruser_embedding*
shape:˙˙˙˙˙˙˙˙˙

&mio_embeddings/c_id_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerc_id_embedding*
shape:˙˙˙˙˙˙˙˙˙

&mio_embeddings/c_id_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerc_id_embedding*
shape:˙˙˙˙˙˙˙˙˙

(mio_embeddings/c_info_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:˙˙˙˙˙˙˙˙˙Ā*
	containerc_info_embedding

(mio_embeddings/c_info_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerc_info_embedding*
shape:˙˙˙˙˙˙˙˙˙Ā

*mio_embeddings/position_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*!
	containerposition_embedding*
shape:˙˙˙˙˙˙˙˙˙

*mio_embeddings/position_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*!
	containerposition_embedding*
shape:˙˙˙˙˙˙˙˙˙
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
concat/axisConst*
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
Ė
concatConcatV2concat/values_0&mio_embeddings/c_id_embedding/variable(mio_embeddings/c_info_embedding/variable*mio_embeddings/position_embedding/variableconcat/axis*
N*

Tidx0*
T0
 
-mio_variable/expand_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
Đ*&
	containerexpand_xtr/dense/kernel
 
-mio_variable/expand_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
Đ*&
	containerexpand_xtr/dense/kernel
U
 Initializer/random_uniform/shapeConst*
valueB"P     *
dtype0
K
Initializer/random_uniform/minConst*
valueB
 *Ü-ÎŊ*
dtype0
K
Initializer/random_uniform/maxConst*
dtype0*
valueB
 *Ü-Î=
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
Ī
AssignAssign-mio_variable/expand_xtr/dense/kernel/gradientInitializer/random_uniform*
use_locking(*
T0*@
_class6
42loc:@mio_variable/expand_xtr/dense/kernel/gradient*
validate_shape(

+mio_variable/expand_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerexpand_xtr/dense/bias*
shape:

+mio_variable/expand_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerexpand_xtr/dense/bias*
shape:
E
Initializer_1/zerosConst*
valueB*    *
dtype0
Æ
Assign_1Assign+mio_variable/expand_xtr/dense/bias/gradientInitializer_1/zeros*
use_locking(*
T0*>
_class4
20loc:@mio_variable/expand_xtr/dense/bias/gradient*
validate_shape(

expand_xtr/dense/MatMulMatMulconcat-mio_variable/expand_xtr/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 

expand_xtr/dense/BiasAddBiasAddexpand_xtr/dense/MatMul+mio_variable/expand_xtr/dense/bias/variable*
T0*
data_formatNHWC
M
 expand_xtr/dense/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍĖL>
j
expand_xtr/dense/LeakyRelu/mulMul expand_xtr/dense/LeakyRelu/alphaexpand_xtr/dense/BiasAdd*
T0
h
expand_xtr/dense/LeakyReluMaximumexpand_xtr/dense/LeakyRelu/mulexpand_xtr/dense/BiasAdd*
T0
¤
/mio_variable/expand_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*(
	containerexpand_xtr/dense_1/kernel
¤
/mio_variable/expand_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*(
	containerexpand_xtr/dense_1/kernel
W
"Initializer_2/random_uniform/shapeConst*
dtype0*
valueB"      
M
 Initializer_2/random_uniform/minConst*
valueB
 *   ž*
dtype0
M
 Initializer_2/random_uniform/maxConst*
valueB
 *   >*
dtype0

*Initializer_2/random_uniform/RandomUniformRandomUniform"Initializer_2/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
t
 Initializer_2/random_uniform/subSub Initializer_2/random_uniform/max Initializer_2/random_uniform/min*
T0
~
 Initializer_2/random_uniform/mulMul*Initializer_2/random_uniform/RandomUniform Initializer_2/random_uniform/sub*
T0
p
Initializer_2/random_uniformAdd Initializer_2/random_uniform/mul Initializer_2/random_uniform/min*
T0
×
Assign_2Assign/mio_variable/expand_xtr/dense_1/kernel/gradientInitializer_2/random_uniform*
validate_shape(*
use_locking(*
T0*B
_class8
64loc:@mio_variable/expand_xtr/dense_1/kernel/gradient

-mio_variable/expand_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerexpand_xtr/dense_1/bias*
shape:

-mio_variable/expand_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerexpand_xtr/dense_1/bias*
shape:
E
Initializer_3/zerosConst*
valueB*    *
dtype0
Ę
Assign_3Assign-mio_variable/expand_xtr/dense_1/bias/gradientInitializer_3/zeros*
T0*@
_class6
42loc:@mio_variable/expand_xtr/dense_1/bias/gradient*
validate_shape(*
use_locking(

expand_xtr/dense_1/MatMulMatMulexpand_xtr/dense/LeakyRelu/mio_variable/expand_xtr/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0

expand_xtr/dense_1/BiasAddBiasAddexpand_xtr/dense_1/MatMul-mio_variable/expand_xtr/dense_1/bias/variable*
T0*
data_formatNHWC
O
"expand_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0
p
 expand_xtr/dense_1/LeakyRelu/mulMul"expand_xtr/dense_1/LeakyRelu/alphaexpand_xtr/dense_1/BiasAdd*
T0
n
expand_xtr/dense_1/LeakyReluMaximum expand_xtr/dense_1/LeakyRelu/mulexpand_xtr/dense_1/BiasAdd*
T0
Ŗ
/mio_variable/expand_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerexpand_xtr/dense_2/kernel*
shape:	@
Ŗ
/mio_variable/expand_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerexpand_xtr/dense_2/kernel*
shape:	@
W
"Initializer_4/random_uniform/shapeConst*
valueB"   @   *
dtype0
M
 Initializer_4/random_uniform/minConst*
dtype0*
valueB
 *ķ5ž
M
 Initializer_4/random_uniform/maxConst*
valueB
 *ķ5>*
dtype0

*Initializer_4/random_uniform/RandomUniformRandomUniform"Initializer_4/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
t
 Initializer_4/random_uniform/subSub Initializer_4/random_uniform/max Initializer_4/random_uniform/min*
T0
~
 Initializer_4/random_uniform/mulMul*Initializer_4/random_uniform/RandomUniform Initializer_4/random_uniform/sub*
T0
p
Initializer_4/random_uniformAdd Initializer_4/random_uniform/mul Initializer_4/random_uniform/min*
T0
×
Assign_4Assign/mio_variable/expand_xtr/dense_2/kernel/gradientInitializer_4/random_uniform*
T0*B
_class8
64loc:@mio_variable/expand_xtr/dense_2/kernel/gradient*
validate_shape(*
use_locking(

-mio_variable/expand_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerexpand_xtr/dense_2/bias*
shape:@

-mio_variable/expand_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerexpand_xtr/dense_2/bias*
shape:@
D
Initializer_5/zerosConst*
valueB@*    *
dtype0
Ę
Assign_5Assign-mio_variable/expand_xtr/dense_2/bias/gradientInitializer_5/zeros*
use_locking(*
T0*@
_class6
42loc:@mio_variable/expand_xtr/dense_2/bias/gradient*
validate_shape(
Ą
expand_xtr/dense_2/MatMulMatMulexpand_xtr/dense_1/LeakyRelu/mio_variable/expand_xtr/dense_2/kernel/variable*
transpose_b( *
T0*
transpose_a( 

expand_xtr/dense_2/BiasAddBiasAddexpand_xtr/dense_2/MatMul-mio_variable/expand_xtr/dense_2/bias/variable*
T0*
data_formatNHWC
O
"expand_xtr/dense_2/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍĖL>
p
 expand_xtr/dense_2/LeakyRelu/mulMul"expand_xtr/dense_2/LeakyRelu/alphaexpand_xtr/dense_2/BiasAdd*
T0
n
expand_xtr/dense_2/LeakyReluMaximum expand_xtr/dense_2/LeakyRelu/mulexpand_xtr/dense_2/BiasAdd*
T0
ĸ
/mio_variable/expand_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerexpand_xtr/dense_3/kernel*
shape
:@
ĸ
/mio_variable/expand_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerexpand_xtr/dense_3/kernel*
shape
:@
W
"Initializer_6/random_uniform/shapeConst*
dtype0*
valueB"@      
M
 Initializer_6/random_uniform/minConst*
dtype0*
valueB
 *ž
M
 Initializer_6/random_uniform/maxConst*
valueB
 *>*
dtype0

*Initializer_6/random_uniform/RandomUniformRandomUniform"Initializer_6/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
t
 Initializer_6/random_uniform/subSub Initializer_6/random_uniform/max Initializer_6/random_uniform/min*
T0
~
 Initializer_6/random_uniform/mulMul*Initializer_6/random_uniform/RandomUniform Initializer_6/random_uniform/sub*
T0
p
Initializer_6/random_uniformAdd Initializer_6/random_uniform/mul Initializer_6/random_uniform/min*
T0
×
Assign_6Assign/mio_variable/expand_xtr/dense_3/kernel/gradientInitializer_6/random_uniform*
use_locking(*
T0*B
_class8
64loc:@mio_variable/expand_xtr/dense_3/kernel/gradient*
validate_shape(

-mio_variable/expand_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerexpand_xtr/dense_3/bias*
shape:

-mio_variable/expand_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerexpand_xtr/dense_3/bias*
shape:
D
Initializer_7/zerosConst*
valueB*    *
dtype0
Ę
Assign_7Assign-mio_variable/expand_xtr/dense_3/bias/gradientInitializer_7/zeros*
T0*@
_class6
42loc:@mio_variable/expand_xtr/dense_3/bias/gradient*
validate_shape(*
use_locking(
Ą
expand_xtr/dense_3/MatMulMatMulexpand_xtr/dense_2/LeakyRelu/mio_variable/expand_xtr/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 

expand_xtr/dense_3/BiasAddBiasAddexpand_xtr/dense_3/MatMul-mio_variable/expand_xtr/dense_3/bias/variable*
T0*
data_formatNHWC
J
expand_xtr/dense_3/SigmoidSigmoidexpand_xtr/dense_3/BiasAdd*
T0

+mio_variable/like_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerlike_xtr/dense/kernel*
shape:
Đ

+mio_variable/like_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerlike_xtr/dense/kernel*
shape:
Đ
W
"Initializer_8/random_uniform/shapeConst*
valueB"P     *
dtype0
M
 Initializer_8/random_uniform/minConst*
valueB
 *Ü-ÎŊ*
dtype0
M
 Initializer_8/random_uniform/maxConst*
valueB
 *Ü-Î=*
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
Ī
Assign_8Assign+mio_variable/like_xtr/dense/kernel/gradientInitializer_8/random_uniform*
T0*>
_class4
20loc:@mio_variable/like_xtr/dense/kernel/gradient*
validate_shape(*
use_locking(

)mio_variable/like_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*"
	containerlike_xtr/dense/bias*
shape:

)mio_variable/like_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*"
	containerlike_xtr/dense/bias*
shape:
E
Initializer_9/zerosConst*
valueB*    *
dtype0
Â
Assign_9Assign)mio_variable/like_xtr/dense/bias/gradientInitializer_9/zeros*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@mio_variable/like_xtr/dense/bias/gradient

like_xtr/dense/MatMulMatMulconcat+mio_variable/like_xtr/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 

like_xtr/dense/BiasAddBiasAddlike_xtr/dense/MatMul)mio_variable/like_xtr/dense/bias/variable*
T0*
data_formatNHWC
K
like_xtr/dense/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0
d
like_xtr/dense/LeakyRelu/mulMullike_xtr/dense/LeakyRelu/alphalike_xtr/dense/BiasAdd*
T0
b
like_xtr/dense/LeakyReluMaximumlike_xtr/dense/LeakyRelu/mullike_xtr/dense/BiasAdd*
T0
 
-mio_variable/like_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerlike_xtr/dense_1/kernel*
shape:

 
-mio_variable/like_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*&
	containerlike_xtr/dense_1/kernel
X
#Initializer_10/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_10/random_uniform/minConst*
dtype0*
valueB
 *   ž
N
!Initializer_10/random_uniform/maxConst*
valueB
 *   >*
dtype0

+Initializer_10/random_uniform/RandomUniformRandomUniform#Initializer_10/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_10/random_uniform/subSub!Initializer_10/random_uniform/max!Initializer_10/random_uniform/min*
T0

!Initializer_10/random_uniform/mulMul+Initializer_10/random_uniform/RandomUniform!Initializer_10/random_uniform/sub*
T0
s
Initializer_10/random_uniformAdd!Initializer_10/random_uniform/mul!Initializer_10/random_uniform/min*
T0
Õ
	Assign_10Assign-mio_variable/like_xtr/dense_1/kernel/gradientInitializer_10/random_uniform*
use_locking(*
T0*@
_class6
42loc:@mio_variable/like_xtr/dense_1/kernel/gradient*
validate_shape(

+mio_variable/like_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerlike_xtr/dense_1/bias*
shape:

+mio_variable/like_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerlike_xtr/dense_1/bias*
shape:
F
Initializer_11/zerosConst*
valueB*    *
dtype0
Č
	Assign_11Assign+mio_variable/like_xtr/dense_1/bias/gradientInitializer_11/zeros*
use_locking(*
T0*>
_class4
20loc:@mio_variable/like_xtr/dense_1/bias/gradient*
validate_shape(

like_xtr/dense_1/MatMulMatMullike_xtr/dense/LeakyRelu-mio_variable/like_xtr/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 

like_xtr/dense_1/BiasAddBiasAddlike_xtr/dense_1/MatMul+mio_variable/like_xtr/dense_1/bias/variable*
T0*
data_formatNHWC
M
 like_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0
j
like_xtr/dense_1/LeakyRelu/mulMul like_xtr/dense_1/LeakyRelu/alphalike_xtr/dense_1/BiasAdd*
T0
h
like_xtr/dense_1/LeakyReluMaximumlike_xtr/dense_1/LeakyRelu/mullike_xtr/dense_1/BiasAdd*
T0

-mio_variable/like_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerlike_xtr/dense_2/kernel*
shape:	@

-mio_variable/like_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerlike_xtr/dense_2/kernel*
shape:	@
X
#Initializer_12/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_12/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
N
!Initializer_12/random_uniform/maxConst*
valueB
 *ķ5>*
dtype0

+Initializer_12/random_uniform/RandomUniformRandomUniform#Initializer_12/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_12/random_uniform/subSub!Initializer_12/random_uniform/max!Initializer_12/random_uniform/min*
T0

!Initializer_12/random_uniform/mulMul+Initializer_12/random_uniform/RandomUniform!Initializer_12/random_uniform/sub*
T0
s
Initializer_12/random_uniformAdd!Initializer_12/random_uniform/mul!Initializer_12/random_uniform/min*
T0
Õ
	Assign_12Assign-mio_variable/like_xtr/dense_2/kernel/gradientInitializer_12/random_uniform*
use_locking(*
T0*@
_class6
42loc:@mio_variable/like_xtr/dense_2/kernel/gradient*
validate_shape(

+mio_variable/like_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*$
	containerlike_xtr/dense_2/bias

+mio_variable/like_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerlike_xtr/dense_2/bias*
shape:@
E
Initializer_13/zerosConst*
valueB@*    *
dtype0
Č
	Assign_13Assign+mio_variable/like_xtr/dense_2/bias/gradientInitializer_13/zeros*
use_locking(*
T0*>
_class4
20loc:@mio_variable/like_xtr/dense_2/bias/gradient*
validate_shape(

like_xtr/dense_2/MatMulMatMullike_xtr/dense_1/LeakyRelu-mio_variable/like_xtr/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 

like_xtr/dense_2/BiasAddBiasAddlike_xtr/dense_2/MatMul+mio_variable/like_xtr/dense_2/bias/variable*
T0*
data_formatNHWC
M
 like_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0
j
like_xtr/dense_2/LeakyRelu/mulMul like_xtr/dense_2/LeakyRelu/alphalike_xtr/dense_2/BiasAdd*
T0
h
like_xtr/dense_2/LeakyReluMaximumlike_xtr/dense_2/LeakyRelu/mullike_xtr/dense_2/BiasAdd*
T0

-mio_variable/like_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerlike_xtr/dense_3/kernel*
shape
:@

-mio_variable/like_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerlike_xtr/dense_3/kernel*
shape
:@
X
#Initializer_14/random_uniform/shapeConst*
dtype0*
valueB"@      
N
!Initializer_14/random_uniform/minConst*
valueB
 *ž*
dtype0
N
!Initializer_14/random_uniform/maxConst*
valueB
 *>*
dtype0

+Initializer_14/random_uniform/RandomUniformRandomUniform#Initializer_14/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_14/random_uniform/subSub!Initializer_14/random_uniform/max!Initializer_14/random_uniform/min*
T0

!Initializer_14/random_uniform/mulMul+Initializer_14/random_uniform/RandomUniform!Initializer_14/random_uniform/sub*
T0
s
Initializer_14/random_uniformAdd!Initializer_14/random_uniform/mul!Initializer_14/random_uniform/min*
T0
Õ
	Assign_14Assign-mio_variable/like_xtr/dense_3/kernel/gradientInitializer_14/random_uniform*
validate_shape(*
use_locking(*
T0*@
_class6
42loc:@mio_variable/like_xtr/dense_3/kernel/gradient

+mio_variable/like_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerlike_xtr/dense_3/bias*
shape:

+mio_variable/like_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerlike_xtr/dense_3/bias*
shape:
E
Initializer_15/zerosConst*
valueB*    *
dtype0
Č
	Assign_15Assign+mio_variable/like_xtr/dense_3/bias/gradientInitializer_15/zeros*
T0*>
_class4
20loc:@mio_variable/like_xtr/dense_3/bias/gradient*
validate_shape(*
use_locking(

like_xtr/dense_3/MatMulMatMullike_xtr/dense_2/LeakyRelu-mio_variable/like_xtr/dense_3/kernel/variable*
transpose_b( *
T0*
transpose_a( 

like_xtr/dense_3/BiasAddBiasAddlike_xtr/dense_3/MatMul+mio_variable/like_xtr/dense_3/bias/variable*
data_formatNHWC*
T0
F
like_xtr/dense_3/SigmoidSigmoidlike_xtr/dense_3/BiasAdd*
T0

,mio_variable/reply_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense/kernel*
shape:
Đ

,mio_variable/reply_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense/kernel*
shape:
Đ
X
#Initializer_16/random_uniform/shapeConst*
dtype0*
valueB"P     
N
!Initializer_16/random_uniform/minConst*
valueB
 *Ü-ÎŊ*
dtype0
N
!Initializer_16/random_uniform/maxConst*
dtype0*
valueB
 *Ü-Î=

+Initializer_16/random_uniform/RandomUniformRandomUniform#Initializer_16/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_16/random_uniform/subSub!Initializer_16/random_uniform/max!Initializer_16/random_uniform/min*
T0

!Initializer_16/random_uniform/mulMul+Initializer_16/random_uniform/RandomUniform!Initializer_16/random_uniform/sub*
T0
s
Initializer_16/random_uniformAdd!Initializer_16/random_uniform/mul!Initializer_16/random_uniform/min*
T0
Ķ
	Assign_16Assign,mio_variable/reply_xtr/dense/kernel/gradientInitializer_16/random_uniform*
use_locking(*
T0*?
_class5
31loc:@mio_variable/reply_xtr/dense/kernel/gradient*
validate_shape(

*mio_variable/reply_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerreply_xtr/dense/bias*
shape:

*mio_variable/reply_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerreply_xtr/dense/bias*
shape:
F
Initializer_17/zerosConst*
dtype0*
valueB*    
Æ
	Assign_17Assign*mio_variable/reply_xtr/dense/bias/gradientInitializer_17/zeros*
T0*=
_class3
1/loc:@mio_variable/reply_xtr/dense/bias/gradient*
validate_shape(*
use_locking(

reply_xtr/dense/MatMulMatMulconcat,mio_variable/reply_xtr/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 

reply_xtr/dense/BiasAddBiasAddreply_xtr/dense/MatMul*mio_variable/reply_xtr/dense/bias/variable*
T0*
data_formatNHWC
L
reply_xtr/dense/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0
g
reply_xtr/dense/LeakyRelu/mulMulreply_xtr/dense/LeakyRelu/alphareply_xtr/dense/BiasAdd*
T0
e
reply_xtr/dense/LeakyReluMaximumreply_xtr/dense/LeakyRelu/mulreply_xtr/dense/BiasAdd*
T0
ĸ
.mio_variable/reply_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*'
	containerreply_xtr/dense_1/kernel
ĸ
.mio_variable/reply_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*'
	containerreply_xtr/dense_1/kernel
X
#Initializer_18/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_18/random_uniform/minConst*
valueB
 *   ž*
dtype0
N
!Initializer_18/random_uniform/maxConst*
valueB
 *   >*
dtype0

+Initializer_18/random_uniform/RandomUniformRandomUniform#Initializer_18/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_18/random_uniform/subSub!Initializer_18/random_uniform/max!Initializer_18/random_uniform/min*
T0

!Initializer_18/random_uniform/mulMul+Initializer_18/random_uniform/RandomUniform!Initializer_18/random_uniform/sub*
T0
s
Initializer_18/random_uniformAdd!Initializer_18/random_uniform/mul!Initializer_18/random_uniform/min*
T0
×
	Assign_18Assign.mio_variable/reply_xtr/dense_1/kernel/gradientInitializer_18/random_uniform*
use_locking(*
T0*A
_class7
53loc:@mio_variable/reply_xtr/dense_1/kernel/gradient*
validate_shape(

,mio_variable/reply_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense_1/bias*
shape:

,mio_variable/reply_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense_1/bias*
shape:
F
Initializer_19/zerosConst*
dtype0*
valueB*    
Ę
	Assign_19Assign,mio_variable/reply_xtr/dense_1/bias/gradientInitializer_19/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/reply_xtr/dense_1/bias/gradient*
validate_shape(

reply_xtr/dense_1/MatMulMatMulreply_xtr/dense/LeakyRelu.mio_variable/reply_xtr/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 

reply_xtr/dense_1/BiasAddBiasAddreply_xtr/dense_1/MatMul,mio_variable/reply_xtr/dense_1/bias/variable*
T0*
data_formatNHWC
N
!reply_xtr/dense_1/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍĖL>
m
reply_xtr/dense_1/LeakyRelu/mulMul!reply_xtr/dense_1/LeakyRelu/alphareply_xtr/dense_1/BiasAdd*
T0
k
reply_xtr/dense_1/LeakyReluMaximumreply_xtr/dense_1/LeakyRelu/mulreply_xtr/dense_1/BiasAdd*
T0
Ą
.mio_variable/reply_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerreply_xtr/dense_2/kernel*
shape:	@
Ą
.mio_variable/reply_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerreply_xtr/dense_2/kernel*
shape:	@
X
#Initializer_20/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_20/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
N
!Initializer_20/random_uniform/maxConst*
valueB
 *ķ5>*
dtype0

+Initializer_20/random_uniform/RandomUniformRandomUniform#Initializer_20/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_20/random_uniform/subSub!Initializer_20/random_uniform/max!Initializer_20/random_uniform/min*
T0

!Initializer_20/random_uniform/mulMul+Initializer_20/random_uniform/RandomUniform!Initializer_20/random_uniform/sub*
T0
s
Initializer_20/random_uniformAdd!Initializer_20/random_uniform/mul!Initializer_20/random_uniform/min*
T0
×
	Assign_20Assign.mio_variable/reply_xtr/dense_2/kernel/gradientInitializer_20/random_uniform*
T0*A
_class7
53loc:@mio_variable/reply_xtr/dense_2/kernel/gradient*
validate_shape(*
use_locking(

,mio_variable/reply_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense_2/bias*
shape:@

,mio_variable/reply_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense_2/bias*
shape:@
E
Initializer_21/zerosConst*
valueB@*    *
dtype0
Ę
	Assign_21Assign,mio_variable/reply_xtr/dense_2/bias/gradientInitializer_21/zeros*
T0*?
_class5
31loc:@mio_variable/reply_xtr/dense_2/bias/gradient*
validate_shape(*
use_locking(

reply_xtr/dense_2/MatMulMatMulreply_xtr/dense_1/LeakyRelu.mio_variable/reply_xtr/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0

reply_xtr/dense_2/BiasAddBiasAddreply_xtr/dense_2/MatMul,mio_variable/reply_xtr/dense_2/bias/variable*
T0*
data_formatNHWC
N
!reply_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0
m
reply_xtr/dense_2/LeakyRelu/mulMul!reply_xtr/dense_2/LeakyRelu/alphareply_xtr/dense_2/BiasAdd*
T0
k
reply_xtr/dense_2/LeakyReluMaximumreply_xtr/dense_2/LeakyRelu/mulreply_xtr/dense_2/BiasAdd*
T0
 
.mio_variable/reply_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerreply_xtr/dense_3/kernel*
shape
:@
 
.mio_variable/reply_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerreply_xtr/dense_3/kernel*
shape
:@
X
#Initializer_22/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_22/random_uniform/minConst*
dtype0*
valueB
 *ž
N
!Initializer_22/random_uniform/maxConst*
valueB
 *>*
dtype0

+Initializer_22/random_uniform/RandomUniformRandomUniform#Initializer_22/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_22/random_uniform/subSub!Initializer_22/random_uniform/max!Initializer_22/random_uniform/min*
T0

!Initializer_22/random_uniform/mulMul+Initializer_22/random_uniform/RandomUniform!Initializer_22/random_uniform/sub*
T0
s
Initializer_22/random_uniformAdd!Initializer_22/random_uniform/mul!Initializer_22/random_uniform/min*
T0
×
	Assign_22Assign.mio_variable/reply_xtr/dense_3/kernel/gradientInitializer_22/random_uniform*
T0*A
_class7
53loc:@mio_variable/reply_xtr/dense_3/kernel/gradient*
validate_shape(*
use_locking(

,mio_variable/reply_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*%
	containerreply_xtr/dense_3/bias

,mio_variable/reply_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense_3/bias*
shape:
E
Initializer_23/zerosConst*
valueB*    *
dtype0
Ę
	Assign_23Assign,mio_variable/reply_xtr/dense_3/bias/gradientInitializer_23/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/reply_xtr/dense_3/bias/gradient*
validate_shape(

reply_xtr/dense_3/MatMulMatMulreply_xtr/dense_2/LeakyRelu.mio_variable/reply_xtr/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 

reply_xtr/dense_3/BiasAddBiasAddreply_xtr/dense_3/MatMul,mio_variable/reply_xtr/dense_3/bias/variable*
data_formatNHWC*
T0
H
reply_xtr/dense_3/SigmoidSigmoidreply_xtr/dense_3/BiasAdd*
T0"