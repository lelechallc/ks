
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
shape:ĸĸĸĸĸĸĸĸĸ
Ĩ
2mio_compress_indices/COMPRESS_INDEX__USER/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerCOMPRESS_INDEX__USER*
shape:ĸĸĸĸĸĸĸĸĸ
h
CastCast2mio_compress_indices/COMPRESS_INDEX__USER/variable*

SrcT0*
Truncate( *

DstT0

&mio_embeddings/user_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containeruser_embedding*
shape:ĸĸĸĸĸĸĸĸĸ

&mio_embeddings/user_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containeruser_embedding*
shape:ĸĸĸĸĸĸĸĸĸ

*mio_embeddings/context_embedding1/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*!
	containercontext_embedding1*
shape:ĸĸĸĸĸĸĸĸĸ 

*mio_embeddings/context_embedding1/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*!
	containercontext_embedding1*
shape:ĸĸĸĸĸĸĸĸĸ 

*mio_embeddings/context_embedding2/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*!
	containercontext_embedding2*
shape:ĸĸĸĸĸĸĸĸĸ

*mio_embeddings/context_embedding2/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:ĸĸĸĸĸĸĸĸĸ*!
	containercontext_embedding2
 
*mio_embeddings/c_udp_id_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:ĸĸĸĸĸĸĸĸĸ*!
	containerc_udp_id_embedding
 
*mio_embeddings/c_udp_id_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:ĸĸĸĸĸĸĸĸĸ*!
	containerc_udp_id_embedding

&mio_embeddings/c_id_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerc_id_embedding*
shape:ĸĸĸĸĸĸĸĸĸ

&mio_embeddings/c_id_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerc_id_embedding*
shape:ĸĸĸĸĸĸĸĸĸ

(mio_embeddings/c_info_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:ĸĸĸĸĸĸĸĸĸĀ*
	containerc_info_embedding

(mio_embeddings/c_info_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerc_info_embedding*
shape:ĸĸĸĸĸĸĸĸĸĀ

*mio_embeddings/position_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*!
	containerposition_embedding*
shape:ĸĸĸĸĸĸĸĸĸ

*mio_embeddings/position_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*!
	containerposition_embedding*
shape:ĸĸĸĸĸĸĸĸĸ
>
concat/values_0/axisConst*
value	B : *
dtype0

concat/values_0GatherV2&mio_embeddings/user_embedding/variableCastconcat/values_0/axis*
Tindices0*
Tparams0*
Taxis0
>
concat/values_4/axisConst*
value	B : *
dtype0

concat/values_4GatherV2*mio_embeddings/c_udp_id_embedding/variableCastconcat/values_4/axis*
Taxis0*
Tindices0*
Tparams0
>
concat/values_5/axisConst*
value	B : *
dtype0

concat/values_5GatherV2*mio_embeddings/context_embedding1/variableCastconcat/values_5/axis*
Tparams0*
Taxis0*
Tindices0
>
concat/values_6/axisConst*
value	B : *
dtype0

concat/values_6GatherV2*mio_embeddings/context_embedding2/variableCastconcat/values_6/axis*
Taxis0*
Tindices0*
Tparams0
>
concat/axisConst*
valueB :
ĸĸĸĸĸĸĸĸĸ*
dtype0
ĸ
concatConcatV2concat/values_0&mio_embeddings/c_id_embedding/variable(mio_embeddings/c_info_embedding/variable*mio_embeddings/position_embedding/variableconcat/values_4concat/values_5concat/values_6concat/axis*
T0*
N*

Tidx0
 
-mio_variable/expand_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerexpand_xtr/dense/kernel*
shape:

 
-mio_variable/expand_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerexpand_xtr/dense/kernel*
shape:

U
 Initializer/random_uniform/shapeConst*
valueB"     *
dtype0
K
Initializer/random_uniform/minConst*
valueB
 *b§―*
dtype0
K
Initializer/random_uniform/maxConst*
valueB
 *b§=*
dtype0

(Initializer/random_uniform/RandomUniformRandomUniform Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
n
Initializer/random_uniform/subSubInitializer/random_uniform/maxInitializer/random_uniform/min*
T0
x
Initializer/random_uniform/mulMul(Initializer/random_uniform/RandomUniformInitializer/random_uniform/sub*
T0
j
Initializer/random_uniformAddInitializer/random_uniform/mulInitializer/random_uniform/min*
T0
Ï
AssignAssign-mio_variable/expand_xtr/dense/kernel/gradientInitializer/random_uniform*
T0*@
_class6
42loc:@mio_variable/expand_xtr/dense/kernel/gradient*
validate_shape(*
use_locking(

+mio_variable/expand_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containerexpand_xtr/dense/bias

+mio_variable/expand_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containerexpand_xtr/dense/bias
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
expand_xtr/dense/MatMulMatMulconcat-mio_variable/expand_xtr/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
Ī
/mio_variable/expand_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*(
	containerexpand_xtr/dense_1/kernel
Ī
/mio_variable/expand_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerexpand_xtr/dense_1/kernel*
shape:

W
"Initializer_2/random_uniform/shapeConst*
dtype0*
valueB"      
M
 Initializer_2/random_uniform/minConst*
valueB
 *   ū*
dtype0
M
 Initializer_2/random_uniform/maxConst*
valueB
 *   >*
dtype0

*Initializer_2/random_uniform/RandomUniformRandomUniform"Initializer_2/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
t
 Initializer_2/random_uniform/subSub Initializer_2/random_uniform/max Initializer_2/random_uniform/min*
T0
~
 Initializer_2/random_uniform/mulMul*Initializer_2/random_uniform/RandomUniform Initializer_2/random_uniform/sub*
T0
p
Initializer_2/random_uniformAdd Initializer_2/random_uniform/mul Initializer_2/random_uniform/min*
T0
Ũ
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
Initializer_3/zerosConst*
dtype0*
valueB*    
Ę
Assign_3Assign-mio_variable/expand_xtr/dense_1/bias/gradientInitializer_3/zeros*
use_locking(*
T0*@
_class6
42loc:@mio_variable/expand_xtr/dense_1/bias/gradient*
validate_shape(
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
Ģ
/mio_variable/expand_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerexpand_xtr/dense_2/kernel*
shape:	@
Ģ
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
 *ó5ū
M
 Initializer_4/random_uniform/maxConst*
valueB
 *ó5>*
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
Ũ
Assign_4Assign/mio_variable/expand_xtr/dense_2/kernel/gradientInitializer_4/random_uniform*
T0*B
_class8
64loc:@mio_variable/expand_xtr/dense_2/kernel/gradient*
validate_shape(*
use_locking(

-mio_variable/expand_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*&
	containerexpand_xtr/dense_2/bias

-mio_variable/expand_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerexpand_xtr/dense_2/bias*
shape:@
D
Initializer_5/zerosConst*
valueB@*    *
dtype0
Ę
Assign_5Assign-mio_variable/expand_xtr/dense_2/bias/gradientInitializer_5/zeros*
validate_shape(*
use_locking(*
T0*@
_class6
42loc:@mio_variable/expand_xtr/dense_2/bias/gradient
Ą
expand_xtr/dense_2/MatMulMatMulexpand_xtr/dense_1/LeakyRelu/mio_variable/expand_xtr/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0

expand_xtr/dense_2/BiasAddBiasAddexpand_xtr/dense_2/MatMul-mio_variable/expand_xtr/dense_2/bias/variable*
T0*
data_formatNHWC
O
"expand_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0
p
 expand_xtr/dense_2/LeakyRelu/mulMul"expand_xtr/dense_2/LeakyRelu/alphaexpand_xtr/dense_2/BiasAdd*
T0
n
expand_xtr/dense_2/LeakyReluMaximum expand_xtr/dense_2/LeakyRelu/mulexpand_xtr/dense_2/BiasAdd*
T0
Ē
/mio_variable/expand_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerexpand_xtr/dense_3/kernel*
shape
:@
Ē
/mio_variable/expand_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerexpand_xtr/dense_3/kernel*
shape
:@
W
"Initializer_6/random_uniform/shapeConst*
valueB"@      *
dtype0
M
 Initializer_6/random_uniform/minConst*
dtype0*
valueB
 *ū
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
Ũ
Assign_6Assign/mio_variable/expand_xtr/dense_3/kernel/gradientInitializer_6/random_uniform*
T0*B
_class8
64loc:@mio_variable/expand_xtr/dense_3/kernel/gradient*
validate_shape(*
use_locking(

-mio_variable/expand_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerexpand_xtr/dense_3/bias*
shape:

-mio_variable/expand_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*&
	containerexpand_xtr/dense_3/bias
D
Initializer_7/zerosConst*
dtype0*
valueB*    
Ę
Assign_7Assign-mio_variable/expand_xtr/dense_3/bias/gradientInitializer_7/zeros*
use_locking(*
T0*@
_class6
42loc:@mio_variable/expand_xtr/dense_3/bias/gradient*
validate_shape(
Ą
expand_xtr/dense_3/MatMulMatMulexpand_xtr/dense_2/LeakyRelu/mio_variable/expand_xtr/dense_3/kernel/variable*
transpose_b( *
T0*
transpose_a( 

expand_xtr/dense_3/BiasAddBiasAddexpand_xtr/dense_3/MatMul-mio_variable/expand_xtr/dense_3/bias/variable*
T0*
data_formatNHWC
J
expand_xtr/dense_3/SigmoidSigmoidexpand_xtr/dense_3/BiasAdd*
T0

+mio_variable/like_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*$
	containerlike_xtr/dense/kernel

+mio_variable/like_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*$
	containerlike_xtr/dense/kernel
W
"Initializer_8/random_uniform/shapeConst*
dtype0*
valueB"     
M
 Initializer_8/random_uniform/minConst*
valueB
 *b§―*
dtype0
M
 Initializer_8/random_uniform/maxConst*
valueB
 *b§=*
dtype0

*Initializer_8/random_uniform/RandomUniformRandomUniform"Initializer_8/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
t
 Initializer_8/random_uniform/subSub Initializer_8/random_uniform/max Initializer_8/random_uniform/min*
T0
~
 Initializer_8/random_uniform/mulMul*Initializer_8/random_uniform/RandomUniform Initializer_8/random_uniform/sub*
T0
p
Initializer_8/random_uniformAdd Initializer_8/random_uniform/mul Initializer_8/random_uniform/min*
T0
Ï
Assign_8Assign+mio_variable/like_xtr/dense/kernel/gradientInitializer_8/random_uniform*
use_locking(*
T0*>
_class4
20loc:@mio_variable/like_xtr/dense/kernel/gradient*
validate_shape(

)mio_variable/like_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*"
	containerlike_xtr/dense/bias*
shape:

)mio_variable/like_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*"
	containerlike_xtr/dense/bias
E
Initializer_9/zerosConst*
valueB*    *
dtype0
Â
Assign_9Assign)mio_variable/like_xtr/dense/bias/gradientInitializer_9/zeros*
use_locking(*
T0*<
_class2
0.loc:@mio_variable/like_xtr/dense/bias/gradient*
validate_shape(
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
!Initializer_10/random_uniform/minConst*
valueB
 *   ū*
dtype0
N
!Initializer_10/random_uniform/maxConst*
valueB
 *   >*
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
Õ
	Assign_10Assign-mio_variable/like_xtr/dense_1/kernel/gradientInitializer_10/random_uniform*
T0*@
_class6
42loc:@mio_variable/like_xtr/dense_1/kernel/gradient*
validate_shape(*
use_locking(

+mio_variable/like_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containerlike_xtr/dense_1/bias

+mio_variable/like_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containerlike_xtr/dense_1/bias
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
dtype0*
valueB"   @   
N
!Initializer_12/random_uniform/minConst*
valueB
 *ó5ū*
dtype0
N
!Initializer_12/random_uniform/maxConst*
valueB
 *ó5>*
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
+mio_variable/like_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerlike_xtr/dense_2/bias*
shape:@

+mio_variable/like_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerlike_xtr/dense_2/bias*
shape:@
E
Initializer_13/zerosConst*
dtype0*
valueB@*    
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
-mio_variable/like_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*&
	containerlike_xtr/dense_3/kernel
X
#Initializer_14/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_14/random_uniform/minConst*
valueB
 *ū*
dtype0
N
!Initializer_14/random_uniform/maxConst*
dtype0*
valueB
 *>
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
	Assign_14Assign-mio_variable/like_xtr/dense_3/kernel/gradientInitializer_14/random_uniform*
use_locking(*
T0*@
_class6
42loc:@mio_variable/like_xtr/dense_3/kernel/gradient*
validate_shape(
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
	Assign_15Assign+mio_variable/like_xtr/dense_3/bias/gradientInitializer_15/zeros*
use_locking(*
T0*>
_class4
20loc:@mio_variable/like_xtr/dense_3/bias/gradient*
validate_shape(

like_xtr/dense_3/MatMulMatMullike_xtr/dense_2/LeakyRelu-mio_variable/like_xtr/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 
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


,mio_variable/reply_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense/kernel*
shape:

X
#Initializer_16/random_uniform/shapeConst*
dtype0*
valueB"     
N
!Initializer_16/random_uniform/minConst*
dtype0*
valueB
 *b§―
N
!Initializer_16/random_uniform/maxConst*
valueB
 *b§=*
dtype0
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
Ó
	Assign_16Assign,mio_variable/reply_xtr/dense/kernel/gradientInitializer_16/random_uniform*
use_locking(*
T0*?
_class5
31loc:@mio_variable/reply_xtr/dense/kernel/gradient*
validate_shape(

*mio_variable/reply_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*#
	containerreply_xtr/dense/bias

*mio_variable/reply_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerreply_xtr/dense/bias*
shape:
F
Initializer_17/zerosConst*
valueB*    *
dtype0
Æ
	Assign_17Assign*mio_variable/reply_xtr/dense/bias/gradientInitializer_17/zeros*
use_locking(*
T0*=
_class3
1/loc:@mio_variable/reply_xtr/dense/bias/gradient*
validate_shape(

reply_xtr/dense/MatMulMatMulconcat,mio_variable/reply_xtr/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
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
Ē
.mio_variable/reply_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerreply_xtr/dense_1/kernel*
shape:

Ē
.mio_variable/reply_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerreply_xtr/dense_1/kernel*
shape:

X
#Initializer_18/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_18/random_uniform/minConst*
dtype0*
valueB
 *   ū
N
!Initializer_18/random_uniform/maxConst*
valueB
 *   >*
dtype0

+Initializer_18/random_uniform/RandomUniformRandomUniform#Initializer_18/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_18/random_uniform/subSub!Initializer_18/random_uniform/max!Initializer_18/random_uniform/min*
T0

!Initializer_18/random_uniform/mulMul+Initializer_18/random_uniform/RandomUniform!Initializer_18/random_uniform/sub*
T0
s
Initializer_18/random_uniformAdd!Initializer_18/random_uniform/mul!Initializer_18/random_uniform/min*
T0
Ũ
	Assign_18Assign.mio_variable/reply_xtr/dense_1/kernel/gradientInitializer_18/random_uniform*
T0*A
_class7
53loc:@mio_variable/reply_xtr/dense_1/kernel/gradient*
validate_shape(*
use_locking(

,mio_variable/reply_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense_1/bias*
shape:

,mio_variable/reply_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*%
	containerreply_xtr/dense_1/bias
F
Initializer_19/zerosConst*
dtype0*
valueB*    
Ę
	Assign_19Assign,mio_variable/reply_xtr/dense_1/bias/gradientInitializer_19/zeros*
T0*?
_class5
31loc:@mio_variable/reply_xtr/dense_1/bias/gradient*
validate_shape(*
use_locking(
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
.mio_variable/reply_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*'
	containerreply_xtr/dense_2/kernel
Ą
.mio_variable/reply_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*'
	containerreply_xtr/dense_2/kernel
X
#Initializer_20/random_uniform/shapeConst*
dtype0*
valueB"   @   
N
!Initializer_20/random_uniform/minConst*
dtype0*
valueB
 *ó5ū
N
!Initializer_20/random_uniform/maxConst*
valueB
 *ó5>*
dtype0

+Initializer_20/random_uniform/RandomUniformRandomUniform#Initializer_20/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_20/random_uniform/subSub!Initializer_20/random_uniform/max!Initializer_20/random_uniform/min*
T0

!Initializer_20/random_uniform/mulMul+Initializer_20/random_uniform/RandomUniform!Initializer_20/random_uniform/sub*
T0
s
Initializer_20/random_uniformAdd!Initializer_20/random_uniform/mul!Initializer_20/random_uniform/min*
T0
Ũ
	Assign_20Assign.mio_variable/reply_xtr/dense_2/kernel/gradientInitializer_20/random_uniform*
use_locking(*
T0*A
_class7
53loc:@mio_variable/reply_xtr/dense_2/kernel/gradient*
validate_shape(

,mio_variable/reply_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense_2/bias*
shape:@

,mio_variable/reply_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*%
	containerreply_xtr/dense_2/bias
E
Initializer_21/zerosConst*
dtype0*
valueB@*    
Ę
	Assign_21Assign,mio_variable/reply_xtr/dense_2/bias/gradientInitializer_21/zeros*
T0*?
_class5
31loc:@mio_variable/reply_xtr/dense_2/bias/gradient*
validate_shape(*
use_locking(

reply_xtr/dense_2/MatMulMatMulreply_xtr/dense_1/LeakyRelu.mio_variable/reply_xtr/dense_2/kernel/variable*
transpose_b( *
T0*
transpose_a( 
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
.mio_variable/reply_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*'
	containerreply_xtr/dense_3/kernel
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
!Initializer_22/random_uniform/minConst*
valueB
 *ū*
dtype0
N
!Initializer_22/random_uniform/maxConst*
dtype0*
valueB
 *>
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
Ũ
	Assign_22Assign.mio_variable/reply_xtr/dense_3/kernel/gradientInitializer_22/random_uniform*
use_locking(*
T0*A
_class7
53loc:@mio_variable/reply_xtr/dense_3/kernel/gradient*
validate_shape(

,mio_variable/reply_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense_3/bias*
shape:

,mio_variable/reply_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense_3/bias*
shape:
E
Initializer_23/zerosConst*
valueB*    *
dtype0
Ę
	Assign_23Assign,mio_variable/reply_xtr/dense_3/bias/gradientInitializer_23/zeros*
T0*?
_class5
31loc:@mio_variable/reply_xtr/dense_3/bias/gradient*
validate_shape(*
use_locking(

reply_xtr/dense_3/MatMulMatMulreply_xtr/dense_2/LeakyRelu.mio_variable/reply_xtr/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 

reply_xtr/dense_3/BiasAddBiasAddreply_xtr/dense_3/MatMul,mio_variable/reply_xtr/dense_3/bias/variable*
T0*
data_formatNHWC
H
reply_xtr/dense_3/SigmoidSigmoidreply_xtr/dense_3/BiasAdd*
T0

+mio_variable/copy_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense/kernel*
shape:


+mio_variable/copy_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense/kernel*
shape:

X
#Initializer_24/random_uniform/shapeConst*
valueB"     *
dtype0
N
!Initializer_24/random_uniform/minConst*
dtype0*
valueB
 *b§―
N
!Initializer_24/random_uniform/maxConst*
valueB
 *b§=*
dtype0

+Initializer_24/random_uniform/RandomUniformRandomUniform#Initializer_24/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_24/random_uniform/subSub!Initializer_24/random_uniform/max!Initializer_24/random_uniform/min*
T0

!Initializer_24/random_uniform/mulMul+Initializer_24/random_uniform/RandomUniform!Initializer_24/random_uniform/sub*
T0
s
Initializer_24/random_uniformAdd!Initializer_24/random_uniform/mul!Initializer_24/random_uniform/min*
T0
Ņ
	Assign_24Assign+mio_variable/copy_xtr/dense/kernel/gradientInitializer_24/random_uniform*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@mio_variable/copy_xtr/dense/kernel/gradient

)mio_variable/copy_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*"
	containercopy_xtr/dense/bias*
shape:

)mio_variable/copy_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*"
	containercopy_xtr/dense/bias
F
Initializer_25/zerosConst*
dtype0*
valueB*    
Ä
	Assign_25Assign)mio_variable/copy_xtr/dense/bias/gradientInitializer_25/zeros*
use_locking(*
T0*<
_class2
0.loc:@mio_variable/copy_xtr/dense/bias/gradient*
validate_shape(

copy_xtr/dense/MatMulMatMulconcat+mio_variable/copy_xtr/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 

copy_xtr/dense/BiasAddBiasAddcopy_xtr/dense/MatMul)mio_variable/copy_xtr/dense/bias/variable*
T0*
data_formatNHWC
K
copy_xtr/dense/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0
d
copy_xtr/dense/LeakyRelu/mulMulcopy_xtr/dense/LeakyRelu/alphacopy_xtr/dense/BiasAdd*
T0
b
copy_xtr/dense/LeakyReluMaximumcopy_xtr/dense/LeakyRelu/mulcopy_xtr/dense/BiasAdd*
T0
 
-mio_variable/copy_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*&
	containercopy_xtr/dense_1/kernel
 
-mio_variable/copy_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*&
	containercopy_xtr/dense_1/kernel
X
#Initializer_26/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_26/random_uniform/minConst*
dtype0*
valueB
 *   ū
N
!Initializer_26/random_uniform/maxConst*
valueB
 *   >*
dtype0

+Initializer_26/random_uniform/RandomUniformRandomUniform#Initializer_26/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_26/random_uniform/subSub!Initializer_26/random_uniform/max!Initializer_26/random_uniform/min*
T0

!Initializer_26/random_uniform/mulMul+Initializer_26/random_uniform/RandomUniform!Initializer_26/random_uniform/sub*
T0
s
Initializer_26/random_uniformAdd!Initializer_26/random_uniform/mul!Initializer_26/random_uniform/min*
T0
Õ
	Assign_26Assign-mio_variable/copy_xtr/dense_1/kernel/gradientInitializer_26/random_uniform*
use_locking(*
T0*@
_class6
42loc:@mio_variable/copy_xtr/dense_1/kernel/gradient*
validate_shape(

+mio_variable/copy_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense_1/bias*
shape:

+mio_variable/copy_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense_1/bias*
shape:
F
Initializer_27/zerosConst*
dtype0*
valueB*    
Č
	Assign_27Assign+mio_variable/copy_xtr/dense_1/bias/gradientInitializer_27/zeros*
T0*>
_class4
20loc:@mio_variable/copy_xtr/dense_1/bias/gradient*
validate_shape(*
use_locking(

copy_xtr/dense_1/MatMulMatMulcopy_xtr/dense/LeakyRelu-mio_variable/copy_xtr/dense_1/kernel/variable*
transpose_b( *
T0*
transpose_a( 

copy_xtr/dense_1/BiasAddBiasAddcopy_xtr/dense_1/MatMul+mio_variable/copy_xtr/dense_1/bias/variable*
T0*
data_formatNHWC
M
 copy_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0
j
copy_xtr/dense_1/LeakyRelu/mulMul copy_xtr/dense_1/LeakyRelu/alphacopy_xtr/dense_1/BiasAdd*
T0
h
copy_xtr/dense_1/LeakyReluMaximumcopy_xtr/dense_1/LeakyRelu/mulcopy_xtr/dense_1/BiasAdd*
T0

-mio_variable/copy_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*&
	containercopy_xtr/dense_2/kernel

-mio_variable/copy_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*&
	containercopy_xtr/dense_2/kernel
X
#Initializer_28/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_28/random_uniform/minConst*
valueB
 *ó5ū*
dtype0
N
!Initializer_28/random_uniform/maxConst*
valueB
 *ó5>*
dtype0

+Initializer_28/random_uniform/RandomUniformRandomUniform#Initializer_28/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_28/random_uniform/subSub!Initializer_28/random_uniform/max!Initializer_28/random_uniform/min*
T0

!Initializer_28/random_uniform/mulMul+Initializer_28/random_uniform/RandomUniform!Initializer_28/random_uniform/sub*
T0
s
Initializer_28/random_uniformAdd!Initializer_28/random_uniform/mul!Initializer_28/random_uniform/min*
T0
Õ
	Assign_28Assign-mio_variable/copy_xtr/dense_2/kernel/gradientInitializer_28/random_uniform*
use_locking(*
T0*@
_class6
42loc:@mio_variable/copy_xtr/dense_2/kernel/gradient*
validate_shape(

+mio_variable/copy_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*$
	containercopy_xtr/dense_2/bias

+mio_variable/copy_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense_2/bias*
shape:@
E
Initializer_29/zerosConst*
valueB@*    *
dtype0
Č
	Assign_29Assign+mio_variable/copy_xtr/dense_2/bias/gradientInitializer_29/zeros*
use_locking(*
T0*>
_class4
20loc:@mio_variable/copy_xtr/dense_2/bias/gradient*
validate_shape(

copy_xtr/dense_2/MatMulMatMulcopy_xtr/dense_1/LeakyRelu-mio_variable/copy_xtr/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 

copy_xtr/dense_2/BiasAddBiasAddcopy_xtr/dense_2/MatMul+mio_variable/copy_xtr/dense_2/bias/variable*
data_formatNHWC*
T0
M
 copy_xtr/dense_2/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍĖL>
j
copy_xtr/dense_2/LeakyRelu/mulMul copy_xtr/dense_2/LeakyRelu/alphacopy_xtr/dense_2/BiasAdd*
T0
h
copy_xtr/dense_2/LeakyReluMaximumcopy_xtr/dense_2/LeakyRelu/mulcopy_xtr/dense_2/BiasAdd*
T0

-mio_variable/copy_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containercopy_xtr/dense_3/kernel*
shape
:@

-mio_variable/copy_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containercopy_xtr/dense_3/kernel*
shape
:@
X
#Initializer_30/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_30/random_uniform/minConst*
valueB
 *ū*
dtype0
N
!Initializer_30/random_uniform/maxConst*
valueB
 *>*
dtype0

+Initializer_30/random_uniform/RandomUniformRandomUniform#Initializer_30/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_30/random_uniform/subSub!Initializer_30/random_uniform/max!Initializer_30/random_uniform/min*
T0

!Initializer_30/random_uniform/mulMul+Initializer_30/random_uniform/RandomUniform!Initializer_30/random_uniform/sub*
T0
s
Initializer_30/random_uniformAdd!Initializer_30/random_uniform/mul!Initializer_30/random_uniform/min*
T0
Õ
	Assign_30Assign-mio_variable/copy_xtr/dense_3/kernel/gradientInitializer_30/random_uniform*
use_locking(*
T0*@
_class6
42loc:@mio_variable/copy_xtr/dense_3/kernel/gradient*
validate_shape(

+mio_variable/copy_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense_3/bias*
shape:

+mio_variable/copy_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense_3/bias*
shape:
E
Initializer_31/zerosConst*
valueB*    *
dtype0
Č
	Assign_31Assign+mio_variable/copy_xtr/dense_3/bias/gradientInitializer_31/zeros*
T0*>
_class4
20loc:@mio_variable/copy_xtr/dense_3/bias/gradient*
validate_shape(*
use_locking(

copy_xtr/dense_3/MatMulMatMulcopy_xtr/dense_2/LeakyRelu-mio_variable/copy_xtr/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 

copy_xtr/dense_3/BiasAddBiasAddcopy_xtr/dense_3/MatMul+mio_variable/copy_xtr/dense_3/bias/variable*
T0*
data_formatNHWC
F
copy_xtr/dense_3/SigmoidSigmoidcopy_xtr/dense_3/BiasAdd*
T0

,mio_variable/share_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containershare_xtr/dense/kernel*
shape:


,mio_variable/share_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containershare_xtr/dense/kernel*
shape:

X
#Initializer_32/random_uniform/shapeConst*
valueB"     *
dtype0
N
!Initializer_32/random_uniform/minConst*
valueB
 *b§―*
dtype0
N
!Initializer_32/random_uniform/maxConst*
valueB
 *b§=*
dtype0

+Initializer_32/random_uniform/RandomUniformRandomUniform#Initializer_32/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_32/random_uniform/subSub!Initializer_32/random_uniform/max!Initializer_32/random_uniform/min*
T0

!Initializer_32/random_uniform/mulMul+Initializer_32/random_uniform/RandomUniform!Initializer_32/random_uniform/sub*
T0
s
Initializer_32/random_uniformAdd!Initializer_32/random_uniform/mul!Initializer_32/random_uniform/min*
T0
Ó
	Assign_32Assign,mio_variable/share_xtr/dense/kernel/gradientInitializer_32/random_uniform*
use_locking(*
T0*?
_class5
31loc:@mio_variable/share_xtr/dense/kernel/gradient*
validate_shape(

*mio_variable/share_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*#
	containershare_xtr/dense/bias*
shape:

*mio_variable/share_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*#
	containershare_xtr/dense/bias
F
Initializer_33/zerosConst*
valueB*    *
dtype0
Æ
	Assign_33Assign*mio_variable/share_xtr/dense/bias/gradientInitializer_33/zeros*
use_locking(*
T0*=
_class3
1/loc:@mio_variable/share_xtr/dense/bias/gradient*
validate_shape(

share_xtr/dense/MatMulMatMulconcat,mio_variable/share_xtr/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 

share_xtr/dense/BiasAddBiasAddshare_xtr/dense/MatMul*mio_variable/share_xtr/dense/bias/variable*
data_formatNHWC*
T0
L
share_xtr/dense/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0
g
share_xtr/dense/LeakyRelu/mulMulshare_xtr/dense/LeakyRelu/alphashare_xtr/dense/BiasAdd*
T0
e
share_xtr/dense/LeakyReluMaximumshare_xtr/dense/LeakyRelu/mulshare_xtr/dense/BiasAdd*
T0
Ē
.mio_variable/share_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*'
	containershare_xtr/dense_1/kernel
Ē
.mio_variable/share_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containershare_xtr/dense_1/kernel*
shape:

X
#Initializer_34/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_34/random_uniform/minConst*
valueB
 *   ū*
dtype0
N
!Initializer_34/random_uniform/maxConst*
valueB
 *   >*
dtype0

+Initializer_34/random_uniform/RandomUniformRandomUniform#Initializer_34/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_34/random_uniform/subSub!Initializer_34/random_uniform/max!Initializer_34/random_uniform/min*
T0

!Initializer_34/random_uniform/mulMul+Initializer_34/random_uniform/RandomUniform!Initializer_34/random_uniform/sub*
T0
s
Initializer_34/random_uniformAdd!Initializer_34/random_uniform/mul!Initializer_34/random_uniform/min*
T0
Ũ
	Assign_34Assign.mio_variable/share_xtr/dense_1/kernel/gradientInitializer_34/random_uniform*
use_locking(*
T0*A
_class7
53loc:@mio_variable/share_xtr/dense_1/kernel/gradient*
validate_shape(

,mio_variable/share_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containershare_xtr/dense_1/bias*
shape:

,mio_variable/share_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containershare_xtr/dense_1/bias*
shape:
F
Initializer_35/zerosConst*
valueB*    *
dtype0
Ę
	Assign_35Assign,mio_variable/share_xtr/dense_1/bias/gradientInitializer_35/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/share_xtr/dense_1/bias/gradient*
validate_shape(

share_xtr/dense_1/MatMulMatMulshare_xtr/dense/LeakyRelu.mio_variable/share_xtr/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 

share_xtr/dense_1/BiasAddBiasAddshare_xtr/dense_1/MatMul,mio_variable/share_xtr/dense_1/bias/variable*
T0*
data_formatNHWC
N
!share_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0
m
share_xtr/dense_1/LeakyRelu/mulMul!share_xtr/dense_1/LeakyRelu/alphashare_xtr/dense_1/BiasAdd*
T0
k
share_xtr/dense_1/LeakyReluMaximumshare_xtr/dense_1/LeakyRelu/mulshare_xtr/dense_1/BiasAdd*
T0
Ą
.mio_variable/share_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containershare_xtr/dense_2/kernel*
shape:	@
Ą
.mio_variable/share_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containershare_xtr/dense_2/kernel*
shape:	@
X
#Initializer_36/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_36/random_uniform/minConst*
valueB
 *ó5ū*
dtype0
N
!Initializer_36/random_uniform/maxConst*
valueB
 *ó5>*
dtype0

+Initializer_36/random_uniform/RandomUniformRandomUniform#Initializer_36/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_36/random_uniform/subSub!Initializer_36/random_uniform/max!Initializer_36/random_uniform/min*
T0

!Initializer_36/random_uniform/mulMul+Initializer_36/random_uniform/RandomUniform!Initializer_36/random_uniform/sub*
T0
s
Initializer_36/random_uniformAdd!Initializer_36/random_uniform/mul!Initializer_36/random_uniform/min*
T0
Ũ
	Assign_36Assign.mio_variable/share_xtr/dense_2/kernel/gradientInitializer_36/random_uniform*
validate_shape(*
use_locking(*
T0*A
_class7
53loc:@mio_variable/share_xtr/dense_2/kernel/gradient

,mio_variable/share_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containershare_xtr/dense_2/bias*
shape:@

,mio_variable/share_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containershare_xtr/dense_2/bias*
shape:@
E
Initializer_37/zerosConst*
valueB@*    *
dtype0
Ę
	Assign_37Assign,mio_variable/share_xtr/dense_2/bias/gradientInitializer_37/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/share_xtr/dense_2/bias/gradient*
validate_shape(

share_xtr/dense_2/MatMulMatMulshare_xtr/dense_1/LeakyRelu.mio_variable/share_xtr/dense_2/kernel/variable*
transpose_b( *
T0*
transpose_a( 

share_xtr/dense_2/BiasAddBiasAddshare_xtr/dense_2/MatMul,mio_variable/share_xtr/dense_2/bias/variable*
data_formatNHWC*
T0
N
!share_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0
m
share_xtr/dense_2/LeakyRelu/mulMul!share_xtr/dense_2/LeakyRelu/alphashare_xtr/dense_2/BiasAdd*
T0
k
share_xtr/dense_2/LeakyReluMaximumshare_xtr/dense_2/LeakyRelu/mulshare_xtr/dense_2/BiasAdd*
T0
 
.mio_variable/share_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containershare_xtr/dense_3/kernel*
shape
:@
 
.mio_variable/share_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*'
	containershare_xtr/dense_3/kernel
X
#Initializer_38/random_uniform/shapeConst*
dtype0*
valueB"@      
N
!Initializer_38/random_uniform/minConst*
valueB
 *ū*
dtype0
N
!Initializer_38/random_uniform/maxConst*
valueB
 *>*
dtype0

+Initializer_38/random_uniform/RandomUniformRandomUniform#Initializer_38/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_38/random_uniform/subSub!Initializer_38/random_uniform/max!Initializer_38/random_uniform/min*
T0

!Initializer_38/random_uniform/mulMul+Initializer_38/random_uniform/RandomUniform!Initializer_38/random_uniform/sub*
T0
s
Initializer_38/random_uniformAdd!Initializer_38/random_uniform/mul!Initializer_38/random_uniform/min*
T0
Ũ
	Assign_38Assign.mio_variable/share_xtr/dense_3/kernel/gradientInitializer_38/random_uniform*
use_locking(*
T0*A
_class7
53loc:@mio_variable/share_xtr/dense_3/kernel/gradient*
validate_shape(

,mio_variable/share_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containershare_xtr/dense_3/bias*
shape:

,mio_variable/share_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containershare_xtr/dense_3/bias*
shape:
E
Initializer_39/zerosConst*
valueB*    *
dtype0
Ę
	Assign_39Assign,mio_variable/share_xtr/dense_3/bias/gradientInitializer_39/zeros*
validate_shape(*
use_locking(*
T0*?
_class5
31loc:@mio_variable/share_xtr/dense_3/bias/gradient

share_xtr/dense_3/MatMulMatMulshare_xtr/dense_2/LeakyRelu.mio_variable/share_xtr/dense_3/kernel/variable*
transpose_a( *
transpose_b( *
T0

share_xtr/dense_3/BiasAddBiasAddshare_xtr/dense_3/MatMul,mio_variable/share_xtr/dense_3/bias/variable*
data_formatNHWC*
T0
H
share_xtr/dense_3/SigmoidSigmoidshare_xtr/dense_3/BiasAdd*
T0
Ī
/mio_variable/audience_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeraudience_xtr/dense/kernel*
shape:

Ī
/mio_variable/audience_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeraudience_xtr/dense/kernel*
shape:

X
#Initializer_40/random_uniform/shapeConst*
valueB"     *
dtype0
N
!Initializer_40/random_uniform/minConst*
valueB
 *b§―*
dtype0
N
!Initializer_40/random_uniform/maxConst*
dtype0*
valueB
 *b§=

+Initializer_40/random_uniform/RandomUniformRandomUniform#Initializer_40/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_40/random_uniform/subSub!Initializer_40/random_uniform/max!Initializer_40/random_uniform/min*
T0

!Initializer_40/random_uniform/mulMul+Initializer_40/random_uniform/RandomUniform!Initializer_40/random_uniform/sub*
T0
s
Initializer_40/random_uniformAdd!Initializer_40/random_uniform/mul!Initializer_40/random_uniform/min*
T0
Ų
	Assign_40Assign/mio_variable/audience_xtr/dense/kernel/gradientInitializer_40/random_uniform*
use_locking(*
T0*B
_class8
64loc:@mio_variable/audience_xtr/dense/kernel/gradient*
validate_shape(

-mio_variable/audience_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*&
	containeraudience_xtr/dense/bias

-mio_variable/audience_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*&
	containeraudience_xtr/dense/bias
F
Initializer_41/zerosConst*
valueB*    *
dtype0
Ė
	Assign_41Assign-mio_variable/audience_xtr/dense/bias/gradientInitializer_41/zeros*
use_locking(*
T0*@
_class6
42loc:@mio_variable/audience_xtr/dense/bias/gradient*
validate_shape(

audience_xtr/dense/MatMulMatMulconcat/mio_variable/audience_xtr/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 

audience_xtr/dense/BiasAddBiasAddaudience_xtr/dense/MatMul-mio_variable/audience_xtr/dense/bias/variable*
T0*
data_formatNHWC
O
"audience_xtr/dense/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0
p
 audience_xtr/dense/LeakyRelu/mulMul"audience_xtr/dense/LeakyRelu/alphaaudience_xtr/dense/BiasAdd*
T0
n
audience_xtr/dense/LeakyReluMaximum audience_xtr/dense/LeakyRelu/mulaudience_xtr/dense/BiasAdd*
T0
Ļ
1mio_variable/audience_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS**
	containeraudience_xtr/dense_1/kernel*
shape:

Ļ
1mio_variable/audience_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
**
	containeraudience_xtr/dense_1/kernel
X
#Initializer_42/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_42/random_uniform/minConst*
valueB
 *   ū*
dtype0
N
!Initializer_42/random_uniform/maxConst*
valueB
 *   >*
dtype0

+Initializer_42/random_uniform/RandomUniformRandomUniform#Initializer_42/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_42/random_uniform/subSub!Initializer_42/random_uniform/max!Initializer_42/random_uniform/min*
T0

!Initializer_42/random_uniform/mulMul+Initializer_42/random_uniform/RandomUniform!Initializer_42/random_uniform/sub*
T0
s
Initializer_42/random_uniformAdd!Initializer_42/random_uniform/mul!Initializer_42/random_uniform/min*
T0
Ý
	Assign_42Assign1mio_variable/audience_xtr/dense_1/kernel/gradientInitializer_42/random_uniform*
T0*D
_class:
86loc:@mio_variable/audience_xtr/dense_1/kernel/gradient*
validate_shape(*
use_locking(

/mio_variable/audience_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeraudience_xtr/dense_1/bias*
shape:

/mio_variable/audience_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeraudience_xtr/dense_1/bias*
shape:
F
Initializer_43/zerosConst*
dtype0*
valueB*    
Ð
	Assign_43Assign/mio_variable/audience_xtr/dense_1/bias/gradientInitializer_43/zeros*
use_locking(*
T0*B
_class8
64loc:@mio_variable/audience_xtr/dense_1/bias/gradient*
validate_shape(
Ĩ
audience_xtr/dense_1/MatMulMatMulaudience_xtr/dense/LeakyRelu1mio_variable/audience_xtr/dense_1/kernel/variable*
transpose_b( *
T0*
transpose_a( 

audience_xtr/dense_1/BiasAddBiasAddaudience_xtr/dense_1/MatMul/mio_variable/audience_xtr/dense_1/bias/variable*
data_formatNHWC*
T0
Q
$audience_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0
v
"audience_xtr/dense_1/LeakyRelu/mulMul$audience_xtr/dense_1/LeakyRelu/alphaaudience_xtr/dense_1/BiasAdd*
T0
t
audience_xtr/dense_1/LeakyReluMaximum"audience_xtr/dense_1/LeakyRelu/mulaudience_xtr/dense_1/BiasAdd*
T0
§
1mio_variable/audience_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@**
	containeraudience_xtr/dense_2/kernel
§
1mio_variable/audience_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS**
	containeraudience_xtr/dense_2/kernel*
shape:	@
X
#Initializer_44/random_uniform/shapeConst*
dtype0*
valueB"   @   
N
!Initializer_44/random_uniform/minConst*
valueB
 *ó5ū*
dtype0
N
!Initializer_44/random_uniform/maxConst*
valueB
 *ó5>*
dtype0

+Initializer_44/random_uniform/RandomUniformRandomUniform#Initializer_44/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_44/random_uniform/subSub!Initializer_44/random_uniform/max!Initializer_44/random_uniform/min*
T0

!Initializer_44/random_uniform/mulMul+Initializer_44/random_uniform/RandomUniform!Initializer_44/random_uniform/sub*
T0
s
Initializer_44/random_uniformAdd!Initializer_44/random_uniform/mul!Initializer_44/random_uniform/min*
T0
Ý
	Assign_44Assign1mio_variable/audience_xtr/dense_2/kernel/gradientInitializer_44/random_uniform*
T0*D
_class:
86loc:@mio_variable/audience_xtr/dense_2/kernel/gradient*
validate_shape(*
use_locking(

/mio_variable/audience_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeraudience_xtr/dense_2/bias*
shape:@

/mio_variable/audience_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*(
	containeraudience_xtr/dense_2/bias
E
Initializer_45/zerosConst*
valueB@*    *
dtype0
Ð
	Assign_45Assign/mio_variable/audience_xtr/dense_2/bias/gradientInitializer_45/zeros*
use_locking(*
T0*B
_class8
64loc:@mio_variable/audience_xtr/dense_2/bias/gradient*
validate_shape(
§
audience_xtr/dense_2/MatMulMatMulaudience_xtr/dense_1/LeakyRelu1mio_variable/audience_xtr/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 

audience_xtr/dense_2/BiasAddBiasAddaudience_xtr/dense_2/MatMul/mio_variable/audience_xtr/dense_2/bias/variable*
T0*
data_formatNHWC
Q
$audience_xtr/dense_2/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍĖL>
v
"audience_xtr/dense_2/LeakyRelu/mulMul$audience_xtr/dense_2/LeakyRelu/alphaaudience_xtr/dense_2/BiasAdd*
T0
t
audience_xtr/dense_2/LeakyReluMaximum"audience_xtr/dense_2/LeakyRelu/mulaudience_xtr/dense_2/BiasAdd*
T0
Ķ
1mio_variable/audience_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS**
	containeraudience_xtr/dense_3/kernel*
shape
:@
Ķ
1mio_variable/audience_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS**
	containeraudience_xtr/dense_3/kernel*
shape
:@
X
#Initializer_46/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_46/random_uniform/minConst*
valueB
 *ū*
dtype0
N
!Initializer_46/random_uniform/maxConst*
valueB
 *>*
dtype0

+Initializer_46/random_uniform/RandomUniformRandomUniform#Initializer_46/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_46/random_uniform/subSub!Initializer_46/random_uniform/max!Initializer_46/random_uniform/min*
T0

!Initializer_46/random_uniform/mulMul+Initializer_46/random_uniform/RandomUniform!Initializer_46/random_uniform/sub*
T0
s
Initializer_46/random_uniformAdd!Initializer_46/random_uniform/mul!Initializer_46/random_uniform/min*
T0
Ý
	Assign_46Assign1mio_variable/audience_xtr/dense_3/kernel/gradientInitializer_46/random_uniform*
use_locking(*
T0*D
_class:
86loc:@mio_variable/audience_xtr/dense_3/kernel/gradient*
validate_shape(

/mio_variable/audience_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*(
	containeraudience_xtr/dense_3/bias

/mio_variable/audience_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeraudience_xtr/dense_3/bias*
shape:
E
Initializer_47/zerosConst*
valueB*    *
dtype0
Ð
	Assign_47Assign/mio_variable/audience_xtr/dense_3/bias/gradientInitializer_47/zeros*
use_locking(*
T0*B
_class8
64loc:@mio_variable/audience_xtr/dense_3/bias/gradient*
validate_shape(
§
audience_xtr/dense_3/MatMulMatMulaudience_xtr/dense_2/LeakyRelu1mio_variable/audience_xtr/dense_3/kernel/variable*
transpose_a( *
transpose_b( *
T0

audience_xtr/dense_3/BiasAddBiasAddaudience_xtr/dense_3/MatMul/mio_variable/audience_xtr/dense_3/bias/variable*
T0*
data_formatNHWC
N
audience_xtr/dense_3/SigmoidSigmoidaudience_xtr/dense_3/BiasAdd*
T0
ķ
8mio_variable/continuous_expand_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"continuous_expand_xtr/dense/kernel*
shape:

ķ
8mio_variable/continuous_expand_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"continuous_expand_xtr/dense/kernel*
shape:

X
#Initializer_48/random_uniform/shapeConst*
valueB"     *
dtype0
N
!Initializer_48/random_uniform/minConst*
valueB
 *b§―*
dtype0
N
!Initializer_48/random_uniform/maxConst*
valueB
 *b§=*
dtype0

+Initializer_48/random_uniform/RandomUniformRandomUniform#Initializer_48/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_48/random_uniform/subSub!Initializer_48/random_uniform/max!Initializer_48/random_uniform/min*
T0

!Initializer_48/random_uniform/mulMul+Initializer_48/random_uniform/RandomUniform!Initializer_48/random_uniform/sub*
T0
s
Initializer_48/random_uniformAdd!Initializer_48/random_uniform/mul!Initializer_48/random_uniform/min*
T0
ë
	Assign_48Assign8mio_variable/continuous_expand_xtr/dense/kernel/gradientInitializer_48/random_uniform*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/continuous_expand_xtr/dense/kernel/gradient*
validate_shape(
­
6mio_variable/continuous_expand_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" continuous_expand_xtr/dense/bias*
shape:
­
6mio_variable/continuous_expand_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" continuous_expand_xtr/dense/bias*
shape:
F
Initializer_49/zerosConst*
valueB*    *
dtype0
Þ
	Assign_49Assign6mio_variable/continuous_expand_xtr/dense/bias/gradientInitializer_49/zeros*
use_locking(*
T0*I
_class?
=;loc:@mio_variable/continuous_expand_xtr/dense/bias/gradient*
validate_shape(

"continuous_expand_xtr/dense/MatMulMatMulconcat8mio_variable/continuous_expand_xtr/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Š
#continuous_expand_xtr/dense/BiasAddBiasAdd"continuous_expand_xtr/dense/MatMul6mio_variable/continuous_expand_xtr/dense/bias/variable*
T0*
data_formatNHWC
X
+continuous_expand_xtr/dense/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0

)continuous_expand_xtr/dense/LeakyRelu/mulMul+continuous_expand_xtr/dense/LeakyRelu/alpha#continuous_expand_xtr/dense/BiasAdd*
T0

%continuous_expand_xtr/dense/LeakyReluMaximum)continuous_expand_xtr/dense/LeakyRelu/mul#continuous_expand_xtr/dense/BiasAdd*
T0
š
:mio_variable/continuous_expand_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*3
	container&$continuous_expand_xtr/dense_1/kernel
š
:mio_variable/continuous_expand_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$continuous_expand_xtr/dense_1/kernel*
shape:

X
#Initializer_50/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_50/random_uniform/minConst*
valueB
 *   ū*
dtype0
N
!Initializer_50/random_uniform/maxConst*
dtype0*
valueB
 *   >

+Initializer_50/random_uniform/RandomUniformRandomUniform#Initializer_50/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_50/random_uniform/subSub!Initializer_50/random_uniform/max!Initializer_50/random_uniform/min*
T0

!Initializer_50/random_uniform/mulMul+Initializer_50/random_uniform/RandomUniform!Initializer_50/random_uniform/sub*
T0
s
Initializer_50/random_uniformAdd!Initializer_50/random_uniform/mul!Initializer_50/random_uniform/min*
T0
ï
	Assign_50Assign:mio_variable/continuous_expand_xtr/dense_1/kernel/gradientInitializer_50/random_uniform*
use_locking(*
T0*M
_classC
A?loc:@mio_variable/continuous_expand_xtr/dense_1/kernel/gradient*
validate_shape(
ą
8mio_variable/continuous_expand_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*1
	container$"continuous_expand_xtr/dense_1/bias
ą
8mio_variable/continuous_expand_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*1
	container$"continuous_expand_xtr/dense_1/bias
F
Initializer_51/zerosConst*
valueB*    *
dtype0
â
	Assign_51Assign8mio_variable/continuous_expand_xtr/dense_1/bias/gradientInitializer_51/zeros*
T0*K
_classA
?=loc:@mio_variable/continuous_expand_xtr/dense_1/bias/gradient*
validate_shape(*
use_locking(
Ā
$continuous_expand_xtr/dense_1/MatMulMatMul%continuous_expand_xtr/dense/LeakyRelu:mio_variable/continuous_expand_xtr/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
°
%continuous_expand_xtr/dense_1/BiasAddBiasAdd$continuous_expand_xtr/dense_1/MatMul8mio_variable/continuous_expand_xtr/dense_1/bias/variable*
T0*
data_formatNHWC
Z
-continuous_expand_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0

+continuous_expand_xtr/dense_1/LeakyRelu/mulMul-continuous_expand_xtr/dense_1/LeakyRelu/alpha%continuous_expand_xtr/dense_1/BiasAdd*
T0

'continuous_expand_xtr/dense_1/LeakyReluMaximum+continuous_expand_xtr/dense_1/LeakyRelu/mul%continuous_expand_xtr/dense_1/BiasAdd*
T0
đ
:mio_variable/continuous_expand_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*3
	container&$continuous_expand_xtr/dense_2/kernel
đ
:mio_variable/continuous_expand_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*3
	container&$continuous_expand_xtr/dense_2/kernel
X
#Initializer_52/random_uniform/shapeConst*
dtype0*
valueB"   @   
N
!Initializer_52/random_uniform/minConst*
valueB
 *ó5ū*
dtype0
N
!Initializer_52/random_uniform/maxConst*
dtype0*
valueB
 *ó5>

+Initializer_52/random_uniform/RandomUniformRandomUniform#Initializer_52/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_52/random_uniform/subSub!Initializer_52/random_uniform/max!Initializer_52/random_uniform/min*
T0

!Initializer_52/random_uniform/mulMul+Initializer_52/random_uniform/RandomUniform!Initializer_52/random_uniform/sub*
T0
s
Initializer_52/random_uniformAdd!Initializer_52/random_uniform/mul!Initializer_52/random_uniform/min*
T0
ï
	Assign_52Assign:mio_variable/continuous_expand_xtr/dense_2/kernel/gradientInitializer_52/random_uniform*
use_locking(*
T0*M
_classC
A?loc:@mio_variable/continuous_expand_xtr/dense_2/kernel/gradient*
validate_shape(
°
8mio_variable/continuous_expand_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"continuous_expand_xtr/dense_2/bias*
shape:@
°
8mio_variable/continuous_expand_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*1
	container$"continuous_expand_xtr/dense_2/bias
E
Initializer_53/zerosConst*
valueB@*    *
dtype0
â
	Assign_53Assign8mio_variable/continuous_expand_xtr/dense_2/bias/gradientInitializer_53/zeros*
T0*K
_classA
?=loc:@mio_variable/continuous_expand_xtr/dense_2/bias/gradient*
validate_shape(*
use_locking(
Â
$continuous_expand_xtr/dense_2/MatMulMatMul'continuous_expand_xtr/dense_1/LeakyRelu:mio_variable/continuous_expand_xtr/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 
°
%continuous_expand_xtr/dense_2/BiasAddBiasAdd$continuous_expand_xtr/dense_2/MatMul8mio_variable/continuous_expand_xtr/dense_2/bias/variable*
T0*
data_formatNHWC
Z
-continuous_expand_xtr/dense_2/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍĖL>

+continuous_expand_xtr/dense_2/LeakyRelu/mulMul-continuous_expand_xtr/dense_2/LeakyRelu/alpha%continuous_expand_xtr/dense_2/BiasAdd*
T0

'continuous_expand_xtr/dense_2/LeakyReluMaximum+continuous_expand_xtr/dense_2/LeakyRelu/mul%continuous_expand_xtr/dense_2/BiasAdd*
T0
ļ
:mio_variable/continuous_expand_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*3
	container&$continuous_expand_xtr/dense_3/kernel
ļ
:mio_variable/continuous_expand_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$continuous_expand_xtr/dense_3/kernel*
shape
:@
X
#Initializer_54/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_54/random_uniform/minConst*
valueB
 *ū*
dtype0
N
!Initializer_54/random_uniform/maxConst*
valueB
 *>*
dtype0

+Initializer_54/random_uniform/RandomUniformRandomUniform#Initializer_54/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_54/random_uniform/subSub!Initializer_54/random_uniform/max!Initializer_54/random_uniform/min*
T0

!Initializer_54/random_uniform/mulMul+Initializer_54/random_uniform/RandomUniform!Initializer_54/random_uniform/sub*
T0
s
Initializer_54/random_uniformAdd!Initializer_54/random_uniform/mul!Initializer_54/random_uniform/min*
T0
ï
	Assign_54Assign:mio_variable/continuous_expand_xtr/dense_3/kernel/gradientInitializer_54/random_uniform*
use_locking(*
T0*M
_classC
A?loc:@mio_variable/continuous_expand_xtr/dense_3/kernel/gradient*
validate_shape(
°
8mio_variable/continuous_expand_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"continuous_expand_xtr/dense_3/bias*
shape:
°
8mio_variable/continuous_expand_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"continuous_expand_xtr/dense_3/bias*
shape:
E
Initializer_55/zerosConst*
valueB*    *
dtype0
â
	Assign_55Assign8mio_variable/continuous_expand_xtr/dense_3/bias/gradientInitializer_55/zeros*
validate_shape(*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/continuous_expand_xtr/dense_3/bias/gradient
Â
$continuous_expand_xtr/dense_3/MatMulMatMul'continuous_expand_xtr/dense_2/LeakyRelu:mio_variable/continuous_expand_xtr/dense_3/kernel/variable*
transpose_b( *
T0*
transpose_a( 
°
%continuous_expand_xtr/dense_3/BiasAddBiasAdd$continuous_expand_xtr/dense_3/MatMul8mio_variable/continuous_expand_xtr/dense_3/bias/variable*
T0*
data_formatNHWC
`
%continuous_expand_xtr/dense_3/SigmoidSigmoid%continuous_expand_xtr/dense_3/BiasAdd*
T0"