
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
MIO_TABLE_ADDRESSConst"/device:CPU:0*
dtype0*
value
B  
¥
2mio_compress_indices/COMPRESS_INDEX__USER/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerCOMPRESS_INDEX__USER*
shape:ÿÿÿÿÿÿÿÿÿ
¥
2mio_compress_indices/COMPRESS_INDEX__USER/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerCOMPRESS_INDEX__USER*
shape:ÿÿÿÿÿÿÿÿÿ
h
CastCast2mio_compress_indices/COMPRESS_INDEX__USER/variable*

SrcT0*
Truncate( *

DstT0

&mio_embeddings/user_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containeruser_embedding*
shape:ÿÿÿÿÿÿÿÿÿ

&mio_embeddings/user_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containeruser_embedding*
shape:ÿÿÿÿÿÿÿÿÿ

%mio_embeddings/pid_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerpid_embedding*
shape:ÿÿÿÿÿÿÿÿÿ@

%mio_embeddings/pid_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerpid_embedding*
shape:ÿÿÿÿÿÿÿÿÿ@

%mio_embeddings/aid_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containeraid_embedding*
shape:ÿÿÿÿÿÿÿÿÿ@

%mio_embeddings/aid_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:ÿÿÿÿÿÿÿÿÿ@*
	containeraid_embedding

%mio_embeddings/uid_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containeruid_embedding*
shape:ÿÿÿÿÿÿÿÿÿ@

%mio_embeddings/uid_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:ÿÿÿÿÿÿÿÿÿ@*
	containeruid_embedding

%mio_embeddings/did_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerdid_embedding*
shape:ÿÿÿÿÿÿÿÿÿ@

%mio_embeddings/did_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerdid_embedding*
shape:ÿÿÿÿÿÿÿÿÿ@

)mio_embeddings/context_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS* 
	containercontext_embedding*
shape:ÿÿÿÿÿÿÿÿÿ@

)mio_embeddings/context_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS* 
	containercontext_embedding*
shape:ÿÿÿÿÿÿÿÿÿ@

&mio_embeddings/c_id_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:ÿÿÿÿÿÿÿÿÿ*
	containerc_id_embedding

&mio_embeddings/c_id_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:ÿÿÿÿÿÿÿÿÿ*
	containerc_id_embedding

(mio_embeddings/c_info_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerc_info_embedding*
shape:ÿÿÿÿÿÿÿÿÿÀ

(mio_embeddings/c_info_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerc_info_embedding*
shape:ÿÿÿÿÿÿÿÿÿÀ

*mio_embeddings/position_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*!
	containerposition_embedding*
shape:ÿÿÿÿÿÿÿÿÿ

*mio_embeddings/position_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*!
	containerposition_embedding*
shape:ÿÿÿÿÿÿÿÿÿ
©
/mio_embeddings/comment_genre_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containercomment_genre_embedding*
shape:ÿÿÿÿÿÿÿÿÿ
©
/mio_embeddings/comment_genre_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containercomment_genre_embedding*
shape:ÿÿÿÿÿÿÿÿÿ
«
0mio_embeddings/comment_length_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containercomment_length_embedding*
shape:ÿÿÿÿÿÿÿÿÿ 
«
0mio_embeddings/comment_length_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:ÿÿÿÿÿÿÿÿÿ *'
	containercomment_length_embedding
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
concat/values_3/axisConst*
value	B : *
dtype0

concat/values_3GatherV2%mio_embeddings/pid_embedding/variableCastconcat/values_3/axis*
Taxis0*
Tindices0*
Tparams0
>
concat/values_4/axisConst*
dtype0*
value	B : 

concat/values_4GatherV2%mio_embeddings/aid_embedding/variableCastconcat/values_4/axis*
Taxis0*
Tindices0*
Tparams0
>
concat/values_5/axisConst*
value	B : *
dtype0

concat/values_5GatherV2%mio_embeddings/uid_embedding/variableCastconcat/values_5/axis*
Taxis0*
Tindices0*
Tparams0
>
concat/values_6/axisConst*
value	B : *
dtype0

concat/values_6GatherV2%mio_embeddings/did_embedding/variableCastconcat/values_6/axis*
Tparams0*
Taxis0*
Tindices0
>
concat/values_7/axisConst*
value	B : *
dtype0

concat/values_7GatherV2)mio_embeddings/context_embedding/variableCastconcat/values_7/axis*
Tindices0*
Tparams0*
Taxis0
>
concat/axisConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0
Ø
concatConcatV2concat/values_0&mio_embeddings/c_id_embedding/variable(mio_embeddings/c_info_embedding/variableconcat/values_3concat/values_4concat/values_5concat/values_6concat/values_7/mio_embeddings/comment_genre_embedding/variable0mio_embeddings/comment_length_embedding/variableconcat/axis*

Tidx0*
T0*
N

@
concat_1/values_0/axisConst*
value	B : *
dtype0

concat_1/values_0GatherV2%mio_embeddings/did_embedding/variableCastconcat_1/values_0/axis*
Taxis0*
Tindices0*
Tparams0
@
concat_1/values_2/axisConst*
value	B : *
dtype0

concat_1/values_2GatherV2)mio_embeddings/context_embedding/variableCastconcat_1/values_2/axis*
Taxis0*
Tindices0*
Tparams0
@
concat_1/axisConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0

concat_1ConcatV2concat_1/values_0*mio_embeddings/position_embedding/variableconcat_1/values_2concat_1/axis*
N*

Tidx0*
T0
 
-mio_variable/expand_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*&
	containerexpand_xtr/dense/kernel
 
-mio_variable/expand_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerexpand_xtr/dense/kernel*
shape:
°
U
 Initializer/random_uniform/shapeConst*
dtype0*
valueB"°     
K
Initializer/random_uniform/minConst*
valueB
 *dF£½*
dtype0
K
Initializer/random_uniform/maxConst*
valueB
 *dF£=*
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
 expand_xtr/dense/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
j
expand_xtr/dense/LeakyRelu/mulMul expand_xtr/dense/LeakyRelu/alphaexpand_xtr/dense/BiasAdd*
T0
h
expand_xtr/dense/LeakyReluMaximumexpand_xtr/dense/LeakyRelu/mulexpand_xtr/dense/BiasAdd*
T0
L
expand_xtr/dropout/IdentityIdentityexpand_xtr/dense/LeakyRelu*
T0
¤
/mio_variable/expand_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*(
	containerexpand_xtr/dense_1/kernel
¤
/mio_variable/expand_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerexpand_xtr/dense_1/kernel*
shape:

W
"Initializer_2/random_uniform/shapeConst*
valueB"      *
dtype0
M
 Initializer_2/random_uniform/minConst*
valueB
 *   ¾*
dtype0
M
 Initializer_2/random_uniform/maxConst*
valueB
 *   >*
dtype0
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
×
Assign_2Assign/mio_variable/expand_xtr/dense_1/kernel/gradientInitializer_2/random_uniform*
T0*B
_class8
64loc:@mio_variable/expand_xtr/dense_1/kernel/gradient*
validate_shape(*
use_locking(
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
Ê
Assign_3Assign-mio_variable/expand_xtr/dense_1/bias/gradientInitializer_3/zeros*
use_locking(*
T0*@
_class6
42loc:@mio_variable/expand_xtr/dense_1/bias/gradient*
validate_shape(
 
expand_xtr/dense_1/MatMulMatMulexpand_xtr/dropout/Identity/mio_variable/expand_xtr/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0

expand_xtr/dense_1/BiasAddBiasAddexpand_xtr/dense_1/MatMul-mio_variable/expand_xtr/dense_1/bias/variable*
data_formatNHWC*
T0
O
"expand_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
p
 expand_xtr/dense_1/LeakyRelu/mulMul"expand_xtr/dense_1/LeakyRelu/alphaexpand_xtr/dense_1/BiasAdd*
T0
n
expand_xtr/dense_1/LeakyReluMaximum expand_xtr/dense_1/LeakyRelu/mulexpand_xtr/dense_1/BiasAdd*
T0
P
expand_xtr/dropout_1/IdentityIdentityexpand_xtr/dense_1/LeakyRelu*
T0
£
/mio_variable/expand_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerexpand_xtr/dense_2/kernel*
shape:	@
£
/mio_variable/expand_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerexpand_xtr/dense_2/kernel*
shape:	@
W
"Initializer_4/random_uniform/shapeConst*
valueB"   @   *
dtype0
M
 Initializer_4/random_uniform/minConst*
valueB
 *ó5¾*
dtype0
M
 Initializer_4/random_uniform/maxConst*
valueB
 *ó5>*
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
Assign_4Assign/mio_variable/expand_xtr/dense_2/kernel/gradientInitializer_4/random_uniform*B
_class8
64loc:@mio_variable/expand_xtr/dense_2/kernel/gradient*
validate_shape(*
use_locking(*
T0
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
Ê
Assign_5Assign-mio_variable/expand_xtr/dense_2/bias/gradientInitializer_5/zeros*
use_locking(*
T0*@
_class6
42loc:@mio_variable/expand_xtr/dense_2/bias/gradient*
validate_shape(
¢
expand_xtr/dense_2/MatMulMatMulexpand_xtr/dropout_1/Identity/mio_variable/expand_xtr/dense_2/kernel/variable*
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
 *ÍÌL>*
dtype0
p
 expand_xtr/dense_2/LeakyRelu/mulMul"expand_xtr/dense_2/LeakyRelu/alphaexpand_xtr/dense_2/BiasAdd*
T0
n
expand_xtr/dense_2/LeakyReluMaximum expand_xtr/dense_2/LeakyRelu/mulexpand_xtr/dense_2/BiasAdd*
T0
¢
/mio_variable/expand_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerexpand_xtr/dense_3/kernel*
shape
:@
¢
/mio_variable/expand_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*(
	containerexpand_xtr/dense_3/kernel
W
"Initializer_6/random_uniform/shapeConst*
valueB"@      *
dtype0
M
 Initializer_6/random_uniform/minConst*
valueB
 *¾*
dtype0
M
 Initializer_6/random_uniform/maxConst*
valueB
 *>*
dtype0

*Initializer_6/random_uniform/RandomUniformRandomUniform"Initializer_6/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
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
Ê
Assign_7Assign-mio_variable/expand_xtr/dense_3/bias/gradientInitializer_7/zeros*
use_locking(*
T0*@
_class6
42loc:@mio_variable/expand_xtr/dense_3/bias/gradient*
validate_shape(
¡
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
+mio_variable/like_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*$
	containerlike_xtr/dense/kernel

+mio_variable/like_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*$
	containerlike_xtr/dense/kernel
W
"Initializer_8/random_uniform/shapeConst*
valueB"°     *
dtype0
M
 Initializer_8/random_uniform/minConst*
dtype0*
valueB
 *dF£½
M
 Initializer_8/random_uniform/maxConst*
dtype0*
valueB
 *dF£=
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
Assign_8Assign+mio_variable/like_xtr/dense/kernel/gradientInitializer_8/random_uniform*>
_class4
20loc:@mio_variable/like_xtr/dense/kernel/gradient*
validate_shape(*
use_locking(*
T0
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
Assign_9Assign)mio_variable/like_xtr/dense/bias/gradientInitializer_9/zeros*
use_locking(*
T0*<
_class2
0.loc:@mio_variable/like_xtr/dense/bias/gradient*
validate_shape(

like_xtr/dense/MatMulMatMulconcat+mio_variable/like_xtr/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 

like_xtr/dense/BiasAddBiasAddlike_xtr/dense/MatMul)mio_variable/like_xtr/dense/bias/variable*
T0*
data_formatNHWC
K
like_xtr/dense/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
d
like_xtr/dense/LeakyRelu/mulMullike_xtr/dense/LeakyRelu/alphalike_xtr/dense/BiasAdd*
T0
b
like_xtr/dense/LeakyReluMaximumlike_xtr/dense/LeakyRelu/mullike_xtr/dense/BiasAdd*
T0
H
like_xtr/dropout/IdentityIdentitylike_xtr/dense/LeakyRelu*
T0
 
-mio_variable/like_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerlike_xtr/dense_1/kernel*
shape:

 
-mio_variable/like_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerlike_xtr/dense_1/kernel*
shape:

X
#Initializer_10/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_10/random_uniform/minConst*
dtype0*
valueB
 *   ¾
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
	Assign_10Assign-mio_variable/like_xtr/dense_1/kernel/gradientInitializer_10/random_uniform*
use_locking(*
T0*@
_class6
42loc:@mio_variable/like_xtr/dense_1/kernel/gradient*
validate_shape(

+mio_variable/like_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containerlike_xtr/dense_1/bias

+mio_variable/like_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerlike_xtr/dense_1/bias*
shape:
F
Initializer_11/zerosConst*
valueB*    *
dtype0
È
	Assign_11Assign+mio_variable/like_xtr/dense_1/bias/gradientInitializer_11/zeros*
use_locking(*
T0*>
_class4
20loc:@mio_variable/like_xtr/dense_1/bias/gradient*
validate_shape(

like_xtr/dense_1/MatMulMatMullike_xtr/dropout/Identity-mio_variable/like_xtr/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 

like_xtr/dense_1/BiasAddBiasAddlike_xtr/dense_1/MatMul+mio_variable/like_xtr/dense_1/bias/variable*
T0*
data_formatNHWC
M
 like_xtr/dense_1/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍÌL>
j
like_xtr/dense_1/LeakyRelu/mulMul like_xtr/dense_1/LeakyRelu/alphalike_xtr/dense_1/BiasAdd*
T0
h
like_xtr/dense_1/LeakyReluMaximumlike_xtr/dense_1/LeakyRelu/mullike_xtr/dense_1/BiasAdd*
T0
L
like_xtr/dropout_1/IdentityIdentitylike_xtr/dense_1/LeakyRelu*
T0

-mio_variable/like_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerlike_xtr/dense_2/kernel*
shape:	@

-mio_variable/like_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*&
	containerlike_xtr/dense_2/kernel
X
#Initializer_12/random_uniform/shapeConst*
dtype0*
valueB"   @   
N
!Initializer_12/random_uniform/minConst*
valueB
 *ó5¾*
dtype0
N
!Initializer_12/random_uniform/maxConst*
dtype0*
valueB
 *ó5>
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
È
	Assign_13Assign+mio_variable/like_xtr/dense_2/bias/gradientInitializer_13/zeros*
use_locking(*
T0*>
_class4
20loc:@mio_variable/like_xtr/dense_2/bias/gradient*
validate_shape(

like_xtr/dense_2/MatMulMatMullike_xtr/dropout_1/Identity-mio_variable/like_xtr/dense_2/kernel/variable*
transpose_b( *
T0*
transpose_a( 

like_xtr/dense_2/BiasAddBiasAddlike_xtr/dense_2/MatMul+mio_variable/like_xtr/dense_2/bias/variable*
data_formatNHWC*
T0
M
 like_xtr/dense_2/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍÌL>
j
like_xtr/dense_2/LeakyRelu/mulMul like_xtr/dense_2/LeakyRelu/alphalike_xtr/dense_2/BiasAdd*
T0
h
like_xtr/dense_2/LeakyReluMaximumlike_xtr/dense_2/LeakyRelu/mullike_xtr/dense_2/BiasAdd*
T0

-mio_variable/like_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*&
	containerlike_xtr/dense_3/kernel

-mio_variable/like_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerlike_xtr/dense_3/kernel*
shape
:@
X
#Initializer_14/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_14/random_uniform/minConst*
dtype0*
valueB
 *¾
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
	Assign_14Assign-mio_variable/like_xtr/dense_3/kernel/gradientInitializer_14/random_uniform*
T0*@
_class6
42loc:@mio_variable/like_xtr/dense_3/kernel/gradient*
validate_shape(*
use_locking(
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
È
	Assign_15Assign+mio_variable/like_xtr/dense_3/bias/gradientInitializer_15/zeros*
T0*>
_class4
20loc:@mio_variable/like_xtr/dense_3/bias/gradient*
validate_shape(*
use_locking(

like_xtr/dense_3/MatMulMatMullike_xtr/dense_2/LeakyRelu-mio_variable/like_xtr/dense_3/kernel/variable*
transpose_a( *
transpose_b( *
T0

like_xtr/dense_3/BiasAddBiasAddlike_xtr/dense_3/MatMul+mio_variable/like_xtr/dense_3/bias/variable*
T0*
data_formatNHWC
F
like_xtr/dense_3/SigmoidSigmoidlike_xtr/dense_3/BiasAdd*
T0

,mio_variable/reply_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*%
	containerreply_xtr/dense/kernel

,mio_variable/reply_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*%
	containerreply_xtr/dense/kernel
X
#Initializer_16/random_uniform/shapeConst*
valueB"°     *
dtype0
N
!Initializer_16/random_uniform/minConst*
valueB
 *dF£½*
dtype0
N
!Initializer_16/random_uniform/maxConst*
valueB
 *dF£=*
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
	Assign_17Assign*mio_variable/reply_xtr/dense/bias/gradientInitializer_17/zeros*
validate_shape(*
use_locking(*
T0*=
_class3
1/loc:@mio_variable/reply_xtr/dense/bias/gradient
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
 *ÍÌL>*
dtype0
g
reply_xtr/dense/LeakyRelu/mulMulreply_xtr/dense/LeakyRelu/alphareply_xtr/dense/BiasAdd*
T0
e
reply_xtr/dense/LeakyReluMaximumreply_xtr/dense/LeakyRelu/mulreply_xtr/dense/BiasAdd*
T0
J
reply_xtr/dropout/IdentityIdentityreply_xtr/dense/LeakyRelu*
T0
¢
.mio_variable/reply_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerreply_xtr/dense_1/kernel*
shape:

¢
.mio_variable/reply_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerreply_xtr/dense_1/kernel*
shape:

X
#Initializer_18/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_18/random_uniform/minConst*
valueB
 *   ¾*
dtype0
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
×
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
,mio_variable/reply_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense_1/bias*
shape:
F
Initializer_19/zerosConst*
dtype0*
valueB*    
Ê
	Assign_19Assign,mio_variable/reply_xtr/dense_1/bias/gradientInitializer_19/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/reply_xtr/dense_1/bias/gradient*
validate_shape(

reply_xtr/dense_1/MatMulMatMulreply_xtr/dropout/Identity.mio_variable/reply_xtr/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0

reply_xtr/dense_1/BiasAddBiasAddreply_xtr/dense_1/MatMul,mio_variable/reply_xtr/dense_1/bias/variable*
T0*
data_formatNHWC
N
!reply_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
m
reply_xtr/dense_1/LeakyRelu/mulMul!reply_xtr/dense_1/LeakyRelu/alphareply_xtr/dense_1/BiasAdd*
T0
k
reply_xtr/dense_1/LeakyReluMaximumreply_xtr/dense_1/LeakyRelu/mulreply_xtr/dense_1/BiasAdd*
T0
N
reply_xtr/dropout_1/IdentityIdentityreply_xtr/dense_1/LeakyRelu*
T0
¡
.mio_variable/reply_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*'
	containerreply_xtr/dense_2/kernel
¡
.mio_variable/reply_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerreply_xtr/dense_2/kernel*
shape:	@
X
#Initializer_20/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_20/random_uniform/minConst*
dtype0*
valueB
 *ó5¾
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
×
	Assign_20Assign.mio_variable/reply_xtr/dense_2/kernel/gradientInitializer_20/random_uniform*
use_locking(*
T0*A
_class7
53loc:@mio_variable/reply_xtr/dense_2/kernel/gradient*
validate_shape(

,mio_variable/reply_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*%
	containerreply_xtr/dense_2/bias

,mio_variable/reply_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*%
	containerreply_xtr/dense_2/bias
E
Initializer_21/zerosConst*
valueB@*    *
dtype0
Ê
	Assign_21Assign,mio_variable/reply_xtr/dense_2/bias/gradientInitializer_21/zeros*
validate_shape(*
use_locking(*
T0*?
_class5
31loc:@mio_variable/reply_xtr/dense_2/bias/gradient

reply_xtr/dense_2/MatMulMatMulreply_xtr/dropout_1/Identity.mio_variable/reply_xtr/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 

reply_xtr/dense_2/BiasAddBiasAddreply_xtr/dense_2/MatMul,mio_variable/reply_xtr/dense_2/bias/variable*
data_formatNHWC*
T0
N
!reply_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
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
!Initializer_22/random_uniform/minConst*
valueB
 *¾*
dtype0
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
Ê
	Assign_23Assign,mio_variable/reply_xtr/dense_3/bias/gradientInitializer_23/zeros*
validate_shape(*
use_locking(*
T0*?
_class5
31loc:@mio_variable/reply_xtr/dense_3/bias/gradient
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
°

+mio_variable/copy_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense/kernel*
shape:
°
X
#Initializer_24/random_uniform/shapeConst*
dtype0*
valueB"°     
N
!Initializer_24/random_uniform/minConst*
valueB
 *dF£½*
dtype0
N
!Initializer_24/random_uniform/maxConst*
dtype0*
valueB
 *dF£=

+Initializer_24/random_uniform/RandomUniformRandomUniform#Initializer_24/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_24/random_uniform/subSub!Initializer_24/random_uniform/max!Initializer_24/random_uniform/min*
T0

!Initializer_24/random_uniform/mulMul+Initializer_24/random_uniform/RandomUniform!Initializer_24/random_uniform/sub*
T0
s
Initializer_24/random_uniformAdd!Initializer_24/random_uniform/mul!Initializer_24/random_uniform/min*
T0
Ñ
	Assign_24Assign+mio_variable/copy_xtr/dense/kernel/gradientInitializer_24/random_uniform*
use_locking(*
T0*>
_class4
20loc:@mio_variable/copy_xtr/dense/kernel/gradient*
validate_shape(

)mio_variable/copy_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*"
	containercopy_xtr/dense/bias*
shape:

)mio_variable/copy_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*"
	containercopy_xtr/dense/bias*
shape:
F
Initializer_25/zerosConst*
valueB*    *
dtype0
Ä
	Assign_25Assign)mio_variable/copy_xtr/dense/bias/gradientInitializer_25/zeros*<
_class2
0.loc:@mio_variable/copy_xtr/dense/bias/gradient*
validate_shape(*
use_locking(*
T0

copy_xtr/dense/MatMulMatMulconcat+mio_variable/copy_xtr/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 

copy_xtr/dense/BiasAddBiasAddcopy_xtr/dense/MatMul)mio_variable/copy_xtr/dense/bias/variable*
T0*
data_formatNHWC
K
copy_xtr/dense/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
d
copy_xtr/dense/LeakyRelu/mulMulcopy_xtr/dense/LeakyRelu/alphacopy_xtr/dense/BiasAdd*
T0
b
copy_xtr/dense/LeakyReluMaximumcopy_xtr/dense/LeakyRelu/mulcopy_xtr/dense/BiasAdd*
T0
H
copy_xtr/dropout/IdentityIdentitycopy_xtr/dense/LeakyRelu*
T0
 
-mio_variable/copy_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containercopy_xtr/dense_1/kernel*
shape:

 
-mio_variable/copy_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containercopy_xtr/dense_1/kernel*
shape:

X
#Initializer_26/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_26/random_uniform/minConst*
valueB
 *   ¾*
dtype0
N
!Initializer_26/random_uniform/maxConst*
valueB
 *   >*
dtype0

+Initializer_26/random_uniform/RandomUniformRandomUniform#Initializer_26/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
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
+mio_variable/copy_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containercopy_xtr/dense_1/bias
F
Initializer_27/zerosConst*
valueB*    *
dtype0
È
	Assign_27Assign+mio_variable/copy_xtr/dense_1/bias/gradientInitializer_27/zeros*>
_class4
20loc:@mio_variable/copy_xtr/dense_1/bias/gradient*
validate_shape(*
use_locking(*
T0

copy_xtr/dense_1/MatMulMatMulcopy_xtr/dropout/Identity-mio_variable/copy_xtr/dense_1/kernel/variable*
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
 *ÍÌL>*
dtype0
j
copy_xtr/dense_1/LeakyRelu/mulMul copy_xtr/dense_1/LeakyRelu/alphacopy_xtr/dense_1/BiasAdd*
T0
h
copy_xtr/dense_1/LeakyReluMaximumcopy_xtr/dense_1/LeakyRelu/mulcopy_xtr/dense_1/BiasAdd*
T0
L
copy_xtr/dropout_1/IdentityIdentitycopy_xtr/dense_1/LeakyRelu*
T0

-mio_variable/copy_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containercopy_xtr/dense_2/kernel*
shape:	@
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
 *ó5¾*
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
	Assign_28Assign-mio_variable/copy_xtr/dense_2/kernel/gradientInitializer_28/random_uniform*@
_class6
42loc:@mio_variable/copy_xtr/dense_2/kernel/gradient*
validate_shape(*
use_locking(*
T0

+mio_variable/copy_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense_2/bias*
shape:@

+mio_variable/copy_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense_2/bias*
shape:@
E
Initializer_29/zerosConst*
valueB@*    *
dtype0
È
	Assign_29Assign+mio_variable/copy_xtr/dense_2/bias/gradientInitializer_29/zeros*
use_locking(*
T0*>
_class4
20loc:@mio_variable/copy_xtr/dense_2/bias/gradient*
validate_shape(

copy_xtr/dense_2/MatMulMatMulcopy_xtr/dropout_1/Identity-mio_variable/copy_xtr/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 

copy_xtr/dense_2/BiasAddBiasAddcopy_xtr/dense_2/MatMul+mio_variable/copy_xtr/dense_2/bias/variable*
T0*
data_formatNHWC
M
 copy_xtr/dense_2/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍÌL>
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
 *¾*
dtype0
N
!Initializer_30/random_uniform/maxConst*
valueB
 *>*
dtype0

+Initializer_30/random_uniform/RandomUniformRandomUniform#Initializer_30/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
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
	Assign_30Assign-mio_variable/copy_xtr/dense_3/kernel/gradientInitializer_30/random_uniform*
T0*@
_class6
42loc:@mio_variable/copy_xtr/dense_3/kernel/gradient*
validate_shape(*
use_locking(
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
È
	Assign_31Assign+mio_variable/copy_xtr/dense_3/bias/gradientInitializer_31/zeros*
use_locking(*
T0*>
_class4
20loc:@mio_variable/copy_xtr/dense_3/bias/gradient*
validate_shape(

copy_xtr/dense_3/MatMulMatMulcopy_xtr/dense_2/LeakyRelu-mio_variable/copy_xtr/dense_3/kernel/variable*
transpose_b( *
T0*
transpose_a( 
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
°

,mio_variable/share_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containershare_xtr/dense/kernel*
shape:
°
X
#Initializer_32/random_uniform/shapeConst*
dtype0*
valueB"°     
N
!Initializer_32/random_uniform/minConst*
valueB
 *dF£½*
dtype0
N
!Initializer_32/random_uniform/maxConst*
valueB
 *dF£=*
dtype0

+Initializer_32/random_uniform/RandomUniformRandomUniform#Initializer_32/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
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
*mio_variable/share_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*#
	containershare_xtr/dense/bias

*mio_variable/share_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*#
	containershare_xtr/dense/bias*
shape:
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
share_xtr/dense/MatMulMatMulconcat,mio_variable/share_xtr/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0

share_xtr/dense/BiasAddBiasAddshare_xtr/dense/MatMul*mio_variable/share_xtr/dense/bias/variable*
T0*
data_formatNHWC
L
share_xtr/dense/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
g
share_xtr/dense/LeakyRelu/mulMulshare_xtr/dense/LeakyRelu/alphashare_xtr/dense/BiasAdd*
T0
e
share_xtr/dense/LeakyReluMaximumshare_xtr/dense/LeakyRelu/mulshare_xtr/dense/BiasAdd*
T0
J
share_xtr/dropout/IdentityIdentityshare_xtr/dense/LeakyRelu*
T0
¢
.mio_variable/share_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containershare_xtr/dense_1/kernel*
shape:

¢
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
 *   ¾*
dtype0
N
!Initializer_34/random_uniform/maxConst*
valueB
 *   >*
dtype0

+Initializer_34/random_uniform/RandomUniformRandomUniform#Initializer_34/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_34/random_uniform/subSub!Initializer_34/random_uniform/max!Initializer_34/random_uniform/min*
T0

!Initializer_34/random_uniform/mulMul+Initializer_34/random_uniform/RandomUniform!Initializer_34/random_uniform/sub*
T0
s
Initializer_34/random_uniformAdd!Initializer_34/random_uniform/mul!Initializer_34/random_uniform/min*
T0
×
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
Ê
	Assign_35Assign,mio_variable/share_xtr/dense_1/bias/gradientInitializer_35/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/share_xtr/dense_1/bias/gradient*
validate_shape(

share_xtr/dense_1/MatMulMatMulshare_xtr/dropout/Identity.mio_variable/share_xtr/dense_1/kernel/variable*
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
 *ÍÌL>*
dtype0
m
share_xtr/dense_1/LeakyRelu/mulMul!share_xtr/dense_1/LeakyRelu/alphashare_xtr/dense_1/BiasAdd*
T0
k
share_xtr/dense_1/LeakyReluMaximumshare_xtr/dense_1/LeakyRelu/mulshare_xtr/dense_1/BiasAdd*
T0
N
share_xtr/dropout_1/IdentityIdentityshare_xtr/dense_1/LeakyRelu*
T0
¡
.mio_variable/share_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containershare_xtr/dense_2/kernel*
shape:	@
¡
.mio_variable/share_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*'
	containershare_xtr/dense_2/kernel
X
#Initializer_36/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_36/random_uniform/minConst*
valueB
 *ó5¾*
dtype0
N
!Initializer_36/random_uniform/maxConst*
valueB
 *ó5>*
dtype0

+Initializer_36/random_uniform/RandomUniformRandomUniform#Initializer_36/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_36/random_uniform/subSub!Initializer_36/random_uniform/max!Initializer_36/random_uniform/min*
T0

!Initializer_36/random_uniform/mulMul+Initializer_36/random_uniform/RandomUniform!Initializer_36/random_uniform/sub*
T0
s
Initializer_36/random_uniformAdd!Initializer_36/random_uniform/mul!Initializer_36/random_uniform/min*
T0
×
	Assign_36Assign.mio_variable/share_xtr/dense_2/kernel/gradientInitializer_36/random_uniform*
use_locking(*
T0*A
_class7
53loc:@mio_variable/share_xtr/dense_2/kernel/gradient*
validate_shape(

,mio_variable/share_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containershare_xtr/dense_2/bias*
shape:@

,mio_variable/share_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*%
	containershare_xtr/dense_2/bias
E
Initializer_37/zerosConst*
valueB@*    *
dtype0
Ê
	Assign_37Assign,mio_variable/share_xtr/dense_2/bias/gradientInitializer_37/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/share_xtr/dense_2/bias/gradient*
validate_shape(

share_xtr/dense_2/MatMulMatMulshare_xtr/dropout_1/Identity.mio_variable/share_xtr/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 

share_xtr/dense_2/BiasAddBiasAddshare_xtr/dense_2/MatMul,mio_variable/share_xtr/dense_2/bias/variable*
T0*
data_formatNHWC
N
!share_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
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
.mio_variable/share_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containershare_xtr/dense_3/kernel*
shape
:@
X
#Initializer_38/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_38/random_uniform/minConst*
valueB
 *¾*
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
×
	Assign_38Assign.mio_variable/share_xtr/dense_3/kernel/gradientInitializer_38/random_uniform*
use_locking(*
T0*A
_class7
53loc:@mio_variable/share_xtr/dense_3/kernel/gradient*
validate_shape(

,mio_variable/share_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*%
	containershare_xtr/dense_3/bias

,mio_variable/share_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containershare_xtr/dense_3/bias*
shape:
E
Initializer_39/zerosConst*
valueB*    *
dtype0
Ê
	Assign_39Assign,mio_variable/share_xtr/dense_3/bias/gradientInitializer_39/zeros*
T0*?
_class5
31loc:@mio_variable/share_xtr/dense_3/bias/gradient*
validate_shape(*
use_locking(

share_xtr/dense_3/MatMulMatMulshare_xtr/dense_2/LeakyRelu.mio_variable/share_xtr/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 

share_xtr/dense_3/BiasAddBiasAddshare_xtr/dense_3/MatMul,mio_variable/share_xtr/dense_3/bias/variable*
T0*
data_formatNHWC
H
share_xtr/dense_3/SigmoidSigmoidshare_xtr/dense_3/BiasAdd*
T0
¤
/mio_variable/audience_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeraudience_xtr/dense/kernel*
shape:
°
¤
/mio_variable/audience_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*(
	containeraudience_xtr/dense/kernel
X
#Initializer_40/random_uniform/shapeConst*
valueB"°     *
dtype0
N
!Initializer_40/random_uniform/minConst*
valueB
 *dF£½*
dtype0
N
!Initializer_40/random_uniform/maxConst*
dtype0*
valueB
 *dF£=
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
Ù
	Assign_40Assign/mio_variable/audience_xtr/dense/kernel/gradientInitializer_40/random_uniform*
use_locking(*
T0*B
_class8
64loc:@mio_variable/audience_xtr/dense/kernel/gradient*
validate_shape(

-mio_variable/audience_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containeraudience_xtr/dense/bias*
shape:

-mio_variable/audience_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containeraudience_xtr/dense/bias*
shape:
F
Initializer_41/zerosConst*
valueB*    *
dtype0
Ì
	Assign_41Assign-mio_variable/audience_xtr/dense/bias/gradientInitializer_41/zeros*
use_locking(*
T0*@
_class6
42loc:@mio_variable/audience_xtr/dense/bias/gradient*
validate_shape(

audience_xtr/dense/MatMulMatMulconcat/mio_variable/audience_xtr/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0

audience_xtr/dense/BiasAddBiasAddaudience_xtr/dense/MatMul-mio_variable/audience_xtr/dense/bias/variable*
T0*
data_formatNHWC
O
"audience_xtr/dense/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
p
 audience_xtr/dense/LeakyRelu/mulMul"audience_xtr/dense/LeakyRelu/alphaaudience_xtr/dense/BiasAdd*
T0
n
audience_xtr/dense/LeakyReluMaximum audience_xtr/dense/LeakyRelu/mulaudience_xtr/dense/BiasAdd*
T0
P
audience_xtr/dropout/IdentityIdentityaudience_xtr/dense/LeakyRelu*
T0
¨
1mio_variable/audience_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS**
	containeraudience_xtr/dense_1/kernel*
shape:

¨
1mio_variable/audience_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS**
	containeraudience_xtr/dense_1/kernel*
shape:

X
#Initializer_42/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_42/random_uniform/minConst*
valueB
 *   ¾*
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
	Assign_42Assign1mio_variable/audience_xtr/dense_1/kernel/gradientInitializer_42/random_uniform*
use_locking(*
T0*D
_class:
86loc:@mio_variable/audience_xtr/dense_1/kernel/gradient*
validate_shape(

/mio_variable/audience_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*(
	containeraudience_xtr/dense_1/bias

/mio_variable/audience_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*(
	containeraudience_xtr/dense_1/bias
F
Initializer_43/zerosConst*
dtype0*
valueB*    
Ð
	Assign_43Assign/mio_variable/audience_xtr/dense_1/bias/gradientInitializer_43/zeros*
validate_shape(*
use_locking(*
T0*B
_class8
64loc:@mio_variable/audience_xtr/dense_1/bias/gradient
¦
audience_xtr/dense_1/MatMulMatMulaudience_xtr/dropout/Identity1mio_variable/audience_xtr/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 

audience_xtr/dense_1/BiasAddBiasAddaudience_xtr/dense_1/MatMul/mio_variable/audience_xtr/dense_1/bias/variable*
T0*
data_formatNHWC
Q
$audience_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
v
"audience_xtr/dense_1/LeakyRelu/mulMul$audience_xtr/dense_1/LeakyRelu/alphaaudience_xtr/dense_1/BiasAdd*
T0
t
audience_xtr/dense_1/LeakyReluMaximum"audience_xtr/dense_1/LeakyRelu/mulaudience_xtr/dense_1/BiasAdd*
T0
T
audience_xtr/dropout_1/IdentityIdentityaudience_xtr/dense_1/LeakyRelu*
T0
§
1mio_variable/audience_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS**
	containeraudience_xtr/dense_2/kernel*
shape:	@
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
 *ó5¾*
dtype0
N
!Initializer_44/random_uniform/maxConst*
valueB
 *ó5>*
dtype0

+Initializer_44/random_uniform/RandomUniformRandomUniform#Initializer_44/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
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
	Assign_44Assign1mio_variable/audience_xtr/dense_2/kernel/gradientInitializer_44/random_uniform*
use_locking(*
T0*D
_class:
86loc:@mio_variable/audience_xtr/dense_2/kernel/gradient*
validate_shape(

/mio_variable/audience_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeraudience_xtr/dense_2/bias*
shape:@

/mio_variable/audience_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeraudience_xtr/dense_2/bias*
shape:@
E
Initializer_45/zerosConst*
valueB@*    *
dtype0
Ð
	Assign_45Assign/mio_variable/audience_xtr/dense_2/bias/gradientInitializer_45/zeros*
validate_shape(*
use_locking(*
T0*B
_class8
64loc:@mio_variable/audience_xtr/dense_2/bias/gradient
¨
audience_xtr/dense_2/MatMulMatMulaudience_xtr/dropout_1/Identity1mio_variable/audience_xtr/dense_2/kernel/variable*
transpose_b( *
T0*
transpose_a( 

audience_xtr/dense_2/BiasAddBiasAddaudience_xtr/dense_2/MatMul/mio_variable/audience_xtr/dense_2/bias/variable*
T0*
data_formatNHWC
Q
$audience_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
v
"audience_xtr/dense_2/LeakyRelu/mulMul$audience_xtr/dense_2/LeakyRelu/alphaaudience_xtr/dense_2/BiasAdd*
T0
t
audience_xtr/dense_2/LeakyReluMaximum"audience_xtr/dense_2/LeakyRelu/mulaudience_xtr/dense_2/BiasAdd*
T0
¦
1mio_variable/audience_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS**
	containeraudience_xtr/dense_3/kernel*
shape
:@
¦
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
 *¾*
dtype0
N
!Initializer_46/random_uniform/maxConst*
valueB
 *>*
dtype0

+Initializer_46/random_uniform/RandomUniformRandomUniform#Initializer_46/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
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
	Assign_46Assign1mio_variable/audience_xtr/dense_3/kernel/gradientInitializer_46/random_uniform*
validate_shape(*
use_locking(*
T0*D
_class:
86loc:@mio_variable/audience_xtr/dense_3/kernel/gradient
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
audience_xtr/dense_3/MatMulMatMulaudience_xtr/dense_2/LeakyRelu1mio_variable/audience_xtr/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 

audience_xtr/dense_3/BiasAddBiasAddaudience_xtr/dense_3/MatMul/mio_variable/audience_xtr/dense_3/bias/variable*
T0*
data_formatNHWC
N
audience_xtr/dense_3/SigmoidSigmoidaudience_xtr/dense_3/BiasAdd*
T0
¶
8mio_variable/continuous_expand_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*1
	container$"continuous_expand_xtr/dense/kernel
¶
8mio_variable/continuous_expand_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*1
	container$"continuous_expand_xtr/dense/kernel
X
#Initializer_48/random_uniform/shapeConst*
valueB"°     *
dtype0
N
!Initializer_48/random_uniform/minConst*
valueB
 *dF£½*
dtype0
N
!Initializer_48/random_uniform/maxConst*
valueB
 *dF£=*
dtype0

+Initializer_48/random_uniform/RandomUniformRandomUniform#Initializer_48/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
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
	Assign_48Assign8mio_variable/continuous_expand_xtr/dense/kernel/gradientInitializer_48/random_uniform*
T0*K
_classA
?=loc:@mio_variable/continuous_expand_xtr/dense/kernel/gradient*
validate_shape(*
use_locking(
­
6mio_variable/continuous_expand_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*/
	container" continuous_expand_xtr/dense/bias
­
6mio_variable/continuous_expand_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*/
	container" continuous_expand_xtr/dense/bias
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
"continuous_expand_xtr/dense/MatMulMatMulconcat8mio_variable/continuous_expand_xtr/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 
ª
#continuous_expand_xtr/dense/BiasAddBiasAdd"continuous_expand_xtr/dense/MatMul6mio_variable/continuous_expand_xtr/dense/bias/variable*
T0*
data_formatNHWC
X
+continuous_expand_xtr/dense/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0

)continuous_expand_xtr/dense/LeakyRelu/mulMul+continuous_expand_xtr/dense/LeakyRelu/alpha#continuous_expand_xtr/dense/BiasAdd*
T0

%continuous_expand_xtr/dense/LeakyReluMaximum)continuous_expand_xtr/dense/LeakyRelu/mul#continuous_expand_xtr/dense/BiasAdd*
T0
b
&continuous_expand_xtr/dropout/IdentityIdentity%continuous_expand_xtr/dense/LeakyRelu*
T0
º
:mio_variable/continuous_expand_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$continuous_expand_xtr/dense_1/kernel*
shape:

º
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
 *   ¾*
dtype0
N
!Initializer_50/random_uniform/maxConst*
valueB
 *   >*
dtype0

+Initializer_50/random_uniform/RandomUniformRandomUniform#Initializer_50/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
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
±
8mio_variable/continuous_expand_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*1
	container$"continuous_expand_xtr/dense_1/bias
±
8mio_variable/continuous_expand_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"continuous_expand_xtr/dense_1/bias*
shape:
F
Initializer_51/zerosConst*
valueB*    *
dtype0
â
	Assign_51Assign8mio_variable/continuous_expand_xtr/dense_1/bias/gradientInitializer_51/zeros*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/continuous_expand_xtr/dense_1/bias/gradient*
validate_shape(
Á
$continuous_expand_xtr/dense_1/MatMulMatMul&continuous_expand_xtr/dropout/Identity:mio_variable/continuous_expand_xtr/dense_1/kernel/variable*
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
 *ÍÌL>*
dtype0

+continuous_expand_xtr/dense_1/LeakyRelu/mulMul-continuous_expand_xtr/dense_1/LeakyRelu/alpha%continuous_expand_xtr/dense_1/BiasAdd*
T0

'continuous_expand_xtr/dense_1/LeakyReluMaximum+continuous_expand_xtr/dense_1/LeakyRelu/mul%continuous_expand_xtr/dense_1/BiasAdd*
T0
f
(continuous_expand_xtr/dropout_1/IdentityIdentity'continuous_expand_xtr/dense_1/LeakyRelu*
T0
¹
:mio_variable/continuous_expand_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*3
	container&$continuous_expand_xtr/dense_2/kernel
¹
:mio_variable/continuous_expand_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$continuous_expand_xtr/dense_2/kernel*
shape:	@
X
#Initializer_52/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_52/random_uniform/minConst*
valueB
 *ó5¾*
dtype0
N
!Initializer_52/random_uniform/maxConst*
valueB
 *ó5>*
dtype0

+Initializer_52/random_uniform/RandomUniformRandomUniform#Initializer_52/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
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
8mio_variable/continuous_expand_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*1
	container$"continuous_expand_xtr/dense_2/bias
°
8mio_variable/continuous_expand_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*1
	container$"continuous_expand_xtr/dense_2/bias
E
Initializer_53/zerosConst*
valueB@*    *
dtype0
â
	Assign_53Assign8mio_variable/continuous_expand_xtr/dense_2/bias/gradientInitializer_53/zeros*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/continuous_expand_xtr/dense_2/bias/gradient*
validate_shape(
Ã
$continuous_expand_xtr/dense_2/MatMulMatMul(continuous_expand_xtr/dropout_1/Identity:mio_variable/continuous_expand_xtr/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0
°
%continuous_expand_xtr/dense_2/BiasAddBiasAdd$continuous_expand_xtr/dense_2/MatMul8mio_variable/continuous_expand_xtr/dense_2/bias/variable*
T0*
data_formatNHWC
Z
-continuous_expand_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0

+continuous_expand_xtr/dense_2/LeakyRelu/mulMul-continuous_expand_xtr/dense_2/LeakyRelu/alpha%continuous_expand_xtr/dense_2/BiasAdd*
T0

'continuous_expand_xtr/dense_2/LeakyReluMaximum+continuous_expand_xtr/dense_2/LeakyRelu/mul%continuous_expand_xtr/dense_2/BiasAdd*
T0
¸
:mio_variable/continuous_expand_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*3
	container&$continuous_expand_xtr/dense_3/kernel
¸
:mio_variable/continuous_expand_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$continuous_expand_xtr/dense_3/kernel*
shape
:@
X
#Initializer_54/random_uniform/shapeConst*
dtype0*
valueB"@      
N
!Initializer_54/random_uniform/minConst*
valueB
 *¾*
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
	Assign_54Assign:mio_variable/continuous_expand_xtr/dense_3/kernel/gradientInitializer_54/random_uniform*
validate_shape(*
use_locking(*
T0*M
_classC
A?loc:@mio_variable/continuous_expand_xtr/dense_3/kernel/gradient
°
8mio_variable/continuous_expand_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*1
	container$"continuous_expand_xtr/dense_3/bias
°
8mio_variable/continuous_expand_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"continuous_expand_xtr/dense_3/bias*
shape:
E
Initializer_55/zerosConst*
valueB*    *
dtype0
â
	Assign_55Assign8mio_variable/continuous_expand_xtr/dense_3/bias/gradientInitializer_55/zeros*
T0*K
_classA
?=loc:@mio_variable/continuous_expand_xtr/dense_3/bias/gradient*
validate_shape(*
use_locking(
Â
$continuous_expand_xtr/dense_3/MatMulMatMul'continuous_expand_xtr/dense_2/LeakyRelu:mio_variable/continuous_expand_xtr/dense_3/kernel/variable*
transpose_b( *
T0*
transpose_a( 
°
%continuous_expand_xtr/dense_3/BiasAddBiasAdd$continuous_expand_xtr/dense_3/MatMul8mio_variable/continuous_expand_xtr/dense_3/bias/variable*
data_formatNHWC*
T0
`
%continuous_expand_xtr/dense_3/SigmoidSigmoid%continuous_expand_xtr/dense_3/BiasAdd*
T0
¬
3mio_variable/duration_predict/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*,
	containerduration_predict/dense/kernel
¬
3mio_variable/duration_predict/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerduration_predict/dense/kernel*
shape:
°
X
#Initializer_56/random_uniform/shapeConst*
valueB"°     *
dtype0
N
!Initializer_56/random_uniform/minConst*
valueB
 *dF£½*
dtype0
N
!Initializer_56/random_uniform/maxConst*
valueB
 *dF£=*
dtype0

+Initializer_56/random_uniform/RandomUniformRandomUniform#Initializer_56/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_56/random_uniform/subSub!Initializer_56/random_uniform/max!Initializer_56/random_uniform/min*
T0

!Initializer_56/random_uniform/mulMul+Initializer_56/random_uniform/RandomUniform!Initializer_56/random_uniform/sub*
T0
s
Initializer_56/random_uniformAdd!Initializer_56/random_uniform/mul!Initializer_56/random_uniform/min*
T0
á
	Assign_56Assign3mio_variable/duration_predict/dense/kernel/gradientInitializer_56/random_uniform*
use_locking(*
T0*F
_class<
:8loc:@mio_variable/duration_predict/dense/kernel/gradient*
validate_shape(
£
1mio_variable/duration_predict/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS**
	containerduration_predict/dense/bias*
shape:
£
1mio_variable/duration_predict/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:**
	containerduration_predict/dense/bias
F
Initializer_57/zerosConst*
valueB*    *
dtype0
Ô
	Assign_57Assign1mio_variable/duration_predict/dense/bias/gradientInitializer_57/zeros*
use_locking(*
T0*D
_class:
86loc:@mio_variable/duration_predict/dense/bias/gradient*
validate_shape(

duration_predict/dense/MatMulMatMulconcat3mio_variable/duration_predict/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 

duration_predict/dense/BiasAddBiasAddduration_predict/dense/MatMul1mio_variable/duration_predict/dense/bias/variable*
T0*
data_formatNHWC
S
&duration_predict/dense/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
|
$duration_predict/dense/LeakyRelu/mulMul&duration_predict/dense/LeakyRelu/alphaduration_predict/dense/BiasAdd*
T0
z
 duration_predict/dense/LeakyReluMaximum$duration_predict/dense/LeakyRelu/mulduration_predict/dense/BiasAdd*
T0
X
!duration_predict/dropout/IdentityIdentity duration_predict/dense/LeakyRelu*
T0
°
5mio_variable/duration_predict/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!duration_predict/dense_1/kernel*
shape:

°
5mio_variable/duration_predict/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!duration_predict/dense_1/kernel*
shape:

X
#Initializer_58/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_58/random_uniform/minConst*
valueB
 *   ¾*
dtype0
N
!Initializer_58/random_uniform/maxConst*
dtype0*
valueB
 *   >

+Initializer_58/random_uniform/RandomUniformRandomUniform#Initializer_58/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_58/random_uniform/subSub!Initializer_58/random_uniform/max!Initializer_58/random_uniform/min*
T0

!Initializer_58/random_uniform/mulMul+Initializer_58/random_uniform/RandomUniform!Initializer_58/random_uniform/sub*
T0
s
Initializer_58/random_uniformAdd!Initializer_58/random_uniform/mul!Initializer_58/random_uniform/min*
T0
å
	Assign_58Assign5mio_variable/duration_predict/dense_1/kernel/gradientInitializer_58/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/duration_predict/dense_1/kernel/gradient*
validate_shape(
§
3mio_variable/duration_predict/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*,
	containerduration_predict/dense_1/bias
§
3mio_variable/duration_predict/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerduration_predict/dense_1/bias*
shape:
F
Initializer_59/zerosConst*
dtype0*
valueB*    
Ø
	Assign_59Assign3mio_variable/duration_predict/dense_1/bias/gradientInitializer_59/zeros*
use_locking(*
T0*F
_class<
:8loc:@mio_variable/duration_predict/dense_1/bias/gradient*
validate_shape(
²
duration_predict/dense_1/MatMulMatMul!duration_predict/dropout/Identity5mio_variable/duration_predict/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
¡
 duration_predict/dense_1/BiasAddBiasAddduration_predict/dense_1/MatMul3mio_variable/duration_predict/dense_1/bias/variable*
data_formatNHWC*
T0
U
(duration_predict/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0

&duration_predict/dense_1/LeakyRelu/mulMul(duration_predict/dense_1/LeakyRelu/alpha duration_predict/dense_1/BiasAdd*
T0

"duration_predict/dense_1/LeakyReluMaximum&duration_predict/dense_1/LeakyRelu/mul duration_predict/dense_1/BiasAdd*
T0
\
#duration_predict/dropout_1/IdentityIdentity"duration_predict/dense_1/LeakyRelu*
T0
¯
5mio_variable/duration_predict/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*.
	container!duration_predict/dense_2/kernel
¯
5mio_variable/duration_predict/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!duration_predict/dense_2/kernel*
shape:	@
X
#Initializer_60/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_60/random_uniform/minConst*
valueB
 *ó5¾*
dtype0
N
!Initializer_60/random_uniform/maxConst*
valueB
 *ó5>*
dtype0

+Initializer_60/random_uniform/RandomUniformRandomUniform#Initializer_60/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_60/random_uniform/subSub!Initializer_60/random_uniform/max!Initializer_60/random_uniform/min*
T0

!Initializer_60/random_uniform/mulMul+Initializer_60/random_uniform/RandomUniform!Initializer_60/random_uniform/sub*
T0
s
Initializer_60/random_uniformAdd!Initializer_60/random_uniform/mul!Initializer_60/random_uniform/min*
T0
å
	Assign_60Assign5mio_variable/duration_predict/dense_2/kernel/gradientInitializer_60/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/duration_predict/dense_2/kernel/gradient*
validate_shape(
¦
3mio_variable/duration_predict/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerduration_predict/dense_2/bias*
shape:@
¦
3mio_variable/duration_predict/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerduration_predict/dense_2/bias*
shape:@
E
Initializer_61/zerosConst*
valueB@*    *
dtype0
Ø
	Assign_61Assign3mio_variable/duration_predict/dense_2/bias/gradientInitializer_61/zeros*
use_locking(*
T0*F
_class<
:8loc:@mio_variable/duration_predict/dense_2/bias/gradient*
validate_shape(
´
duration_predict/dense_2/MatMulMatMul#duration_predict/dropout_1/Identity5mio_variable/duration_predict/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0
¡
 duration_predict/dense_2/BiasAddBiasAddduration_predict/dense_2/MatMul3mio_variable/duration_predict/dense_2/bias/variable*
data_formatNHWC*
T0
U
(duration_predict/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0

&duration_predict/dense_2/LeakyRelu/mulMul(duration_predict/dense_2/LeakyRelu/alpha duration_predict/dense_2/BiasAdd*
T0

"duration_predict/dense_2/LeakyReluMaximum&duration_predict/dense_2/LeakyRelu/mul duration_predict/dense_2/BiasAdd*
T0
®
5mio_variable/duration_predict/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!duration_predict/dense_3/kernel*
shape
:@
®
5mio_variable/duration_predict/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*.
	container!duration_predict/dense_3/kernel
X
#Initializer_62/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_62/random_uniform/minConst*
valueB
 *¾*
dtype0
N
!Initializer_62/random_uniform/maxConst*
valueB
 *>*
dtype0

+Initializer_62/random_uniform/RandomUniformRandomUniform#Initializer_62/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_62/random_uniform/subSub!Initializer_62/random_uniform/max!Initializer_62/random_uniform/min*
T0

!Initializer_62/random_uniform/mulMul+Initializer_62/random_uniform/RandomUniform!Initializer_62/random_uniform/sub*
T0
s
Initializer_62/random_uniformAdd!Initializer_62/random_uniform/mul!Initializer_62/random_uniform/min*
T0
å
	Assign_62Assign5mio_variable/duration_predict/dense_3/kernel/gradientInitializer_62/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/duration_predict/dense_3/kernel/gradient*
validate_shape(
¦
3mio_variable/duration_predict/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerduration_predict/dense_3/bias*
shape:
¦
3mio_variable/duration_predict/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*,
	containerduration_predict/dense_3/bias
E
Initializer_63/zerosConst*
valueB*    *
dtype0
Ø
	Assign_63Assign3mio_variable/duration_predict/dense_3/bias/gradientInitializer_63/zeros*
use_locking(*
T0*F
_class<
:8loc:@mio_variable/duration_predict/dense_3/bias/gradient*
validate_shape(
³
duration_predict/dense_3/MatMulMatMul"duration_predict/dense_2/LeakyRelu5mio_variable/duration_predict/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 
¡
 duration_predict/dense_3/BiasAddBiasAddduration_predict/dense_3/MatMul3mio_variable/duration_predict/dense_3/bias/variable*
data_formatNHWC*
T0
P
duration_predict/dense_3/ReluRelu duration_predict/dense_3/BiasAdd*
T0
¾
<mio_variable/duration_pos_bias_predict/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*5
	container(&duration_pos_bias_predict/dense/kernel
¾
<mio_variable/duration_pos_bias_predict/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*5
	container(&duration_pos_bias_predict/dense/kernel
X
#Initializer_64/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_64/random_uniform/minConst*
dtype0*
valueB
 *²_¾
N
!Initializer_64/random_uniform/maxConst*
valueB
 *²_>*
dtype0

+Initializer_64/random_uniform/RandomUniformRandomUniform#Initializer_64/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_64/random_uniform/subSub!Initializer_64/random_uniform/max!Initializer_64/random_uniform/min*
T0

!Initializer_64/random_uniform/mulMul+Initializer_64/random_uniform/RandomUniform!Initializer_64/random_uniform/sub*
T0
s
Initializer_64/random_uniformAdd!Initializer_64/random_uniform/mul!Initializer_64/random_uniform/min*
T0
ó
	Assign_64Assign<mio_variable/duration_pos_bias_predict/dense/kernel/gradientInitializer_64/random_uniform*
T0*O
_classE
CAloc:@mio_variable/duration_pos_bias_predict/dense/kernel/gradient*
validate_shape(*
use_locking(
µ
:mio_variable/duration_pos_bias_predict/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$duration_pos_bias_predict/dense/bias*
shape:
µ
:mio_variable/duration_pos_bias_predict/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$duration_pos_bias_predict/dense/bias*
shape:
F
Initializer_65/zerosConst*
valueB*    *
dtype0
æ
	Assign_65Assign:mio_variable/duration_pos_bias_predict/dense/bias/gradientInitializer_65/zeros*
use_locking(*
T0*M
_classC
A?loc:@mio_variable/duration_pos_bias_predict/dense/bias/gradient*
validate_shape(
§
&duration_pos_bias_predict/dense/MatMulMatMulconcat_1<mio_variable/duration_pos_bias_predict/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
¶
'duration_pos_bias_predict/dense/BiasAddBiasAdd&duration_pos_bias_predict/dense/MatMul:mio_variable/duration_pos_bias_predict/dense/bias/variable*
T0*
data_formatNHWC
\
/duration_pos_bias_predict/dense/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0

-duration_pos_bias_predict/dense/LeakyRelu/mulMul/duration_pos_bias_predict/dense/LeakyRelu/alpha'duration_pos_bias_predict/dense/BiasAdd*
T0

)duration_pos_bias_predict/dense/LeakyReluMaximum-duration_pos_bias_predict/dense/LeakyRelu/mul'duration_pos_bias_predict/dense/BiasAdd*
T0
j
*duration_pos_bias_predict/dropout/IdentityIdentity)duration_pos_bias_predict/dense/LeakyRelu*
T0
Á
>mio_variable/duration_pos_bias_predict/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(duration_pos_bias_predict/dense_1/kernel*
shape:	@
Á
>mio_variable/duration_pos_bias_predict/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(duration_pos_bias_predict/dense_1/kernel*
shape:	@
X
#Initializer_66/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_66/random_uniform/minConst*
valueB
 *ó5¾*
dtype0
N
!Initializer_66/random_uniform/maxConst*
valueB
 *ó5>*
dtype0

+Initializer_66/random_uniform/RandomUniformRandomUniform#Initializer_66/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_66/random_uniform/subSub!Initializer_66/random_uniform/max!Initializer_66/random_uniform/min*
T0

!Initializer_66/random_uniform/mulMul+Initializer_66/random_uniform/RandomUniform!Initializer_66/random_uniform/sub*
T0
s
Initializer_66/random_uniformAdd!Initializer_66/random_uniform/mul!Initializer_66/random_uniform/min*
T0
÷
	Assign_66Assign>mio_variable/duration_pos_bias_predict/dense_1/kernel/gradientInitializer_66/random_uniform*
use_locking(*
T0*Q
_classG
ECloc:@mio_variable/duration_pos_bias_predict/dense_1/kernel/gradient*
validate_shape(
¸
<mio_variable/duration_pos_bias_predict/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&duration_pos_bias_predict/dense_1/bias*
shape:@
¸
<mio_variable/duration_pos_bias_predict/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&duration_pos_bias_predict/dense_1/bias*
shape:@
E
Initializer_67/zerosConst*
valueB@*    *
dtype0
ê
	Assign_67Assign<mio_variable/duration_pos_bias_predict/dense_1/bias/gradientInitializer_67/zeros*
use_locking(*
T0*O
_classE
CAloc:@mio_variable/duration_pos_bias_predict/dense_1/bias/gradient*
validate_shape(
Í
(duration_pos_bias_predict/dense_1/MatMulMatMul*duration_pos_bias_predict/dropout/Identity>mio_variable/duration_pos_bias_predict/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0
¼
)duration_pos_bias_predict/dense_1/BiasAddBiasAdd(duration_pos_bias_predict/dense_1/MatMul<mio_variable/duration_pos_bias_predict/dense_1/bias/variable*
T0*
data_formatNHWC
^
1duration_pos_bias_predict/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0

/duration_pos_bias_predict/dense_1/LeakyRelu/mulMul1duration_pos_bias_predict/dense_1/LeakyRelu/alpha)duration_pos_bias_predict/dense_1/BiasAdd*
T0

+duration_pos_bias_predict/dense_1/LeakyReluMaximum/duration_pos_bias_predict/dense_1/LeakyRelu/mul)duration_pos_bias_predict/dense_1/BiasAdd*
T0
À
>mio_variable/duration_pos_bias_predict/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(duration_pos_bias_predict/dense_2/kernel*
shape
:@
À
>mio_variable/duration_pos_bias_predict/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(duration_pos_bias_predict/dense_2/kernel*
shape
:@
X
#Initializer_68/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_68/random_uniform/minConst*
dtype0*
valueB
 *¾
N
!Initializer_68/random_uniform/maxConst*
valueB
 *>*
dtype0

+Initializer_68/random_uniform/RandomUniformRandomUniform#Initializer_68/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_68/random_uniform/subSub!Initializer_68/random_uniform/max!Initializer_68/random_uniform/min*
T0

!Initializer_68/random_uniform/mulMul+Initializer_68/random_uniform/RandomUniform!Initializer_68/random_uniform/sub*
T0
s
Initializer_68/random_uniformAdd!Initializer_68/random_uniform/mul!Initializer_68/random_uniform/min*
T0
÷
	Assign_68Assign>mio_variable/duration_pos_bias_predict/dense_2/kernel/gradientInitializer_68/random_uniform*
validate_shape(*
use_locking(*
T0*Q
_classG
ECloc:@mio_variable/duration_pos_bias_predict/dense_2/kernel/gradient
¸
<mio_variable/duration_pos_bias_predict/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&duration_pos_bias_predict/dense_2/bias*
shape:
¸
<mio_variable/duration_pos_bias_predict/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*5
	container(&duration_pos_bias_predict/dense_2/bias
E
Initializer_69/zerosConst*
valueB*    *
dtype0
ê
	Assign_69Assign<mio_variable/duration_pos_bias_predict/dense_2/bias/gradientInitializer_69/zeros*
T0*O
_classE
CAloc:@mio_variable/duration_pos_bias_predict/dense_2/bias/gradient*
validate_shape(*
use_locking(
Î
(duration_pos_bias_predict/dense_2/MatMulMatMul+duration_pos_bias_predict/dense_1/LeakyRelu>mio_variable/duration_pos_bias_predict/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0
¼
)duration_pos_bias_predict/dense_2/BiasAddBiasAdd(duration_pos_bias_predict/dense_2/MatMul<mio_variable/duration_pos_bias_predict/dense_2/bias/variable*
data_formatNHWC*
T0
b
&duration_pos_bias_predict/dense_2/ReluRelu)duration_pos_bias_predict/dense_2/BiasAdd*
T0

+mio_variable/hate_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerhate_xtr/dense/kernel*
shape:
°

+mio_variable/hate_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerhate_xtr/dense/kernel*
shape:
°
X
#Initializer_70/random_uniform/shapeConst*
valueB"°     *
dtype0
N
!Initializer_70/random_uniform/minConst*
valueB
 *dF£½*
dtype0
N
!Initializer_70/random_uniform/maxConst*
valueB
 *dF£=*
dtype0

+Initializer_70/random_uniform/RandomUniformRandomUniform#Initializer_70/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_70/random_uniform/subSub!Initializer_70/random_uniform/max!Initializer_70/random_uniform/min*
T0

!Initializer_70/random_uniform/mulMul+Initializer_70/random_uniform/RandomUniform!Initializer_70/random_uniform/sub*
T0
s
Initializer_70/random_uniformAdd!Initializer_70/random_uniform/mul!Initializer_70/random_uniform/min*
T0
Ñ
	Assign_70Assign+mio_variable/hate_xtr/dense/kernel/gradientInitializer_70/random_uniform*
use_locking(*
T0*>
_class4
20loc:@mio_variable/hate_xtr/dense/kernel/gradient*
validate_shape(

)mio_variable/hate_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*"
	containerhate_xtr/dense/bias*
shape:

)mio_variable/hate_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*"
	containerhate_xtr/dense/bias*
shape:
F
Initializer_71/zerosConst*
valueB*    *
dtype0
Ä
	Assign_71Assign)mio_variable/hate_xtr/dense/bias/gradientInitializer_71/zeros*
use_locking(*
T0*<
_class2
0.loc:@mio_variable/hate_xtr/dense/bias/gradient*
validate_shape(

hate_xtr/dense/MatMulMatMulconcat+mio_variable/hate_xtr/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0

hate_xtr/dense/BiasAddBiasAddhate_xtr/dense/MatMul)mio_variable/hate_xtr/dense/bias/variable*
T0*
data_formatNHWC
K
hate_xtr/dense/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
d
hate_xtr/dense/LeakyRelu/mulMulhate_xtr/dense/LeakyRelu/alphahate_xtr/dense/BiasAdd*
T0
b
hate_xtr/dense/LeakyReluMaximumhate_xtr/dense/LeakyRelu/mulhate_xtr/dense/BiasAdd*
T0
 
-mio_variable/hate_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerhate_xtr/dense_1/kernel*
shape:

 
-mio_variable/hate_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerhate_xtr/dense_1/kernel*
shape:

X
#Initializer_72/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_72/random_uniform/minConst*
valueB
 *   ¾*
dtype0
N
!Initializer_72/random_uniform/maxConst*
valueB
 *   >*
dtype0

+Initializer_72/random_uniform/RandomUniformRandomUniform#Initializer_72/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_72/random_uniform/subSub!Initializer_72/random_uniform/max!Initializer_72/random_uniform/min*
T0

!Initializer_72/random_uniform/mulMul+Initializer_72/random_uniform/RandomUniform!Initializer_72/random_uniform/sub*
T0
s
Initializer_72/random_uniformAdd!Initializer_72/random_uniform/mul!Initializer_72/random_uniform/min*
T0
Õ
	Assign_72Assign-mio_variable/hate_xtr/dense_1/kernel/gradientInitializer_72/random_uniform*@
_class6
42loc:@mio_variable/hate_xtr/dense_1/kernel/gradient*
validate_shape(*
use_locking(*
T0

+mio_variable/hate_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containerhate_xtr/dense_1/bias

+mio_variable/hate_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containerhate_xtr/dense_1/bias
F
Initializer_73/zerosConst*
valueB*    *
dtype0
È
	Assign_73Assign+mio_variable/hate_xtr/dense_1/bias/gradientInitializer_73/zeros*
use_locking(*
T0*>
_class4
20loc:@mio_variable/hate_xtr/dense_1/bias/gradient*
validate_shape(

hate_xtr/dense_1/MatMulMatMulhate_xtr/dense/LeakyRelu-mio_variable/hate_xtr/dense_1/kernel/variable*
transpose_b( *
T0*
transpose_a( 

hate_xtr/dense_1/BiasAddBiasAddhate_xtr/dense_1/MatMul+mio_variable/hate_xtr/dense_1/bias/variable*
T0*
data_formatNHWC
M
 hate_xtr/dense_1/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍÌL>
j
hate_xtr/dense_1/LeakyRelu/mulMul hate_xtr/dense_1/LeakyRelu/alphahate_xtr/dense_1/BiasAdd*
T0
h
hate_xtr/dense_1/LeakyReluMaximumhate_xtr/dense_1/LeakyRelu/mulhate_xtr/dense_1/BiasAdd*
T0

-mio_variable/hate_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerhate_xtr/dense_2/kernel*
shape:	@

-mio_variable/hate_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerhate_xtr/dense_2/kernel*
shape:	@
X
#Initializer_74/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_74/random_uniform/minConst*
valueB
 *ó5¾*
dtype0
N
!Initializer_74/random_uniform/maxConst*
dtype0*
valueB
 *ó5>

+Initializer_74/random_uniform/RandomUniformRandomUniform#Initializer_74/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_74/random_uniform/subSub!Initializer_74/random_uniform/max!Initializer_74/random_uniform/min*
T0

!Initializer_74/random_uniform/mulMul+Initializer_74/random_uniform/RandomUniform!Initializer_74/random_uniform/sub*
T0
s
Initializer_74/random_uniformAdd!Initializer_74/random_uniform/mul!Initializer_74/random_uniform/min*
T0
Õ
	Assign_74Assign-mio_variable/hate_xtr/dense_2/kernel/gradientInitializer_74/random_uniform*
use_locking(*
T0*@
_class6
42loc:@mio_variable/hate_xtr/dense_2/kernel/gradient*
validate_shape(

+mio_variable/hate_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerhate_xtr/dense_2/bias*
shape:@

+mio_variable/hate_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerhate_xtr/dense_2/bias*
shape:@
E
Initializer_75/zerosConst*
valueB@*    *
dtype0
È
	Assign_75Assign+mio_variable/hate_xtr/dense_2/bias/gradientInitializer_75/zeros*
T0*>
_class4
20loc:@mio_variable/hate_xtr/dense_2/bias/gradient*
validate_shape(*
use_locking(

hate_xtr/dense_2/MatMulMatMulhate_xtr/dense_1/LeakyRelu-mio_variable/hate_xtr/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0

hate_xtr/dense_2/BiasAddBiasAddhate_xtr/dense_2/MatMul+mio_variable/hate_xtr/dense_2/bias/variable*
data_formatNHWC*
T0
M
 hate_xtr/dense_2/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍÌL>
j
hate_xtr/dense_2/LeakyRelu/mulMul hate_xtr/dense_2/LeakyRelu/alphahate_xtr/dense_2/BiasAdd*
T0
h
hate_xtr/dense_2/LeakyReluMaximumhate_xtr/dense_2/LeakyRelu/mulhate_xtr/dense_2/BiasAdd*
T0

-mio_variable/hate_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerhate_xtr/dense_3/kernel*
shape
:@

-mio_variable/hate_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*&
	containerhate_xtr/dense_3/kernel
X
#Initializer_76/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_76/random_uniform/minConst*
valueB
 *¾*
dtype0
N
!Initializer_76/random_uniform/maxConst*
valueB
 *>*
dtype0

+Initializer_76/random_uniform/RandomUniformRandomUniform#Initializer_76/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_76/random_uniform/subSub!Initializer_76/random_uniform/max!Initializer_76/random_uniform/min*
T0

!Initializer_76/random_uniform/mulMul+Initializer_76/random_uniform/RandomUniform!Initializer_76/random_uniform/sub*
T0
s
Initializer_76/random_uniformAdd!Initializer_76/random_uniform/mul!Initializer_76/random_uniform/min*
T0
Õ
	Assign_76Assign-mio_variable/hate_xtr/dense_3/kernel/gradientInitializer_76/random_uniform*
use_locking(*
T0*@
_class6
42loc:@mio_variable/hate_xtr/dense_3/kernel/gradient*
validate_shape(

+mio_variable/hate_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerhate_xtr/dense_3/bias*
shape:

+mio_variable/hate_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerhate_xtr/dense_3/bias*
shape:
E
Initializer_77/zerosConst*
valueB*    *
dtype0
È
	Assign_77Assign+mio_variable/hate_xtr/dense_3/bias/gradientInitializer_77/zeros*
use_locking(*
T0*>
_class4
20loc:@mio_variable/hate_xtr/dense_3/bias/gradient*
validate_shape(

hate_xtr/dense_3/MatMulMatMulhate_xtr/dense_2/LeakyRelu-mio_variable/hate_xtr/dense_3/kernel/variable*
transpose_a( *
transpose_b( *
T0

hate_xtr/dense_3/BiasAddBiasAddhate_xtr/dense_3/MatMul+mio_variable/hate_xtr/dense_3/bias/variable*
T0*
data_formatNHWC
F
hate_xtr/dense_3/SigmoidSigmoidhate_xtr/dense_3/BiasAdd*
T0
 
-mio_variable/report_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*&
	containerreport_xtr/dense/kernel
 
-mio_variable/report_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreport_xtr/dense/kernel*
shape:
°
X
#Initializer_78/random_uniform/shapeConst*
valueB"°     *
dtype0
N
!Initializer_78/random_uniform/minConst*
valueB
 *dF£½*
dtype0
N
!Initializer_78/random_uniform/maxConst*
valueB
 *dF£=*
dtype0

+Initializer_78/random_uniform/RandomUniformRandomUniform#Initializer_78/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_78/random_uniform/subSub!Initializer_78/random_uniform/max!Initializer_78/random_uniform/min*
T0

!Initializer_78/random_uniform/mulMul+Initializer_78/random_uniform/RandomUniform!Initializer_78/random_uniform/sub*
T0
s
Initializer_78/random_uniformAdd!Initializer_78/random_uniform/mul!Initializer_78/random_uniform/min*
T0
Õ
	Assign_78Assign-mio_variable/report_xtr/dense/kernel/gradientInitializer_78/random_uniform*
use_locking(*
T0*@
_class6
42loc:@mio_variable/report_xtr/dense/kernel/gradient*
validate_shape(

+mio_variable/report_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerreport_xtr/dense/bias*
shape:

+mio_variable/report_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerreport_xtr/dense/bias*
shape:
F
Initializer_79/zerosConst*
valueB*    *
dtype0
È
	Assign_79Assign+mio_variable/report_xtr/dense/bias/gradientInitializer_79/zeros*
T0*>
_class4
20loc:@mio_variable/report_xtr/dense/bias/gradient*
validate_shape(*
use_locking(

report_xtr/dense/MatMulMatMulconcat-mio_variable/report_xtr/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0

report_xtr/dense/BiasAddBiasAddreport_xtr/dense/MatMul+mio_variable/report_xtr/dense/bias/variable*
data_formatNHWC*
T0
M
 report_xtr/dense/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍÌL>
j
report_xtr/dense/LeakyRelu/mulMul report_xtr/dense/LeakyRelu/alphareport_xtr/dense/BiasAdd*
T0
h
report_xtr/dense/LeakyReluMaximumreport_xtr/dense/LeakyRelu/mulreport_xtr/dense/BiasAdd*
T0
¤
/mio_variable/report_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*(
	containerreport_xtr/dense_1/kernel
¤
/mio_variable/report_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*(
	containerreport_xtr/dense_1/kernel
X
#Initializer_80/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_80/random_uniform/minConst*
valueB
 *   ¾*
dtype0
N
!Initializer_80/random_uniform/maxConst*
valueB
 *   >*
dtype0

+Initializer_80/random_uniform/RandomUniformRandomUniform#Initializer_80/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_80/random_uniform/subSub!Initializer_80/random_uniform/max!Initializer_80/random_uniform/min*
T0

!Initializer_80/random_uniform/mulMul+Initializer_80/random_uniform/RandomUniform!Initializer_80/random_uniform/sub*
T0
s
Initializer_80/random_uniformAdd!Initializer_80/random_uniform/mul!Initializer_80/random_uniform/min*
T0
Ù
	Assign_80Assign/mio_variable/report_xtr/dense_1/kernel/gradientInitializer_80/random_uniform*
use_locking(*
T0*B
_class8
64loc:@mio_variable/report_xtr/dense_1/kernel/gradient*
validate_shape(

-mio_variable/report_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreport_xtr/dense_1/bias*
shape:

-mio_variable/report_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreport_xtr/dense_1/bias*
shape:
F
Initializer_81/zerosConst*
valueB*    *
dtype0
Ì
	Assign_81Assign-mio_variable/report_xtr/dense_1/bias/gradientInitializer_81/zeros*
use_locking(*
T0*@
_class6
42loc:@mio_variable/report_xtr/dense_1/bias/gradient*
validate_shape(

report_xtr/dense_1/MatMulMatMulreport_xtr/dense/LeakyRelu/mio_variable/report_xtr/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 

report_xtr/dense_1/BiasAddBiasAddreport_xtr/dense_1/MatMul-mio_variable/report_xtr/dense_1/bias/variable*
T0*
data_formatNHWC
O
"report_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
p
 report_xtr/dense_1/LeakyRelu/mulMul"report_xtr/dense_1/LeakyRelu/alphareport_xtr/dense_1/BiasAdd*
T0
n
report_xtr/dense_1/LeakyReluMaximum report_xtr/dense_1/LeakyRelu/mulreport_xtr/dense_1/BiasAdd*
T0
£
/mio_variable/report_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerreport_xtr/dense_2/kernel*
shape:	@
£
/mio_variable/report_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerreport_xtr/dense_2/kernel*
shape:	@
X
#Initializer_82/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_82/random_uniform/minConst*
valueB
 *ó5¾*
dtype0
N
!Initializer_82/random_uniform/maxConst*
dtype0*
valueB
 *ó5>

+Initializer_82/random_uniform/RandomUniformRandomUniform#Initializer_82/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_82/random_uniform/subSub!Initializer_82/random_uniform/max!Initializer_82/random_uniform/min*
T0

!Initializer_82/random_uniform/mulMul+Initializer_82/random_uniform/RandomUniform!Initializer_82/random_uniform/sub*
T0
s
Initializer_82/random_uniformAdd!Initializer_82/random_uniform/mul!Initializer_82/random_uniform/min*
T0
Ù
	Assign_82Assign/mio_variable/report_xtr/dense_2/kernel/gradientInitializer_82/random_uniform*
use_locking(*
T0*B
_class8
64loc:@mio_variable/report_xtr/dense_2/kernel/gradient*
validate_shape(

-mio_variable/report_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreport_xtr/dense_2/bias*
shape:@

-mio_variable/report_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreport_xtr/dense_2/bias*
shape:@
E
Initializer_83/zerosConst*
dtype0*
valueB@*    
Ì
	Assign_83Assign-mio_variable/report_xtr/dense_2/bias/gradientInitializer_83/zeros*
T0*@
_class6
42loc:@mio_variable/report_xtr/dense_2/bias/gradient*
validate_shape(*
use_locking(
¡
report_xtr/dense_2/MatMulMatMulreport_xtr/dense_1/LeakyRelu/mio_variable/report_xtr/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0

report_xtr/dense_2/BiasAddBiasAddreport_xtr/dense_2/MatMul-mio_variable/report_xtr/dense_2/bias/variable*
T0*
data_formatNHWC
O
"report_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
p
 report_xtr/dense_2/LeakyRelu/mulMul"report_xtr/dense_2/LeakyRelu/alphareport_xtr/dense_2/BiasAdd*
T0
n
report_xtr/dense_2/LeakyReluMaximum report_xtr/dense_2/LeakyRelu/mulreport_xtr/dense_2/BiasAdd*
T0
¢
/mio_variable/report_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerreport_xtr/dense_3/kernel*
shape
:@
¢
/mio_variable/report_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerreport_xtr/dense_3/kernel*
shape
:@
X
#Initializer_84/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_84/random_uniform/minConst*
valueB
 *¾*
dtype0
N
!Initializer_84/random_uniform/maxConst*
valueB
 *>*
dtype0

+Initializer_84/random_uniform/RandomUniformRandomUniform#Initializer_84/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_84/random_uniform/subSub!Initializer_84/random_uniform/max!Initializer_84/random_uniform/min*
T0

!Initializer_84/random_uniform/mulMul+Initializer_84/random_uniform/RandomUniform!Initializer_84/random_uniform/sub*
T0
s
Initializer_84/random_uniformAdd!Initializer_84/random_uniform/mul!Initializer_84/random_uniform/min*
T0
Ù
	Assign_84Assign/mio_variable/report_xtr/dense_3/kernel/gradientInitializer_84/random_uniform*
validate_shape(*
use_locking(*
T0*B
_class8
64loc:@mio_variable/report_xtr/dense_3/kernel/gradient

-mio_variable/report_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreport_xtr/dense_3/bias*
shape:

-mio_variable/report_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreport_xtr/dense_3/bias*
shape:
E
Initializer_85/zerosConst*
valueB*    *
dtype0
Ì
	Assign_85Assign-mio_variable/report_xtr/dense_3/bias/gradientInitializer_85/zeros*
use_locking(*
T0*@
_class6
42loc:@mio_variable/report_xtr/dense_3/bias/gradient*
validate_shape(
¡
report_xtr/dense_3/MatMulMatMulreport_xtr/dense_2/LeakyRelu/mio_variable/report_xtr/dense_3/kernel/variable*
transpose_b( *
T0*
transpose_a( 

report_xtr/dense_3/BiasAddBiasAddreport_xtr/dense_3/MatMul-mio_variable/report_xtr/dense_3/bias/variable*
T0*
data_formatNHWC
J
report_xtr/dense_3/SigmoidSigmoidreport_xtr/dense_3/BiasAdd*
T0
®
4mio_variable/page_time_predict/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*-
	container page_time_predict/dense/kernel
®
4mio_variable/page_time_predict/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*-
	container page_time_predict/dense/kernel
X
#Initializer_86/random_uniform/shapeConst*
dtype0*
valueB"°     
N
!Initializer_86/random_uniform/minConst*
dtype0*
valueB
 *dF£½
N
!Initializer_86/random_uniform/maxConst*
valueB
 *dF£=*
dtype0

+Initializer_86/random_uniform/RandomUniformRandomUniform#Initializer_86/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_86/random_uniform/subSub!Initializer_86/random_uniform/max!Initializer_86/random_uniform/min*
T0

!Initializer_86/random_uniform/mulMul+Initializer_86/random_uniform/RandomUniform!Initializer_86/random_uniform/sub*
T0
s
Initializer_86/random_uniformAdd!Initializer_86/random_uniform/mul!Initializer_86/random_uniform/min*
T0
ã
	Assign_86Assign4mio_variable/page_time_predict/dense/kernel/gradientInitializer_86/random_uniform*
T0*G
_class=
;9loc:@mio_variable/page_time_predict/dense/kernel/gradient*
validate_shape(*
use_locking(
¥
2mio_variable/page_time_predict/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*+
	containerpage_time_predict/dense/bias*
shape:
¥
2mio_variable/page_time_predict/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*+
	containerpage_time_predict/dense/bias
F
Initializer_87/zerosConst*
valueB*    *
dtype0
Ö
	Assign_87Assign2mio_variable/page_time_predict/dense/bias/gradientInitializer_87/zeros*
use_locking(*
T0*E
_class;
97loc:@mio_variable/page_time_predict/dense/bias/gradient*
validate_shape(

page_time_predict/dense/MatMulMatMulconcat4mio_variable/page_time_predict/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 

page_time_predict/dense/BiasAddBiasAddpage_time_predict/dense/MatMul2mio_variable/page_time_predict/dense/bias/variable*
T0*
data_formatNHWC
T
'page_time_predict/dense/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0

%page_time_predict/dense/LeakyRelu/mulMul'page_time_predict/dense/LeakyRelu/alphapage_time_predict/dense/BiasAdd*
T0
}
!page_time_predict/dense/LeakyReluMaximum%page_time_predict/dense/LeakyRelu/mulpage_time_predict/dense/BiasAdd*
T0
Z
"page_time_predict/dropout/IdentityIdentity!page_time_predict/dense/LeakyRelu*
T0
²
6mio_variable/page_time_predict/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*/
	container" page_time_predict/dense_1/kernel
²
6mio_variable/page_time_predict/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" page_time_predict/dense_1/kernel*
shape:

X
#Initializer_88/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_88/random_uniform/minConst*
valueB
 *   ¾*
dtype0
N
!Initializer_88/random_uniform/maxConst*
valueB
 *   >*
dtype0

+Initializer_88/random_uniform/RandomUniformRandomUniform#Initializer_88/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_88/random_uniform/subSub!Initializer_88/random_uniform/max!Initializer_88/random_uniform/min*
T0

!Initializer_88/random_uniform/mulMul+Initializer_88/random_uniform/RandomUniform!Initializer_88/random_uniform/sub*
T0
s
Initializer_88/random_uniformAdd!Initializer_88/random_uniform/mul!Initializer_88/random_uniform/min*
T0
ç
	Assign_88Assign6mio_variable/page_time_predict/dense_1/kernel/gradientInitializer_88/random_uniform*
validate_shape(*
use_locking(*
T0*I
_class?
=;loc:@mio_variable/page_time_predict/dense_1/kernel/gradient
©
4mio_variable/page_time_predict/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*-
	container page_time_predict/dense_1/bias*
shape:
©
4mio_variable/page_time_predict/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*-
	container page_time_predict/dense_1/bias*
shape:
F
Initializer_89/zerosConst*
valueB*    *
dtype0
Ú
	Assign_89Assign4mio_variable/page_time_predict/dense_1/bias/gradientInitializer_89/zeros*
validate_shape(*
use_locking(*
T0*G
_class=
;9loc:@mio_variable/page_time_predict/dense_1/bias/gradient
µ
 page_time_predict/dense_1/MatMulMatMul"page_time_predict/dropout/Identity6mio_variable/page_time_predict/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0
¤
!page_time_predict/dense_1/BiasAddBiasAdd page_time_predict/dense_1/MatMul4mio_variable/page_time_predict/dense_1/bias/variable*
T0*
data_formatNHWC
V
)page_time_predict/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0

'page_time_predict/dense_1/LeakyRelu/mulMul)page_time_predict/dense_1/LeakyRelu/alpha!page_time_predict/dense_1/BiasAdd*
T0

#page_time_predict/dense_1/LeakyReluMaximum'page_time_predict/dense_1/LeakyRelu/mul!page_time_predict/dense_1/BiasAdd*
T0
^
$page_time_predict/dropout_1/IdentityIdentity#page_time_predict/dense_1/LeakyRelu*
T0
±
6mio_variable/page_time_predict/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" page_time_predict/dense_2/kernel*
shape:	@
±
6mio_variable/page_time_predict/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" page_time_predict/dense_2/kernel*
shape:	@
X
#Initializer_90/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_90/random_uniform/minConst*
valueB
 *ó5¾*
dtype0
N
!Initializer_90/random_uniform/maxConst*
valueB
 *ó5>*
dtype0

+Initializer_90/random_uniform/RandomUniformRandomUniform#Initializer_90/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_90/random_uniform/subSub!Initializer_90/random_uniform/max!Initializer_90/random_uniform/min*
T0

!Initializer_90/random_uniform/mulMul+Initializer_90/random_uniform/RandomUniform!Initializer_90/random_uniform/sub*
T0
s
Initializer_90/random_uniformAdd!Initializer_90/random_uniform/mul!Initializer_90/random_uniform/min*
T0
ç
	Assign_90Assign6mio_variable/page_time_predict/dense_2/kernel/gradientInitializer_90/random_uniform*
use_locking(*
T0*I
_class?
=;loc:@mio_variable/page_time_predict/dense_2/kernel/gradient*
validate_shape(
¨
4mio_variable/page_time_predict/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*-
	container page_time_predict/dense_2/bias
¨
4mio_variable/page_time_predict/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*-
	container page_time_predict/dense_2/bias
E
Initializer_91/zerosConst*
dtype0*
valueB@*    
Ú
	Assign_91Assign4mio_variable/page_time_predict/dense_2/bias/gradientInitializer_91/zeros*
use_locking(*
T0*G
_class=
;9loc:@mio_variable/page_time_predict/dense_2/bias/gradient*
validate_shape(
·
 page_time_predict/dense_2/MatMulMatMul$page_time_predict/dropout_1/Identity6mio_variable/page_time_predict/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0
¤
!page_time_predict/dense_2/BiasAddBiasAdd page_time_predict/dense_2/MatMul4mio_variable/page_time_predict/dense_2/bias/variable*
T0*
data_formatNHWC
V
)page_time_predict/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0

'page_time_predict/dense_2/LeakyRelu/mulMul)page_time_predict/dense_2/LeakyRelu/alpha!page_time_predict/dense_2/BiasAdd*
T0

#page_time_predict/dense_2/LeakyReluMaximum'page_time_predict/dense_2/LeakyRelu/mul!page_time_predict/dense_2/BiasAdd*
T0
°
6mio_variable/page_time_predict/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" page_time_predict/dense_3/kernel*
shape
:@
°
6mio_variable/page_time_predict/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" page_time_predict/dense_3/kernel*
shape
:@
X
#Initializer_92/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_92/random_uniform/minConst*
valueB
 *¾*
dtype0
N
!Initializer_92/random_uniform/maxConst*
valueB
 *>*
dtype0

+Initializer_92/random_uniform/RandomUniformRandomUniform#Initializer_92/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_92/random_uniform/subSub!Initializer_92/random_uniform/max!Initializer_92/random_uniform/min*
T0

!Initializer_92/random_uniform/mulMul+Initializer_92/random_uniform/RandomUniform!Initializer_92/random_uniform/sub*
T0
s
Initializer_92/random_uniformAdd!Initializer_92/random_uniform/mul!Initializer_92/random_uniform/min*
T0
ç
	Assign_92Assign6mio_variable/page_time_predict/dense_3/kernel/gradientInitializer_92/random_uniform*
validate_shape(*
use_locking(*
T0*I
_class?
=;loc:@mio_variable/page_time_predict/dense_3/kernel/gradient
¨
4mio_variable/page_time_predict/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*-
	container page_time_predict/dense_3/bias*
shape:
¨
4mio_variable/page_time_predict/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*-
	container page_time_predict/dense_3/bias*
shape:
E
Initializer_93/zerosConst*
valueB*    *
dtype0
Ú
	Assign_93Assign4mio_variable/page_time_predict/dense_3/bias/gradientInitializer_93/zeros*
use_locking(*
T0*G
_class=
;9loc:@mio_variable/page_time_predict/dense_3/bias/gradient*
validate_shape(
¶
 page_time_predict/dense_3/MatMulMatMul#page_time_predict/dense_2/LeakyRelu6mio_variable/page_time_predict/dense_3/kernel/variable*
transpose_a( *
transpose_b( *
T0
¤
!page_time_predict/dense_3/BiasAddBiasAdd page_time_predict/dense_3/MatMul4mio_variable/page_time_predict/dense_3/bias/variable*
T0*
data_formatNHWC
X
!page_time_predict/dense_3/SigmoidSigmoid!page_time_predict/dense_3/BiasAdd*
T0
¦
0mio_variable/effe_read_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*)
	containereffe_read_xtr/dense/kernel*
shape:
°
¦
0mio_variable/effe_read_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*)
	containereffe_read_xtr/dense/kernel
X
#Initializer_94/random_uniform/shapeConst*
valueB"°     *
dtype0
N
!Initializer_94/random_uniform/minConst*
valueB
 *dF£½*
dtype0
N
!Initializer_94/random_uniform/maxConst*
dtype0*
valueB
 *dF£=

+Initializer_94/random_uniform/RandomUniformRandomUniform#Initializer_94/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_94/random_uniform/subSub!Initializer_94/random_uniform/max!Initializer_94/random_uniform/min*
T0

!Initializer_94/random_uniform/mulMul+Initializer_94/random_uniform/RandomUniform!Initializer_94/random_uniform/sub*
T0
s
Initializer_94/random_uniformAdd!Initializer_94/random_uniform/mul!Initializer_94/random_uniform/min*
T0
Û
	Assign_94Assign0mio_variable/effe_read_xtr/dense/kernel/gradientInitializer_94/random_uniform*
use_locking(*
T0*C
_class9
75loc:@mio_variable/effe_read_xtr/dense/kernel/gradient*
validate_shape(

.mio_variable/effe_read_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containereffe_read_xtr/dense/bias*
shape:

.mio_variable/effe_read_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containereffe_read_xtr/dense/bias*
shape:
F
Initializer_95/zerosConst*
dtype0*
valueB*    
Î
	Assign_95Assign.mio_variable/effe_read_xtr/dense/bias/gradientInitializer_95/zeros*
use_locking(*
T0*A
_class7
53loc:@mio_variable/effe_read_xtr/dense/bias/gradient*
validate_shape(

effe_read_xtr/dense/MatMulMatMulconcat0mio_variable/effe_read_xtr/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0

effe_read_xtr/dense/BiasAddBiasAddeffe_read_xtr/dense/MatMul.mio_variable/effe_read_xtr/dense/bias/variable*
T0*
data_formatNHWC
P
#effe_read_xtr/dense/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
s
!effe_read_xtr/dense/LeakyRelu/mulMul#effe_read_xtr/dense/LeakyRelu/alphaeffe_read_xtr/dense/BiasAdd*
T0
q
effe_read_xtr/dense/LeakyReluMaximum!effe_read_xtr/dense/LeakyRelu/muleffe_read_xtr/dense/BiasAdd*
T0
R
effe_read_xtr/dropout/IdentityIdentityeffe_read_xtr/dense/LeakyRelu*
T0
ª
2mio_variable/effe_read_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*+
	containereffe_read_xtr/dense_1/kernel
ª
2mio_variable/effe_read_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*+
	containereffe_read_xtr/dense_1/kernel
X
#Initializer_96/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_96/random_uniform/minConst*
valueB
 *   ¾*
dtype0
N
!Initializer_96/random_uniform/maxConst*
valueB
 *   >*
dtype0

+Initializer_96/random_uniform/RandomUniformRandomUniform#Initializer_96/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_96/random_uniform/subSub!Initializer_96/random_uniform/max!Initializer_96/random_uniform/min*
T0

!Initializer_96/random_uniform/mulMul+Initializer_96/random_uniform/RandomUniform!Initializer_96/random_uniform/sub*
T0
s
Initializer_96/random_uniformAdd!Initializer_96/random_uniform/mul!Initializer_96/random_uniform/min*
T0
ß
	Assign_96Assign2mio_variable/effe_read_xtr/dense_1/kernel/gradientInitializer_96/random_uniform*
validate_shape(*
use_locking(*
T0*E
_class;
97loc:@mio_variable/effe_read_xtr/dense_1/kernel/gradient
¡
0mio_variable/effe_read_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*)
	containereffe_read_xtr/dense_1/bias*
shape:
¡
0mio_variable/effe_read_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*)
	containereffe_read_xtr/dense_1/bias
F
Initializer_97/zerosConst*
valueB*    *
dtype0
Ò
	Assign_97Assign0mio_variable/effe_read_xtr/dense_1/bias/gradientInitializer_97/zeros*
validate_shape(*
use_locking(*
T0*C
_class9
75loc:@mio_variable/effe_read_xtr/dense_1/bias/gradient
©
effe_read_xtr/dense_1/MatMulMatMuleffe_read_xtr/dropout/Identity2mio_variable/effe_read_xtr/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 

effe_read_xtr/dense_1/BiasAddBiasAddeffe_read_xtr/dense_1/MatMul0mio_variable/effe_read_xtr/dense_1/bias/variable*
T0*
data_formatNHWC
R
%effe_read_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
y
#effe_read_xtr/dense_1/LeakyRelu/mulMul%effe_read_xtr/dense_1/LeakyRelu/alphaeffe_read_xtr/dense_1/BiasAdd*
T0
w
effe_read_xtr/dense_1/LeakyReluMaximum#effe_read_xtr/dense_1/LeakyRelu/muleffe_read_xtr/dense_1/BiasAdd*
T0
V
 effe_read_xtr/dropout_1/IdentityIdentityeffe_read_xtr/dense_1/LeakyRelu*
T0
©
2mio_variable/effe_read_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*+
	containereffe_read_xtr/dense_2/kernel
©
2mio_variable/effe_read_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*+
	containereffe_read_xtr/dense_2/kernel
X
#Initializer_98/random_uniform/shapeConst*
dtype0*
valueB"   @   
N
!Initializer_98/random_uniform/minConst*
valueB
 *ó5¾*
dtype0
N
!Initializer_98/random_uniform/maxConst*
valueB
 *ó5>*
dtype0

+Initializer_98/random_uniform/RandomUniformRandomUniform#Initializer_98/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_98/random_uniform/subSub!Initializer_98/random_uniform/max!Initializer_98/random_uniform/min*
T0

!Initializer_98/random_uniform/mulMul+Initializer_98/random_uniform/RandomUniform!Initializer_98/random_uniform/sub*
T0
s
Initializer_98/random_uniformAdd!Initializer_98/random_uniform/mul!Initializer_98/random_uniform/min*
T0
ß
	Assign_98Assign2mio_variable/effe_read_xtr/dense_2/kernel/gradientInitializer_98/random_uniform*
T0*E
_class;
97loc:@mio_variable/effe_read_xtr/dense_2/kernel/gradient*
validate_shape(*
use_locking(
 
0mio_variable/effe_read_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*)
	containereffe_read_xtr/dense_2/bias*
shape:@
 
0mio_variable/effe_read_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*)
	containereffe_read_xtr/dense_2/bias
E
Initializer_99/zerosConst*
valueB@*    *
dtype0
Ò
	Assign_99Assign0mio_variable/effe_read_xtr/dense_2/bias/gradientInitializer_99/zeros*
validate_shape(*
use_locking(*
T0*C
_class9
75loc:@mio_variable/effe_read_xtr/dense_2/bias/gradient
«
effe_read_xtr/dense_2/MatMulMatMul effe_read_xtr/dropout_1/Identity2mio_variable/effe_read_xtr/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0

effe_read_xtr/dense_2/BiasAddBiasAddeffe_read_xtr/dense_2/MatMul0mio_variable/effe_read_xtr/dense_2/bias/variable*
T0*
data_formatNHWC
R
%effe_read_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
y
#effe_read_xtr/dense_2/LeakyRelu/mulMul%effe_read_xtr/dense_2/LeakyRelu/alphaeffe_read_xtr/dense_2/BiasAdd*
T0
w
effe_read_xtr/dense_2/LeakyReluMaximum#effe_read_xtr/dense_2/LeakyRelu/muleffe_read_xtr/dense_2/BiasAdd*
T0
¨
2mio_variable/effe_read_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*+
	containereffe_read_xtr/dense_3/kernel
¨
2mio_variable/effe_read_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*+
	containereffe_read_xtr/dense_3/kernel*
shape
:@
Y
$Initializer_100/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_100/random_uniform/minConst*
dtype0*
valueB
 *¾
O
"Initializer_100/random_uniform/maxConst*
valueB
 *>*
dtype0

,Initializer_100/random_uniform/RandomUniformRandomUniform$Initializer_100/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_100/random_uniform/subSub"Initializer_100/random_uniform/max"Initializer_100/random_uniform/min*
T0

"Initializer_100/random_uniform/mulMul,Initializer_100/random_uniform/RandomUniform"Initializer_100/random_uniform/sub*
T0
v
Initializer_100/random_uniformAdd"Initializer_100/random_uniform/mul"Initializer_100/random_uniform/min*
T0
á

Assign_100Assign2mio_variable/effe_read_xtr/dense_3/kernel/gradientInitializer_100/random_uniform*
use_locking(*
T0*E
_class;
97loc:@mio_variable/effe_read_xtr/dense_3/kernel/gradient*
validate_shape(
 
0mio_variable/effe_read_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*)
	containereffe_read_xtr/dense_3/bias*
shape:
 
0mio_variable/effe_read_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*)
	containereffe_read_xtr/dense_3/bias*
shape:
F
Initializer_101/zerosConst*
valueB*    *
dtype0
Ô

Assign_101Assign0mio_variable/effe_read_xtr/dense_3/bias/gradientInitializer_101/zeros*
T0*C
_class9
75loc:@mio_variable/effe_read_xtr/dense_3/bias/gradient*
validate_shape(*
use_locking(
ª
effe_read_xtr/dense_3/MatMulMatMuleffe_read_xtr/dense_2/LeakyRelu2mio_variable/effe_read_xtr/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 

effe_read_xtr/dense_3/BiasAddBiasAddeffe_read_xtr/dense_3/MatMul0mio_variable/effe_read_xtr/dense_3/bias/variable*
data_formatNHWC*
T0
P
effe_read_xtr/dense_3/SigmoidSigmoideffe_read_xtr/dense_3/BiasAdd*
T0
¤
/mio_variable/readtime_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerreadtime_xtr/dense/kernel*
shape:
°
¤
/mio_variable/readtime_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerreadtime_xtr/dense/kernel*
shape:
°
Y
$Initializer_102/random_uniform/shapeConst*
valueB"°     *
dtype0
O
"Initializer_102/random_uniform/minConst*
valueB
 *dF£½*
dtype0
O
"Initializer_102/random_uniform/maxConst*
valueB
 *dF£=*
dtype0

,Initializer_102/random_uniform/RandomUniformRandomUniform$Initializer_102/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_102/random_uniform/subSub"Initializer_102/random_uniform/max"Initializer_102/random_uniform/min*
T0

"Initializer_102/random_uniform/mulMul,Initializer_102/random_uniform/RandomUniform"Initializer_102/random_uniform/sub*
T0
v
Initializer_102/random_uniformAdd"Initializer_102/random_uniform/mul"Initializer_102/random_uniform/min*
T0
Û

Assign_102Assign/mio_variable/readtime_xtr/dense/kernel/gradientInitializer_102/random_uniform*B
_class8
64loc:@mio_variable/readtime_xtr/dense/kernel/gradient*
validate_shape(*
use_locking(*
T0

-mio_variable/readtime_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreadtime_xtr/dense/bias*
shape:

-mio_variable/readtime_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreadtime_xtr/dense/bias*
shape:
G
Initializer_103/zerosConst*
valueB*    *
dtype0
Î

Assign_103Assign-mio_variable/readtime_xtr/dense/bias/gradientInitializer_103/zeros*
use_locking(*
T0*@
_class6
42loc:@mio_variable/readtime_xtr/dense/bias/gradient*
validate_shape(

readtime_xtr/dense/MatMulMatMulconcat/mio_variable/readtime_xtr/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 

readtime_xtr/dense/BiasAddBiasAddreadtime_xtr/dense/MatMul-mio_variable/readtime_xtr/dense/bias/variable*
data_formatNHWC*
T0
O
"readtime_xtr/dense/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
p
 readtime_xtr/dense/LeakyRelu/mulMul"readtime_xtr/dense/LeakyRelu/alphareadtime_xtr/dense/BiasAdd*
T0
n
readtime_xtr/dense/LeakyReluMaximum readtime_xtr/dense/LeakyRelu/mulreadtime_xtr/dense/BiasAdd*
T0
P
readtime_xtr/dropout/IdentityIdentityreadtime_xtr/dense/LeakyRelu*
T0
¨
1mio_variable/readtime_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS**
	containerreadtime_xtr/dense_1/kernel*
shape:

¨
1mio_variable/readtime_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
**
	containerreadtime_xtr/dense_1/kernel
Y
$Initializer_104/random_uniform/shapeConst*
valueB"      *
dtype0
O
"Initializer_104/random_uniform/minConst*
valueB
 *   ¾*
dtype0
O
"Initializer_104/random_uniform/maxConst*
dtype0*
valueB
 *   >

,Initializer_104/random_uniform/RandomUniformRandomUniform$Initializer_104/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_104/random_uniform/subSub"Initializer_104/random_uniform/max"Initializer_104/random_uniform/min*
T0

"Initializer_104/random_uniform/mulMul,Initializer_104/random_uniform/RandomUniform"Initializer_104/random_uniform/sub*
T0
v
Initializer_104/random_uniformAdd"Initializer_104/random_uniform/mul"Initializer_104/random_uniform/min*
T0
ß

Assign_104Assign1mio_variable/readtime_xtr/dense_1/kernel/gradientInitializer_104/random_uniform*
use_locking(*
T0*D
_class:
86loc:@mio_variable/readtime_xtr/dense_1/kernel/gradient*
validate_shape(

/mio_variable/readtime_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerreadtime_xtr/dense_1/bias*
shape:

/mio_variable/readtime_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerreadtime_xtr/dense_1/bias*
shape:
G
Initializer_105/zerosConst*
valueB*    *
dtype0
Ò

Assign_105Assign/mio_variable/readtime_xtr/dense_1/bias/gradientInitializer_105/zeros*
use_locking(*
T0*B
_class8
64loc:@mio_variable/readtime_xtr/dense_1/bias/gradient*
validate_shape(
¦
readtime_xtr/dense_1/MatMulMatMulreadtime_xtr/dropout/Identity1mio_variable/readtime_xtr/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 

readtime_xtr/dense_1/BiasAddBiasAddreadtime_xtr/dense_1/MatMul/mio_variable/readtime_xtr/dense_1/bias/variable*
T0*
data_formatNHWC
Q
$readtime_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
v
"readtime_xtr/dense_1/LeakyRelu/mulMul$readtime_xtr/dense_1/LeakyRelu/alphareadtime_xtr/dense_1/BiasAdd*
T0
t
readtime_xtr/dense_1/LeakyReluMaximum"readtime_xtr/dense_1/LeakyRelu/mulreadtime_xtr/dense_1/BiasAdd*
T0
T
readtime_xtr/dropout_1/IdentityIdentityreadtime_xtr/dense_1/LeakyRelu*
T0
§
1mio_variable/readtime_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS**
	containerreadtime_xtr/dense_2/kernel*
shape:	@
§
1mio_variable/readtime_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS**
	containerreadtime_xtr/dense_2/kernel*
shape:	@
Y
$Initializer_106/random_uniform/shapeConst*
valueB"   @   *
dtype0
O
"Initializer_106/random_uniform/minConst*
valueB
 *ó5¾*
dtype0
O
"Initializer_106/random_uniform/maxConst*
valueB
 *ó5>*
dtype0

,Initializer_106/random_uniform/RandomUniformRandomUniform$Initializer_106/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_106/random_uniform/subSub"Initializer_106/random_uniform/max"Initializer_106/random_uniform/min*
T0

"Initializer_106/random_uniform/mulMul,Initializer_106/random_uniform/RandomUniform"Initializer_106/random_uniform/sub*
T0
v
Initializer_106/random_uniformAdd"Initializer_106/random_uniform/mul"Initializer_106/random_uniform/min*
T0
ß

Assign_106Assign1mio_variable/readtime_xtr/dense_2/kernel/gradientInitializer_106/random_uniform*
use_locking(*
T0*D
_class:
86loc:@mio_variable/readtime_xtr/dense_2/kernel/gradient*
validate_shape(

/mio_variable/readtime_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerreadtime_xtr/dense_2/bias*
shape:@

/mio_variable/readtime_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerreadtime_xtr/dense_2/bias*
shape:@
F
Initializer_107/zerosConst*
valueB@*    *
dtype0
Ò

Assign_107Assign/mio_variable/readtime_xtr/dense_2/bias/gradientInitializer_107/zeros*
use_locking(*
T0*B
_class8
64loc:@mio_variable/readtime_xtr/dense_2/bias/gradient*
validate_shape(
¨
readtime_xtr/dense_2/MatMulMatMulreadtime_xtr/dropout_1/Identity1mio_variable/readtime_xtr/dense_2/kernel/variable*
transpose_b( *
T0*
transpose_a( 

readtime_xtr/dense_2/BiasAddBiasAddreadtime_xtr/dense_2/MatMul/mio_variable/readtime_xtr/dense_2/bias/variable*
T0*
data_formatNHWC
Q
$readtime_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
v
"readtime_xtr/dense_2/LeakyRelu/mulMul$readtime_xtr/dense_2/LeakyRelu/alphareadtime_xtr/dense_2/BiasAdd*
T0
t
readtime_xtr/dense_2/LeakyReluMaximum"readtime_xtr/dense_2/LeakyRelu/mulreadtime_xtr/dense_2/BiasAdd*
T0
¦
1mio_variable/readtime_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS**
	containerreadtime_xtr/dense_3/kernel*
shape
:@
¦
1mio_variable/readtime_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS**
	containerreadtime_xtr/dense_3/kernel*
shape
:@
Y
$Initializer_108/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_108/random_uniform/minConst*
valueB
 *¾*
dtype0
O
"Initializer_108/random_uniform/maxConst*
valueB
 *>*
dtype0

,Initializer_108/random_uniform/RandomUniformRandomUniform$Initializer_108/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_108/random_uniform/subSub"Initializer_108/random_uniform/max"Initializer_108/random_uniform/min*
T0

"Initializer_108/random_uniform/mulMul,Initializer_108/random_uniform/RandomUniform"Initializer_108/random_uniform/sub*
T0
v
Initializer_108/random_uniformAdd"Initializer_108/random_uniform/mul"Initializer_108/random_uniform/min*
T0
ß

Assign_108Assign1mio_variable/readtime_xtr/dense_3/kernel/gradientInitializer_108/random_uniform*
T0*D
_class:
86loc:@mio_variable/readtime_xtr/dense_3/kernel/gradient*
validate_shape(*
use_locking(

/mio_variable/readtime_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerreadtime_xtr/dense_3/bias*
shape:

/mio_variable/readtime_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*(
	containerreadtime_xtr/dense_3/bias
F
Initializer_109/zerosConst*
valueB*    *
dtype0
Ò

Assign_109Assign/mio_variable/readtime_xtr/dense_3/bias/gradientInitializer_109/zeros*
use_locking(*
T0*B
_class8
64loc:@mio_variable/readtime_xtr/dense_3/bias/gradient*
validate_shape(
§
readtime_xtr/dense_3/MatMulMatMulreadtime_xtr/dense_2/LeakyRelu1mio_variable/readtime_xtr/dense_3/kernel/variable*
transpose_b( *
T0*
transpose_a( 

readtime_xtr/dense_3/BiasAddBiasAddreadtime_xtr/dense_3/MatMul/mio_variable/readtime_xtr/dense_3/bias/variable*
data_formatNHWC*
T0
N
readtime_xtr/dense_3/SigmoidSigmoidreadtime_xtr/dense_3/BiasAdd*
T0
2
sub/xConst*
valueB
 *  ?*
dtype0
=
subSubsub/x!page_time_predict/dense_3/Sigmoid*
T0
C
truedivRealDiv!page_time_predict/dense_3/Sigmoidsub*
T0
4
sub_1/xConst*
valueB
 *  ?*
dtype0
<
sub_1Subsub_1/xreadtime_xtr/dense_3/Sigmoid*
T0
B
	truediv_1RealDivreadtime_xtr/dense_3/Sigmoidsub_1*
T0"