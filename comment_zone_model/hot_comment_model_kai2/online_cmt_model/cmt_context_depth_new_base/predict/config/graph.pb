
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
CastCast2mio_compress_indices/COMPRESS_INDEX__USER/variable*
Truncate( *

DstT0*

SrcT0

&mio_embeddings/user_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containeruser_embedding*
shape:ÿÿÿÿÿÿÿÿÿ

&mio_embeddings/user_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:ÿÿÿÿÿÿÿÿÿ*
	containeruser_embedding

%mio_embeddings/pid_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:ÿÿÿÿÿÿÿÿÿ@*
	containerpid_embedding

%mio_embeddings/pid_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:ÿÿÿÿÿÿÿÿÿ@*
	containerpid_embedding
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
	containeruid_embedding*
shape:ÿÿÿÿÿÿÿÿÿ@

%mio_embeddings/did_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerdid_embedding*
shape:ÿÿÿÿÿÿÿÿÿ@

%mio_embeddings/did_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:ÿÿÿÿÿÿÿÿÿ@*
	containerdid_embedding

)mio_embeddings/context_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:ÿÿÿÿÿÿÿÿÿ@* 
	containercontext_embedding

)mio_embeddings/context_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS* 
	containercontext_embedding*
shape:ÿÿÿÿÿÿÿÿÿ@

&mio_embeddings/c_id_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerc_id_embedding*
shape:ÿÿÿÿÿÿÿÿÿ

&mio_embeddings/c_id_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerc_id_embedding*
shape:ÿÿÿÿÿÿÿÿÿ

(mio_embeddings/c_info_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerc_info_embedding*
shape:ÿÿÿÿÿÿÿÿÿÀ

(mio_embeddings/c_info_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerc_info_embedding*
shape:ÿÿÿÿÿÿÿÿÿÀ

*mio_embeddings/position_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:ÿÿÿÿÿÿÿÿÿ*!
	containerposition_embedding
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
0mio_embeddings/comment_length_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containercomment_length_embedding*
shape:ÿÿÿÿÿÿÿÿÿ 
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
concat/values_3/axisConst*
value	B : *
dtype0

concat/values_3GatherV2%mio_embeddings/pid_embedding/variableCastconcat/values_3/axis*
Taxis0*
Tindices0*
Tparams0
>
concat/values_4/axisConst*
value	B : *
dtype0

concat/values_4GatherV2%mio_embeddings/aid_embedding/variableCastconcat/values_4/axis*
Taxis0*
Tindices0*
Tparams0
>
concat/values_5/axisConst*
dtype0*
value	B : 
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
Taxis0*
Tindices0*
Tparams0
>
concat/values_7/axisConst*
value	B : *
dtype0

concat/values_7GatherV2)mio_embeddings/context_embedding/variableCastconcat/values_7/axis*
Taxis0*
Tindices0*
Tparams0
>
concat/axisConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0
Ø
concatConcatV2concat/values_0&mio_embeddings/c_id_embedding/variable(mio_embeddings/c_info_embedding/variableconcat/values_3concat/values_4concat/values_5concat/values_6concat/values_7/mio_embeddings/comment_genre_embedding/variable0mio_embeddings/comment_length_embedding/variableconcat/axis*
T0*
N
*

Tidx0
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
concat_1/values_2GatherV2)mio_embeddings/context_embedding/variableCastconcat_1/values_2/axis*
Tindices0*
Tparams0*
Taxis0
@
concat_1/axisConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0

concat_1ConcatV2concat_1/values_0*mio_embeddings/position_embedding/variableconcat_1/values_2concat_1/axis*

Tidx0*
T0*
N
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
valueB"°     *
dtype0
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
+mio_variable/expand_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containerexpand_xtr/dense/bias
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
expand_xtr/dense/BiasAddBiasAddexpand_xtr/dense/MatMul+mio_variable/expand_xtr/dense/bias/variable*
data_formatNHWC*
T0
M
 expand_xtr/dense/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍÌL>
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
/mio_variable/expand_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerexpand_xtr/dense_1/kernel*
shape:

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
Assign_2Assign/mio_variable/expand_xtr/dense_1/kernel/gradientInitializer_2/random_uniform*B
_class8
64loc:@mio_variable/expand_xtr/dense_1/kernel/gradient*
validate_shape(*
use_locking(*
T0
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
expand_xtr/dense_1/MatMulMatMulexpand_xtr/dropout/Identity/mio_variable/expand_xtr/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 

expand_xtr/dense_1/BiasAddBiasAddexpand_xtr/dense_1/MatMul-mio_variable/expand_xtr/dense_1/bias/variable*
data_formatNHWC*
T0
O
"expand_xtr/dense_1/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍÌL>
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
dtype0*
valueB"   @   
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
Assign_4Assign/mio_variable/expand_xtr/dense_2/kernel/gradientInitializer_4/random_uniform*
use_locking(*
T0*B
_class8
64loc:@mio_variable/expand_xtr/dense_2/kernel/gradient*
validate_shape(

-mio_variable/expand_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerexpand_xtr/dense_2/bias*
shape:@

-mio_variable/expand_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*&
	containerexpand_xtr/dense_2/bias
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
/mio_variable/expand_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerexpand_xtr/dense_3/kernel*
shape
:@
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

seed *
T0*
dtype0*
seed2 
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
-mio_variable/expand_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*&
	containerexpand_xtr/dense_3/bias
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
expand_xtr/dense_3/BiasAddBiasAddexpand_xtr/dense_3/MatMul-mio_variable/expand_xtr/dense_3/bias/variable*
data_formatNHWC*
T0
J
expand_xtr/dense_3/SigmoidSigmoidexpand_xtr/dense_3/BiasAdd*
T0

+mio_variable/like_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerlike_xtr/dense/kernel*
shape:
°

+mio_variable/like_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerlike_xtr/dense/kernel*
shape:
°
W
"Initializer_8/random_uniform/shapeConst*
valueB"°     *
dtype0
M
 Initializer_8/random_uniform/minConst*
valueB
 *dF£½*
dtype0
M
 Initializer_8/random_uniform/maxConst*
valueB
 *dF£=*
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
Ï
Assign_8Assign+mio_variable/like_xtr/dense/kernel/gradientInitializer_8/random_uniform*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@mio_variable/like_xtr/dense/kernel/gradient
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
like_xtr/dense/BiasAddBiasAddlike_xtr/dense/MatMul)mio_variable/like_xtr/dense/bias/variable*
data_formatNHWC*
T0
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
dtype0*
valueB"      
N
!Initializer_10/random_uniform/minConst*
valueB
 *   ¾*
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
	Assign_10Assign-mio_variable/like_xtr/dense_1/kernel/gradientInitializer_10/random_uniform*@
_class6
42loc:@mio_variable/like_xtr/dense_1/kernel/gradient*
validate_shape(*
use_locking(*
T0

+mio_variable/like_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerlike_xtr/dense_1/bias*
shape:

+mio_variable/like_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containerlike_xtr/dense_1/bias
F
Initializer_11/zerosConst*
valueB*    *
dtype0
È
	Assign_11Assign+mio_variable/like_xtr/dense_1/bias/gradientInitializer_11/zeros*
T0*>
_class4
20loc:@mio_variable/like_xtr/dense_1/bias/gradient*
validate_shape(*
use_locking(

like_xtr/dense_1/MatMulMatMullike_xtr/dropout/Identity-mio_variable/like_xtr/dense_1/kernel/variable*
transpose_b( *
T0*
transpose_a( 

like_xtr/dense_1/BiasAddBiasAddlike_xtr/dense_1/MatMul+mio_variable/like_xtr/dense_1/bias/variable*
data_formatNHWC*
T0
M
 like_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
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
 *ó5¾*
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
	Assign_12Assign-mio_variable/like_xtr/dense_2/kernel/gradientInitializer_12/random_uniform*
T0*@
_class6
42loc:@mio_variable/like_xtr/dense_2/kernel/gradient*
validate_shape(*
use_locking(

+mio_variable/like_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerlike_xtr/dense_2/bias*
shape:@
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
like_xtr/dense_2/MatMulMatMullike_xtr/dropout_1/Identity-mio_variable/like_xtr/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 

like_xtr/dense_2/BiasAddBiasAddlike_xtr/dense_2/MatMul+mio_variable/like_xtr/dense_2/bias/variable*
data_formatNHWC*
T0
M
 like_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
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
!Initializer_14/random_uniform/minConst*
valueB
 *¾*
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
	Assign_14Assign-mio_variable/like_xtr/dense_3/kernel/gradientInitializer_14/random_uniform*@
_class6
42loc:@mio_variable/like_xtr/dense_3/kernel/gradient*
validate_shape(*
use_locking(*
T0
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
	Assign_15Assign+mio_variable/like_xtr/dense_3/bias/gradientInitializer_15/zeros*
use_locking(*
T0*>
_class4
20loc:@mio_variable/like_xtr/dense_3/bias/gradient*
validate_shape(

like_xtr/dense_3/MatMulMatMullike_xtr/dense_2/LeakyRelu-mio_variable/like_xtr/dense_3/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
°

,mio_variable/reply_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense/kernel*
shape:
°
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
+Initializer_16/random_uniform/RandomUniformRandomUniform#Initializer_16/random_uniform/shape*
seed2 *

seed *
T0*
dtype0
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
*mio_variable/reply_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerreply_xtr/dense/bias*
shape:

*mio_variable/reply_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*#
	containerreply_xtr/dense/bias
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
,mio_variable/reply_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*%
	containerreply_xtr/dense_1/bias

,mio_variable/reply_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense_1/bias*
shape:
F
Initializer_19/zerosConst*
valueB*    *
dtype0
Ê
	Assign_19Assign,mio_variable/reply_xtr/dense_1/bias/gradientInitializer_19/zeros*?
_class5
31loc:@mio_variable/reply_xtr/dense_1/bias/gradient*
validate_shape(*
use_locking(*
T0

reply_xtr/dense_1/MatMulMatMulreply_xtr/dropout/Identity.mio_variable/reply_xtr/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
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
.mio_variable/reply_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerreply_xtr/dense_2/kernel*
shape:	@
¡
.mio_variable/reply_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*'
	containerreply_xtr/dense_2/kernel
X
#Initializer_20/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_20/random_uniform/minConst*
valueB
 *ó5¾*
dtype0
N
!Initializer_20/random_uniform/maxConst*
valueB
 *ó5>*
dtype0

+Initializer_20/random_uniform/RandomUniformRandomUniform#Initializer_20/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
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
Ê
	Assign_21Assign,mio_variable/reply_xtr/dense_2/bias/gradientInitializer_21/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/reply_xtr/dense_2/bias/gradient*
validate_shape(

reply_xtr/dense_2/MatMulMatMulreply_xtr/dropout_1/Identity.mio_variable/reply_xtr/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0

reply_xtr/dense_2/BiasAddBiasAddreply_xtr/dense_2/MatMul,mio_variable/reply_xtr/dense_2/bias/variable*
T0*
data_formatNHWC
N
!reply_xtr/dense_2/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍÌL>
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
+Initializer_22/random_uniform/RandomUniformRandomUniform#Initializer_22/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
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
	Assign_22Assign.mio_variable/reply_xtr/dense_3/kernel/gradientInitializer_22/random_uniform*
validate_shape(*
use_locking(*
T0*A
_class7
53loc:@mio_variable/reply_xtr/dense_3/kernel/gradient

,mio_variable/reply_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense_3/bias*
shape:

,mio_variable/reply_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense_3/bias*
shape:
E
Initializer_23/zerosConst*
dtype0*
valueB*    
Ê
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
T0

+mio_variable/copy_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense/kernel*
shape:
°

+mio_variable/copy_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*$
	containercopy_xtr/dense/kernel
X
#Initializer_24/random_uniform/shapeConst*
valueB"°     *
dtype0
N
!Initializer_24/random_uniform/minConst*
valueB
 *dF£½*
dtype0
N
!Initializer_24/random_uniform/maxConst*
valueB
 *dF£=*
dtype0

+Initializer_24/random_uniform/RandomUniformRandomUniform#Initializer_24/random_uniform/shape*
seed2 *

seed *
T0*
dtype0
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
	Assign_26Assign-mio_variable/copy_xtr/dense_1/kernel/gradientInitializer_26/random_uniform*@
_class6
42loc:@mio_variable/copy_xtr/dense_1/kernel/gradient*
validate_shape(*
use_locking(*
T0

+mio_variable/copy_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense_1/bias*
shape:

+mio_variable/copy_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense_1/bias*
shape:
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
copy_xtr/dense_1/MatMulMatMulcopy_xtr/dropout/Identity-mio_variable/copy_xtr/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
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
-mio_variable/copy_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containercopy_xtr/dense_2/kernel*
shape:	@
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
+Initializer_28/random_uniform/RandomUniformRandomUniform#Initializer_28/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
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
+mio_variable/copy_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*$
	containercopy_xtr/dense_2/bias
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
copy_xtr/dense_2/BiasAddBiasAddcopy_xtr/dense_2/MatMul+mio_variable/copy_xtr/dense_2/bias/variable*
data_formatNHWC*
T0
M
 copy_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
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
È
	Assign_31Assign+mio_variable/copy_xtr/dense_3/bias/gradientInitializer_31/zeros*>
_class4
20loc:@mio_variable/copy_xtr/dense_3/bias/gradient*
validate_shape(*
use_locking(*
T0

copy_xtr/dense_3/MatMulMatMulcopy_xtr/dense_2/LeakyRelu-mio_variable/copy_xtr/dense_3/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
,mio_variable/share_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*%
	containershare_xtr/dense/kernel
X
#Initializer_32/random_uniform/shapeConst*
dtype0*
valueB"°     
N
!Initializer_32/random_uniform/minConst*
dtype0*
valueB
 *dF£½
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
*mio_variable/share_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*#
	containershare_xtr/dense/bias*
shape:

*mio_variable/share_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*#
	containershare_xtr/dense/bias*
shape:
F
Initializer_33/zerosConst*
valueB*    *
dtype0
Æ
	Assign_33Assign*mio_variable/share_xtr/dense/bias/gradientInitializer_33/zeros*
validate_shape(*
use_locking(*
T0*=
_class3
1/loc:@mio_variable/share_xtr/dense/bias/gradient

share_xtr/dense/MatMulMatMulconcat,mio_variable/share_xtr/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
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
!Initializer_34/random_uniform/minConst*
dtype0*
valueB
 *   ¾
N
!Initializer_34/random_uniform/maxConst*
valueB
 *   >*
dtype0

+Initializer_34/random_uniform/RandomUniformRandomUniform#Initializer_34/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
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
	Assign_34Assign.mio_variable/share_xtr/dense_1/kernel/gradientInitializer_34/random_uniform*
T0*A
_class7
53loc:@mio_variable/share_xtr/dense_1/kernel/gradient*
validate_shape(*
use_locking(

,mio_variable/share_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*%
	containershare_xtr/dense_1/bias
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
×
	Assign_36Assign.mio_variable/share_xtr/dense_2/kernel/gradientInitializer_36/random_uniform*
T0*A
_class7
53loc:@mio_variable/share_xtr/dense_2/kernel/gradient*
validate_shape(*
use_locking(

,mio_variable/share_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containershare_xtr/dense_2/bias*
shape:@

,mio_variable/share_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*%
	containershare_xtr/dense_2/bias
E
Initializer_37/zerosConst*
dtype0*
valueB@*    
Ê
	Assign_37Assign,mio_variable/share_xtr/dense_2/bias/gradientInitializer_37/zeros*
validate_shape(*
use_locking(*
T0*?
_class5
31loc:@mio_variable/share_xtr/dense_2/bias/gradient

share_xtr/dense_2/MatMulMatMulshare_xtr/dropout_1/Identity.mio_variable/share_xtr/dense_2/kernel/variable*
transpose_b( *
T0*
transpose_a( 
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
.mio_variable/share_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*'
	containershare_xtr/dense_3/kernel
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

seed *
T0*
dtype0*
seed2 
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
	Assign_38Assign.mio_variable/share_xtr/dense_3/kernel/gradientInitializer_38/random_uniform*
T0*A
_class7
53loc:@mio_variable/share_xtr/dense_3/kernel/gradient*
validate_shape(*
use_locking(
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
Ê
	Assign_39Assign,mio_variable/share_xtr/dense_3/bias/gradientInitializer_39/zeros*
validate_shape(*
use_locking(*
T0*?
_class5
31loc:@mio_variable/share_xtr/dense_3/bias/gradient
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
/mio_variable/audience_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeraudience_xtr/dense/kernel*
shape:
°
X
#Initializer_40/random_uniform/shapeConst*
valueB"°     *
dtype0
N
!Initializer_40/random_uniform/minConst*
dtype0*
valueB
 *dF£½
N
!Initializer_40/random_uniform/maxConst*
valueB
 *dF£=*
dtype0
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
1mio_variable/audience_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
**
	containeraudience_xtr/dense_1/kernel
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
+Initializer_42/random_uniform/RandomUniformRandomUniform#Initializer_42/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
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
/mio_variable/audience_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeraudience_xtr/dense_1/bias*
shape:

/mio_variable/audience_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeraudience_xtr/dense_1/bias*
shape:
F
Initializer_43/zerosConst*
valueB*    *
dtype0
Ð
	Assign_43Assign/mio_variable/audience_xtr/dense_1/bias/gradientInitializer_43/zeros*
T0*B
_class8
64loc:@mio_variable/audience_xtr/dense_1/bias/gradient*
validate_shape(*
use_locking(
¦
audience_xtr/dense_1/MatMulMatMulaudience_xtr/dropout/Identity1mio_variable/audience_xtr/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 

audience_xtr/dense_1/BiasAddBiasAddaudience_xtr/dense_1/MatMul/mio_variable/audience_xtr/dense_1/bias/variable*
data_formatNHWC*
T0
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
1mio_variable/audience_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@**
	containeraudience_xtr/dense_2/kernel
X
#Initializer_44/random_uniform/shapeConst*
valueB"   @   *
dtype0
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
	Assign_44Assign1mio_variable/audience_xtr/dense_2/kernel/gradientInitializer_44/random_uniform*
use_locking(*
T0*D
_class:
86loc:@mio_variable/audience_xtr/dense_2/kernel/gradient*
validate_shape(

/mio_variable/audience_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*(
	containeraudience_xtr/dense_2/bias
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
¨
audience_xtr/dense_2/MatMulMatMulaudience_xtr/dropout_1/Identity1mio_variable/audience_xtr/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 
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
1mio_variable/audience_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@**
	containeraudience_xtr/dense_3/kernel
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

seed *
T0*
dtype0*
seed2 
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
	Assign_46Assign1mio_variable/audience_xtr/dense_3/kernel/gradientInitializer_46/random_uniform*
T0*D
_class:
86loc:@mio_variable/audience_xtr/dense_3/kernel/gradient*
validate_shape(*
use_locking(

/mio_variable/audience_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeraudience_xtr/dense_3/bias*
shape:
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
¶
8mio_variable/continuous_expand_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*1
	container$"continuous_expand_xtr/dense/kernel
¶
8mio_variable/continuous_expand_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"continuous_expand_xtr/dense/kernel*
shape:
°
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
Initializer_49/zerosConst*
dtype0*
valueB*    
Þ
	Assign_49Assign6mio_variable/continuous_expand_xtr/dense/bias/gradientInitializer_49/zeros*I
_class?
=;loc:@mio_variable/continuous_expand_xtr/dense/bias/gradient*
validate_shape(*
use_locking(*
T0

"continuous_expand_xtr/dense/MatMulMatMulconcat8mio_variable/continuous_expand_xtr/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
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
	Assign_50Assign:mio_variable/continuous_expand_xtr/dense_1/kernel/gradientInitializer_50/random_uniform*M
_classC
A?loc:@mio_variable/continuous_expand_xtr/dense_1/kernel/gradient*
validate_shape(*
use_locking(*
T0
±
8mio_variable/continuous_expand_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"continuous_expand_xtr/dense_1/bias*
shape:
±
8mio_variable/continuous_expand_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"continuous_expand_xtr/dense_1/bias*
shape:
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
Á
$continuous_expand_xtr/dense_1/MatMulMatMul&continuous_expand_xtr/dropout/Identity:mio_variable/continuous_expand_xtr/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
°
%continuous_expand_xtr/dense_1/BiasAddBiasAdd$continuous_expand_xtr/dense_1/MatMul8mio_variable/continuous_expand_xtr/dense_1/bias/variable*
data_formatNHWC*
T0
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
:mio_variable/continuous_expand_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$continuous_expand_xtr/dense_2/kernel*
shape:	@
¹
:mio_variable/continuous_expand_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*3
	container&$continuous_expand_xtr/dense_2/kernel
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
8mio_variable/continuous_expand_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*1
	container$"continuous_expand_xtr/dense_2/bias
°
8mio_variable/continuous_expand_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"continuous_expand_xtr/dense_2/bias*
shape:@
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
$continuous_expand_xtr/dense_2/MatMulMatMul(continuous_expand_xtr/dropout_1/Identity:mio_variable/continuous_expand_xtr/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 
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
:mio_variable/continuous_expand_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$continuous_expand_xtr/dense_3/kernel*
shape
:@
¸
:mio_variable/continuous_expand_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*3
	container&$continuous_expand_xtr/dense_3/kernel
X
#Initializer_54/random_uniform/shapeConst*
valueB"@      *
dtype0
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
+Initializer_54/random_uniform/RandomUniformRandomUniform#Initializer_54/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
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
8mio_variable/continuous_expand_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"continuous_expand_xtr/dense_3/bias*
shape:
°
8mio_variable/continuous_expand_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*1
	container$"continuous_expand_xtr/dense_3/bias
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
$continuous_expand_xtr/dense_3/MatMulMatMul'continuous_expand_xtr/dense_2/LeakyRelu:mio_variable/continuous_expand_xtr/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 
°
%continuous_expand_xtr/dense_3/BiasAddBiasAdd$continuous_expand_xtr/dense_3/MatMul8mio_variable/continuous_expand_xtr/dense_3/bias/variable*
T0*
data_formatNHWC
`
%continuous_expand_xtr/dense_3/SigmoidSigmoid%continuous_expand_xtr/dense_3/BiasAdd*
T0
¬
3mio_variable/duration_predict/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerduration_predict/dense/kernel*
shape:
°
¬
3mio_variable/duration_predict/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*,
	containerduration_predict/dense/kernel
X
#Initializer_56/random_uniform/shapeConst*
valueB"°     *
dtype0
N
!Initializer_56/random_uniform/minConst*
dtype0*
valueB
 *dF£½
N
!Initializer_56/random_uniform/maxConst*
valueB
 *dF£=*
dtype0

+Initializer_56/random_uniform/RandomUniformRandomUniform#Initializer_56/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
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
1mio_variable/duration_predict/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:**
	containerduration_predict/dense/bias
£
1mio_variable/duration_predict/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:**
	containerduration_predict/dense/bias
F
Initializer_57/zerosConst*
valueB*    *
dtype0
Ô
	Assign_57Assign1mio_variable/duration_predict/dense/bias/gradientInitializer_57/zeros*D
_class:
86loc:@mio_variable/duration_predict/dense/bias/gradient*
validate_shape(*
use_locking(*
T0

duration_predict/dense/MatMulMatMulconcat3mio_variable/duration_predict/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 
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
!Initializer_58/random_uniform/maxConst*
valueB
 *   >*
dtype0

+Initializer_58/random_uniform/RandomUniformRandomUniform#Initializer_58/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
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
3mio_variable/duration_predict/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerduration_predict/dense_1/bias*
shape:
§
3mio_variable/duration_predict/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerduration_predict/dense_1/bias*
shape:
F
Initializer_59/zerosConst*
valueB*    *
dtype0
Ø
	Assign_59Assign3mio_variable/duration_predict/dense_1/bias/gradientInitializer_59/zeros*
T0*F
_class<
:8loc:@mio_variable/duration_predict/dense_1/bias/gradient*
validate_shape(*
use_locking(
²
duration_predict/dense_1/MatMulMatMul!duration_predict/dropout/Identity5mio_variable/duration_predict/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
¡
 duration_predict/dense_1/BiasAddBiasAddduration_predict/dense_1/MatMul3mio_variable/duration_predict/dense_1/bias/variable*
T0*
data_formatNHWC
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
!Initializer_60/random_uniform/maxConst*
dtype0*
valueB
 *ó5>

+Initializer_60/random_uniform/RandomUniformRandomUniform#Initializer_60/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
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
3mio_variable/duration_predict/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*,
	containerduration_predict/dense_2/bias
¦
3mio_variable/duration_predict/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerduration_predict/dense_2/bias*
shape:@
E
Initializer_61/zerosConst*
valueB@*    *
dtype0
Ø
	Assign_61Assign3mio_variable/duration_predict/dense_2/bias/gradientInitializer_61/zeros*
T0*F
_class<
:8loc:@mio_variable/duration_predict/dense_2/bias/gradient*
validate_shape(*
use_locking(
´
duration_predict/dense_2/MatMulMatMul#duration_predict/dropout_1/Identity5mio_variable/duration_predict/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0
¡
 duration_predict/dense_2/BiasAddBiasAddduration_predict/dense_2/MatMul3mio_variable/duration_predict/dense_2/bias/variable*
T0*
data_formatNHWC
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
5mio_variable/duration_predict/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!duration_predict/dense_3/kernel*
shape
:@
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
3mio_variable/duration_predict/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*,
	containerduration_predict/dense_3/bias
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
<mio_variable/duration_pos_bias_predict/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&duration_pos_bias_predict/dense/kernel*
shape:

¾
<mio_variable/duration_pos_bias_predict/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&duration_pos_bias_predict/dense/kernel*
shape:

X
#Initializer_64/random_uniform/shapeConst*
dtype0*
valueB"      
N
!Initializer_64/random_uniform/minConst*
dtype0*
valueB
 *²_¾
N
!Initializer_64/random_uniform/maxConst*
dtype0*
valueB
 *²_>

+Initializer_64/random_uniform/RandomUniformRandomUniform#Initializer_64/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
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
	Assign_64Assign<mio_variable/duration_pos_bias_predict/dense/kernel/gradientInitializer_64/random_uniform*
use_locking(*
T0*O
_classE
CAloc:@mio_variable/duration_pos_bias_predict/dense/kernel/gradient*
validate_shape(
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
&duration_pos_bias_predict/dense/MatMulMatMulconcat_1<mio_variable/duration_pos_bias_predict/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
>mio_variable/duration_pos_bias_predict/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*7
	container*(duration_pos_bias_predict/dense_1/kernel
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
+Initializer_66/random_uniform/RandomUniformRandomUniform#Initializer_66/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
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
<mio_variable/duration_pos_bias_predict/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*5
	container(&duration_pos_bias_predict/dense_1/bias
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
(duration_pos_bias_predict/dense_1/MatMulMatMul*duration_pos_bias_predict/dropout/Identity>mio_variable/duration_pos_bias_predict/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
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
!Initializer_68/random_uniform/minConst*
valueB
 *¾*
dtype0
N
!Initializer_68/random_uniform/maxConst*
valueB
 *>*
dtype0

+Initializer_68/random_uniform/RandomUniformRandomUniform#Initializer_68/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
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
	Assign_68Assign>mio_variable/duration_pos_bias_predict/dense_2/kernel/gradientInitializer_68/random_uniform*
use_locking(*
T0*Q
_classG
ECloc:@mio_variable/duration_pos_bias_predict/dense_2/kernel/gradient*
validate_shape(
¸
<mio_variable/duration_pos_bias_predict/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&duration_pos_bias_predict/dense_2/bias*
shape:
¸
<mio_variable/duration_pos_bias_predict/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&duration_pos_bias_predict/dense_2/bias*
shape:
E
Initializer_69/zerosConst*
valueB*    *
dtype0
ê
	Assign_69Assign<mio_variable/duration_pos_bias_predict/dense_2/bias/gradientInitializer_69/zeros*
use_locking(*
T0*O
_classE
CAloc:@mio_variable/duration_pos_bias_predict/dense_2/bias/gradient*
validate_shape(
Î
(duration_pos_bias_predict/dense_2/MatMulMatMul+duration_pos_bias_predict/dense_1/LeakyRelu>mio_variable/duration_pos_bias_predict/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 
¼
)duration_pos_bias_predict/dense_2/BiasAddBiasAdd(duration_pos_bias_predict/dense_2/MatMul<mio_variable/duration_pos_bias_predict/dense_2/bias/variable*
T0*
data_formatNHWC
b
&duration_pos_bias_predict/dense_2/ReluRelu)duration_pos_bias_predict/dense_2/BiasAdd*
T0

,mio_variable/depth_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*%
	containerdepth_xtr/dense/kernel

,mio_variable/depth_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerdepth_xtr/dense/kernel*
shape:
°
X
#Initializer_70/random_uniform/shapeConst*
dtype0*
valueB"°     
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
Ó
	Assign_70Assign,mio_variable/depth_xtr/dense/kernel/gradientInitializer_70/random_uniform*
T0*?
_class5
31loc:@mio_variable/depth_xtr/dense/kernel/gradient*
validate_shape(*
use_locking(

*mio_variable/depth_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerdepth_xtr/dense/bias*
shape:

*mio_variable/depth_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerdepth_xtr/dense/bias*
shape:
F
Initializer_71/zerosConst*
valueB*    *
dtype0
Æ
	Assign_71Assign*mio_variable/depth_xtr/dense/bias/gradientInitializer_71/zeros*
use_locking(*
T0*=
_class3
1/loc:@mio_variable/depth_xtr/dense/bias/gradient*
validate_shape(

depth_xtr/dense/MatMulMatMulconcat,mio_variable/depth_xtr/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 

depth_xtr/dense/BiasAddBiasAdddepth_xtr/dense/MatMul*mio_variable/depth_xtr/dense/bias/variable*
data_formatNHWC*
T0
L
depth_xtr/dense/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
g
depth_xtr/dense/LeakyRelu/mulMuldepth_xtr/dense/LeakyRelu/alphadepth_xtr/dense/BiasAdd*
T0
e
depth_xtr/dense/LeakyReluMaximumdepth_xtr/dense/LeakyRelu/muldepth_xtr/dense/BiasAdd*
T0
J
depth_xtr/dropout/IdentityIdentitydepth_xtr/dense/LeakyRelu*
T0
¢
.mio_variable/depth_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*'
	containerdepth_xtr/dense_1/kernel
¢
.mio_variable/depth_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerdepth_xtr/dense_1/kernel*
shape:

X
#Initializer_72/random_uniform/shapeConst*
dtype0*
valueB"      
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
×
	Assign_72Assign.mio_variable/depth_xtr/dense_1/kernel/gradientInitializer_72/random_uniform*
use_locking(*
T0*A
_class7
53loc:@mio_variable/depth_xtr/dense_1/kernel/gradient*
validate_shape(

,mio_variable/depth_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*%
	containerdepth_xtr/dense_1/bias

,mio_variable/depth_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*%
	containerdepth_xtr/dense_1/bias
F
Initializer_73/zerosConst*
dtype0*
valueB*    
Ê
	Assign_73Assign,mio_variable/depth_xtr/dense_1/bias/gradientInitializer_73/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/depth_xtr/dense_1/bias/gradient*
validate_shape(

depth_xtr/dense_1/MatMulMatMuldepth_xtr/dropout/Identity.mio_variable/depth_xtr/dense_1/kernel/variable*
transpose_b( *
T0*
transpose_a( 

depth_xtr/dense_1/BiasAddBiasAdddepth_xtr/dense_1/MatMul,mio_variable/depth_xtr/dense_1/bias/variable*
T0*
data_formatNHWC
N
!depth_xtr/dense_1/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍÌL>
m
depth_xtr/dense_1/LeakyRelu/mulMul!depth_xtr/dense_1/LeakyRelu/alphadepth_xtr/dense_1/BiasAdd*
T0
k
depth_xtr/dense_1/LeakyReluMaximumdepth_xtr/dense_1/LeakyRelu/muldepth_xtr/dense_1/BiasAdd*
T0
N
depth_xtr/dropout_1/IdentityIdentitydepth_xtr/dense_1/LeakyRelu*
T0
¡
.mio_variable/depth_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerdepth_xtr/dense_2/kernel*
shape:	@
¡
.mio_variable/depth_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerdepth_xtr/dense_2/kernel*
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
!Initializer_74/random_uniform/maxConst*
valueB
 *ó5>*
dtype0

+Initializer_74/random_uniform/RandomUniformRandomUniform#Initializer_74/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_74/random_uniform/subSub!Initializer_74/random_uniform/max!Initializer_74/random_uniform/min*
T0

!Initializer_74/random_uniform/mulMul+Initializer_74/random_uniform/RandomUniform!Initializer_74/random_uniform/sub*
T0
s
Initializer_74/random_uniformAdd!Initializer_74/random_uniform/mul!Initializer_74/random_uniform/min*
T0
×
	Assign_74Assign.mio_variable/depth_xtr/dense_2/kernel/gradientInitializer_74/random_uniform*
use_locking(*
T0*A
_class7
53loc:@mio_variable/depth_xtr/dense_2/kernel/gradient*
validate_shape(

,mio_variable/depth_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*%
	containerdepth_xtr/dense_2/bias

,mio_variable/depth_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerdepth_xtr/dense_2/bias*
shape:@
E
Initializer_75/zerosConst*
valueB@*    *
dtype0
Ê
	Assign_75Assign,mio_variable/depth_xtr/dense_2/bias/gradientInitializer_75/zeros*?
_class5
31loc:@mio_variable/depth_xtr/dense_2/bias/gradient*
validate_shape(*
use_locking(*
T0

depth_xtr/dense_2/MatMulMatMuldepth_xtr/dropout_1/Identity.mio_variable/depth_xtr/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 

depth_xtr/dense_2/BiasAddBiasAdddepth_xtr/dense_2/MatMul,mio_variable/depth_xtr/dense_2/bias/variable*
data_formatNHWC*
T0
N
!depth_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
m
depth_xtr/dense_2/LeakyRelu/mulMul!depth_xtr/dense_2/LeakyRelu/alphadepth_xtr/dense_2/BiasAdd*
T0
k
depth_xtr/dense_2/LeakyReluMaximumdepth_xtr/dense_2/LeakyRelu/muldepth_xtr/dense_2/BiasAdd*
T0
 
.mio_variable/depth_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerdepth_xtr/dense_3/kernel*
shape
:@
 
.mio_variable/depth_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerdepth_xtr/dense_3/kernel*
shape
:@
X
#Initializer_76/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_76/random_uniform/minConst*
dtype0*
valueB
 *¾
N
!Initializer_76/random_uniform/maxConst*
dtype0*
valueB
 *>
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
×
	Assign_76Assign.mio_variable/depth_xtr/dense_3/kernel/gradientInitializer_76/random_uniform*
use_locking(*
T0*A
_class7
53loc:@mio_variable/depth_xtr/dense_3/kernel/gradient*
validate_shape(

,mio_variable/depth_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerdepth_xtr/dense_3/bias*
shape:

,mio_variable/depth_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerdepth_xtr/dense_3/bias*
shape:
E
Initializer_77/zerosConst*
valueB*    *
dtype0
Ê
	Assign_77Assign,mio_variable/depth_xtr/dense_3/bias/gradientInitializer_77/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/depth_xtr/dense_3/bias/gradient*
validate_shape(

depth_xtr/dense_3/MatMulMatMuldepth_xtr/dense_2/LeakyRelu.mio_variable/depth_xtr/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 

depth_xtr/dense_3/BiasAddBiasAdddepth_xtr/dense_3/MatMul,mio_variable/depth_xtr/dense_3/bias/variable*
data_formatNHWC*
T0
H
depth_xtr/dense_3/SigmoidSigmoiddepth_xtr/dense_3/BiasAdd*
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
+Initializer_78/random_uniform/RandomUniformRandomUniform#Initializer_78/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_78/random_uniform/subSub!Initializer_78/random_uniform/max!Initializer_78/random_uniform/min*
T0

!Initializer_78/random_uniform/mulMul+Initializer_78/random_uniform/RandomUniform!Initializer_78/random_uniform/sub*
T0
s
Initializer_78/random_uniformAdd!Initializer_78/random_uniform/mul!Initializer_78/random_uniform/min*
T0
Ñ
	Assign_78Assign+mio_variable/hate_xtr/dense/kernel/gradientInitializer_78/random_uniform*
use_locking(*
T0*>
_class4
20loc:@mio_variable/hate_xtr/dense/kernel/gradient*
validate_shape(

)mio_variable/hate_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*"
	containerhate_xtr/dense/bias

)mio_variable/hate_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*"
	containerhate_xtr/dense/bias*
shape:
F
Initializer_79/zerosConst*
valueB*    *
dtype0
Ä
	Assign_79Assign)mio_variable/hate_xtr/dense/bias/gradientInitializer_79/zeros*
use_locking(*
T0*<
_class2
0.loc:@mio_variable/hate_xtr/dense/bias/gradient*
validate_shape(

hate_xtr/dense/MatMulMatMulconcat+mio_variable/hate_xtr/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
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
-mio_variable/hate_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*&
	containerhate_xtr/dense_1/kernel
 
-mio_variable/hate_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerhate_xtr/dense_1/kernel*
shape:

X
#Initializer_80/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_80/random_uniform/minConst*
dtype0*
valueB
 *   ¾
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
Õ
	Assign_80Assign-mio_variable/hate_xtr/dense_1/kernel/gradientInitializer_80/random_uniform*
validate_shape(*
use_locking(*
T0*@
_class6
42loc:@mio_variable/hate_xtr/dense_1/kernel/gradient

+mio_variable/hate_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerhate_xtr/dense_1/bias*
shape:

+mio_variable/hate_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerhate_xtr/dense_1/bias*
shape:
F
Initializer_81/zerosConst*
valueB*    *
dtype0
È
	Assign_81Assign+mio_variable/hate_xtr/dense_1/bias/gradientInitializer_81/zeros*
use_locking(*
T0*>
_class4
20loc:@mio_variable/hate_xtr/dense_1/bias/gradient*
validate_shape(

hate_xtr/dense_1/MatMulMatMulhate_xtr/dense/LeakyRelu-mio_variable/hate_xtr/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 

hate_xtr/dense_1/BiasAddBiasAddhate_xtr/dense_1/MatMul+mio_variable/hate_xtr/dense_1/bias/variable*
data_formatNHWC*
T0
M
 hate_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
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
#Initializer_82/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_82/random_uniform/minConst*
valueB
 *ó5¾*
dtype0
N
!Initializer_82/random_uniform/maxConst*
valueB
 *ó5>*
dtype0

+Initializer_82/random_uniform/RandomUniformRandomUniform#Initializer_82/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_82/random_uniform/subSub!Initializer_82/random_uniform/max!Initializer_82/random_uniform/min*
T0

!Initializer_82/random_uniform/mulMul+Initializer_82/random_uniform/RandomUniform!Initializer_82/random_uniform/sub*
T0
s
Initializer_82/random_uniformAdd!Initializer_82/random_uniform/mul!Initializer_82/random_uniform/min*
T0
Õ
	Assign_82Assign-mio_variable/hate_xtr/dense_2/kernel/gradientInitializer_82/random_uniform*@
_class6
42loc:@mio_variable/hate_xtr/dense_2/kernel/gradient*
validate_shape(*
use_locking(*
T0

+mio_variable/hate_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerhate_xtr/dense_2/bias*
shape:@

+mio_variable/hate_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerhate_xtr/dense_2/bias*
shape:@
E
Initializer_83/zerosConst*
dtype0*
valueB@*    
È
	Assign_83Assign+mio_variable/hate_xtr/dense_2/bias/gradientInitializer_83/zeros*
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
hate_xtr/dense_2/BiasAddBiasAddhate_xtr/dense_2/MatMul+mio_variable/hate_xtr/dense_2/bias/variable*
T0*
data_formatNHWC
M
 hate_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
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
-mio_variable/hate_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerhate_xtr/dense_3/kernel*
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
Õ
	Assign_84Assign-mio_variable/hate_xtr/dense_3/kernel/gradientInitializer_84/random_uniform*
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
Initializer_85/zerosConst*
valueB*    *
dtype0
È
	Assign_85Assign+mio_variable/hate_xtr/dense_3/bias/gradientInitializer_85/zeros*
T0*>
_class4
20loc:@mio_variable/hate_xtr/dense_3/bias/gradient*
validate_shape(*
use_locking(

hate_xtr/dense_3/MatMulMatMulhate_xtr/dense_2/LeakyRelu-mio_variable/hate_xtr/dense_3/kernel/variable*
transpose_b( *
T0*
transpose_a( 

hate_xtr/dense_3/BiasAddBiasAddhate_xtr/dense_3/MatMul+mio_variable/hate_xtr/dense_3/bias/variable*
T0*
data_formatNHWC
F
hate_xtr/dense_3/SigmoidSigmoidhate_xtr/dense_3/BiasAdd*
T0
 
-mio_variable/report_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreport_xtr/dense/kernel*
shape:
°
 
-mio_variable/report_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreport_xtr/dense/kernel*
shape:
°
X
#Initializer_86/random_uniform/shapeConst*
dtype0*
valueB"°     
N
!Initializer_86/random_uniform/minConst*
valueB
 *dF£½*
dtype0
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
Õ
	Assign_86Assign-mio_variable/report_xtr/dense/kernel/gradientInitializer_86/random_uniform*
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
Initializer_87/zerosConst*
dtype0*
valueB*    
È
	Assign_87Assign+mio_variable/report_xtr/dense/bias/gradientInitializer_87/zeros*>
_class4
20loc:@mio_variable/report_xtr/dense/bias/gradient*
validate_shape(*
use_locking(*
T0

report_xtr/dense/MatMulMatMulconcat-mio_variable/report_xtr/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 

report_xtr/dense/BiasAddBiasAddreport_xtr/dense/MatMul+mio_variable/report_xtr/dense/bias/variable*
data_formatNHWC*
T0
M
 report_xtr/dense/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
j
report_xtr/dense/LeakyRelu/mulMul report_xtr/dense/LeakyRelu/alphareport_xtr/dense/BiasAdd*
T0
h
report_xtr/dense/LeakyReluMaximumreport_xtr/dense/LeakyRelu/mulreport_xtr/dense/BiasAdd*
T0
¤
/mio_variable/report_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerreport_xtr/dense_1/kernel*
shape:

¤
/mio_variable/report_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerreport_xtr/dense_1/kernel*
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
Ù
	Assign_88Assign/mio_variable/report_xtr/dense_1/kernel/gradientInitializer_88/random_uniform*
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
-mio_variable/report_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*&
	containerreport_xtr/dense_1/bias
F
Initializer_89/zerosConst*
valueB*    *
dtype0
Ì
	Assign_89Assign-mio_variable/report_xtr/dense_1/bias/gradientInitializer_89/zeros*
use_locking(*
T0*@
_class6
42loc:@mio_variable/report_xtr/dense_1/bias/gradient*
validate_shape(

report_xtr/dense_1/MatMulMatMulreport_xtr/dense/LeakyRelu/mio_variable/report_xtr/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0

report_xtr/dense_1/BiasAddBiasAddreport_xtr/dense_1/MatMul-mio_variable/report_xtr/dense_1/bias/variable*
data_formatNHWC*
T0
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
/mio_variable/report_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*(
	containerreport_xtr/dense_2/kernel
£
/mio_variable/report_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerreport_xtr/dense_2/kernel*
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
+Initializer_90/random_uniform/RandomUniformRandomUniform#Initializer_90/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_90/random_uniform/subSub!Initializer_90/random_uniform/max!Initializer_90/random_uniform/min*
T0

!Initializer_90/random_uniform/mulMul+Initializer_90/random_uniform/RandomUniform!Initializer_90/random_uniform/sub*
T0
s
Initializer_90/random_uniformAdd!Initializer_90/random_uniform/mul!Initializer_90/random_uniform/min*
T0
Ù
	Assign_90Assign/mio_variable/report_xtr/dense_2/kernel/gradientInitializer_90/random_uniform*
validate_shape(*
use_locking(*
T0*B
_class8
64loc:@mio_variable/report_xtr/dense_2/kernel/gradient

-mio_variable/report_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreport_xtr/dense_2/bias*
shape:@

-mio_variable/report_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*&
	containerreport_xtr/dense_2/bias
E
Initializer_91/zerosConst*
valueB@*    *
dtype0
Ì
	Assign_91Assign-mio_variable/report_xtr/dense_2/bias/gradientInitializer_91/zeros*
T0*@
_class6
42loc:@mio_variable/report_xtr/dense_2/bias/gradient*
validate_shape(*
use_locking(
¡
report_xtr/dense_2/MatMulMatMulreport_xtr/dense_1/LeakyRelu/mio_variable/report_xtr/dense_2/kernel/variable*
transpose_b( *
T0*
transpose_a( 

report_xtr/dense_2/BiasAddBiasAddreport_xtr/dense_2/MatMul-mio_variable/report_xtr/dense_2/bias/variable*
T0*
data_formatNHWC
O
"report_xtr/dense_2/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍÌL>
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
/mio_variable/report_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*(
	containerreport_xtr/dense_3/kernel
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
+Initializer_92/random_uniform/RandomUniformRandomUniform#Initializer_92/random_uniform/shape*
seed2 *

seed *
T0*
dtype0
w
!Initializer_92/random_uniform/subSub!Initializer_92/random_uniform/max!Initializer_92/random_uniform/min*
T0

!Initializer_92/random_uniform/mulMul+Initializer_92/random_uniform/RandomUniform!Initializer_92/random_uniform/sub*
T0
s
Initializer_92/random_uniformAdd!Initializer_92/random_uniform/mul!Initializer_92/random_uniform/min*
T0
Ù
	Assign_92Assign/mio_variable/report_xtr/dense_3/kernel/gradientInitializer_92/random_uniform*B
_class8
64loc:@mio_variable/report_xtr/dense_3/kernel/gradient*
validate_shape(*
use_locking(*
T0

-mio_variable/report_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreport_xtr/dense_3/bias*
shape:

-mio_variable/report_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreport_xtr/dense_3/bias*
shape:
E
Initializer_93/zerosConst*
valueB*    *
dtype0
Ì
	Assign_93Assign-mio_variable/report_xtr/dense_3/bias/gradientInitializer_93/zeros*
use_locking(*
T0*@
_class6
42loc:@mio_variable/report_xtr/dense_3/bias/gradient*
validate_shape(
¡
report_xtr/dense_3/MatMulMatMulreport_xtr/dense_2/LeakyRelu/mio_variable/report_xtr/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 

report_xtr/dense_3/BiasAddBiasAddreport_xtr/dense_3/MatMul-mio_variable/report_xtr/dense_3/bias/variable*
data_formatNHWC*
T0
J
report_xtr/dense_3/SigmoidSigmoidreport_xtr/dense_3/BiasAdd*
T0
°
5mio_variable/depth_interact_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*.
	container!depth_interact_xtr/dense/kernel
°
5mio_variable/depth_interact_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!depth_interact_xtr/dense/kernel*
shape:
°
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
!Initializer_94/random_uniform/maxConst*
valueB
 *dF£=*
dtype0

+Initializer_94/random_uniform/RandomUniformRandomUniform#Initializer_94/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_94/random_uniform/subSub!Initializer_94/random_uniform/max!Initializer_94/random_uniform/min*
T0

!Initializer_94/random_uniform/mulMul+Initializer_94/random_uniform/RandomUniform!Initializer_94/random_uniform/sub*
T0
s
Initializer_94/random_uniformAdd!Initializer_94/random_uniform/mul!Initializer_94/random_uniform/min*
T0
å
	Assign_94Assign5mio_variable/depth_interact_xtr/dense/kernel/gradientInitializer_94/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/depth_interact_xtr/dense/kernel/gradient*
validate_shape(
§
3mio_variable/depth_interact_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerdepth_interact_xtr/dense/bias*
shape:
§
3mio_variable/depth_interact_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*,
	containerdepth_interact_xtr/dense/bias
F
Initializer_95/zerosConst*
valueB*    *
dtype0
Ø
	Assign_95Assign3mio_variable/depth_interact_xtr/dense/bias/gradientInitializer_95/zeros*
use_locking(*
T0*F
_class<
:8loc:@mio_variable/depth_interact_xtr/dense/bias/gradient*
validate_shape(

depth_interact_xtr/dense/MatMulMatMulconcat5mio_variable/depth_interact_xtr/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
¡
 depth_interact_xtr/dense/BiasAddBiasAdddepth_interact_xtr/dense/MatMul3mio_variable/depth_interact_xtr/dense/bias/variable*
T0*
data_formatNHWC
U
(depth_interact_xtr/dense/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0

&depth_interact_xtr/dense/LeakyRelu/mulMul(depth_interact_xtr/dense/LeakyRelu/alpha depth_interact_xtr/dense/BiasAdd*
T0

"depth_interact_xtr/dense/LeakyReluMaximum&depth_interact_xtr/dense/LeakyRelu/mul depth_interact_xtr/dense/BiasAdd*
T0
\
#depth_interact_xtr/dropout/IdentityIdentity"depth_interact_xtr/dense/LeakyRelu*
T0
´
7mio_variable/depth_interact_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!depth_interact_xtr/dense_1/kernel*
shape:

´
7mio_variable/depth_interact_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*0
	container#!depth_interact_xtr/dense_1/kernel
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
+Initializer_96/random_uniform/RandomUniformRandomUniform#Initializer_96/random_uniform/shape*
seed2 *

seed *
T0*
dtype0
w
!Initializer_96/random_uniform/subSub!Initializer_96/random_uniform/max!Initializer_96/random_uniform/min*
T0

!Initializer_96/random_uniform/mulMul+Initializer_96/random_uniform/RandomUniform!Initializer_96/random_uniform/sub*
T0
s
Initializer_96/random_uniformAdd!Initializer_96/random_uniform/mul!Initializer_96/random_uniform/min*
T0
é
	Assign_96Assign7mio_variable/depth_interact_xtr/dense_1/kernel/gradientInitializer_96/random_uniform*
use_locking(*
T0*J
_class@
><loc:@mio_variable/depth_interact_xtr/dense_1/kernel/gradient*
validate_shape(
«
5mio_variable/depth_interact_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*.
	container!depth_interact_xtr/dense_1/bias
«
5mio_variable/depth_interact_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!depth_interact_xtr/dense_1/bias*
shape:
F
Initializer_97/zerosConst*
valueB*    *
dtype0
Ü
	Assign_97Assign5mio_variable/depth_interact_xtr/dense_1/bias/gradientInitializer_97/zeros*
validate_shape(*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/depth_interact_xtr/dense_1/bias/gradient
¸
!depth_interact_xtr/dense_1/MatMulMatMul#depth_interact_xtr/dropout/Identity7mio_variable/depth_interact_xtr/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0
§
"depth_interact_xtr/dense_1/BiasAddBiasAdd!depth_interact_xtr/dense_1/MatMul5mio_variable/depth_interact_xtr/dense_1/bias/variable*
data_formatNHWC*
T0
W
*depth_interact_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0

(depth_interact_xtr/dense_1/LeakyRelu/mulMul*depth_interact_xtr/dense_1/LeakyRelu/alpha"depth_interact_xtr/dense_1/BiasAdd*
T0

$depth_interact_xtr/dense_1/LeakyReluMaximum(depth_interact_xtr/dense_1/LeakyRelu/mul"depth_interact_xtr/dense_1/BiasAdd*
T0
`
%depth_interact_xtr/dropout_1/IdentityIdentity$depth_interact_xtr/dense_1/LeakyRelu*
T0
³
7mio_variable/depth_interact_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*0
	container#!depth_interact_xtr/dense_2/kernel
³
7mio_variable/depth_interact_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!depth_interact_xtr/dense_2/kernel*
shape:	@
X
#Initializer_98/random_uniform/shapeConst*
valueB"   @   *
dtype0
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
é
	Assign_98Assign7mio_variable/depth_interact_xtr/dense_2/kernel/gradientInitializer_98/random_uniform*
use_locking(*
T0*J
_class@
><loc:@mio_variable/depth_interact_xtr/dense_2/kernel/gradient*
validate_shape(
ª
5mio_variable/depth_interact_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!depth_interact_xtr/dense_2/bias*
shape:@
ª
5mio_variable/depth_interact_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!depth_interact_xtr/dense_2/bias*
shape:@
E
Initializer_99/zerosConst*
valueB@*    *
dtype0
Ü
	Assign_99Assign5mio_variable/depth_interact_xtr/dense_2/bias/gradientInitializer_99/zeros*
validate_shape(*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/depth_interact_xtr/dense_2/bias/gradient
º
!depth_interact_xtr/dense_2/MatMulMatMul%depth_interact_xtr/dropout_1/Identity7mio_variable/depth_interact_xtr/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 
§
"depth_interact_xtr/dense_2/BiasAddBiasAdd!depth_interact_xtr/dense_2/MatMul5mio_variable/depth_interact_xtr/dense_2/bias/variable*
T0*
data_formatNHWC
W
*depth_interact_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0

(depth_interact_xtr/dense_2/LeakyRelu/mulMul*depth_interact_xtr/dense_2/LeakyRelu/alpha"depth_interact_xtr/dense_2/BiasAdd*
T0

$depth_interact_xtr/dense_2/LeakyReluMaximum(depth_interact_xtr/dense_2/LeakyRelu/mul"depth_interact_xtr/dense_2/BiasAdd*
T0
²
7mio_variable/depth_interact_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!depth_interact_xtr/dense_3/kernel*
shape
:@
²
7mio_variable/depth_interact_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*0
	container#!depth_interact_xtr/dense_3/kernel
Y
$Initializer_100/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_100/random_uniform/minConst*
valueB
 *¾*
dtype0
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
ë

Assign_100Assign7mio_variable/depth_interact_xtr/dense_3/kernel/gradientInitializer_100/random_uniform*
use_locking(*
T0*J
_class@
><loc:@mio_variable/depth_interact_xtr/dense_3/kernel/gradient*
validate_shape(
ª
5mio_variable/depth_interact_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!depth_interact_xtr/dense_3/bias*
shape:
ª
5mio_variable/depth_interact_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!depth_interact_xtr/dense_3/bias*
shape:
F
Initializer_101/zerosConst*
valueB*    *
dtype0
Þ

Assign_101Assign5mio_variable/depth_interact_xtr/dense_3/bias/gradientInitializer_101/zeros*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/depth_interact_xtr/dense_3/bias/gradient*
validate_shape(
¹
!depth_interact_xtr/dense_3/MatMulMatMul$depth_interact_xtr/dense_2/LeakyRelu7mio_variable/depth_interact_xtr/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 
§
"depth_interact_xtr/dense_3/BiasAddBiasAdd!depth_interact_xtr/dense_3/MatMul5mio_variable/depth_interact_xtr/dense_3/bias/variable*
data_formatNHWC*
T0
Z
"depth_interact_xtr/dense_3/SigmoidSigmoid"depth_interact_xtr/dense_3/BiasAdd*
T0
2
sub/xConst*
valueB
 *  ?*
dtype0
5
subSubsub/xdepth_xtr/dense_3/Sigmoid*
T0
;
truedivRealDivdepth_xtr/dense_3/Sigmoidsub*
T0
4
sub_1/xConst*
valueB
 *  ?*
dtype0
B
sub_1Subsub_1/x"depth_interact_xtr/dense_3/Sigmoid*
T0
H
	truediv_1RealDiv"depth_interact_xtr/dense_3/Sigmoidsub_1*
T0"