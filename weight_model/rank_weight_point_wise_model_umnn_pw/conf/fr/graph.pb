
/
ConstConst*
value	B : *
dtype0
K
MIO_TABLE_ADDRESSConst"/device:CPU:0*
value
B � *
dtype0
�
2mio_compress_indices/COMPRESS_INDEX__USER/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerCOMPRESS_INDEX__USER*
shape:���������
�
2mio_compress_indices/COMPRESS_INDEX__USER/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:���������*#
	containerCOMPRESS_INDEX__USER
h
CastCast2mio_compress_indices/COMPRESS_INDEX__USER/variable*
Truncate( *

DstT0*

SrcT0
�
mio_embeddings/uid_emb/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	container	uid_emb*
shape:���������@
�
mio_embeddings/uid_emb/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	container	uid_emb*
shape:���������@
�
 mio_embeddings/uid_stat/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:���������*
	container
uid_stat
�
 mio_embeddings/uid_stat/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	container
uid_stat*
shape:���������
�
 mio_embeddings/did_stat/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:���������0*
	container
did_stat
�
 mio_embeddings/did_stat/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:���������0*
	container
did_stat
�
#mio_embeddings/u_mean_stat/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containeru_mean_stat*
shape:���������0
�
#mio_embeddings/u_mean_stat/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containeru_mean_stat*
shape:���������0
�
"mio_embeddings/u_std_stat/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	container
u_std_stat*
shape:���������0
�
"mio_embeddings/u_std_stat/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	container
u_std_stat*
shape:���������0
�
mio_embeddings/pid_emb/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:���������@*
	container	pid_emb
�
mio_embeddings/pid_emb/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:���������@*
	container	pid_emb
�
mio_embeddings/pid_xtr/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	container	pid_xtr*
shape:���������H
�
mio_embeddings/pid_xtr/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	container	pid_xtr*
shape:���������H
�
 mio_embeddings/pid_stat/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	container
pid_stat*
shape:���������@
�
 mio_embeddings/pid_stat/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	container
pid_stat*
shape:���������@
�
 mio_embeddings/pid_gate/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:��������� *
	container
pid_gate
�
 mio_embeddings/pid_gate/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:��������� *
	container
pid_gate
�
 mio_embeddings/pid_pxtr/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:���������`*
	container
pid_pxtr
�
 mio_embeddings/pid_pxtr/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	container
pid_pxtr*
shape:���������`
�
 mio_embeddings/top_bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:��������� *
	container
top_bias
�
 mio_embeddings/top_bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:��������� *
	container
top_bias
�
&mio_embeddings/photo_category/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerphoto_category*
shape:���������
�
&mio_embeddings/photo_category/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:���������*
	containerphoto_category
�
)mio_embeddings/uid_action_list_1/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS* 
	containeruid_action_list_1*
shape:����������
�
)mio_embeddings/uid_action_list_1/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS* 
	containeruid_action_list_1*
shape:����������
=
Reshape/tensor/axisConst*
dtype0*
value	B : 
�
Reshape/tensorGatherV2)mio_embeddings/uid_action_list_1/variableCastReshape/tensor/axis*
Taxis0*
Tindices0*
Tparams0
F
Reshape/shapeConst*!
valueB"����      *
dtype0
H
ReshapeReshapeReshape/tensorReshape/shape*
T0*
Tshape0
�
)mio_embeddings/uid_action_list_2/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS* 
	containeruid_action_list_2*
shape:����������
�
)mio_embeddings/uid_action_list_2/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS* 
	containeruid_action_list_2*
shape:����������
?
Reshape_1/tensor/axisConst*
value	B : *
dtype0
�
Reshape_1/tensorGatherV2)mio_embeddings/uid_action_list_2/variableCastReshape_1/tensor/axis*
Tindices0*
Tparams0*
Taxis0
H
Reshape_1/shapeConst*!
valueB"����      *
dtype0
N
	Reshape_1ReshapeReshape_1/tensorReshape_1/shape*
T0*
Tshape0
�
)mio_embeddings/uid_action_list_3/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS* 
	containeruid_action_list_3*
shape:����������
�
)mio_embeddings/uid_action_list_3/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS* 
	containeruid_action_list_3*
shape:����������
?
Reshape_2/tensor/axisConst*
value	B : *
dtype0
�
Reshape_2/tensorGatherV2)mio_embeddings/uid_action_list_3/variableCastReshape_2/tensor/axis*
Tparams0*
Taxis0*
Tindices0
H
Reshape_2/shapeConst*!
valueB"����      *
dtype0
N
	Reshape_2ReshapeReshape_2/tensorReshape_2/shape*
T0*
Tshape0
�
)mio_embeddings/uid_action_list_4/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:����������* 
	containeruid_action_list_4
�
)mio_embeddings/uid_action_list_4/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS* 
	containeruid_action_list_4*
shape:����������
?
Reshape_3/tensor/axisConst*
value	B : *
dtype0
�
Reshape_3/tensorGatherV2)mio_embeddings/uid_action_list_4/variableCastReshape_3/tensor/axis*
Tindices0*
Tparams0*
Taxis0
H
Reshape_3/shapeConst*
dtype0*!
valueB"����      
N
	Reshape_3ReshapeReshape_3/tensorReshape_3/shape*
T0*
Tshape0
�
)mio_embeddings/uid_action_list_5/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS* 
	containeruid_action_list_5*
shape:����������
�
)mio_embeddings/uid_action_list_5/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS* 
	containeruid_action_list_5*
shape:����������
?
Reshape_4/tensor/axisConst*
value	B : *
dtype0
�
Reshape_4/tensorGatherV2)mio_embeddings/uid_action_list_5/variableCastReshape_4/tensor/axis*
Taxis0*
Tindices0*
Tparams0
H
Reshape_4/shapeConst*
dtype0*!
valueB"����      
N
	Reshape_4ReshapeReshape_4/tensorReshape_4/shape*
T0*
Tshape0
�
)mio_embeddings/uid_action_list_6/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:����������* 
	containeruid_action_list_6
�
)mio_embeddings/uid_action_list_6/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:����������* 
	containeruid_action_list_6
?
Reshape_5/tensor/axisConst*
dtype0*
value	B : 
�
Reshape_5/tensorGatherV2)mio_embeddings/uid_action_list_6/variableCastReshape_5/tensor/axis*
Taxis0*
Tindices0*
Tparams0
H
Reshape_5/shapeConst*!
valueB"����      *
dtype0
N
	Reshape_5ReshapeReshape_5/tensorReshape_5/shape*
T0*
Tshape0
�
)mio_embeddings/uid_action_list_7/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:����������* 
	containeruid_action_list_7
�
)mio_embeddings/uid_action_list_7/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS* 
	containeruid_action_list_7*
shape:����������
?
Reshape_6/tensor/axisConst*
value	B : *
dtype0
�
Reshape_6/tensorGatherV2)mio_embeddings/uid_action_list_7/variableCastReshape_6/tensor/axis*
Tparams0*
Taxis0*
Tindices0
H
Reshape_6/shapeConst*!
valueB"����      *
dtype0
N
	Reshape_6ReshapeReshape_6/tensorReshape_6/shape*
T0*
Tshape0
�
)mio_embeddings/uid_action_list_8/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:����������* 
	containeruid_action_list_8
�
)mio_embeddings/uid_action_list_8/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS* 
	containeruid_action_list_8*
shape:����������
?
Reshape_7/tensor/axisConst*
value	B : *
dtype0
�
Reshape_7/tensorGatherV2)mio_embeddings/uid_action_list_8/variableCastReshape_7/tensor/axis*
Tindices0*
Tparams0*
Taxis0
H
Reshape_7/shapeConst*!
valueB"����      *
dtype0
N
	Reshape_7ReshapeReshape_7/tensorReshape_7/shape*
T0*
Tshape0
�
)mio_embeddings/uid_action_list_9/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS* 
	containeruid_action_list_9*
shape:����������
�
)mio_embeddings/uid_action_list_9/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS* 
	containeruid_action_list_9*
shape:����������
?
Reshape_8/tensor/axisConst*
value	B : *
dtype0
�
Reshape_8/tensorGatherV2)mio_embeddings/uid_action_list_9/variableCastReshape_8/tensor/axis*
Taxis0*
Tindices0*
Tparams0
H
Reshape_8/shapeConst*!
valueB"����      *
dtype0
N
	Reshape_8ReshapeReshape_8/tensorReshape_8/shape*
T0*
Tshape0
�
*mio_embeddings/uid_action_list_10/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_action_list_10*
shape:����������
�
*mio_embeddings/uid_action_list_10/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_action_list_10*
shape:����������
?
Reshape_9/tensor/axisConst*
value	B : *
dtype0
�
Reshape_9/tensorGatherV2*mio_embeddings/uid_action_list_10/variableCastReshape_9/tensor/axis*
Taxis0*
Tindices0*
Tparams0
H
Reshape_9/shapeConst*!
valueB"����      *
dtype0
N
	Reshape_9ReshapeReshape_9/tensorReshape_9/shape*
T0*
Tshape0
�
*mio_embeddings/uid_action_list_11/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_action_list_11*
shape:����������
�
*mio_embeddings/uid_action_list_11/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_action_list_11*
shape:����������
@
Reshape_10/tensor/axisConst*
value	B : *
dtype0
�
Reshape_10/tensorGatherV2*mio_embeddings/uid_action_list_11/variableCastReshape_10/tensor/axis*
Taxis0*
Tindices0*
Tparams0
I
Reshape_10/shapeConst*
dtype0*!
valueB"����      
Q

Reshape_10ReshapeReshape_10/tensorReshape_10/shape*
T0*
Tshape0
�
*mio_embeddings/uid_action_list_12/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_action_list_12*
shape:����������
�
*mio_embeddings/uid_action_list_12/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_action_list_12*
shape:����������
@
Reshape_11/tensor/axisConst*
value	B : *
dtype0
�
Reshape_11/tensorGatherV2*mio_embeddings/uid_action_list_12/variableCastReshape_11/tensor/axis*
Taxis0*
Tindices0*
Tparams0
I
Reshape_11/shapeConst*!
valueB"����      *
dtype0
Q

Reshape_11ReshapeReshape_11/tensorReshape_11/shape*
T0*
Tshape0
�
*mio_embeddings/uid_action_list_13/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_action_list_13*
shape:����������
�
*mio_embeddings/uid_action_list_13/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_action_list_13*
shape:����������
@
Reshape_12/tensor/axisConst*
dtype0*
value	B : 
�
Reshape_12/tensorGatherV2*mio_embeddings/uid_action_list_13/variableCastReshape_12/tensor/axis*
Tparams0*
Taxis0*
Tindices0
I
Reshape_12/shapeConst*!
valueB"����      *
dtype0
Q

Reshape_12ReshapeReshape_12/tensorReshape_12/shape*
T0*
Tshape0
�
*mio_embeddings/uid_action_list_14/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:����������*!
	containeruid_action_list_14
�
*mio_embeddings/uid_action_list_14/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_action_list_14*
shape:����������
@
Reshape_13/tensor/axisConst*
dtype0*
value	B : 
�
Reshape_13/tensorGatherV2*mio_embeddings/uid_action_list_14/variableCastReshape_13/tensor/axis*
Taxis0*
Tindices0*
Tparams0
I
Reshape_13/shapeConst*
dtype0*!
valueB"����      
Q

Reshape_13ReshapeReshape_13/tensorReshape_13/shape*
T0*
Tshape0
�
*mio_embeddings/uid_action_list_15/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:����������*!
	containeruid_action_list_15
�
*mio_embeddings/uid_action_list_15/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_action_list_15*
shape:����������
@
Reshape_14/tensor/axisConst*
value	B : *
dtype0
�
Reshape_14/tensorGatherV2*mio_embeddings/uid_action_list_15/variableCastReshape_14/tensor/axis*
Tindices0*
Tparams0*
Taxis0
I
Reshape_14/shapeConst*!
valueB"����      *
dtype0
Q

Reshape_14ReshapeReshape_14/tensorReshape_14/shape*
T0*
Tshape0
�
*mio_embeddings/uid_action_list_16/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_action_list_16*
shape:����������
�
*mio_embeddings/uid_action_list_16/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_action_list_16*
shape:����������
@
Reshape_15/tensor/axisConst*
value	B : *
dtype0
�
Reshape_15/tensorGatherV2*mio_embeddings/uid_action_list_16/variableCastReshape_15/tensor/axis*
Taxis0*
Tindices0*
Tparams0
I
Reshape_15/shapeConst*!
valueB"����      *
dtype0
Q

Reshape_15ReshapeReshape_15/tensorReshape_15/shape*
T0*
Tshape0
�
*mio_embeddings/uid_action_list_17/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_action_list_17*
shape:����������
�
*mio_embeddings/uid_action_list_17/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_action_list_17*
shape:����������
@
Reshape_16/tensor/axisConst*
dtype0*
value	B : 
�
Reshape_16/tensorGatherV2*mio_embeddings/uid_action_list_17/variableCastReshape_16/tensor/axis*
Taxis0*
Tindices0*
Tparams0
I
Reshape_16/shapeConst*!
valueB"����      *
dtype0
Q

Reshape_16ReshapeReshape_16/tensorReshape_16/shape*
T0*
Tshape0
�
*mio_embeddings/uid_action_list_18/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_action_list_18*
shape:����������
�
*mio_embeddings/uid_action_list_18/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_action_list_18*
shape:����������
@
Reshape_17/tensor/axisConst*
dtype0*
value	B : 
�
Reshape_17/tensorGatherV2*mio_embeddings/uid_action_list_18/variableCastReshape_17/tensor/axis*
Taxis0*
Tindices0*
Tparams0
I
Reshape_17/shapeConst*
dtype0*!
valueB"����      
Q

Reshape_17ReshapeReshape_17/tensorReshape_17/shape*
T0*
Tshape0
�
*mio_embeddings/uid_action_list_19/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_action_list_19*
shape:����������
�
*mio_embeddings/uid_action_list_19/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:����������*!
	containeruid_action_list_19
@
Reshape_18/tensor/axisConst*
value	B : *
dtype0
�
Reshape_18/tensorGatherV2*mio_embeddings/uid_action_list_19/variableCastReshape_18/tensor/axis*
Taxis0*
Tindices0*
Tparams0
I
Reshape_18/shapeConst*
dtype0*!
valueB"����      
Q

Reshape_18ReshapeReshape_18/tensorReshape_18/shape*
T0*
Tshape0
�
*mio_embeddings/uid_action_list_20/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_action_list_20*
shape:����������
�
*mio_embeddings/uid_action_list_20/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:����������*!
	containeruid_action_list_20
@
Reshape_19/tensor/axisConst*
value	B : *
dtype0
�
Reshape_19/tensorGatherV2*mio_embeddings/uid_action_list_20/variableCastReshape_19/tensor/axis*
Taxis0*
Tindices0*
Tparams0
I
Reshape_19/shapeConst*!
valueB"����      *
dtype0
Q

Reshape_19ReshapeReshape_19/tensorReshape_19/shape*
T0*
Tshape0
�
*mio_embeddings/uid_action_list_21/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_action_list_21*
shape:����������
�
*mio_embeddings/uid_action_list_21/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:����������*!
	containeruid_action_list_21
@
Reshape_20/tensor/axisConst*
value	B : *
dtype0
�
Reshape_20/tensorGatherV2*mio_embeddings/uid_action_list_21/variableCastReshape_20/tensor/axis*
Taxis0*
Tindices0*
Tparams0
I
Reshape_20/shapeConst*
dtype0*!
valueB"����      
Q

Reshape_20ReshapeReshape_20/tensorReshape_20/shape*
T0*
Tshape0
5
concat/axisConst*
value	B :*
dtype0
�
concatConcatV2Reshape	Reshape_1	Reshape_2	Reshape_3	Reshape_4	Reshape_5	Reshape_6	Reshape_7	Reshape_8	Reshape_9
Reshape_10
Reshape_11
Reshape_12
Reshape_13
Reshape_14
Reshape_15
Reshape_16
Reshape_17
Reshape_18
Reshape_19
Reshape_20concat/axis*

Tidx0*
T0*
N
@
Mean/reduction_indicesConst*
value	B :*
dtype0
R
MeanMeanconcatMean/reduction_indices*

Tidx0*
	keep_dims( *
T0
�
.mio_variable/seq_encoder/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	� *'
	containerseq_encoder/dense/kernel
�
.mio_variable/seq_encoder/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerseq_encoder/dense/kernel*
shape:	� 
W
"Initializer/truncated_normal/shapeConst*
valueB"�       *
dtype0
N
!Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0
P
#Initializer/truncated_normal/stddevConst*
valueB
 *   ?*
dtype0
�
,Initializer/truncated_normal/TruncatedNormalTruncatedNormal"Initializer/truncated_normal/shape*
dtype0*
seed2*
seed���)*
T0
�
 Initializer/truncated_normal/mulMul,Initializer/truncated_normal/TruncatedNormal#Initializer/truncated_normal/stddev*
T0
q
Initializer/truncated_normalAdd Initializer/truncated_normal/mul!Initializer/truncated_normal/mean*
T0
�
AssignAssign.mio_variable/seq_encoder/dense/kernel/gradientInitializer/truncated_normal*
validate_shape(*
use_locking(*
T0*A
_class7
53loc:@mio_variable/seq_encoder/dense/kernel/gradient
�
,mio_variable/seq_encoder/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape: *%
	containerseq_encoder/dense/bias
�
,mio_variable/seq_encoder/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerseq_encoder/dense/bias*
shape: 
D
Initializer_1/zerosConst*
valueB *    *
dtype0
�
Assign_1Assign,mio_variable/seq_encoder/dense/bias/gradientInitializer_1/zeros*
T0*?
_class5
31loc:@mio_variable/seq_encoder/dense/bias/gradient*
validate_shape(*
use_locking(
�
seq_encoder/dense/MatMulMatMulMean.mio_variable/seq_encoder/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
seq_encoder/dense/BiasAddBiasAddseq_encoder/dense/MatMul,mio_variable/seq_encoder/dense/bias/variable*
T0*
data_formatNHWC
N
!seq_encoder/dense/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
m
seq_encoder/dense/LeakyRelu/mulMul!seq_encoder/dense/LeakyRelu/alphaseq_encoder/dense/BiasAdd*
T0
k
seq_encoder/dense/LeakyReluMaximumseq_encoder/dense/LeakyRelu/mulseq_encoder/dense/BiasAdd*
T0
�
mio_extra_param/pltr/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:���������*
	containerpltr
�
mio_extra_param/pltr/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerpltr*
shape:���������
�
mio_extra_param/pwtr/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerpwtr*
shape:���������
�
mio_extra_param/pwtr/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerpwtr*
shape:���������
�
mio_extra_param/pftr/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerpftr*
shape:���������
�
mio_extra_param/pftr/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerpftr*
shape:���������
�
mio_extra_param/pcmtr/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerpcmtr*
shape:���������
�
mio_extra_param/pcmtr/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerpcmtr*
shape:���������
�
mio_extra_param/plvtr/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerplvtr*
shape:���������
�
mio_extra_param/plvtr/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerplvtr*
shape:���������
�
mio_extra_param/pctr/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerpctr*
shape:���������
�
mio_extra_param/pctr/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerpctr*
shape:���������
7
concat_1/axisConst*
dtype0*
value	B :
�
concat_1ConcatV2mio_extra_param/pltr/variablemio_extra_param/pwtr/variablemio_extra_param/pftr/variablemio_extra_param/pcmtr/variablemio_extra_param/plvtr/variablemio_extra_param/pctr/variableconcat_1/axis*
T0*
N*

Tidx0
@
concat_2/values_1/axisConst*
value	B : *
dtype0
�
concat_2/values_1GatherV2#mio_embeddings/u_mean_stat/variableCastconcat_2/values_1/axis*
Taxis0*
Tindices0*
Tparams0
@
concat_2/values_2/axisConst*
value	B : *
dtype0
�
concat_2/values_2GatherV2"mio_embeddings/u_std_stat/variableCastconcat_2/values_2/axis*
Taxis0*
Tindices0*
Tparams0
7
concat_2/axisConst*
value	B :*
dtype0
�
concat_2ConcatV2seq_encoder/dense/LeakyReluconcat_2/values_1concat_2/values_2concat_2/axis*
T0*
N*

Tidx0
�
3mio_variable/intent_predictor/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerintent_predictor/dense/kernel*
shape:	�
�
3mio_variable/intent_predictor/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerintent_predictor/dense/kernel*
shape:	�
Y
$Initializer_2/truncated_normal/shapeConst*
dtype0*
valueB"�      
P
#Initializer_2/truncated_normal/meanConst*
dtype0*
valueB
 *    
R
%Initializer_2/truncated_normal/stddevConst*
valueB
 *   ?*
dtype0
�
.Initializer_2/truncated_normal/TruncatedNormalTruncatedNormal$Initializer_2/truncated_normal/shape*
T0*
dtype0*
seed2*
seed���)
�
"Initializer_2/truncated_normal/mulMul.Initializer_2/truncated_normal/TruncatedNormal%Initializer_2/truncated_normal/stddev*
T0
w
Initializer_2/truncated_normalAdd"Initializer_2/truncated_normal/mul#Initializer_2/truncated_normal/mean*
T0
�
Assign_2Assign3mio_variable/intent_predictor/dense/kernel/gradientInitializer_2/truncated_normal*
T0*F
_class<
:8loc:@mio_variable/intent_predictor/dense/kernel/gradient*
validate_shape(*
use_locking(
�
1mio_variable/intent_predictor/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS**
	containerintent_predictor/dense/bias*
shape:
�
1mio_variable/intent_predictor/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS**
	containerintent_predictor/dense/bias*
shape:
D
Initializer_3/zerosConst*
valueB*    *
dtype0
�
Assign_3Assign1mio_variable/intent_predictor/dense/bias/gradientInitializer_3/zeros*
use_locking(*
T0*D
_class:
86loc:@mio_variable/intent_predictor/dense/bias/gradient*
validate_shape(
�
intent_predictor/dense/MatMulMatMulconcat_23mio_variable/intent_predictor/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
intent_predictor/dense/BiasAddBiasAddintent_predictor/dense/MatMul1mio_variable/intent_predictor/dense/bias/variable*
data_formatNHWC*
T0
R
intent_predictor/dense/SoftmaxSoftmaxintent_predictor/dense/BiasAdd*
T0
�
-mio_variable/intent_emb/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:*&
	containerintent_emb/dense/kernel
�
-mio_variable/intent_emb/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerintent_emb/dense/kernel*
shape
:
Y
$Initializer_4/truncated_normal/shapeConst*
dtype0*
valueB"      
P
#Initializer_4/truncated_normal/meanConst*
valueB
 *    *
dtype0
R
%Initializer_4/truncated_normal/stddevConst*
valueB
 *   ?*
dtype0
�
.Initializer_4/truncated_normal/TruncatedNormalTruncatedNormal$Initializer_4/truncated_normal/shape*
T0*
dtype0*
seed2*
seed���)
�
"Initializer_4/truncated_normal/mulMul.Initializer_4/truncated_normal/TruncatedNormal%Initializer_4/truncated_normal/stddev*
T0
w
Initializer_4/truncated_normalAdd"Initializer_4/truncated_normal/mul#Initializer_4/truncated_normal/mean*
T0
�
Assign_4Assign-mio_variable/intent_emb/dense/kernel/gradientInitializer_4/truncated_normal*
use_locking(*
T0*@
_class6
42loc:@mio_variable/intent_emb/dense/kernel/gradient*
validate_shape(
�
+mio_variable/intent_emb/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerintent_emb/dense/bias*
shape:
�
+mio_variable/intent_emb/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerintent_emb/dense/bias*
shape:
D
Initializer_5/zerosConst*
dtype0*
valueB*    
�
Assign_5Assign+mio_variable/intent_emb/dense/bias/gradientInitializer_5/zeros*
T0*>
_class4
20loc:@mio_variable/intent_emb/dense/bias/gradient*
validate_shape(*
use_locking(
�
intent_emb/dense/MatMulMatMulintent_predictor/dense/Softmax-mio_variable/intent_emb/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
intent_emb/dense/BiasAddBiasAddintent_emb/dense/MatMul+mio_variable/intent_emb/dense/bias/variable*
T0*
data_formatNHWC
F
intent_emb/dense/SigmoidSigmoidintent_emb/dense/BiasAdd*
T0
8
ExpandDims/dimConst*
value	B :*
dtype0
G

ExpandDims
ExpandDimsconcat_1ExpandDims/dim*

Tdim0*
T0
:
ExpandDims_1/dimConst*
value	B :*
dtype0
[
ExpandDims_1
ExpandDimsintent_emb/dense/SigmoidExpandDims_1/dim*

Tdim0*
T0
:
ExpandDims_2/dimConst*
value	B :*
dtype0
i
ExpandDims_2
ExpandDims&mio_embeddings/photo_category/variableExpandDims_2/dim*

Tdim0*
T0
�
6mio_variable/pxtr_self_attention/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" pxtr_self_attention/dense/kernel*
shape
:
�
6mio_variable/pxtr_self_attention/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" pxtr_self_attention/dense/kernel*
shape
:
W
"Initializer_6/random_uniform/shapeConst*
valueB"      *
dtype0
M
 Initializer_6/random_uniform/minConst*
valueB
 *��*
dtype0
M
 Initializer_6/random_uniform/maxConst*
valueB
 *�?*
dtype0
�
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
�
Assign_6Assign6mio_variable/pxtr_self_attention/dense/kernel/gradientInitializer_6/random_uniform*
use_locking(*
T0*I
_class?
=;loc:@mio_variable/pxtr_self_attention/dense/kernel/gradient*
validate_shape(
V
(pxtr_self_attention/dense/Tensordot/axesConst*
valueB:*
dtype0
]
(pxtr_self_attention/dense/Tensordot/freeConst*
dtype0*
valueB"       
W
)pxtr_self_attention/dense/Tensordot/ShapeShape
ExpandDims*
T0*
out_type0
[
1pxtr_self_attention/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0
�
,pxtr_self_attention/dense/Tensordot/GatherV2GatherV2)pxtr_self_attention/dense/Tensordot/Shape(pxtr_self_attention/dense/Tensordot/free1pxtr_self_attention/dense/Tensordot/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0
]
3pxtr_self_attention/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0
�
.pxtr_self_attention/dense/Tensordot/GatherV2_1GatherV2)pxtr_self_attention/dense/Tensordot/Shape(pxtr_self_attention/dense/Tensordot/axes3pxtr_self_attention/dense/Tensordot/GatherV2_1/axis*
Tparams0*
Taxis0*
Tindices0
W
)pxtr_self_attention/dense/Tensordot/ConstConst*
valueB: *
dtype0
�
(pxtr_self_attention/dense/Tensordot/ProdProd,pxtr_self_attention/dense/Tensordot/GatherV2)pxtr_self_attention/dense/Tensordot/Const*
T0*

Tidx0*
	keep_dims( 
Y
+pxtr_self_attention/dense/Tensordot/Const_1Const*
dtype0*
valueB: 
�
*pxtr_self_attention/dense/Tensordot/Prod_1Prod.pxtr_self_attention/dense/Tensordot/GatherV2_1+pxtr_self_attention/dense/Tensordot/Const_1*
T0*

Tidx0*
	keep_dims( 
Y
/pxtr_self_attention/dense/Tensordot/concat/axisConst*
dtype0*
value	B : 
�
*pxtr_self_attention/dense/Tensordot/concatConcatV2(pxtr_self_attention/dense/Tensordot/free(pxtr_self_attention/dense/Tensordot/axes/pxtr_self_attention/dense/Tensordot/concat/axis*
T0*
N*

Tidx0
�
)pxtr_self_attention/dense/Tensordot/stackPack(pxtr_self_attention/dense/Tensordot/Prod*pxtr_self_attention/dense/Tensordot/Prod_1*
N*
T0*

axis 
�
-pxtr_self_attention/dense/Tensordot/transpose	Transpose
ExpandDims*pxtr_self_attention/dense/Tensordot/concat*
Tperm0*
T0
�
+pxtr_self_attention/dense/Tensordot/ReshapeReshape-pxtr_self_attention/dense/Tensordot/transpose)pxtr_self_attention/dense/Tensordot/stack*
T0*
Tshape0
i
4pxtr_self_attention/dense/Tensordot/transpose_1/permConst*
dtype0*
valueB"       
�
/pxtr_self_attention/dense/Tensordot/transpose_1	Transpose6mio_variable/pxtr_self_attention/dense/kernel/variable4pxtr_self_attention/dense/Tensordot/transpose_1/perm*
Tperm0*
T0
h
3pxtr_self_attention/dense/Tensordot/Reshape_1/shapeConst*
dtype0*
valueB"      
�
-pxtr_self_attention/dense/Tensordot/Reshape_1Reshape/pxtr_self_attention/dense/Tensordot/transpose_13pxtr_self_attention/dense/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
*pxtr_self_attention/dense/Tensordot/MatMulMatMul+pxtr_self_attention/dense/Tensordot/Reshape-pxtr_self_attention/dense/Tensordot/Reshape_1*
transpose_a( *
transpose_b( *
T0
Y
+pxtr_self_attention/dense/Tensordot/Const_2Const*
valueB:*
dtype0
[
1pxtr_self_attention/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0
�
,pxtr_self_attention/dense/Tensordot/concat_1ConcatV2,pxtr_self_attention/dense/Tensordot/GatherV2+pxtr_self_attention/dense/Tensordot/Const_21pxtr_self_attention/dense/Tensordot/concat_1/axis*
T0*
N*

Tidx0
�
#pxtr_self_attention/dense/TensordotReshape*pxtr_self_attention/dense/Tensordot/MatMul,pxtr_self_attention/dense/Tensordot/concat_1*
T0*
Tshape0
�
8mio_variable/pxtr_self_attention/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:*1
	container$"pxtr_self_attention/dense_1/kernel
�
8mio_variable/pxtr_self_attention/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"pxtr_self_attention/dense_1/kernel*
shape
:
W
"Initializer_7/random_uniform/shapeConst*
dtype0*
valueB"      
M
 Initializer_7/random_uniform/minConst*
valueB
 *��*
dtype0
M
 Initializer_7/random_uniform/maxConst*
valueB
 *�?*
dtype0
�
*Initializer_7/random_uniform/RandomUniformRandomUniform"Initializer_7/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
t
 Initializer_7/random_uniform/subSub Initializer_7/random_uniform/max Initializer_7/random_uniform/min*
T0
~
 Initializer_7/random_uniform/mulMul*Initializer_7/random_uniform/RandomUniform Initializer_7/random_uniform/sub*
T0
p
Initializer_7/random_uniformAdd Initializer_7/random_uniform/mul Initializer_7/random_uniform/min*
T0
�
Assign_7Assign8mio_variable/pxtr_self_attention/dense_1/kernel/gradientInitializer_7/random_uniform*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/pxtr_self_attention/dense_1/kernel/gradient*
validate_shape(
X
*pxtr_self_attention/dense_1/Tensordot/axesConst*
valueB:*
dtype0
_
*pxtr_self_attention/dense_1/Tensordot/freeConst*
valueB"       *
dtype0
Y
+pxtr_self_attention/dense_1/Tensordot/ShapeShape
ExpandDims*
T0*
out_type0
]
3pxtr_self_attention/dense_1/Tensordot/GatherV2/axisConst*
value	B : *
dtype0
�
.pxtr_self_attention/dense_1/Tensordot/GatherV2GatherV2+pxtr_self_attention/dense_1/Tensordot/Shape*pxtr_self_attention/dense_1/Tensordot/free3pxtr_self_attention/dense_1/Tensordot/GatherV2/axis*
Tparams0*
Taxis0*
Tindices0
_
5pxtr_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0
�
0pxtr_self_attention/dense_1/Tensordot/GatherV2_1GatherV2+pxtr_self_attention/dense_1/Tensordot/Shape*pxtr_self_attention/dense_1/Tensordot/axes5pxtr_self_attention/dense_1/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0
Y
+pxtr_self_attention/dense_1/Tensordot/ConstConst*
valueB: *
dtype0
�
*pxtr_self_attention/dense_1/Tensordot/ProdProd.pxtr_self_attention/dense_1/Tensordot/GatherV2+pxtr_self_attention/dense_1/Tensordot/Const*

Tidx0*
	keep_dims( *
T0
[
-pxtr_self_attention/dense_1/Tensordot/Const_1Const*
dtype0*
valueB: 
�
,pxtr_self_attention/dense_1/Tensordot/Prod_1Prod0pxtr_self_attention/dense_1/Tensordot/GatherV2_1-pxtr_self_attention/dense_1/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0
[
1pxtr_self_attention/dense_1/Tensordot/concat/axisConst*
value	B : *
dtype0
�
,pxtr_self_attention/dense_1/Tensordot/concatConcatV2*pxtr_self_attention/dense_1/Tensordot/free*pxtr_self_attention/dense_1/Tensordot/axes1pxtr_self_attention/dense_1/Tensordot/concat/axis*

Tidx0*
T0*
N
�
+pxtr_self_attention/dense_1/Tensordot/stackPack*pxtr_self_attention/dense_1/Tensordot/Prod,pxtr_self_attention/dense_1/Tensordot/Prod_1*
T0*

axis *
N
�
/pxtr_self_attention/dense_1/Tensordot/transpose	Transpose
ExpandDims,pxtr_self_attention/dense_1/Tensordot/concat*
Tperm0*
T0
�
-pxtr_self_attention/dense_1/Tensordot/ReshapeReshape/pxtr_self_attention/dense_1/Tensordot/transpose+pxtr_self_attention/dense_1/Tensordot/stack*
T0*
Tshape0
k
6pxtr_self_attention/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
1pxtr_self_attention/dense_1/Tensordot/transpose_1	Transpose8mio_variable/pxtr_self_attention/dense_1/kernel/variable6pxtr_self_attention/dense_1/Tensordot/transpose_1/perm*
T0*
Tperm0
j
5pxtr_self_attention/dense_1/Tensordot/Reshape_1/shapeConst*
valueB"      *
dtype0
�
/pxtr_self_attention/dense_1/Tensordot/Reshape_1Reshape1pxtr_self_attention/dense_1/Tensordot/transpose_15pxtr_self_attention/dense_1/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
,pxtr_self_attention/dense_1/Tensordot/MatMulMatMul-pxtr_self_attention/dense_1/Tensordot/Reshape/pxtr_self_attention/dense_1/Tensordot/Reshape_1*
transpose_b( *
T0*
transpose_a( 
[
-pxtr_self_attention/dense_1/Tensordot/Const_2Const*
valueB:*
dtype0
]
3pxtr_self_attention/dense_1/Tensordot/concat_1/axisConst*
value	B : *
dtype0
�
.pxtr_self_attention/dense_1/Tensordot/concat_1ConcatV2.pxtr_self_attention/dense_1/Tensordot/GatherV2-pxtr_self_attention/dense_1/Tensordot/Const_23pxtr_self_attention/dense_1/Tensordot/concat_1/axis*
T0*
N*

Tidx0
�
%pxtr_self_attention/dense_1/TensordotReshape,pxtr_self_attention/dense_1/Tensordot/MatMul.pxtr_self_attention/dense_1/Tensordot/concat_1*
T0*
Tshape0
�
8mio_variable/pxtr_self_attention/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"pxtr_self_attention/dense_2/kernel*
shape
:
�
8mio_variable/pxtr_self_attention/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:*1
	container$"pxtr_self_attention/dense_2/kernel
W
"Initializer_8/random_uniform/shapeConst*
dtype0*
valueB"      
M
 Initializer_8/random_uniform/minConst*
valueB
 *��*
dtype0
M
 Initializer_8/random_uniform/maxConst*
dtype0*
valueB
 *�?
�
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
�
Assign_8Assign8mio_variable/pxtr_self_attention/dense_2/kernel/gradientInitializer_8/random_uniform*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/pxtr_self_attention/dense_2/kernel/gradient*
validate_shape(
X
*pxtr_self_attention/dense_2/Tensordot/axesConst*
valueB:*
dtype0
_
*pxtr_self_attention/dense_2/Tensordot/freeConst*
valueB"       *
dtype0
Y
+pxtr_self_attention/dense_2/Tensordot/ShapeShape
ExpandDims*
T0*
out_type0
]
3pxtr_self_attention/dense_2/Tensordot/GatherV2/axisConst*
value	B : *
dtype0
�
.pxtr_self_attention/dense_2/Tensordot/GatherV2GatherV2+pxtr_self_attention/dense_2/Tensordot/Shape*pxtr_self_attention/dense_2/Tensordot/free3pxtr_self_attention/dense_2/Tensordot/GatherV2/axis*
Tparams0*
Taxis0*
Tindices0
_
5pxtr_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0
�
0pxtr_self_attention/dense_2/Tensordot/GatherV2_1GatherV2+pxtr_self_attention/dense_2/Tensordot/Shape*pxtr_self_attention/dense_2/Tensordot/axes5pxtr_self_attention/dense_2/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0
Y
+pxtr_self_attention/dense_2/Tensordot/ConstConst*
valueB: *
dtype0
�
*pxtr_self_attention/dense_2/Tensordot/ProdProd.pxtr_self_attention/dense_2/Tensordot/GatherV2+pxtr_self_attention/dense_2/Tensordot/Const*
T0*

Tidx0*
	keep_dims( 
[
-pxtr_self_attention/dense_2/Tensordot/Const_1Const*
valueB: *
dtype0
�
,pxtr_self_attention/dense_2/Tensordot/Prod_1Prod0pxtr_self_attention/dense_2/Tensordot/GatherV2_1-pxtr_self_attention/dense_2/Tensordot/Const_1*
T0*

Tidx0*
	keep_dims( 
[
1pxtr_self_attention/dense_2/Tensordot/concat/axisConst*
dtype0*
value	B : 
�
,pxtr_self_attention/dense_2/Tensordot/concatConcatV2*pxtr_self_attention/dense_2/Tensordot/free*pxtr_self_attention/dense_2/Tensordot/axes1pxtr_self_attention/dense_2/Tensordot/concat/axis*

Tidx0*
T0*
N
�
+pxtr_self_attention/dense_2/Tensordot/stackPack*pxtr_self_attention/dense_2/Tensordot/Prod,pxtr_self_attention/dense_2/Tensordot/Prod_1*
T0*

axis *
N
�
/pxtr_self_attention/dense_2/Tensordot/transpose	Transpose
ExpandDims,pxtr_self_attention/dense_2/Tensordot/concat*
Tperm0*
T0
�
-pxtr_self_attention/dense_2/Tensordot/ReshapeReshape/pxtr_self_attention/dense_2/Tensordot/transpose+pxtr_self_attention/dense_2/Tensordot/stack*
T0*
Tshape0
k
6pxtr_self_attention/dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
1pxtr_self_attention/dense_2/Tensordot/transpose_1	Transpose8mio_variable/pxtr_self_attention/dense_2/kernel/variable6pxtr_self_attention/dense_2/Tensordot/transpose_1/perm*
Tperm0*
T0
j
5pxtr_self_attention/dense_2/Tensordot/Reshape_1/shapeConst*
valueB"      *
dtype0
�
/pxtr_self_attention/dense_2/Tensordot/Reshape_1Reshape1pxtr_self_attention/dense_2/Tensordot/transpose_15pxtr_self_attention/dense_2/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
,pxtr_self_attention/dense_2/Tensordot/MatMulMatMul-pxtr_self_attention/dense_2/Tensordot/Reshape/pxtr_self_attention/dense_2/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
[
-pxtr_self_attention/dense_2/Tensordot/Const_2Const*
valueB:*
dtype0
]
3pxtr_self_attention/dense_2/Tensordot/concat_1/axisConst*
value	B : *
dtype0
�
.pxtr_self_attention/dense_2/Tensordot/concat_1ConcatV2.pxtr_self_attention/dense_2/Tensordot/GatherV2-pxtr_self_attention/dense_2/Tensordot/Const_23pxtr_self_attention/dense_2/Tensordot/concat_1/axis*
N*

Tidx0*
T0
�
%pxtr_self_attention/dense_2/TensordotReshape,pxtr_self_attention/dense_2/Tensordot/MatMul.pxtr_self_attention/dense_2/Tensordot/concat_1*
T0*
Tshape0
C
pxtr_self_attention/ConstConst*
dtype0*
value	B :
V
#pxtr_self_attention/split/split_dimConst*
valueB :
���������*
dtype0
�
pxtr_self_attention/splitSplit#pxtr_self_attention/split/split_dim#pxtr_self_attention/dense/Tensordot*
T0*
	num_split
I
pxtr_self_attention/concat/axisConst*
value	B : *
dtype0
�
pxtr_self_attention/concatConcatV2pxtr_self_attention/splitpxtr_self_attention/split:1pxtr_self_attention/concat/axis*
N*

Tidx0*
T0
E
pxtr_self_attention/Const_1Const*
value	B :*
dtype0
X
%pxtr_self_attention/split_1/split_dimConst*
valueB :
���������*
dtype0
�
pxtr_self_attention/split_1Split%pxtr_self_attention/split_1/split_dim%pxtr_self_attention/dense_1/Tensordot*
T0*
	num_split
K
!pxtr_self_attention/concat_1/axisConst*
value	B : *
dtype0
�
pxtr_self_attention/concat_1ConcatV2pxtr_self_attention/split_1pxtr_self_attention/split_1:1!pxtr_self_attention/concat_1/axis*
T0*
N*

Tidx0
E
pxtr_self_attention/Const_2Const*
value	B :*
dtype0
X
%pxtr_self_attention/split_2/split_dimConst*
valueB :
���������*
dtype0
�
pxtr_self_attention/split_2Split%pxtr_self_attention/split_2/split_dim%pxtr_self_attention/dense_2/Tensordot*
T0*
	num_split
K
!pxtr_self_attention/concat_2/axisConst*
value	B : *
dtype0
�
pxtr_self_attention/concat_2ConcatV2pxtr_self_attention/split_2pxtr_self_attention/split_2:1!pxtr_self_attention/concat_2/axis*
T0*
N*

Tidx0
[
"pxtr_self_attention/transpose/permConst*!
valueB"          *
dtype0
�
pxtr_self_attention/transpose	Transposepxtr_self_attention/concat_1"pxtr_self_attention/transpose/perm*
Tperm0*
T0
�
pxtr_self_attention/MatMulBatchMatMulpxtr_self_attention/concatpxtr_self_attention/transpose*
adj_x( *
adj_y( *
T0
J
pxtr_self_attention/truediv/yConst*
dtype0*
valueB
 *�5@
j
pxtr_self_attention/truedivRealDivpxtr_self_attention/MatMulpxtr_self_attention/truediv/y*
T0
L
pxtr_self_attention/SoftmaxSoftmaxpxtr_self_attention/truediv*
T0
�
pxtr_self_attention/MatMul_1BatchMatMulpxtr_self_attention/Softmaxpxtr_self_attention/concat_2*
adj_x( *
adj_y( *
T0
E
pxtr_self_attention/Const_3Const*
value	B :*
dtype0
O
%pxtr_self_attention/split_3/split_dimConst*
value	B : *
dtype0
�
pxtr_self_attention/split_3Split%pxtr_self_attention/split_3/split_dimpxtr_self_attention/MatMul_1*
T0*
	num_split
K
!pxtr_self_attention/concat_3/axisConst*
value	B :*
dtype0
�
pxtr_self_attention/concat_3ConcatV2pxtr_self_attention/split_3pxtr_self_attention/split_3:1!pxtr_self_attention/concat_3/axis*
N*

Tidx0*
T0
�
8mio_variable/pxtr_self_attention/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"pxtr_self_attention/dense_3/kernel*
shape
:
�
8mio_variable/pxtr_self_attention/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"pxtr_self_attention/dense_3/kernel*
shape
:
W
"Initializer_9/random_uniform/shapeConst*
valueB"      *
dtype0
M
 Initializer_9/random_uniform/minConst*
valueB
 *׳ݾ*
dtype0
M
 Initializer_9/random_uniform/maxConst*
valueB
 *׳�>*
dtype0
�
*Initializer_9/random_uniform/RandomUniformRandomUniform"Initializer_9/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
t
 Initializer_9/random_uniform/subSub Initializer_9/random_uniform/max Initializer_9/random_uniform/min*
T0
~
 Initializer_9/random_uniform/mulMul*Initializer_9/random_uniform/RandomUniform Initializer_9/random_uniform/sub*
T0
p
Initializer_9/random_uniformAdd Initializer_9/random_uniform/mul Initializer_9/random_uniform/min*
T0
�
Assign_9Assign8mio_variable/pxtr_self_attention/dense_3/kernel/gradientInitializer_9/random_uniform*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/pxtr_self_attention/dense_3/kernel/gradient*
validate_shape(
�
6mio_variable/pxtr_self_attention/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" pxtr_self_attention/dense_3/bias*
shape:
�
6mio_variable/pxtr_self_attention/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" pxtr_self_attention/dense_3/bias*
shape:
E
Initializer_10/zerosConst*
valueB*    *
dtype0
�
	Assign_10Assign6mio_variable/pxtr_self_attention/dense_3/bias/gradientInitializer_10/zeros*
T0*I
_class?
=;loc:@mio_variable/pxtr_self_attention/dense_3/bias/gradient*
validate_shape(*
use_locking(
X
*pxtr_self_attention/dense_3/Tensordot/axesConst*
valueB:*
dtype0
_
*pxtr_self_attention/dense_3/Tensordot/freeConst*
valueB"       *
dtype0
k
+pxtr_self_attention/dense_3/Tensordot/ShapeShapepxtr_self_attention/concat_3*
T0*
out_type0
]
3pxtr_self_attention/dense_3/Tensordot/GatherV2/axisConst*
value	B : *
dtype0
�
.pxtr_self_attention/dense_3/Tensordot/GatherV2GatherV2+pxtr_self_attention/dense_3/Tensordot/Shape*pxtr_self_attention/dense_3/Tensordot/free3pxtr_self_attention/dense_3/Tensordot/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0
_
5pxtr_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0
�
0pxtr_self_attention/dense_3/Tensordot/GatherV2_1GatherV2+pxtr_self_attention/dense_3/Tensordot/Shape*pxtr_self_attention/dense_3/Tensordot/axes5pxtr_self_attention/dense_3/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0
Y
+pxtr_self_attention/dense_3/Tensordot/ConstConst*
dtype0*
valueB: 
�
*pxtr_self_attention/dense_3/Tensordot/ProdProd.pxtr_self_attention/dense_3/Tensordot/GatherV2+pxtr_self_attention/dense_3/Tensordot/Const*

Tidx0*
	keep_dims( *
T0
[
-pxtr_self_attention/dense_3/Tensordot/Const_1Const*
dtype0*
valueB: 
�
,pxtr_self_attention/dense_3/Tensordot/Prod_1Prod0pxtr_self_attention/dense_3/Tensordot/GatherV2_1-pxtr_self_attention/dense_3/Tensordot/Const_1*
T0*

Tidx0*
	keep_dims( 
[
1pxtr_self_attention/dense_3/Tensordot/concat/axisConst*
value	B : *
dtype0
�
,pxtr_self_attention/dense_3/Tensordot/concatConcatV2*pxtr_self_attention/dense_3/Tensordot/free*pxtr_self_attention/dense_3/Tensordot/axes1pxtr_self_attention/dense_3/Tensordot/concat/axis*
T0*
N*

Tidx0
�
+pxtr_self_attention/dense_3/Tensordot/stackPack*pxtr_self_attention/dense_3/Tensordot/Prod,pxtr_self_attention/dense_3/Tensordot/Prod_1*
N*
T0*

axis 
�
/pxtr_self_attention/dense_3/Tensordot/transpose	Transposepxtr_self_attention/concat_3,pxtr_self_attention/dense_3/Tensordot/concat*
T0*
Tperm0
�
-pxtr_self_attention/dense_3/Tensordot/ReshapeReshape/pxtr_self_attention/dense_3/Tensordot/transpose+pxtr_self_attention/dense_3/Tensordot/stack*
T0*
Tshape0
k
6pxtr_self_attention/dense_3/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
1pxtr_self_attention/dense_3/Tensordot/transpose_1	Transpose8mio_variable/pxtr_self_attention/dense_3/kernel/variable6pxtr_self_attention/dense_3/Tensordot/transpose_1/perm*
Tperm0*
T0
j
5pxtr_self_attention/dense_3/Tensordot/Reshape_1/shapeConst*
valueB"      *
dtype0
�
/pxtr_self_attention/dense_3/Tensordot/Reshape_1Reshape1pxtr_self_attention/dense_3/Tensordot/transpose_15pxtr_self_attention/dense_3/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
,pxtr_self_attention/dense_3/Tensordot/MatMulMatMul-pxtr_self_attention/dense_3/Tensordot/Reshape/pxtr_self_attention/dense_3/Tensordot/Reshape_1*
transpose_a( *
transpose_b( *
T0
[
-pxtr_self_attention/dense_3/Tensordot/Const_2Const*
valueB:*
dtype0
]
3pxtr_self_attention/dense_3/Tensordot/concat_1/axisConst*
dtype0*
value	B : 
�
.pxtr_self_attention/dense_3/Tensordot/concat_1ConcatV2.pxtr_self_attention/dense_3/Tensordot/GatherV2-pxtr_self_attention/dense_3/Tensordot/Const_23pxtr_self_attention/dense_3/Tensordot/concat_1/axis*
T0*
N*

Tidx0
�
%pxtr_self_attention/dense_3/TensordotReshape,pxtr_self_attention/dense_3/Tensordot/MatMul.pxtr_self_attention/dense_3/Tensordot/concat_1*
T0*
Tshape0
�
#pxtr_self_attention/dense_3/BiasAddBiasAdd%pxtr_self_attention/dense_3/Tensordot6mio_variable/pxtr_self_attention/dense_3/bias/variable*
T0*
data_formatNHWC
�
Dmio_variable/intent_aware_cross_pxtr_attention/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.intent_aware_cross_pxtr_attention/dense/kernel*
shape
:
�
Dmio_variable/intent_aware_cross_pxtr_attention/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.intent_aware_cross_pxtr_attention/dense/kernel*
shape
:
X
#Initializer_11/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_11/random_uniform/minConst*
valueB
 *׳ݾ*
dtype0
N
!Initializer_11/random_uniform/maxConst*
valueB
 *׳�>*
dtype0
�
+Initializer_11/random_uniform/RandomUniformRandomUniform#Initializer_11/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_11/random_uniform/subSub!Initializer_11/random_uniform/max!Initializer_11/random_uniform/min*
T0
�
!Initializer_11/random_uniform/mulMul+Initializer_11/random_uniform/RandomUniform!Initializer_11/random_uniform/sub*
T0
s
Initializer_11/random_uniformAdd!Initializer_11/random_uniform/mul!Initializer_11/random_uniform/min*
T0
�
	Assign_11AssignDmio_variable/intent_aware_cross_pxtr_attention/dense/kernel/gradientInitializer_11/random_uniform*
T0*W
_classM
KIloc:@mio_variable/intent_aware_cross_pxtr_attention/dense/kernel/gradient*
validate_shape(*
use_locking(
d
6intent_aware_cross_pxtr_attention/dense/Tensordot/axesConst*
dtype0*
valueB:
k
6intent_aware_cross_pxtr_attention/dense/Tensordot/freeConst*
valueB"       *
dtype0
g
7intent_aware_cross_pxtr_attention/dense/Tensordot/ShapeShapeExpandDims_1*
T0*
out_type0
i
?intent_aware_cross_pxtr_attention/dense/Tensordot/GatherV2/axisConst*
dtype0*
value	B : 
�
:intent_aware_cross_pxtr_attention/dense/Tensordot/GatherV2GatherV27intent_aware_cross_pxtr_attention/dense/Tensordot/Shape6intent_aware_cross_pxtr_attention/dense/Tensordot/free?intent_aware_cross_pxtr_attention/dense/Tensordot/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0
k
Aintent_aware_cross_pxtr_attention/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0
�
<intent_aware_cross_pxtr_attention/dense/Tensordot/GatherV2_1GatherV27intent_aware_cross_pxtr_attention/dense/Tensordot/Shape6intent_aware_cross_pxtr_attention/dense/Tensordot/axesAintent_aware_cross_pxtr_attention/dense/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0
e
7intent_aware_cross_pxtr_attention/dense/Tensordot/ConstConst*
valueB: *
dtype0
�
6intent_aware_cross_pxtr_attention/dense/Tensordot/ProdProd:intent_aware_cross_pxtr_attention/dense/Tensordot/GatherV27intent_aware_cross_pxtr_attention/dense/Tensordot/Const*

Tidx0*
	keep_dims( *
T0
g
9intent_aware_cross_pxtr_attention/dense/Tensordot/Const_1Const*
valueB: *
dtype0
�
8intent_aware_cross_pxtr_attention/dense/Tensordot/Prod_1Prod<intent_aware_cross_pxtr_attention/dense/Tensordot/GatherV2_19intent_aware_cross_pxtr_attention/dense/Tensordot/Const_1*
T0*

Tidx0*
	keep_dims( 
g
=intent_aware_cross_pxtr_attention/dense/Tensordot/concat/axisConst*
value	B : *
dtype0
�
8intent_aware_cross_pxtr_attention/dense/Tensordot/concatConcatV26intent_aware_cross_pxtr_attention/dense/Tensordot/free6intent_aware_cross_pxtr_attention/dense/Tensordot/axes=intent_aware_cross_pxtr_attention/dense/Tensordot/concat/axis*
T0*
N*

Tidx0
�
7intent_aware_cross_pxtr_attention/dense/Tensordot/stackPack6intent_aware_cross_pxtr_attention/dense/Tensordot/Prod8intent_aware_cross_pxtr_attention/dense/Tensordot/Prod_1*
N*
T0*

axis 
�
;intent_aware_cross_pxtr_attention/dense/Tensordot/transpose	TransposeExpandDims_18intent_aware_cross_pxtr_attention/dense/Tensordot/concat*
Tperm0*
T0
�
9intent_aware_cross_pxtr_attention/dense/Tensordot/ReshapeReshape;intent_aware_cross_pxtr_attention/dense/Tensordot/transpose7intent_aware_cross_pxtr_attention/dense/Tensordot/stack*
T0*
Tshape0
w
Bintent_aware_cross_pxtr_attention/dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
=intent_aware_cross_pxtr_attention/dense/Tensordot/transpose_1	TransposeDmio_variable/intent_aware_cross_pxtr_attention/dense/kernel/variableBintent_aware_cross_pxtr_attention/dense/Tensordot/transpose_1/perm*
Tperm0*
T0
v
Aintent_aware_cross_pxtr_attention/dense/Tensordot/Reshape_1/shapeConst*
valueB"      *
dtype0
�
;intent_aware_cross_pxtr_attention/dense/Tensordot/Reshape_1Reshape=intent_aware_cross_pxtr_attention/dense/Tensordot/transpose_1Aintent_aware_cross_pxtr_attention/dense/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
8intent_aware_cross_pxtr_attention/dense/Tensordot/MatMulMatMul9intent_aware_cross_pxtr_attention/dense/Tensordot/Reshape;intent_aware_cross_pxtr_attention/dense/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
g
9intent_aware_cross_pxtr_attention/dense/Tensordot/Const_2Const*
valueB:*
dtype0
i
?intent_aware_cross_pxtr_attention/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0
�
:intent_aware_cross_pxtr_attention/dense/Tensordot/concat_1ConcatV2:intent_aware_cross_pxtr_attention/dense/Tensordot/GatherV29intent_aware_cross_pxtr_attention/dense/Tensordot/Const_2?intent_aware_cross_pxtr_attention/dense/Tensordot/concat_1/axis*
N*

Tidx0*
T0
�
1intent_aware_cross_pxtr_attention/dense/TensordotReshape8intent_aware_cross_pxtr_attention/dense/Tensordot/MatMul:intent_aware_cross_pxtr_attention/dense/Tensordot/concat_1*
T0*
Tshape0
�
Fmio_variable/intent_aware_cross_pxtr_attention/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20intent_aware_cross_pxtr_attention/dense_1/kernel*
shape
:
�
Fmio_variable/intent_aware_cross_pxtr_attention/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20intent_aware_cross_pxtr_attention/dense_1/kernel*
shape
:
X
#Initializer_12/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_12/random_uniform/minConst*
valueB
 *׳ݾ*
dtype0
N
!Initializer_12/random_uniform/maxConst*
valueB
 *׳�>*
dtype0
�
+Initializer_12/random_uniform/RandomUniformRandomUniform#Initializer_12/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_12/random_uniform/subSub!Initializer_12/random_uniform/max!Initializer_12/random_uniform/min*
T0
�
!Initializer_12/random_uniform/mulMul+Initializer_12/random_uniform/RandomUniform!Initializer_12/random_uniform/sub*
T0
s
Initializer_12/random_uniformAdd!Initializer_12/random_uniform/mul!Initializer_12/random_uniform/min*
T0
�
	Assign_12AssignFmio_variable/intent_aware_cross_pxtr_attention/dense_1/kernel/gradientInitializer_12/random_uniform*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/intent_aware_cross_pxtr_attention/dense_1/kernel/gradient*
validate_shape(
f
8intent_aware_cross_pxtr_attention/dense_1/Tensordot/axesConst*
valueB:*
dtype0
m
8intent_aware_cross_pxtr_attention/dense_1/Tensordot/freeConst*
dtype0*
valueB"       
�
9intent_aware_cross_pxtr_attention/dense_1/Tensordot/ShapeShape#pxtr_self_attention/dense_3/BiasAdd*
T0*
out_type0
k
Aintent_aware_cross_pxtr_attention/dense_1/Tensordot/GatherV2/axisConst*
value	B : *
dtype0
�
<intent_aware_cross_pxtr_attention/dense_1/Tensordot/GatherV2GatherV29intent_aware_cross_pxtr_attention/dense_1/Tensordot/Shape8intent_aware_cross_pxtr_attention/dense_1/Tensordot/freeAintent_aware_cross_pxtr_attention/dense_1/Tensordot/GatherV2/axis*
Tindices0*
Tparams0*
Taxis0
m
Cintent_aware_cross_pxtr_attention/dense_1/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0
�
>intent_aware_cross_pxtr_attention/dense_1/Tensordot/GatherV2_1GatherV29intent_aware_cross_pxtr_attention/dense_1/Tensordot/Shape8intent_aware_cross_pxtr_attention/dense_1/Tensordot/axesCintent_aware_cross_pxtr_attention/dense_1/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0
g
9intent_aware_cross_pxtr_attention/dense_1/Tensordot/ConstConst*
valueB: *
dtype0
�
8intent_aware_cross_pxtr_attention/dense_1/Tensordot/ProdProd<intent_aware_cross_pxtr_attention/dense_1/Tensordot/GatherV29intent_aware_cross_pxtr_attention/dense_1/Tensordot/Const*

Tidx0*
	keep_dims( *
T0
i
;intent_aware_cross_pxtr_attention/dense_1/Tensordot/Const_1Const*
valueB: *
dtype0
�
:intent_aware_cross_pxtr_attention/dense_1/Tensordot/Prod_1Prod>intent_aware_cross_pxtr_attention/dense_1/Tensordot/GatherV2_1;intent_aware_cross_pxtr_attention/dense_1/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0
i
?intent_aware_cross_pxtr_attention/dense_1/Tensordot/concat/axisConst*
value	B : *
dtype0
�
:intent_aware_cross_pxtr_attention/dense_1/Tensordot/concatConcatV28intent_aware_cross_pxtr_attention/dense_1/Tensordot/free8intent_aware_cross_pxtr_attention/dense_1/Tensordot/axes?intent_aware_cross_pxtr_attention/dense_1/Tensordot/concat/axis*

Tidx0*
T0*
N
�
9intent_aware_cross_pxtr_attention/dense_1/Tensordot/stackPack8intent_aware_cross_pxtr_attention/dense_1/Tensordot/Prod:intent_aware_cross_pxtr_attention/dense_1/Tensordot/Prod_1*
T0*

axis *
N
�
=intent_aware_cross_pxtr_attention/dense_1/Tensordot/transpose	Transpose#pxtr_self_attention/dense_3/BiasAdd:intent_aware_cross_pxtr_attention/dense_1/Tensordot/concat*
Tperm0*
T0
�
;intent_aware_cross_pxtr_attention/dense_1/Tensordot/ReshapeReshape=intent_aware_cross_pxtr_attention/dense_1/Tensordot/transpose9intent_aware_cross_pxtr_attention/dense_1/Tensordot/stack*
T0*
Tshape0
y
Dintent_aware_cross_pxtr_attention/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
?intent_aware_cross_pxtr_attention/dense_1/Tensordot/transpose_1	TransposeFmio_variable/intent_aware_cross_pxtr_attention/dense_1/kernel/variableDintent_aware_cross_pxtr_attention/dense_1/Tensordot/transpose_1/perm*
Tperm0*
T0
x
Cintent_aware_cross_pxtr_attention/dense_1/Tensordot/Reshape_1/shapeConst*
dtype0*
valueB"      
�
=intent_aware_cross_pxtr_attention/dense_1/Tensordot/Reshape_1Reshape?intent_aware_cross_pxtr_attention/dense_1/Tensordot/transpose_1Cintent_aware_cross_pxtr_attention/dense_1/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
:intent_aware_cross_pxtr_attention/dense_1/Tensordot/MatMulMatMul;intent_aware_cross_pxtr_attention/dense_1/Tensordot/Reshape=intent_aware_cross_pxtr_attention/dense_1/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
i
;intent_aware_cross_pxtr_attention/dense_1/Tensordot/Const_2Const*
valueB:*
dtype0
k
Aintent_aware_cross_pxtr_attention/dense_1/Tensordot/concat_1/axisConst*
value	B : *
dtype0
�
<intent_aware_cross_pxtr_attention/dense_1/Tensordot/concat_1ConcatV2<intent_aware_cross_pxtr_attention/dense_1/Tensordot/GatherV2;intent_aware_cross_pxtr_attention/dense_1/Tensordot/Const_2Aintent_aware_cross_pxtr_attention/dense_1/Tensordot/concat_1/axis*
T0*
N*

Tidx0
�
3intent_aware_cross_pxtr_attention/dense_1/TensordotReshape:intent_aware_cross_pxtr_attention/dense_1/Tensordot/MatMul<intent_aware_cross_pxtr_attention/dense_1/Tensordot/concat_1*
T0*
Tshape0
�
Fmio_variable/intent_aware_cross_pxtr_attention/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20intent_aware_cross_pxtr_attention/dense_2/kernel*
shape
:
�
Fmio_variable/intent_aware_cross_pxtr_attention/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20intent_aware_cross_pxtr_attention/dense_2/kernel*
shape
:
X
#Initializer_13/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_13/random_uniform/minConst*
dtype0*
valueB
 *׳ݾ
N
!Initializer_13/random_uniform/maxConst*
valueB
 *׳�>*
dtype0
�
+Initializer_13/random_uniform/RandomUniformRandomUniform#Initializer_13/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_13/random_uniform/subSub!Initializer_13/random_uniform/max!Initializer_13/random_uniform/min*
T0
�
!Initializer_13/random_uniform/mulMul+Initializer_13/random_uniform/RandomUniform!Initializer_13/random_uniform/sub*
T0
s
Initializer_13/random_uniformAdd!Initializer_13/random_uniform/mul!Initializer_13/random_uniform/min*
T0
�
	Assign_13AssignFmio_variable/intent_aware_cross_pxtr_attention/dense_2/kernel/gradientInitializer_13/random_uniform*
validate_shape(*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/intent_aware_cross_pxtr_attention/dense_2/kernel/gradient
f
8intent_aware_cross_pxtr_attention/dense_2/Tensordot/axesConst*
valueB:*
dtype0
m
8intent_aware_cross_pxtr_attention/dense_2/Tensordot/freeConst*
valueB"       *
dtype0
�
9intent_aware_cross_pxtr_attention/dense_2/Tensordot/ShapeShape#pxtr_self_attention/dense_3/BiasAdd*
T0*
out_type0
k
Aintent_aware_cross_pxtr_attention/dense_2/Tensordot/GatherV2/axisConst*
dtype0*
value	B : 
�
<intent_aware_cross_pxtr_attention/dense_2/Tensordot/GatherV2GatherV29intent_aware_cross_pxtr_attention/dense_2/Tensordot/Shape8intent_aware_cross_pxtr_attention/dense_2/Tensordot/freeAintent_aware_cross_pxtr_attention/dense_2/Tensordot/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0
m
Cintent_aware_cross_pxtr_attention/dense_2/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0
�
>intent_aware_cross_pxtr_attention/dense_2/Tensordot/GatherV2_1GatherV29intent_aware_cross_pxtr_attention/dense_2/Tensordot/Shape8intent_aware_cross_pxtr_attention/dense_2/Tensordot/axesCintent_aware_cross_pxtr_attention/dense_2/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0
g
9intent_aware_cross_pxtr_attention/dense_2/Tensordot/ConstConst*
valueB: *
dtype0
�
8intent_aware_cross_pxtr_attention/dense_2/Tensordot/ProdProd<intent_aware_cross_pxtr_attention/dense_2/Tensordot/GatherV29intent_aware_cross_pxtr_attention/dense_2/Tensordot/Const*
T0*

Tidx0*
	keep_dims( 
i
;intent_aware_cross_pxtr_attention/dense_2/Tensordot/Const_1Const*
dtype0*
valueB: 
�
:intent_aware_cross_pxtr_attention/dense_2/Tensordot/Prod_1Prod>intent_aware_cross_pxtr_attention/dense_2/Tensordot/GatherV2_1;intent_aware_cross_pxtr_attention/dense_2/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0
i
?intent_aware_cross_pxtr_attention/dense_2/Tensordot/concat/axisConst*
value	B : *
dtype0
�
:intent_aware_cross_pxtr_attention/dense_2/Tensordot/concatConcatV28intent_aware_cross_pxtr_attention/dense_2/Tensordot/free8intent_aware_cross_pxtr_attention/dense_2/Tensordot/axes?intent_aware_cross_pxtr_attention/dense_2/Tensordot/concat/axis*
T0*
N*

Tidx0
�
9intent_aware_cross_pxtr_attention/dense_2/Tensordot/stackPack8intent_aware_cross_pxtr_attention/dense_2/Tensordot/Prod:intent_aware_cross_pxtr_attention/dense_2/Tensordot/Prod_1*
T0*

axis *
N
�
=intent_aware_cross_pxtr_attention/dense_2/Tensordot/transpose	Transpose#pxtr_self_attention/dense_3/BiasAdd:intent_aware_cross_pxtr_attention/dense_2/Tensordot/concat*
Tperm0*
T0
�
;intent_aware_cross_pxtr_attention/dense_2/Tensordot/ReshapeReshape=intent_aware_cross_pxtr_attention/dense_2/Tensordot/transpose9intent_aware_cross_pxtr_attention/dense_2/Tensordot/stack*
T0*
Tshape0
y
Dintent_aware_cross_pxtr_attention/dense_2/Tensordot/transpose_1/permConst*
dtype0*
valueB"       
�
?intent_aware_cross_pxtr_attention/dense_2/Tensordot/transpose_1	TransposeFmio_variable/intent_aware_cross_pxtr_attention/dense_2/kernel/variableDintent_aware_cross_pxtr_attention/dense_2/Tensordot/transpose_1/perm*
T0*
Tperm0
x
Cintent_aware_cross_pxtr_attention/dense_2/Tensordot/Reshape_1/shapeConst*
valueB"      *
dtype0
�
=intent_aware_cross_pxtr_attention/dense_2/Tensordot/Reshape_1Reshape?intent_aware_cross_pxtr_attention/dense_2/Tensordot/transpose_1Cintent_aware_cross_pxtr_attention/dense_2/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
:intent_aware_cross_pxtr_attention/dense_2/Tensordot/MatMulMatMul;intent_aware_cross_pxtr_attention/dense_2/Tensordot/Reshape=intent_aware_cross_pxtr_attention/dense_2/Tensordot/Reshape_1*
transpose_a( *
transpose_b( *
T0
i
;intent_aware_cross_pxtr_attention/dense_2/Tensordot/Const_2Const*
valueB:*
dtype0
k
Aintent_aware_cross_pxtr_attention/dense_2/Tensordot/concat_1/axisConst*
dtype0*
value	B : 
�
<intent_aware_cross_pxtr_attention/dense_2/Tensordot/concat_1ConcatV2<intent_aware_cross_pxtr_attention/dense_2/Tensordot/GatherV2;intent_aware_cross_pxtr_attention/dense_2/Tensordot/Const_2Aintent_aware_cross_pxtr_attention/dense_2/Tensordot/concat_1/axis*
N*

Tidx0*
T0
�
3intent_aware_cross_pxtr_attention/dense_2/TensordotReshape:intent_aware_cross_pxtr_attention/dense_2/Tensordot/MatMul<intent_aware_cross_pxtr_attention/dense_2/Tensordot/concat_1*
T0*
Tshape0
Q
'intent_aware_cross_pxtr_attention/ConstConst*
dtype0*
value	B :
d
1intent_aware_cross_pxtr_attention/split/split_dimConst*
dtype0*
valueB :
���������
�
'intent_aware_cross_pxtr_attention/splitSplit1intent_aware_cross_pxtr_attention/split/split_dim1intent_aware_cross_pxtr_attention/dense/Tensordot*
T0*
	num_split
W
-intent_aware_cross_pxtr_attention/concat/axisConst*
value	B : *
dtype0
�
(intent_aware_cross_pxtr_attention/concatConcatV2'intent_aware_cross_pxtr_attention/split)intent_aware_cross_pxtr_attention/split:1-intent_aware_cross_pxtr_attention/concat/axis*

Tidx0*
T0*
N
S
)intent_aware_cross_pxtr_attention/Const_1Const*
value	B :*
dtype0
f
3intent_aware_cross_pxtr_attention/split_1/split_dimConst*
valueB :
���������*
dtype0
�
)intent_aware_cross_pxtr_attention/split_1Split3intent_aware_cross_pxtr_attention/split_1/split_dim3intent_aware_cross_pxtr_attention/dense_1/Tensordot*
T0*
	num_split
Y
/intent_aware_cross_pxtr_attention/concat_1/axisConst*
value	B : *
dtype0
�
*intent_aware_cross_pxtr_attention/concat_1ConcatV2)intent_aware_cross_pxtr_attention/split_1+intent_aware_cross_pxtr_attention/split_1:1/intent_aware_cross_pxtr_attention/concat_1/axis*
T0*
N*

Tidx0
S
)intent_aware_cross_pxtr_attention/Const_2Const*
value	B :*
dtype0
f
3intent_aware_cross_pxtr_attention/split_2/split_dimConst*
valueB :
���������*
dtype0
�
)intent_aware_cross_pxtr_attention/split_2Split3intent_aware_cross_pxtr_attention/split_2/split_dim3intent_aware_cross_pxtr_attention/dense_2/Tensordot*
T0*
	num_split
Y
/intent_aware_cross_pxtr_attention/concat_2/axisConst*
value	B : *
dtype0
�
*intent_aware_cross_pxtr_attention/concat_2ConcatV2)intent_aware_cross_pxtr_attention/split_2+intent_aware_cross_pxtr_attention/split_2:1/intent_aware_cross_pxtr_attention/concat_2/axis*
T0*
N*

Tidx0
i
0intent_aware_cross_pxtr_attention/transpose/permConst*
dtype0*!
valueB"          
�
+intent_aware_cross_pxtr_attention/transpose	Transpose*intent_aware_cross_pxtr_attention/concat_10intent_aware_cross_pxtr_attention/transpose/perm*
Tperm0*
T0
�
(intent_aware_cross_pxtr_attention/MatMulBatchMatMul(intent_aware_cross_pxtr_attention/concat+intent_aware_cross_pxtr_attention/transpose*
adj_x( *
adj_y( *
T0
X
+intent_aware_cross_pxtr_attention/truediv/yConst*
valueB
 *�5@*
dtype0
�
)intent_aware_cross_pxtr_attention/truedivRealDiv(intent_aware_cross_pxtr_attention/MatMul+intent_aware_cross_pxtr_attention/truediv/y*
T0
h
)intent_aware_cross_pxtr_attention/SoftmaxSoftmax)intent_aware_cross_pxtr_attention/truediv*
T0
�
*intent_aware_cross_pxtr_attention/MatMul_1BatchMatMul)intent_aware_cross_pxtr_attention/Softmax*intent_aware_cross_pxtr_attention/concat_2*
adj_x( *
adj_y( *
T0
S
)intent_aware_cross_pxtr_attention/Const_3Const*
dtype0*
value	B :
]
3intent_aware_cross_pxtr_attention/split_3/split_dimConst*
value	B : *
dtype0
�
)intent_aware_cross_pxtr_attention/split_3Split3intent_aware_cross_pxtr_attention/split_3/split_dim*intent_aware_cross_pxtr_attention/MatMul_1*
T0*
	num_split
Y
/intent_aware_cross_pxtr_attention/concat_3/axisConst*
value	B :*
dtype0
�
*intent_aware_cross_pxtr_attention/concat_3ConcatV2)intent_aware_cross_pxtr_attention/split_3+intent_aware_cross_pxtr_attention/split_3:1/intent_aware_cross_pxtr_attention/concat_3/axis*
N*

Tidx0*
T0
�
Fmio_variable/intent_aware_cross_pxtr_attention/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:*?
	container20intent_aware_cross_pxtr_attention/dense_3/kernel
�
Fmio_variable/intent_aware_cross_pxtr_attention/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20intent_aware_cross_pxtr_attention/dense_3/kernel*
shape
:
X
#Initializer_14/random_uniform/shapeConst*
dtype0*
valueB"      
N
!Initializer_14/random_uniform/minConst*
dtype0*
valueB
 *׳ݾ
N
!Initializer_14/random_uniform/maxConst*
valueB
 *׳�>*
dtype0
�
+Initializer_14/random_uniform/RandomUniformRandomUniform#Initializer_14/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_14/random_uniform/subSub!Initializer_14/random_uniform/max!Initializer_14/random_uniform/min*
T0
�
!Initializer_14/random_uniform/mulMul+Initializer_14/random_uniform/RandomUniform!Initializer_14/random_uniform/sub*
T0
s
Initializer_14/random_uniformAdd!Initializer_14/random_uniform/mul!Initializer_14/random_uniform/min*
T0
�
	Assign_14AssignFmio_variable/intent_aware_cross_pxtr_attention/dense_3/kernel/gradientInitializer_14/random_uniform*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/intent_aware_cross_pxtr_attention/dense_3/kernel/gradient*
validate_shape(
�
Dmio_variable/intent_aware_cross_pxtr_attention/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.intent_aware_cross_pxtr_attention/dense_3/bias*
shape:
�
Dmio_variable/intent_aware_cross_pxtr_attention/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*=
	container0.intent_aware_cross_pxtr_attention/dense_3/bias
E
Initializer_15/zerosConst*
valueB*    *
dtype0
�
	Assign_15AssignDmio_variable/intent_aware_cross_pxtr_attention/dense_3/bias/gradientInitializer_15/zeros*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/intent_aware_cross_pxtr_attention/dense_3/bias/gradient*
validate_shape(
f
8intent_aware_cross_pxtr_attention/dense_3/Tensordot/axesConst*
dtype0*
valueB:
m
8intent_aware_cross_pxtr_attention/dense_3/Tensordot/freeConst*
valueB"       *
dtype0
�
9intent_aware_cross_pxtr_attention/dense_3/Tensordot/ShapeShape*intent_aware_cross_pxtr_attention/concat_3*
T0*
out_type0
k
Aintent_aware_cross_pxtr_attention/dense_3/Tensordot/GatherV2/axisConst*
value	B : *
dtype0
�
<intent_aware_cross_pxtr_attention/dense_3/Tensordot/GatherV2GatherV29intent_aware_cross_pxtr_attention/dense_3/Tensordot/Shape8intent_aware_cross_pxtr_attention/dense_3/Tensordot/freeAintent_aware_cross_pxtr_attention/dense_3/Tensordot/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0
m
Cintent_aware_cross_pxtr_attention/dense_3/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0
�
>intent_aware_cross_pxtr_attention/dense_3/Tensordot/GatherV2_1GatherV29intent_aware_cross_pxtr_attention/dense_3/Tensordot/Shape8intent_aware_cross_pxtr_attention/dense_3/Tensordot/axesCintent_aware_cross_pxtr_attention/dense_3/Tensordot/GatherV2_1/axis*
Tindices0*
Tparams0*
Taxis0
g
9intent_aware_cross_pxtr_attention/dense_3/Tensordot/ConstConst*
dtype0*
valueB: 
�
8intent_aware_cross_pxtr_attention/dense_3/Tensordot/ProdProd<intent_aware_cross_pxtr_attention/dense_3/Tensordot/GatherV29intent_aware_cross_pxtr_attention/dense_3/Tensordot/Const*

Tidx0*
	keep_dims( *
T0
i
;intent_aware_cross_pxtr_attention/dense_3/Tensordot/Const_1Const*
dtype0*
valueB: 
�
:intent_aware_cross_pxtr_attention/dense_3/Tensordot/Prod_1Prod>intent_aware_cross_pxtr_attention/dense_3/Tensordot/GatherV2_1;intent_aware_cross_pxtr_attention/dense_3/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0
i
?intent_aware_cross_pxtr_attention/dense_3/Tensordot/concat/axisConst*
value	B : *
dtype0
�
:intent_aware_cross_pxtr_attention/dense_3/Tensordot/concatConcatV28intent_aware_cross_pxtr_attention/dense_3/Tensordot/free8intent_aware_cross_pxtr_attention/dense_3/Tensordot/axes?intent_aware_cross_pxtr_attention/dense_3/Tensordot/concat/axis*
T0*
N*

Tidx0
�
9intent_aware_cross_pxtr_attention/dense_3/Tensordot/stackPack8intent_aware_cross_pxtr_attention/dense_3/Tensordot/Prod:intent_aware_cross_pxtr_attention/dense_3/Tensordot/Prod_1*
T0*

axis *
N
�
=intent_aware_cross_pxtr_attention/dense_3/Tensordot/transpose	Transpose*intent_aware_cross_pxtr_attention/concat_3:intent_aware_cross_pxtr_attention/dense_3/Tensordot/concat*
T0*
Tperm0
�
;intent_aware_cross_pxtr_attention/dense_3/Tensordot/ReshapeReshape=intent_aware_cross_pxtr_attention/dense_3/Tensordot/transpose9intent_aware_cross_pxtr_attention/dense_3/Tensordot/stack*
T0*
Tshape0
y
Dintent_aware_cross_pxtr_attention/dense_3/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
?intent_aware_cross_pxtr_attention/dense_3/Tensordot/transpose_1	TransposeFmio_variable/intent_aware_cross_pxtr_attention/dense_3/kernel/variableDintent_aware_cross_pxtr_attention/dense_3/Tensordot/transpose_1/perm*
Tperm0*
T0
x
Cintent_aware_cross_pxtr_attention/dense_3/Tensordot/Reshape_1/shapeConst*
valueB"      *
dtype0
�
=intent_aware_cross_pxtr_attention/dense_3/Tensordot/Reshape_1Reshape?intent_aware_cross_pxtr_attention/dense_3/Tensordot/transpose_1Cintent_aware_cross_pxtr_attention/dense_3/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
:intent_aware_cross_pxtr_attention/dense_3/Tensordot/MatMulMatMul;intent_aware_cross_pxtr_attention/dense_3/Tensordot/Reshape=intent_aware_cross_pxtr_attention/dense_3/Tensordot/Reshape_1*
transpose_a( *
transpose_b( *
T0
i
;intent_aware_cross_pxtr_attention/dense_3/Tensordot/Const_2Const*
valueB:*
dtype0
k
Aintent_aware_cross_pxtr_attention/dense_3/Tensordot/concat_1/axisConst*
dtype0*
value	B : 
�
<intent_aware_cross_pxtr_attention/dense_3/Tensordot/concat_1ConcatV2<intent_aware_cross_pxtr_attention/dense_3/Tensordot/GatherV2;intent_aware_cross_pxtr_attention/dense_3/Tensordot/Const_2Aintent_aware_cross_pxtr_attention/dense_3/Tensordot/concat_1/axis*
T0*
N*

Tidx0
�
3intent_aware_cross_pxtr_attention/dense_3/TensordotReshape:intent_aware_cross_pxtr_attention/dense_3/Tensordot/MatMul<intent_aware_cross_pxtr_attention/dense_3/Tensordot/concat_1*
T0*
Tshape0
�
1intent_aware_cross_pxtr_attention/dense_3/BiasAddBiasAdd3intent_aware_cross_pxtr_attention/dense_3/TensordotDmio_variable/intent_aware_cross_pxtr_attention/dense_3/bias/variable*
T0*
data_formatNHWC
�
Hmio_variable/intent_aware_cross_category_attention/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42intent_aware_cross_category_attention/dense/kernel*
shape
:
�
Hmio_variable/intent_aware_cross_category_attention/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42intent_aware_cross_category_attention/dense/kernel*
shape
:
X
#Initializer_16/random_uniform/shapeConst*
dtype0*
valueB"      
N
!Initializer_16/random_uniform/minConst*
valueB
 *׳ݾ*
dtype0
N
!Initializer_16/random_uniform/maxConst*
valueB
 *׳�>*
dtype0
�
+Initializer_16/random_uniform/RandomUniformRandomUniform#Initializer_16/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_16/random_uniform/subSub!Initializer_16/random_uniform/max!Initializer_16/random_uniform/min*
T0
�
!Initializer_16/random_uniform/mulMul+Initializer_16/random_uniform/RandomUniform!Initializer_16/random_uniform/sub*
T0
s
Initializer_16/random_uniformAdd!Initializer_16/random_uniform/mul!Initializer_16/random_uniform/min*
T0
�
	Assign_16AssignHmio_variable/intent_aware_cross_category_attention/dense/kernel/gradientInitializer_16/random_uniform*
use_locking(*
T0*[
_classQ
OMloc:@mio_variable/intent_aware_cross_category_attention/dense/kernel/gradient*
validate_shape(
h
:intent_aware_cross_category_attention/dense/Tensordot/axesConst*
dtype0*
valueB:
o
:intent_aware_cross_category_attention/dense/Tensordot/freeConst*
dtype0*
valueB"       
k
;intent_aware_cross_category_attention/dense/Tensordot/ShapeShapeExpandDims_1*
T0*
out_type0
m
Cintent_aware_cross_category_attention/dense/Tensordot/GatherV2/axisConst*
dtype0*
value	B : 
�
>intent_aware_cross_category_attention/dense/Tensordot/GatherV2GatherV2;intent_aware_cross_category_attention/dense/Tensordot/Shape:intent_aware_cross_category_attention/dense/Tensordot/freeCintent_aware_cross_category_attention/dense/Tensordot/GatherV2/axis*
Tparams0*
Taxis0*
Tindices0
o
Eintent_aware_cross_category_attention/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0
�
@intent_aware_cross_category_attention/dense/Tensordot/GatherV2_1GatherV2;intent_aware_cross_category_attention/dense/Tensordot/Shape:intent_aware_cross_category_attention/dense/Tensordot/axesEintent_aware_cross_category_attention/dense/Tensordot/GatherV2_1/axis*
Tindices0*
Tparams0*
Taxis0
i
;intent_aware_cross_category_attention/dense/Tensordot/ConstConst*
valueB: *
dtype0
�
:intent_aware_cross_category_attention/dense/Tensordot/ProdProd>intent_aware_cross_category_attention/dense/Tensordot/GatherV2;intent_aware_cross_category_attention/dense/Tensordot/Const*
T0*

Tidx0*
	keep_dims( 
k
=intent_aware_cross_category_attention/dense/Tensordot/Const_1Const*
valueB: *
dtype0
�
<intent_aware_cross_category_attention/dense/Tensordot/Prod_1Prod@intent_aware_cross_category_attention/dense/Tensordot/GatherV2_1=intent_aware_cross_category_attention/dense/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0
k
Aintent_aware_cross_category_attention/dense/Tensordot/concat/axisConst*
value	B : *
dtype0
�
<intent_aware_cross_category_attention/dense/Tensordot/concatConcatV2:intent_aware_cross_category_attention/dense/Tensordot/free:intent_aware_cross_category_attention/dense/Tensordot/axesAintent_aware_cross_category_attention/dense/Tensordot/concat/axis*

Tidx0*
T0*
N
�
;intent_aware_cross_category_attention/dense/Tensordot/stackPack:intent_aware_cross_category_attention/dense/Tensordot/Prod<intent_aware_cross_category_attention/dense/Tensordot/Prod_1*
T0*

axis *
N
�
?intent_aware_cross_category_attention/dense/Tensordot/transpose	TransposeExpandDims_1<intent_aware_cross_category_attention/dense/Tensordot/concat*
Tperm0*
T0
�
=intent_aware_cross_category_attention/dense/Tensordot/ReshapeReshape?intent_aware_cross_category_attention/dense/Tensordot/transpose;intent_aware_cross_category_attention/dense/Tensordot/stack*
T0*
Tshape0
{
Fintent_aware_cross_category_attention/dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
Aintent_aware_cross_category_attention/dense/Tensordot/transpose_1	TransposeHmio_variable/intent_aware_cross_category_attention/dense/kernel/variableFintent_aware_cross_category_attention/dense/Tensordot/transpose_1/perm*
T0*
Tperm0
z
Eintent_aware_cross_category_attention/dense/Tensordot/Reshape_1/shapeConst*
valueB"      *
dtype0
�
?intent_aware_cross_category_attention/dense/Tensordot/Reshape_1ReshapeAintent_aware_cross_category_attention/dense/Tensordot/transpose_1Eintent_aware_cross_category_attention/dense/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
<intent_aware_cross_category_attention/dense/Tensordot/MatMulMatMul=intent_aware_cross_category_attention/dense/Tensordot/Reshape?intent_aware_cross_category_attention/dense/Tensordot/Reshape_1*
transpose_a( *
transpose_b( *
T0
k
=intent_aware_cross_category_attention/dense/Tensordot/Const_2Const*
valueB:*
dtype0
m
Cintent_aware_cross_category_attention/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0
�
>intent_aware_cross_category_attention/dense/Tensordot/concat_1ConcatV2>intent_aware_cross_category_attention/dense/Tensordot/GatherV2=intent_aware_cross_category_attention/dense/Tensordot/Const_2Cintent_aware_cross_category_attention/dense/Tensordot/concat_1/axis*

Tidx0*
T0*
N
�
5intent_aware_cross_category_attention/dense/TensordotReshape<intent_aware_cross_category_attention/dense/Tensordot/MatMul>intent_aware_cross_category_attention/dense/Tensordot/concat_1*
T0*
Tshape0
�
Jmio_variable/intent_aware_cross_category_attention/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:*C
	container64intent_aware_cross_category_attention/dense_1/kernel
�
Jmio_variable/intent_aware_cross_category_attention/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64intent_aware_cross_category_attention/dense_1/kernel*
shape
:
X
#Initializer_17/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_17/random_uniform/minConst*
dtype0*
valueB
 *׳ݾ
N
!Initializer_17/random_uniform/maxConst*
valueB
 *׳�>*
dtype0
�
+Initializer_17/random_uniform/RandomUniformRandomUniform#Initializer_17/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_17/random_uniform/subSub!Initializer_17/random_uniform/max!Initializer_17/random_uniform/min*
T0
�
!Initializer_17/random_uniform/mulMul+Initializer_17/random_uniform/RandomUniform!Initializer_17/random_uniform/sub*
T0
s
Initializer_17/random_uniformAdd!Initializer_17/random_uniform/mul!Initializer_17/random_uniform/min*
T0
�
	Assign_17AssignJmio_variable/intent_aware_cross_category_attention/dense_1/kernel/gradientInitializer_17/random_uniform*
T0*]
_classS
QOloc:@mio_variable/intent_aware_cross_category_attention/dense_1/kernel/gradient*
validate_shape(*
use_locking(
j
<intent_aware_cross_category_attention/dense_1/Tensordot/axesConst*
dtype0*
valueB:
q
<intent_aware_cross_category_attention/dense_1/Tensordot/freeConst*
valueB"       *
dtype0
m
=intent_aware_cross_category_attention/dense_1/Tensordot/ShapeShapeExpandDims_2*
T0*
out_type0
o
Eintent_aware_cross_category_attention/dense_1/Tensordot/GatherV2/axisConst*
value	B : *
dtype0
�
@intent_aware_cross_category_attention/dense_1/Tensordot/GatherV2GatherV2=intent_aware_cross_category_attention/dense_1/Tensordot/Shape<intent_aware_cross_category_attention/dense_1/Tensordot/freeEintent_aware_cross_category_attention/dense_1/Tensordot/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0
q
Gintent_aware_cross_category_attention/dense_1/Tensordot/GatherV2_1/axisConst*
dtype0*
value	B : 
�
Bintent_aware_cross_category_attention/dense_1/Tensordot/GatherV2_1GatherV2=intent_aware_cross_category_attention/dense_1/Tensordot/Shape<intent_aware_cross_category_attention/dense_1/Tensordot/axesGintent_aware_cross_category_attention/dense_1/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0
k
=intent_aware_cross_category_attention/dense_1/Tensordot/ConstConst*
valueB: *
dtype0
�
<intent_aware_cross_category_attention/dense_1/Tensordot/ProdProd@intent_aware_cross_category_attention/dense_1/Tensordot/GatherV2=intent_aware_cross_category_attention/dense_1/Tensordot/Const*

Tidx0*
	keep_dims( *
T0
m
?intent_aware_cross_category_attention/dense_1/Tensordot/Const_1Const*
dtype0*
valueB: 
�
>intent_aware_cross_category_attention/dense_1/Tensordot/Prod_1ProdBintent_aware_cross_category_attention/dense_1/Tensordot/GatherV2_1?intent_aware_cross_category_attention/dense_1/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0
m
Cintent_aware_cross_category_attention/dense_1/Tensordot/concat/axisConst*
value	B : *
dtype0
�
>intent_aware_cross_category_attention/dense_1/Tensordot/concatConcatV2<intent_aware_cross_category_attention/dense_1/Tensordot/free<intent_aware_cross_category_attention/dense_1/Tensordot/axesCintent_aware_cross_category_attention/dense_1/Tensordot/concat/axis*
T0*
N*

Tidx0
�
=intent_aware_cross_category_attention/dense_1/Tensordot/stackPack<intent_aware_cross_category_attention/dense_1/Tensordot/Prod>intent_aware_cross_category_attention/dense_1/Tensordot/Prod_1*
T0*

axis *
N
�
Aintent_aware_cross_category_attention/dense_1/Tensordot/transpose	TransposeExpandDims_2>intent_aware_cross_category_attention/dense_1/Tensordot/concat*
T0*
Tperm0
�
?intent_aware_cross_category_attention/dense_1/Tensordot/ReshapeReshapeAintent_aware_cross_category_attention/dense_1/Tensordot/transpose=intent_aware_cross_category_attention/dense_1/Tensordot/stack*
T0*
Tshape0
}
Hintent_aware_cross_category_attention/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
Cintent_aware_cross_category_attention/dense_1/Tensordot/transpose_1	TransposeJmio_variable/intent_aware_cross_category_attention/dense_1/kernel/variableHintent_aware_cross_category_attention/dense_1/Tensordot/transpose_1/perm*
T0*
Tperm0
|
Gintent_aware_cross_category_attention/dense_1/Tensordot/Reshape_1/shapeConst*
valueB"      *
dtype0
�
Aintent_aware_cross_category_attention/dense_1/Tensordot/Reshape_1ReshapeCintent_aware_cross_category_attention/dense_1/Tensordot/transpose_1Gintent_aware_cross_category_attention/dense_1/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
>intent_aware_cross_category_attention/dense_1/Tensordot/MatMulMatMul?intent_aware_cross_category_attention/dense_1/Tensordot/ReshapeAintent_aware_cross_category_attention/dense_1/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
m
?intent_aware_cross_category_attention/dense_1/Tensordot/Const_2Const*
valueB:*
dtype0
o
Eintent_aware_cross_category_attention/dense_1/Tensordot/concat_1/axisConst*
dtype0*
value	B : 
�
@intent_aware_cross_category_attention/dense_1/Tensordot/concat_1ConcatV2@intent_aware_cross_category_attention/dense_1/Tensordot/GatherV2?intent_aware_cross_category_attention/dense_1/Tensordot/Const_2Eintent_aware_cross_category_attention/dense_1/Tensordot/concat_1/axis*

Tidx0*
T0*
N
�
7intent_aware_cross_category_attention/dense_1/TensordotReshape>intent_aware_cross_category_attention/dense_1/Tensordot/MatMul@intent_aware_cross_category_attention/dense_1/Tensordot/concat_1*
T0*
Tshape0
�
Jmio_variable/intent_aware_cross_category_attention/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64intent_aware_cross_category_attention/dense_2/kernel*
shape
:
�
Jmio_variable/intent_aware_cross_category_attention/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64intent_aware_cross_category_attention/dense_2/kernel*
shape
:
X
#Initializer_18/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_18/random_uniform/minConst*
valueB
 *׳ݾ*
dtype0
N
!Initializer_18/random_uniform/maxConst*
dtype0*
valueB
 *׳�>
�
+Initializer_18/random_uniform/RandomUniformRandomUniform#Initializer_18/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_18/random_uniform/subSub!Initializer_18/random_uniform/max!Initializer_18/random_uniform/min*
T0
�
!Initializer_18/random_uniform/mulMul+Initializer_18/random_uniform/RandomUniform!Initializer_18/random_uniform/sub*
T0
s
Initializer_18/random_uniformAdd!Initializer_18/random_uniform/mul!Initializer_18/random_uniform/min*
T0
�
	Assign_18AssignJmio_variable/intent_aware_cross_category_attention/dense_2/kernel/gradientInitializer_18/random_uniform*
use_locking(*
T0*]
_classS
QOloc:@mio_variable/intent_aware_cross_category_attention/dense_2/kernel/gradient*
validate_shape(
j
<intent_aware_cross_category_attention/dense_2/Tensordot/axesConst*
valueB:*
dtype0
q
<intent_aware_cross_category_attention/dense_2/Tensordot/freeConst*
valueB"       *
dtype0
m
=intent_aware_cross_category_attention/dense_2/Tensordot/ShapeShapeExpandDims_2*
T0*
out_type0
o
Eintent_aware_cross_category_attention/dense_2/Tensordot/GatherV2/axisConst*
value	B : *
dtype0
�
@intent_aware_cross_category_attention/dense_2/Tensordot/GatherV2GatherV2=intent_aware_cross_category_attention/dense_2/Tensordot/Shape<intent_aware_cross_category_attention/dense_2/Tensordot/freeEintent_aware_cross_category_attention/dense_2/Tensordot/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0
q
Gintent_aware_cross_category_attention/dense_2/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0
�
Bintent_aware_cross_category_attention/dense_2/Tensordot/GatherV2_1GatherV2=intent_aware_cross_category_attention/dense_2/Tensordot/Shape<intent_aware_cross_category_attention/dense_2/Tensordot/axesGintent_aware_cross_category_attention/dense_2/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0
k
=intent_aware_cross_category_attention/dense_2/Tensordot/ConstConst*
valueB: *
dtype0
�
<intent_aware_cross_category_attention/dense_2/Tensordot/ProdProd@intent_aware_cross_category_attention/dense_2/Tensordot/GatherV2=intent_aware_cross_category_attention/dense_2/Tensordot/Const*

Tidx0*
	keep_dims( *
T0
m
?intent_aware_cross_category_attention/dense_2/Tensordot/Const_1Const*
valueB: *
dtype0
�
>intent_aware_cross_category_attention/dense_2/Tensordot/Prod_1ProdBintent_aware_cross_category_attention/dense_2/Tensordot/GatherV2_1?intent_aware_cross_category_attention/dense_2/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0
m
Cintent_aware_cross_category_attention/dense_2/Tensordot/concat/axisConst*
value	B : *
dtype0
�
>intent_aware_cross_category_attention/dense_2/Tensordot/concatConcatV2<intent_aware_cross_category_attention/dense_2/Tensordot/free<intent_aware_cross_category_attention/dense_2/Tensordot/axesCintent_aware_cross_category_attention/dense_2/Tensordot/concat/axis*
N*

Tidx0*
T0
�
=intent_aware_cross_category_attention/dense_2/Tensordot/stackPack<intent_aware_cross_category_attention/dense_2/Tensordot/Prod>intent_aware_cross_category_attention/dense_2/Tensordot/Prod_1*
T0*

axis *
N
�
Aintent_aware_cross_category_attention/dense_2/Tensordot/transpose	TransposeExpandDims_2>intent_aware_cross_category_attention/dense_2/Tensordot/concat*
Tperm0*
T0
�
?intent_aware_cross_category_attention/dense_2/Tensordot/ReshapeReshapeAintent_aware_cross_category_attention/dense_2/Tensordot/transpose=intent_aware_cross_category_attention/dense_2/Tensordot/stack*
T0*
Tshape0
}
Hintent_aware_cross_category_attention/dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
Cintent_aware_cross_category_attention/dense_2/Tensordot/transpose_1	TransposeJmio_variable/intent_aware_cross_category_attention/dense_2/kernel/variableHintent_aware_cross_category_attention/dense_2/Tensordot/transpose_1/perm*
Tperm0*
T0
|
Gintent_aware_cross_category_attention/dense_2/Tensordot/Reshape_1/shapeConst*
valueB"      *
dtype0
�
Aintent_aware_cross_category_attention/dense_2/Tensordot/Reshape_1ReshapeCintent_aware_cross_category_attention/dense_2/Tensordot/transpose_1Gintent_aware_cross_category_attention/dense_2/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
>intent_aware_cross_category_attention/dense_2/Tensordot/MatMulMatMul?intent_aware_cross_category_attention/dense_2/Tensordot/ReshapeAintent_aware_cross_category_attention/dense_2/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
m
?intent_aware_cross_category_attention/dense_2/Tensordot/Const_2Const*
valueB:*
dtype0
o
Eintent_aware_cross_category_attention/dense_2/Tensordot/concat_1/axisConst*
value	B : *
dtype0
�
@intent_aware_cross_category_attention/dense_2/Tensordot/concat_1ConcatV2@intent_aware_cross_category_attention/dense_2/Tensordot/GatherV2?intent_aware_cross_category_attention/dense_2/Tensordot/Const_2Eintent_aware_cross_category_attention/dense_2/Tensordot/concat_1/axis*

Tidx0*
T0*
N
�
7intent_aware_cross_category_attention/dense_2/TensordotReshape>intent_aware_cross_category_attention/dense_2/Tensordot/MatMul@intent_aware_cross_category_attention/dense_2/Tensordot/concat_1*
T0*
Tshape0
U
+intent_aware_cross_category_attention/ConstConst*
value	B :*
dtype0
h
5intent_aware_cross_category_attention/split/split_dimConst*
valueB :
���������*
dtype0
�
+intent_aware_cross_category_attention/splitSplit5intent_aware_cross_category_attention/split/split_dim5intent_aware_cross_category_attention/dense/Tensordot*
	num_split*
T0
[
1intent_aware_cross_category_attention/concat/axisConst*
dtype0*
value	B : 
�
,intent_aware_cross_category_attention/concatConcatV2+intent_aware_cross_category_attention/split-intent_aware_cross_category_attention/split:11intent_aware_cross_category_attention/concat/axis*
N*

Tidx0*
T0
W
-intent_aware_cross_category_attention/Const_1Const*
dtype0*
value	B :
j
7intent_aware_cross_category_attention/split_1/split_dimConst*
valueB :
���������*
dtype0
�
-intent_aware_cross_category_attention/split_1Split7intent_aware_cross_category_attention/split_1/split_dim7intent_aware_cross_category_attention/dense_1/Tensordot*
T0*
	num_split
]
3intent_aware_cross_category_attention/concat_1/axisConst*
value	B : *
dtype0
�
.intent_aware_cross_category_attention/concat_1ConcatV2-intent_aware_cross_category_attention/split_1/intent_aware_cross_category_attention/split_1:13intent_aware_cross_category_attention/concat_1/axis*
T0*
N*

Tidx0
W
-intent_aware_cross_category_attention/Const_2Const*
value	B :*
dtype0
j
7intent_aware_cross_category_attention/split_2/split_dimConst*
valueB :
���������*
dtype0
�
-intent_aware_cross_category_attention/split_2Split7intent_aware_cross_category_attention/split_2/split_dim7intent_aware_cross_category_attention/dense_2/Tensordot*
T0*
	num_split
]
3intent_aware_cross_category_attention/concat_2/axisConst*
value	B : *
dtype0
�
.intent_aware_cross_category_attention/concat_2ConcatV2-intent_aware_cross_category_attention/split_2/intent_aware_cross_category_attention/split_2:13intent_aware_cross_category_attention/concat_2/axis*
T0*
N*

Tidx0
m
4intent_aware_cross_category_attention/transpose/permConst*!
valueB"          *
dtype0
�
/intent_aware_cross_category_attention/transpose	Transpose.intent_aware_cross_category_attention/concat_14intent_aware_cross_category_attention/transpose/perm*
Tperm0*
T0
�
,intent_aware_cross_category_attention/MatMulBatchMatMul,intent_aware_cross_category_attention/concat/intent_aware_cross_category_attention/transpose*
adj_x( *
adj_y( *
T0
\
/intent_aware_cross_category_attention/truediv/yConst*
valueB
 *�5@*
dtype0
�
-intent_aware_cross_category_attention/truedivRealDiv,intent_aware_cross_category_attention/MatMul/intent_aware_cross_category_attention/truediv/y*
T0
p
-intent_aware_cross_category_attention/SoftmaxSoftmax-intent_aware_cross_category_attention/truediv*
T0
�
.intent_aware_cross_category_attention/MatMul_1BatchMatMul-intent_aware_cross_category_attention/Softmax.intent_aware_cross_category_attention/concat_2*
adj_x( *
adj_y( *
T0
W
-intent_aware_cross_category_attention/Const_3Const*
value	B :*
dtype0
a
7intent_aware_cross_category_attention/split_3/split_dimConst*
value	B : *
dtype0
�
-intent_aware_cross_category_attention/split_3Split7intent_aware_cross_category_attention/split_3/split_dim.intent_aware_cross_category_attention/MatMul_1*
T0*
	num_split
]
3intent_aware_cross_category_attention/concat_3/axisConst*
dtype0*
value	B :
�
.intent_aware_cross_category_attention/concat_3ConcatV2-intent_aware_cross_category_attention/split_3/intent_aware_cross_category_attention/split_3:13intent_aware_cross_category_attention/concat_3/axis*
T0*
N*

Tidx0
�
Jmio_variable/intent_aware_cross_category_attention/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:*C
	container64intent_aware_cross_category_attention/dense_3/kernel
�
Jmio_variable/intent_aware_cross_category_attention/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64intent_aware_cross_category_attention/dense_3/kernel*
shape
:
X
#Initializer_19/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_19/random_uniform/minConst*
valueB
 *׳ݾ*
dtype0
N
!Initializer_19/random_uniform/maxConst*
valueB
 *׳�>*
dtype0
�
+Initializer_19/random_uniform/RandomUniformRandomUniform#Initializer_19/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_19/random_uniform/subSub!Initializer_19/random_uniform/max!Initializer_19/random_uniform/min*
T0
�
!Initializer_19/random_uniform/mulMul+Initializer_19/random_uniform/RandomUniform!Initializer_19/random_uniform/sub*
T0
s
Initializer_19/random_uniformAdd!Initializer_19/random_uniform/mul!Initializer_19/random_uniform/min*
T0
�
	Assign_19AssignJmio_variable/intent_aware_cross_category_attention/dense_3/kernel/gradientInitializer_19/random_uniform*
use_locking(*
T0*]
_classS
QOloc:@mio_variable/intent_aware_cross_category_attention/dense_3/kernel/gradient*
validate_shape(
�
Hmio_variable/intent_aware_cross_category_attention/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42intent_aware_cross_category_attention/dense_3/bias*
shape:
�
Hmio_variable/intent_aware_cross_category_attention/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42intent_aware_cross_category_attention/dense_3/bias*
shape:
E
Initializer_20/zerosConst*
valueB*    *
dtype0
�
	Assign_20AssignHmio_variable/intent_aware_cross_category_attention/dense_3/bias/gradientInitializer_20/zeros*
use_locking(*
T0*[
_classQ
OMloc:@mio_variable/intent_aware_cross_category_attention/dense_3/bias/gradient*
validate_shape(
j
<intent_aware_cross_category_attention/dense_3/Tensordot/axesConst*
valueB:*
dtype0
q
<intent_aware_cross_category_attention/dense_3/Tensordot/freeConst*
valueB"       *
dtype0
�
=intent_aware_cross_category_attention/dense_3/Tensordot/ShapeShape.intent_aware_cross_category_attention/concat_3*
T0*
out_type0
o
Eintent_aware_cross_category_attention/dense_3/Tensordot/GatherV2/axisConst*
value	B : *
dtype0
�
@intent_aware_cross_category_attention/dense_3/Tensordot/GatherV2GatherV2=intent_aware_cross_category_attention/dense_3/Tensordot/Shape<intent_aware_cross_category_attention/dense_3/Tensordot/freeEintent_aware_cross_category_attention/dense_3/Tensordot/GatherV2/axis*
Tindices0*
Tparams0*
Taxis0
q
Gintent_aware_cross_category_attention/dense_3/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0
�
Bintent_aware_cross_category_attention/dense_3/Tensordot/GatherV2_1GatherV2=intent_aware_cross_category_attention/dense_3/Tensordot/Shape<intent_aware_cross_category_attention/dense_3/Tensordot/axesGintent_aware_cross_category_attention/dense_3/Tensordot/GatherV2_1/axis*
Tindices0*
Tparams0*
Taxis0
k
=intent_aware_cross_category_attention/dense_3/Tensordot/ConstConst*
valueB: *
dtype0
�
<intent_aware_cross_category_attention/dense_3/Tensordot/ProdProd@intent_aware_cross_category_attention/dense_3/Tensordot/GatherV2=intent_aware_cross_category_attention/dense_3/Tensordot/Const*
T0*

Tidx0*
	keep_dims( 
m
?intent_aware_cross_category_attention/dense_3/Tensordot/Const_1Const*
valueB: *
dtype0
�
>intent_aware_cross_category_attention/dense_3/Tensordot/Prod_1ProdBintent_aware_cross_category_attention/dense_3/Tensordot/GatherV2_1?intent_aware_cross_category_attention/dense_3/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0
m
Cintent_aware_cross_category_attention/dense_3/Tensordot/concat/axisConst*
value	B : *
dtype0
�
>intent_aware_cross_category_attention/dense_3/Tensordot/concatConcatV2<intent_aware_cross_category_attention/dense_3/Tensordot/free<intent_aware_cross_category_attention/dense_3/Tensordot/axesCintent_aware_cross_category_attention/dense_3/Tensordot/concat/axis*

Tidx0*
T0*
N
�
=intent_aware_cross_category_attention/dense_3/Tensordot/stackPack<intent_aware_cross_category_attention/dense_3/Tensordot/Prod>intent_aware_cross_category_attention/dense_3/Tensordot/Prod_1*
N*
T0*

axis 
�
Aintent_aware_cross_category_attention/dense_3/Tensordot/transpose	Transpose.intent_aware_cross_category_attention/concat_3>intent_aware_cross_category_attention/dense_3/Tensordot/concat*
Tperm0*
T0
�
?intent_aware_cross_category_attention/dense_3/Tensordot/ReshapeReshapeAintent_aware_cross_category_attention/dense_3/Tensordot/transpose=intent_aware_cross_category_attention/dense_3/Tensordot/stack*
T0*
Tshape0
}
Hintent_aware_cross_category_attention/dense_3/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
Cintent_aware_cross_category_attention/dense_3/Tensordot/transpose_1	TransposeJmio_variable/intent_aware_cross_category_attention/dense_3/kernel/variableHintent_aware_cross_category_attention/dense_3/Tensordot/transpose_1/perm*
T0*
Tperm0
|
Gintent_aware_cross_category_attention/dense_3/Tensordot/Reshape_1/shapeConst*
valueB"      *
dtype0
�
Aintent_aware_cross_category_attention/dense_3/Tensordot/Reshape_1ReshapeCintent_aware_cross_category_attention/dense_3/Tensordot/transpose_1Gintent_aware_cross_category_attention/dense_3/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
>intent_aware_cross_category_attention/dense_3/Tensordot/MatMulMatMul?intent_aware_cross_category_attention/dense_3/Tensordot/ReshapeAintent_aware_cross_category_attention/dense_3/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
m
?intent_aware_cross_category_attention/dense_3/Tensordot/Const_2Const*
valueB:*
dtype0
o
Eintent_aware_cross_category_attention/dense_3/Tensordot/concat_1/axisConst*
value	B : *
dtype0
�
@intent_aware_cross_category_attention/dense_3/Tensordot/concat_1ConcatV2@intent_aware_cross_category_attention/dense_3/Tensordot/GatherV2?intent_aware_cross_category_attention/dense_3/Tensordot/Const_2Eintent_aware_cross_category_attention/dense_3/Tensordot/concat_1/axis*
N*

Tidx0*
T0
�
7intent_aware_cross_category_attention/dense_3/TensordotReshape>intent_aware_cross_category_attention/dense_3/Tensordot/MatMul@intent_aware_cross_category_attention/dense_3/Tensordot/concat_1*
T0*
Tshape0
�
5intent_aware_cross_category_attention/dense_3/BiasAddBiasAdd7intent_aware_cross_category_attention/dense_3/TensordotHmio_variable/intent_aware_cross_category_attention/dense_3/bias/variable*
T0*
data_formatNHWC
e
SqueezeSqueeze1intent_aware_cross_pxtr_attention/dense_3/BiasAdd*
squeeze_dims
*
T0
k
	Squeeze_1Squeeze5intent_aware_cross_category_attention/dense_3/BiasAdd*
squeeze_dims
*
T0
7
concat_3/axisConst*
value	B :*
dtype0
o
concat_3ConcatV2Squeeze	Squeeze_1intent_emb/dense/Sigmoidconcat_3/axis*

Tidx0*
T0*
N
�
-mio_variable/projection/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerprojection/dense/kernel*
shape
:0
�
-mio_variable/projection/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerprojection/dense/kernel*
shape
:0
Z
%Initializer_21/truncated_normal/shapeConst*
valueB"0      *
dtype0
Q
$Initializer_21/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_21/truncated_normal/stddevConst*
valueB
 *   ?*
dtype0
�
/Initializer_21/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_21/truncated_normal/shape*
T0*
dtype0*
seed2*
seed���)
�
#Initializer_21/truncated_normal/mulMul/Initializer_21/truncated_normal/TruncatedNormal&Initializer_21/truncated_normal/stddev*
T0
z
Initializer_21/truncated_normalAdd#Initializer_21/truncated_normal/mul$Initializer_21/truncated_normal/mean*
T0
�
	Assign_21Assign-mio_variable/projection/dense/kernel/gradientInitializer_21/truncated_normal*
use_locking(*
T0*@
_class6
42loc:@mio_variable/projection/dense/kernel/gradient*
validate_shape(
�
+mio_variable/projection/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containerprojection/dense/bias
�
+mio_variable/projection/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerprojection/dense/bias*
shape:
E
Initializer_22/zerosConst*
dtype0*
valueB*    
�
	Assign_22Assign+mio_variable/projection/dense/bias/gradientInitializer_22/zeros*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@mio_variable/projection/dense/bias/gradient
�
projection/dense/MatMulMatMulconcat_3-mio_variable/projection/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
projection/dense/BiasAddBiasAddprojection/dense/MatMul+mio_variable/projection/dense/bias/variable*
T0*
data_formatNHWC
>
projection/dense/EluEluprojection/dense/BiasAdd*
T0
2
add/xConst*
dtype0*
valueB
 *   @
0
addAddadd/xprojection/dense/Elu*
T0

LogLogadd*
T0
"
MulMulconcat_1Log*
T0
?
Sum/reduction_indicesConst*
dtype0*
value	B :
L
SumSumMulSum/reduction_indices*

Tidx0*
	keep_dims(*
T0
�
1mio_variable/ensemble_score/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS**
	containerensemble_score/dense/kernel*
shape
:
�
1mio_variable/ensemble_score/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:**
	containerensemble_score/dense/kernel
Z
%Initializer_23/truncated_normal/shapeConst*
dtype0*
valueB"      
Q
$Initializer_23/truncated_normal/meanConst*
dtype0*
valueB
 *    
S
&Initializer_23/truncated_normal/stddevConst*
valueB
 *   ?*
dtype0
�
/Initializer_23/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_23/truncated_normal/shape*
dtype0*
seed2*
seed���)*
T0
�
#Initializer_23/truncated_normal/mulMul/Initializer_23/truncated_normal/TruncatedNormal&Initializer_23/truncated_normal/stddev*
T0
z
Initializer_23/truncated_normalAdd#Initializer_23/truncated_normal/mul$Initializer_23/truncated_normal/mean*
T0
�
	Assign_23Assign1mio_variable/ensemble_score/dense/kernel/gradientInitializer_23/truncated_normal*
use_locking(*
T0*D
_class:
86loc:@mio_variable/ensemble_score/dense/kernel/gradient*
validate_shape(
�
/mio_variable/ensemble_score/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*(
	containerensemble_score/dense/bias
�
/mio_variable/ensemble_score/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerensemble_score/dense/bias*
shape:
E
Initializer_24/zerosConst*
dtype0*
valueB*    
�
	Assign_24Assign/mio_variable/ensemble_score/dense/bias/gradientInitializer_24/zeros*
use_locking(*
T0*B
_class8
64loc:@mio_variable/ensemble_score/dense/bias/gradient*
validate_shape(
�
ensemble_score/dense/MatMulMatMulSum1mio_variable/ensemble_score/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
ensemble_score/dense/BiasAddBiasAddensemble_score/dense/MatMul/mio_variable/ensemble_score/dense/bias/variable*
T0*
data_formatNHWC
N
ensemble_score/dense/SigmoidSigmoidensemble_score/dense/BiasAdd*
T0
�
'mio_extra_param/interact_label/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerinteract_label*
shape:���������
�
'mio_extra_param/interact_label/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:���������*
	containerinteract_label
�
/mio_extra_param/comment_effective_stay/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containercomment_effective_stay*
shape:���������
�
/mio_extra_param/comment_effective_stay/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containercomment_effective_stay*
shape:���������
�
"mio_extra_param/long_view/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	container	long_view*
shape:���������
�
"mio_extra_param/long_view/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	container	long_view*
shape:���������
�
'mio_extra_param/effective_view/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containereffective_view*
shape:���������
�
'mio_extra_param/effective_view/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containereffective_view*
shape:���������
�
mio_extra_param/follow/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerfollow*
shape:���������
�
mio_extra_param/follow/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerfollow*
shape:���������
U
ones_like/ShapeShape"mio_extra_param/long_view/variable*
T0*
out_type0
<
ones_like/ConstConst*
valueB
 *  �?*
dtype0
N
	ones_likeFillones_like/Shapeones_like/Const*
T0*

index_type0
D

zeros_like	ZerosLike"mio_extra_param/long_view/variable*
T0
6
	Greater/yConst*
valueB
 *    *
dtype0
W
GreaterGreater/mio_extra_param/comment_effective_stay/variable	Greater/y*
T0
9
SelectSelectGreater	ones_like
zeros_like*
T0
�
+mio_extra_param/comment_watch_time/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*!
	containercomment_watch_time*
shape:���������
�
+mio_extra_param/comment_watch_time/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:���������*!
	containercomment_watch_time
�
-mio_extra_param/comment_action_coeff/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*#
	containercomment_action_coeff*
shape:���������
�
-mio_extra_param/comment_action_coeff/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:���������*#
	containercomment_action_coeff
�
+mio_extra_param/comment_stay_coeff/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*!
	containercomment_stay_coeff*
shape:���������
�
+mio_extra_param/comment_stay_coeff/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*!
	containercomment_stay_coeff*
shape:���������
�
.mio_extra_param/comment_action_weight/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercomment_action_weight*
shape:���������
�
.mio_extra_param/comment_action_weight/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercomment_action_weight*
shape:���������
�
 mio_extra_param/comment/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	container	comment*
shape:���������
�
 mio_extra_param/comment/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	container	comment*
shape:���������
�
&mio_extra_param/comment_coeff/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containercomment_coeff*
shape:���������
�
&mio_extra_param/comment_coeff/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containercomment_coeff*
shape:���������
o
mul_1Mul+mio_extra_param/comment_watch_time/variable+mio_extra_param/comment_stay_coeff/variable*
T0
t
mul_2Mul.mio_extra_param/comment_action_weight/variable-mio_extra_param/comment_action_coeff/variable*
T0
#
add_1Addmul_1mul_2*
T0
_
mul_3Mul mio_extra_param/comment/variable&mio_extra_param/comment_coeff/variable*
T0
#
add_2Addadd_1mul_3*
T0
�
%mio_extra_param/good_quality/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containergood_quality*
shape:���������
�
%mio_extra_param/good_quality/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:���������*
	containergood_quality
1
Const_1Const*
dtype0*
value	B :
9
split/split_dimConst*
value	B :*
dtype0
O
splitSplitsplit/split_dimprojection/dense/Elu*
T0*
	num_split"