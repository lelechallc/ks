
/
ConstConst*
value	B : *
dtype0
8
is_train/inputConst*
dtype0*
value	B : 
L
is_trainPlaceholderWithDefaultis_train/input*
shape: *
dtype0
K
MIO_TABLE_ADDRESSConst"/device:CPU:0*
value
B � *
dtype0
�
mio_embeddings/pid_emb/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	container	pid_emb*
shape:��������� 
�
mio_embeddings/pid_emb/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:��������� *
	container	pid_emb
�
mio_embeddings/aid_emb/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	container	aid_emb*
shape:���������@
�
mio_embeddings/aid_emb/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	container	aid_emb*
shape:���������@
�
mio_embeddings/pid_xtr/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	container	pid_xtr*
shape:���������8
�
mio_embeddings/pid_xtr/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:���������8*
	container	pid_xtr
�
 mio_embeddings/pid_stat/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	container
pid_stat*
shape:���������8
�
 mio_embeddings/pid_stat/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:���������8*
	container
pid_stat
�
 mio_embeddings/pid_hetu/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	container
pid_hetu*
shape:���������
�
 mio_embeddings/pid_hetu/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	container
pid_hetu*
shape:���������
�
mio_embeddings/pid_cnt/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	container	pid_cnt*
shape:���������
�
mio_embeddings/pid_cnt/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	container	pid_cnt*
shape:���������
�
 mio_embeddings/pid_pxtr/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:����������*
	container
pid_pxtr
�
 mio_embeddings/pid_pxtr/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	container
pid_pxtr*
shape:����������
�
 mio_embeddings/top_bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	container
top_bias*
shape:��������� 
�
 mio_embeddings/top_bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	container
top_bias*
shape:��������� 
�
"mio_embeddings/pid_play_f/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:��������� *
	container
pid_play_f
�
"mio_embeddings/pid_play_f/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	container
pid_play_f*
shape:��������� 
5
concat/axisConst*
value	B :*
dtype0
�
concatConcatV2mio_embeddings/pid_emb/variablemio_embeddings/aid_emb/variablemio_embeddings/pid_xtr/variable mio_embeddings/pid_stat/variable mio_embeddings/pid_hetu/variablemio_embeddings/pid_cnt/variable mio_embeddings/top_bias/variable mio_embeddings/pid_pxtr/variable"mio_embeddings/pid_play_f/variableconcat/axis*
T0*
N	*

Tidx0
�
2mio_compress_indices/COMPRESS_INDEX__USER/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerCOMPRESS_INDEX__USER*
shape:���������
�
2mio_compress_indices/COMPRESS_INDEX__USER/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerCOMPRESS_INDEX__USER*
shape:���������
h
CastCast2mio_compress_indices/COMPRESS_INDEX__USER/variable*

SrcT0*
Truncate( *

DstT0
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
 mio_embeddings/did_stat/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	container
did_stat*
shape:���������0
�
 mio_embeddings/did_stat/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	container
did_stat*
shape:���������0
�
"mio_embeddings/uid_live_f/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:���������0*
	container
uid_live_f
�
"mio_embeddings/uid_live_f/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	container
uid_live_f*
shape:���������0
�
!mio_embeddings/uid_loc_f/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	container	uid_loc_f*
shape:���������
�
!mio_embeddings/uid_loc_f/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	container	uid_loc_f*
shape:���������
�
$mio_embeddings/uid_viewid_f/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containeruid_viewid_f*
shape:���������@
�
$mio_embeddings/uid_viewid_f/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:���������@*
	containeruid_viewid_f
�
%mio_embeddings/realshow_tags/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerrealshow_tags*
shape:����������
�
%mio_embeddings/realshow_tags/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerrealshow_tags*
shape:����������
�
'mio_embeddings/short_term_pids/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:����������*
	containershort_term_pids
�
'mio_embeddings/short_term_pids/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:����������*
	containershort_term_pids
�
'mio_embeddings/short_term_aids/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containershort_term_aids*
shape:����������
�
'mio_embeddings/short_term_aids/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containershort_term_aids*
shape:����������
�
(mio_embeddings/short_term_times/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containershort_term_times*
shape:����������
�
(mio_embeddings/short_term_times/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containershort_term_times*
shape:����������
=
Reshape/tensor/axisConst*
value	B : *
dtype0
�
Reshape/tensorGatherV2'mio_embeddings/short_term_pids/variableCastReshape/tensor/axis*
Tparams0*
Taxis0*
Tindices0
F
Reshape/shapeConst*!
valueB"����2       *
dtype0
H
ReshapeReshapeReshape/tensorReshape/shape*
T0*
Tshape0
?
Reshape_1/tensor/axisConst*
dtype0*
value	B : 
�
Reshape_1/tensorGatherV2'mio_embeddings/short_term_aids/variableCastReshape_1/tensor/axis*
Tindices0*
Tparams0*
Taxis0
H
Reshape_1/shapeConst*!
valueB"����2       *
dtype0
N
	Reshape_1ReshapeReshape_1/tensorReshape_1/shape*
T0*
Tshape0
?
Reshape_2/tensor/axisConst*
value	B : *
dtype0
�
Reshape_2/tensorGatherV2%mio_embeddings/realshow_tags/variableCastReshape_2/tensor/axis*
Taxis0*
Tindices0*
Tparams0
H
Reshape_2/shapeConst*!
valueB"����2      *
dtype0
N
	Reshape_2ReshapeReshape_2/tensorReshape_2/shape*
T0*
Tshape0
?
Reshape_3/tensor/axisConst*
value	B : *
dtype0
�
Reshape_3/tensorGatherV2(mio_embeddings/short_term_times/variableCastReshape_3/tensor/axis*
Taxis0*
Tindices0*
Tparams0
H
Reshape_3/shapeConst*!
valueB"����2      *
dtype0
N
	Reshape_3ReshapeReshape_3/tensorReshape_3/shape*
T0*
Tshape0
7
concat_1/axisConst*
value	B :*
dtype0
k
concat_1ConcatV2Reshape	Reshape_1	Reshape_2	Reshape_3concat_1/axis*

Tidx0*
T0*
N
?
Sum/reduction_indicesConst*
value	B :*
dtype0
Q
SumSumconcat_1Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
�
(mio_embeddings/uid_like_list_id/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containeruid_like_list_id*
shape:����������
�
(mio_embeddings/uid_like_list_id/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containeruid_like_list_id*
shape:����������
?
Reshape_4/tensor/axisConst*
value	B : *
dtype0
�
Reshape_4/tensorGatherV2(mio_embeddings/uid_like_list_id/variableCastReshape_4/tensor/axis*
Tindices0*
Tparams0*
Taxis0
H
Reshape_4/shapeConst*!
valueB"����2       *
dtype0
N
	Reshape_4ReshapeReshape_4/tensorReshape_4/shape*
T0*
Tshape0
�
+mio_embeddings/uid_like_list_hetu1/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*"
	containeruid_like_list_hetu1*
shape:����������
�
+mio_embeddings/uid_like_list_hetu1/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*"
	containeruid_like_list_hetu1*
shape:����������
?
Reshape_5/tensor/axisConst*
value	B : *
dtype0
�
Reshape_5/tensorGatherV2+mio_embeddings/uid_like_list_hetu1/variableCastReshape_5/tensor/axis*
Taxis0*
Tindices0*
Tparams0
H
Reshape_5/shapeConst*!
valueB"����2      *
dtype0
N
	Reshape_5ReshapeReshape_5/tensorReshape_5/shape*
T0*
Tshape0
�
+mio_embeddings/uid_like_list_hetu2/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*"
	containeruid_like_list_hetu2*
shape:����������
�
+mio_embeddings/uid_like_list_hetu2/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*"
	containeruid_like_list_hetu2*
shape:����������
?
Reshape_6/tensor/axisConst*
dtype0*
value	B : 
�
Reshape_6/tensorGatherV2+mio_embeddings/uid_like_list_hetu2/variableCastReshape_6/tensor/axis*
Taxis0*
Tindices0*
Tparams0
H
Reshape_6/shapeConst*!
valueB"����2      *
dtype0
N
	Reshape_6ReshapeReshape_6/tensorReshape_6/shape*
T0*
Tshape0
7
concat_2/axisConst*
dtype0*
value	B :
b
concat_2ConcatV2	Reshape_4	Reshape_5	Reshape_6concat_2/axis*
T0*
N*

Tidx0
�
*mio_embeddings/uid_follow_list_id/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_follow_list_id*
shape:����������
�
*mio_embeddings/uid_follow_list_id/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*!
	containeruid_follow_list_id*
shape:����������
?
Reshape_7/tensor/axisConst*
dtype0*
value	B : 
�
Reshape_7/tensorGatherV2*mio_embeddings/uid_follow_list_id/variableCastReshape_7/tensor/axis*
Taxis0*
Tindices0*
Tparams0
H
Reshape_7/shapeConst*
dtype0*!
valueB"����2       
N
	Reshape_7ReshapeReshape_7/tensorReshape_7/shape*
T0*
Tshape0
�
-mio_embeddings/uid_follow_list_hetu1/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containeruid_follow_list_hetu1*
shape:����������
�
-mio_embeddings/uid_follow_list_hetu1/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containeruid_follow_list_hetu1*
shape:����������
?
Reshape_8/tensor/axisConst*
value	B : *
dtype0
�
Reshape_8/tensorGatherV2-mio_embeddings/uid_follow_list_hetu1/variableCastReshape_8/tensor/axis*
Tparams0*
Taxis0*
Tindices0
H
Reshape_8/shapeConst*!
valueB"����2      *
dtype0
N
	Reshape_8ReshapeReshape_8/tensorReshape_8/shape*
T0*
Tshape0
�
-mio_embeddings/uid_follow_list_hetu2/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:����������*$
	containeruid_follow_list_hetu2
�
-mio_embeddings/uid_follow_list_hetu2/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containeruid_follow_list_hetu2*
shape:����������
?
Reshape_9/tensor/axisConst*
value	B : *
dtype0
�
Reshape_9/tensorGatherV2-mio_embeddings/uid_follow_list_hetu2/variableCastReshape_9/tensor/axis*
Tparams0*
Taxis0*
Tindices0
H
Reshape_9/shapeConst*!
valueB"����2      *
dtype0
N
	Reshape_9ReshapeReshape_9/tensorReshape_9/shape*
T0*
Tshape0
7
concat_3/axisConst*
value	B :*
dtype0
b
concat_3ConcatV2	Reshape_7	Reshape_8	Reshape_9concat_3/axis*
T0*
N*

Tidx0
�
+mio_embeddings/uid_forward_list_id/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*"
	containeruid_forward_list_id*
shape:����������
�
+mio_embeddings/uid_forward_list_id/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*"
	containeruid_forward_list_id*
shape:����������
@
Reshape_10/tensor/axisConst*
dtype0*
value	B : 
�
Reshape_10/tensorGatherV2+mio_embeddings/uid_forward_list_id/variableCastReshape_10/tensor/axis*
Taxis0*
Tindices0*
Tparams0
I
Reshape_10/shapeConst*
dtype0*!
valueB"����2       
Q

Reshape_10ReshapeReshape_10/tensorReshape_10/shape*
T0*
Tshape0
�
.mio_embeddings/uid_forward_list_hetu1/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containeruid_forward_list_hetu1*
shape:����������
�
.mio_embeddings/uid_forward_list_hetu1/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containeruid_forward_list_hetu1*
shape:����������
@
Reshape_11/tensor/axisConst*
value	B : *
dtype0
�
Reshape_11/tensorGatherV2.mio_embeddings/uid_forward_list_hetu1/variableCastReshape_11/tensor/axis*
Tparams0*
Taxis0*
Tindices0
I
Reshape_11/shapeConst*!
valueB"����2      *
dtype0
Q

Reshape_11ReshapeReshape_11/tensorReshape_11/shape*
T0*
Tshape0
�
.mio_embeddings/uid_forward_list_hetu2/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containeruid_forward_list_hetu2*
shape:����������
�
.mio_embeddings/uid_forward_list_hetu2/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containeruid_forward_list_hetu2*
shape:����������
@
Reshape_12/tensor/axisConst*
value	B : *
dtype0
�
Reshape_12/tensorGatherV2.mio_embeddings/uid_forward_list_hetu2/variableCastReshape_12/tensor/axis*
Taxis0*
Tindices0*
Tparams0
I
Reshape_12/shapeConst*!
valueB"����2      *
dtype0
Q

Reshape_12ReshapeReshape_12/tensorReshape_12/shape*
T0*
Tshape0
7
concat_4/axisConst*
dtype0*
value	B :
e
concat_4ConcatV2
Reshape_10
Reshape_11
Reshape_12concat_4/axis*
T0*
N*

Tidx0
�
+mio_embeddings/uid_comment_list_id/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*"
	containeruid_comment_list_id*
shape:����������
�
+mio_embeddings/uid_comment_list_id/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*"
	containeruid_comment_list_id*
shape:����������
@
Reshape_13/tensor/axisConst*
value	B : *
dtype0
�
Reshape_13/tensorGatherV2+mio_embeddings/uid_comment_list_id/variableCastReshape_13/tensor/axis*
Tparams0*
Taxis0*
Tindices0
I
Reshape_13/shapeConst*!
valueB"����2       *
dtype0
Q

Reshape_13ReshapeReshape_13/tensorReshape_13/shape*
T0*
Tshape0
�
.mio_embeddings/uid_comment_list_hetu1/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containeruid_comment_list_hetu1*
shape:����������
�
.mio_embeddings/uid_comment_list_hetu1/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containeruid_comment_list_hetu1*
shape:����������
@
Reshape_14/tensor/axisConst*
dtype0*
value	B : 
�
Reshape_14/tensorGatherV2.mio_embeddings/uid_comment_list_hetu1/variableCastReshape_14/tensor/axis*
Taxis0*
Tindices0*
Tparams0
I
Reshape_14/shapeConst*!
valueB"����2      *
dtype0
Q

Reshape_14ReshapeReshape_14/tensorReshape_14/shape*
T0*
Tshape0
�
.mio_embeddings/uid_comment_list_hetu2/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containeruid_comment_list_hetu2*
shape:����������
�
.mio_embeddings/uid_comment_list_hetu2/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:����������*%
	containeruid_comment_list_hetu2
@
Reshape_15/tensor/axisConst*
value	B : *
dtype0
�
Reshape_15/tensorGatherV2.mio_embeddings/uid_comment_list_hetu2/variableCastReshape_15/tensor/axis*
Taxis0*
Tindices0*
Tparams0
I
Reshape_15/shapeConst*
dtype0*!
valueB"����2      
Q

Reshape_15ReshapeReshape_15/tensorReshape_15/shape*
T0*
Tshape0
7
concat_5/axisConst*
value	B :*
dtype0
e
concat_5ConcatV2
Reshape_13
Reshape_14
Reshape_15concat_5/axis*

Tidx0*
T0*
N
�
+mio_embeddings/uid_collect_list_id/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*"
	containeruid_collect_list_id*
shape:����������
�
+mio_embeddings/uid_collect_list_id/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*"
	containeruid_collect_list_id*
shape:����������
@
Reshape_16/tensor/axisConst*
value	B : *
dtype0
�
Reshape_16/tensorGatherV2+mio_embeddings/uid_collect_list_id/variableCastReshape_16/tensor/axis*
Tparams0*
Taxis0*
Tindices0
I
Reshape_16/shapeConst*!
valueB"����2       *
dtype0
Q

Reshape_16ReshapeReshape_16/tensorReshape_16/shape*
T0*
Tshape0
�
.mio_embeddings/uid_collect_list_hetu1/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containeruid_collect_list_hetu1*
shape:����������
�
.mio_embeddings/uid_collect_list_hetu1/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containeruid_collect_list_hetu1*
shape:����������
@
Reshape_17/tensor/axisConst*
value	B : *
dtype0
�
Reshape_17/tensorGatherV2.mio_embeddings/uid_collect_list_hetu1/variableCastReshape_17/tensor/axis*
Taxis0*
Tindices0*
Tparams0
I
Reshape_17/shapeConst*!
valueB"����2      *
dtype0
Q

Reshape_17ReshapeReshape_17/tensorReshape_17/shape*
T0*
Tshape0
�
.mio_embeddings/uid_collect_list_hetu2/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containeruid_collect_list_hetu2*
shape:����������
�
.mio_embeddings/uid_collect_list_hetu2/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containeruid_collect_list_hetu2*
shape:����������
@
Reshape_18/tensor/axisConst*
value	B : *
dtype0
�
Reshape_18/tensorGatherV2.mio_embeddings/uid_collect_list_hetu2/variableCastReshape_18/tensor/axis*
Taxis0*
Tindices0*
Tparams0
I
Reshape_18/shapeConst*!
valueB"����2      *
dtype0
Q

Reshape_18ReshapeReshape_18/tensorReshape_18/shape*
T0*
Tshape0
7
concat_6/axisConst*
value	B :*
dtype0
e
concat_6ConcatV2
Reshape_16
Reshape_17
Reshape_18concat_6/axis*
T0*
N*

Tidx0
�
1mio_embeddings/uid_profile_enter_list_id/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeruid_profile_enter_list_id*
shape:����������
�
1mio_embeddings/uid_profile_enter_list_id/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:����������*(
	containeruid_profile_enter_list_id
@
Reshape_19/tensor/axisConst*
value	B : *
dtype0
�
Reshape_19/tensorGatherV21mio_embeddings/uid_profile_enter_list_id/variableCastReshape_19/tensor/axis*
Taxis0*
Tindices0*
Tparams0
I
Reshape_19/shapeConst*!
valueB"����2       *
dtype0
Q

Reshape_19ReshapeReshape_19/tensorReshape_19/shape*
T0*
Tshape0
�
4mio_embeddings/uid_profile_enter_list_hetu1/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*+
	containeruid_profile_enter_list_hetu1*
shape:����������
�
4mio_embeddings/uid_profile_enter_list_hetu1/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*+
	containeruid_profile_enter_list_hetu1*
shape:����������
@
Reshape_20/tensor/axisConst*
value	B : *
dtype0
�
Reshape_20/tensorGatherV24mio_embeddings/uid_profile_enter_list_hetu1/variableCastReshape_20/tensor/axis*
Tindices0*
Tparams0*
Taxis0
I
Reshape_20/shapeConst*!
valueB"����2      *
dtype0
Q

Reshape_20ReshapeReshape_20/tensorReshape_20/shape*
T0*
Tshape0
�
4mio_embeddings/uid_profile_enter_list_hetu2/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*+
	containeruid_profile_enter_list_hetu2*
shape:����������
�
4mio_embeddings/uid_profile_enter_list_hetu2/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*+
	containeruid_profile_enter_list_hetu2*
shape:����������
@
Reshape_21/tensor/axisConst*
value	B : *
dtype0
�
Reshape_21/tensorGatherV24mio_embeddings/uid_profile_enter_list_hetu2/variableCastReshape_21/tensor/axis*
Taxis0*
Tindices0*
Tparams0
I
Reshape_21/shapeConst*!
valueB"����2      *
dtype0
Q

Reshape_21ReshapeReshape_21/tensorReshape_21/shape*
T0*
Tshape0
7
concat_7/axisConst*
value	B :*
dtype0
e
concat_7ConcatV2
Reshape_19
Reshape_20
Reshape_21concat_7/axis*

Tidx0*
T0*
N
7
concat_8/axisConst*
dtype0*
value	B :
�
concat_8ConcatV2mio_embeddings/pid_emb/variable mio_embeddings/pid_hetu/variableconcat_8/axis*

Tidx0*
T0*
N
8
ExpandDims/dimConst*
value	B : *
dtype0
G

ExpandDims
ExpandDimsconcat_8ExpandDims/dim*
T0*

Tdim0
L
strided_slice/stackConst*!
valueB"            *
dtype0
N
strided_slice/stack_1Const*!
valueB"           *
dtype0
N
strided_slice/stack_2Const*
dtype0*!
valueB"         
�
strided_sliceStridedSliceconcat_2strided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
:
ExpandDims_1/dimConst*
dtype0*
value	B : 
P
ExpandDims_1
ExpandDimsstrided_sliceExpandDims_1/dim*

Tdim0*
T0
N
strided_slice_1/stackConst*!
valueB"            *
dtype0
P
strided_slice_1/stack_1Const*
dtype0*!
valueB"           
P
strided_slice_1/stack_2Const*
dtype0*!
valueB"         
�
strided_slice_1StridedSliceconcat_3strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
end_mask*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
:
ExpandDims_2/dimConst*
value	B : *
dtype0
R
ExpandDims_2
ExpandDimsstrided_slice_1ExpandDims_2/dim*

Tdim0*
T0
N
strided_slice_2/stackConst*!
valueB"            *
dtype0
P
strided_slice_2/stack_1Const*!
valueB"           *
dtype0
P
strided_slice_2/stack_2Const*
dtype0*!
valueB"         
�
strided_slice_2StridedSliceconcat_4strided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
:
ExpandDims_3/dimConst*
value	B : *
dtype0
R
ExpandDims_3
ExpandDimsstrided_slice_2ExpandDims_3/dim*
T0*

Tdim0
N
strided_slice_3/stackConst*
dtype0*!
valueB"            
P
strided_slice_3/stack_1Const*!
valueB"           *
dtype0
P
strided_slice_3/stack_2Const*!
valueB"         *
dtype0
�
strided_slice_3StridedSliceconcat_5strided_slice_3/stackstrided_slice_3/stack_1strided_slice_3/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
:
ExpandDims_4/dimConst*
value	B : *
dtype0
R
ExpandDims_4
ExpandDimsstrided_slice_3ExpandDims_4/dim*

Tdim0*
T0
N
strided_slice_4/stackConst*!
valueB"            *
dtype0
P
strided_slice_4/stack_1Const*!
valueB"           *
dtype0
P
strided_slice_4/stack_2Const*!
valueB"         *
dtype0
�
strided_slice_4StridedSliceconcat_6strided_slice_4/stackstrided_slice_4/stack_1strided_slice_4/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
:
ExpandDims_5/dimConst*
value	B : *
dtype0
R
ExpandDims_5
ExpandDimsstrided_slice_4ExpandDims_5/dim*

Tdim0*
T0
N
strided_slice_5/stackConst*
dtype0*!
valueB"            
P
strided_slice_5/stack_1Const*
dtype0*!
valueB"           
P
strided_slice_5/stack_2Const*
dtype0*!
valueB"         
�
strided_slice_5StridedSliceconcat_7strided_slice_5/stackstrided_slice_5/stack_1strided_slice_5/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
:
ExpandDims_6/dimConst*
value	B : *
dtype0
R
ExpandDims_6
ExpandDimsstrided_slice_5ExpandDims_6/dim*

Tdim0*
T0
F
like_seq_attention/ShapeShape
ExpandDims*
T0*
out_type0
T
&like_seq_attention/strided_slice/stackConst*
valueB:*
dtype0
V
(like_seq_attention/strided_slice/stack_1Const*
valueB:*
dtype0
V
(like_seq_attention/strided_slice/stack_2Const*
valueB:*
dtype0
�
 like_seq_attention/strided_sliceStridedSlicelike_seq_attention/Shape&like_seq_attention/strided_slice/stack(like_seq_attention/strided_slice/stack_1(like_seq_attention/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
S
like_seq_attention/Shape_1Const*
dtype0*!
valueB"   2   0   
V
(like_seq_attention/strided_slice_1/stackConst*
valueB:*
dtype0
X
*like_seq_attention/strided_slice_1/stack_1Const*
valueB:*
dtype0
X
*like_seq_attention/strided_slice_1/stack_2Const*
valueB:*
dtype0
�
"like_seq_attention/strided_slice_1StridedSlicelike_seq_attention/Shape_1(like_seq_attention/strided_slice_1/stack*like_seq_attention/strided_slice_1/stack_1*like_seq_attention/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
�
5mio_variable/like_seq_attention/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!like_seq_attention/dense/kernel*
shape
:0 
�
5mio_variable/like_seq_attention/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!like_seq_attention/dense/kernel*
shape
:0 
U
 Initializer/random_uniform/shapeConst*
valueB"0       *
dtype0
K
Initializer/random_uniform/minConst*
dtype0*
valueB
 *�7��
K
Initializer/random_uniform/maxConst*
valueB
 *�7�>*
dtype0
�
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
�
AssignAssign5mio_variable/like_seq_attention/dense/kernel/gradientInitializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/like_seq_attention/dense/kernel/gradient*
validate_shape(
�
3mio_variable/like_seq_attention/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerlike_seq_attention/dense/bias*
shape: 
�
3mio_variable/like_seq_attention/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape: *,
	containerlike_seq_attention/dense/bias
D
Initializer_1/zerosConst*
valueB *    *
dtype0
�
Assign_1Assign3mio_variable/like_seq_attention/dense/bias/gradientInitializer_1/zeros*
use_locking(*
T0*F
_class<
:8loc:@mio_variable/like_seq_attention/dense/bias/gradient*
validate_shape(
U
'like_seq_attention/dense/Tensordot/axesConst*
valueB:*
dtype0
\
'like_seq_attention/dense/Tensordot/freeConst*
valueB"       *
dtype0
V
(like_seq_attention/dense/Tensordot/ShapeShape
ExpandDims*
T0*
out_type0
Z
0like_seq_attention/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0
�
+like_seq_attention/dense/Tensordot/GatherV2GatherV2(like_seq_attention/dense/Tensordot/Shape'like_seq_attention/dense/Tensordot/free0like_seq_attention/dense/Tensordot/GatherV2/axis*
Tindices0*
Tparams0*
Taxis0
\
2like_seq_attention/dense/Tensordot/GatherV2_1/axisConst*
dtype0*
value	B : 
�
-like_seq_attention/dense/Tensordot/GatherV2_1GatherV2(like_seq_attention/dense/Tensordot/Shape'like_seq_attention/dense/Tensordot/axes2like_seq_attention/dense/Tensordot/GatherV2_1/axis*
Tindices0*
Tparams0*
Taxis0
V
(like_seq_attention/dense/Tensordot/ConstConst*
valueB: *
dtype0
�
'like_seq_attention/dense/Tensordot/ProdProd+like_seq_attention/dense/Tensordot/GatherV2(like_seq_attention/dense/Tensordot/Const*
T0*

Tidx0*
	keep_dims( 
X
*like_seq_attention/dense/Tensordot/Const_1Const*
valueB: *
dtype0
�
)like_seq_attention/dense/Tensordot/Prod_1Prod-like_seq_attention/dense/Tensordot/GatherV2_1*like_seq_attention/dense/Tensordot/Const_1*
T0*

Tidx0*
	keep_dims( 
X
.like_seq_attention/dense/Tensordot/concat/axisConst*
dtype0*
value	B : 
�
)like_seq_attention/dense/Tensordot/concatConcatV2'like_seq_attention/dense/Tensordot/free'like_seq_attention/dense/Tensordot/axes.like_seq_attention/dense/Tensordot/concat/axis*
N*

Tidx0*
T0
�
(like_seq_attention/dense/Tensordot/stackPack'like_seq_attention/dense/Tensordot/Prod)like_seq_attention/dense/Tensordot/Prod_1*
N*
T0*

axis 
�
,like_seq_attention/dense/Tensordot/transpose	Transpose
ExpandDims)like_seq_attention/dense/Tensordot/concat*
T0*
Tperm0
�
*like_seq_attention/dense/Tensordot/ReshapeReshape,like_seq_attention/dense/Tensordot/transpose(like_seq_attention/dense/Tensordot/stack*
T0*
Tshape0
h
3like_seq_attention/dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
.like_seq_attention/dense/Tensordot/transpose_1	Transpose5mio_variable/like_seq_attention/dense/kernel/variable3like_seq_attention/dense/Tensordot/transpose_1/perm*
Tperm0*
T0
g
2like_seq_attention/dense/Tensordot/Reshape_1/shapeConst*
valueB"0       *
dtype0
�
,like_seq_attention/dense/Tensordot/Reshape_1Reshape.like_seq_attention/dense/Tensordot/transpose_12like_seq_attention/dense/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
)like_seq_attention/dense/Tensordot/MatMulMatMul*like_seq_attention/dense/Tensordot/Reshape,like_seq_attention/dense/Tensordot/Reshape_1*
transpose_a( *
transpose_b( *
T0
X
*like_seq_attention/dense/Tensordot/Const_2Const*
valueB: *
dtype0
Z
0like_seq_attention/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0
�
+like_seq_attention/dense/Tensordot/concat_1ConcatV2+like_seq_attention/dense/Tensordot/GatherV2*like_seq_attention/dense/Tensordot/Const_20like_seq_attention/dense/Tensordot/concat_1/axis*
N*

Tidx0*
T0
�
"like_seq_attention/dense/TensordotReshape)like_seq_attention/dense/Tensordot/MatMul+like_seq_attention/dense/Tensordot/concat_1*
T0*
Tshape0
�
 like_seq_attention/dense/BiasAddBiasAdd"like_seq_attention/dense/Tensordot3mio_variable/like_seq_attention/dense/bias/variable*
T0*
data_formatNHWC
�
7mio_variable/like_seq_attention/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!like_seq_attention/dense_1/kernel*
shape
:0 
�
7mio_variable/like_seq_attention/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!like_seq_attention/dense_1/kernel*
shape
:0 
W
"Initializer_2/random_uniform/shapeConst*
valueB"0       *
dtype0
M
 Initializer_2/random_uniform/minConst*
valueB
 *�7��*
dtype0
M
 Initializer_2/random_uniform/maxConst*
valueB
 *�7�>*
dtype0
�
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
�
Assign_2Assign7mio_variable/like_seq_attention/dense_1/kernel/gradientInitializer_2/random_uniform*
T0*J
_class@
><loc:@mio_variable/like_seq_attention/dense_1/kernel/gradient*
validate_shape(*
use_locking(
�
5mio_variable/like_seq_attention/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!like_seq_attention/dense_1/bias*
shape: 
�
5mio_variable/like_seq_attention/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape: *.
	container!like_seq_attention/dense_1/bias
D
Initializer_3/zerosConst*
valueB *    *
dtype0
�
Assign_3Assign5mio_variable/like_seq_attention/dense_1/bias/gradientInitializer_3/zeros*
T0*H
_class>
<:loc:@mio_variable/like_seq_attention/dense_1/bias/gradient*
validate_shape(*
use_locking(
l
3like_seq_attention/dense_1/Tensordot/transpose/permConst*!
valueB"          *
dtype0
�
.like_seq_attention/dense_1/Tensordot/transpose	TransposeExpandDims_13like_seq_attention/dense_1/Tensordot/transpose/perm*
Tperm0*
T0
g
2like_seq_attention/dense_1/Tensordot/Reshape/shapeConst*
valueB"2   0   *
dtype0
�
,like_seq_attention/dense_1/Tensordot/ReshapeReshape.like_seq_attention/dense_1/Tensordot/transpose2like_seq_attention/dense_1/Tensordot/Reshape/shape*
T0*
Tshape0
j
5like_seq_attention/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
0like_seq_attention/dense_1/Tensordot/transpose_1	Transpose7mio_variable/like_seq_attention/dense_1/kernel/variable5like_seq_attention/dense_1/Tensordot/transpose_1/perm*
Tperm0*
T0
i
4like_seq_attention/dense_1/Tensordot/Reshape_1/shapeConst*
valueB"0       *
dtype0
�
.like_seq_attention/dense_1/Tensordot/Reshape_1Reshape0like_seq_attention/dense_1/Tensordot/transpose_14like_seq_attention/dense_1/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
+like_seq_attention/dense_1/Tensordot/MatMulMatMul,like_seq_attention/dense_1/Tensordot/Reshape.like_seq_attention/dense_1/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
c
*like_seq_attention/dense_1/Tensordot/shapeConst*!
valueB"   2       *
dtype0
�
$like_seq_attention/dense_1/TensordotReshape+like_seq_attention/dense_1/Tensordot/MatMul*like_seq_attention/dense_1/Tensordot/shape*
T0*
Tshape0
�
"like_seq_attention/dense_1/BiasAddBiasAdd$like_seq_attention/dense_1/Tensordot5mio_variable/like_seq_attention/dense_1/bias/variable*
T0*
data_formatNHWC
�
7mio_variable/like_seq_attention/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!like_seq_attention/dense_2/kernel*
shape
:0 
�
7mio_variable/like_seq_attention/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:0 *0
	container#!like_seq_attention/dense_2/kernel
W
"Initializer_4/random_uniform/shapeConst*
valueB"0       *
dtype0
M
 Initializer_4/random_uniform/minConst*
dtype0*
valueB
 *�7��
M
 Initializer_4/random_uniform/maxConst*
valueB
 *�7�>*
dtype0
�
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
�
Assign_4Assign7mio_variable/like_seq_attention/dense_2/kernel/gradientInitializer_4/random_uniform*
use_locking(*
T0*J
_class@
><loc:@mio_variable/like_seq_attention/dense_2/kernel/gradient*
validate_shape(
�
5mio_variable/like_seq_attention/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!like_seq_attention/dense_2/bias*
shape: 
�
5mio_variable/like_seq_attention/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape: *.
	container!like_seq_attention/dense_2/bias
D
Initializer_5/zerosConst*
valueB *    *
dtype0
�
Assign_5Assign5mio_variable/like_seq_attention/dense_2/bias/gradientInitializer_5/zeros*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/like_seq_attention/dense_2/bias/gradient*
validate_shape(
l
3like_seq_attention/dense_2/Tensordot/transpose/permConst*!
valueB"          *
dtype0
�
.like_seq_attention/dense_2/Tensordot/transpose	TransposeExpandDims_13like_seq_attention/dense_2/Tensordot/transpose/perm*
T0*
Tperm0
g
2like_seq_attention/dense_2/Tensordot/Reshape/shapeConst*
dtype0*
valueB"2   0   
�
,like_seq_attention/dense_2/Tensordot/ReshapeReshape.like_seq_attention/dense_2/Tensordot/transpose2like_seq_attention/dense_2/Tensordot/Reshape/shape*
T0*
Tshape0
j
5like_seq_attention/dense_2/Tensordot/transpose_1/permConst*
dtype0*
valueB"       
�
0like_seq_attention/dense_2/Tensordot/transpose_1	Transpose7mio_variable/like_seq_attention/dense_2/kernel/variable5like_seq_attention/dense_2/Tensordot/transpose_1/perm*
T0*
Tperm0
i
4like_seq_attention/dense_2/Tensordot/Reshape_1/shapeConst*
dtype0*
valueB"0       
�
.like_seq_attention/dense_2/Tensordot/Reshape_1Reshape0like_seq_attention/dense_2/Tensordot/transpose_14like_seq_attention/dense_2/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
+like_seq_attention/dense_2/Tensordot/MatMulMatMul,like_seq_attention/dense_2/Tensordot/Reshape.like_seq_attention/dense_2/Tensordot/Reshape_1*
transpose_b( *
T0*
transpose_a( 
c
*like_seq_attention/dense_2/Tensordot/shapeConst*!
valueB"   2       *
dtype0
�
$like_seq_attention/dense_2/TensordotReshape+like_seq_attention/dense_2/Tensordot/MatMul*like_seq_attention/dense_2/Tensordot/shape*
T0*
Tshape0
�
"like_seq_attention/dense_2/BiasAddBiasAdd$like_seq_attention/dense_2/Tensordot5mio_variable/like_seq_attention/dense_2/bias/variable*
T0*
data_formatNHWC
U
"like_seq_attention/Reshape/shape/0Const*
valueB :
���������*
dtype0
L
"like_seq_attention/Reshape/shape/2Const*
dtype0*
value	B :
L
"like_seq_attention/Reshape/shape/3Const*
dtype0*
value	B :
�
 like_seq_attention/Reshape/shapePack"like_seq_attention/Reshape/shape/0 like_seq_attention/strided_slice"like_seq_attention/Reshape/shape/2"like_seq_attention/Reshape/shape/3*
T0*

axis *
N
�
like_seq_attention/ReshapeReshape like_seq_attention/dense/BiasAdd like_seq_attention/Reshape/shape*
T0*
Tshape0
^
!like_seq_attention/transpose/permConst*%
valueB"             *
dtype0
~
like_seq_attention/transpose	Transposelike_seq_attention/Reshape!like_seq_attention/transpose/perm*
T0*
Tperm0
W
$like_seq_attention/Reshape_1/shape/0Const*
valueB :
���������*
dtype0
N
$like_seq_attention/Reshape_1/shape/2Const*
value	B :*
dtype0
N
$like_seq_attention/Reshape_1/shape/3Const*
value	B :*
dtype0
�
"like_seq_attention/Reshape_1/shapePack$like_seq_attention/Reshape_1/shape/0"like_seq_attention/strided_slice_1$like_seq_attention/Reshape_1/shape/2$like_seq_attention/Reshape_1/shape/3*
T0*

axis *
N
�
like_seq_attention/Reshape_1Reshape"like_seq_attention/dense_1/BiasAdd"like_seq_attention/Reshape_1/shape*
T0*
Tshape0
`
#like_seq_attention/transpose_1/permConst*%
valueB"             *
dtype0
�
like_seq_attention/transpose_1	Transposelike_seq_attention/Reshape_1#like_seq_attention/transpose_1/perm*
Tperm0*
T0
W
$like_seq_attention/Reshape_2/shape/0Const*
valueB :
���������*
dtype0
N
$like_seq_attention/Reshape_2/shape/2Const*
value	B :*
dtype0
N
$like_seq_attention/Reshape_2/shape/3Const*
value	B :*
dtype0
�
"like_seq_attention/Reshape_2/shapePack$like_seq_attention/Reshape_2/shape/0"like_seq_attention/strided_slice_1$like_seq_attention/Reshape_2/shape/2$like_seq_attention/Reshape_2/shape/3*
T0*

axis *
N
�
like_seq_attention/Reshape_2Reshape"like_seq_attention/dense_2/BiasAdd"like_seq_attention/Reshape_2/shape*
T0*
Tshape0
`
#like_seq_attention/transpose_2/permConst*%
valueB"             *
dtype0
�
like_seq_attention/transpose_2	Transposelike_seq_attention/Reshape_2#like_seq_attention/transpose_2/perm*
T0*
Tperm0
�
like_seq_attention/MatMulBatchMatMullike_seq_attention/transposelike_seq_attention/transpose_1*
adj_x( *
adj_y(*
T0
C
like_seq_attention/Cast/xConst*
value	B :*
dtype0
b
like_seq_attention/CastCastlike_seq_attention/Cast/x*

SrcT0*
Truncate( *

DstT0
A
like_seq_attention/SqrtSqrtlike_seq_attention/Cast*
T0
b
like_seq_attention/truedivRealDivlike_seq_attention/MatMullike_seq_attention/Sqrt*
T0
J
like_seq_attention/SoftmaxSoftmaxlike_seq_attention/truediv*
T0
�
like_seq_attention/MatMul_1BatchMatMullike_seq_attention/Softmaxlike_seq_attention/transpose_2*
adj_x( *
adj_y( *
T0
`
#like_seq_attention/transpose_3/permConst*%
valueB"             *
dtype0
�
like_seq_attention/transpose_3	Transposelike_seq_attention/MatMul_1#like_seq_attention/transpose_3/perm*
Tperm0*
T0
W
$like_seq_attention/Reshape_3/shape/0Const*
dtype0*
valueB :
���������
N
$like_seq_attention/Reshape_3/shape/2Const*
dtype0*
value	B : 
�
"like_seq_attention/Reshape_3/shapePack$like_seq_attention/Reshape_3/shape/0 like_seq_attention/strided_slice$like_seq_attention/Reshape_3/shape/2*
T0*

axis *
N
�
like_seq_attention/Reshape_3Reshapelike_seq_attention/transpose_3"like_seq_attention/Reshape_3/shape*
T0*
Tshape0
H
follow_seq_attention/ShapeShape
ExpandDims*
T0*
out_type0
V
(follow_seq_attention/strided_slice/stackConst*
valueB:*
dtype0
X
*follow_seq_attention/strided_slice/stack_1Const*
valueB:*
dtype0
X
*follow_seq_attention/strided_slice/stack_2Const*
dtype0*
valueB:
�
"follow_seq_attention/strided_sliceStridedSlicefollow_seq_attention/Shape(follow_seq_attention/strided_slice/stack*follow_seq_attention/strided_slice/stack_1*follow_seq_attention/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0
U
follow_seq_attention/Shape_1Const*!
valueB"   2   0   *
dtype0
X
*follow_seq_attention/strided_slice_1/stackConst*
dtype0*
valueB:
Z
,follow_seq_attention/strided_slice_1/stack_1Const*
valueB:*
dtype0
Z
,follow_seq_attention/strided_slice_1/stack_2Const*
dtype0*
valueB:
�
$follow_seq_attention/strided_slice_1StridedSlicefollow_seq_attention/Shape_1*follow_seq_attention/strided_slice_1/stack,follow_seq_attention/strided_slice_1/stack_1,follow_seq_attention/strided_slice_1/stack_2*
end_mask *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
�
7mio_variable/follow_seq_attention/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!follow_seq_attention/dense/kernel*
shape
:0 
�
7mio_variable/follow_seq_attention/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:0 *0
	container#!follow_seq_attention/dense/kernel
W
"Initializer_6/random_uniform/shapeConst*
dtype0*
valueB"0       
M
 Initializer_6/random_uniform/minConst*
valueB
 *�7��*
dtype0
M
 Initializer_6/random_uniform/maxConst*
dtype0*
valueB
 *�7�>
�
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
�
Assign_6Assign7mio_variable/follow_seq_attention/dense/kernel/gradientInitializer_6/random_uniform*
use_locking(*
T0*J
_class@
><loc:@mio_variable/follow_seq_attention/dense/kernel/gradient*
validate_shape(
�
5mio_variable/follow_seq_attention/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape: *.
	container!follow_seq_attention/dense/bias
�
5mio_variable/follow_seq_attention/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!follow_seq_attention/dense/bias*
shape: 
D
Initializer_7/zerosConst*
valueB *    *
dtype0
�
Assign_7Assign5mio_variable/follow_seq_attention/dense/bias/gradientInitializer_7/zeros*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/follow_seq_attention/dense/bias/gradient*
validate_shape(
W
)follow_seq_attention/dense/Tensordot/axesConst*
dtype0*
valueB:
^
)follow_seq_attention/dense/Tensordot/freeConst*
valueB"       *
dtype0
X
*follow_seq_attention/dense/Tensordot/ShapeShape
ExpandDims*
T0*
out_type0
\
2follow_seq_attention/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0
�
-follow_seq_attention/dense/Tensordot/GatherV2GatherV2*follow_seq_attention/dense/Tensordot/Shape)follow_seq_attention/dense/Tensordot/free2follow_seq_attention/dense/Tensordot/GatherV2/axis*
Tparams0*
Taxis0*
Tindices0
^
4follow_seq_attention/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0
�
/follow_seq_attention/dense/Tensordot/GatherV2_1GatherV2*follow_seq_attention/dense/Tensordot/Shape)follow_seq_attention/dense/Tensordot/axes4follow_seq_attention/dense/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0
X
*follow_seq_attention/dense/Tensordot/ConstConst*
valueB: *
dtype0
�
)follow_seq_attention/dense/Tensordot/ProdProd-follow_seq_attention/dense/Tensordot/GatherV2*follow_seq_attention/dense/Tensordot/Const*
T0*

Tidx0*
	keep_dims( 
Z
,follow_seq_attention/dense/Tensordot/Const_1Const*
valueB: *
dtype0
�
+follow_seq_attention/dense/Tensordot/Prod_1Prod/follow_seq_attention/dense/Tensordot/GatherV2_1,follow_seq_attention/dense/Tensordot/Const_1*
T0*

Tidx0*
	keep_dims( 
Z
0follow_seq_attention/dense/Tensordot/concat/axisConst*
value	B : *
dtype0
�
+follow_seq_attention/dense/Tensordot/concatConcatV2)follow_seq_attention/dense/Tensordot/free)follow_seq_attention/dense/Tensordot/axes0follow_seq_attention/dense/Tensordot/concat/axis*

Tidx0*
T0*
N
�
*follow_seq_attention/dense/Tensordot/stackPack)follow_seq_attention/dense/Tensordot/Prod+follow_seq_attention/dense/Tensordot/Prod_1*
T0*

axis *
N
�
.follow_seq_attention/dense/Tensordot/transpose	Transpose
ExpandDims+follow_seq_attention/dense/Tensordot/concat*
T0*
Tperm0
�
,follow_seq_attention/dense/Tensordot/ReshapeReshape.follow_seq_attention/dense/Tensordot/transpose*follow_seq_attention/dense/Tensordot/stack*
T0*
Tshape0
j
5follow_seq_attention/dense/Tensordot/transpose_1/permConst*
dtype0*
valueB"       
�
0follow_seq_attention/dense/Tensordot/transpose_1	Transpose7mio_variable/follow_seq_attention/dense/kernel/variable5follow_seq_attention/dense/Tensordot/transpose_1/perm*
T0*
Tperm0
i
4follow_seq_attention/dense/Tensordot/Reshape_1/shapeConst*
dtype0*
valueB"0       
�
.follow_seq_attention/dense/Tensordot/Reshape_1Reshape0follow_seq_attention/dense/Tensordot/transpose_14follow_seq_attention/dense/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
+follow_seq_attention/dense/Tensordot/MatMulMatMul,follow_seq_attention/dense/Tensordot/Reshape.follow_seq_attention/dense/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
Z
,follow_seq_attention/dense/Tensordot/Const_2Const*
valueB: *
dtype0
\
2follow_seq_attention/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0
�
-follow_seq_attention/dense/Tensordot/concat_1ConcatV2-follow_seq_attention/dense/Tensordot/GatherV2,follow_seq_attention/dense/Tensordot/Const_22follow_seq_attention/dense/Tensordot/concat_1/axis*
T0*
N*

Tidx0
�
$follow_seq_attention/dense/TensordotReshape+follow_seq_attention/dense/Tensordot/MatMul-follow_seq_attention/dense/Tensordot/concat_1*
T0*
Tshape0
�
"follow_seq_attention/dense/BiasAddBiasAdd$follow_seq_attention/dense/Tensordot5mio_variable/follow_seq_attention/dense/bias/variable*
T0*
data_formatNHWC
�
9mio_variable/follow_seq_attention/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#follow_seq_attention/dense_1/kernel*
shape
:0 
�
9mio_variable/follow_seq_attention/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#follow_seq_attention/dense_1/kernel*
shape
:0 
W
"Initializer_8/random_uniform/shapeConst*
valueB"0       *
dtype0
M
 Initializer_8/random_uniform/minConst*
valueB
 *�7��*
dtype0
M
 Initializer_8/random_uniform/maxConst*
valueB
 *�7�>*
dtype0
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
Assign_8Assign9mio_variable/follow_seq_attention/dense_1/kernel/gradientInitializer_8/random_uniform*
use_locking(*
T0*L
_classB
@>loc:@mio_variable/follow_seq_attention/dense_1/kernel/gradient*
validate_shape(
�
7mio_variable/follow_seq_attention/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!follow_seq_attention/dense_1/bias*
shape: 
�
7mio_variable/follow_seq_attention/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!follow_seq_attention/dense_1/bias*
shape: 
D
Initializer_9/zerosConst*
dtype0*
valueB *    
�
Assign_9Assign7mio_variable/follow_seq_attention/dense_1/bias/gradientInitializer_9/zeros*
use_locking(*
T0*J
_class@
><loc:@mio_variable/follow_seq_attention/dense_1/bias/gradient*
validate_shape(
n
5follow_seq_attention/dense_1/Tensordot/transpose/permConst*!
valueB"          *
dtype0
�
0follow_seq_attention/dense_1/Tensordot/transpose	TransposeExpandDims_25follow_seq_attention/dense_1/Tensordot/transpose/perm*
T0*
Tperm0
i
4follow_seq_attention/dense_1/Tensordot/Reshape/shapeConst*
valueB"2   0   *
dtype0
�
.follow_seq_attention/dense_1/Tensordot/ReshapeReshape0follow_seq_attention/dense_1/Tensordot/transpose4follow_seq_attention/dense_1/Tensordot/Reshape/shape*
T0*
Tshape0
l
7follow_seq_attention/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
2follow_seq_attention/dense_1/Tensordot/transpose_1	Transpose9mio_variable/follow_seq_attention/dense_1/kernel/variable7follow_seq_attention/dense_1/Tensordot/transpose_1/perm*
Tperm0*
T0
k
6follow_seq_attention/dense_1/Tensordot/Reshape_1/shapeConst*
valueB"0       *
dtype0
�
0follow_seq_attention/dense_1/Tensordot/Reshape_1Reshape2follow_seq_attention/dense_1/Tensordot/transpose_16follow_seq_attention/dense_1/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
-follow_seq_attention/dense_1/Tensordot/MatMulMatMul.follow_seq_attention/dense_1/Tensordot/Reshape0follow_seq_attention/dense_1/Tensordot/Reshape_1*
transpose_b( *
T0*
transpose_a( 
e
,follow_seq_attention/dense_1/Tensordot/shapeConst*!
valueB"   2       *
dtype0
�
&follow_seq_attention/dense_1/TensordotReshape-follow_seq_attention/dense_1/Tensordot/MatMul,follow_seq_attention/dense_1/Tensordot/shape*
T0*
Tshape0
�
$follow_seq_attention/dense_1/BiasAddBiasAdd&follow_seq_attention/dense_1/Tensordot7mio_variable/follow_seq_attention/dense_1/bias/variable*
T0*
data_formatNHWC
�
9mio_variable/follow_seq_attention/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#follow_seq_attention/dense_2/kernel*
shape
:0 
�
9mio_variable/follow_seq_attention/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:0 *2
	container%#follow_seq_attention/dense_2/kernel
X
#Initializer_10/random_uniform/shapeConst*
dtype0*
valueB"0       
N
!Initializer_10/random_uniform/minConst*
valueB
 *�7��*
dtype0
N
!Initializer_10/random_uniform/maxConst*
valueB
 *�7�>*
dtype0
�
+Initializer_10/random_uniform/RandomUniformRandomUniform#Initializer_10/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_10/random_uniform/subSub!Initializer_10/random_uniform/max!Initializer_10/random_uniform/min*
T0
�
!Initializer_10/random_uniform/mulMul+Initializer_10/random_uniform/RandomUniform!Initializer_10/random_uniform/sub*
T0
s
Initializer_10/random_uniformAdd!Initializer_10/random_uniform/mul!Initializer_10/random_uniform/min*
T0
�
	Assign_10Assign9mio_variable/follow_seq_attention/dense_2/kernel/gradientInitializer_10/random_uniform*
T0*L
_classB
@>loc:@mio_variable/follow_seq_attention/dense_2/kernel/gradient*
validate_shape(*
use_locking(
�
7mio_variable/follow_seq_attention/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape: *0
	container#!follow_seq_attention/dense_2/bias
�
7mio_variable/follow_seq_attention/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape: *0
	container#!follow_seq_attention/dense_2/bias
E
Initializer_11/zerosConst*
dtype0*
valueB *    
�
	Assign_11Assign7mio_variable/follow_seq_attention/dense_2/bias/gradientInitializer_11/zeros*
use_locking(*
T0*J
_class@
><loc:@mio_variable/follow_seq_attention/dense_2/bias/gradient*
validate_shape(
n
5follow_seq_attention/dense_2/Tensordot/transpose/permConst*!
valueB"          *
dtype0
�
0follow_seq_attention/dense_2/Tensordot/transpose	TransposeExpandDims_25follow_seq_attention/dense_2/Tensordot/transpose/perm*
Tperm0*
T0
i
4follow_seq_attention/dense_2/Tensordot/Reshape/shapeConst*
dtype0*
valueB"2   0   
�
.follow_seq_attention/dense_2/Tensordot/ReshapeReshape0follow_seq_attention/dense_2/Tensordot/transpose4follow_seq_attention/dense_2/Tensordot/Reshape/shape*
T0*
Tshape0
l
7follow_seq_attention/dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
2follow_seq_attention/dense_2/Tensordot/transpose_1	Transpose9mio_variable/follow_seq_attention/dense_2/kernel/variable7follow_seq_attention/dense_2/Tensordot/transpose_1/perm*
T0*
Tperm0
k
6follow_seq_attention/dense_2/Tensordot/Reshape_1/shapeConst*
valueB"0       *
dtype0
�
0follow_seq_attention/dense_2/Tensordot/Reshape_1Reshape2follow_seq_attention/dense_2/Tensordot/transpose_16follow_seq_attention/dense_2/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
-follow_seq_attention/dense_2/Tensordot/MatMulMatMul.follow_seq_attention/dense_2/Tensordot/Reshape0follow_seq_attention/dense_2/Tensordot/Reshape_1*
transpose_a( *
transpose_b( *
T0
e
,follow_seq_attention/dense_2/Tensordot/shapeConst*!
valueB"   2       *
dtype0
�
&follow_seq_attention/dense_2/TensordotReshape-follow_seq_attention/dense_2/Tensordot/MatMul,follow_seq_attention/dense_2/Tensordot/shape*
T0*
Tshape0
�
$follow_seq_attention/dense_2/BiasAddBiasAdd&follow_seq_attention/dense_2/Tensordot7mio_variable/follow_seq_attention/dense_2/bias/variable*
data_formatNHWC*
T0
W
$follow_seq_attention/Reshape/shape/0Const*
valueB :
���������*
dtype0
N
$follow_seq_attention/Reshape/shape/2Const*
dtype0*
value	B :
N
$follow_seq_attention/Reshape/shape/3Const*
value	B :*
dtype0
�
"follow_seq_attention/Reshape/shapePack$follow_seq_attention/Reshape/shape/0"follow_seq_attention/strided_slice$follow_seq_attention/Reshape/shape/2$follow_seq_attention/Reshape/shape/3*
T0*

axis *
N
�
follow_seq_attention/ReshapeReshape"follow_seq_attention/dense/BiasAdd"follow_seq_attention/Reshape/shape*
T0*
Tshape0
`
#follow_seq_attention/transpose/permConst*%
valueB"             *
dtype0
�
follow_seq_attention/transpose	Transposefollow_seq_attention/Reshape#follow_seq_attention/transpose/perm*
Tperm0*
T0
Y
&follow_seq_attention/Reshape_1/shape/0Const*
valueB :
���������*
dtype0
P
&follow_seq_attention/Reshape_1/shape/2Const*
dtype0*
value	B :
P
&follow_seq_attention/Reshape_1/shape/3Const*
value	B :*
dtype0
�
$follow_seq_attention/Reshape_1/shapePack&follow_seq_attention/Reshape_1/shape/0$follow_seq_attention/strided_slice_1&follow_seq_attention/Reshape_1/shape/2&follow_seq_attention/Reshape_1/shape/3*
T0*

axis *
N
�
follow_seq_attention/Reshape_1Reshape$follow_seq_attention/dense_1/BiasAdd$follow_seq_attention/Reshape_1/shape*
T0*
Tshape0
b
%follow_seq_attention/transpose_1/permConst*%
valueB"             *
dtype0
�
 follow_seq_attention/transpose_1	Transposefollow_seq_attention/Reshape_1%follow_seq_attention/transpose_1/perm*
T0*
Tperm0
Y
&follow_seq_attention/Reshape_2/shape/0Const*
dtype0*
valueB :
���������
P
&follow_seq_attention/Reshape_2/shape/2Const*
value	B :*
dtype0
P
&follow_seq_attention/Reshape_2/shape/3Const*
dtype0*
value	B :
�
$follow_seq_attention/Reshape_2/shapePack&follow_seq_attention/Reshape_2/shape/0$follow_seq_attention/strided_slice_1&follow_seq_attention/Reshape_2/shape/2&follow_seq_attention/Reshape_2/shape/3*
T0*

axis *
N
�
follow_seq_attention/Reshape_2Reshape$follow_seq_attention/dense_2/BiasAdd$follow_seq_attention/Reshape_2/shape*
T0*
Tshape0
b
%follow_seq_attention/transpose_2/permConst*
dtype0*%
valueB"             
�
 follow_seq_attention/transpose_2	Transposefollow_seq_attention/Reshape_2%follow_seq_attention/transpose_2/perm*
T0*
Tperm0
�
follow_seq_attention/MatMulBatchMatMulfollow_seq_attention/transpose follow_seq_attention/transpose_1*
T0*
adj_x( *
adj_y(
E
follow_seq_attention/Cast/xConst*
dtype0*
value	B :
f
follow_seq_attention/CastCastfollow_seq_attention/Cast/x*

SrcT0*
Truncate( *

DstT0
E
follow_seq_attention/SqrtSqrtfollow_seq_attention/Cast*
T0
h
follow_seq_attention/truedivRealDivfollow_seq_attention/MatMulfollow_seq_attention/Sqrt*
T0
N
follow_seq_attention/SoftmaxSoftmaxfollow_seq_attention/truediv*
T0
�
follow_seq_attention/MatMul_1BatchMatMulfollow_seq_attention/Softmax follow_seq_attention/transpose_2*
T0*
adj_x( *
adj_y( 
b
%follow_seq_attention/transpose_3/permConst*%
valueB"             *
dtype0
�
 follow_seq_attention/transpose_3	Transposefollow_seq_attention/MatMul_1%follow_seq_attention/transpose_3/perm*
T0*
Tperm0
Y
&follow_seq_attention/Reshape_3/shape/0Const*
valueB :
���������*
dtype0
P
&follow_seq_attention/Reshape_3/shape/2Const*
value	B : *
dtype0
�
$follow_seq_attention/Reshape_3/shapePack&follow_seq_attention/Reshape_3/shape/0"follow_seq_attention/strided_slice&follow_seq_attention/Reshape_3/shape/2*
T0*

axis *
N
�
follow_seq_attention/Reshape_3Reshape follow_seq_attention/transpose_3$follow_seq_attention/Reshape_3/shape*
T0*
Tshape0
I
forward_seq_attention/ShapeShape
ExpandDims*
T0*
out_type0
W
)forward_seq_attention/strided_slice/stackConst*
valueB:*
dtype0
Y
+forward_seq_attention/strided_slice/stack_1Const*
valueB:*
dtype0
Y
+forward_seq_attention/strided_slice/stack_2Const*
valueB:*
dtype0
�
#forward_seq_attention/strided_sliceStridedSliceforward_seq_attention/Shape)forward_seq_attention/strided_slice/stack+forward_seq_attention/strided_slice/stack_1+forward_seq_attention/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask
V
forward_seq_attention/Shape_1Const*!
valueB"   2   0   *
dtype0
Y
+forward_seq_attention/strided_slice_1/stackConst*
valueB:*
dtype0
[
-forward_seq_attention/strided_slice_1/stack_1Const*
dtype0*
valueB:
[
-forward_seq_attention/strided_slice_1/stack_2Const*
dtype0*
valueB:
�
%forward_seq_attention/strided_slice_1StridedSliceforward_seq_attention/Shape_1+forward_seq_attention/strided_slice_1/stack-forward_seq_attention/strided_slice_1/stack_1-forward_seq_attention/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
�
8mio_variable/forward_seq_attention/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:0 *1
	container$"forward_seq_attention/dense/kernel
�
8mio_variable/forward_seq_attention/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"forward_seq_attention/dense/kernel*
shape
:0 
X
#Initializer_12/random_uniform/shapeConst*
valueB"0       *
dtype0
N
!Initializer_12/random_uniform/minConst*
valueB
 *�7��*
dtype0
N
!Initializer_12/random_uniform/maxConst*
dtype0*
valueB
 *�7�>
�
+Initializer_12/random_uniform/RandomUniformRandomUniform#Initializer_12/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_12/random_uniform/subSub!Initializer_12/random_uniform/max!Initializer_12/random_uniform/min*
T0
�
!Initializer_12/random_uniform/mulMul+Initializer_12/random_uniform/RandomUniform!Initializer_12/random_uniform/sub*
T0
s
Initializer_12/random_uniformAdd!Initializer_12/random_uniform/mul!Initializer_12/random_uniform/min*
T0
�
	Assign_12Assign8mio_variable/forward_seq_attention/dense/kernel/gradientInitializer_12/random_uniform*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/forward_seq_attention/dense/kernel/gradient*
validate_shape(
�
6mio_variable/forward_seq_attention/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape: */
	container" forward_seq_attention/dense/bias
�
6mio_variable/forward_seq_attention/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape: */
	container" forward_seq_attention/dense/bias
E
Initializer_13/zerosConst*
valueB *    *
dtype0
�
	Assign_13Assign6mio_variable/forward_seq_attention/dense/bias/gradientInitializer_13/zeros*
use_locking(*
T0*I
_class?
=;loc:@mio_variable/forward_seq_attention/dense/bias/gradient*
validate_shape(
X
*forward_seq_attention/dense/Tensordot/axesConst*
valueB:*
dtype0
_
*forward_seq_attention/dense/Tensordot/freeConst*
valueB"       *
dtype0
Y
+forward_seq_attention/dense/Tensordot/ShapeShape
ExpandDims*
T0*
out_type0
]
3forward_seq_attention/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0
�
.forward_seq_attention/dense/Tensordot/GatherV2GatherV2+forward_seq_attention/dense/Tensordot/Shape*forward_seq_attention/dense/Tensordot/free3forward_seq_attention/dense/Tensordot/GatherV2/axis*
Tparams0*
Taxis0*
Tindices0
_
5forward_seq_attention/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0
�
0forward_seq_attention/dense/Tensordot/GatherV2_1GatherV2+forward_seq_attention/dense/Tensordot/Shape*forward_seq_attention/dense/Tensordot/axes5forward_seq_attention/dense/Tensordot/GatherV2_1/axis*
Tparams0*
Taxis0*
Tindices0
Y
+forward_seq_attention/dense/Tensordot/ConstConst*
dtype0*
valueB: 
�
*forward_seq_attention/dense/Tensordot/ProdProd.forward_seq_attention/dense/Tensordot/GatherV2+forward_seq_attention/dense/Tensordot/Const*

Tidx0*
	keep_dims( *
T0
[
-forward_seq_attention/dense/Tensordot/Const_1Const*
valueB: *
dtype0
�
,forward_seq_attention/dense/Tensordot/Prod_1Prod0forward_seq_attention/dense/Tensordot/GatherV2_1-forward_seq_attention/dense/Tensordot/Const_1*
T0*

Tidx0*
	keep_dims( 
[
1forward_seq_attention/dense/Tensordot/concat/axisConst*
value	B : *
dtype0
�
,forward_seq_attention/dense/Tensordot/concatConcatV2*forward_seq_attention/dense/Tensordot/free*forward_seq_attention/dense/Tensordot/axes1forward_seq_attention/dense/Tensordot/concat/axis*

Tidx0*
T0*
N
�
+forward_seq_attention/dense/Tensordot/stackPack*forward_seq_attention/dense/Tensordot/Prod,forward_seq_attention/dense/Tensordot/Prod_1*
T0*

axis *
N
�
/forward_seq_attention/dense/Tensordot/transpose	Transpose
ExpandDims,forward_seq_attention/dense/Tensordot/concat*
T0*
Tperm0
�
-forward_seq_attention/dense/Tensordot/ReshapeReshape/forward_seq_attention/dense/Tensordot/transpose+forward_seq_attention/dense/Tensordot/stack*
T0*
Tshape0
k
6forward_seq_attention/dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
1forward_seq_attention/dense/Tensordot/transpose_1	Transpose8mio_variable/forward_seq_attention/dense/kernel/variable6forward_seq_attention/dense/Tensordot/transpose_1/perm*
Tperm0*
T0
j
5forward_seq_attention/dense/Tensordot/Reshape_1/shapeConst*
valueB"0       *
dtype0
�
/forward_seq_attention/dense/Tensordot/Reshape_1Reshape1forward_seq_attention/dense/Tensordot/transpose_15forward_seq_attention/dense/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
,forward_seq_attention/dense/Tensordot/MatMulMatMul-forward_seq_attention/dense/Tensordot/Reshape/forward_seq_attention/dense/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
[
-forward_seq_attention/dense/Tensordot/Const_2Const*
valueB: *
dtype0
]
3forward_seq_attention/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0
�
.forward_seq_attention/dense/Tensordot/concat_1ConcatV2.forward_seq_attention/dense/Tensordot/GatherV2-forward_seq_attention/dense/Tensordot/Const_23forward_seq_attention/dense/Tensordot/concat_1/axis*
T0*
N*

Tidx0
�
%forward_seq_attention/dense/TensordotReshape,forward_seq_attention/dense/Tensordot/MatMul.forward_seq_attention/dense/Tensordot/concat_1*
T0*
Tshape0
�
#forward_seq_attention/dense/BiasAddBiasAdd%forward_seq_attention/dense/Tensordot6mio_variable/forward_seq_attention/dense/bias/variable*
data_formatNHWC*
T0
�
:mio_variable/forward_seq_attention/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$forward_seq_attention/dense_1/kernel*
shape
:0 
�
:mio_variable/forward_seq_attention/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$forward_seq_attention/dense_1/kernel*
shape
:0 
X
#Initializer_14/random_uniform/shapeConst*
valueB"0       *
dtype0
N
!Initializer_14/random_uniform/minConst*
valueB
 *�7��*
dtype0
N
!Initializer_14/random_uniform/maxConst*
valueB
 *�7�>*
dtype0
�
+Initializer_14/random_uniform/RandomUniformRandomUniform#Initializer_14/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_14/random_uniform/subSub!Initializer_14/random_uniform/max!Initializer_14/random_uniform/min*
T0
�
!Initializer_14/random_uniform/mulMul+Initializer_14/random_uniform/RandomUniform!Initializer_14/random_uniform/sub*
T0
s
Initializer_14/random_uniformAdd!Initializer_14/random_uniform/mul!Initializer_14/random_uniform/min*
T0
�
	Assign_14Assign:mio_variable/forward_seq_attention/dense_1/kernel/gradientInitializer_14/random_uniform*
validate_shape(*
use_locking(*
T0*M
_classC
A?loc:@mio_variable/forward_seq_attention/dense_1/kernel/gradient
�
8mio_variable/forward_seq_attention/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"forward_seq_attention/dense_1/bias*
shape: 
�
8mio_variable/forward_seq_attention/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape: *1
	container$"forward_seq_attention/dense_1/bias
E
Initializer_15/zerosConst*
valueB *    *
dtype0
�
	Assign_15Assign8mio_variable/forward_seq_attention/dense_1/bias/gradientInitializer_15/zeros*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/forward_seq_attention/dense_1/bias/gradient*
validate_shape(
o
6forward_seq_attention/dense_1/Tensordot/transpose/permConst*
dtype0*!
valueB"          
�
1forward_seq_attention/dense_1/Tensordot/transpose	TransposeExpandDims_36forward_seq_attention/dense_1/Tensordot/transpose/perm*
T0*
Tperm0
j
5forward_seq_attention/dense_1/Tensordot/Reshape/shapeConst*
valueB"2   0   *
dtype0
�
/forward_seq_attention/dense_1/Tensordot/ReshapeReshape1forward_seq_attention/dense_1/Tensordot/transpose5forward_seq_attention/dense_1/Tensordot/Reshape/shape*
T0*
Tshape0
m
8forward_seq_attention/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
3forward_seq_attention/dense_1/Tensordot/transpose_1	Transpose:mio_variable/forward_seq_attention/dense_1/kernel/variable8forward_seq_attention/dense_1/Tensordot/transpose_1/perm*
Tperm0*
T0
l
7forward_seq_attention/dense_1/Tensordot/Reshape_1/shapeConst*
valueB"0       *
dtype0
�
1forward_seq_attention/dense_1/Tensordot/Reshape_1Reshape3forward_seq_attention/dense_1/Tensordot/transpose_17forward_seq_attention/dense_1/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
.forward_seq_attention/dense_1/Tensordot/MatMulMatMul/forward_seq_attention/dense_1/Tensordot/Reshape1forward_seq_attention/dense_1/Tensordot/Reshape_1*
transpose_b( *
T0*
transpose_a( 
f
-forward_seq_attention/dense_1/Tensordot/shapeConst*
dtype0*!
valueB"   2       
�
'forward_seq_attention/dense_1/TensordotReshape.forward_seq_attention/dense_1/Tensordot/MatMul-forward_seq_attention/dense_1/Tensordot/shape*
T0*
Tshape0
�
%forward_seq_attention/dense_1/BiasAddBiasAdd'forward_seq_attention/dense_1/Tensordot8mio_variable/forward_seq_attention/dense_1/bias/variable*
T0*
data_formatNHWC
�
:mio_variable/forward_seq_attention/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$forward_seq_attention/dense_2/kernel*
shape
:0 
�
:mio_variable/forward_seq_attention/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:0 *3
	container&$forward_seq_attention/dense_2/kernel
X
#Initializer_16/random_uniform/shapeConst*
valueB"0       *
dtype0
N
!Initializer_16/random_uniform/minConst*
valueB
 *�7��*
dtype0
N
!Initializer_16/random_uniform/maxConst*
valueB
 *�7�>*
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
�
	Assign_16Assign:mio_variable/forward_seq_attention/dense_2/kernel/gradientInitializer_16/random_uniform*
use_locking(*
T0*M
_classC
A?loc:@mio_variable/forward_seq_attention/dense_2/kernel/gradient*
validate_shape(
�
8mio_variable/forward_seq_attention/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"forward_seq_attention/dense_2/bias*
shape: 
�
8mio_variable/forward_seq_attention/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"forward_seq_attention/dense_2/bias*
shape: 
E
Initializer_17/zerosConst*
valueB *    *
dtype0
�
	Assign_17Assign8mio_variable/forward_seq_attention/dense_2/bias/gradientInitializer_17/zeros*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/forward_seq_attention/dense_2/bias/gradient*
validate_shape(
o
6forward_seq_attention/dense_2/Tensordot/transpose/permConst*
dtype0*!
valueB"          
�
1forward_seq_attention/dense_2/Tensordot/transpose	TransposeExpandDims_36forward_seq_attention/dense_2/Tensordot/transpose/perm*
T0*
Tperm0
j
5forward_seq_attention/dense_2/Tensordot/Reshape/shapeConst*
valueB"2   0   *
dtype0
�
/forward_seq_attention/dense_2/Tensordot/ReshapeReshape1forward_seq_attention/dense_2/Tensordot/transpose5forward_seq_attention/dense_2/Tensordot/Reshape/shape*
T0*
Tshape0
m
8forward_seq_attention/dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
3forward_seq_attention/dense_2/Tensordot/transpose_1	Transpose:mio_variable/forward_seq_attention/dense_2/kernel/variable8forward_seq_attention/dense_2/Tensordot/transpose_1/perm*
Tperm0*
T0
l
7forward_seq_attention/dense_2/Tensordot/Reshape_1/shapeConst*
valueB"0       *
dtype0
�
1forward_seq_attention/dense_2/Tensordot/Reshape_1Reshape3forward_seq_attention/dense_2/Tensordot/transpose_17forward_seq_attention/dense_2/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
.forward_seq_attention/dense_2/Tensordot/MatMulMatMul/forward_seq_attention/dense_2/Tensordot/Reshape1forward_seq_attention/dense_2/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
f
-forward_seq_attention/dense_2/Tensordot/shapeConst*!
valueB"   2       *
dtype0
�
'forward_seq_attention/dense_2/TensordotReshape.forward_seq_attention/dense_2/Tensordot/MatMul-forward_seq_attention/dense_2/Tensordot/shape*
T0*
Tshape0
�
%forward_seq_attention/dense_2/BiasAddBiasAdd'forward_seq_attention/dense_2/Tensordot8mio_variable/forward_seq_attention/dense_2/bias/variable*
T0*
data_formatNHWC
X
%forward_seq_attention/Reshape/shape/0Const*
valueB :
���������*
dtype0
O
%forward_seq_attention/Reshape/shape/2Const*
dtype0*
value	B :
O
%forward_seq_attention/Reshape/shape/3Const*
value	B :*
dtype0
�
#forward_seq_attention/Reshape/shapePack%forward_seq_attention/Reshape/shape/0#forward_seq_attention/strided_slice%forward_seq_attention/Reshape/shape/2%forward_seq_attention/Reshape/shape/3*
T0*

axis *
N
�
forward_seq_attention/ReshapeReshape#forward_seq_attention/dense/BiasAdd#forward_seq_attention/Reshape/shape*
T0*
Tshape0
a
$forward_seq_attention/transpose/permConst*%
valueB"             *
dtype0
�
forward_seq_attention/transpose	Transposeforward_seq_attention/Reshape$forward_seq_attention/transpose/perm*
T0*
Tperm0
Z
'forward_seq_attention/Reshape_1/shape/0Const*
valueB :
���������*
dtype0
Q
'forward_seq_attention/Reshape_1/shape/2Const*
dtype0*
value	B :
Q
'forward_seq_attention/Reshape_1/shape/3Const*
value	B :*
dtype0
�
%forward_seq_attention/Reshape_1/shapePack'forward_seq_attention/Reshape_1/shape/0%forward_seq_attention/strided_slice_1'forward_seq_attention/Reshape_1/shape/2'forward_seq_attention/Reshape_1/shape/3*
T0*

axis *
N
�
forward_seq_attention/Reshape_1Reshape%forward_seq_attention/dense_1/BiasAdd%forward_seq_attention/Reshape_1/shape*
T0*
Tshape0
c
&forward_seq_attention/transpose_1/permConst*%
valueB"             *
dtype0
�
!forward_seq_attention/transpose_1	Transposeforward_seq_attention/Reshape_1&forward_seq_attention/transpose_1/perm*
Tperm0*
T0
Z
'forward_seq_attention/Reshape_2/shape/0Const*
valueB :
���������*
dtype0
Q
'forward_seq_attention/Reshape_2/shape/2Const*
value	B :*
dtype0
Q
'forward_seq_attention/Reshape_2/shape/3Const*
value	B :*
dtype0
�
%forward_seq_attention/Reshape_2/shapePack'forward_seq_attention/Reshape_2/shape/0%forward_seq_attention/strided_slice_1'forward_seq_attention/Reshape_2/shape/2'forward_seq_attention/Reshape_2/shape/3*
T0*

axis *
N
�
forward_seq_attention/Reshape_2Reshape%forward_seq_attention/dense_2/BiasAdd%forward_seq_attention/Reshape_2/shape*
T0*
Tshape0
c
&forward_seq_attention/transpose_2/permConst*%
valueB"             *
dtype0
�
!forward_seq_attention/transpose_2	Transposeforward_seq_attention/Reshape_2&forward_seq_attention/transpose_2/perm*
T0*
Tperm0
�
forward_seq_attention/MatMulBatchMatMulforward_seq_attention/transpose!forward_seq_attention/transpose_1*
T0*
adj_x( *
adj_y(
F
forward_seq_attention/Cast/xConst*
dtype0*
value	B :
h
forward_seq_attention/CastCastforward_seq_attention/Cast/x*

SrcT0*
Truncate( *

DstT0
G
forward_seq_attention/SqrtSqrtforward_seq_attention/Cast*
T0
k
forward_seq_attention/truedivRealDivforward_seq_attention/MatMulforward_seq_attention/Sqrt*
T0
P
forward_seq_attention/SoftmaxSoftmaxforward_seq_attention/truediv*
T0
�
forward_seq_attention/MatMul_1BatchMatMulforward_seq_attention/Softmax!forward_seq_attention/transpose_2*
adj_x( *
adj_y( *
T0
c
&forward_seq_attention/transpose_3/permConst*%
valueB"             *
dtype0
�
!forward_seq_attention/transpose_3	Transposeforward_seq_attention/MatMul_1&forward_seq_attention/transpose_3/perm*
T0*
Tperm0
Z
'forward_seq_attention/Reshape_3/shape/0Const*
valueB :
���������*
dtype0
Q
'forward_seq_attention/Reshape_3/shape/2Const*
value	B : *
dtype0
�
%forward_seq_attention/Reshape_3/shapePack'forward_seq_attention/Reshape_3/shape/0#forward_seq_attention/strided_slice'forward_seq_attention/Reshape_3/shape/2*
T0*

axis *
N
�
forward_seq_attention/Reshape_3Reshape!forward_seq_attention/transpose_3%forward_seq_attention/Reshape_3/shape*
T0*
Tshape0
I
comment_seq_attention/ShapeShape
ExpandDims*
T0*
out_type0
W
)comment_seq_attention/strided_slice/stackConst*
valueB:*
dtype0
Y
+comment_seq_attention/strided_slice/stack_1Const*
valueB:*
dtype0
Y
+comment_seq_attention/strided_slice/stack_2Const*
valueB:*
dtype0
�
#comment_seq_attention/strided_sliceStridedSlicecomment_seq_attention/Shape)comment_seq_attention/strided_slice/stack+comment_seq_attention/strided_slice/stack_1+comment_seq_attention/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0
V
comment_seq_attention/Shape_1Const*!
valueB"   2   0   *
dtype0
Y
+comment_seq_attention/strided_slice_1/stackConst*
valueB:*
dtype0
[
-comment_seq_attention/strided_slice_1/stack_1Const*
valueB:*
dtype0
[
-comment_seq_attention/strided_slice_1/stack_2Const*
dtype0*
valueB:
�
%comment_seq_attention/strided_slice_1StridedSlicecomment_seq_attention/Shape_1+comment_seq_attention/strided_slice_1/stack-comment_seq_attention/strided_slice_1/stack_1-comment_seq_attention/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
�
8mio_variable/comment_seq_attention/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:0 *1
	container$"comment_seq_attention/dense/kernel
�
8mio_variable/comment_seq_attention/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"comment_seq_attention/dense/kernel*
shape
:0 
X
#Initializer_18/random_uniform/shapeConst*
valueB"0       *
dtype0
N
!Initializer_18/random_uniform/minConst*
valueB
 *�7��*
dtype0
N
!Initializer_18/random_uniform/maxConst*
valueB
 *�7�>*
dtype0
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
�
	Assign_18Assign8mio_variable/comment_seq_attention/dense/kernel/gradientInitializer_18/random_uniform*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/comment_seq_attention/dense/kernel/gradient*
validate_shape(
�
6mio_variable/comment_seq_attention/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" comment_seq_attention/dense/bias*
shape: 
�
6mio_variable/comment_seq_attention/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" comment_seq_attention/dense/bias*
shape: 
E
Initializer_19/zerosConst*
dtype0*
valueB *    
�
	Assign_19Assign6mio_variable/comment_seq_attention/dense/bias/gradientInitializer_19/zeros*
use_locking(*
T0*I
_class?
=;loc:@mio_variable/comment_seq_attention/dense/bias/gradient*
validate_shape(
X
*comment_seq_attention/dense/Tensordot/axesConst*
valueB:*
dtype0
_
*comment_seq_attention/dense/Tensordot/freeConst*
valueB"       *
dtype0
Y
+comment_seq_attention/dense/Tensordot/ShapeShape
ExpandDims*
T0*
out_type0
]
3comment_seq_attention/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0
�
.comment_seq_attention/dense/Tensordot/GatherV2GatherV2+comment_seq_attention/dense/Tensordot/Shape*comment_seq_attention/dense/Tensordot/free3comment_seq_attention/dense/Tensordot/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0
_
5comment_seq_attention/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0
�
0comment_seq_attention/dense/Tensordot/GatherV2_1GatherV2+comment_seq_attention/dense/Tensordot/Shape*comment_seq_attention/dense/Tensordot/axes5comment_seq_attention/dense/Tensordot/GatherV2_1/axis*
Tindices0*
Tparams0*
Taxis0
Y
+comment_seq_attention/dense/Tensordot/ConstConst*
valueB: *
dtype0
�
*comment_seq_attention/dense/Tensordot/ProdProd.comment_seq_attention/dense/Tensordot/GatherV2+comment_seq_attention/dense/Tensordot/Const*

Tidx0*
	keep_dims( *
T0
[
-comment_seq_attention/dense/Tensordot/Const_1Const*
valueB: *
dtype0
�
,comment_seq_attention/dense/Tensordot/Prod_1Prod0comment_seq_attention/dense/Tensordot/GatherV2_1-comment_seq_attention/dense/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0
[
1comment_seq_attention/dense/Tensordot/concat/axisConst*
value	B : *
dtype0
�
,comment_seq_attention/dense/Tensordot/concatConcatV2*comment_seq_attention/dense/Tensordot/free*comment_seq_attention/dense/Tensordot/axes1comment_seq_attention/dense/Tensordot/concat/axis*
T0*
N*

Tidx0
�
+comment_seq_attention/dense/Tensordot/stackPack*comment_seq_attention/dense/Tensordot/Prod,comment_seq_attention/dense/Tensordot/Prod_1*
T0*

axis *
N
�
/comment_seq_attention/dense/Tensordot/transpose	Transpose
ExpandDims,comment_seq_attention/dense/Tensordot/concat*
T0*
Tperm0
�
-comment_seq_attention/dense/Tensordot/ReshapeReshape/comment_seq_attention/dense/Tensordot/transpose+comment_seq_attention/dense/Tensordot/stack*
T0*
Tshape0
k
6comment_seq_attention/dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
1comment_seq_attention/dense/Tensordot/transpose_1	Transpose8mio_variable/comment_seq_attention/dense/kernel/variable6comment_seq_attention/dense/Tensordot/transpose_1/perm*
Tperm0*
T0
j
5comment_seq_attention/dense/Tensordot/Reshape_1/shapeConst*
valueB"0       *
dtype0
�
/comment_seq_attention/dense/Tensordot/Reshape_1Reshape1comment_seq_attention/dense/Tensordot/transpose_15comment_seq_attention/dense/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
,comment_seq_attention/dense/Tensordot/MatMulMatMul-comment_seq_attention/dense/Tensordot/Reshape/comment_seq_attention/dense/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
[
-comment_seq_attention/dense/Tensordot/Const_2Const*
dtype0*
valueB: 
]
3comment_seq_attention/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0
�
.comment_seq_attention/dense/Tensordot/concat_1ConcatV2.comment_seq_attention/dense/Tensordot/GatherV2-comment_seq_attention/dense/Tensordot/Const_23comment_seq_attention/dense/Tensordot/concat_1/axis*

Tidx0*
T0*
N
�
%comment_seq_attention/dense/TensordotReshape,comment_seq_attention/dense/Tensordot/MatMul.comment_seq_attention/dense/Tensordot/concat_1*
T0*
Tshape0
�
#comment_seq_attention/dense/BiasAddBiasAdd%comment_seq_attention/dense/Tensordot6mio_variable/comment_seq_attention/dense/bias/variable*
T0*
data_formatNHWC
�
:mio_variable/comment_seq_attention/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$comment_seq_attention/dense_1/kernel*
shape
:0 
�
:mio_variable/comment_seq_attention/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$comment_seq_attention/dense_1/kernel*
shape
:0 
X
#Initializer_20/random_uniform/shapeConst*
valueB"0       *
dtype0
N
!Initializer_20/random_uniform/minConst*
dtype0*
valueB
 *�7��
N
!Initializer_20/random_uniform/maxConst*
valueB
 *�7�>*
dtype0
�
+Initializer_20/random_uniform/RandomUniformRandomUniform#Initializer_20/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_20/random_uniform/subSub!Initializer_20/random_uniform/max!Initializer_20/random_uniform/min*
T0
�
!Initializer_20/random_uniform/mulMul+Initializer_20/random_uniform/RandomUniform!Initializer_20/random_uniform/sub*
T0
s
Initializer_20/random_uniformAdd!Initializer_20/random_uniform/mul!Initializer_20/random_uniform/min*
T0
�
	Assign_20Assign:mio_variable/comment_seq_attention/dense_1/kernel/gradientInitializer_20/random_uniform*
use_locking(*
T0*M
_classC
A?loc:@mio_variable/comment_seq_attention/dense_1/kernel/gradient*
validate_shape(
�
8mio_variable/comment_seq_attention/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"comment_seq_attention/dense_1/bias*
shape: 
�
8mio_variable/comment_seq_attention/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape: *1
	container$"comment_seq_attention/dense_1/bias
E
Initializer_21/zerosConst*
valueB *    *
dtype0
�
	Assign_21Assign8mio_variable/comment_seq_attention/dense_1/bias/gradientInitializer_21/zeros*
validate_shape(*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/comment_seq_attention/dense_1/bias/gradient
o
6comment_seq_attention/dense_1/Tensordot/transpose/permConst*!
valueB"          *
dtype0
�
1comment_seq_attention/dense_1/Tensordot/transpose	TransposeExpandDims_46comment_seq_attention/dense_1/Tensordot/transpose/perm*
Tperm0*
T0
j
5comment_seq_attention/dense_1/Tensordot/Reshape/shapeConst*
dtype0*
valueB"2   0   
�
/comment_seq_attention/dense_1/Tensordot/ReshapeReshape1comment_seq_attention/dense_1/Tensordot/transpose5comment_seq_attention/dense_1/Tensordot/Reshape/shape*
T0*
Tshape0
m
8comment_seq_attention/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
3comment_seq_attention/dense_1/Tensordot/transpose_1	Transpose:mio_variable/comment_seq_attention/dense_1/kernel/variable8comment_seq_attention/dense_1/Tensordot/transpose_1/perm*
T0*
Tperm0
l
7comment_seq_attention/dense_1/Tensordot/Reshape_1/shapeConst*
valueB"0       *
dtype0
�
1comment_seq_attention/dense_1/Tensordot/Reshape_1Reshape3comment_seq_attention/dense_1/Tensordot/transpose_17comment_seq_attention/dense_1/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
.comment_seq_attention/dense_1/Tensordot/MatMulMatMul/comment_seq_attention/dense_1/Tensordot/Reshape1comment_seq_attention/dense_1/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
f
-comment_seq_attention/dense_1/Tensordot/shapeConst*
dtype0*!
valueB"   2       
�
'comment_seq_attention/dense_1/TensordotReshape.comment_seq_attention/dense_1/Tensordot/MatMul-comment_seq_attention/dense_1/Tensordot/shape*
T0*
Tshape0
�
%comment_seq_attention/dense_1/BiasAddBiasAdd'comment_seq_attention/dense_1/Tensordot8mio_variable/comment_seq_attention/dense_1/bias/variable*
T0*
data_formatNHWC
�
:mio_variable/comment_seq_attention/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:0 *3
	container&$comment_seq_attention/dense_2/kernel
�
:mio_variable/comment_seq_attention/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$comment_seq_attention/dense_2/kernel*
shape
:0 
X
#Initializer_22/random_uniform/shapeConst*
dtype0*
valueB"0       
N
!Initializer_22/random_uniform/minConst*
valueB
 *�7��*
dtype0
N
!Initializer_22/random_uniform/maxConst*
valueB
 *�7�>*
dtype0
�
+Initializer_22/random_uniform/RandomUniformRandomUniform#Initializer_22/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_22/random_uniform/subSub!Initializer_22/random_uniform/max!Initializer_22/random_uniform/min*
T0
�
!Initializer_22/random_uniform/mulMul+Initializer_22/random_uniform/RandomUniform!Initializer_22/random_uniform/sub*
T0
s
Initializer_22/random_uniformAdd!Initializer_22/random_uniform/mul!Initializer_22/random_uniform/min*
T0
�
	Assign_22Assign:mio_variable/comment_seq_attention/dense_2/kernel/gradientInitializer_22/random_uniform*
T0*M
_classC
A?loc:@mio_variable/comment_seq_attention/dense_2/kernel/gradient*
validate_shape(*
use_locking(
�
8mio_variable/comment_seq_attention/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"comment_seq_attention/dense_2/bias*
shape: 
�
8mio_variable/comment_seq_attention/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape: *1
	container$"comment_seq_attention/dense_2/bias
E
Initializer_23/zerosConst*
dtype0*
valueB *    
�
	Assign_23Assign8mio_variable/comment_seq_attention/dense_2/bias/gradientInitializer_23/zeros*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/comment_seq_attention/dense_2/bias/gradient*
validate_shape(
o
6comment_seq_attention/dense_2/Tensordot/transpose/permConst*!
valueB"          *
dtype0
�
1comment_seq_attention/dense_2/Tensordot/transpose	TransposeExpandDims_46comment_seq_attention/dense_2/Tensordot/transpose/perm*
T0*
Tperm0
j
5comment_seq_attention/dense_2/Tensordot/Reshape/shapeConst*
valueB"2   0   *
dtype0
�
/comment_seq_attention/dense_2/Tensordot/ReshapeReshape1comment_seq_attention/dense_2/Tensordot/transpose5comment_seq_attention/dense_2/Tensordot/Reshape/shape*
T0*
Tshape0
m
8comment_seq_attention/dense_2/Tensordot/transpose_1/permConst*
dtype0*
valueB"       
�
3comment_seq_attention/dense_2/Tensordot/transpose_1	Transpose:mio_variable/comment_seq_attention/dense_2/kernel/variable8comment_seq_attention/dense_2/Tensordot/transpose_1/perm*
Tperm0*
T0
l
7comment_seq_attention/dense_2/Tensordot/Reshape_1/shapeConst*
dtype0*
valueB"0       
�
1comment_seq_attention/dense_2/Tensordot/Reshape_1Reshape3comment_seq_attention/dense_2/Tensordot/transpose_17comment_seq_attention/dense_2/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
.comment_seq_attention/dense_2/Tensordot/MatMulMatMul/comment_seq_attention/dense_2/Tensordot/Reshape1comment_seq_attention/dense_2/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
f
-comment_seq_attention/dense_2/Tensordot/shapeConst*!
valueB"   2       *
dtype0
�
'comment_seq_attention/dense_2/TensordotReshape.comment_seq_attention/dense_2/Tensordot/MatMul-comment_seq_attention/dense_2/Tensordot/shape*
T0*
Tshape0
�
%comment_seq_attention/dense_2/BiasAddBiasAdd'comment_seq_attention/dense_2/Tensordot8mio_variable/comment_seq_attention/dense_2/bias/variable*
T0*
data_formatNHWC
X
%comment_seq_attention/Reshape/shape/0Const*
valueB :
���������*
dtype0
O
%comment_seq_attention/Reshape/shape/2Const*
value	B :*
dtype0
O
%comment_seq_attention/Reshape/shape/3Const*
value	B :*
dtype0
�
#comment_seq_attention/Reshape/shapePack%comment_seq_attention/Reshape/shape/0#comment_seq_attention/strided_slice%comment_seq_attention/Reshape/shape/2%comment_seq_attention/Reshape/shape/3*
N*
T0*

axis 
�
comment_seq_attention/ReshapeReshape#comment_seq_attention/dense/BiasAdd#comment_seq_attention/Reshape/shape*
T0*
Tshape0
a
$comment_seq_attention/transpose/permConst*%
valueB"             *
dtype0
�
comment_seq_attention/transpose	Transposecomment_seq_attention/Reshape$comment_seq_attention/transpose/perm*
Tperm0*
T0
Z
'comment_seq_attention/Reshape_1/shape/0Const*
dtype0*
valueB :
���������
Q
'comment_seq_attention/Reshape_1/shape/2Const*
value	B :*
dtype0
Q
'comment_seq_attention/Reshape_1/shape/3Const*
dtype0*
value	B :
�
%comment_seq_attention/Reshape_1/shapePack'comment_seq_attention/Reshape_1/shape/0%comment_seq_attention/strided_slice_1'comment_seq_attention/Reshape_1/shape/2'comment_seq_attention/Reshape_1/shape/3*
N*
T0*

axis 
�
comment_seq_attention/Reshape_1Reshape%comment_seq_attention/dense_1/BiasAdd%comment_seq_attention/Reshape_1/shape*
T0*
Tshape0
c
&comment_seq_attention/transpose_1/permConst*%
valueB"             *
dtype0
�
!comment_seq_attention/transpose_1	Transposecomment_seq_attention/Reshape_1&comment_seq_attention/transpose_1/perm*
T0*
Tperm0
Z
'comment_seq_attention/Reshape_2/shape/0Const*
valueB :
���������*
dtype0
Q
'comment_seq_attention/Reshape_2/shape/2Const*
value	B :*
dtype0
Q
'comment_seq_attention/Reshape_2/shape/3Const*
value	B :*
dtype0
�
%comment_seq_attention/Reshape_2/shapePack'comment_seq_attention/Reshape_2/shape/0%comment_seq_attention/strided_slice_1'comment_seq_attention/Reshape_2/shape/2'comment_seq_attention/Reshape_2/shape/3*
T0*

axis *
N
�
comment_seq_attention/Reshape_2Reshape%comment_seq_attention/dense_2/BiasAdd%comment_seq_attention/Reshape_2/shape*
T0*
Tshape0
c
&comment_seq_attention/transpose_2/permConst*%
valueB"             *
dtype0
�
!comment_seq_attention/transpose_2	Transposecomment_seq_attention/Reshape_2&comment_seq_attention/transpose_2/perm*
T0*
Tperm0
�
comment_seq_attention/MatMulBatchMatMulcomment_seq_attention/transpose!comment_seq_attention/transpose_1*
T0*
adj_x( *
adj_y(
F
comment_seq_attention/Cast/xConst*
value	B :*
dtype0
h
comment_seq_attention/CastCastcomment_seq_attention/Cast/x*

SrcT0*
Truncate( *

DstT0
G
comment_seq_attention/SqrtSqrtcomment_seq_attention/Cast*
T0
k
comment_seq_attention/truedivRealDivcomment_seq_attention/MatMulcomment_seq_attention/Sqrt*
T0
P
comment_seq_attention/SoftmaxSoftmaxcomment_seq_attention/truediv*
T0
�
comment_seq_attention/MatMul_1BatchMatMulcomment_seq_attention/Softmax!comment_seq_attention/transpose_2*
adj_x( *
adj_y( *
T0
c
&comment_seq_attention/transpose_3/permConst*%
valueB"             *
dtype0
�
!comment_seq_attention/transpose_3	Transposecomment_seq_attention/MatMul_1&comment_seq_attention/transpose_3/perm*
T0*
Tperm0
Z
'comment_seq_attention/Reshape_3/shape/0Const*
valueB :
���������*
dtype0
Q
'comment_seq_attention/Reshape_3/shape/2Const*
value	B : *
dtype0
�
%comment_seq_attention/Reshape_3/shapePack'comment_seq_attention/Reshape_3/shape/0#comment_seq_attention/strided_slice'comment_seq_attention/Reshape_3/shape/2*
T0*

axis *
N
�
comment_seq_attention/Reshape_3Reshape!comment_seq_attention/transpose_3%comment_seq_attention/Reshape_3/shape*
T0*
Tshape0
I
collect_seq_attention/ShapeShape
ExpandDims*
T0*
out_type0
W
)collect_seq_attention/strided_slice/stackConst*
valueB:*
dtype0
Y
+collect_seq_attention/strided_slice/stack_1Const*
valueB:*
dtype0
Y
+collect_seq_attention/strided_slice/stack_2Const*
valueB:*
dtype0
�
#collect_seq_attention/strided_sliceStridedSlicecollect_seq_attention/Shape)collect_seq_attention/strided_slice/stack+collect_seq_attention/strided_slice/stack_1+collect_seq_attention/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
V
collect_seq_attention/Shape_1Const*!
valueB"   2   0   *
dtype0
Y
+collect_seq_attention/strided_slice_1/stackConst*
valueB:*
dtype0
[
-collect_seq_attention/strided_slice_1/stack_1Const*
valueB:*
dtype0
[
-collect_seq_attention/strided_slice_1/stack_2Const*
valueB:*
dtype0
�
%collect_seq_attention/strided_slice_1StridedSlicecollect_seq_attention/Shape_1+collect_seq_attention/strided_slice_1/stack-collect_seq_attention/strided_slice_1/stack_1-collect_seq_attention/strided_slice_1/stack_2*
end_mask *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
�
8mio_variable/collect_seq_attention/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"collect_seq_attention/dense/kernel*
shape
:0 
�
8mio_variable/collect_seq_attention/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"collect_seq_attention/dense/kernel*
shape
:0 
X
#Initializer_24/random_uniform/shapeConst*
valueB"0       *
dtype0
N
!Initializer_24/random_uniform/minConst*
valueB
 *�7��*
dtype0
N
!Initializer_24/random_uniform/maxConst*
dtype0*
valueB
 *�7�>
�
+Initializer_24/random_uniform/RandomUniformRandomUniform#Initializer_24/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_24/random_uniform/subSub!Initializer_24/random_uniform/max!Initializer_24/random_uniform/min*
T0
�
!Initializer_24/random_uniform/mulMul+Initializer_24/random_uniform/RandomUniform!Initializer_24/random_uniform/sub*
T0
s
Initializer_24/random_uniformAdd!Initializer_24/random_uniform/mul!Initializer_24/random_uniform/min*
T0
�
	Assign_24Assign8mio_variable/collect_seq_attention/dense/kernel/gradientInitializer_24/random_uniform*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/collect_seq_attention/dense/kernel/gradient*
validate_shape(
�
6mio_variable/collect_seq_attention/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" collect_seq_attention/dense/bias*
shape: 
�
6mio_variable/collect_seq_attention/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" collect_seq_attention/dense/bias*
shape: 
E
Initializer_25/zerosConst*
dtype0*
valueB *    
�
	Assign_25Assign6mio_variable/collect_seq_attention/dense/bias/gradientInitializer_25/zeros*
T0*I
_class?
=;loc:@mio_variable/collect_seq_attention/dense/bias/gradient*
validate_shape(*
use_locking(
X
*collect_seq_attention/dense/Tensordot/axesConst*
dtype0*
valueB:
_
*collect_seq_attention/dense/Tensordot/freeConst*
dtype0*
valueB"       
Y
+collect_seq_attention/dense/Tensordot/ShapeShape
ExpandDims*
T0*
out_type0
]
3collect_seq_attention/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0
�
.collect_seq_attention/dense/Tensordot/GatherV2GatherV2+collect_seq_attention/dense/Tensordot/Shape*collect_seq_attention/dense/Tensordot/free3collect_seq_attention/dense/Tensordot/GatherV2/axis*
Tparams0*
Taxis0*
Tindices0
_
5collect_seq_attention/dense/Tensordot/GatherV2_1/axisConst*
dtype0*
value	B : 
�
0collect_seq_attention/dense/Tensordot/GatherV2_1GatherV2+collect_seq_attention/dense/Tensordot/Shape*collect_seq_attention/dense/Tensordot/axes5collect_seq_attention/dense/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0
Y
+collect_seq_attention/dense/Tensordot/ConstConst*
valueB: *
dtype0
�
*collect_seq_attention/dense/Tensordot/ProdProd.collect_seq_attention/dense/Tensordot/GatherV2+collect_seq_attention/dense/Tensordot/Const*

Tidx0*
	keep_dims( *
T0
[
-collect_seq_attention/dense/Tensordot/Const_1Const*
valueB: *
dtype0
�
,collect_seq_attention/dense/Tensordot/Prod_1Prod0collect_seq_attention/dense/Tensordot/GatherV2_1-collect_seq_attention/dense/Tensordot/Const_1*
T0*

Tidx0*
	keep_dims( 
[
1collect_seq_attention/dense/Tensordot/concat/axisConst*
dtype0*
value	B : 
�
,collect_seq_attention/dense/Tensordot/concatConcatV2*collect_seq_attention/dense/Tensordot/free*collect_seq_attention/dense/Tensordot/axes1collect_seq_attention/dense/Tensordot/concat/axis*
N*

Tidx0*
T0
�
+collect_seq_attention/dense/Tensordot/stackPack*collect_seq_attention/dense/Tensordot/Prod,collect_seq_attention/dense/Tensordot/Prod_1*
T0*

axis *
N
�
/collect_seq_attention/dense/Tensordot/transpose	Transpose
ExpandDims,collect_seq_attention/dense/Tensordot/concat*
Tperm0*
T0
�
-collect_seq_attention/dense/Tensordot/ReshapeReshape/collect_seq_attention/dense/Tensordot/transpose+collect_seq_attention/dense/Tensordot/stack*
T0*
Tshape0
k
6collect_seq_attention/dense/Tensordot/transpose_1/permConst*
dtype0*
valueB"       
�
1collect_seq_attention/dense/Tensordot/transpose_1	Transpose8mio_variable/collect_seq_attention/dense/kernel/variable6collect_seq_attention/dense/Tensordot/transpose_1/perm*
Tperm0*
T0
j
5collect_seq_attention/dense/Tensordot/Reshape_1/shapeConst*
valueB"0       *
dtype0
�
/collect_seq_attention/dense/Tensordot/Reshape_1Reshape1collect_seq_attention/dense/Tensordot/transpose_15collect_seq_attention/dense/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
,collect_seq_attention/dense/Tensordot/MatMulMatMul-collect_seq_attention/dense/Tensordot/Reshape/collect_seq_attention/dense/Tensordot/Reshape_1*
transpose_a( *
transpose_b( *
T0
[
-collect_seq_attention/dense/Tensordot/Const_2Const*
valueB: *
dtype0
]
3collect_seq_attention/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0
�
.collect_seq_attention/dense/Tensordot/concat_1ConcatV2.collect_seq_attention/dense/Tensordot/GatherV2-collect_seq_attention/dense/Tensordot/Const_23collect_seq_attention/dense/Tensordot/concat_1/axis*
T0*
N*

Tidx0
�
%collect_seq_attention/dense/TensordotReshape,collect_seq_attention/dense/Tensordot/MatMul.collect_seq_attention/dense/Tensordot/concat_1*
T0*
Tshape0
�
#collect_seq_attention/dense/BiasAddBiasAdd%collect_seq_attention/dense/Tensordot6mio_variable/collect_seq_attention/dense/bias/variable*
T0*
data_formatNHWC
�
:mio_variable/collect_seq_attention/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$collect_seq_attention/dense_1/kernel*
shape
:0 
�
:mio_variable/collect_seq_attention/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:0 *3
	container&$collect_seq_attention/dense_1/kernel
X
#Initializer_26/random_uniform/shapeConst*
valueB"0       *
dtype0
N
!Initializer_26/random_uniform/minConst*
valueB
 *�7��*
dtype0
N
!Initializer_26/random_uniform/maxConst*
valueB
 *�7�>*
dtype0
�
+Initializer_26/random_uniform/RandomUniformRandomUniform#Initializer_26/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_26/random_uniform/subSub!Initializer_26/random_uniform/max!Initializer_26/random_uniform/min*
T0
�
!Initializer_26/random_uniform/mulMul+Initializer_26/random_uniform/RandomUniform!Initializer_26/random_uniform/sub*
T0
s
Initializer_26/random_uniformAdd!Initializer_26/random_uniform/mul!Initializer_26/random_uniform/min*
T0
�
	Assign_26Assign:mio_variable/collect_seq_attention/dense_1/kernel/gradientInitializer_26/random_uniform*
use_locking(*
T0*M
_classC
A?loc:@mio_variable/collect_seq_attention/dense_1/kernel/gradient*
validate_shape(
�
8mio_variable/collect_seq_attention/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"collect_seq_attention/dense_1/bias*
shape: 
�
8mio_variable/collect_seq_attention/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"collect_seq_attention/dense_1/bias*
shape: 
E
Initializer_27/zerosConst*
dtype0*
valueB *    
�
	Assign_27Assign8mio_variable/collect_seq_attention/dense_1/bias/gradientInitializer_27/zeros*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/collect_seq_attention/dense_1/bias/gradient*
validate_shape(
o
6collect_seq_attention/dense_1/Tensordot/transpose/permConst*
dtype0*!
valueB"          
�
1collect_seq_attention/dense_1/Tensordot/transpose	TransposeExpandDims_56collect_seq_attention/dense_1/Tensordot/transpose/perm*
Tperm0*
T0
j
5collect_seq_attention/dense_1/Tensordot/Reshape/shapeConst*
dtype0*
valueB"2   0   
�
/collect_seq_attention/dense_1/Tensordot/ReshapeReshape1collect_seq_attention/dense_1/Tensordot/transpose5collect_seq_attention/dense_1/Tensordot/Reshape/shape*
T0*
Tshape0
m
8collect_seq_attention/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
3collect_seq_attention/dense_1/Tensordot/transpose_1	Transpose:mio_variable/collect_seq_attention/dense_1/kernel/variable8collect_seq_attention/dense_1/Tensordot/transpose_1/perm*
Tperm0*
T0
l
7collect_seq_attention/dense_1/Tensordot/Reshape_1/shapeConst*
valueB"0       *
dtype0
�
1collect_seq_attention/dense_1/Tensordot/Reshape_1Reshape3collect_seq_attention/dense_1/Tensordot/transpose_17collect_seq_attention/dense_1/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
.collect_seq_attention/dense_1/Tensordot/MatMulMatMul/collect_seq_attention/dense_1/Tensordot/Reshape1collect_seq_attention/dense_1/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
f
-collect_seq_attention/dense_1/Tensordot/shapeConst*
dtype0*!
valueB"   2       
�
'collect_seq_attention/dense_1/TensordotReshape.collect_seq_attention/dense_1/Tensordot/MatMul-collect_seq_attention/dense_1/Tensordot/shape*
T0*
Tshape0
�
%collect_seq_attention/dense_1/BiasAddBiasAdd'collect_seq_attention/dense_1/Tensordot8mio_variable/collect_seq_attention/dense_1/bias/variable*
T0*
data_formatNHWC
�
:mio_variable/collect_seq_attention/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$collect_seq_attention/dense_2/kernel*
shape
:0 
�
:mio_variable/collect_seq_attention/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$collect_seq_attention/dense_2/kernel*
shape
:0 
X
#Initializer_28/random_uniform/shapeConst*
valueB"0       *
dtype0
N
!Initializer_28/random_uniform/minConst*
dtype0*
valueB
 *�7��
N
!Initializer_28/random_uniform/maxConst*
dtype0*
valueB
 *�7�>
�
+Initializer_28/random_uniform/RandomUniformRandomUniform#Initializer_28/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_28/random_uniform/subSub!Initializer_28/random_uniform/max!Initializer_28/random_uniform/min*
T0
�
!Initializer_28/random_uniform/mulMul+Initializer_28/random_uniform/RandomUniform!Initializer_28/random_uniform/sub*
T0
s
Initializer_28/random_uniformAdd!Initializer_28/random_uniform/mul!Initializer_28/random_uniform/min*
T0
�
	Assign_28Assign:mio_variable/collect_seq_attention/dense_2/kernel/gradientInitializer_28/random_uniform*
T0*M
_classC
A?loc:@mio_variable/collect_seq_attention/dense_2/kernel/gradient*
validate_shape(*
use_locking(
�
8mio_variable/collect_seq_attention/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"collect_seq_attention/dense_2/bias*
shape: 
�
8mio_variable/collect_seq_attention/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"collect_seq_attention/dense_2/bias*
shape: 
E
Initializer_29/zerosConst*
dtype0*
valueB *    
�
	Assign_29Assign8mio_variable/collect_seq_attention/dense_2/bias/gradientInitializer_29/zeros*
T0*K
_classA
?=loc:@mio_variable/collect_seq_attention/dense_2/bias/gradient*
validate_shape(*
use_locking(
o
6collect_seq_attention/dense_2/Tensordot/transpose/permConst*!
valueB"          *
dtype0
�
1collect_seq_attention/dense_2/Tensordot/transpose	TransposeExpandDims_56collect_seq_attention/dense_2/Tensordot/transpose/perm*
T0*
Tperm0
j
5collect_seq_attention/dense_2/Tensordot/Reshape/shapeConst*
dtype0*
valueB"2   0   
�
/collect_seq_attention/dense_2/Tensordot/ReshapeReshape1collect_seq_attention/dense_2/Tensordot/transpose5collect_seq_attention/dense_2/Tensordot/Reshape/shape*
T0*
Tshape0
m
8collect_seq_attention/dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
3collect_seq_attention/dense_2/Tensordot/transpose_1	Transpose:mio_variable/collect_seq_attention/dense_2/kernel/variable8collect_seq_attention/dense_2/Tensordot/transpose_1/perm*
Tperm0*
T0
l
7collect_seq_attention/dense_2/Tensordot/Reshape_1/shapeConst*
valueB"0       *
dtype0
�
1collect_seq_attention/dense_2/Tensordot/Reshape_1Reshape3collect_seq_attention/dense_2/Tensordot/transpose_17collect_seq_attention/dense_2/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
.collect_seq_attention/dense_2/Tensordot/MatMulMatMul/collect_seq_attention/dense_2/Tensordot/Reshape1collect_seq_attention/dense_2/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
f
-collect_seq_attention/dense_2/Tensordot/shapeConst*
dtype0*!
valueB"   2       
�
'collect_seq_attention/dense_2/TensordotReshape.collect_seq_attention/dense_2/Tensordot/MatMul-collect_seq_attention/dense_2/Tensordot/shape*
T0*
Tshape0
�
%collect_seq_attention/dense_2/BiasAddBiasAdd'collect_seq_attention/dense_2/Tensordot8mio_variable/collect_seq_attention/dense_2/bias/variable*
T0*
data_formatNHWC
X
%collect_seq_attention/Reshape/shape/0Const*
valueB :
���������*
dtype0
O
%collect_seq_attention/Reshape/shape/2Const*
dtype0*
value	B :
O
%collect_seq_attention/Reshape/shape/3Const*
value	B :*
dtype0
�
#collect_seq_attention/Reshape/shapePack%collect_seq_attention/Reshape/shape/0#collect_seq_attention/strided_slice%collect_seq_attention/Reshape/shape/2%collect_seq_attention/Reshape/shape/3*
T0*

axis *
N
�
collect_seq_attention/ReshapeReshape#collect_seq_attention/dense/BiasAdd#collect_seq_attention/Reshape/shape*
T0*
Tshape0
a
$collect_seq_attention/transpose/permConst*
dtype0*%
valueB"             
�
collect_seq_attention/transpose	Transposecollect_seq_attention/Reshape$collect_seq_attention/transpose/perm*
T0*
Tperm0
Z
'collect_seq_attention/Reshape_1/shape/0Const*
valueB :
���������*
dtype0
Q
'collect_seq_attention/Reshape_1/shape/2Const*
value	B :*
dtype0
Q
'collect_seq_attention/Reshape_1/shape/3Const*
value	B :*
dtype0
�
%collect_seq_attention/Reshape_1/shapePack'collect_seq_attention/Reshape_1/shape/0%collect_seq_attention/strided_slice_1'collect_seq_attention/Reshape_1/shape/2'collect_seq_attention/Reshape_1/shape/3*
T0*

axis *
N
�
collect_seq_attention/Reshape_1Reshape%collect_seq_attention/dense_1/BiasAdd%collect_seq_attention/Reshape_1/shape*
T0*
Tshape0
c
&collect_seq_attention/transpose_1/permConst*%
valueB"             *
dtype0
�
!collect_seq_attention/transpose_1	Transposecollect_seq_attention/Reshape_1&collect_seq_attention/transpose_1/perm*
Tperm0*
T0
Z
'collect_seq_attention/Reshape_2/shape/0Const*
valueB :
���������*
dtype0
Q
'collect_seq_attention/Reshape_2/shape/2Const*
dtype0*
value	B :
Q
'collect_seq_attention/Reshape_2/shape/3Const*
value	B :*
dtype0
�
%collect_seq_attention/Reshape_2/shapePack'collect_seq_attention/Reshape_2/shape/0%collect_seq_attention/strided_slice_1'collect_seq_attention/Reshape_2/shape/2'collect_seq_attention/Reshape_2/shape/3*
T0*

axis *
N
�
collect_seq_attention/Reshape_2Reshape%collect_seq_attention/dense_2/BiasAdd%collect_seq_attention/Reshape_2/shape*
T0*
Tshape0
c
&collect_seq_attention/transpose_2/permConst*
dtype0*%
valueB"             
�
!collect_seq_attention/transpose_2	Transposecollect_seq_attention/Reshape_2&collect_seq_attention/transpose_2/perm*
Tperm0*
T0
�
collect_seq_attention/MatMulBatchMatMulcollect_seq_attention/transpose!collect_seq_attention/transpose_1*
adj_x( *
adj_y(*
T0
F
collect_seq_attention/Cast/xConst*
value	B :*
dtype0
h
collect_seq_attention/CastCastcollect_seq_attention/Cast/x*

SrcT0*
Truncate( *

DstT0
G
collect_seq_attention/SqrtSqrtcollect_seq_attention/Cast*
T0
k
collect_seq_attention/truedivRealDivcollect_seq_attention/MatMulcollect_seq_attention/Sqrt*
T0
P
collect_seq_attention/SoftmaxSoftmaxcollect_seq_attention/truediv*
T0
�
collect_seq_attention/MatMul_1BatchMatMulcollect_seq_attention/Softmax!collect_seq_attention/transpose_2*
T0*
adj_x( *
adj_y( 
c
&collect_seq_attention/transpose_3/permConst*%
valueB"             *
dtype0
�
!collect_seq_attention/transpose_3	Transposecollect_seq_attention/MatMul_1&collect_seq_attention/transpose_3/perm*
T0*
Tperm0
Z
'collect_seq_attention/Reshape_3/shape/0Const*
valueB :
���������*
dtype0
Q
'collect_seq_attention/Reshape_3/shape/2Const*
value	B : *
dtype0
�
%collect_seq_attention/Reshape_3/shapePack'collect_seq_attention/Reshape_3/shape/0#collect_seq_attention/strided_slice'collect_seq_attention/Reshape_3/shape/2*
T0*

axis *
N
�
collect_seq_attention/Reshape_3Reshape!collect_seq_attention/transpose_3%collect_seq_attention/Reshape_3/shape*
T0*
Tshape0
O
!profile_enter_seq_attention/ShapeShape
ExpandDims*
T0*
out_type0
]
/profile_enter_seq_attention/strided_slice/stackConst*
valueB:*
dtype0
_
1profile_enter_seq_attention/strided_slice/stack_1Const*
dtype0*
valueB:
_
1profile_enter_seq_attention/strided_slice/stack_2Const*
valueB:*
dtype0
�
)profile_enter_seq_attention/strided_sliceStridedSlice!profile_enter_seq_attention/Shape/profile_enter_seq_attention/strided_slice/stack1profile_enter_seq_attention/strided_slice/stack_11profile_enter_seq_attention/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
\
#profile_enter_seq_attention/Shape_1Const*!
valueB"   2   0   *
dtype0
_
1profile_enter_seq_attention/strided_slice_1/stackConst*
valueB:*
dtype0
a
3profile_enter_seq_attention/strided_slice_1/stack_1Const*
valueB:*
dtype0
a
3profile_enter_seq_attention/strided_slice_1/stack_2Const*
valueB:*
dtype0
�
+profile_enter_seq_attention/strided_slice_1StridedSlice#profile_enter_seq_attention/Shape_11profile_enter_seq_attention/strided_slice_1/stack3profile_enter_seq_attention/strided_slice_1/stack_13profile_enter_seq_attention/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
�
>mio_variable/profile_enter_seq_attention/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(profile_enter_seq_attention/dense/kernel*
shape
:0 
�
>mio_variable/profile_enter_seq_attention/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(profile_enter_seq_attention/dense/kernel*
shape
:0 
X
#Initializer_30/random_uniform/shapeConst*
valueB"0       *
dtype0
N
!Initializer_30/random_uniform/minConst*
valueB
 *�7��*
dtype0
N
!Initializer_30/random_uniform/maxConst*
valueB
 *�7�>*
dtype0
�
+Initializer_30/random_uniform/RandomUniformRandomUniform#Initializer_30/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_30/random_uniform/subSub!Initializer_30/random_uniform/max!Initializer_30/random_uniform/min*
T0
�
!Initializer_30/random_uniform/mulMul+Initializer_30/random_uniform/RandomUniform!Initializer_30/random_uniform/sub*
T0
s
Initializer_30/random_uniformAdd!Initializer_30/random_uniform/mul!Initializer_30/random_uniform/min*
T0
�
	Assign_30Assign>mio_variable/profile_enter_seq_attention/dense/kernel/gradientInitializer_30/random_uniform*
T0*Q
_classG
ECloc:@mio_variable/profile_enter_seq_attention/dense/kernel/gradient*
validate_shape(*
use_locking(
�
<mio_variable/profile_enter_seq_attention/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&profile_enter_seq_attention/dense/bias*
shape: 
�
<mio_variable/profile_enter_seq_attention/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&profile_enter_seq_attention/dense/bias*
shape: 
E
Initializer_31/zerosConst*
valueB *    *
dtype0
�
	Assign_31Assign<mio_variable/profile_enter_seq_attention/dense/bias/gradientInitializer_31/zeros*
validate_shape(*
use_locking(*
T0*O
_classE
CAloc:@mio_variable/profile_enter_seq_attention/dense/bias/gradient
^
0profile_enter_seq_attention/dense/Tensordot/axesConst*
dtype0*
valueB:
e
0profile_enter_seq_attention/dense/Tensordot/freeConst*
valueB"       *
dtype0
_
1profile_enter_seq_attention/dense/Tensordot/ShapeShape
ExpandDims*
T0*
out_type0
c
9profile_enter_seq_attention/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0
�
4profile_enter_seq_attention/dense/Tensordot/GatherV2GatherV21profile_enter_seq_attention/dense/Tensordot/Shape0profile_enter_seq_attention/dense/Tensordot/free9profile_enter_seq_attention/dense/Tensordot/GatherV2/axis*
Tparams0*
Taxis0*
Tindices0
e
;profile_enter_seq_attention/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0
�
6profile_enter_seq_attention/dense/Tensordot/GatherV2_1GatherV21profile_enter_seq_attention/dense/Tensordot/Shape0profile_enter_seq_attention/dense/Tensordot/axes;profile_enter_seq_attention/dense/Tensordot/GatherV2_1/axis*
Tindices0*
Tparams0*
Taxis0
_
1profile_enter_seq_attention/dense/Tensordot/ConstConst*
valueB: *
dtype0
�
0profile_enter_seq_attention/dense/Tensordot/ProdProd4profile_enter_seq_attention/dense/Tensordot/GatherV21profile_enter_seq_attention/dense/Tensordot/Const*
T0*

Tidx0*
	keep_dims( 
a
3profile_enter_seq_attention/dense/Tensordot/Const_1Const*
valueB: *
dtype0
�
2profile_enter_seq_attention/dense/Tensordot/Prod_1Prod6profile_enter_seq_attention/dense/Tensordot/GatherV2_13profile_enter_seq_attention/dense/Tensordot/Const_1*
T0*

Tidx0*
	keep_dims( 
a
7profile_enter_seq_attention/dense/Tensordot/concat/axisConst*
value	B : *
dtype0
�
2profile_enter_seq_attention/dense/Tensordot/concatConcatV20profile_enter_seq_attention/dense/Tensordot/free0profile_enter_seq_attention/dense/Tensordot/axes7profile_enter_seq_attention/dense/Tensordot/concat/axis*
T0*
N*

Tidx0
�
1profile_enter_seq_attention/dense/Tensordot/stackPack0profile_enter_seq_attention/dense/Tensordot/Prod2profile_enter_seq_attention/dense/Tensordot/Prod_1*
T0*

axis *
N
�
5profile_enter_seq_attention/dense/Tensordot/transpose	Transpose
ExpandDims2profile_enter_seq_attention/dense/Tensordot/concat*
T0*
Tperm0
�
3profile_enter_seq_attention/dense/Tensordot/ReshapeReshape5profile_enter_seq_attention/dense/Tensordot/transpose1profile_enter_seq_attention/dense/Tensordot/stack*
T0*
Tshape0
q
<profile_enter_seq_attention/dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
7profile_enter_seq_attention/dense/Tensordot/transpose_1	Transpose>mio_variable/profile_enter_seq_attention/dense/kernel/variable<profile_enter_seq_attention/dense/Tensordot/transpose_1/perm*
T0*
Tperm0
p
;profile_enter_seq_attention/dense/Tensordot/Reshape_1/shapeConst*
valueB"0       *
dtype0
�
5profile_enter_seq_attention/dense/Tensordot/Reshape_1Reshape7profile_enter_seq_attention/dense/Tensordot/transpose_1;profile_enter_seq_attention/dense/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
2profile_enter_seq_attention/dense/Tensordot/MatMulMatMul3profile_enter_seq_attention/dense/Tensordot/Reshape5profile_enter_seq_attention/dense/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
a
3profile_enter_seq_attention/dense/Tensordot/Const_2Const*
valueB: *
dtype0
c
9profile_enter_seq_attention/dense/Tensordot/concat_1/axisConst*
dtype0*
value	B : 
�
4profile_enter_seq_attention/dense/Tensordot/concat_1ConcatV24profile_enter_seq_attention/dense/Tensordot/GatherV23profile_enter_seq_attention/dense/Tensordot/Const_29profile_enter_seq_attention/dense/Tensordot/concat_1/axis*
T0*
N*

Tidx0
�
+profile_enter_seq_attention/dense/TensordotReshape2profile_enter_seq_attention/dense/Tensordot/MatMul4profile_enter_seq_attention/dense/Tensordot/concat_1*
T0*
Tshape0
�
)profile_enter_seq_attention/dense/BiasAddBiasAdd+profile_enter_seq_attention/dense/Tensordot<mio_variable/profile_enter_seq_attention/dense/bias/variable*
data_formatNHWC*
T0
�
@mio_variable/profile_enter_seq_attention/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*profile_enter_seq_attention/dense_1/kernel*
shape
:0 
�
@mio_variable/profile_enter_seq_attention/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*profile_enter_seq_attention/dense_1/kernel*
shape
:0 
X
#Initializer_32/random_uniform/shapeConst*
valueB"0       *
dtype0
N
!Initializer_32/random_uniform/minConst*
valueB
 *�7��*
dtype0
N
!Initializer_32/random_uniform/maxConst*
valueB
 *�7�>*
dtype0
�
+Initializer_32/random_uniform/RandomUniformRandomUniform#Initializer_32/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_32/random_uniform/subSub!Initializer_32/random_uniform/max!Initializer_32/random_uniform/min*
T0
�
!Initializer_32/random_uniform/mulMul+Initializer_32/random_uniform/RandomUniform!Initializer_32/random_uniform/sub*
T0
s
Initializer_32/random_uniformAdd!Initializer_32/random_uniform/mul!Initializer_32/random_uniform/min*
T0
�
	Assign_32Assign@mio_variable/profile_enter_seq_attention/dense_1/kernel/gradientInitializer_32/random_uniform*
T0*S
_classI
GEloc:@mio_variable/profile_enter_seq_attention/dense_1/kernel/gradient*
validate_shape(*
use_locking(
�
>mio_variable/profile_enter_seq_attention/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(profile_enter_seq_attention/dense_1/bias*
shape: 
�
>mio_variable/profile_enter_seq_attention/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(profile_enter_seq_attention/dense_1/bias*
shape: 
E
Initializer_33/zerosConst*
valueB *    *
dtype0
�
	Assign_33Assign>mio_variable/profile_enter_seq_attention/dense_1/bias/gradientInitializer_33/zeros*
use_locking(*
T0*Q
_classG
ECloc:@mio_variable/profile_enter_seq_attention/dense_1/bias/gradient*
validate_shape(
u
<profile_enter_seq_attention/dense_1/Tensordot/transpose/permConst*
dtype0*!
valueB"          
�
7profile_enter_seq_attention/dense_1/Tensordot/transpose	TransposeExpandDims_6<profile_enter_seq_attention/dense_1/Tensordot/transpose/perm*
T0*
Tperm0
p
;profile_enter_seq_attention/dense_1/Tensordot/Reshape/shapeConst*
dtype0*
valueB"2   0   
�
5profile_enter_seq_attention/dense_1/Tensordot/ReshapeReshape7profile_enter_seq_attention/dense_1/Tensordot/transpose;profile_enter_seq_attention/dense_1/Tensordot/Reshape/shape*
T0*
Tshape0
s
>profile_enter_seq_attention/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
9profile_enter_seq_attention/dense_1/Tensordot/transpose_1	Transpose@mio_variable/profile_enter_seq_attention/dense_1/kernel/variable>profile_enter_seq_attention/dense_1/Tensordot/transpose_1/perm*
Tperm0*
T0
r
=profile_enter_seq_attention/dense_1/Tensordot/Reshape_1/shapeConst*
valueB"0       *
dtype0
�
7profile_enter_seq_attention/dense_1/Tensordot/Reshape_1Reshape9profile_enter_seq_attention/dense_1/Tensordot/transpose_1=profile_enter_seq_attention/dense_1/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
4profile_enter_seq_attention/dense_1/Tensordot/MatMulMatMul5profile_enter_seq_attention/dense_1/Tensordot/Reshape7profile_enter_seq_attention/dense_1/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
l
3profile_enter_seq_attention/dense_1/Tensordot/shapeConst*!
valueB"   2       *
dtype0
�
-profile_enter_seq_attention/dense_1/TensordotReshape4profile_enter_seq_attention/dense_1/Tensordot/MatMul3profile_enter_seq_attention/dense_1/Tensordot/shape*
T0*
Tshape0
�
+profile_enter_seq_attention/dense_1/BiasAddBiasAdd-profile_enter_seq_attention/dense_1/Tensordot>mio_variable/profile_enter_seq_attention/dense_1/bias/variable*
T0*
data_formatNHWC
�
@mio_variable/profile_enter_seq_attention/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:0 *9
	container,*profile_enter_seq_attention/dense_2/kernel
�
@mio_variable/profile_enter_seq_attention/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:0 *9
	container,*profile_enter_seq_attention/dense_2/kernel
X
#Initializer_34/random_uniform/shapeConst*
valueB"0       *
dtype0
N
!Initializer_34/random_uniform/minConst*
valueB
 *�7��*
dtype0
N
!Initializer_34/random_uniform/maxConst*
valueB
 *�7�>*
dtype0
�
+Initializer_34/random_uniform/RandomUniformRandomUniform#Initializer_34/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_34/random_uniform/subSub!Initializer_34/random_uniform/max!Initializer_34/random_uniform/min*
T0
�
!Initializer_34/random_uniform/mulMul+Initializer_34/random_uniform/RandomUniform!Initializer_34/random_uniform/sub*
T0
s
Initializer_34/random_uniformAdd!Initializer_34/random_uniform/mul!Initializer_34/random_uniform/min*
T0
�
	Assign_34Assign@mio_variable/profile_enter_seq_attention/dense_2/kernel/gradientInitializer_34/random_uniform*
use_locking(*
T0*S
_classI
GEloc:@mio_variable/profile_enter_seq_attention/dense_2/kernel/gradient*
validate_shape(
�
>mio_variable/profile_enter_seq_attention/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(profile_enter_seq_attention/dense_2/bias*
shape: 
�
>mio_variable/profile_enter_seq_attention/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(profile_enter_seq_attention/dense_2/bias*
shape: 
E
Initializer_35/zerosConst*
valueB *    *
dtype0
�
	Assign_35Assign>mio_variable/profile_enter_seq_attention/dense_2/bias/gradientInitializer_35/zeros*
use_locking(*
T0*Q
_classG
ECloc:@mio_variable/profile_enter_seq_attention/dense_2/bias/gradient*
validate_shape(
u
<profile_enter_seq_attention/dense_2/Tensordot/transpose/permConst*
dtype0*!
valueB"          
�
7profile_enter_seq_attention/dense_2/Tensordot/transpose	TransposeExpandDims_6<profile_enter_seq_attention/dense_2/Tensordot/transpose/perm*
T0*
Tperm0
p
;profile_enter_seq_attention/dense_2/Tensordot/Reshape/shapeConst*
dtype0*
valueB"2   0   
�
5profile_enter_seq_attention/dense_2/Tensordot/ReshapeReshape7profile_enter_seq_attention/dense_2/Tensordot/transpose;profile_enter_seq_attention/dense_2/Tensordot/Reshape/shape*
T0*
Tshape0
s
>profile_enter_seq_attention/dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0
�
9profile_enter_seq_attention/dense_2/Tensordot/transpose_1	Transpose@mio_variable/profile_enter_seq_attention/dense_2/kernel/variable>profile_enter_seq_attention/dense_2/Tensordot/transpose_1/perm*
T0*
Tperm0
r
=profile_enter_seq_attention/dense_2/Tensordot/Reshape_1/shapeConst*
valueB"0       *
dtype0
�
7profile_enter_seq_attention/dense_2/Tensordot/Reshape_1Reshape9profile_enter_seq_attention/dense_2/Tensordot/transpose_1=profile_enter_seq_attention/dense_2/Tensordot/Reshape_1/shape*
T0*
Tshape0
�
4profile_enter_seq_attention/dense_2/Tensordot/MatMulMatMul5profile_enter_seq_attention/dense_2/Tensordot/Reshape7profile_enter_seq_attention/dense_2/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( 
l
3profile_enter_seq_attention/dense_2/Tensordot/shapeConst*!
valueB"   2       *
dtype0
�
-profile_enter_seq_attention/dense_2/TensordotReshape4profile_enter_seq_attention/dense_2/Tensordot/MatMul3profile_enter_seq_attention/dense_2/Tensordot/shape*
T0*
Tshape0
�
+profile_enter_seq_attention/dense_2/BiasAddBiasAdd-profile_enter_seq_attention/dense_2/Tensordot>mio_variable/profile_enter_seq_attention/dense_2/bias/variable*
T0*
data_formatNHWC
^
+profile_enter_seq_attention/Reshape/shape/0Const*
valueB :
���������*
dtype0
U
+profile_enter_seq_attention/Reshape/shape/2Const*
value	B :*
dtype0
U
+profile_enter_seq_attention/Reshape/shape/3Const*
value	B :*
dtype0
�
)profile_enter_seq_attention/Reshape/shapePack+profile_enter_seq_attention/Reshape/shape/0)profile_enter_seq_attention/strided_slice+profile_enter_seq_attention/Reshape/shape/2+profile_enter_seq_attention/Reshape/shape/3*
N*
T0*

axis 
�
#profile_enter_seq_attention/ReshapeReshape)profile_enter_seq_attention/dense/BiasAdd)profile_enter_seq_attention/Reshape/shape*
T0*
Tshape0
g
*profile_enter_seq_attention/transpose/permConst*
dtype0*%
valueB"             
�
%profile_enter_seq_attention/transpose	Transpose#profile_enter_seq_attention/Reshape*profile_enter_seq_attention/transpose/perm*
T0*
Tperm0
`
-profile_enter_seq_attention/Reshape_1/shape/0Const*
valueB :
���������*
dtype0
W
-profile_enter_seq_attention/Reshape_1/shape/2Const*
value	B :*
dtype0
W
-profile_enter_seq_attention/Reshape_1/shape/3Const*
value	B :*
dtype0
�
+profile_enter_seq_attention/Reshape_1/shapePack-profile_enter_seq_attention/Reshape_1/shape/0+profile_enter_seq_attention/strided_slice_1-profile_enter_seq_attention/Reshape_1/shape/2-profile_enter_seq_attention/Reshape_1/shape/3*
N*
T0*

axis 
�
%profile_enter_seq_attention/Reshape_1Reshape+profile_enter_seq_attention/dense_1/BiasAdd+profile_enter_seq_attention/Reshape_1/shape*
T0*
Tshape0
i
,profile_enter_seq_attention/transpose_1/permConst*%
valueB"             *
dtype0
�
'profile_enter_seq_attention/transpose_1	Transpose%profile_enter_seq_attention/Reshape_1,profile_enter_seq_attention/transpose_1/perm*
T0*
Tperm0
`
-profile_enter_seq_attention/Reshape_2/shape/0Const*
valueB :
���������*
dtype0
W
-profile_enter_seq_attention/Reshape_2/shape/2Const*
dtype0*
value	B :
W
-profile_enter_seq_attention/Reshape_2/shape/3Const*
value	B :*
dtype0
�
+profile_enter_seq_attention/Reshape_2/shapePack-profile_enter_seq_attention/Reshape_2/shape/0+profile_enter_seq_attention/strided_slice_1-profile_enter_seq_attention/Reshape_2/shape/2-profile_enter_seq_attention/Reshape_2/shape/3*
T0*

axis *
N
�
%profile_enter_seq_attention/Reshape_2Reshape+profile_enter_seq_attention/dense_2/BiasAdd+profile_enter_seq_attention/Reshape_2/shape*
T0*
Tshape0
i
,profile_enter_seq_attention/transpose_2/permConst*%
valueB"             *
dtype0
�
'profile_enter_seq_attention/transpose_2	Transpose%profile_enter_seq_attention/Reshape_2,profile_enter_seq_attention/transpose_2/perm*
Tperm0*
T0
�
"profile_enter_seq_attention/MatMulBatchMatMul%profile_enter_seq_attention/transpose'profile_enter_seq_attention/transpose_1*
T0*
adj_x( *
adj_y(
L
"profile_enter_seq_attention/Cast/xConst*
value	B :*
dtype0
t
 profile_enter_seq_attention/CastCast"profile_enter_seq_attention/Cast/x*
Truncate( *

DstT0*

SrcT0
S
 profile_enter_seq_attention/SqrtSqrt profile_enter_seq_attention/Cast*
T0
}
#profile_enter_seq_attention/truedivRealDiv"profile_enter_seq_attention/MatMul profile_enter_seq_attention/Sqrt*
T0
\
#profile_enter_seq_attention/SoftmaxSoftmax#profile_enter_seq_attention/truediv*
T0
�
$profile_enter_seq_attention/MatMul_1BatchMatMul#profile_enter_seq_attention/Softmax'profile_enter_seq_attention/transpose_2*
T0*
adj_x( *
adj_y( 
i
,profile_enter_seq_attention/transpose_3/permConst*
dtype0*%
valueB"             
�
'profile_enter_seq_attention/transpose_3	Transpose$profile_enter_seq_attention/MatMul_1,profile_enter_seq_attention/transpose_3/perm*
T0*
Tperm0
`
-profile_enter_seq_attention/Reshape_3/shape/0Const*
valueB :
���������*
dtype0
W
-profile_enter_seq_attention/Reshape_3/shape/2Const*
dtype0*
value	B : 
�
+profile_enter_seq_attention/Reshape_3/shapePack-profile_enter_seq_attention/Reshape_3/shape/0)profile_enter_seq_attention/strided_slice-profile_enter_seq_attention/Reshape_3/shape/2*
T0*

axis *
N
�
%profile_enter_seq_attention/Reshape_3Reshape'profile_enter_seq_attention/transpose_3+profile_enter_seq_attention/Reshape_3/shape*
T0*
Tshape0
7
concat_9/axisConst*
value	B :*
dtype0
�
concat_9ConcatV2like_seq_attention/Reshape_3follow_seq_attention/Reshape_3forward_seq_attention/Reshape_3comment_seq_attention/Reshape_3collect_seq_attention/Reshape_3%profile_enter_seq_attention/Reshape_3concat_9/axis*
N*

Tidx0*
T0
E
Reshape_22/shapeConst*
valueB"�����   *
dtype0
H

Reshape_22Reshapeconcat_9Reshape_22/shape*
T0*
Tshape0
A
concat_10/values_0/axisConst*
value	B : *
dtype0
�
concat_10/values_0GatherV2mio_embeddings/uid_emb/variableCastconcat_10/values_0/axis*
Tparams0*
Taxis0*
Tindices0
A
concat_10/values_1/axisConst*
value	B : *
dtype0
�
concat_10/values_1GatherV2 mio_embeddings/uid_stat/variableCastconcat_10/values_1/axis*
Taxis0*
Tindices0*
Tparams0
A
concat_10/values_2/axisConst*
value	B : *
dtype0
�
concat_10/values_2GatherV2 mio_embeddings/did_stat/variableCastconcat_10/values_2/axis*
Taxis0*
Tindices0*
Tparams0
A
concat_10/values_3/axisConst*
value	B : *
dtype0
�
concat_10/values_3GatherV2"mio_embeddings/uid_live_f/variableCastconcat_10/values_3/axis*
Taxis0*
Tindices0*
Tparams0
A
concat_10/values_4/axisConst*
value	B : *
dtype0
�
concat_10/values_4GatherV2!mio_embeddings/uid_loc_f/variableCastconcat_10/values_4/axis*
Tparams0*
Taxis0*
Tindices0
A
concat_10/values_5/axisConst*
value	B : *
dtype0
�
concat_10/values_5GatherV2$mio_embeddings/uid_viewid_f/variableCastconcat_10/values_5/axis*
Taxis0*
Tindices0*
Tparams0
8
concat_10/axisConst*
value	B :*
dtype0
�
	concat_10ConcatV2concat_10/values_0concat_10/values_1concat_10/values_2concat_10/values_3concat_10/values_4concat_10/values_5Sum
Reshape_22concat_10/axis*
T0*
N*

Tidx0
8
concat_11/axisConst*
value	B :*
dtype0
V
	concat_11ConcatV2	concat_10concatconcat_11/axis*

Tidx0*
T0*
N
�
3mio_extra_param/user_comment_cluster_level/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*)
	containeruser_comment_cluster_level*
shape:���������
�
3mio_extra_param/user_comment_cluster_level/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*)
	containeruser_comment_cluster_level*
shape:���������
�
/mio_extra_param/user_app_cluster_level/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containeruser_app_cluster_level*
shape:���������
�
/mio_extra_param/user_app_cluster_level/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containeruser_app_cluster_level*
shape:���������
>
concat_12/concat_dimConst*
value	B :*
dtype0
O
	concat_12Identity/mio_extra_param/user_app_cluster_level/variable*
T0
A
concat_13/values_1/axisConst*
value	B : *
dtype0
�
concat_13/values_1GatherV2mio_embeddings/uid_emb/variableCastconcat_13/values_1/axis*
Tindices0*
Tparams0*
Taxis0
8
concat_13/axisConst*
value	B :*
dtype0
�
	concat_13ConcatV2/mio_extra_param/user_app_cluster_level/variableconcat_13/values_1mio_embeddings/pid_emb/variablemio_embeddings/aid_emb/variableconcat_13/axis*
T0*
N*

Tidx0
�
0mio_variable/follow_layers/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*)
	containerfollow_layers/dense/kernel*
shape:
��
�
0mio_variable/follow_layers/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
��*)
	containerfollow_layers/dense/kernel
X
#Initializer_36/random_uniform/shapeConst*
valueB"0     *
dtype0
N
!Initializer_36/random_uniform/minConst*
valueB
 *ܨ��*
dtype0
N
!Initializer_36/random_uniform/maxConst*
dtype0*
valueB
 *ܨ�=
�
+Initializer_36/random_uniform/RandomUniformRandomUniform#Initializer_36/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_36/random_uniform/subSub!Initializer_36/random_uniform/max!Initializer_36/random_uniform/min*
T0
�
!Initializer_36/random_uniform/mulMul+Initializer_36/random_uniform/RandomUniform!Initializer_36/random_uniform/sub*
T0
s
Initializer_36/random_uniformAdd!Initializer_36/random_uniform/mul!Initializer_36/random_uniform/min*
T0
�
	Assign_36Assign0mio_variable/follow_layers/dense/kernel/gradientInitializer_36/random_uniform*
T0*C
_class9
75loc:@mio_variable/follow_layers/dense/kernel/gradient*
validate_shape(*
use_locking(
�
.mio_variable/follow_layers/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:�*'
	containerfollow_layers/dense/bias
�
.mio_variable/follow_layers/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerfollow_layers/dense/bias*
shape:�
F
Initializer_37/zerosConst*
valueB�*    *
dtype0
�
	Assign_37Assign.mio_variable/follow_layers/dense/bias/gradientInitializer_37/zeros*
use_locking(*
T0*A
_class7
53loc:@mio_variable/follow_layers/dense/bias/gradient*
validate_shape(
�
 model/follow_layers/dense/MatMulMatMul	concat_110mio_variable/follow_layers/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
!model/follow_layers/dense/BiasAddBiasAdd model/follow_layers/dense/MatMul.mio_variable/follow_layers/dense/bias/variable*
T0*
data_formatNHWC
V
)model/follow_layers/dense/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
'model/follow_layers/dense/LeakyRelu/mulMul)model/follow_layers/dense/LeakyRelu/alpha!model/follow_layers/dense/BiasAdd*
T0
�
#model/follow_layers/dense/LeakyReluMaximum'model/follow_layers/dense/LeakyRelu/mul!model/follow_layers/dense/BiasAdd*
T0
�
2mio_variable/follow_layers/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*+
	containerfollow_layers/dense_1/kernel*
shape:
��
�
2mio_variable/follow_layers/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*+
	containerfollow_layers/dense_1/kernel*
shape:
��
X
#Initializer_38/random_uniform/shapeConst*
valueB"   �   *
dtype0
N
!Initializer_38/random_uniform/minConst*
valueB
 *   �*
dtype0
N
!Initializer_38/random_uniform/maxConst*
valueB
 *   >*
dtype0
�
+Initializer_38/random_uniform/RandomUniformRandomUniform#Initializer_38/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_38/random_uniform/subSub!Initializer_38/random_uniform/max!Initializer_38/random_uniform/min*
T0
�
!Initializer_38/random_uniform/mulMul+Initializer_38/random_uniform/RandomUniform!Initializer_38/random_uniform/sub*
T0
s
Initializer_38/random_uniformAdd!Initializer_38/random_uniform/mul!Initializer_38/random_uniform/min*
T0
�
	Assign_38Assign2mio_variable/follow_layers/dense_1/kernel/gradientInitializer_38/random_uniform*
use_locking(*
T0*E
_class;
97loc:@mio_variable/follow_layers/dense_1/kernel/gradient*
validate_shape(
�
0mio_variable/follow_layers/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*)
	containerfollow_layers/dense_1/bias*
shape:�
�
0mio_variable/follow_layers/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*)
	containerfollow_layers/dense_1/bias*
shape:�
F
Initializer_39/zerosConst*
dtype0*
valueB�*    
�
	Assign_39Assign0mio_variable/follow_layers/dense_1/bias/gradientInitializer_39/zeros*
T0*C
_class9
75loc:@mio_variable/follow_layers/dense_1/bias/gradient*
validate_shape(*
use_locking(
�
"model/follow_layers/dense_1/MatMulMatMul#model/follow_layers/dense/LeakyRelu2mio_variable/follow_layers/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
#model/follow_layers/dense_1/BiasAddBiasAdd"model/follow_layers/dense_1/MatMul0mio_variable/follow_layers/dense_1/bias/variable*
T0*
data_formatNHWC
X
+model/follow_layers/dense_1/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
)model/follow_layers/dense_1/LeakyRelu/mulMul+model/follow_layers/dense_1/LeakyRelu/alpha#model/follow_layers/dense_1/BiasAdd*
T0
�
%model/follow_layers/dense_1/LeakyReluMaximum)model/follow_layers/dense_1/LeakyRelu/mul#model/follow_layers/dense_1/BiasAdd*
T0
�
2mio_variable/follow_layers/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�@*+
	containerfollow_layers/dense_2/kernel
�
2mio_variable/follow_layers/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�@*+
	containerfollow_layers/dense_2/kernel
X
#Initializer_40/random_uniform/shapeConst*
valueB"�   @   *
dtype0
N
!Initializer_40/random_uniform/minConst*
dtype0*
valueB
 *�5�
N
!Initializer_40/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
+Initializer_40/random_uniform/RandomUniformRandomUniform#Initializer_40/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_40/random_uniform/subSub!Initializer_40/random_uniform/max!Initializer_40/random_uniform/min*
T0
�
!Initializer_40/random_uniform/mulMul+Initializer_40/random_uniform/RandomUniform!Initializer_40/random_uniform/sub*
T0
s
Initializer_40/random_uniformAdd!Initializer_40/random_uniform/mul!Initializer_40/random_uniform/min*
T0
�
	Assign_40Assign2mio_variable/follow_layers/dense_2/kernel/gradientInitializer_40/random_uniform*
use_locking(*
T0*E
_class;
97loc:@mio_variable/follow_layers/dense_2/kernel/gradient*
validate_shape(
�
0mio_variable/follow_layers/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*)
	containerfollow_layers/dense_2/bias*
shape:@
�
0mio_variable/follow_layers/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*)
	containerfollow_layers/dense_2/bias*
shape:@
E
Initializer_41/zerosConst*
valueB@*    *
dtype0
�
	Assign_41Assign0mio_variable/follow_layers/dense_2/bias/gradientInitializer_41/zeros*
validate_shape(*
use_locking(*
T0*C
_class9
75loc:@mio_variable/follow_layers/dense_2/bias/gradient
�
"model/follow_layers/dense_2/MatMulMatMul%model/follow_layers/dense_1/LeakyRelu2mio_variable/follow_layers/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
#model/follow_layers/dense_2/BiasAddBiasAdd"model/follow_layers/dense_2/MatMul0mio_variable/follow_layers/dense_2/bias/variable*
T0*
data_formatNHWC
X
+model/follow_layers/dense_2/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
)model/follow_layers/dense_2/LeakyRelu/mulMul+model/follow_layers/dense_2/LeakyRelu/alpha#model/follow_layers/dense_2/BiasAdd*
T0
�
%model/follow_layers/dense_2/LeakyReluMaximum)model/follow_layers/dense_2/LeakyRelu/mul#model/follow_layers/dense_2/BiasAdd*
T0
�
2mio_variable/follow_layers/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*+
	containerfollow_layers/dense_3/kernel*
shape
:@
�
2mio_variable/follow_layers/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*+
	containerfollow_layers/dense_3/kernel*
shape
:@
X
#Initializer_42/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_42/random_uniform/minConst*
valueB
 *����*
dtype0
N
!Initializer_42/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
+Initializer_42/random_uniform/RandomUniformRandomUniform#Initializer_42/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_42/random_uniform/subSub!Initializer_42/random_uniform/max!Initializer_42/random_uniform/min*
T0
�
!Initializer_42/random_uniform/mulMul+Initializer_42/random_uniform/RandomUniform!Initializer_42/random_uniform/sub*
T0
s
Initializer_42/random_uniformAdd!Initializer_42/random_uniform/mul!Initializer_42/random_uniform/min*
T0
�
	Assign_42Assign2mio_variable/follow_layers/dense_3/kernel/gradientInitializer_42/random_uniform*
use_locking(*
T0*E
_class;
97loc:@mio_variable/follow_layers/dense_3/kernel/gradient*
validate_shape(
�
0mio_variable/follow_layers/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*)
	containerfollow_layers/dense_3/bias*
shape:
�
0mio_variable/follow_layers/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*)
	containerfollow_layers/dense_3/bias*
shape:
E
Initializer_43/zerosConst*
valueB*    *
dtype0
�
	Assign_43Assign0mio_variable/follow_layers/dense_3/bias/gradientInitializer_43/zeros*
validate_shape(*
use_locking(*
T0*C
_class9
75loc:@mio_variable/follow_layers/dense_3/bias/gradient
�
"model/follow_layers/dense_3/MatMulMatMul%model/follow_layers/dense_2/LeakyRelu2mio_variable/follow_layers/dense_3/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
#model/follow_layers/dense_3/BiasAddBiasAdd"model/follow_layers/dense_3/MatMul0mio_variable/follow_layers/dense_3/bias/variable*
T0*
data_formatNHWC
�
8mio_variable/forward_inside_layers/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
��*1
	container$"forward_inside_layers/dense/kernel
�
8mio_variable/forward_inside_layers/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"forward_inside_layers/dense/kernel*
shape:
��
X
#Initializer_44/random_uniform/shapeConst*
valueB"0     *
dtype0
N
!Initializer_44/random_uniform/minConst*
dtype0*
valueB
 *ܨ��
N
!Initializer_44/random_uniform/maxConst*
dtype0*
valueB
 *ܨ�=
�
+Initializer_44/random_uniform/RandomUniformRandomUniform#Initializer_44/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_44/random_uniform/subSub!Initializer_44/random_uniform/max!Initializer_44/random_uniform/min*
T0
�
!Initializer_44/random_uniform/mulMul+Initializer_44/random_uniform/RandomUniform!Initializer_44/random_uniform/sub*
T0
s
Initializer_44/random_uniformAdd!Initializer_44/random_uniform/mul!Initializer_44/random_uniform/min*
T0
�
	Assign_44Assign8mio_variable/forward_inside_layers/dense/kernel/gradientInitializer_44/random_uniform*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/forward_inside_layers/dense/kernel/gradient*
validate_shape(
�
6mio_variable/forward_inside_layers/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" forward_inside_layers/dense/bias*
shape:�
�
6mio_variable/forward_inside_layers/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" forward_inside_layers/dense/bias*
shape:�
F
Initializer_45/zerosConst*
dtype0*
valueB�*    
�
	Assign_45Assign6mio_variable/forward_inside_layers/dense/bias/gradientInitializer_45/zeros*
validate_shape(*
use_locking(*
T0*I
_class?
=;loc:@mio_variable/forward_inside_layers/dense/bias/gradient
�
(model/forward_inside_layers/dense/MatMulMatMul	concat_118mio_variable/forward_inside_layers/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
)model/forward_inside_layers/dense/BiasAddBiasAdd(model/forward_inside_layers/dense/MatMul6mio_variable/forward_inside_layers/dense/bias/variable*
data_formatNHWC*
T0
^
1model/forward_inside_layers/dense/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
/model/forward_inside_layers/dense/LeakyRelu/mulMul1model/forward_inside_layers/dense/LeakyRelu/alpha)model/forward_inside_layers/dense/BiasAdd*
T0
�
+model/forward_inside_layers/dense/LeakyReluMaximum/model/forward_inside_layers/dense/LeakyRelu/mul)model/forward_inside_layers/dense/BiasAdd*
T0
�
:mio_variable/forward_inside_layers/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$forward_inside_layers/dense_1/kernel*
shape:
��
�
:mio_variable/forward_inside_layers/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$forward_inside_layers/dense_1/kernel*
shape:
��
X
#Initializer_46/random_uniform/shapeConst*
valueB"   �   *
dtype0
N
!Initializer_46/random_uniform/minConst*
valueB
 *   �*
dtype0
N
!Initializer_46/random_uniform/maxConst*
valueB
 *   >*
dtype0
�
+Initializer_46/random_uniform/RandomUniformRandomUniform#Initializer_46/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_46/random_uniform/subSub!Initializer_46/random_uniform/max!Initializer_46/random_uniform/min*
T0
�
!Initializer_46/random_uniform/mulMul+Initializer_46/random_uniform/RandomUniform!Initializer_46/random_uniform/sub*
T0
s
Initializer_46/random_uniformAdd!Initializer_46/random_uniform/mul!Initializer_46/random_uniform/min*
T0
�
	Assign_46Assign:mio_variable/forward_inside_layers/dense_1/kernel/gradientInitializer_46/random_uniform*
use_locking(*
T0*M
_classC
A?loc:@mio_variable/forward_inside_layers/dense_1/kernel/gradient*
validate_shape(
�
8mio_variable/forward_inside_layers/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"forward_inside_layers/dense_1/bias*
shape:�
�
8mio_variable/forward_inside_layers/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"forward_inside_layers/dense_1/bias*
shape:�
F
Initializer_47/zerosConst*
valueB�*    *
dtype0
�
	Assign_47Assign8mio_variable/forward_inside_layers/dense_1/bias/gradientInitializer_47/zeros*
T0*K
_classA
?=loc:@mio_variable/forward_inside_layers/dense_1/bias/gradient*
validate_shape(*
use_locking(
�
*model/forward_inside_layers/dense_1/MatMulMatMul+model/forward_inside_layers/dense/LeakyRelu:mio_variable/forward_inside_layers/dense_1/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
+model/forward_inside_layers/dense_1/BiasAddBiasAdd*model/forward_inside_layers/dense_1/MatMul8mio_variable/forward_inside_layers/dense_1/bias/variable*
T0*
data_formatNHWC
`
3model/forward_inside_layers/dense_1/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
1model/forward_inside_layers/dense_1/LeakyRelu/mulMul3model/forward_inside_layers/dense_1/LeakyRelu/alpha+model/forward_inside_layers/dense_1/BiasAdd*
T0
�
-model/forward_inside_layers/dense_1/LeakyReluMaximum1model/forward_inside_layers/dense_1/LeakyRelu/mul+model/forward_inside_layers/dense_1/BiasAdd*
T0
�
:mio_variable/forward_inside_layers/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�@*3
	container&$forward_inside_layers/dense_2/kernel
�
:mio_variable/forward_inside_layers/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$forward_inside_layers/dense_2/kernel*
shape:	�@
X
#Initializer_48/random_uniform/shapeConst*
valueB"�   @   *
dtype0
N
!Initializer_48/random_uniform/minConst*
valueB
 *�5�*
dtype0
N
!Initializer_48/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
+Initializer_48/random_uniform/RandomUniformRandomUniform#Initializer_48/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_48/random_uniform/subSub!Initializer_48/random_uniform/max!Initializer_48/random_uniform/min*
T0
�
!Initializer_48/random_uniform/mulMul+Initializer_48/random_uniform/RandomUniform!Initializer_48/random_uniform/sub*
T0
s
Initializer_48/random_uniformAdd!Initializer_48/random_uniform/mul!Initializer_48/random_uniform/min*
T0
�
	Assign_48Assign:mio_variable/forward_inside_layers/dense_2/kernel/gradientInitializer_48/random_uniform*
use_locking(*
T0*M
_classC
A?loc:@mio_variable/forward_inside_layers/dense_2/kernel/gradient*
validate_shape(
�
8mio_variable/forward_inside_layers/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"forward_inside_layers/dense_2/bias*
shape:@
�
8mio_variable/forward_inside_layers/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"forward_inside_layers/dense_2/bias*
shape:@
E
Initializer_49/zerosConst*
valueB@*    *
dtype0
�
	Assign_49Assign8mio_variable/forward_inside_layers/dense_2/bias/gradientInitializer_49/zeros*
validate_shape(*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/forward_inside_layers/dense_2/bias/gradient
�
*model/forward_inside_layers/dense_2/MatMulMatMul-model/forward_inside_layers/dense_1/LeakyRelu:mio_variable/forward_inside_layers/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
+model/forward_inside_layers/dense_2/BiasAddBiasAdd*model/forward_inside_layers/dense_2/MatMul8mio_variable/forward_inside_layers/dense_2/bias/variable*
T0*
data_formatNHWC
`
3model/forward_inside_layers/dense_2/LeakyRelu/alphaConst*
dtype0*
valueB
 *��L>
�
1model/forward_inside_layers/dense_2/LeakyRelu/mulMul3model/forward_inside_layers/dense_2/LeakyRelu/alpha+model/forward_inside_layers/dense_2/BiasAdd*
T0
�
-model/forward_inside_layers/dense_2/LeakyReluMaximum1model/forward_inside_layers/dense_2/LeakyRelu/mul+model/forward_inside_layers/dense_2/BiasAdd*
T0
�
:mio_variable/forward_inside_layers/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*3
	container&$forward_inside_layers/dense_3/kernel
�
:mio_variable/forward_inside_layers/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$forward_inside_layers/dense_3/kernel*
shape
:@
X
#Initializer_50/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_50/random_uniform/minConst*
valueB
 *����*
dtype0
N
!Initializer_50/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
+Initializer_50/random_uniform/RandomUniformRandomUniform#Initializer_50/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_50/random_uniform/subSub!Initializer_50/random_uniform/max!Initializer_50/random_uniform/min*
T0
�
!Initializer_50/random_uniform/mulMul+Initializer_50/random_uniform/RandomUniform!Initializer_50/random_uniform/sub*
T0
s
Initializer_50/random_uniformAdd!Initializer_50/random_uniform/mul!Initializer_50/random_uniform/min*
T0
�
	Assign_50Assign:mio_variable/forward_inside_layers/dense_3/kernel/gradientInitializer_50/random_uniform*
T0*M
_classC
A?loc:@mio_variable/forward_inside_layers/dense_3/kernel/gradient*
validate_shape(*
use_locking(
�
8mio_variable/forward_inside_layers/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"forward_inside_layers/dense_3/bias*
shape:
�
8mio_variable/forward_inside_layers/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"forward_inside_layers/dense_3/bias*
shape:
E
Initializer_51/zerosConst*
dtype0*
valueB*    
�
	Assign_51Assign8mio_variable/forward_inside_layers/dense_3/bias/gradientInitializer_51/zeros*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/forward_inside_layers/dense_3/bias/gradient*
validate_shape(
�
*model/forward_inside_layers/dense_3/MatMulMatMul-model/forward_inside_layers/dense_2/LeakyRelu:mio_variable/forward_inside_layers/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
+model/forward_inside_layers/dense_3/BiasAddBiasAdd*model/forward_inside_layers/dense_3/MatMul8mio_variable/forward_inside_layers/dense_3/bias/variable*
data_formatNHWC*
T0
�
2mio_variable/interact_layers/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*+
	containerinteract_layers/dense/kernel*
shape:
��
�
2mio_variable/interact_layers/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*+
	containerinteract_layers/dense/kernel*
shape:
��
X
#Initializer_52/random_uniform/shapeConst*
valueB"0     *
dtype0
N
!Initializer_52/random_uniform/minConst*
valueB
 *ܨ��*
dtype0
N
!Initializer_52/random_uniform/maxConst*
valueB
 *ܨ�=*
dtype0
�
+Initializer_52/random_uniform/RandomUniformRandomUniform#Initializer_52/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_52/random_uniform/subSub!Initializer_52/random_uniform/max!Initializer_52/random_uniform/min*
T0
�
!Initializer_52/random_uniform/mulMul+Initializer_52/random_uniform/RandomUniform!Initializer_52/random_uniform/sub*
T0
s
Initializer_52/random_uniformAdd!Initializer_52/random_uniform/mul!Initializer_52/random_uniform/min*
T0
�
	Assign_52Assign2mio_variable/interact_layers/dense/kernel/gradientInitializer_52/random_uniform*
validate_shape(*
use_locking(*
T0*E
_class;
97loc:@mio_variable/interact_layers/dense/kernel/gradient
�
0mio_variable/interact_layers/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:�*)
	containerinteract_layers/dense/bias
�
0mio_variable/interact_layers/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:�*)
	containerinteract_layers/dense/bias
F
Initializer_53/zerosConst*
dtype0*
valueB�*    
�
	Assign_53Assign0mio_variable/interact_layers/dense/bias/gradientInitializer_53/zeros*
use_locking(*
T0*C
_class9
75loc:@mio_variable/interact_layers/dense/bias/gradient*
validate_shape(
�
"model/interact_layers/dense/MatMulMatMul	concat_112mio_variable/interact_layers/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
#model/interact_layers/dense/BiasAddBiasAdd"model/interact_layers/dense/MatMul0mio_variable/interact_layers/dense/bias/variable*
T0*
data_formatNHWC
X
+model/interact_layers/dense/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
)model/interact_layers/dense/LeakyRelu/mulMul+model/interact_layers/dense/LeakyRelu/alpha#model/interact_layers/dense/BiasAdd*
T0
�
%model/interact_layers/dense/LeakyReluMaximum)model/interact_layers/dense/LeakyRelu/mul#model/interact_layers/dense/BiasAdd*
T0
�
4mio_variable/interact_layers/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*-
	container interact_layers/dense_1/kernel*
shape:
��
�
4mio_variable/interact_layers/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*-
	container interact_layers/dense_1/kernel*
shape:
��
X
#Initializer_54/random_uniform/shapeConst*
valueB"   �   *
dtype0
N
!Initializer_54/random_uniform/minConst*
valueB
 *   �*
dtype0
N
!Initializer_54/random_uniform/maxConst*
valueB
 *   >*
dtype0
�
+Initializer_54/random_uniform/RandomUniformRandomUniform#Initializer_54/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_54/random_uniform/subSub!Initializer_54/random_uniform/max!Initializer_54/random_uniform/min*
T0
�
!Initializer_54/random_uniform/mulMul+Initializer_54/random_uniform/RandomUniform!Initializer_54/random_uniform/sub*
T0
s
Initializer_54/random_uniformAdd!Initializer_54/random_uniform/mul!Initializer_54/random_uniform/min*
T0
�
	Assign_54Assign4mio_variable/interact_layers/dense_1/kernel/gradientInitializer_54/random_uniform*
use_locking(*
T0*G
_class=
;9loc:@mio_variable/interact_layers/dense_1/kernel/gradient*
validate_shape(
�
2mio_variable/interact_layers/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*+
	containerinteract_layers/dense_1/bias*
shape:�
�
2mio_variable/interact_layers/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*+
	containerinteract_layers/dense_1/bias*
shape:�
F
Initializer_55/zerosConst*
dtype0*
valueB�*    
�
	Assign_55Assign2mio_variable/interact_layers/dense_1/bias/gradientInitializer_55/zeros*
validate_shape(*
use_locking(*
T0*E
_class;
97loc:@mio_variable/interact_layers/dense_1/bias/gradient
�
$model/interact_layers/dense_1/MatMulMatMul%model/interact_layers/dense/LeakyRelu4mio_variable/interact_layers/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
%model/interact_layers/dense_1/BiasAddBiasAdd$model/interact_layers/dense_1/MatMul2mio_variable/interact_layers/dense_1/bias/variable*
T0*
data_formatNHWC
Z
-model/interact_layers/dense_1/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
+model/interact_layers/dense_1/LeakyRelu/mulMul-model/interact_layers/dense_1/LeakyRelu/alpha%model/interact_layers/dense_1/BiasAdd*
T0
�
'model/interact_layers/dense_1/LeakyReluMaximum+model/interact_layers/dense_1/LeakyRelu/mul%model/interact_layers/dense_1/BiasAdd*
T0
�
4mio_variable/interact_layers/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*-
	container interact_layers/dense_2/kernel*
shape:	�@
�
4mio_variable/interact_layers/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�@*-
	container interact_layers/dense_2/kernel
X
#Initializer_56/random_uniform/shapeConst*
valueB"�   @   *
dtype0
N
!Initializer_56/random_uniform/minConst*
valueB
 *�5�*
dtype0
N
!Initializer_56/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
+Initializer_56/random_uniform/RandomUniformRandomUniform#Initializer_56/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_56/random_uniform/subSub!Initializer_56/random_uniform/max!Initializer_56/random_uniform/min*
T0
�
!Initializer_56/random_uniform/mulMul+Initializer_56/random_uniform/RandomUniform!Initializer_56/random_uniform/sub*
T0
s
Initializer_56/random_uniformAdd!Initializer_56/random_uniform/mul!Initializer_56/random_uniform/min*
T0
�
	Assign_56Assign4mio_variable/interact_layers/dense_2/kernel/gradientInitializer_56/random_uniform*
T0*G
_class=
;9loc:@mio_variable/interact_layers/dense_2/kernel/gradient*
validate_shape(*
use_locking(
�
2mio_variable/interact_layers/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*+
	containerinteract_layers/dense_2/bias*
shape:@
�
2mio_variable/interact_layers/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*+
	containerinteract_layers/dense_2/bias*
shape:@
E
Initializer_57/zerosConst*
valueB@*    *
dtype0
�
	Assign_57Assign2mio_variable/interact_layers/dense_2/bias/gradientInitializer_57/zeros*
use_locking(*
T0*E
_class;
97loc:@mio_variable/interact_layers/dense_2/bias/gradient*
validate_shape(
�
$model/interact_layers/dense_2/MatMulMatMul'model/interact_layers/dense_1/LeakyRelu4mio_variable/interact_layers/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
%model/interact_layers/dense_2/BiasAddBiasAdd$model/interact_layers/dense_2/MatMul2mio_variable/interact_layers/dense_2/bias/variable*
T0*
data_formatNHWC
Z
-model/interact_layers/dense_2/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
+model/interact_layers/dense_2/LeakyRelu/mulMul-model/interact_layers/dense_2/LeakyRelu/alpha%model/interact_layers/dense_2/BiasAdd*
T0
�
'model/interact_layers/dense_2/LeakyReluMaximum+model/interact_layers/dense_2/LeakyRelu/mul%model/interact_layers/dense_2/BiasAdd*
T0
�
4mio_variable/interact_layers/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*-
	container interact_layers/dense_3/kernel*
shape
:@
�
4mio_variable/interact_layers/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*-
	container interact_layers/dense_3/kernel
X
#Initializer_58/random_uniform/shapeConst*
dtype0*
valueB"@      
N
!Initializer_58/random_uniform/minConst*
valueB
 *����*
dtype0
N
!Initializer_58/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
+Initializer_58/random_uniform/RandomUniformRandomUniform#Initializer_58/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_58/random_uniform/subSub!Initializer_58/random_uniform/max!Initializer_58/random_uniform/min*
T0
�
!Initializer_58/random_uniform/mulMul+Initializer_58/random_uniform/RandomUniform!Initializer_58/random_uniform/sub*
T0
s
Initializer_58/random_uniformAdd!Initializer_58/random_uniform/mul!Initializer_58/random_uniform/min*
T0
�
	Assign_58Assign4mio_variable/interact_layers/dense_3/kernel/gradientInitializer_58/random_uniform*
validate_shape(*
use_locking(*
T0*G
_class=
;9loc:@mio_variable/interact_layers/dense_3/kernel/gradient
�
2mio_variable/interact_layers/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*+
	containerinteract_layers/dense_3/bias*
shape:
�
2mio_variable/interact_layers/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*+
	containerinteract_layers/dense_3/bias*
shape:
E
Initializer_59/zerosConst*
dtype0*
valueB*    
�
	Assign_59Assign2mio_variable/interact_layers/dense_3/bias/gradientInitializer_59/zeros*
validate_shape(*
use_locking(*
T0*E
_class;
97loc:@mio_variable/interact_layers/dense_3/bias/gradient
�
$model/interact_layers/dense_3/MatMulMatMul'model/interact_layers/dense_2/LeakyRelu4mio_variable/interact_layers/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
%model/interact_layers/dense_3/BiasAddBiasAdd$model/interact_layers/dense_3/MatMul2mio_variable/interact_layers/dense_3/bias/variable*
data_formatNHWC*
T0
�
7mio_variable/click_comment_layers/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!click_comment_layers/dense/kernel*
shape:
��
�
7mio_variable/click_comment_layers/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
��*0
	container#!click_comment_layers/dense/kernel
X
#Initializer_60/random_uniform/shapeConst*
valueB"0     *
dtype0
N
!Initializer_60/random_uniform/minConst*
valueB
 *ܨ��*
dtype0
N
!Initializer_60/random_uniform/maxConst*
valueB
 *ܨ�=*
dtype0
�
+Initializer_60/random_uniform/RandomUniformRandomUniform#Initializer_60/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_60/random_uniform/subSub!Initializer_60/random_uniform/max!Initializer_60/random_uniform/min*
T0
�
!Initializer_60/random_uniform/mulMul+Initializer_60/random_uniform/RandomUniform!Initializer_60/random_uniform/sub*
T0
s
Initializer_60/random_uniformAdd!Initializer_60/random_uniform/mul!Initializer_60/random_uniform/min*
T0
�
	Assign_60Assign7mio_variable/click_comment_layers/dense/kernel/gradientInitializer_60/random_uniform*
use_locking(*
T0*J
_class@
><loc:@mio_variable/click_comment_layers/dense/kernel/gradient*
validate_shape(
�
5mio_variable/click_comment_layers/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!click_comment_layers/dense/bias*
shape:�
�
5mio_variable/click_comment_layers/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!click_comment_layers/dense/bias*
shape:�
F
Initializer_61/zerosConst*
dtype0*
valueB�*    
�
	Assign_61Assign5mio_variable/click_comment_layers/dense/bias/gradientInitializer_61/zeros*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/click_comment_layers/dense/bias/gradient*
validate_shape(
�
'model/click_comment_layers/dense/MatMulMatMul	concat_117mio_variable/click_comment_layers/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
(model/click_comment_layers/dense/BiasAddBiasAdd'model/click_comment_layers/dense/MatMul5mio_variable/click_comment_layers/dense/bias/variable*
data_formatNHWC*
T0
]
0model/click_comment_layers/dense/LeakyRelu/alphaConst*
dtype0*
valueB
 *��L>
�
.model/click_comment_layers/dense/LeakyRelu/mulMul0model/click_comment_layers/dense/LeakyRelu/alpha(model/click_comment_layers/dense/BiasAdd*
T0
�
*model/click_comment_layers/dense/LeakyReluMaximum.model/click_comment_layers/dense/LeakyRelu/mul(model/click_comment_layers/dense/BiasAdd*
T0
�
9mio_variable/click_comment_layers/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#click_comment_layers/dense_1/kernel*
shape:
��
�
9mio_variable/click_comment_layers/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#click_comment_layers/dense_1/kernel*
shape:
��
X
#Initializer_62/random_uniform/shapeConst*
valueB"   �   *
dtype0
N
!Initializer_62/random_uniform/minConst*
valueB
 *   �*
dtype0
N
!Initializer_62/random_uniform/maxConst*
valueB
 *   >*
dtype0
�
+Initializer_62/random_uniform/RandomUniformRandomUniform#Initializer_62/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_62/random_uniform/subSub!Initializer_62/random_uniform/max!Initializer_62/random_uniform/min*
T0
�
!Initializer_62/random_uniform/mulMul+Initializer_62/random_uniform/RandomUniform!Initializer_62/random_uniform/sub*
T0
s
Initializer_62/random_uniformAdd!Initializer_62/random_uniform/mul!Initializer_62/random_uniform/min*
T0
�
	Assign_62Assign9mio_variable/click_comment_layers/dense_1/kernel/gradientInitializer_62/random_uniform*
validate_shape(*
use_locking(*
T0*L
_classB
@>loc:@mio_variable/click_comment_layers/dense_1/kernel/gradient
�
7mio_variable/click_comment_layers/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!click_comment_layers/dense_1/bias*
shape:�
�
7mio_variable/click_comment_layers/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:�*0
	container#!click_comment_layers/dense_1/bias
F
Initializer_63/zerosConst*
valueB�*    *
dtype0
�
	Assign_63Assign7mio_variable/click_comment_layers/dense_1/bias/gradientInitializer_63/zeros*
T0*J
_class@
><loc:@mio_variable/click_comment_layers/dense_1/bias/gradient*
validate_shape(*
use_locking(
�
)model/click_comment_layers/dense_1/MatMulMatMul*model/click_comment_layers/dense/LeakyRelu9mio_variable/click_comment_layers/dense_1/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
*model/click_comment_layers/dense_1/BiasAddBiasAdd)model/click_comment_layers/dense_1/MatMul7mio_variable/click_comment_layers/dense_1/bias/variable*
T0*
data_formatNHWC
_
2model/click_comment_layers/dense_1/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
0model/click_comment_layers/dense_1/LeakyRelu/mulMul2model/click_comment_layers/dense_1/LeakyRelu/alpha*model/click_comment_layers/dense_1/BiasAdd*
T0
�
,model/click_comment_layers/dense_1/LeakyReluMaximum0model/click_comment_layers/dense_1/LeakyRelu/mul*model/click_comment_layers/dense_1/BiasAdd*
T0
�
9mio_variable/click_comment_layers/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#click_comment_layers/dense_2/kernel*
shape:	�@
�
9mio_variable/click_comment_layers/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#click_comment_layers/dense_2/kernel*
shape:	�@
X
#Initializer_64/random_uniform/shapeConst*
valueB"�   @   *
dtype0
N
!Initializer_64/random_uniform/minConst*
valueB
 *�5�*
dtype0
N
!Initializer_64/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
+Initializer_64/random_uniform/RandomUniformRandomUniform#Initializer_64/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_64/random_uniform/subSub!Initializer_64/random_uniform/max!Initializer_64/random_uniform/min*
T0
�
!Initializer_64/random_uniform/mulMul+Initializer_64/random_uniform/RandomUniform!Initializer_64/random_uniform/sub*
T0
s
Initializer_64/random_uniformAdd!Initializer_64/random_uniform/mul!Initializer_64/random_uniform/min*
T0
�
	Assign_64Assign9mio_variable/click_comment_layers/dense_2/kernel/gradientInitializer_64/random_uniform*
validate_shape(*
use_locking(*
T0*L
_classB
@>loc:@mio_variable/click_comment_layers/dense_2/kernel/gradient
�
7mio_variable/click_comment_layers/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*0
	container#!click_comment_layers/dense_2/bias
�
7mio_variable/click_comment_layers/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!click_comment_layers/dense_2/bias*
shape:@
E
Initializer_65/zerosConst*
valueB@*    *
dtype0
�
	Assign_65Assign7mio_variable/click_comment_layers/dense_2/bias/gradientInitializer_65/zeros*
use_locking(*
T0*J
_class@
><loc:@mio_variable/click_comment_layers/dense_2/bias/gradient*
validate_shape(
�
)model/click_comment_layers/dense_2/MatMulMatMul,model/click_comment_layers/dense_1/LeakyRelu9mio_variable/click_comment_layers/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
*model/click_comment_layers/dense_2/BiasAddBiasAdd)model/click_comment_layers/dense_2/MatMul7mio_variable/click_comment_layers/dense_2/bias/variable*
T0*
data_formatNHWC
_
2model/click_comment_layers/dense_2/LeakyRelu/alphaConst*
dtype0*
valueB
 *��L>
�
0model/click_comment_layers/dense_2/LeakyRelu/mulMul2model/click_comment_layers/dense_2/LeakyRelu/alpha*model/click_comment_layers/dense_2/BiasAdd*
T0
�
,model/click_comment_layers/dense_2/LeakyReluMaximum0model/click_comment_layers/dense_2/LeakyRelu/mul*model/click_comment_layers/dense_2/BiasAdd*
T0
�
9mio_variable/click_comment_layers/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#click_comment_layers/dense_3/kernel*
shape
:@
�
9mio_variable/click_comment_layers/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#click_comment_layers/dense_3/kernel*
shape
:@
X
#Initializer_66/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_66/random_uniform/minConst*
valueB
 *����*
dtype0
N
!Initializer_66/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
+Initializer_66/random_uniform/RandomUniformRandomUniform#Initializer_66/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_66/random_uniform/subSub!Initializer_66/random_uniform/max!Initializer_66/random_uniform/min*
T0
�
!Initializer_66/random_uniform/mulMul+Initializer_66/random_uniform/RandomUniform!Initializer_66/random_uniform/sub*
T0
s
Initializer_66/random_uniformAdd!Initializer_66/random_uniform/mul!Initializer_66/random_uniform/min*
T0
�
	Assign_66Assign9mio_variable/click_comment_layers/dense_3/kernel/gradientInitializer_66/random_uniform*
T0*L
_classB
@>loc:@mio_variable/click_comment_layers/dense_3/kernel/gradient*
validate_shape(*
use_locking(
�
7mio_variable/click_comment_layers/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!click_comment_layers/dense_3/bias*
shape:
�
7mio_variable/click_comment_layers/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!click_comment_layers/dense_3/bias*
shape:
E
Initializer_67/zerosConst*
valueB*    *
dtype0
�
	Assign_67Assign7mio_variable/click_comment_layers/dense_3/bias/gradientInitializer_67/zeros*
T0*J
_class@
><loc:@mio_variable/click_comment_layers/dense_3/bias/gradient*
validate_shape(*
use_locking(
�
)model/click_comment_layers/dense_3/MatMulMatMul,model/click_comment_layers/dense_2/LeakyRelu9mio_variable/click_comment_layers/dense_3/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
*model/click_comment_layers/dense_3/BiasAddBiasAdd)model/click_comment_layers/dense_3/MatMul7mio_variable/click_comment_layers/dense_3/bias/variable*
T0*
data_formatNHWC
�
6mio_variable/comment_time_layers/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" comment_time_layers/dense/kernel*
shape:
��
�
6mio_variable/comment_time_layers/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
��*/
	container" comment_time_layers/dense/kernel
X
#Initializer_68/random_uniform/shapeConst*
valueB"0     *
dtype0
N
!Initializer_68/random_uniform/minConst*
valueB
 *ܨ��*
dtype0
N
!Initializer_68/random_uniform/maxConst*
valueB
 *ܨ�=*
dtype0
�
+Initializer_68/random_uniform/RandomUniformRandomUniform#Initializer_68/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_68/random_uniform/subSub!Initializer_68/random_uniform/max!Initializer_68/random_uniform/min*
T0
�
!Initializer_68/random_uniform/mulMul+Initializer_68/random_uniform/RandomUniform!Initializer_68/random_uniform/sub*
T0
s
Initializer_68/random_uniformAdd!Initializer_68/random_uniform/mul!Initializer_68/random_uniform/min*
T0
�
	Assign_68Assign6mio_variable/comment_time_layers/dense/kernel/gradientInitializer_68/random_uniform*
validate_shape(*
use_locking(*
T0*I
_class?
=;loc:@mio_variable/comment_time_layers/dense/kernel/gradient
�
4mio_variable/comment_time_layers/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*-
	container comment_time_layers/dense/bias*
shape:�
�
4mio_variable/comment_time_layers/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*-
	container comment_time_layers/dense/bias*
shape:�
F
Initializer_69/zerosConst*
dtype0*
valueB�*    
�
	Assign_69Assign4mio_variable/comment_time_layers/dense/bias/gradientInitializer_69/zeros*
validate_shape(*
use_locking(*
T0*G
_class=
;9loc:@mio_variable/comment_time_layers/dense/bias/gradient
�
&model/comment_time_layers/dense/MatMulMatMul	concat_116mio_variable/comment_time_layers/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
'model/comment_time_layers/dense/BiasAddBiasAdd&model/comment_time_layers/dense/MatMul4mio_variable/comment_time_layers/dense/bias/variable*
data_formatNHWC*
T0
\
/model/comment_time_layers/dense/LeakyRelu/alphaConst*
dtype0*
valueB
 *��L>
�
-model/comment_time_layers/dense/LeakyRelu/mulMul/model/comment_time_layers/dense/LeakyRelu/alpha'model/comment_time_layers/dense/BiasAdd*
T0
�
)model/comment_time_layers/dense/LeakyReluMaximum-model/comment_time_layers/dense/LeakyRelu/mul'model/comment_time_layers/dense/BiasAdd*
T0
�
8mio_variable/comment_time_layers/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
��*1
	container$"comment_time_layers/dense_1/kernel
�
8mio_variable/comment_time_layers/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"comment_time_layers/dense_1/kernel*
shape:
��
X
#Initializer_70/random_uniform/shapeConst*
valueB"   �   *
dtype0
N
!Initializer_70/random_uniform/minConst*
valueB
 *   �*
dtype0
N
!Initializer_70/random_uniform/maxConst*
valueB
 *   >*
dtype0
�
+Initializer_70/random_uniform/RandomUniformRandomUniform#Initializer_70/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_70/random_uniform/subSub!Initializer_70/random_uniform/max!Initializer_70/random_uniform/min*
T0
�
!Initializer_70/random_uniform/mulMul+Initializer_70/random_uniform/RandomUniform!Initializer_70/random_uniform/sub*
T0
s
Initializer_70/random_uniformAdd!Initializer_70/random_uniform/mul!Initializer_70/random_uniform/min*
T0
�
	Assign_70Assign8mio_variable/comment_time_layers/dense_1/kernel/gradientInitializer_70/random_uniform*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/comment_time_layers/dense_1/kernel/gradient*
validate_shape(
�
6mio_variable/comment_time_layers/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:�*/
	container" comment_time_layers/dense_1/bias
�
6mio_variable/comment_time_layers/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" comment_time_layers/dense_1/bias*
shape:�
F
Initializer_71/zerosConst*
dtype0*
valueB�*    
�
	Assign_71Assign6mio_variable/comment_time_layers/dense_1/bias/gradientInitializer_71/zeros*
validate_shape(*
use_locking(*
T0*I
_class?
=;loc:@mio_variable/comment_time_layers/dense_1/bias/gradient
�
(model/comment_time_layers/dense_1/MatMulMatMul)model/comment_time_layers/dense/LeakyRelu8mio_variable/comment_time_layers/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
)model/comment_time_layers/dense_1/BiasAddBiasAdd(model/comment_time_layers/dense_1/MatMul6mio_variable/comment_time_layers/dense_1/bias/variable*
data_formatNHWC*
T0
^
1model/comment_time_layers/dense_1/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
/model/comment_time_layers/dense_1/LeakyRelu/mulMul1model/comment_time_layers/dense_1/LeakyRelu/alpha)model/comment_time_layers/dense_1/BiasAdd*
T0
�
+model/comment_time_layers/dense_1/LeakyReluMaximum/model/comment_time_layers/dense_1/LeakyRelu/mul)model/comment_time_layers/dense_1/BiasAdd*
T0
�
8mio_variable/comment_time_layers/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"comment_time_layers/dense_2/kernel*
shape:	�@
�
8mio_variable/comment_time_layers/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�@*1
	container$"comment_time_layers/dense_2/kernel
X
#Initializer_72/random_uniform/shapeConst*
valueB"�   @   *
dtype0
N
!Initializer_72/random_uniform/minConst*
valueB
 *�5�*
dtype0
N
!Initializer_72/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
+Initializer_72/random_uniform/RandomUniformRandomUniform#Initializer_72/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_72/random_uniform/subSub!Initializer_72/random_uniform/max!Initializer_72/random_uniform/min*
T0
�
!Initializer_72/random_uniform/mulMul+Initializer_72/random_uniform/RandomUniform!Initializer_72/random_uniform/sub*
T0
s
Initializer_72/random_uniformAdd!Initializer_72/random_uniform/mul!Initializer_72/random_uniform/min*
T0
�
	Assign_72Assign8mio_variable/comment_time_layers/dense_2/kernel/gradientInitializer_72/random_uniform*
T0*K
_classA
?=loc:@mio_variable/comment_time_layers/dense_2/kernel/gradient*
validate_shape(*
use_locking(
�
6mio_variable/comment_time_layers/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*/
	container" comment_time_layers/dense_2/bias
�
6mio_variable/comment_time_layers/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*/
	container" comment_time_layers/dense_2/bias
E
Initializer_73/zerosConst*
valueB@*    *
dtype0
�
	Assign_73Assign6mio_variable/comment_time_layers/dense_2/bias/gradientInitializer_73/zeros*
use_locking(*
T0*I
_class?
=;loc:@mio_variable/comment_time_layers/dense_2/bias/gradient*
validate_shape(
�
(model/comment_time_layers/dense_2/MatMulMatMul+model/comment_time_layers/dense_1/LeakyRelu8mio_variable/comment_time_layers/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
)model/comment_time_layers/dense_2/BiasAddBiasAdd(model/comment_time_layers/dense_2/MatMul6mio_variable/comment_time_layers/dense_2/bias/variable*
data_formatNHWC*
T0
^
1model/comment_time_layers/dense_2/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
/model/comment_time_layers/dense_2/LeakyRelu/mulMul1model/comment_time_layers/dense_2/LeakyRelu/alpha)model/comment_time_layers/dense_2/BiasAdd*
T0
�
+model/comment_time_layers/dense_2/LeakyReluMaximum/model/comment_time_layers/dense_2/LeakyRelu/mul)model/comment_time_layers/dense_2/BiasAdd*
T0
�
8mio_variable/comment_time_layers/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"comment_time_layers/dense_3/kernel*
shape
:@
�
8mio_variable/comment_time_layers/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"comment_time_layers/dense_3/kernel*
shape
:@
X
#Initializer_74/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_74/random_uniform/minConst*
valueB
 *����*
dtype0
N
!Initializer_74/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
+Initializer_74/random_uniform/RandomUniformRandomUniform#Initializer_74/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_74/random_uniform/subSub!Initializer_74/random_uniform/max!Initializer_74/random_uniform/min*
T0
�
!Initializer_74/random_uniform/mulMul+Initializer_74/random_uniform/RandomUniform!Initializer_74/random_uniform/sub*
T0
s
Initializer_74/random_uniformAdd!Initializer_74/random_uniform/mul!Initializer_74/random_uniform/min*
T0
�
	Assign_74Assign8mio_variable/comment_time_layers/dense_3/kernel/gradientInitializer_74/random_uniform*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/comment_time_layers/dense_3/kernel/gradient*
validate_shape(
�
6mio_variable/comment_time_layers/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" comment_time_layers/dense_3/bias*
shape:
�
6mio_variable/comment_time_layers/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*/
	container" comment_time_layers/dense_3/bias
E
Initializer_75/zerosConst*
dtype0*
valueB*    
�
	Assign_75Assign6mio_variable/comment_time_layers/dense_3/bias/gradientInitializer_75/zeros*
use_locking(*
T0*I
_class?
=;loc:@mio_variable/comment_time_layers/dense_3/bias/gradient*
validate_shape(
�
(model/comment_time_layers/dense_3/MatMulMatMul+model/comment_time_layers/dense_2/LeakyRelu8mio_variable/comment_time_layers/dense_3/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
)model/comment_time_layers/dense_3/BiasAddBiasAdd(model/comment_time_layers/dense_3/MatMul6mio_variable/comment_time_layers/dense_3/bias/variable*
T0*
data_formatNHWC
�
;mio_variable/long_view_wiz_cmt_layers/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*4
	container'%long_view_wiz_cmt_layers/dense/kernel*
shape:
��
�
;mio_variable/long_view_wiz_cmt_layers/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*4
	container'%long_view_wiz_cmt_layers/dense/kernel*
shape:
��
X
#Initializer_76/random_uniform/shapeConst*
valueB"0     *
dtype0
N
!Initializer_76/random_uniform/minConst*
dtype0*
valueB
 *ܨ��
N
!Initializer_76/random_uniform/maxConst*
dtype0*
valueB
 *ܨ�=
�
+Initializer_76/random_uniform/RandomUniformRandomUniform#Initializer_76/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_76/random_uniform/subSub!Initializer_76/random_uniform/max!Initializer_76/random_uniform/min*
T0
�
!Initializer_76/random_uniform/mulMul+Initializer_76/random_uniform/RandomUniform!Initializer_76/random_uniform/sub*
T0
s
Initializer_76/random_uniformAdd!Initializer_76/random_uniform/mul!Initializer_76/random_uniform/min*
T0
�
	Assign_76Assign;mio_variable/long_view_wiz_cmt_layers/dense/kernel/gradientInitializer_76/random_uniform*
use_locking(*
T0*N
_classD
B@loc:@mio_variable/long_view_wiz_cmt_layers/dense/kernel/gradient*
validate_shape(
�
9mio_variable/long_view_wiz_cmt_layers/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:�*2
	container%#long_view_wiz_cmt_layers/dense/bias
�
9mio_variable/long_view_wiz_cmt_layers/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#long_view_wiz_cmt_layers/dense/bias*
shape:�
F
Initializer_77/zerosConst*
valueB�*    *
dtype0
�
	Assign_77Assign9mio_variable/long_view_wiz_cmt_layers/dense/bias/gradientInitializer_77/zeros*
T0*L
_classB
@>loc:@mio_variable/long_view_wiz_cmt_layers/dense/bias/gradient*
validate_shape(*
use_locking(
�
+model/long_view_wiz_cmt_layers/dense/MatMulMatMul	concat_11;mio_variable/long_view_wiz_cmt_layers/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
,model/long_view_wiz_cmt_layers/dense/BiasAddBiasAdd+model/long_view_wiz_cmt_layers/dense/MatMul9mio_variable/long_view_wiz_cmt_layers/dense/bias/variable*
T0*
data_formatNHWC
a
4model/long_view_wiz_cmt_layers/dense/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
2model/long_view_wiz_cmt_layers/dense/LeakyRelu/mulMul4model/long_view_wiz_cmt_layers/dense/LeakyRelu/alpha,model/long_view_wiz_cmt_layers/dense/BiasAdd*
T0
�
.model/long_view_wiz_cmt_layers/dense/LeakyReluMaximum2model/long_view_wiz_cmt_layers/dense/LeakyRelu/mul,model/long_view_wiz_cmt_layers/dense/BiasAdd*
T0
�
=mio_variable/long_view_wiz_cmt_layers/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*6
	container)'long_view_wiz_cmt_layers/dense_1/kernel*
shape:
��
�
=mio_variable/long_view_wiz_cmt_layers/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*6
	container)'long_view_wiz_cmt_layers/dense_1/kernel*
shape:
��
X
#Initializer_78/random_uniform/shapeConst*
valueB"   �   *
dtype0
N
!Initializer_78/random_uniform/minConst*
valueB
 *   �*
dtype0
N
!Initializer_78/random_uniform/maxConst*
valueB
 *   >*
dtype0
�
+Initializer_78/random_uniform/RandomUniformRandomUniform#Initializer_78/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_78/random_uniform/subSub!Initializer_78/random_uniform/max!Initializer_78/random_uniform/min*
T0
�
!Initializer_78/random_uniform/mulMul+Initializer_78/random_uniform/RandomUniform!Initializer_78/random_uniform/sub*
T0
s
Initializer_78/random_uniformAdd!Initializer_78/random_uniform/mul!Initializer_78/random_uniform/min*
T0
�
	Assign_78Assign=mio_variable/long_view_wiz_cmt_layers/dense_1/kernel/gradientInitializer_78/random_uniform*
T0*P
_classF
DBloc:@mio_variable/long_view_wiz_cmt_layers/dense_1/kernel/gradient*
validate_shape(*
use_locking(
�
;mio_variable/long_view_wiz_cmt_layers/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:�*4
	container'%long_view_wiz_cmt_layers/dense_1/bias
�
;mio_variable/long_view_wiz_cmt_layers/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*4
	container'%long_view_wiz_cmt_layers/dense_1/bias*
shape:�
F
Initializer_79/zerosConst*
valueB�*    *
dtype0
�
	Assign_79Assign;mio_variable/long_view_wiz_cmt_layers/dense_1/bias/gradientInitializer_79/zeros*
T0*N
_classD
B@loc:@mio_variable/long_view_wiz_cmt_layers/dense_1/bias/gradient*
validate_shape(*
use_locking(
�
-model/long_view_wiz_cmt_layers/dense_1/MatMulMatMul.model/long_view_wiz_cmt_layers/dense/LeakyRelu=mio_variable/long_view_wiz_cmt_layers/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
.model/long_view_wiz_cmt_layers/dense_1/BiasAddBiasAdd-model/long_view_wiz_cmt_layers/dense_1/MatMul;mio_variable/long_view_wiz_cmt_layers/dense_1/bias/variable*
T0*
data_formatNHWC
c
6model/long_view_wiz_cmt_layers/dense_1/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
4model/long_view_wiz_cmt_layers/dense_1/LeakyRelu/mulMul6model/long_view_wiz_cmt_layers/dense_1/LeakyRelu/alpha.model/long_view_wiz_cmt_layers/dense_1/BiasAdd*
T0
�
0model/long_view_wiz_cmt_layers/dense_1/LeakyReluMaximum4model/long_view_wiz_cmt_layers/dense_1/LeakyRelu/mul.model/long_view_wiz_cmt_layers/dense_1/BiasAdd*
T0
�
=mio_variable/long_view_wiz_cmt_layers/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�@*6
	container)'long_view_wiz_cmt_layers/dense_2/kernel
�
=mio_variable/long_view_wiz_cmt_layers/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�@*6
	container)'long_view_wiz_cmt_layers/dense_2/kernel
X
#Initializer_80/random_uniform/shapeConst*
dtype0*
valueB"�   @   
N
!Initializer_80/random_uniform/minConst*
valueB
 *�5�*
dtype0
N
!Initializer_80/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
+Initializer_80/random_uniform/RandomUniformRandomUniform#Initializer_80/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_80/random_uniform/subSub!Initializer_80/random_uniform/max!Initializer_80/random_uniform/min*
T0
�
!Initializer_80/random_uniform/mulMul+Initializer_80/random_uniform/RandomUniform!Initializer_80/random_uniform/sub*
T0
s
Initializer_80/random_uniformAdd!Initializer_80/random_uniform/mul!Initializer_80/random_uniform/min*
T0
�
	Assign_80Assign=mio_variable/long_view_wiz_cmt_layers/dense_2/kernel/gradientInitializer_80/random_uniform*
use_locking(*
T0*P
_classF
DBloc:@mio_variable/long_view_wiz_cmt_layers/dense_2/kernel/gradient*
validate_shape(
�
;mio_variable/long_view_wiz_cmt_layers/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*4
	container'%long_view_wiz_cmt_layers/dense_2/bias*
shape:@
�
;mio_variable/long_view_wiz_cmt_layers/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*4
	container'%long_view_wiz_cmt_layers/dense_2/bias*
shape:@
E
Initializer_81/zerosConst*
valueB@*    *
dtype0
�
	Assign_81Assign;mio_variable/long_view_wiz_cmt_layers/dense_2/bias/gradientInitializer_81/zeros*
validate_shape(*
use_locking(*
T0*N
_classD
B@loc:@mio_variable/long_view_wiz_cmt_layers/dense_2/bias/gradient
�
-model/long_view_wiz_cmt_layers/dense_2/MatMulMatMul0model/long_view_wiz_cmt_layers/dense_1/LeakyRelu=mio_variable/long_view_wiz_cmt_layers/dense_2/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
.model/long_view_wiz_cmt_layers/dense_2/BiasAddBiasAdd-model/long_view_wiz_cmt_layers/dense_2/MatMul;mio_variable/long_view_wiz_cmt_layers/dense_2/bias/variable*
data_formatNHWC*
T0
c
6model/long_view_wiz_cmt_layers/dense_2/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
4model/long_view_wiz_cmt_layers/dense_2/LeakyRelu/mulMul6model/long_view_wiz_cmt_layers/dense_2/LeakyRelu/alpha.model/long_view_wiz_cmt_layers/dense_2/BiasAdd*
T0
�
0model/long_view_wiz_cmt_layers/dense_2/LeakyReluMaximum4model/long_view_wiz_cmt_layers/dense_2/LeakyRelu/mul.model/long_view_wiz_cmt_layers/dense_2/BiasAdd*
T0
�
=mio_variable/long_view_wiz_cmt_layers/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*6
	container)'long_view_wiz_cmt_layers/dense_3/kernel*
shape
:@
�
=mio_variable/long_view_wiz_cmt_layers/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*6
	container)'long_view_wiz_cmt_layers/dense_3/kernel
X
#Initializer_82/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_82/random_uniform/minConst*
dtype0*
valueB
 *����
N
!Initializer_82/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
+Initializer_82/random_uniform/RandomUniformRandomUniform#Initializer_82/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_82/random_uniform/subSub!Initializer_82/random_uniform/max!Initializer_82/random_uniform/min*
T0
�
!Initializer_82/random_uniform/mulMul+Initializer_82/random_uniform/RandomUniform!Initializer_82/random_uniform/sub*
T0
s
Initializer_82/random_uniformAdd!Initializer_82/random_uniform/mul!Initializer_82/random_uniform/min*
T0
�
	Assign_82Assign=mio_variable/long_view_wiz_cmt_layers/dense_3/kernel/gradientInitializer_82/random_uniform*
validate_shape(*
use_locking(*
T0*P
_classF
DBloc:@mio_variable/long_view_wiz_cmt_layers/dense_3/kernel/gradient
�
;mio_variable/long_view_wiz_cmt_layers/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*4
	container'%long_view_wiz_cmt_layers/dense_3/bias*
shape:
�
;mio_variable/long_view_wiz_cmt_layers/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*4
	container'%long_view_wiz_cmt_layers/dense_3/bias*
shape:
E
Initializer_83/zerosConst*
valueB*    *
dtype0
�
	Assign_83Assign;mio_variable/long_view_wiz_cmt_layers/dense_3/bias/gradientInitializer_83/zeros*
validate_shape(*
use_locking(*
T0*N
_classD
B@loc:@mio_variable/long_view_wiz_cmt_layers/dense_3/bias/gradient
�
-model/long_view_wiz_cmt_layers/dense_3/MatMulMatMul0model/long_view_wiz_cmt_layers/dense_2/LeakyRelu=mio_variable/long_view_wiz_cmt_layers/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
.model/long_view_wiz_cmt_layers/dense_3/BiasAddBiasAdd-model/long_view_wiz_cmt_layers/dense_3/MatMul;mio_variable/long_view_wiz_cmt_layers/dense_3/bias/variable*
T0*
data_formatNHWC
�
>mio_variable/long_view_wiz_no_cmt_layers/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
��*7
	container*(long_view_wiz_no_cmt_layers/dense/kernel
�
>mio_variable/long_view_wiz_no_cmt_layers/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(long_view_wiz_no_cmt_layers/dense/kernel*
shape:
��
X
#Initializer_84/random_uniform/shapeConst*
valueB"0     *
dtype0
N
!Initializer_84/random_uniform/minConst*
valueB
 *ܨ��*
dtype0
N
!Initializer_84/random_uniform/maxConst*
valueB
 *ܨ�=*
dtype0
�
+Initializer_84/random_uniform/RandomUniformRandomUniform#Initializer_84/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_84/random_uniform/subSub!Initializer_84/random_uniform/max!Initializer_84/random_uniform/min*
T0
�
!Initializer_84/random_uniform/mulMul+Initializer_84/random_uniform/RandomUniform!Initializer_84/random_uniform/sub*
T0
s
Initializer_84/random_uniformAdd!Initializer_84/random_uniform/mul!Initializer_84/random_uniform/min*
T0
�
	Assign_84Assign>mio_variable/long_view_wiz_no_cmt_layers/dense/kernel/gradientInitializer_84/random_uniform*
use_locking(*
T0*Q
_classG
ECloc:@mio_variable/long_view_wiz_no_cmt_layers/dense/kernel/gradient*
validate_shape(
�
<mio_variable/long_view_wiz_no_cmt_layers/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&long_view_wiz_no_cmt_layers/dense/bias*
shape:�
�
<mio_variable/long_view_wiz_no_cmt_layers/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&long_view_wiz_no_cmt_layers/dense/bias*
shape:�
F
Initializer_85/zerosConst*
valueB�*    *
dtype0
�
	Assign_85Assign<mio_variable/long_view_wiz_no_cmt_layers/dense/bias/gradientInitializer_85/zeros*
use_locking(*
T0*O
_classE
CAloc:@mio_variable/long_view_wiz_no_cmt_layers/dense/bias/gradient*
validate_shape(
�
.model/long_view_wiz_no_cmt_layers/dense/MatMulMatMul	concat_11>mio_variable/long_view_wiz_no_cmt_layers/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
/model/long_view_wiz_no_cmt_layers/dense/BiasAddBiasAdd.model/long_view_wiz_no_cmt_layers/dense/MatMul<mio_variable/long_view_wiz_no_cmt_layers/dense/bias/variable*
T0*
data_formatNHWC
d
7model/long_view_wiz_no_cmt_layers/dense/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
5model/long_view_wiz_no_cmt_layers/dense/LeakyRelu/mulMul7model/long_view_wiz_no_cmt_layers/dense/LeakyRelu/alpha/model/long_view_wiz_no_cmt_layers/dense/BiasAdd*
T0
�
1model/long_view_wiz_no_cmt_layers/dense/LeakyReluMaximum5model/long_view_wiz_no_cmt_layers/dense/LeakyRelu/mul/model/long_view_wiz_no_cmt_layers/dense/BiasAdd*
T0
�
@mio_variable/long_view_wiz_no_cmt_layers/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
��*9
	container,*long_view_wiz_no_cmt_layers/dense_1/kernel
�
@mio_variable/long_view_wiz_no_cmt_layers/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*long_view_wiz_no_cmt_layers/dense_1/kernel*
shape:
��
X
#Initializer_86/random_uniform/shapeConst*
valueB"   �   *
dtype0
N
!Initializer_86/random_uniform/minConst*
valueB
 *   �*
dtype0
N
!Initializer_86/random_uniform/maxConst*
dtype0*
valueB
 *   >
�
+Initializer_86/random_uniform/RandomUniformRandomUniform#Initializer_86/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_86/random_uniform/subSub!Initializer_86/random_uniform/max!Initializer_86/random_uniform/min*
T0
�
!Initializer_86/random_uniform/mulMul+Initializer_86/random_uniform/RandomUniform!Initializer_86/random_uniform/sub*
T0
s
Initializer_86/random_uniformAdd!Initializer_86/random_uniform/mul!Initializer_86/random_uniform/min*
T0
�
	Assign_86Assign@mio_variable/long_view_wiz_no_cmt_layers/dense_1/kernel/gradientInitializer_86/random_uniform*
use_locking(*
T0*S
_classI
GEloc:@mio_variable/long_view_wiz_no_cmt_layers/dense_1/kernel/gradient*
validate_shape(
�
>mio_variable/long_view_wiz_no_cmt_layers/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(long_view_wiz_no_cmt_layers/dense_1/bias*
shape:�
�
>mio_variable/long_view_wiz_no_cmt_layers/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(long_view_wiz_no_cmt_layers/dense_1/bias*
shape:�
F
Initializer_87/zerosConst*
valueB�*    *
dtype0
�
	Assign_87Assign>mio_variable/long_view_wiz_no_cmt_layers/dense_1/bias/gradientInitializer_87/zeros*
use_locking(*
T0*Q
_classG
ECloc:@mio_variable/long_view_wiz_no_cmt_layers/dense_1/bias/gradient*
validate_shape(
�
0model/long_view_wiz_no_cmt_layers/dense_1/MatMulMatMul1model/long_view_wiz_no_cmt_layers/dense/LeakyRelu@mio_variable/long_view_wiz_no_cmt_layers/dense_1/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
1model/long_view_wiz_no_cmt_layers/dense_1/BiasAddBiasAdd0model/long_view_wiz_no_cmt_layers/dense_1/MatMul>mio_variable/long_view_wiz_no_cmt_layers/dense_1/bias/variable*
T0*
data_formatNHWC
f
9model/long_view_wiz_no_cmt_layers/dense_1/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
7model/long_view_wiz_no_cmt_layers/dense_1/LeakyRelu/mulMul9model/long_view_wiz_no_cmt_layers/dense_1/LeakyRelu/alpha1model/long_view_wiz_no_cmt_layers/dense_1/BiasAdd*
T0
�
3model/long_view_wiz_no_cmt_layers/dense_1/LeakyReluMaximum7model/long_view_wiz_no_cmt_layers/dense_1/LeakyRelu/mul1model/long_view_wiz_no_cmt_layers/dense_1/BiasAdd*
T0
�
@mio_variable/long_view_wiz_no_cmt_layers/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*long_view_wiz_no_cmt_layers/dense_2/kernel*
shape:	�@
�
@mio_variable/long_view_wiz_no_cmt_layers/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*long_view_wiz_no_cmt_layers/dense_2/kernel*
shape:	�@
X
#Initializer_88/random_uniform/shapeConst*
valueB"�   @   *
dtype0
N
!Initializer_88/random_uniform/minConst*
valueB
 *�5�*
dtype0
N
!Initializer_88/random_uniform/maxConst*
dtype0*
valueB
 *�5>
�
+Initializer_88/random_uniform/RandomUniformRandomUniform#Initializer_88/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_88/random_uniform/subSub!Initializer_88/random_uniform/max!Initializer_88/random_uniform/min*
T0
�
!Initializer_88/random_uniform/mulMul+Initializer_88/random_uniform/RandomUniform!Initializer_88/random_uniform/sub*
T0
s
Initializer_88/random_uniformAdd!Initializer_88/random_uniform/mul!Initializer_88/random_uniform/min*
T0
�
	Assign_88Assign@mio_variable/long_view_wiz_no_cmt_layers/dense_2/kernel/gradientInitializer_88/random_uniform*
validate_shape(*
use_locking(*
T0*S
_classI
GEloc:@mio_variable/long_view_wiz_no_cmt_layers/dense_2/kernel/gradient
�
>mio_variable/long_view_wiz_no_cmt_layers/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*7
	container*(long_view_wiz_no_cmt_layers/dense_2/bias
�
>mio_variable/long_view_wiz_no_cmt_layers/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(long_view_wiz_no_cmt_layers/dense_2/bias*
shape:@
E
Initializer_89/zerosConst*
valueB@*    *
dtype0
�
	Assign_89Assign>mio_variable/long_view_wiz_no_cmt_layers/dense_2/bias/gradientInitializer_89/zeros*
use_locking(*
T0*Q
_classG
ECloc:@mio_variable/long_view_wiz_no_cmt_layers/dense_2/bias/gradient*
validate_shape(
�
0model/long_view_wiz_no_cmt_layers/dense_2/MatMulMatMul3model/long_view_wiz_no_cmt_layers/dense_1/LeakyRelu@mio_variable/long_view_wiz_no_cmt_layers/dense_2/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
1model/long_view_wiz_no_cmt_layers/dense_2/BiasAddBiasAdd0model/long_view_wiz_no_cmt_layers/dense_2/MatMul>mio_variable/long_view_wiz_no_cmt_layers/dense_2/bias/variable*
T0*
data_formatNHWC
f
9model/long_view_wiz_no_cmt_layers/dense_2/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
7model/long_view_wiz_no_cmt_layers/dense_2/LeakyRelu/mulMul9model/long_view_wiz_no_cmt_layers/dense_2/LeakyRelu/alpha1model/long_view_wiz_no_cmt_layers/dense_2/BiasAdd*
T0
�
3model/long_view_wiz_no_cmt_layers/dense_2/LeakyReluMaximum7model/long_view_wiz_no_cmt_layers/dense_2/LeakyRelu/mul1model/long_view_wiz_no_cmt_layers/dense_2/BiasAdd*
T0
�
@mio_variable/long_view_wiz_no_cmt_layers/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*long_view_wiz_no_cmt_layers/dense_3/kernel*
shape
:@
�
@mio_variable/long_view_wiz_no_cmt_layers/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*long_view_wiz_no_cmt_layers/dense_3/kernel*
shape
:@
X
#Initializer_90/random_uniform/shapeConst*
dtype0*
valueB"@      
N
!Initializer_90/random_uniform/minConst*
dtype0*
valueB
 *����
N
!Initializer_90/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
+Initializer_90/random_uniform/RandomUniformRandomUniform#Initializer_90/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_90/random_uniform/subSub!Initializer_90/random_uniform/max!Initializer_90/random_uniform/min*
T0
�
!Initializer_90/random_uniform/mulMul+Initializer_90/random_uniform/RandomUniform!Initializer_90/random_uniform/sub*
T0
s
Initializer_90/random_uniformAdd!Initializer_90/random_uniform/mul!Initializer_90/random_uniform/min*
T0
�
	Assign_90Assign@mio_variable/long_view_wiz_no_cmt_layers/dense_3/kernel/gradientInitializer_90/random_uniform*
validate_shape(*
use_locking(*
T0*S
_classI
GEloc:@mio_variable/long_view_wiz_no_cmt_layers/dense_3/kernel/gradient
�
>mio_variable/long_view_wiz_no_cmt_layers/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(long_view_wiz_no_cmt_layers/dense_3/bias*
shape:
�
>mio_variable/long_view_wiz_no_cmt_layers/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*7
	container*(long_view_wiz_no_cmt_layers/dense_3/bias
E
Initializer_91/zerosConst*
valueB*    *
dtype0
�
	Assign_91Assign>mio_variable/long_view_wiz_no_cmt_layers/dense_3/bias/gradientInitializer_91/zeros*
validate_shape(*
use_locking(*
T0*Q
_classG
ECloc:@mio_variable/long_view_wiz_no_cmt_layers/dense_3/bias/gradient
�
0model/long_view_wiz_no_cmt_layers/dense_3/MatMulMatMul3model/long_view_wiz_no_cmt_layers/dense_2/LeakyRelu@mio_variable/long_view_wiz_no_cmt_layers/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
1model/long_view_wiz_no_cmt_layers/dense_3/BiasAddBiasAdd0model/long_view_wiz_no_cmt_layers/dense_3/MatMul>mio_variable/long_view_wiz_no_cmt_layers/dense_3/bias/variable*
T0*
data_formatNHWC
�
?mio_variable/comment_top_net_cluster_gate/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*8
	container+)comment_top_net_cluster_gate/dense/kernel*
shape:	�
�
?mio_variable/comment_top_net_cluster_gate/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*8
	container+)comment_top_net_cluster_gate/dense/kernel*
shape:	�
X
#Initializer_92/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_92/random_uniform/minConst*
valueB
 *Iv�*
dtype0
N
!Initializer_92/random_uniform/maxConst*
valueB
 *Iv>*
dtype0
�
+Initializer_92/random_uniform/RandomUniformRandomUniform#Initializer_92/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_92/random_uniform/subSub!Initializer_92/random_uniform/max!Initializer_92/random_uniform/min*
T0
�
!Initializer_92/random_uniform/mulMul+Initializer_92/random_uniform/RandomUniform!Initializer_92/random_uniform/sub*
T0
s
Initializer_92/random_uniformAdd!Initializer_92/random_uniform/mul!Initializer_92/random_uniform/min*
T0
�
	Assign_92Assign?mio_variable/comment_top_net_cluster_gate/dense/kernel/gradientInitializer_92/random_uniform*
use_locking(*
T0*R
_classH
FDloc:@mio_variable/comment_top_net_cluster_gate/dense/kernel/gradient*
validate_shape(
�
=mio_variable/comment_top_net_cluster_gate/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*6
	container)'comment_top_net_cluster_gate/dense/bias*
shape:�
�
=mio_variable/comment_top_net_cluster_gate/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*6
	container)'comment_top_net_cluster_gate/dense/bias*
shape:�
F
Initializer_93/zerosConst*
valueB�*    *
dtype0
�
	Assign_93Assign=mio_variable/comment_top_net_cluster_gate/dense/bias/gradientInitializer_93/zeros*
validate_shape(*
use_locking(*
T0*P
_classF
DBloc:@mio_variable/comment_top_net_cluster_gate/dense/bias/gradient
�
/model/comment_top_net_cluster_gate/dense/MatMulMatMul	concat_12?mio_variable/comment_top_net_cluster_gate/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
0model/comment_top_net_cluster_gate/dense/BiasAddBiasAdd/model/comment_top_net_cluster_gate/dense/MatMul=mio_variable/comment_top_net_cluster_gate/dense/bias/variable*
T0*
data_formatNHWC
p
-model/comment_top_net_cluster_gate/dense/ReluRelu0model/comment_top_net_cluster_gate/dense/BiasAdd*
T0
�
Amio_variable/comment_top_net_cluster_gate/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+comment_top_net_cluster_gate/dense_1/kernel*
shape:
��
�
Amio_variable/comment_top_net_cluster_gate/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
��*:
	container-+comment_top_net_cluster_gate/dense_1/kernel
X
#Initializer_94/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_94/random_uniform/minConst*
valueB
 *׳ݽ*
dtype0
N
!Initializer_94/random_uniform/maxConst*
valueB
 *׳�=*
dtype0
�
+Initializer_94/random_uniform/RandomUniformRandomUniform#Initializer_94/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_94/random_uniform/subSub!Initializer_94/random_uniform/max!Initializer_94/random_uniform/min*
T0
�
!Initializer_94/random_uniform/mulMul+Initializer_94/random_uniform/RandomUniform!Initializer_94/random_uniform/sub*
T0
s
Initializer_94/random_uniformAdd!Initializer_94/random_uniform/mul!Initializer_94/random_uniform/min*
T0
�
	Assign_94AssignAmio_variable/comment_top_net_cluster_gate/dense_1/kernel/gradientInitializer_94/random_uniform*
use_locking(*
T0*T
_classJ
HFloc:@mio_variable/comment_top_net_cluster_gate/dense_1/kernel/gradient*
validate_shape(
�
?mio_variable/comment_top_net_cluster_gate/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*8
	container+)comment_top_net_cluster_gate/dense_1/bias*
shape:�
�
?mio_variable/comment_top_net_cluster_gate/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*8
	container+)comment_top_net_cluster_gate/dense_1/bias*
shape:�
F
Initializer_95/zerosConst*
valueB�*    *
dtype0
�
	Assign_95Assign?mio_variable/comment_top_net_cluster_gate/dense_1/bias/gradientInitializer_95/zeros*
validate_shape(*
use_locking(*
T0*R
_classH
FDloc:@mio_variable/comment_top_net_cluster_gate/dense_1/bias/gradient
�
1model/comment_top_net_cluster_gate/dense_1/MatMulMatMul-model/comment_top_net_cluster_gate/dense/ReluAmio_variable/comment_top_net_cluster_gate/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
2model/comment_top_net_cluster_gate/dense_1/BiasAddBiasAdd1model/comment_top_net_cluster_gate/dense_1/MatMul?mio_variable/comment_top_net_cluster_gate/dense_1/bias/variable*
data_formatNHWC*
T0
z
2model/comment_top_net_cluster_gate/dense_1/SigmoidSigmoid2model/comment_top_net_cluster_gate/dense_1/BiasAdd*
T0
U
(model/comment_top_net_cluster_gate/mul/xConst*
valueB
 *   @*
dtype0
�
&model/comment_top_net_cluster_gate/mulMul(model/comment_top_net_cluster_gate/mul/x2model/comment_top_net_cluster_gate/dense_1/Sigmoid*
T0
�
2mio_variable/comment_top_net/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*+
	containercomment_top_net/dense/kernel*
shape:
��
�
2mio_variable/comment_top_net/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*+
	containercomment_top_net/dense/kernel*
shape:
��
X
#Initializer_96/random_uniform/shapeConst*
valueB"0     *
dtype0
N
!Initializer_96/random_uniform/minConst*
valueB
 *ܨ��*
dtype0
N
!Initializer_96/random_uniform/maxConst*
valueB
 *ܨ�=*
dtype0
�
+Initializer_96/random_uniform/RandomUniformRandomUniform#Initializer_96/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_96/random_uniform/subSub!Initializer_96/random_uniform/max!Initializer_96/random_uniform/min*
T0
�
!Initializer_96/random_uniform/mulMul+Initializer_96/random_uniform/RandomUniform!Initializer_96/random_uniform/sub*
T0
s
Initializer_96/random_uniformAdd!Initializer_96/random_uniform/mul!Initializer_96/random_uniform/min*
T0
�
	Assign_96Assign2mio_variable/comment_top_net/dense/kernel/gradientInitializer_96/random_uniform*
validate_shape(*
use_locking(*
T0*E
_class;
97loc:@mio_variable/comment_top_net/dense/kernel/gradient
�
0mio_variable/comment_top_net/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*)
	containercomment_top_net/dense/bias*
shape:�
�
0mio_variable/comment_top_net/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:�*)
	containercomment_top_net/dense/bias
F
Initializer_97/zerosConst*
valueB�*    *
dtype0
�
	Assign_97Assign0mio_variable/comment_top_net/dense/bias/gradientInitializer_97/zeros*
validate_shape(*
use_locking(*
T0*C
_class9
75loc:@mio_variable/comment_top_net/dense/bias/gradient
�
"model/comment_top_net/dense/MatMulMatMul	concat_112mio_variable/comment_top_net/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
#model/comment_top_net/dense/BiasAddBiasAdd"model/comment_top_net/dense/MatMul0mio_variable/comment_top_net/dense/bias/variable*
T0*
data_formatNHWC
X
+model/comment_top_net/dense/LeakyRelu/alphaConst*
dtype0*
valueB
 *��L>
�
)model/comment_top_net/dense/LeakyRelu/mulMul+model/comment_top_net/dense/LeakyRelu/alpha#model/comment_top_net/dense/BiasAdd*
T0
�
%model/comment_top_net/dense/LeakyReluMaximum)model/comment_top_net/dense/LeakyRelu/mul#model/comment_top_net/dense/BiasAdd*
T0
x
model/comment_top_net/MulMul%model/comment_top_net/dense/LeakyRelu&model/comment_top_net_cluster_gate/mul*
T0
�
4mio_variable/comment_top_net/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*-
	container comment_top_net/dense_1/kernel*
shape:
��
�
4mio_variable/comment_top_net/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*-
	container comment_top_net/dense_1/kernel*
shape:
��
X
#Initializer_98/random_uniform/shapeConst*
valueB"   �   *
dtype0
N
!Initializer_98/random_uniform/minConst*
dtype0*
valueB
 *   �
N
!Initializer_98/random_uniform/maxConst*
valueB
 *   >*
dtype0
�
+Initializer_98/random_uniform/RandomUniformRandomUniform#Initializer_98/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_98/random_uniform/subSub!Initializer_98/random_uniform/max!Initializer_98/random_uniform/min*
T0
�
!Initializer_98/random_uniform/mulMul+Initializer_98/random_uniform/RandomUniform!Initializer_98/random_uniform/sub*
T0
s
Initializer_98/random_uniformAdd!Initializer_98/random_uniform/mul!Initializer_98/random_uniform/min*
T0
�
	Assign_98Assign4mio_variable/comment_top_net/dense_1/kernel/gradientInitializer_98/random_uniform*
use_locking(*
T0*G
_class=
;9loc:@mio_variable/comment_top_net/dense_1/kernel/gradient*
validate_shape(
�
2mio_variable/comment_top_net/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*+
	containercomment_top_net/dense_1/bias*
shape:�
�
2mio_variable/comment_top_net/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*+
	containercomment_top_net/dense_1/bias*
shape:�
F
Initializer_99/zerosConst*
valueB�*    *
dtype0
�
	Assign_99Assign2mio_variable/comment_top_net/dense_1/bias/gradientInitializer_99/zeros*
validate_shape(*
use_locking(*
T0*E
_class;
97loc:@mio_variable/comment_top_net/dense_1/bias/gradient
�
$model/comment_top_net/dense_1/MatMulMatMulmodel/comment_top_net/Mul4mio_variable/comment_top_net/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
%model/comment_top_net/dense_1/BiasAddBiasAdd$model/comment_top_net/dense_1/MatMul2mio_variable/comment_top_net/dense_1/bias/variable*
T0*
data_formatNHWC
Z
-model/comment_top_net/dense_1/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
+model/comment_top_net/dense_1/LeakyRelu/mulMul-model/comment_top_net/dense_1/LeakyRelu/alpha%model/comment_top_net/dense_1/BiasAdd*
T0
�
'model/comment_top_net/dense_1/LeakyReluMaximum+model/comment_top_net/dense_1/LeakyRelu/mul%model/comment_top_net/dense_1/BiasAdd*
T0
�
Fmio_variable/uplift_comment_consume_depth_layers/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20uplift_comment_consume_depth_layers/dense/kernel*
shape:	�@
�
Fmio_variable/uplift_comment_consume_depth_layers/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20uplift_comment_consume_depth_layers/dense/kernel*
shape:	�@
Y
$Initializer_100/random_uniform/shapeConst*
valueB"�   @   *
dtype0
O
"Initializer_100/random_uniform/minConst*
valueB
 *�5�*
dtype0
O
"Initializer_100/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_100/random_uniform/RandomUniformRandomUniform$Initializer_100/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_100/random_uniform/subSub"Initializer_100/random_uniform/max"Initializer_100/random_uniform/min*
T0
�
"Initializer_100/random_uniform/mulMul,Initializer_100/random_uniform/RandomUniform"Initializer_100/random_uniform/sub*
T0
v
Initializer_100/random_uniformAdd"Initializer_100/random_uniform/mul"Initializer_100/random_uniform/min*
T0
�

Assign_100AssignFmio_variable/uplift_comment_consume_depth_layers/dense/kernel/gradientInitializer_100/random_uniform*
validate_shape(*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/uplift_comment_consume_depth_layers/dense/kernel/gradient
�
Dmio_variable/uplift_comment_consume_depth_layers/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*=
	container0.uplift_comment_consume_depth_layers/dense/bias
�
Dmio_variable/uplift_comment_consume_depth_layers/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.uplift_comment_consume_depth_layers/dense/bias*
shape:@
F
Initializer_101/zerosConst*
valueB@*    *
dtype0
�

Assign_101AssignDmio_variable/uplift_comment_consume_depth_layers/dense/bias/gradientInitializer_101/zeros*
T0*W
_classM
KIloc:@mio_variable/uplift_comment_consume_depth_layers/dense/bias/gradient*
validate_shape(*
use_locking(
�
6model/uplift_comment_consume_depth_layers/dense/MatMulMatMul'model/comment_top_net/dense_1/LeakyReluFmio_variable/uplift_comment_consume_depth_layers/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
7model/uplift_comment_consume_depth_layers/dense/BiasAddBiasAdd6model/uplift_comment_consume_depth_layers/dense/MatMulDmio_variable/uplift_comment_consume_depth_layers/dense/bias/variable*
data_formatNHWC*
T0
l
?model/uplift_comment_consume_depth_layers/dense/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
=model/uplift_comment_consume_depth_layers/dense/LeakyRelu/mulMul?model/uplift_comment_consume_depth_layers/dense/LeakyRelu/alpha7model/uplift_comment_consume_depth_layers/dense/BiasAdd*
T0
�
9model/uplift_comment_consume_depth_layers/dense/LeakyReluMaximum=model/uplift_comment_consume_depth_layers/dense/LeakyRelu/mul7model/uplift_comment_consume_depth_layers/dense/BiasAdd*
T0
�
Hmio_variable/uplift_comment_consume_depth_layers/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42uplift_comment_consume_depth_layers/dense_1/kernel*
shape
:@
�
Hmio_variable/uplift_comment_consume_depth_layers/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42uplift_comment_consume_depth_layers/dense_1/kernel*
shape
:@
Y
$Initializer_102/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_102/random_uniform/minConst*
valueB
 *����*
dtype0
O
"Initializer_102/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
,Initializer_102/random_uniform/RandomUniformRandomUniform$Initializer_102/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_102/random_uniform/subSub"Initializer_102/random_uniform/max"Initializer_102/random_uniform/min*
T0
�
"Initializer_102/random_uniform/mulMul,Initializer_102/random_uniform/RandomUniform"Initializer_102/random_uniform/sub*
T0
v
Initializer_102/random_uniformAdd"Initializer_102/random_uniform/mul"Initializer_102/random_uniform/min*
T0
�

Assign_102AssignHmio_variable/uplift_comment_consume_depth_layers/dense_1/kernel/gradientInitializer_102/random_uniform*
T0*[
_classQ
OMloc:@mio_variable/uplift_comment_consume_depth_layers/dense_1/kernel/gradient*
validate_shape(*
use_locking(
�
Fmio_variable/uplift_comment_consume_depth_layers/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20uplift_comment_consume_depth_layers/dense_1/bias*
shape:
�
Fmio_variable/uplift_comment_consume_depth_layers/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20uplift_comment_consume_depth_layers/dense_1/bias*
shape:
F
Initializer_103/zerosConst*
valueB*    *
dtype0
�

Assign_103AssignFmio_variable/uplift_comment_consume_depth_layers/dense_1/bias/gradientInitializer_103/zeros*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/uplift_comment_consume_depth_layers/dense_1/bias/gradient*
validate_shape(
�
8model/uplift_comment_consume_depth_layers/dense_1/MatMulMatMul9model/uplift_comment_consume_depth_layers/dense/LeakyReluHmio_variable/uplift_comment_consume_depth_layers/dense_1/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
9model/uplift_comment_consume_depth_layers/dense_1/BiasAddBiasAdd8model/uplift_comment_consume_depth_layers/dense_1/MatMulFmio_variable/uplift_comment_consume_depth_layers/dense_1/bias/variable*
T0*
data_formatNHWC
�
Fmio_variable/uplift_comment_stay_duration_layers/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20uplift_comment_stay_duration_layers/dense/kernel*
shape:	�@
�
Fmio_variable/uplift_comment_stay_duration_layers/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20uplift_comment_stay_duration_layers/dense/kernel*
shape:	�@
Y
$Initializer_104/random_uniform/shapeConst*
valueB"�   @   *
dtype0
O
"Initializer_104/random_uniform/minConst*
valueB
 *�5�*
dtype0
O
"Initializer_104/random_uniform/maxConst*
dtype0*
valueB
 *�5>
�
,Initializer_104/random_uniform/RandomUniformRandomUniform$Initializer_104/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_104/random_uniform/subSub"Initializer_104/random_uniform/max"Initializer_104/random_uniform/min*
T0
�
"Initializer_104/random_uniform/mulMul,Initializer_104/random_uniform/RandomUniform"Initializer_104/random_uniform/sub*
T0
v
Initializer_104/random_uniformAdd"Initializer_104/random_uniform/mul"Initializer_104/random_uniform/min*
T0
�

Assign_104AssignFmio_variable/uplift_comment_stay_duration_layers/dense/kernel/gradientInitializer_104/random_uniform*
T0*Y
_classO
MKloc:@mio_variable/uplift_comment_stay_duration_layers/dense/kernel/gradient*
validate_shape(*
use_locking(
�
Dmio_variable/uplift_comment_stay_duration_layers/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.uplift_comment_stay_duration_layers/dense/bias*
shape:@
�
Dmio_variable/uplift_comment_stay_duration_layers/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.uplift_comment_stay_duration_layers/dense/bias*
shape:@
F
Initializer_105/zerosConst*
valueB@*    *
dtype0
�

Assign_105AssignDmio_variable/uplift_comment_stay_duration_layers/dense/bias/gradientInitializer_105/zeros*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/uplift_comment_stay_duration_layers/dense/bias/gradient*
validate_shape(
�
6model/uplift_comment_stay_duration_layers/dense/MatMulMatMul'model/comment_top_net/dense_1/LeakyReluFmio_variable/uplift_comment_stay_duration_layers/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
7model/uplift_comment_stay_duration_layers/dense/BiasAddBiasAdd6model/uplift_comment_stay_duration_layers/dense/MatMulDmio_variable/uplift_comment_stay_duration_layers/dense/bias/variable*
data_formatNHWC*
T0
l
?model/uplift_comment_stay_duration_layers/dense/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
=model/uplift_comment_stay_duration_layers/dense/LeakyRelu/mulMul?model/uplift_comment_stay_duration_layers/dense/LeakyRelu/alpha7model/uplift_comment_stay_duration_layers/dense/BiasAdd*
T0
�
9model/uplift_comment_stay_duration_layers/dense/LeakyReluMaximum=model/uplift_comment_stay_duration_layers/dense/LeakyRelu/mul7model/uplift_comment_stay_duration_layers/dense/BiasAdd*
T0
�
Hmio_variable/uplift_comment_stay_duration_layers/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42uplift_comment_stay_duration_layers/dense_1/kernel*
shape
:@
�
Hmio_variable/uplift_comment_stay_duration_layers/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42uplift_comment_stay_duration_layers/dense_1/kernel*
shape
:@
Y
$Initializer_106/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_106/random_uniform/minConst*
valueB
 *����*
dtype0
O
"Initializer_106/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
,Initializer_106/random_uniform/RandomUniformRandomUniform$Initializer_106/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_106/random_uniform/subSub"Initializer_106/random_uniform/max"Initializer_106/random_uniform/min*
T0
�
"Initializer_106/random_uniform/mulMul,Initializer_106/random_uniform/RandomUniform"Initializer_106/random_uniform/sub*
T0
v
Initializer_106/random_uniformAdd"Initializer_106/random_uniform/mul"Initializer_106/random_uniform/min*
T0
�

Assign_106AssignHmio_variable/uplift_comment_stay_duration_layers/dense_1/kernel/gradientInitializer_106/random_uniform*
use_locking(*
T0*[
_classQ
OMloc:@mio_variable/uplift_comment_stay_duration_layers/dense_1/kernel/gradient*
validate_shape(
�
Fmio_variable/uplift_comment_stay_duration_layers/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20uplift_comment_stay_duration_layers/dense_1/bias*
shape:
�
Fmio_variable/uplift_comment_stay_duration_layers/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*?
	container20uplift_comment_stay_duration_layers/dense_1/bias
F
Initializer_107/zerosConst*
valueB*    *
dtype0
�

Assign_107AssignFmio_variable/uplift_comment_stay_duration_layers/dense_1/bias/gradientInitializer_107/zeros*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/uplift_comment_stay_duration_layers/dense_1/bias/gradient*
validate_shape(
�
8model/uplift_comment_stay_duration_layers/dense_1/MatMulMatMul9model/uplift_comment_stay_duration_layers/dense/LeakyReluHmio_variable/uplift_comment_stay_duration_layers/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
9model/uplift_comment_stay_duration_layers/dense_1/BiasAddBiasAdd8model/uplift_comment_stay_duration_layers/dense_1/MatMulFmio_variable/uplift_comment_stay_duration_layers/dense_1/bias/variable*
T0*
data_formatNHWC
�
Lmio_variable/effective_read_comment_fresh_label_layers/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*E
	container86effective_read_comment_fresh_label_layers/dense/kernel*
shape:	�@
�
Lmio_variable/effective_read_comment_fresh_label_layers/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�@*E
	container86effective_read_comment_fresh_label_layers/dense/kernel
Y
$Initializer_108/random_uniform/shapeConst*
dtype0*
valueB"�   @   
O
"Initializer_108/random_uniform/minConst*
dtype0*
valueB
 *�5�
O
"Initializer_108/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_108/random_uniform/RandomUniformRandomUniform$Initializer_108/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_108/random_uniform/subSub"Initializer_108/random_uniform/max"Initializer_108/random_uniform/min*
T0
�
"Initializer_108/random_uniform/mulMul,Initializer_108/random_uniform/RandomUniform"Initializer_108/random_uniform/sub*
T0
v
Initializer_108/random_uniformAdd"Initializer_108/random_uniform/mul"Initializer_108/random_uniform/min*
T0
�

Assign_108AssignLmio_variable/effective_read_comment_fresh_label_layers/dense/kernel/gradientInitializer_108/random_uniform*
use_locking(*
T0*_
_classU
SQloc:@mio_variable/effective_read_comment_fresh_label_layers/dense/kernel/gradient*
validate_shape(
�
Jmio_variable/effective_read_comment_fresh_label_layers/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64effective_read_comment_fresh_label_layers/dense/bias*
shape:@
�
Jmio_variable/effective_read_comment_fresh_label_layers/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64effective_read_comment_fresh_label_layers/dense/bias*
shape:@
F
Initializer_109/zerosConst*
valueB@*    *
dtype0
�

Assign_109AssignJmio_variable/effective_read_comment_fresh_label_layers/dense/bias/gradientInitializer_109/zeros*
T0*]
_classS
QOloc:@mio_variable/effective_read_comment_fresh_label_layers/dense/bias/gradient*
validate_shape(*
use_locking(
�
<model/effective_read_comment_fresh_label_layers/dense/MatMulMatMul'model/comment_top_net/dense_1/LeakyReluLmio_variable/effective_read_comment_fresh_label_layers/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
=model/effective_read_comment_fresh_label_layers/dense/BiasAddBiasAdd<model/effective_read_comment_fresh_label_layers/dense/MatMulJmio_variable/effective_read_comment_fresh_label_layers/dense/bias/variable*
T0*
data_formatNHWC
r
Emodel/effective_read_comment_fresh_label_layers/dense/LeakyRelu/alphaConst*
dtype0*
valueB
 *��L>
�
Cmodel/effective_read_comment_fresh_label_layers/dense/LeakyRelu/mulMulEmodel/effective_read_comment_fresh_label_layers/dense/LeakyRelu/alpha=model/effective_read_comment_fresh_label_layers/dense/BiasAdd*
T0
�
?model/effective_read_comment_fresh_label_layers/dense/LeakyReluMaximumCmodel/effective_read_comment_fresh_label_layers/dense/LeakyRelu/mul=model/effective_read_comment_fresh_label_layers/dense/BiasAdd*
T0
�
Nmio_variable/effective_read_comment_fresh_label_layers/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*G
	container:8effective_read_comment_fresh_label_layers/dense_1/kernel*
shape
:@
�
Nmio_variable/effective_read_comment_fresh_label_layers/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*G
	container:8effective_read_comment_fresh_label_layers/dense_1/kernel*
shape
:@
Y
$Initializer_110/random_uniform/shapeConst*
dtype0*
valueB"@      
O
"Initializer_110/random_uniform/minConst*
valueB
 *����*
dtype0
O
"Initializer_110/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
,Initializer_110/random_uniform/RandomUniformRandomUniform$Initializer_110/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_110/random_uniform/subSub"Initializer_110/random_uniform/max"Initializer_110/random_uniform/min*
T0
�
"Initializer_110/random_uniform/mulMul,Initializer_110/random_uniform/RandomUniform"Initializer_110/random_uniform/sub*
T0
v
Initializer_110/random_uniformAdd"Initializer_110/random_uniform/mul"Initializer_110/random_uniform/min*
T0
�

Assign_110AssignNmio_variable/effective_read_comment_fresh_label_layers/dense_1/kernel/gradientInitializer_110/random_uniform*
validate_shape(*
use_locking(*
T0*a
_classW
USloc:@mio_variable/effective_read_comment_fresh_label_layers/dense_1/kernel/gradient
�
Lmio_variable/effective_read_comment_fresh_label_layers/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*E
	container86effective_read_comment_fresh_label_layers/dense_1/bias*
shape:
�
Lmio_variable/effective_read_comment_fresh_label_layers/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*E
	container86effective_read_comment_fresh_label_layers/dense_1/bias*
shape:
F
Initializer_111/zerosConst*
valueB*    *
dtype0
�

Assign_111AssignLmio_variable/effective_read_comment_fresh_label_layers/dense_1/bias/gradientInitializer_111/zeros*
use_locking(*
T0*_
_classU
SQloc:@mio_variable/effective_read_comment_fresh_label_layers/dense_1/bias/gradient*
validate_shape(
�
>model/effective_read_comment_fresh_label_layers/dense_1/MatMulMatMul?model/effective_read_comment_fresh_label_layers/dense/LeakyReluNmio_variable/effective_read_comment_fresh_label_layers/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
?model/effective_read_comment_fresh_label_layers/dense_1/BiasAddBiasAdd>model/effective_read_comment_fresh_label_layers/dense_1/MatMulLmio_variable/effective_read_comment_fresh_label_layers/dense_1/bias/variable*
T0*
data_formatNHWC
�
Jmio_variable/comment_unfold_score_logit_cluster_gate/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�*C
	container64comment_unfold_score_logit_cluster_gate/dense/kernel
�
Jmio_variable/comment_unfold_score_logit_cluster_gate/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64comment_unfold_score_logit_cluster_gate/dense/kernel*
shape:	�
Y
$Initializer_112/random_uniform/shapeConst*
valueB"   �   *
dtype0
O
"Initializer_112/random_uniform/minConst*
valueB
 *n�\�*
dtype0
O
"Initializer_112/random_uniform/maxConst*
valueB
 *n�\>*
dtype0
�
,Initializer_112/random_uniform/RandomUniformRandomUniform$Initializer_112/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_112/random_uniform/subSub"Initializer_112/random_uniform/max"Initializer_112/random_uniform/min*
T0
�
"Initializer_112/random_uniform/mulMul,Initializer_112/random_uniform/RandomUniform"Initializer_112/random_uniform/sub*
T0
v
Initializer_112/random_uniformAdd"Initializer_112/random_uniform/mul"Initializer_112/random_uniform/min*
T0
�

Assign_112AssignJmio_variable/comment_unfold_score_logit_cluster_gate/dense/kernel/gradientInitializer_112/random_uniform*
use_locking(*
T0*]
_classS
QOloc:@mio_variable/comment_unfold_score_logit_cluster_gate/dense/kernel/gradient*
validate_shape(
�
Hmio_variable/comment_unfold_score_logit_cluster_gate/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42comment_unfold_score_logit_cluster_gate/dense/bias*
shape:�
�
Hmio_variable/comment_unfold_score_logit_cluster_gate/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:�*A
	container42comment_unfold_score_logit_cluster_gate/dense/bias
G
Initializer_113/zerosConst*
valueB�*    *
dtype0
�

Assign_113AssignHmio_variable/comment_unfold_score_logit_cluster_gate/dense/bias/gradientInitializer_113/zeros*
T0*[
_classQ
OMloc:@mio_variable/comment_unfold_score_logit_cluster_gate/dense/bias/gradient*
validate_shape(*
use_locking(
�
:model/comment_unfold_score_logit_cluster_gate/dense/MatMulMatMul	concat_12Jmio_variable/comment_unfold_score_logit_cluster_gate/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
;model/comment_unfold_score_logit_cluster_gate/dense/BiasAddBiasAdd:model/comment_unfold_score_logit_cluster_gate/dense/MatMulHmio_variable/comment_unfold_score_logit_cluster_gate/dense/bias/variable*
T0*
data_formatNHWC
�
8model/comment_unfold_score_logit_cluster_gate/dense/ReluRelu;model/comment_unfold_score_logit_cluster_gate/dense/BiasAdd*
T0
�
Lmio_variable/comment_unfold_score_logit_cluster_gate/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*E
	container86comment_unfold_score_logit_cluster_gate/dense_1/kernel*
shape:
��
�
Lmio_variable/comment_unfold_score_logit_cluster_gate/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*E
	container86comment_unfold_score_logit_cluster_gate/dense_1/kernel*
shape:
��
Y
$Initializer_114/random_uniform/shapeConst*
valueB"�   �   *
dtype0
O
"Initializer_114/random_uniform/minConst*
valueB
 *q��*
dtype0
O
"Initializer_114/random_uniform/maxConst*
valueB
 *q�>*
dtype0
�
,Initializer_114/random_uniform/RandomUniformRandomUniform$Initializer_114/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_114/random_uniform/subSub"Initializer_114/random_uniform/max"Initializer_114/random_uniform/min*
T0
�
"Initializer_114/random_uniform/mulMul,Initializer_114/random_uniform/RandomUniform"Initializer_114/random_uniform/sub*
T0
v
Initializer_114/random_uniformAdd"Initializer_114/random_uniform/mul"Initializer_114/random_uniform/min*
T0
�

Assign_114AssignLmio_variable/comment_unfold_score_logit_cluster_gate/dense_1/kernel/gradientInitializer_114/random_uniform*
use_locking(*
T0*_
_classU
SQloc:@mio_variable/comment_unfold_score_logit_cluster_gate/dense_1/kernel/gradient*
validate_shape(
�
Jmio_variable/comment_unfold_score_logit_cluster_gate/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64comment_unfold_score_logit_cluster_gate/dense_1/bias*
shape:�
�
Jmio_variable/comment_unfold_score_logit_cluster_gate/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64comment_unfold_score_logit_cluster_gate/dense_1/bias*
shape:�
G
Initializer_115/zerosConst*
valueB�*    *
dtype0
�

Assign_115AssignJmio_variable/comment_unfold_score_logit_cluster_gate/dense_1/bias/gradientInitializer_115/zeros*
use_locking(*
T0*]
_classS
QOloc:@mio_variable/comment_unfold_score_logit_cluster_gate/dense_1/bias/gradient*
validate_shape(
�
<model/comment_unfold_score_logit_cluster_gate/dense_1/MatMulMatMul8model/comment_unfold_score_logit_cluster_gate/dense/ReluLmio_variable/comment_unfold_score_logit_cluster_gate/dense_1/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
=model/comment_unfold_score_logit_cluster_gate/dense_1/BiasAddBiasAdd<model/comment_unfold_score_logit_cluster_gate/dense_1/MatMulJmio_variable/comment_unfold_score_logit_cluster_gate/dense_1/bias/variable*
T0*
data_formatNHWC
�
=model/comment_unfold_score_logit_cluster_gate/dense_1/SigmoidSigmoid=model/comment_unfold_score_logit_cluster_gate/dense_1/BiasAdd*
T0
`
3model/comment_unfold_score_logit_cluster_gate/mul/xConst*
dtype0*
valueB
 *   @
�
1model/comment_unfold_score_logit_cluster_gate/mulMul3model/comment_unfold_score_logit_cluster_gate/mul/x=model/comment_unfold_score_logit_cluster_gate/dense_1/Sigmoid*
T0
�
Lmio_variable/comment_unfold_score_logit_cluster_gate/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*E
	container86comment_unfold_score_logit_cluster_gate/dense_2/kernel*
shape:	�@
�
Lmio_variable/comment_unfold_score_logit_cluster_gate/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*E
	container86comment_unfold_score_logit_cluster_gate/dense_2/kernel*
shape:	�@
Y
$Initializer_116/random_uniform/shapeConst*
valueB"�   @   *
dtype0
O
"Initializer_116/random_uniform/minConst*
valueB
 *�5�*
dtype0
O
"Initializer_116/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_116/random_uniform/RandomUniformRandomUniform$Initializer_116/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_116/random_uniform/subSub"Initializer_116/random_uniform/max"Initializer_116/random_uniform/min*
T0
�
"Initializer_116/random_uniform/mulMul,Initializer_116/random_uniform/RandomUniform"Initializer_116/random_uniform/sub*
T0
v
Initializer_116/random_uniformAdd"Initializer_116/random_uniform/mul"Initializer_116/random_uniform/min*
T0
�

Assign_116AssignLmio_variable/comment_unfold_score_logit_cluster_gate/dense_2/kernel/gradientInitializer_116/random_uniform*
use_locking(*
T0*_
_classU
SQloc:@mio_variable/comment_unfold_score_logit_cluster_gate/dense_2/kernel/gradient*
validate_shape(
�
Jmio_variable/comment_unfold_score_logit_cluster_gate/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64comment_unfold_score_logit_cluster_gate/dense_2/bias*
shape:@
�
Jmio_variable/comment_unfold_score_logit_cluster_gate/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64comment_unfold_score_logit_cluster_gate/dense_2/bias*
shape:@
F
Initializer_117/zerosConst*
dtype0*
valueB@*    
�

Assign_117AssignJmio_variable/comment_unfold_score_logit_cluster_gate/dense_2/bias/gradientInitializer_117/zeros*
use_locking(*
T0*]
_classS
QOloc:@mio_variable/comment_unfold_score_logit_cluster_gate/dense_2/bias/gradient*
validate_shape(
�
<model/comment_unfold_score_logit_cluster_gate/dense_2/MatMulMatMul=model/comment_unfold_score_logit_cluster_gate/dense_1/SigmoidLmio_variable/comment_unfold_score_logit_cluster_gate/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
=model/comment_unfold_score_logit_cluster_gate/dense_2/BiasAddBiasAdd<model/comment_unfold_score_logit_cluster_gate/dense_2/MatMulJmio_variable/comment_unfold_score_logit_cluster_gate/dense_2/bias/variable*
data_formatNHWC*
T0
�
:model/comment_unfold_score_logit_cluster_gate/dense_2/ReluRelu=model/comment_unfold_score_logit_cluster_gate/dense_2/BiasAdd*
T0
�
Lmio_variable/comment_unfold_score_logit_cluster_gate/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@@*E
	container86comment_unfold_score_logit_cluster_gate/dense_3/kernel
�
Lmio_variable/comment_unfold_score_logit_cluster_gate/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*E
	container86comment_unfold_score_logit_cluster_gate/dense_3/kernel*
shape
:@@
Y
$Initializer_118/random_uniform/shapeConst*
valueB"@   @   *
dtype0
O
"Initializer_118/random_uniform/minConst*
dtype0*
valueB
 *׳]�
O
"Initializer_118/random_uniform/maxConst*
valueB
 *׳]>*
dtype0
�
,Initializer_118/random_uniform/RandomUniformRandomUniform$Initializer_118/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_118/random_uniform/subSub"Initializer_118/random_uniform/max"Initializer_118/random_uniform/min*
T0
�
"Initializer_118/random_uniform/mulMul,Initializer_118/random_uniform/RandomUniform"Initializer_118/random_uniform/sub*
T0
v
Initializer_118/random_uniformAdd"Initializer_118/random_uniform/mul"Initializer_118/random_uniform/min*
T0
�

Assign_118AssignLmio_variable/comment_unfold_score_logit_cluster_gate/dense_3/kernel/gradientInitializer_118/random_uniform*
validate_shape(*
use_locking(*
T0*_
_classU
SQloc:@mio_variable/comment_unfold_score_logit_cluster_gate/dense_3/kernel/gradient
�
Jmio_variable/comment_unfold_score_logit_cluster_gate/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64comment_unfold_score_logit_cluster_gate/dense_3/bias*
shape:@
�
Jmio_variable/comment_unfold_score_logit_cluster_gate/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*C
	container64comment_unfold_score_logit_cluster_gate/dense_3/bias
F
Initializer_119/zerosConst*
dtype0*
valueB@*    
�

Assign_119AssignJmio_variable/comment_unfold_score_logit_cluster_gate/dense_3/bias/gradientInitializer_119/zeros*
use_locking(*
T0*]
_classS
QOloc:@mio_variable/comment_unfold_score_logit_cluster_gate/dense_3/bias/gradient*
validate_shape(
�
<model/comment_unfold_score_logit_cluster_gate/dense_3/MatMulMatMul:model/comment_unfold_score_logit_cluster_gate/dense_2/ReluLmio_variable/comment_unfold_score_logit_cluster_gate/dense_3/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
=model/comment_unfold_score_logit_cluster_gate/dense_3/BiasAddBiasAdd<model/comment_unfold_score_logit_cluster_gate/dense_3/MatMulJmio_variable/comment_unfold_score_logit_cluster_gate/dense_3/bias/variable*
T0*
data_formatNHWC
�
=model/comment_unfold_score_logit_cluster_gate/dense_3/SigmoidSigmoid=model/comment_unfold_score_logit_cluster_gate/dense_3/BiasAdd*
T0
b
5model/comment_unfold_score_logit_cluster_gate/mul_1/xConst*
dtype0*
valueB
 *   @
�
3model/comment_unfold_score_logit_cluster_gate/mul_1Mul5model/comment_unfold_score_logit_cluster_gate/mul_1/x=model/comment_unfold_score_logit_cluster_gate/dense_3/Sigmoid*
T0
�
$model/comment_unfold_score_logit/MulMul'model/comment_top_net/dense_1/LeakyRelu1model/comment_unfold_score_logit_cluster_gate/mul*
T0
�
=mio_variable/comment_unfold_score_logit/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�@*6
	container)'comment_unfold_score_logit/dense/kernel
�
=mio_variable/comment_unfold_score_logit/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*6
	container)'comment_unfold_score_logit/dense/kernel*
shape:	�@
Y
$Initializer_120/random_uniform/shapeConst*
valueB"�   @   *
dtype0
O
"Initializer_120/random_uniform/minConst*
dtype0*
valueB
 *�5�
O
"Initializer_120/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_120/random_uniform/RandomUniformRandomUniform$Initializer_120/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_120/random_uniform/subSub"Initializer_120/random_uniform/max"Initializer_120/random_uniform/min*
T0
�
"Initializer_120/random_uniform/mulMul,Initializer_120/random_uniform/RandomUniform"Initializer_120/random_uniform/sub*
T0
v
Initializer_120/random_uniformAdd"Initializer_120/random_uniform/mul"Initializer_120/random_uniform/min*
T0
�

Assign_120Assign=mio_variable/comment_unfold_score_logit/dense/kernel/gradientInitializer_120/random_uniform*
validate_shape(*
use_locking(*
T0*P
_classF
DBloc:@mio_variable/comment_unfold_score_logit/dense/kernel/gradient
�
;mio_variable/comment_unfold_score_logit/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*4
	container'%comment_unfold_score_logit/dense/bias
�
;mio_variable/comment_unfold_score_logit/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*4
	container'%comment_unfold_score_logit/dense/bias
F
Initializer_121/zerosConst*
valueB@*    *
dtype0
�

Assign_121Assign;mio_variable/comment_unfold_score_logit/dense/bias/gradientInitializer_121/zeros*
use_locking(*
T0*N
_classD
B@loc:@mio_variable/comment_unfold_score_logit/dense/bias/gradient*
validate_shape(
�
-model/comment_unfold_score_logit/dense/MatMulMatMul$model/comment_unfold_score_logit/Mul=mio_variable/comment_unfold_score_logit/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
.model/comment_unfold_score_logit/dense/BiasAddBiasAdd-model/comment_unfold_score_logit/dense/MatMul;mio_variable/comment_unfold_score_logit/dense/bias/variable*
T0*
data_formatNHWC
c
6model/comment_unfold_score_logit/dense/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
4model/comment_unfold_score_logit/dense/LeakyRelu/mulMul6model/comment_unfold_score_logit/dense/LeakyRelu/alpha.model/comment_unfold_score_logit/dense/BiasAdd*
T0
�
0model/comment_unfold_score_logit/dense/LeakyReluMaximum4model/comment_unfold_score_logit/dense/LeakyRelu/mul.model/comment_unfold_score_logit/dense/BiasAdd*
T0
�
&model/comment_unfold_score_logit/Mul_1Mul0model/comment_unfold_score_logit/dense/LeakyRelu3model/comment_unfold_score_logit_cluster_gate/mul_1*
T0
�
?mio_variable/comment_unfold_score_logit/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*8
	container+)comment_unfold_score_logit/dense_1/kernel*
shape
:@
�
?mio_variable/comment_unfold_score_logit/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*8
	container+)comment_unfold_score_logit/dense_1/kernel*
shape
:@
Y
$Initializer_122/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_122/random_uniform/minConst*
valueB
 *����*
dtype0
O
"Initializer_122/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
,Initializer_122/random_uniform/RandomUniformRandomUniform$Initializer_122/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_122/random_uniform/subSub"Initializer_122/random_uniform/max"Initializer_122/random_uniform/min*
T0
�
"Initializer_122/random_uniform/mulMul,Initializer_122/random_uniform/RandomUniform"Initializer_122/random_uniform/sub*
T0
v
Initializer_122/random_uniformAdd"Initializer_122/random_uniform/mul"Initializer_122/random_uniform/min*
T0
�

Assign_122Assign?mio_variable/comment_unfold_score_logit/dense_1/kernel/gradientInitializer_122/random_uniform*
use_locking(*
T0*R
_classH
FDloc:@mio_variable/comment_unfold_score_logit/dense_1/kernel/gradient*
validate_shape(
�
=mio_variable/comment_unfold_score_logit/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*6
	container)'comment_unfold_score_logit/dense_1/bias*
shape:
�
=mio_variable/comment_unfold_score_logit/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*6
	container)'comment_unfold_score_logit/dense_1/bias*
shape:
F
Initializer_123/zerosConst*
valueB*    *
dtype0
�

Assign_123Assign=mio_variable/comment_unfold_score_logit/dense_1/bias/gradientInitializer_123/zeros*
validate_shape(*
use_locking(*
T0*P
_classF
DBloc:@mio_variable/comment_unfold_score_logit/dense_1/bias/gradient
�
/model/comment_unfold_score_logit/dense_1/MatMulMatMul&model/comment_unfold_score_logit/Mul_1?mio_variable/comment_unfold_score_logit/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
0model/comment_unfold_score_logit/dense_1/BiasAddBiasAdd/model/comment_unfold_score_logit/dense_1/MatMul=mio_variable/comment_unfold_score_logit/dense_1/bias/variable*
T0*
data_formatNHWC
e
8model/comment_unfold_score_logit/dense_1/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
6model/comment_unfold_score_logit/dense_1/LeakyRelu/mulMul8model/comment_unfold_score_logit/dense_1/LeakyRelu/alpha0model/comment_unfold_score_logit/dense_1/BiasAdd*
T0
�
2model/comment_unfold_score_logit/dense_1/LeakyReluMaximum6model/comment_unfold_score_logit/dense_1/LeakyRelu/mul0model/comment_unfold_score_logit/dense_1/BiasAdd*
T0
�
Hmio_variable/comment_like_score_logit_cluster_gate/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42comment_like_score_logit_cluster_gate/dense/kernel*
shape:	�
�
Hmio_variable/comment_like_score_logit_cluster_gate/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�*A
	container42comment_like_score_logit_cluster_gate/dense/kernel
Y
$Initializer_124/random_uniform/shapeConst*
valueB"   �   *
dtype0
O
"Initializer_124/random_uniform/minConst*
valueB
 *n�\�*
dtype0
O
"Initializer_124/random_uniform/maxConst*
valueB
 *n�\>*
dtype0
�
,Initializer_124/random_uniform/RandomUniformRandomUniform$Initializer_124/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_124/random_uniform/subSub"Initializer_124/random_uniform/max"Initializer_124/random_uniform/min*
T0
�
"Initializer_124/random_uniform/mulMul,Initializer_124/random_uniform/RandomUniform"Initializer_124/random_uniform/sub*
T0
v
Initializer_124/random_uniformAdd"Initializer_124/random_uniform/mul"Initializer_124/random_uniform/min*
T0
�

Assign_124AssignHmio_variable/comment_like_score_logit_cluster_gate/dense/kernel/gradientInitializer_124/random_uniform*
T0*[
_classQ
OMloc:@mio_variable/comment_like_score_logit_cluster_gate/dense/kernel/gradient*
validate_shape(*
use_locking(
�
Fmio_variable/comment_like_score_logit_cluster_gate/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20comment_like_score_logit_cluster_gate/dense/bias*
shape:�
�
Fmio_variable/comment_like_score_logit_cluster_gate/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20comment_like_score_logit_cluster_gate/dense/bias*
shape:�
G
Initializer_125/zerosConst*
valueB�*    *
dtype0
�

Assign_125AssignFmio_variable/comment_like_score_logit_cluster_gate/dense/bias/gradientInitializer_125/zeros*
T0*Y
_classO
MKloc:@mio_variable/comment_like_score_logit_cluster_gate/dense/bias/gradient*
validate_shape(*
use_locking(
�
8model/comment_like_score_logit_cluster_gate/dense/MatMulMatMul	concat_12Hmio_variable/comment_like_score_logit_cluster_gate/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
9model/comment_like_score_logit_cluster_gate/dense/BiasAddBiasAdd8model/comment_like_score_logit_cluster_gate/dense/MatMulFmio_variable/comment_like_score_logit_cluster_gate/dense/bias/variable*
T0*
data_formatNHWC
�
6model/comment_like_score_logit_cluster_gate/dense/ReluRelu9model/comment_like_score_logit_cluster_gate/dense/BiasAdd*
T0
�
Jmio_variable/comment_like_score_logit_cluster_gate/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64comment_like_score_logit_cluster_gate/dense_1/kernel*
shape:
��
�
Jmio_variable/comment_like_score_logit_cluster_gate/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64comment_like_score_logit_cluster_gate/dense_1/kernel*
shape:
��
Y
$Initializer_126/random_uniform/shapeConst*
valueB"�   �   *
dtype0
O
"Initializer_126/random_uniform/minConst*
valueB
 *q��*
dtype0
O
"Initializer_126/random_uniform/maxConst*
valueB
 *q�>*
dtype0
�
,Initializer_126/random_uniform/RandomUniformRandomUniform$Initializer_126/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_126/random_uniform/subSub"Initializer_126/random_uniform/max"Initializer_126/random_uniform/min*
T0
�
"Initializer_126/random_uniform/mulMul,Initializer_126/random_uniform/RandomUniform"Initializer_126/random_uniform/sub*
T0
v
Initializer_126/random_uniformAdd"Initializer_126/random_uniform/mul"Initializer_126/random_uniform/min*
T0
�

Assign_126AssignJmio_variable/comment_like_score_logit_cluster_gate/dense_1/kernel/gradientInitializer_126/random_uniform*
use_locking(*
T0*]
_classS
QOloc:@mio_variable/comment_like_score_logit_cluster_gate/dense_1/kernel/gradient*
validate_shape(
�
Hmio_variable/comment_like_score_logit_cluster_gate/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42comment_like_score_logit_cluster_gate/dense_1/bias*
shape:�
�
Hmio_variable/comment_like_score_logit_cluster_gate/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:�*A
	container42comment_like_score_logit_cluster_gate/dense_1/bias
G
Initializer_127/zerosConst*
dtype0*
valueB�*    
�

Assign_127AssignHmio_variable/comment_like_score_logit_cluster_gate/dense_1/bias/gradientInitializer_127/zeros*
T0*[
_classQ
OMloc:@mio_variable/comment_like_score_logit_cluster_gate/dense_1/bias/gradient*
validate_shape(*
use_locking(
�
:model/comment_like_score_logit_cluster_gate/dense_1/MatMulMatMul6model/comment_like_score_logit_cluster_gate/dense/ReluJmio_variable/comment_like_score_logit_cluster_gate/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
;model/comment_like_score_logit_cluster_gate/dense_1/BiasAddBiasAdd:model/comment_like_score_logit_cluster_gate/dense_1/MatMulHmio_variable/comment_like_score_logit_cluster_gate/dense_1/bias/variable*
T0*
data_formatNHWC
�
;model/comment_like_score_logit_cluster_gate/dense_1/SigmoidSigmoid;model/comment_like_score_logit_cluster_gate/dense_1/BiasAdd*
T0
^
1model/comment_like_score_logit_cluster_gate/mul/xConst*
dtype0*
valueB
 *   @
�
/model/comment_like_score_logit_cluster_gate/mulMul1model/comment_like_score_logit_cluster_gate/mul/x;model/comment_like_score_logit_cluster_gate/dense_1/Sigmoid*
T0
�
Jmio_variable/comment_like_score_logit_cluster_gate/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64comment_like_score_logit_cluster_gate/dense_2/kernel*
shape:	�@
�
Jmio_variable/comment_like_score_logit_cluster_gate/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64comment_like_score_logit_cluster_gate/dense_2/kernel*
shape:	�@
Y
$Initializer_128/random_uniform/shapeConst*
valueB"�   @   *
dtype0
O
"Initializer_128/random_uniform/minConst*
valueB
 *�5�*
dtype0
O
"Initializer_128/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_128/random_uniform/RandomUniformRandomUniform$Initializer_128/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_128/random_uniform/subSub"Initializer_128/random_uniform/max"Initializer_128/random_uniform/min*
T0
�
"Initializer_128/random_uniform/mulMul,Initializer_128/random_uniform/RandomUniform"Initializer_128/random_uniform/sub*
T0
v
Initializer_128/random_uniformAdd"Initializer_128/random_uniform/mul"Initializer_128/random_uniform/min*
T0
�

Assign_128AssignJmio_variable/comment_like_score_logit_cluster_gate/dense_2/kernel/gradientInitializer_128/random_uniform*
T0*]
_classS
QOloc:@mio_variable/comment_like_score_logit_cluster_gate/dense_2/kernel/gradient*
validate_shape(*
use_locking(
�
Hmio_variable/comment_like_score_logit_cluster_gate/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42comment_like_score_logit_cluster_gate/dense_2/bias*
shape:@
�
Hmio_variable/comment_like_score_logit_cluster_gate/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*A
	container42comment_like_score_logit_cluster_gate/dense_2/bias
F
Initializer_129/zerosConst*
valueB@*    *
dtype0
�

Assign_129AssignHmio_variable/comment_like_score_logit_cluster_gate/dense_2/bias/gradientInitializer_129/zeros*
validate_shape(*
use_locking(*
T0*[
_classQ
OMloc:@mio_variable/comment_like_score_logit_cluster_gate/dense_2/bias/gradient
�
:model/comment_like_score_logit_cluster_gate/dense_2/MatMulMatMul;model/comment_like_score_logit_cluster_gate/dense_1/SigmoidJmio_variable/comment_like_score_logit_cluster_gate/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
;model/comment_like_score_logit_cluster_gate/dense_2/BiasAddBiasAdd:model/comment_like_score_logit_cluster_gate/dense_2/MatMulHmio_variable/comment_like_score_logit_cluster_gate/dense_2/bias/variable*
T0*
data_formatNHWC
�
8model/comment_like_score_logit_cluster_gate/dense_2/ReluRelu;model/comment_like_score_logit_cluster_gate/dense_2/BiasAdd*
T0
�
Jmio_variable/comment_like_score_logit_cluster_gate/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64comment_like_score_logit_cluster_gate/dense_3/kernel*
shape
:@@
�
Jmio_variable/comment_like_score_logit_cluster_gate/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64comment_like_score_logit_cluster_gate/dense_3/kernel*
shape
:@@
Y
$Initializer_130/random_uniform/shapeConst*
dtype0*
valueB"@   @   
O
"Initializer_130/random_uniform/minConst*
dtype0*
valueB
 *׳]�
O
"Initializer_130/random_uniform/maxConst*
valueB
 *׳]>*
dtype0
�
,Initializer_130/random_uniform/RandomUniformRandomUniform$Initializer_130/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_130/random_uniform/subSub"Initializer_130/random_uniform/max"Initializer_130/random_uniform/min*
T0
�
"Initializer_130/random_uniform/mulMul,Initializer_130/random_uniform/RandomUniform"Initializer_130/random_uniform/sub*
T0
v
Initializer_130/random_uniformAdd"Initializer_130/random_uniform/mul"Initializer_130/random_uniform/min*
T0
�

Assign_130AssignJmio_variable/comment_like_score_logit_cluster_gate/dense_3/kernel/gradientInitializer_130/random_uniform*
T0*]
_classS
QOloc:@mio_variable/comment_like_score_logit_cluster_gate/dense_3/kernel/gradient*
validate_shape(*
use_locking(
�
Hmio_variable/comment_like_score_logit_cluster_gate/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*A
	container42comment_like_score_logit_cluster_gate/dense_3/bias
�
Hmio_variable/comment_like_score_logit_cluster_gate/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42comment_like_score_logit_cluster_gate/dense_3/bias*
shape:@
F
Initializer_131/zerosConst*
dtype0*
valueB@*    
�

Assign_131AssignHmio_variable/comment_like_score_logit_cluster_gate/dense_3/bias/gradientInitializer_131/zeros*
use_locking(*
T0*[
_classQ
OMloc:@mio_variable/comment_like_score_logit_cluster_gate/dense_3/bias/gradient*
validate_shape(
�
:model/comment_like_score_logit_cluster_gate/dense_3/MatMulMatMul8model/comment_like_score_logit_cluster_gate/dense_2/ReluJmio_variable/comment_like_score_logit_cluster_gate/dense_3/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
;model/comment_like_score_logit_cluster_gate/dense_3/BiasAddBiasAdd:model/comment_like_score_logit_cluster_gate/dense_3/MatMulHmio_variable/comment_like_score_logit_cluster_gate/dense_3/bias/variable*
T0*
data_formatNHWC
�
;model/comment_like_score_logit_cluster_gate/dense_3/SigmoidSigmoid;model/comment_like_score_logit_cluster_gate/dense_3/BiasAdd*
T0
`
3model/comment_like_score_logit_cluster_gate/mul_1/xConst*
dtype0*
valueB
 *   @
�
1model/comment_like_score_logit_cluster_gate/mul_1Mul3model/comment_like_score_logit_cluster_gate/mul_1/x;model/comment_like_score_logit_cluster_gate/dense_3/Sigmoid*
T0
�
"model/comment_like_score_logit/MulMul'model/comment_top_net/dense_1/LeakyRelu/model/comment_like_score_logit_cluster_gate/mul*
T0
�
;mio_variable/comment_like_score_logit/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*4
	container'%comment_like_score_logit/dense/kernel*
shape:	�@
�
;mio_variable/comment_like_score_logit/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*4
	container'%comment_like_score_logit/dense/kernel*
shape:	�@
Y
$Initializer_132/random_uniform/shapeConst*
valueB"�   @   *
dtype0
O
"Initializer_132/random_uniform/minConst*
valueB
 *�5�*
dtype0
O
"Initializer_132/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_132/random_uniform/RandomUniformRandomUniform$Initializer_132/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_132/random_uniform/subSub"Initializer_132/random_uniform/max"Initializer_132/random_uniform/min*
T0
�
"Initializer_132/random_uniform/mulMul,Initializer_132/random_uniform/RandomUniform"Initializer_132/random_uniform/sub*
T0
v
Initializer_132/random_uniformAdd"Initializer_132/random_uniform/mul"Initializer_132/random_uniform/min*
T0
�

Assign_132Assign;mio_variable/comment_like_score_logit/dense/kernel/gradientInitializer_132/random_uniform*
use_locking(*
T0*N
_classD
B@loc:@mio_variable/comment_like_score_logit/dense/kernel/gradient*
validate_shape(
�
9mio_variable/comment_like_score_logit/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#comment_like_score_logit/dense/bias*
shape:@
�
9mio_variable/comment_like_score_logit/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*2
	container%#comment_like_score_logit/dense/bias
F
Initializer_133/zerosConst*
valueB@*    *
dtype0
�

Assign_133Assign9mio_variable/comment_like_score_logit/dense/bias/gradientInitializer_133/zeros*
use_locking(*
T0*L
_classB
@>loc:@mio_variable/comment_like_score_logit/dense/bias/gradient*
validate_shape(
�
+model/comment_like_score_logit/dense/MatMulMatMul"model/comment_like_score_logit/Mul;mio_variable/comment_like_score_logit/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
,model/comment_like_score_logit/dense/BiasAddBiasAdd+model/comment_like_score_logit/dense/MatMul9mio_variable/comment_like_score_logit/dense/bias/variable*
T0*
data_formatNHWC
a
4model/comment_like_score_logit/dense/LeakyRelu/alphaConst*
dtype0*
valueB
 *��L>
�
2model/comment_like_score_logit/dense/LeakyRelu/mulMul4model/comment_like_score_logit/dense/LeakyRelu/alpha,model/comment_like_score_logit/dense/BiasAdd*
T0
�
.model/comment_like_score_logit/dense/LeakyReluMaximum2model/comment_like_score_logit/dense/LeakyRelu/mul,model/comment_like_score_logit/dense/BiasAdd*
T0
�
$model/comment_like_score_logit/Mul_1Mul.model/comment_like_score_logit/dense/LeakyRelu1model/comment_like_score_logit_cluster_gate/mul_1*
T0
�
=mio_variable/comment_like_score_logit/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*6
	container)'comment_like_score_logit/dense_1/kernel*
shape
:@
�
=mio_variable/comment_like_score_logit/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*6
	container)'comment_like_score_logit/dense_1/kernel*
shape
:@
Y
$Initializer_134/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_134/random_uniform/minConst*
valueB
 *����*
dtype0
O
"Initializer_134/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
,Initializer_134/random_uniform/RandomUniformRandomUniform$Initializer_134/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_134/random_uniform/subSub"Initializer_134/random_uniform/max"Initializer_134/random_uniform/min*
T0
�
"Initializer_134/random_uniform/mulMul,Initializer_134/random_uniform/RandomUniform"Initializer_134/random_uniform/sub*
T0
v
Initializer_134/random_uniformAdd"Initializer_134/random_uniform/mul"Initializer_134/random_uniform/min*
T0
�

Assign_134Assign=mio_variable/comment_like_score_logit/dense_1/kernel/gradientInitializer_134/random_uniform*
use_locking(*
T0*P
_classF
DBloc:@mio_variable/comment_like_score_logit/dense_1/kernel/gradient*
validate_shape(
�
;mio_variable/comment_like_score_logit/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*4
	container'%comment_like_score_logit/dense_1/bias*
shape:
�
;mio_variable/comment_like_score_logit/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*4
	container'%comment_like_score_logit/dense_1/bias*
shape:
F
Initializer_135/zerosConst*
valueB*    *
dtype0
�

Assign_135Assign;mio_variable/comment_like_score_logit/dense_1/bias/gradientInitializer_135/zeros*
use_locking(*
T0*N
_classD
B@loc:@mio_variable/comment_like_score_logit/dense_1/bias/gradient*
validate_shape(
�
-model/comment_like_score_logit/dense_1/MatMulMatMul$model/comment_like_score_logit/Mul_1=mio_variable/comment_like_score_logit/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
.model/comment_like_score_logit/dense_1/BiasAddBiasAdd-model/comment_like_score_logit/dense_1/MatMul;mio_variable/comment_like_score_logit/dense_1/bias/variable*
T0*
data_formatNHWC
c
6model/comment_like_score_logit/dense_1/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
4model/comment_like_score_logit/dense_1/LeakyRelu/mulMul6model/comment_like_score_logit/dense_1/LeakyRelu/alpha.model/comment_like_score_logit/dense_1/BiasAdd*
T0
�
0model/comment_like_score_logit/dense_1/LeakyReluMaximum4model/comment_like_score_logit/dense_1/LeakyRelu/mul.model/comment_like_score_logit/dense_1/BiasAdd*
T0
�
Tmio_variable/comment_content_copyward_score_logit_cluster_gate/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�*M
	container@>comment_content_copyward_score_logit_cluster_gate/dense/kernel
�
Tmio_variable/comment_content_copyward_score_logit_cluster_gate/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*M
	container@>comment_content_copyward_score_logit_cluster_gate/dense/kernel*
shape:	�
Y
$Initializer_136/random_uniform/shapeConst*
valueB"   �   *
dtype0
O
"Initializer_136/random_uniform/minConst*
dtype0*
valueB
 *n�\�
O
"Initializer_136/random_uniform/maxConst*
valueB
 *n�\>*
dtype0
�
,Initializer_136/random_uniform/RandomUniformRandomUniform$Initializer_136/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_136/random_uniform/subSub"Initializer_136/random_uniform/max"Initializer_136/random_uniform/min*
T0
�
"Initializer_136/random_uniform/mulMul,Initializer_136/random_uniform/RandomUniform"Initializer_136/random_uniform/sub*
T0
v
Initializer_136/random_uniformAdd"Initializer_136/random_uniform/mul"Initializer_136/random_uniform/min*
T0
�

Assign_136AssignTmio_variable/comment_content_copyward_score_logit_cluster_gate/dense/kernel/gradientInitializer_136/random_uniform*
validate_shape(*
use_locking(*
T0*g
_class]
[Yloc:@mio_variable/comment_content_copyward_score_logit_cluster_gate/dense/kernel/gradient
�
Rmio_variable/comment_content_copyward_score_logit_cluster_gate/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*K
	container><comment_content_copyward_score_logit_cluster_gate/dense/bias*
shape:�
�
Rmio_variable/comment_content_copyward_score_logit_cluster_gate/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*K
	container><comment_content_copyward_score_logit_cluster_gate/dense/bias*
shape:�
G
Initializer_137/zerosConst*
valueB�*    *
dtype0
�

Assign_137AssignRmio_variable/comment_content_copyward_score_logit_cluster_gate/dense/bias/gradientInitializer_137/zeros*
validate_shape(*
use_locking(*
T0*e
_class[
YWloc:@mio_variable/comment_content_copyward_score_logit_cluster_gate/dense/bias/gradient
�
Dmodel/comment_content_copyward_score_logit_cluster_gate/dense/MatMulMatMul	concat_12Tmio_variable/comment_content_copyward_score_logit_cluster_gate/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
Emodel/comment_content_copyward_score_logit_cluster_gate/dense/BiasAddBiasAddDmodel/comment_content_copyward_score_logit_cluster_gate/dense/MatMulRmio_variable/comment_content_copyward_score_logit_cluster_gate/dense/bias/variable*
data_formatNHWC*
T0
�
Bmodel/comment_content_copyward_score_logit_cluster_gate/dense/ReluReluEmodel/comment_content_copyward_score_logit_cluster_gate/dense/BiasAdd*
T0
�
Vmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
��*O
	containerB@comment_content_copyward_score_logit_cluster_gate/dense_1/kernel
�
Vmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*O
	containerB@comment_content_copyward_score_logit_cluster_gate/dense_1/kernel*
shape:
��
Y
$Initializer_138/random_uniform/shapeConst*
dtype0*
valueB"�   �   
O
"Initializer_138/random_uniform/minConst*
valueB
 *q��*
dtype0
O
"Initializer_138/random_uniform/maxConst*
valueB
 *q�>*
dtype0
�
,Initializer_138/random_uniform/RandomUniformRandomUniform$Initializer_138/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_138/random_uniform/subSub"Initializer_138/random_uniform/max"Initializer_138/random_uniform/min*
T0
�
"Initializer_138/random_uniform/mulMul,Initializer_138/random_uniform/RandomUniform"Initializer_138/random_uniform/sub*
T0
v
Initializer_138/random_uniformAdd"Initializer_138/random_uniform/mul"Initializer_138/random_uniform/min*
T0
�

Assign_138AssignVmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_1/kernel/gradientInitializer_138/random_uniform*
use_locking(*
T0*i
_class_
][loc:@mio_variable/comment_content_copyward_score_logit_cluster_gate/dense_1/kernel/gradient*
validate_shape(
�
Tmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:�*M
	container@>comment_content_copyward_score_logit_cluster_gate/dense_1/bias
�
Tmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*M
	container@>comment_content_copyward_score_logit_cluster_gate/dense_1/bias*
shape:�
G
Initializer_139/zerosConst*
dtype0*
valueB�*    
�

Assign_139AssignTmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_1/bias/gradientInitializer_139/zeros*
T0*g
_class]
[Yloc:@mio_variable/comment_content_copyward_score_logit_cluster_gate/dense_1/bias/gradient*
validate_shape(*
use_locking(
�
Fmodel/comment_content_copyward_score_logit_cluster_gate/dense_1/MatMulMatMulBmodel/comment_content_copyward_score_logit_cluster_gate/dense/ReluVmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
Gmodel/comment_content_copyward_score_logit_cluster_gate/dense_1/BiasAddBiasAddFmodel/comment_content_copyward_score_logit_cluster_gate/dense_1/MatMulTmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_1/bias/variable*
data_formatNHWC*
T0
�
Gmodel/comment_content_copyward_score_logit_cluster_gate/dense_1/SigmoidSigmoidGmodel/comment_content_copyward_score_logit_cluster_gate/dense_1/BiasAdd*
T0
j
=model/comment_content_copyward_score_logit_cluster_gate/mul/xConst*
valueB
 *   @*
dtype0
�
;model/comment_content_copyward_score_logit_cluster_gate/mulMul=model/comment_content_copyward_score_logit_cluster_gate/mul/xGmodel/comment_content_copyward_score_logit_cluster_gate/dense_1/Sigmoid*
T0
�
Vmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*O
	containerB@comment_content_copyward_score_logit_cluster_gate/dense_2/kernel*
shape:	�@
�
Vmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*O
	containerB@comment_content_copyward_score_logit_cluster_gate/dense_2/kernel*
shape:	�@
Y
$Initializer_140/random_uniform/shapeConst*
valueB"�   @   *
dtype0
O
"Initializer_140/random_uniform/minConst*
dtype0*
valueB
 *�5�
O
"Initializer_140/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_140/random_uniform/RandomUniformRandomUniform$Initializer_140/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_140/random_uniform/subSub"Initializer_140/random_uniform/max"Initializer_140/random_uniform/min*
T0
�
"Initializer_140/random_uniform/mulMul,Initializer_140/random_uniform/RandomUniform"Initializer_140/random_uniform/sub*
T0
v
Initializer_140/random_uniformAdd"Initializer_140/random_uniform/mul"Initializer_140/random_uniform/min*
T0
�

Assign_140AssignVmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_2/kernel/gradientInitializer_140/random_uniform*
validate_shape(*
use_locking(*
T0*i
_class_
][loc:@mio_variable/comment_content_copyward_score_logit_cluster_gate/dense_2/kernel/gradient
�
Tmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*M
	container@>comment_content_copyward_score_logit_cluster_gate/dense_2/bias
�
Tmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*M
	container@>comment_content_copyward_score_logit_cluster_gate/dense_2/bias*
shape:@
F
Initializer_141/zerosConst*
valueB@*    *
dtype0
�

Assign_141AssignTmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_2/bias/gradientInitializer_141/zeros*
validate_shape(*
use_locking(*
T0*g
_class]
[Yloc:@mio_variable/comment_content_copyward_score_logit_cluster_gate/dense_2/bias/gradient
�
Fmodel/comment_content_copyward_score_logit_cluster_gate/dense_2/MatMulMatMulGmodel/comment_content_copyward_score_logit_cluster_gate/dense_1/SigmoidVmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
Gmodel/comment_content_copyward_score_logit_cluster_gate/dense_2/BiasAddBiasAddFmodel/comment_content_copyward_score_logit_cluster_gate/dense_2/MatMulTmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_2/bias/variable*
data_formatNHWC*
T0
�
Dmodel/comment_content_copyward_score_logit_cluster_gate/dense_2/ReluReluGmodel/comment_content_copyward_score_logit_cluster_gate/dense_2/BiasAdd*
T0
�
Vmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*O
	containerB@comment_content_copyward_score_logit_cluster_gate/dense_3/kernel*
shape
:@@
�
Vmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@@*O
	containerB@comment_content_copyward_score_logit_cluster_gate/dense_3/kernel
Y
$Initializer_142/random_uniform/shapeConst*
dtype0*
valueB"@   @   
O
"Initializer_142/random_uniform/minConst*
valueB
 *׳]�*
dtype0
O
"Initializer_142/random_uniform/maxConst*
valueB
 *׳]>*
dtype0
�
,Initializer_142/random_uniform/RandomUniformRandomUniform$Initializer_142/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_142/random_uniform/subSub"Initializer_142/random_uniform/max"Initializer_142/random_uniform/min*
T0
�
"Initializer_142/random_uniform/mulMul,Initializer_142/random_uniform/RandomUniform"Initializer_142/random_uniform/sub*
T0
v
Initializer_142/random_uniformAdd"Initializer_142/random_uniform/mul"Initializer_142/random_uniform/min*
T0
�

Assign_142AssignVmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_3/kernel/gradientInitializer_142/random_uniform*
use_locking(*
T0*i
_class_
][loc:@mio_variable/comment_content_copyward_score_logit_cluster_gate/dense_3/kernel/gradient*
validate_shape(
�
Tmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*M
	container@>comment_content_copyward_score_logit_cluster_gate/dense_3/bias*
shape:@
�
Tmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*M
	container@>comment_content_copyward_score_logit_cluster_gate/dense_3/bias*
shape:@
F
Initializer_143/zerosConst*
valueB@*    *
dtype0
�

Assign_143AssignTmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_3/bias/gradientInitializer_143/zeros*
validate_shape(*
use_locking(*
T0*g
_class]
[Yloc:@mio_variable/comment_content_copyward_score_logit_cluster_gate/dense_3/bias/gradient
�
Fmodel/comment_content_copyward_score_logit_cluster_gate/dense_3/MatMulMatMulDmodel/comment_content_copyward_score_logit_cluster_gate/dense_2/ReluVmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
Gmodel/comment_content_copyward_score_logit_cluster_gate/dense_3/BiasAddBiasAddFmodel/comment_content_copyward_score_logit_cluster_gate/dense_3/MatMulTmio_variable/comment_content_copyward_score_logit_cluster_gate/dense_3/bias/variable*
T0*
data_formatNHWC
�
Gmodel/comment_content_copyward_score_logit_cluster_gate/dense_3/SigmoidSigmoidGmodel/comment_content_copyward_score_logit_cluster_gate/dense_3/BiasAdd*
T0
l
?model/comment_content_copyward_score_logit_cluster_gate/mul_1/xConst*
valueB
 *   @*
dtype0
�
=model/comment_content_copyward_score_logit_cluster_gate/mul_1Mul?model/comment_content_copyward_score_logit_cluster_gate/mul_1/xGmodel/comment_content_copyward_score_logit_cluster_gate/dense_3/Sigmoid*
T0
�
.model/comment_content_copyward_score_logit/MulMul'model/comment_top_net/dense_1/LeakyRelu;model/comment_content_copyward_score_logit_cluster_gate/mul*
T0
�
Gmio_variable/comment_content_copyward_score_logit/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*@
	container31comment_content_copyward_score_logit/dense/kernel*
shape:	�@
�
Gmio_variable/comment_content_copyward_score_logit/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�@*@
	container31comment_content_copyward_score_logit/dense/kernel
Y
$Initializer_144/random_uniform/shapeConst*
valueB"�   @   *
dtype0
O
"Initializer_144/random_uniform/minConst*
valueB
 *�5�*
dtype0
O
"Initializer_144/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_144/random_uniform/RandomUniformRandomUniform$Initializer_144/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_144/random_uniform/subSub"Initializer_144/random_uniform/max"Initializer_144/random_uniform/min*
T0
�
"Initializer_144/random_uniform/mulMul,Initializer_144/random_uniform/RandomUniform"Initializer_144/random_uniform/sub*
T0
v
Initializer_144/random_uniformAdd"Initializer_144/random_uniform/mul"Initializer_144/random_uniform/min*
T0
�

Assign_144AssignGmio_variable/comment_content_copyward_score_logit/dense/kernel/gradientInitializer_144/random_uniform*
T0*Z
_classP
NLloc:@mio_variable/comment_content_copyward_score_logit/dense/kernel/gradient*
validate_shape(*
use_locking(
�
Emio_variable/comment_content_copyward_score_logit/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*>
	container1/comment_content_copyward_score_logit/dense/bias*
shape:@
�
Emio_variable/comment_content_copyward_score_logit/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*>
	container1/comment_content_copyward_score_logit/dense/bias*
shape:@
F
Initializer_145/zerosConst*
valueB@*    *
dtype0
�

Assign_145AssignEmio_variable/comment_content_copyward_score_logit/dense/bias/gradientInitializer_145/zeros*
use_locking(*
T0*X
_classN
LJloc:@mio_variable/comment_content_copyward_score_logit/dense/bias/gradient*
validate_shape(
�
7model/comment_content_copyward_score_logit/dense/MatMulMatMul.model/comment_content_copyward_score_logit/MulGmio_variable/comment_content_copyward_score_logit/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
8model/comment_content_copyward_score_logit/dense/BiasAddBiasAdd7model/comment_content_copyward_score_logit/dense/MatMulEmio_variable/comment_content_copyward_score_logit/dense/bias/variable*
T0*
data_formatNHWC
m
@model/comment_content_copyward_score_logit/dense/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
>model/comment_content_copyward_score_logit/dense/LeakyRelu/mulMul@model/comment_content_copyward_score_logit/dense/LeakyRelu/alpha8model/comment_content_copyward_score_logit/dense/BiasAdd*
T0
�
:model/comment_content_copyward_score_logit/dense/LeakyReluMaximum>model/comment_content_copyward_score_logit/dense/LeakyRelu/mul8model/comment_content_copyward_score_logit/dense/BiasAdd*
T0
�
0model/comment_content_copyward_score_logit/Mul_1Mul:model/comment_content_copyward_score_logit/dense/LeakyRelu=model/comment_content_copyward_score_logit_cluster_gate/mul_1*
T0
�
Imio_variable/comment_content_copyward_score_logit/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*B
	container53comment_content_copyward_score_logit/dense_1/kernel*
shape
:@
�
Imio_variable/comment_content_copyward_score_logit/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*B
	container53comment_content_copyward_score_logit/dense_1/kernel*
shape
:@
Y
$Initializer_146/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_146/random_uniform/minConst*
valueB
 *����*
dtype0
O
"Initializer_146/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
,Initializer_146/random_uniform/RandomUniformRandomUniform$Initializer_146/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_146/random_uniform/subSub"Initializer_146/random_uniform/max"Initializer_146/random_uniform/min*
T0
�
"Initializer_146/random_uniform/mulMul,Initializer_146/random_uniform/RandomUniform"Initializer_146/random_uniform/sub*
T0
v
Initializer_146/random_uniformAdd"Initializer_146/random_uniform/mul"Initializer_146/random_uniform/min*
T0
�

Assign_146AssignImio_variable/comment_content_copyward_score_logit/dense_1/kernel/gradientInitializer_146/random_uniform*
T0*\
_classR
PNloc:@mio_variable/comment_content_copyward_score_logit/dense_1/kernel/gradient*
validate_shape(*
use_locking(
�
Gmio_variable/comment_content_copyward_score_logit/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*@
	container31comment_content_copyward_score_logit/dense_1/bias*
shape:
�
Gmio_variable/comment_content_copyward_score_logit/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*@
	container31comment_content_copyward_score_logit/dense_1/bias*
shape:
F
Initializer_147/zerosConst*
dtype0*
valueB*    
�

Assign_147AssignGmio_variable/comment_content_copyward_score_logit/dense_1/bias/gradientInitializer_147/zeros*
validate_shape(*
use_locking(*
T0*Z
_classP
NLloc:@mio_variable/comment_content_copyward_score_logit/dense_1/bias/gradient
�
9model/comment_content_copyward_score_logit/dense_1/MatMulMatMul0model/comment_content_copyward_score_logit/Mul_1Imio_variable/comment_content_copyward_score_logit/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
:model/comment_content_copyward_score_logit/dense_1/BiasAddBiasAdd9model/comment_content_copyward_score_logit/dense_1/MatMulGmio_variable/comment_content_copyward_score_logit/dense_1/bias/variable*
T0*
data_formatNHWC
o
Bmodel/comment_content_copyward_score_logit/dense_1/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
@model/comment_content_copyward_score_logit/dense_1/LeakyRelu/mulMulBmodel/comment_content_copyward_score_logit/dense_1/LeakyRelu/alpha:model/comment_content_copyward_score_logit/dense_1/BiasAdd*
T0
�
<model/comment_content_copyward_score_logit/dense_1/LeakyReluMaximum@model/comment_content_copyward_score_logit/dense_1/LeakyRelu/mul:model/comment_content_copyward_score_logit/dense_1/BiasAdd*
T0
�
Rmio_variable/comment_effective_read_score_logit_cluster_gate/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�*K
	container><comment_effective_read_score_logit_cluster_gate/dense/kernel
�
Rmio_variable/comment_effective_read_score_logit_cluster_gate/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*K
	container><comment_effective_read_score_logit_cluster_gate/dense/kernel*
shape:	�
Y
$Initializer_148/random_uniform/shapeConst*
valueB"   �   *
dtype0
O
"Initializer_148/random_uniform/minConst*
valueB
 *n�\�*
dtype0
O
"Initializer_148/random_uniform/maxConst*
valueB
 *n�\>*
dtype0
�
,Initializer_148/random_uniform/RandomUniformRandomUniform$Initializer_148/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_148/random_uniform/subSub"Initializer_148/random_uniform/max"Initializer_148/random_uniform/min*
T0
�
"Initializer_148/random_uniform/mulMul,Initializer_148/random_uniform/RandomUniform"Initializer_148/random_uniform/sub*
T0
v
Initializer_148/random_uniformAdd"Initializer_148/random_uniform/mul"Initializer_148/random_uniform/min*
T0
�

Assign_148AssignRmio_variable/comment_effective_read_score_logit_cluster_gate/dense/kernel/gradientInitializer_148/random_uniform*
validate_shape(*
use_locking(*
T0*e
_class[
YWloc:@mio_variable/comment_effective_read_score_logit_cluster_gate/dense/kernel/gradient
�
Pmio_variable/comment_effective_read_score_logit_cluster_gate/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*I
	container<:comment_effective_read_score_logit_cluster_gate/dense/bias*
shape:�
�
Pmio_variable/comment_effective_read_score_logit_cluster_gate/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*I
	container<:comment_effective_read_score_logit_cluster_gate/dense/bias*
shape:�
G
Initializer_149/zerosConst*
valueB�*    *
dtype0
�

Assign_149AssignPmio_variable/comment_effective_read_score_logit_cluster_gate/dense/bias/gradientInitializer_149/zeros*
use_locking(*
T0*c
_classY
WUloc:@mio_variable/comment_effective_read_score_logit_cluster_gate/dense/bias/gradient*
validate_shape(
�
Bmodel/comment_effective_read_score_logit_cluster_gate/dense/MatMulMatMul	concat_12Rmio_variable/comment_effective_read_score_logit_cluster_gate/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
Cmodel/comment_effective_read_score_logit_cluster_gate/dense/BiasAddBiasAddBmodel/comment_effective_read_score_logit_cluster_gate/dense/MatMulPmio_variable/comment_effective_read_score_logit_cluster_gate/dense/bias/variable*
data_formatNHWC*
T0
�
@model/comment_effective_read_score_logit_cluster_gate/dense/ReluReluCmodel/comment_effective_read_score_logit_cluster_gate/dense/BiasAdd*
T0
�
Tmio_variable/comment_effective_read_score_logit_cluster_gate/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*M
	container@>comment_effective_read_score_logit_cluster_gate/dense_1/kernel*
shape:
��
�
Tmio_variable/comment_effective_read_score_logit_cluster_gate/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*M
	container@>comment_effective_read_score_logit_cluster_gate/dense_1/kernel*
shape:
��
Y
$Initializer_150/random_uniform/shapeConst*
dtype0*
valueB"�   �   
O
"Initializer_150/random_uniform/minConst*
valueB
 *q��*
dtype0
O
"Initializer_150/random_uniform/maxConst*
valueB
 *q�>*
dtype0
�
,Initializer_150/random_uniform/RandomUniformRandomUniform$Initializer_150/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_150/random_uniform/subSub"Initializer_150/random_uniform/max"Initializer_150/random_uniform/min*
T0
�
"Initializer_150/random_uniform/mulMul,Initializer_150/random_uniform/RandomUniform"Initializer_150/random_uniform/sub*
T0
v
Initializer_150/random_uniformAdd"Initializer_150/random_uniform/mul"Initializer_150/random_uniform/min*
T0
�

Assign_150AssignTmio_variable/comment_effective_read_score_logit_cluster_gate/dense_1/kernel/gradientInitializer_150/random_uniform*
validate_shape(*
use_locking(*
T0*g
_class]
[Yloc:@mio_variable/comment_effective_read_score_logit_cluster_gate/dense_1/kernel/gradient
�
Rmio_variable/comment_effective_read_score_logit_cluster_gate/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*K
	container><comment_effective_read_score_logit_cluster_gate/dense_1/bias*
shape:�
�
Rmio_variable/comment_effective_read_score_logit_cluster_gate/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*K
	container><comment_effective_read_score_logit_cluster_gate/dense_1/bias*
shape:�
G
Initializer_151/zerosConst*
valueB�*    *
dtype0
�

Assign_151AssignRmio_variable/comment_effective_read_score_logit_cluster_gate/dense_1/bias/gradientInitializer_151/zeros*
use_locking(*
T0*e
_class[
YWloc:@mio_variable/comment_effective_read_score_logit_cluster_gate/dense_1/bias/gradient*
validate_shape(
�
Dmodel/comment_effective_read_score_logit_cluster_gate/dense_1/MatMulMatMul@model/comment_effective_read_score_logit_cluster_gate/dense/ReluTmio_variable/comment_effective_read_score_logit_cluster_gate/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
Emodel/comment_effective_read_score_logit_cluster_gate/dense_1/BiasAddBiasAddDmodel/comment_effective_read_score_logit_cluster_gate/dense_1/MatMulRmio_variable/comment_effective_read_score_logit_cluster_gate/dense_1/bias/variable*
T0*
data_formatNHWC
�
Emodel/comment_effective_read_score_logit_cluster_gate/dense_1/SigmoidSigmoidEmodel/comment_effective_read_score_logit_cluster_gate/dense_1/BiasAdd*
T0
h
;model/comment_effective_read_score_logit_cluster_gate/mul/xConst*
valueB
 *   @*
dtype0
�
9model/comment_effective_read_score_logit_cluster_gate/mulMul;model/comment_effective_read_score_logit_cluster_gate/mul/xEmodel/comment_effective_read_score_logit_cluster_gate/dense_1/Sigmoid*
T0
�
Tmio_variable/comment_effective_read_score_logit_cluster_gate/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*M
	container@>comment_effective_read_score_logit_cluster_gate/dense_2/kernel*
shape:	�@
�
Tmio_variable/comment_effective_read_score_logit_cluster_gate/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*M
	container@>comment_effective_read_score_logit_cluster_gate/dense_2/kernel*
shape:	�@
Y
$Initializer_152/random_uniform/shapeConst*
dtype0*
valueB"�   @   
O
"Initializer_152/random_uniform/minConst*
valueB
 *�5�*
dtype0
O
"Initializer_152/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_152/random_uniform/RandomUniformRandomUniform$Initializer_152/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_152/random_uniform/subSub"Initializer_152/random_uniform/max"Initializer_152/random_uniform/min*
T0
�
"Initializer_152/random_uniform/mulMul,Initializer_152/random_uniform/RandomUniform"Initializer_152/random_uniform/sub*
T0
v
Initializer_152/random_uniformAdd"Initializer_152/random_uniform/mul"Initializer_152/random_uniform/min*
T0
�

Assign_152AssignTmio_variable/comment_effective_read_score_logit_cluster_gate/dense_2/kernel/gradientInitializer_152/random_uniform*
use_locking(*
T0*g
_class]
[Yloc:@mio_variable/comment_effective_read_score_logit_cluster_gate/dense_2/kernel/gradient*
validate_shape(
�
Rmio_variable/comment_effective_read_score_logit_cluster_gate/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*K
	container><comment_effective_read_score_logit_cluster_gate/dense_2/bias*
shape:@
�
Rmio_variable/comment_effective_read_score_logit_cluster_gate/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*K
	container><comment_effective_read_score_logit_cluster_gate/dense_2/bias*
shape:@
F
Initializer_153/zerosConst*
valueB@*    *
dtype0
�

Assign_153AssignRmio_variable/comment_effective_read_score_logit_cluster_gate/dense_2/bias/gradientInitializer_153/zeros*
use_locking(*
T0*e
_class[
YWloc:@mio_variable/comment_effective_read_score_logit_cluster_gate/dense_2/bias/gradient*
validate_shape(
�
Dmodel/comment_effective_read_score_logit_cluster_gate/dense_2/MatMulMatMulEmodel/comment_effective_read_score_logit_cluster_gate/dense_1/SigmoidTmio_variable/comment_effective_read_score_logit_cluster_gate/dense_2/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
Emodel/comment_effective_read_score_logit_cluster_gate/dense_2/BiasAddBiasAddDmodel/comment_effective_read_score_logit_cluster_gate/dense_2/MatMulRmio_variable/comment_effective_read_score_logit_cluster_gate/dense_2/bias/variable*
data_formatNHWC*
T0
�
Bmodel/comment_effective_read_score_logit_cluster_gate/dense_2/ReluReluEmodel/comment_effective_read_score_logit_cluster_gate/dense_2/BiasAdd*
T0
�
Tmio_variable/comment_effective_read_score_logit_cluster_gate/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@@*M
	container@>comment_effective_read_score_logit_cluster_gate/dense_3/kernel
�
Tmio_variable/comment_effective_read_score_logit_cluster_gate/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@@*M
	container@>comment_effective_read_score_logit_cluster_gate/dense_3/kernel
Y
$Initializer_154/random_uniform/shapeConst*
valueB"@   @   *
dtype0
O
"Initializer_154/random_uniform/minConst*
valueB
 *׳]�*
dtype0
O
"Initializer_154/random_uniform/maxConst*
valueB
 *׳]>*
dtype0
�
,Initializer_154/random_uniform/RandomUniformRandomUniform$Initializer_154/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_154/random_uniform/subSub"Initializer_154/random_uniform/max"Initializer_154/random_uniform/min*
T0
�
"Initializer_154/random_uniform/mulMul,Initializer_154/random_uniform/RandomUniform"Initializer_154/random_uniform/sub*
T0
v
Initializer_154/random_uniformAdd"Initializer_154/random_uniform/mul"Initializer_154/random_uniform/min*
T0
�

Assign_154AssignTmio_variable/comment_effective_read_score_logit_cluster_gate/dense_3/kernel/gradientInitializer_154/random_uniform*
use_locking(*
T0*g
_class]
[Yloc:@mio_variable/comment_effective_read_score_logit_cluster_gate/dense_3/kernel/gradient*
validate_shape(
�
Rmio_variable/comment_effective_read_score_logit_cluster_gate/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*K
	container><comment_effective_read_score_logit_cluster_gate/dense_3/bias*
shape:@
�
Rmio_variable/comment_effective_read_score_logit_cluster_gate/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*K
	container><comment_effective_read_score_logit_cluster_gate/dense_3/bias
F
Initializer_155/zerosConst*
valueB@*    *
dtype0
�

Assign_155AssignRmio_variable/comment_effective_read_score_logit_cluster_gate/dense_3/bias/gradientInitializer_155/zeros*
use_locking(*
T0*e
_class[
YWloc:@mio_variable/comment_effective_read_score_logit_cluster_gate/dense_3/bias/gradient*
validate_shape(
�
Dmodel/comment_effective_read_score_logit_cluster_gate/dense_3/MatMulMatMulBmodel/comment_effective_read_score_logit_cluster_gate/dense_2/ReluTmio_variable/comment_effective_read_score_logit_cluster_gate/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
Emodel/comment_effective_read_score_logit_cluster_gate/dense_3/BiasAddBiasAddDmodel/comment_effective_read_score_logit_cluster_gate/dense_3/MatMulRmio_variable/comment_effective_read_score_logit_cluster_gate/dense_3/bias/variable*
T0*
data_formatNHWC
�
Emodel/comment_effective_read_score_logit_cluster_gate/dense_3/SigmoidSigmoidEmodel/comment_effective_read_score_logit_cluster_gate/dense_3/BiasAdd*
T0
j
=model/comment_effective_read_score_logit_cluster_gate/mul_1/xConst*
valueB
 *   @*
dtype0
�
;model/comment_effective_read_score_logit_cluster_gate/mul_1Mul=model/comment_effective_read_score_logit_cluster_gate/mul_1/xEmodel/comment_effective_read_score_logit_cluster_gate/dense_3/Sigmoid*
T0
�
,model/comment_effective_read_score_logit/MulMul'model/comment_top_net/dense_1/LeakyRelu9model/comment_effective_read_score_logit_cluster_gate/mul*
T0
�
Emio_variable/comment_effective_read_score_logit/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*>
	container1/comment_effective_read_score_logit/dense/kernel*
shape:	�@
�
Emio_variable/comment_effective_read_score_logit/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�@*>
	container1/comment_effective_read_score_logit/dense/kernel
Y
$Initializer_156/random_uniform/shapeConst*
valueB"�   @   *
dtype0
O
"Initializer_156/random_uniform/minConst*
valueB
 *�5�*
dtype0
O
"Initializer_156/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_156/random_uniform/RandomUniformRandomUniform$Initializer_156/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_156/random_uniform/subSub"Initializer_156/random_uniform/max"Initializer_156/random_uniform/min*
T0
�
"Initializer_156/random_uniform/mulMul,Initializer_156/random_uniform/RandomUniform"Initializer_156/random_uniform/sub*
T0
v
Initializer_156/random_uniformAdd"Initializer_156/random_uniform/mul"Initializer_156/random_uniform/min*
T0
�

Assign_156AssignEmio_variable/comment_effective_read_score_logit/dense/kernel/gradientInitializer_156/random_uniform*
use_locking(*
T0*X
_classN
LJloc:@mio_variable/comment_effective_read_score_logit/dense/kernel/gradient*
validate_shape(
�
Cmio_variable/comment_effective_read_score_logit/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*<
	container/-comment_effective_read_score_logit/dense/bias*
shape:@
�
Cmio_variable/comment_effective_read_score_logit/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*<
	container/-comment_effective_read_score_logit/dense/bias*
shape:@
F
Initializer_157/zerosConst*
valueB@*    *
dtype0
�

Assign_157AssignCmio_variable/comment_effective_read_score_logit/dense/bias/gradientInitializer_157/zeros*
use_locking(*
T0*V
_classL
JHloc:@mio_variable/comment_effective_read_score_logit/dense/bias/gradient*
validate_shape(
�
5model/comment_effective_read_score_logit/dense/MatMulMatMul,model/comment_effective_read_score_logit/MulEmio_variable/comment_effective_read_score_logit/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
6model/comment_effective_read_score_logit/dense/BiasAddBiasAdd5model/comment_effective_read_score_logit/dense/MatMulCmio_variable/comment_effective_read_score_logit/dense/bias/variable*
T0*
data_formatNHWC
k
>model/comment_effective_read_score_logit/dense/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
<model/comment_effective_read_score_logit/dense/LeakyRelu/mulMul>model/comment_effective_read_score_logit/dense/LeakyRelu/alpha6model/comment_effective_read_score_logit/dense/BiasAdd*
T0
�
8model/comment_effective_read_score_logit/dense/LeakyReluMaximum<model/comment_effective_read_score_logit/dense/LeakyRelu/mul6model/comment_effective_read_score_logit/dense/BiasAdd*
T0
�
.model/comment_effective_read_score_logit/Mul_1Mul8model/comment_effective_read_score_logit/dense/LeakyRelu;model/comment_effective_read_score_logit_cluster_gate/mul_1*
T0
�
Gmio_variable/comment_effective_read_score_logit/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*@
	container31comment_effective_read_score_logit/dense_1/kernel*
shape
:@
�
Gmio_variable/comment_effective_read_score_logit/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*@
	container31comment_effective_read_score_logit/dense_1/kernel*
shape
:@
Y
$Initializer_158/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_158/random_uniform/minConst*
valueB
 *����*
dtype0
O
"Initializer_158/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
,Initializer_158/random_uniform/RandomUniformRandomUniform$Initializer_158/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_158/random_uniform/subSub"Initializer_158/random_uniform/max"Initializer_158/random_uniform/min*
T0
�
"Initializer_158/random_uniform/mulMul,Initializer_158/random_uniform/RandomUniform"Initializer_158/random_uniform/sub*
T0
v
Initializer_158/random_uniformAdd"Initializer_158/random_uniform/mul"Initializer_158/random_uniform/min*
T0
�

Assign_158AssignGmio_variable/comment_effective_read_score_logit/dense_1/kernel/gradientInitializer_158/random_uniform*
validate_shape(*
use_locking(*
T0*Z
_classP
NLloc:@mio_variable/comment_effective_read_score_logit/dense_1/kernel/gradient
�
Emio_variable/comment_effective_read_score_logit/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*>
	container1/comment_effective_read_score_logit/dense_1/bias*
shape:
�
Emio_variable/comment_effective_read_score_logit/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*>
	container1/comment_effective_read_score_logit/dense_1/bias*
shape:
F
Initializer_159/zerosConst*
valueB*    *
dtype0
�

Assign_159AssignEmio_variable/comment_effective_read_score_logit/dense_1/bias/gradientInitializer_159/zeros*
use_locking(*
T0*X
_classN
LJloc:@mio_variable/comment_effective_read_score_logit/dense_1/bias/gradient*
validate_shape(
�
7model/comment_effective_read_score_logit/dense_1/MatMulMatMul.model/comment_effective_read_score_logit/Mul_1Gmio_variable/comment_effective_read_score_logit/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
8model/comment_effective_read_score_logit/dense_1/BiasAddBiasAdd7model/comment_effective_read_score_logit/dense_1/MatMulEmio_variable/comment_effective_read_score_logit/dense_1/bias/variable*
T0*
data_formatNHWC
m
@model/comment_effective_read_score_logit/dense_1/LeakyRelu/alphaConst*
dtype0*
valueB
 *��L>
�
>model/comment_effective_read_score_logit/dense_1/LeakyRelu/mulMul@model/comment_effective_read_score_logit/dense_1/LeakyRelu/alpha8model/comment_effective_read_score_logit/dense_1/BiasAdd*
T0
�
:model/comment_effective_read_score_logit/dense_1/LeakyReluMaximum>model/comment_effective_read_score_logit/dense_1/LeakyRelu/mul8model/comment_effective_read_score_logit/dense_1/BiasAdd*
T0
�
Nmio_variable/comment_slide_down_score_logit_cluster_gate/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*G
	container:8comment_slide_down_score_logit_cluster_gate/dense/kernel*
shape:	�
�
Nmio_variable/comment_slide_down_score_logit_cluster_gate/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*G
	container:8comment_slide_down_score_logit_cluster_gate/dense/kernel*
shape:	�
Y
$Initializer_160/random_uniform/shapeConst*
valueB"   �   *
dtype0
O
"Initializer_160/random_uniform/minConst*
dtype0*
valueB
 *n�\�
O
"Initializer_160/random_uniform/maxConst*
valueB
 *n�\>*
dtype0
�
,Initializer_160/random_uniform/RandomUniformRandomUniform$Initializer_160/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_160/random_uniform/subSub"Initializer_160/random_uniform/max"Initializer_160/random_uniform/min*
T0
�
"Initializer_160/random_uniform/mulMul,Initializer_160/random_uniform/RandomUniform"Initializer_160/random_uniform/sub*
T0
v
Initializer_160/random_uniformAdd"Initializer_160/random_uniform/mul"Initializer_160/random_uniform/min*
T0
�

Assign_160AssignNmio_variable/comment_slide_down_score_logit_cluster_gate/dense/kernel/gradientInitializer_160/random_uniform*
use_locking(*
T0*a
_classW
USloc:@mio_variable/comment_slide_down_score_logit_cluster_gate/dense/kernel/gradient*
validate_shape(
�
Lmio_variable/comment_slide_down_score_logit_cluster_gate/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*E
	container86comment_slide_down_score_logit_cluster_gate/dense/bias*
shape:�
�
Lmio_variable/comment_slide_down_score_logit_cluster_gate/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:�*E
	container86comment_slide_down_score_logit_cluster_gate/dense/bias
G
Initializer_161/zerosConst*
valueB�*    *
dtype0
�

Assign_161AssignLmio_variable/comment_slide_down_score_logit_cluster_gate/dense/bias/gradientInitializer_161/zeros*
T0*_
_classU
SQloc:@mio_variable/comment_slide_down_score_logit_cluster_gate/dense/bias/gradient*
validate_shape(*
use_locking(
�
>model/comment_slide_down_score_logit_cluster_gate/dense/MatMulMatMul	concat_12Nmio_variable/comment_slide_down_score_logit_cluster_gate/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
?model/comment_slide_down_score_logit_cluster_gate/dense/BiasAddBiasAdd>model/comment_slide_down_score_logit_cluster_gate/dense/MatMulLmio_variable/comment_slide_down_score_logit_cluster_gate/dense/bias/variable*
data_formatNHWC*
T0
�
<model/comment_slide_down_score_logit_cluster_gate/dense/ReluRelu?model/comment_slide_down_score_logit_cluster_gate/dense/BiasAdd*
T0
�
Pmio_variable/comment_slide_down_score_logit_cluster_gate/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*I
	container<:comment_slide_down_score_logit_cluster_gate/dense_1/kernel*
shape:
��
�
Pmio_variable/comment_slide_down_score_logit_cluster_gate/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
��*I
	container<:comment_slide_down_score_logit_cluster_gate/dense_1/kernel
Y
$Initializer_162/random_uniform/shapeConst*
valueB"�   �   *
dtype0
O
"Initializer_162/random_uniform/minConst*
valueB
 *q��*
dtype0
O
"Initializer_162/random_uniform/maxConst*
valueB
 *q�>*
dtype0
�
,Initializer_162/random_uniform/RandomUniformRandomUniform$Initializer_162/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_162/random_uniform/subSub"Initializer_162/random_uniform/max"Initializer_162/random_uniform/min*
T0
�
"Initializer_162/random_uniform/mulMul,Initializer_162/random_uniform/RandomUniform"Initializer_162/random_uniform/sub*
T0
v
Initializer_162/random_uniformAdd"Initializer_162/random_uniform/mul"Initializer_162/random_uniform/min*
T0
�

Assign_162AssignPmio_variable/comment_slide_down_score_logit_cluster_gate/dense_1/kernel/gradientInitializer_162/random_uniform*
use_locking(*
T0*c
_classY
WUloc:@mio_variable/comment_slide_down_score_logit_cluster_gate/dense_1/kernel/gradient*
validate_shape(
�
Nmio_variable/comment_slide_down_score_logit_cluster_gate/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*G
	container:8comment_slide_down_score_logit_cluster_gate/dense_1/bias*
shape:�
�
Nmio_variable/comment_slide_down_score_logit_cluster_gate/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*G
	container:8comment_slide_down_score_logit_cluster_gate/dense_1/bias*
shape:�
G
Initializer_163/zerosConst*
valueB�*    *
dtype0
�

Assign_163AssignNmio_variable/comment_slide_down_score_logit_cluster_gate/dense_1/bias/gradientInitializer_163/zeros*
use_locking(*
T0*a
_classW
USloc:@mio_variable/comment_slide_down_score_logit_cluster_gate/dense_1/bias/gradient*
validate_shape(
�
@model/comment_slide_down_score_logit_cluster_gate/dense_1/MatMulMatMul<model/comment_slide_down_score_logit_cluster_gate/dense/ReluPmio_variable/comment_slide_down_score_logit_cluster_gate/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
Amodel/comment_slide_down_score_logit_cluster_gate/dense_1/BiasAddBiasAdd@model/comment_slide_down_score_logit_cluster_gate/dense_1/MatMulNmio_variable/comment_slide_down_score_logit_cluster_gate/dense_1/bias/variable*
data_formatNHWC*
T0
�
Amodel/comment_slide_down_score_logit_cluster_gate/dense_1/SigmoidSigmoidAmodel/comment_slide_down_score_logit_cluster_gate/dense_1/BiasAdd*
T0
d
7model/comment_slide_down_score_logit_cluster_gate/mul/xConst*
valueB
 *   @*
dtype0
�
5model/comment_slide_down_score_logit_cluster_gate/mulMul7model/comment_slide_down_score_logit_cluster_gate/mul/xAmodel/comment_slide_down_score_logit_cluster_gate/dense_1/Sigmoid*
T0
�
Pmio_variable/comment_slide_down_score_logit_cluster_gate/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*I
	container<:comment_slide_down_score_logit_cluster_gate/dense_2/kernel*
shape:	�@
�
Pmio_variable/comment_slide_down_score_logit_cluster_gate/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�@*I
	container<:comment_slide_down_score_logit_cluster_gate/dense_2/kernel
Y
$Initializer_164/random_uniform/shapeConst*
valueB"�   @   *
dtype0
O
"Initializer_164/random_uniform/minConst*
dtype0*
valueB
 *�5�
O
"Initializer_164/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_164/random_uniform/RandomUniformRandomUniform$Initializer_164/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_164/random_uniform/subSub"Initializer_164/random_uniform/max"Initializer_164/random_uniform/min*
T0
�
"Initializer_164/random_uniform/mulMul,Initializer_164/random_uniform/RandomUniform"Initializer_164/random_uniform/sub*
T0
v
Initializer_164/random_uniformAdd"Initializer_164/random_uniform/mul"Initializer_164/random_uniform/min*
T0
�

Assign_164AssignPmio_variable/comment_slide_down_score_logit_cluster_gate/dense_2/kernel/gradientInitializer_164/random_uniform*
use_locking(*
T0*c
_classY
WUloc:@mio_variable/comment_slide_down_score_logit_cluster_gate/dense_2/kernel/gradient*
validate_shape(
�
Nmio_variable/comment_slide_down_score_logit_cluster_gate/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*G
	container:8comment_slide_down_score_logit_cluster_gate/dense_2/bias
�
Nmio_variable/comment_slide_down_score_logit_cluster_gate/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*G
	container:8comment_slide_down_score_logit_cluster_gate/dense_2/bias
F
Initializer_165/zerosConst*
dtype0*
valueB@*    
�

Assign_165AssignNmio_variable/comment_slide_down_score_logit_cluster_gate/dense_2/bias/gradientInitializer_165/zeros*
validate_shape(*
use_locking(*
T0*a
_classW
USloc:@mio_variable/comment_slide_down_score_logit_cluster_gate/dense_2/bias/gradient
�
@model/comment_slide_down_score_logit_cluster_gate/dense_2/MatMulMatMulAmodel/comment_slide_down_score_logit_cluster_gate/dense_1/SigmoidPmio_variable/comment_slide_down_score_logit_cluster_gate/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
Amodel/comment_slide_down_score_logit_cluster_gate/dense_2/BiasAddBiasAdd@model/comment_slide_down_score_logit_cluster_gate/dense_2/MatMulNmio_variable/comment_slide_down_score_logit_cluster_gate/dense_2/bias/variable*
T0*
data_formatNHWC
�
>model/comment_slide_down_score_logit_cluster_gate/dense_2/ReluReluAmodel/comment_slide_down_score_logit_cluster_gate/dense_2/BiasAdd*
T0
�
Pmio_variable/comment_slide_down_score_logit_cluster_gate/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@@*I
	container<:comment_slide_down_score_logit_cluster_gate/dense_3/kernel
�
Pmio_variable/comment_slide_down_score_logit_cluster_gate/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*I
	container<:comment_slide_down_score_logit_cluster_gate/dense_3/kernel*
shape
:@@
Y
$Initializer_166/random_uniform/shapeConst*
dtype0*
valueB"@   @   
O
"Initializer_166/random_uniform/minConst*
valueB
 *׳]�*
dtype0
O
"Initializer_166/random_uniform/maxConst*
dtype0*
valueB
 *׳]>
�
,Initializer_166/random_uniform/RandomUniformRandomUniform$Initializer_166/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_166/random_uniform/subSub"Initializer_166/random_uniform/max"Initializer_166/random_uniform/min*
T0
�
"Initializer_166/random_uniform/mulMul,Initializer_166/random_uniform/RandomUniform"Initializer_166/random_uniform/sub*
T0
v
Initializer_166/random_uniformAdd"Initializer_166/random_uniform/mul"Initializer_166/random_uniform/min*
T0
�

Assign_166AssignPmio_variable/comment_slide_down_score_logit_cluster_gate/dense_3/kernel/gradientInitializer_166/random_uniform*
validate_shape(*
use_locking(*
T0*c
_classY
WUloc:@mio_variable/comment_slide_down_score_logit_cluster_gate/dense_3/kernel/gradient
�
Nmio_variable/comment_slide_down_score_logit_cluster_gate/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*G
	container:8comment_slide_down_score_logit_cluster_gate/dense_3/bias
�
Nmio_variable/comment_slide_down_score_logit_cluster_gate/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*G
	container:8comment_slide_down_score_logit_cluster_gate/dense_3/bias
F
Initializer_167/zerosConst*
valueB@*    *
dtype0
�

Assign_167AssignNmio_variable/comment_slide_down_score_logit_cluster_gate/dense_3/bias/gradientInitializer_167/zeros*
use_locking(*
T0*a
_classW
USloc:@mio_variable/comment_slide_down_score_logit_cluster_gate/dense_3/bias/gradient*
validate_shape(
�
@model/comment_slide_down_score_logit_cluster_gate/dense_3/MatMulMatMul>model/comment_slide_down_score_logit_cluster_gate/dense_2/ReluPmio_variable/comment_slide_down_score_logit_cluster_gate/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
Amodel/comment_slide_down_score_logit_cluster_gate/dense_3/BiasAddBiasAdd@model/comment_slide_down_score_logit_cluster_gate/dense_3/MatMulNmio_variable/comment_slide_down_score_logit_cluster_gate/dense_3/bias/variable*
T0*
data_formatNHWC
�
Amodel/comment_slide_down_score_logit_cluster_gate/dense_3/SigmoidSigmoidAmodel/comment_slide_down_score_logit_cluster_gate/dense_3/BiasAdd*
T0
f
9model/comment_slide_down_score_logit_cluster_gate/mul_1/xConst*
dtype0*
valueB
 *   @
�
7model/comment_slide_down_score_logit_cluster_gate/mul_1Mul9model/comment_slide_down_score_logit_cluster_gate/mul_1/xAmodel/comment_slide_down_score_logit_cluster_gate/dense_3/Sigmoid*
T0
�
(model/comment_slide_down_score_logit/MulMul'model/comment_top_net/dense_1/LeakyRelu5model/comment_slide_down_score_logit_cluster_gate/mul*
T0
�
Amio_variable/comment_slide_down_score_logit/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+comment_slide_down_score_logit/dense/kernel*
shape:	�@
�
Amio_variable/comment_slide_down_score_logit/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+comment_slide_down_score_logit/dense/kernel*
shape:	�@
Y
$Initializer_168/random_uniform/shapeConst*
valueB"�   @   *
dtype0
O
"Initializer_168/random_uniform/minConst*
valueB
 *�5�*
dtype0
O
"Initializer_168/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_168/random_uniform/RandomUniformRandomUniform$Initializer_168/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_168/random_uniform/subSub"Initializer_168/random_uniform/max"Initializer_168/random_uniform/min*
T0
�
"Initializer_168/random_uniform/mulMul,Initializer_168/random_uniform/RandomUniform"Initializer_168/random_uniform/sub*
T0
v
Initializer_168/random_uniformAdd"Initializer_168/random_uniform/mul"Initializer_168/random_uniform/min*
T0
�

Assign_168AssignAmio_variable/comment_slide_down_score_logit/dense/kernel/gradientInitializer_168/random_uniform*
use_locking(*
T0*T
_classJ
HFloc:@mio_variable/comment_slide_down_score_logit/dense/kernel/gradient*
validate_shape(
�
?mio_variable/comment_slide_down_score_logit/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*8
	container+)comment_slide_down_score_logit/dense/bias*
shape:@
�
?mio_variable/comment_slide_down_score_logit/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*8
	container+)comment_slide_down_score_logit/dense/bias*
shape:@
F
Initializer_169/zerosConst*
dtype0*
valueB@*    
�

Assign_169Assign?mio_variable/comment_slide_down_score_logit/dense/bias/gradientInitializer_169/zeros*
validate_shape(*
use_locking(*
T0*R
_classH
FDloc:@mio_variable/comment_slide_down_score_logit/dense/bias/gradient
�
1model/comment_slide_down_score_logit/dense/MatMulMatMul(model/comment_slide_down_score_logit/MulAmio_variable/comment_slide_down_score_logit/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
2model/comment_slide_down_score_logit/dense/BiasAddBiasAdd1model/comment_slide_down_score_logit/dense/MatMul?mio_variable/comment_slide_down_score_logit/dense/bias/variable*
T0*
data_formatNHWC
g
:model/comment_slide_down_score_logit/dense/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
8model/comment_slide_down_score_logit/dense/LeakyRelu/mulMul:model/comment_slide_down_score_logit/dense/LeakyRelu/alpha2model/comment_slide_down_score_logit/dense/BiasAdd*
T0
�
4model/comment_slide_down_score_logit/dense/LeakyReluMaximum8model/comment_slide_down_score_logit/dense/LeakyRelu/mul2model/comment_slide_down_score_logit/dense/BiasAdd*
T0
�
*model/comment_slide_down_score_logit/Mul_1Mul4model/comment_slide_down_score_logit/dense/LeakyRelu7model/comment_slide_down_score_logit_cluster_gate/mul_1*
T0
�
Cmio_variable/comment_slide_down_score_logit/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*<
	container/-comment_slide_down_score_logit/dense_1/kernel*
shape
:@
�
Cmio_variable/comment_slide_down_score_logit/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*<
	container/-comment_slide_down_score_logit/dense_1/kernel*
shape
:@
Y
$Initializer_170/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_170/random_uniform/minConst*
valueB
 *����*
dtype0
O
"Initializer_170/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
,Initializer_170/random_uniform/RandomUniformRandomUniform$Initializer_170/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_170/random_uniform/subSub"Initializer_170/random_uniform/max"Initializer_170/random_uniform/min*
T0
�
"Initializer_170/random_uniform/mulMul,Initializer_170/random_uniform/RandomUniform"Initializer_170/random_uniform/sub*
T0
v
Initializer_170/random_uniformAdd"Initializer_170/random_uniform/mul"Initializer_170/random_uniform/min*
T0
�

Assign_170AssignCmio_variable/comment_slide_down_score_logit/dense_1/kernel/gradientInitializer_170/random_uniform*
use_locking(*
T0*V
_classL
JHloc:@mio_variable/comment_slide_down_score_logit/dense_1/kernel/gradient*
validate_shape(
�
Amio_variable/comment_slide_down_score_logit/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+comment_slide_down_score_logit/dense_1/bias*
shape:
�
Amio_variable/comment_slide_down_score_logit/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+comment_slide_down_score_logit/dense_1/bias*
shape:
F
Initializer_171/zerosConst*
valueB*    *
dtype0
�

Assign_171AssignAmio_variable/comment_slide_down_score_logit/dense_1/bias/gradientInitializer_171/zeros*
use_locking(*
T0*T
_classJ
HFloc:@mio_variable/comment_slide_down_score_logit/dense_1/bias/gradient*
validate_shape(
�
3model/comment_slide_down_score_logit/dense_1/MatMulMatMul*model/comment_slide_down_score_logit/Mul_1Cmio_variable/comment_slide_down_score_logit/dense_1/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
4model/comment_slide_down_score_logit/dense_1/BiasAddBiasAdd3model/comment_slide_down_score_logit/dense_1/MatMulAmio_variable/comment_slide_down_score_logit/dense_1/bias/variable*
T0*
data_formatNHWC
i
<model/comment_slide_down_score_logit/dense_1/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
:model/comment_slide_down_score_logit/dense_1/LeakyRelu/mulMul<model/comment_slide_down_score_logit/dense_1/LeakyRelu/alpha4model/comment_slide_down_score_logit/dense_1/BiasAdd*
T0
�
6model/comment_slide_down_score_logit/dense_1/LeakyReluMaximum:model/comment_slide_down_score_logit/dense_1/LeakyRelu/mul4model/comment_slide_down_score_logit/dense_1/BiasAdd*
T0
�
0mio_variable/eft_click_cmt/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*)
	containereft_click_cmt/dense/kernel*
shape:	�@
�
0mio_variable/eft_click_cmt/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*)
	containereft_click_cmt/dense/kernel*
shape:	�@
Y
$Initializer_172/random_uniform/shapeConst*
dtype0*
valueB"�   @   
O
"Initializer_172/random_uniform/minConst*
valueB
 *�5�*
dtype0
O
"Initializer_172/random_uniform/maxConst*
dtype0*
valueB
 *�5>
�
,Initializer_172/random_uniform/RandomUniformRandomUniform$Initializer_172/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_172/random_uniform/subSub"Initializer_172/random_uniform/max"Initializer_172/random_uniform/min*
T0
�
"Initializer_172/random_uniform/mulMul,Initializer_172/random_uniform/RandomUniform"Initializer_172/random_uniform/sub*
T0
v
Initializer_172/random_uniformAdd"Initializer_172/random_uniform/mul"Initializer_172/random_uniform/min*
T0
�

Assign_172Assign0mio_variable/eft_click_cmt/dense/kernel/gradientInitializer_172/random_uniform*
use_locking(*
T0*C
_class9
75loc:@mio_variable/eft_click_cmt/dense/kernel/gradient*
validate_shape(
�
.mio_variable/eft_click_cmt/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containereft_click_cmt/dense/bias*
shape:@
�
.mio_variable/eft_click_cmt/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containereft_click_cmt/dense/bias*
shape:@
F
Initializer_173/zerosConst*
valueB@*    *
dtype0
�

Assign_173Assign.mio_variable/eft_click_cmt/dense/bias/gradientInitializer_173/zeros*
use_locking(*
T0*A
_class7
53loc:@mio_variable/eft_click_cmt/dense/bias/gradient*
validate_shape(
�
 model/eft_click_cmt/dense/MatMulMatMul'model/comment_top_net/dense_1/LeakyRelu0mio_variable/eft_click_cmt/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
!model/eft_click_cmt/dense/BiasAddBiasAdd model/eft_click_cmt/dense/MatMul.mio_variable/eft_click_cmt/dense/bias/variable*
T0*
data_formatNHWC
V
)model/eft_click_cmt/dense/LeakyRelu/alphaConst*
dtype0*
valueB
 *��L>
�
'model/eft_click_cmt/dense/LeakyRelu/mulMul)model/eft_click_cmt/dense/LeakyRelu/alpha!model/eft_click_cmt/dense/BiasAdd*
T0
�
#model/eft_click_cmt/dense/LeakyReluMaximum'model/eft_click_cmt/dense/LeakyRelu/mul!model/eft_click_cmt/dense/BiasAdd*
T0
�
2mio_variable/eft_click_cmt/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*+
	containereft_click_cmt/dense_1/kernel
�
2mio_variable/eft_click_cmt/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*+
	containereft_click_cmt/dense_1/kernel
Y
$Initializer_174/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_174/random_uniform/minConst*
valueB
 *����*
dtype0
O
"Initializer_174/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
,Initializer_174/random_uniform/RandomUniformRandomUniform$Initializer_174/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_174/random_uniform/subSub"Initializer_174/random_uniform/max"Initializer_174/random_uniform/min*
T0
�
"Initializer_174/random_uniform/mulMul,Initializer_174/random_uniform/RandomUniform"Initializer_174/random_uniform/sub*
T0
v
Initializer_174/random_uniformAdd"Initializer_174/random_uniform/mul"Initializer_174/random_uniform/min*
T0
�

Assign_174Assign2mio_variable/eft_click_cmt/dense_1/kernel/gradientInitializer_174/random_uniform*
validate_shape(*
use_locking(*
T0*E
_class;
97loc:@mio_variable/eft_click_cmt/dense_1/kernel/gradient
�
0mio_variable/eft_click_cmt/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*)
	containereft_click_cmt/dense_1/bias*
shape:
�
0mio_variable/eft_click_cmt/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*)
	containereft_click_cmt/dense_1/bias*
shape:
F
Initializer_175/zerosConst*
valueB*    *
dtype0
�

Assign_175Assign0mio_variable/eft_click_cmt/dense_1/bias/gradientInitializer_175/zeros*
T0*C
_class9
75loc:@mio_variable/eft_click_cmt/dense_1/bias/gradient*
validate_shape(*
use_locking(
�
"model/eft_click_cmt/dense_1/MatMulMatMul#model/eft_click_cmt/dense/LeakyRelu2mio_variable/eft_click_cmt/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
#model/eft_click_cmt/dense_1/BiasAddBiasAdd"model/eft_click_cmt/dense_1/MatMul0mio_variable/eft_click_cmt/dense_1/bias/variable*
T0*
data_formatNHWC
�
0mio_variable/eft_write_cmt/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*)
	containereft_write_cmt/dense/kernel*
shape:	�@
�
0mio_variable/eft_write_cmt/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*)
	containereft_write_cmt/dense/kernel*
shape:	�@
Y
$Initializer_176/random_uniform/shapeConst*
valueB"�   @   *
dtype0
O
"Initializer_176/random_uniform/minConst*
valueB
 *�5�*
dtype0
O
"Initializer_176/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_176/random_uniform/RandomUniformRandomUniform$Initializer_176/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_176/random_uniform/subSub"Initializer_176/random_uniform/max"Initializer_176/random_uniform/min*
T0
�
"Initializer_176/random_uniform/mulMul,Initializer_176/random_uniform/RandomUniform"Initializer_176/random_uniform/sub*
T0
v
Initializer_176/random_uniformAdd"Initializer_176/random_uniform/mul"Initializer_176/random_uniform/min*
T0
�

Assign_176Assign0mio_variable/eft_write_cmt/dense/kernel/gradientInitializer_176/random_uniform*
use_locking(*
T0*C
_class9
75loc:@mio_variable/eft_write_cmt/dense/kernel/gradient*
validate_shape(
�
.mio_variable/eft_write_cmt/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containereft_write_cmt/dense/bias*
shape:@
�
.mio_variable/eft_write_cmt/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*'
	containereft_write_cmt/dense/bias
F
Initializer_177/zerosConst*
valueB@*    *
dtype0
�

Assign_177Assign.mio_variable/eft_write_cmt/dense/bias/gradientInitializer_177/zeros*
use_locking(*
T0*A
_class7
53loc:@mio_variable/eft_write_cmt/dense/bias/gradient*
validate_shape(
�
 model/eft_write_cmt/dense/MatMulMatMul'model/comment_top_net/dense_1/LeakyRelu0mio_variable/eft_write_cmt/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
!model/eft_write_cmt/dense/BiasAddBiasAdd model/eft_write_cmt/dense/MatMul.mio_variable/eft_write_cmt/dense/bias/variable*
T0*
data_formatNHWC
V
)model/eft_write_cmt/dense/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
'model/eft_write_cmt/dense/LeakyRelu/mulMul)model/eft_write_cmt/dense/LeakyRelu/alpha!model/eft_write_cmt/dense/BiasAdd*
T0
�
#model/eft_write_cmt/dense/LeakyReluMaximum'model/eft_write_cmt/dense/LeakyRelu/mul!model/eft_write_cmt/dense/BiasAdd*
T0
�
2mio_variable/eft_write_cmt/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*+
	containereft_write_cmt/dense_1/kernel*
shape
:@
�
2mio_variable/eft_write_cmt/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*+
	containereft_write_cmt/dense_1/kernel
Y
$Initializer_178/random_uniform/shapeConst*
dtype0*
valueB"@      
O
"Initializer_178/random_uniform/minConst*
valueB
 *����*
dtype0
O
"Initializer_178/random_uniform/maxConst*
dtype0*
valueB
 *���>
�
,Initializer_178/random_uniform/RandomUniformRandomUniform$Initializer_178/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_178/random_uniform/subSub"Initializer_178/random_uniform/max"Initializer_178/random_uniform/min*
T0
�
"Initializer_178/random_uniform/mulMul,Initializer_178/random_uniform/RandomUniform"Initializer_178/random_uniform/sub*
T0
v
Initializer_178/random_uniformAdd"Initializer_178/random_uniform/mul"Initializer_178/random_uniform/min*
T0
�

Assign_178Assign2mio_variable/eft_write_cmt/dense_1/kernel/gradientInitializer_178/random_uniform*
T0*E
_class;
97loc:@mio_variable/eft_write_cmt/dense_1/kernel/gradient*
validate_shape(*
use_locking(
�
0mio_variable/eft_write_cmt/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*)
	containereft_write_cmt/dense_1/bias*
shape:
�
0mio_variable/eft_write_cmt/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*)
	containereft_write_cmt/dense_1/bias*
shape:
F
Initializer_179/zerosConst*
valueB*    *
dtype0
�

Assign_179Assign0mio_variable/eft_write_cmt/dense_1/bias/gradientInitializer_179/zeros*
use_locking(*
T0*C
_class9
75loc:@mio_variable/eft_write_cmt/dense_1/bias/gradient*
validate_shape(
�
"model/eft_write_cmt/dense_1/MatMulMatMul#model/eft_write_cmt/dense/LeakyRelu2mio_variable/eft_write_cmt/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
#model/eft_write_cmt/dense_1/BiasAddBiasAdd"model/eft_write_cmt/dense_1/MatMul0mio_variable/eft_write_cmt/dense_1/bias/variable*
T0*
data_formatNHWC
�
7mio_variable/comment_genre_layers/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!comment_genre_layers/dense/kernel*
shape:
��
�
7mio_variable/comment_genre_layers/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!comment_genre_layers/dense/kernel*
shape:
��
Y
$Initializer_180/random_uniform/shapeConst*
valueB"0     *
dtype0
O
"Initializer_180/random_uniform/minConst*
valueB
 *ܨ��*
dtype0
O
"Initializer_180/random_uniform/maxConst*
valueB
 *ܨ�=*
dtype0
�
,Initializer_180/random_uniform/RandomUniformRandomUniform$Initializer_180/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_180/random_uniform/subSub"Initializer_180/random_uniform/max"Initializer_180/random_uniform/min*
T0
�
"Initializer_180/random_uniform/mulMul,Initializer_180/random_uniform/RandomUniform"Initializer_180/random_uniform/sub*
T0
v
Initializer_180/random_uniformAdd"Initializer_180/random_uniform/mul"Initializer_180/random_uniform/min*
T0
�

Assign_180Assign7mio_variable/comment_genre_layers/dense/kernel/gradientInitializer_180/random_uniform*
validate_shape(*
use_locking(*
T0*J
_class@
><loc:@mio_variable/comment_genre_layers/dense/kernel/gradient
�
5mio_variable/comment_genre_layers/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!comment_genre_layers/dense/bias*
shape:�
�
5mio_variable/comment_genre_layers/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!comment_genre_layers/dense/bias*
shape:�
G
Initializer_181/zerosConst*
valueB�*    *
dtype0
�

Assign_181Assign5mio_variable/comment_genre_layers/dense/bias/gradientInitializer_181/zeros*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/comment_genre_layers/dense/bias/gradient*
validate_shape(
�
'model/comment_genre_layers/dense/MatMulMatMul	concat_117mio_variable/comment_genre_layers/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
(model/comment_genre_layers/dense/BiasAddBiasAdd'model/comment_genre_layers/dense/MatMul5mio_variable/comment_genre_layers/dense/bias/variable*
data_formatNHWC*
T0
]
0model/comment_genre_layers/dense/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
.model/comment_genre_layers/dense/LeakyRelu/mulMul0model/comment_genre_layers/dense/LeakyRelu/alpha(model/comment_genre_layers/dense/BiasAdd*
T0
�
*model/comment_genre_layers/dense/LeakyReluMaximum.model/comment_genre_layers/dense/LeakyRelu/mul(model/comment_genre_layers/dense/BiasAdd*
T0
�
9mio_variable/comment_genre_layers/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#comment_genre_layers/dense_1/kernel*
shape:
��
�
9mio_variable/comment_genre_layers/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
��*2
	container%#comment_genre_layers/dense_1/kernel
Y
$Initializer_182/random_uniform/shapeConst*
valueB"   �   *
dtype0
O
"Initializer_182/random_uniform/minConst*
valueB
 *   �*
dtype0
O
"Initializer_182/random_uniform/maxConst*
valueB
 *   >*
dtype0
�
,Initializer_182/random_uniform/RandomUniformRandomUniform$Initializer_182/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_182/random_uniform/subSub"Initializer_182/random_uniform/max"Initializer_182/random_uniform/min*
T0
�
"Initializer_182/random_uniform/mulMul,Initializer_182/random_uniform/RandomUniform"Initializer_182/random_uniform/sub*
T0
v
Initializer_182/random_uniformAdd"Initializer_182/random_uniform/mul"Initializer_182/random_uniform/min*
T0
�

Assign_182Assign9mio_variable/comment_genre_layers/dense_1/kernel/gradientInitializer_182/random_uniform*
use_locking(*
T0*L
_classB
@>loc:@mio_variable/comment_genre_layers/dense_1/kernel/gradient*
validate_shape(
�
7mio_variable/comment_genre_layers/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!comment_genre_layers/dense_1/bias*
shape:�
�
7mio_variable/comment_genre_layers/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!comment_genre_layers/dense_1/bias*
shape:�
G
Initializer_183/zerosConst*
valueB�*    *
dtype0
�

Assign_183Assign7mio_variable/comment_genre_layers/dense_1/bias/gradientInitializer_183/zeros*
T0*J
_class@
><loc:@mio_variable/comment_genre_layers/dense_1/bias/gradient*
validate_shape(*
use_locking(
�
)model/comment_genre_layers/dense_1/MatMulMatMul*model/comment_genre_layers/dense/LeakyRelu9mio_variable/comment_genre_layers/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
*model/comment_genre_layers/dense_1/BiasAddBiasAdd)model/comment_genre_layers/dense_1/MatMul7mio_variable/comment_genre_layers/dense_1/bias/variable*
data_formatNHWC*
T0
_
2model/comment_genre_layers/dense_1/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
0model/comment_genre_layers/dense_1/LeakyRelu/mulMul2model/comment_genre_layers/dense_1/LeakyRelu/alpha*model/comment_genre_layers/dense_1/BiasAdd*
T0
�
,model/comment_genre_layers/dense_1/LeakyReluMaximum0model/comment_genre_layers/dense_1/LeakyRelu/mul*model/comment_genre_layers/dense_1/BiasAdd*
T0
�
5mio_variable/sub_comment_layers/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�@*.
	container!sub_comment_layers/dense/kernel
�
5mio_variable/sub_comment_layers/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!sub_comment_layers/dense/kernel*
shape:	�@
Y
$Initializer_184/random_uniform/shapeConst*
dtype0*
valueB"�   @   
O
"Initializer_184/random_uniform/minConst*
dtype0*
valueB
 *�5�
O
"Initializer_184/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_184/random_uniform/RandomUniformRandomUniform$Initializer_184/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_184/random_uniform/subSub"Initializer_184/random_uniform/max"Initializer_184/random_uniform/min*
T0
�
"Initializer_184/random_uniform/mulMul,Initializer_184/random_uniform/RandomUniform"Initializer_184/random_uniform/sub*
T0
v
Initializer_184/random_uniformAdd"Initializer_184/random_uniform/mul"Initializer_184/random_uniform/min*
T0
�

Assign_184Assign5mio_variable/sub_comment_layers/dense/kernel/gradientInitializer_184/random_uniform*
T0*H
_class>
<:loc:@mio_variable/sub_comment_layers/dense/kernel/gradient*
validate_shape(*
use_locking(
�
3mio_variable/sub_comment_layers/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*,
	containersub_comment_layers/dense/bias*
shape:@
�
3mio_variable/sub_comment_layers/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*,
	containersub_comment_layers/dense/bias*
shape:@
F
Initializer_185/zerosConst*
dtype0*
valueB@*    
�

Assign_185Assign3mio_variable/sub_comment_layers/dense/bias/gradientInitializer_185/zeros*
validate_shape(*
use_locking(*
T0*F
_class<
:8loc:@mio_variable/sub_comment_layers/dense/bias/gradient
�
%model/sub_comment_layers/dense/MatMulMatMul,model/comment_genre_layers/dense_1/LeakyRelu5mio_variable/sub_comment_layers/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
&model/sub_comment_layers/dense/BiasAddBiasAdd%model/sub_comment_layers/dense/MatMul3mio_variable/sub_comment_layers/dense/bias/variable*
T0*
data_formatNHWC
[
.model/sub_comment_layers/dense/LeakyRelu/alphaConst*
dtype0*
valueB
 *��L>
�
,model/sub_comment_layers/dense/LeakyRelu/mulMul.model/sub_comment_layers/dense/LeakyRelu/alpha&model/sub_comment_layers/dense/BiasAdd*
T0
�
(model/sub_comment_layers/dense/LeakyReluMaximum,model/sub_comment_layers/dense/LeakyRelu/mul&model/sub_comment_layers/dense/BiasAdd*
T0
�
7mio_variable/sub_comment_layers/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*0
	container#!sub_comment_layers/dense_1/kernel
�
7mio_variable/sub_comment_layers/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!sub_comment_layers/dense_1/kernel*
shape
:@
Y
$Initializer_186/random_uniform/shapeConst*
dtype0*
valueB"@      
O
"Initializer_186/random_uniform/minConst*
dtype0*
valueB
 *����
O
"Initializer_186/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
,Initializer_186/random_uniform/RandomUniformRandomUniform$Initializer_186/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_186/random_uniform/subSub"Initializer_186/random_uniform/max"Initializer_186/random_uniform/min*
T0
�
"Initializer_186/random_uniform/mulMul,Initializer_186/random_uniform/RandomUniform"Initializer_186/random_uniform/sub*
T0
v
Initializer_186/random_uniformAdd"Initializer_186/random_uniform/mul"Initializer_186/random_uniform/min*
T0
�

Assign_186Assign7mio_variable/sub_comment_layers/dense_1/kernel/gradientInitializer_186/random_uniform*
validate_shape(*
use_locking(*
T0*J
_class@
><loc:@mio_variable/sub_comment_layers/dense_1/kernel/gradient
�
5mio_variable/sub_comment_layers/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!sub_comment_layers/dense_1/bias*
shape:
�
5mio_variable/sub_comment_layers/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!sub_comment_layers/dense_1/bias*
shape:
F
Initializer_187/zerosConst*
valueB*    *
dtype0
�

Assign_187Assign5mio_variable/sub_comment_layers/dense_1/bias/gradientInitializer_187/zeros*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/sub_comment_layers/dense_1/bias/gradient*
validate_shape(
�
'model/sub_comment_layers/dense_1/MatMulMatMul(model/sub_comment_layers/dense/LeakyRelu7mio_variable/sub_comment_layers/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
(model/sub_comment_layers/dense_1/BiasAddBiasAdd'model/sub_comment_layers/dense_1/MatMul5mio_variable/sub_comment_layers/dense_1/bias/variable*
T0*
data_formatNHWC
�
7mio_variable/emoji_comment_layers/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!emoji_comment_layers/dense/kernel*
shape:	�@
�
7mio_variable/emoji_comment_layers/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!emoji_comment_layers/dense/kernel*
shape:	�@
Y
$Initializer_188/random_uniform/shapeConst*
valueB"�   @   *
dtype0
O
"Initializer_188/random_uniform/minConst*
valueB
 *�5�*
dtype0
O
"Initializer_188/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_188/random_uniform/RandomUniformRandomUniform$Initializer_188/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_188/random_uniform/subSub"Initializer_188/random_uniform/max"Initializer_188/random_uniform/min*
T0
�
"Initializer_188/random_uniform/mulMul,Initializer_188/random_uniform/RandomUniform"Initializer_188/random_uniform/sub*
T0
v
Initializer_188/random_uniformAdd"Initializer_188/random_uniform/mul"Initializer_188/random_uniform/min*
T0
�

Assign_188Assign7mio_variable/emoji_comment_layers/dense/kernel/gradientInitializer_188/random_uniform*
use_locking(*
T0*J
_class@
><loc:@mio_variable/emoji_comment_layers/dense/kernel/gradient*
validate_shape(
�
5mio_variable/emoji_comment_layers/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!emoji_comment_layers/dense/bias*
shape:@
�
5mio_variable/emoji_comment_layers/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!emoji_comment_layers/dense/bias*
shape:@
F
Initializer_189/zerosConst*
valueB@*    *
dtype0
�

Assign_189Assign5mio_variable/emoji_comment_layers/dense/bias/gradientInitializer_189/zeros*
T0*H
_class>
<:loc:@mio_variable/emoji_comment_layers/dense/bias/gradient*
validate_shape(*
use_locking(
�
'model/emoji_comment_layers/dense/MatMulMatMul,model/comment_genre_layers/dense_1/LeakyRelu7mio_variable/emoji_comment_layers/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
(model/emoji_comment_layers/dense/BiasAddBiasAdd'model/emoji_comment_layers/dense/MatMul5mio_variable/emoji_comment_layers/dense/bias/variable*
T0*
data_formatNHWC
]
0model/emoji_comment_layers/dense/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
.model/emoji_comment_layers/dense/LeakyRelu/mulMul0model/emoji_comment_layers/dense/LeakyRelu/alpha(model/emoji_comment_layers/dense/BiasAdd*
T0
�
*model/emoji_comment_layers/dense/LeakyReluMaximum.model/emoji_comment_layers/dense/LeakyRelu/mul(model/emoji_comment_layers/dense/BiasAdd*
T0
�
9mio_variable/emoji_comment_layers/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*2
	container%#emoji_comment_layers/dense_1/kernel
�
9mio_variable/emoji_comment_layers/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#emoji_comment_layers/dense_1/kernel*
shape
:@
Y
$Initializer_190/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_190/random_uniform/minConst*
valueB
 *����*
dtype0
O
"Initializer_190/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
,Initializer_190/random_uniform/RandomUniformRandomUniform$Initializer_190/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_190/random_uniform/subSub"Initializer_190/random_uniform/max"Initializer_190/random_uniform/min*
T0
�
"Initializer_190/random_uniform/mulMul,Initializer_190/random_uniform/RandomUniform"Initializer_190/random_uniform/sub*
T0
v
Initializer_190/random_uniformAdd"Initializer_190/random_uniform/mul"Initializer_190/random_uniform/min*
T0
�

Assign_190Assign9mio_variable/emoji_comment_layers/dense_1/kernel/gradientInitializer_190/random_uniform*
T0*L
_classB
@>loc:@mio_variable/emoji_comment_layers/dense_1/kernel/gradient*
validate_shape(*
use_locking(
�
7mio_variable/emoji_comment_layers/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*0
	container#!emoji_comment_layers/dense_1/bias
�
7mio_variable/emoji_comment_layers/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!emoji_comment_layers/dense_1/bias*
shape:
F
Initializer_191/zerosConst*
valueB*    *
dtype0
�

Assign_191Assign7mio_variable/emoji_comment_layers/dense_1/bias/gradientInitializer_191/zeros*
validate_shape(*
use_locking(*
T0*J
_class@
><loc:@mio_variable/emoji_comment_layers/dense_1/bias/gradient
�
)model/emoji_comment_layers/dense_1/MatMulMatMul*model/emoji_comment_layers/dense/LeakyRelu9mio_variable/emoji_comment_layers/dense_1/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
*model/emoji_comment_layers/dense_1/BiasAddBiasAdd)model/emoji_comment_layers/dense_1/MatMul7mio_variable/emoji_comment_layers/dense_1/bias/variable*
data_formatNHWC*
T0
�
5mio_variable/gif_comment_layers/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!gif_comment_layers/dense/kernel*
shape:	�@
�
5mio_variable/gif_comment_layers/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!gif_comment_layers/dense/kernel*
shape:	�@
Y
$Initializer_192/random_uniform/shapeConst*
valueB"�   @   *
dtype0
O
"Initializer_192/random_uniform/minConst*
valueB
 *�5�*
dtype0
O
"Initializer_192/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_192/random_uniform/RandomUniformRandomUniform$Initializer_192/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_192/random_uniform/subSub"Initializer_192/random_uniform/max"Initializer_192/random_uniform/min*
T0
�
"Initializer_192/random_uniform/mulMul,Initializer_192/random_uniform/RandomUniform"Initializer_192/random_uniform/sub*
T0
v
Initializer_192/random_uniformAdd"Initializer_192/random_uniform/mul"Initializer_192/random_uniform/min*
T0
�

Assign_192Assign5mio_variable/gif_comment_layers/dense/kernel/gradientInitializer_192/random_uniform*
T0*H
_class>
<:loc:@mio_variable/gif_comment_layers/dense/kernel/gradient*
validate_shape(*
use_locking(
�
3mio_variable/gif_comment_layers/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*,
	containergif_comment_layers/dense/bias*
shape:@
�
3mio_variable/gif_comment_layers/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*,
	containergif_comment_layers/dense/bias*
shape:@
F
Initializer_193/zerosConst*
dtype0*
valueB@*    
�

Assign_193Assign3mio_variable/gif_comment_layers/dense/bias/gradientInitializer_193/zeros*
validate_shape(*
use_locking(*
T0*F
_class<
:8loc:@mio_variable/gif_comment_layers/dense/bias/gradient
�
%model/gif_comment_layers/dense/MatMulMatMul,model/comment_genre_layers/dense_1/LeakyRelu5mio_variable/gif_comment_layers/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
&model/gif_comment_layers/dense/BiasAddBiasAdd%model/gif_comment_layers/dense/MatMul3mio_variable/gif_comment_layers/dense/bias/variable*
T0*
data_formatNHWC
[
.model/gif_comment_layers/dense/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
,model/gif_comment_layers/dense/LeakyRelu/mulMul.model/gif_comment_layers/dense/LeakyRelu/alpha&model/gif_comment_layers/dense/BiasAdd*
T0
�
(model/gif_comment_layers/dense/LeakyReluMaximum,model/gif_comment_layers/dense/LeakyRelu/mul&model/gif_comment_layers/dense/BiasAdd*
T0
�
7mio_variable/gif_comment_layers/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!gif_comment_layers/dense_1/kernel*
shape
:@
�
7mio_variable/gif_comment_layers/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*0
	container#!gif_comment_layers/dense_1/kernel
Y
$Initializer_194/random_uniform/shapeConst*
dtype0*
valueB"@      
O
"Initializer_194/random_uniform/minConst*
dtype0*
valueB
 *����
O
"Initializer_194/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
,Initializer_194/random_uniform/RandomUniformRandomUniform$Initializer_194/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_194/random_uniform/subSub"Initializer_194/random_uniform/max"Initializer_194/random_uniform/min*
T0
�
"Initializer_194/random_uniform/mulMul,Initializer_194/random_uniform/RandomUniform"Initializer_194/random_uniform/sub*
T0
v
Initializer_194/random_uniformAdd"Initializer_194/random_uniform/mul"Initializer_194/random_uniform/min*
T0
�

Assign_194Assign7mio_variable/gif_comment_layers/dense_1/kernel/gradientInitializer_194/random_uniform*
validate_shape(*
use_locking(*
T0*J
_class@
><loc:@mio_variable/gif_comment_layers/dense_1/kernel/gradient
�
5mio_variable/gif_comment_layers/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!gif_comment_layers/dense_1/bias*
shape:
�
5mio_variable/gif_comment_layers/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!gif_comment_layers/dense_1/bias*
shape:
F
Initializer_195/zerosConst*
dtype0*
valueB*    
�

Assign_195Assign5mio_variable/gif_comment_layers/dense_1/bias/gradientInitializer_195/zeros*
validate_shape(*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/gif_comment_layers/dense_1/bias/gradient
�
'model/gif_comment_layers/dense_1/MatMulMatMul(model/gif_comment_layers/dense/LeakyRelu7mio_variable/gif_comment_layers/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
(model/gif_comment_layers/dense_1/BiasAddBiasAdd'model/gif_comment_layers/dense_1/MatMul5mio_variable/gif_comment_layers/dense_1/bias/variable*
T0*
data_formatNHWC
�
4mio_variable/at_comment_layers/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*-
	container at_comment_layers/dense/kernel*
shape:	�@
�
4mio_variable/at_comment_layers/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*-
	container at_comment_layers/dense/kernel*
shape:	�@
Y
$Initializer_196/random_uniform/shapeConst*
valueB"�   @   *
dtype0
O
"Initializer_196/random_uniform/minConst*
valueB
 *�5�*
dtype0
O
"Initializer_196/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_196/random_uniform/RandomUniformRandomUniform$Initializer_196/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_196/random_uniform/subSub"Initializer_196/random_uniform/max"Initializer_196/random_uniform/min*
T0
�
"Initializer_196/random_uniform/mulMul,Initializer_196/random_uniform/RandomUniform"Initializer_196/random_uniform/sub*
T0
v
Initializer_196/random_uniformAdd"Initializer_196/random_uniform/mul"Initializer_196/random_uniform/min*
T0
�

Assign_196Assign4mio_variable/at_comment_layers/dense/kernel/gradientInitializer_196/random_uniform*
use_locking(*
T0*G
_class=
;9loc:@mio_variable/at_comment_layers/dense/kernel/gradient*
validate_shape(
�
2mio_variable/at_comment_layers/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*+
	containerat_comment_layers/dense/bias
�
2mio_variable/at_comment_layers/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*+
	containerat_comment_layers/dense/bias
F
Initializer_197/zerosConst*
valueB@*    *
dtype0
�

Assign_197Assign2mio_variable/at_comment_layers/dense/bias/gradientInitializer_197/zeros*
validate_shape(*
use_locking(*
T0*E
_class;
97loc:@mio_variable/at_comment_layers/dense/bias/gradient
�
$model/at_comment_layers/dense/MatMulMatMul,model/comment_genre_layers/dense_1/LeakyRelu4mio_variable/at_comment_layers/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
%model/at_comment_layers/dense/BiasAddBiasAdd$model/at_comment_layers/dense/MatMul2mio_variable/at_comment_layers/dense/bias/variable*
T0*
data_formatNHWC
Z
-model/at_comment_layers/dense/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
+model/at_comment_layers/dense/LeakyRelu/mulMul-model/at_comment_layers/dense/LeakyRelu/alpha%model/at_comment_layers/dense/BiasAdd*
T0
�
'model/at_comment_layers/dense/LeakyReluMaximum+model/at_comment_layers/dense/LeakyRelu/mul%model/at_comment_layers/dense/BiasAdd*
T0
�
6mio_variable/at_comment_layers/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" at_comment_layers/dense_1/kernel*
shape
:@
�
6mio_variable/at_comment_layers/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" at_comment_layers/dense_1/kernel*
shape
:@
Y
$Initializer_198/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_198/random_uniform/minConst*
dtype0*
valueB
 *����
O
"Initializer_198/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
,Initializer_198/random_uniform/RandomUniformRandomUniform$Initializer_198/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_198/random_uniform/subSub"Initializer_198/random_uniform/max"Initializer_198/random_uniform/min*
T0
�
"Initializer_198/random_uniform/mulMul,Initializer_198/random_uniform/RandomUniform"Initializer_198/random_uniform/sub*
T0
v
Initializer_198/random_uniformAdd"Initializer_198/random_uniform/mul"Initializer_198/random_uniform/min*
T0
�

Assign_198Assign6mio_variable/at_comment_layers/dense_1/kernel/gradientInitializer_198/random_uniform*
use_locking(*
T0*I
_class?
=;loc:@mio_variable/at_comment_layers/dense_1/kernel/gradient*
validate_shape(
�
4mio_variable/at_comment_layers/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*-
	container at_comment_layers/dense_1/bias
�
4mio_variable/at_comment_layers/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*-
	container at_comment_layers/dense_1/bias
F
Initializer_199/zerosConst*
dtype0*
valueB*    
�

Assign_199Assign4mio_variable/at_comment_layers/dense_1/bias/gradientInitializer_199/zeros*
validate_shape(*
use_locking(*
T0*G
_class=
;9loc:@mio_variable/at_comment_layers/dense_1/bias/gradient
�
&model/at_comment_layers/dense_1/MatMulMatMul'model/at_comment_layers/dense/LeakyRelu6mio_variable/at_comment_layers/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
'model/at_comment_layers/dense_1/BiasAddBiasAdd&model/at_comment_layers/dense_1/MatMul4mio_variable/at_comment_layers/dense_1/bias/variable*
T0*
data_formatNHWC
�
7mio_variable/image_comment_layers/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!image_comment_layers/dense/kernel*
shape:	�@
�
7mio_variable/image_comment_layers/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!image_comment_layers/dense/kernel*
shape:	�@
Y
$Initializer_200/random_uniform/shapeConst*
dtype0*
valueB"�   @   
O
"Initializer_200/random_uniform/minConst*
valueB
 *�5�*
dtype0
O
"Initializer_200/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_200/random_uniform/RandomUniformRandomUniform$Initializer_200/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_200/random_uniform/subSub"Initializer_200/random_uniform/max"Initializer_200/random_uniform/min*
T0
�
"Initializer_200/random_uniform/mulMul,Initializer_200/random_uniform/RandomUniform"Initializer_200/random_uniform/sub*
T0
v
Initializer_200/random_uniformAdd"Initializer_200/random_uniform/mul"Initializer_200/random_uniform/min*
T0
�

Assign_200Assign7mio_variable/image_comment_layers/dense/kernel/gradientInitializer_200/random_uniform*
validate_shape(*
use_locking(*
T0*J
_class@
><loc:@mio_variable/image_comment_layers/dense/kernel/gradient
�
5mio_variable/image_comment_layers/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!image_comment_layers/dense/bias*
shape:@
�
5mio_variable/image_comment_layers/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!image_comment_layers/dense/bias*
shape:@
F
Initializer_201/zerosConst*
valueB@*    *
dtype0
�

Assign_201Assign5mio_variable/image_comment_layers/dense/bias/gradientInitializer_201/zeros*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/image_comment_layers/dense/bias/gradient*
validate_shape(
�
'model/image_comment_layers/dense/MatMulMatMul,model/comment_genre_layers/dense_1/LeakyRelu7mio_variable/image_comment_layers/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
(model/image_comment_layers/dense/BiasAddBiasAdd'model/image_comment_layers/dense/MatMul5mio_variable/image_comment_layers/dense/bias/variable*
data_formatNHWC*
T0
]
0model/image_comment_layers/dense/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
�
.model/image_comment_layers/dense/LeakyRelu/mulMul0model/image_comment_layers/dense/LeakyRelu/alpha(model/image_comment_layers/dense/BiasAdd*
T0
�
*model/image_comment_layers/dense/LeakyReluMaximum.model/image_comment_layers/dense/LeakyRelu/mul(model/image_comment_layers/dense/BiasAdd*
T0
�
9mio_variable/image_comment_layers/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#image_comment_layers/dense_1/kernel*
shape
:@
�
9mio_variable/image_comment_layers/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*2
	container%#image_comment_layers/dense_1/kernel
Y
$Initializer_202/random_uniform/shapeConst*
dtype0*
valueB"@      
O
"Initializer_202/random_uniform/minConst*
valueB
 *����*
dtype0
O
"Initializer_202/random_uniform/maxConst*
dtype0*
valueB
 *���>
�
,Initializer_202/random_uniform/RandomUniformRandomUniform$Initializer_202/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_202/random_uniform/subSub"Initializer_202/random_uniform/max"Initializer_202/random_uniform/min*
T0
�
"Initializer_202/random_uniform/mulMul,Initializer_202/random_uniform/RandomUniform"Initializer_202/random_uniform/sub*
T0
v
Initializer_202/random_uniformAdd"Initializer_202/random_uniform/mul"Initializer_202/random_uniform/min*
T0
�

Assign_202Assign9mio_variable/image_comment_layers/dense_1/kernel/gradientInitializer_202/random_uniform*
use_locking(*
T0*L
_classB
@>loc:@mio_variable/image_comment_layers/dense_1/kernel/gradient*
validate_shape(
�
7mio_variable/image_comment_layers/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*0
	container#!image_comment_layers/dense_1/bias
�
7mio_variable/image_comment_layers/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*0
	container#!image_comment_layers/dense_1/bias
F
Initializer_203/zerosConst*
valueB*    *
dtype0
�

Assign_203Assign7mio_variable/image_comment_layers/dense_1/bias/gradientInitializer_203/zeros*
T0*J
_class@
><loc:@mio_variable/image_comment_layers/dense_1/bias/gradient*
validate_shape(*
use_locking(
�
)model/image_comment_layers/dense_1/MatMulMatMul*model/image_comment_layers/dense/LeakyRelu9mio_variable/image_comment_layers/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
*model/image_comment_layers/dense_1/BiasAddBiasAdd)model/image_comment_layers/dense_1/MatMul7mio_variable/image_comment_layers/dense_1/bias/variable*
data_formatNHWC*
T0
�
6mio_variable/text_comment_layers/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" text_comment_layers/dense/kernel*
shape:	�@
�
6mio_variable/text_comment_layers/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�@*/
	container" text_comment_layers/dense/kernel
Y
$Initializer_204/random_uniform/shapeConst*
valueB"�   @   *
dtype0
O
"Initializer_204/random_uniform/minConst*
dtype0*
valueB
 *�5�
O
"Initializer_204/random_uniform/maxConst*
dtype0*
valueB
 *�5>
�
,Initializer_204/random_uniform/RandomUniformRandomUniform$Initializer_204/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_204/random_uniform/subSub"Initializer_204/random_uniform/max"Initializer_204/random_uniform/min*
T0
�
"Initializer_204/random_uniform/mulMul,Initializer_204/random_uniform/RandomUniform"Initializer_204/random_uniform/sub*
T0
v
Initializer_204/random_uniformAdd"Initializer_204/random_uniform/mul"Initializer_204/random_uniform/min*
T0
�

Assign_204Assign6mio_variable/text_comment_layers/dense/kernel/gradientInitializer_204/random_uniform*
use_locking(*
T0*I
_class?
=;loc:@mio_variable/text_comment_layers/dense/kernel/gradient*
validate_shape(
�
4mio_variable/text_comment_layers/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*-
	container text_comment_layers/dense/bias*
shape:@
�
4mio_variable/text_comment_layers/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*-
	container text_comment_layers/dense/bias*
shape:@
F
Initializer_205/zerosConst*
valueB@*    *
dtype0
�

Assign_205Assign4mio_variable/text_comment_layers/dense/bias/gradientInitializer_205/zeros*
use_locking(*
T0*G
_class=
;9loc:@mio_variable/text_comment_layers/dense/bias/gradient*
validate_shape(
�
&model/text_comment_layers/dense/MatMulMatMul,model/comment_genre_layers/dense_1/LeakyRelu6mio_variable/text_comment_layers/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
'model/text_comment_layers/dense/BiasAddBiasAdd&model/text_comment_layers/dense/MatMul4mio_variable/text_comment_layers/dense/bias/variable*
T0*
data_formatNHWC
\
/model/text_comment_layers/dense/LeakyRelu/alphaConst*
dtype0*
valueB
 *��L>
�
-model/text_comment_layers/dense/LeakyRelu/mulMul/model/text_comment_layers/dense/LeakyRelu/alpha'model/text_comment_layers/dense/BiasAdd*
T0
�
)model/text_comment_layers/dense/LeakyReluMaximum-model/text_comment_layers/dense/LeakyRelu/mul'model/text_comment_layers/dense/BiasAdd*
T0
�
8mio_variable/text_comment_layers/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"text_comment_layers/dense_1/kernel*
shape
:@
�
8mio_variable/text_comment_layers/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*1
	container$"text_comment_layers/dense_1/kernel
Y
$Initializer_206/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_206/random_uniform/minConst*
valueB
 *����*
dtype0
O
"Initializer_206/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
,Initializer_206/random_uniform/RandomUniformRandomUniform$Initializer_206/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_206/random_uniform/subSub"Initializer_206/random_uniform/max"Initializer_206/random_uniform/min*
T0
�
"Initializer_206/random_uniform/mulMul,Initializer_206/random_uniform/RandomUniform"Initializer_206/random_uniform/sub*
T0
v
Initializer_206/random_uniformAdd"Initializer_206/random_uniform/mul"Initializer_206/random_uniform/min*
T0
�

Assign_206Assign8mio_variable/text_comment_layers/dense_1/kernel/gradientInitializer_206/random_uniform*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/text_comment_layers/dense_1/kernel/gradient*
validate_shape(
�
6mio_variable/text_comment_layers/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" text_comment_layers/dense_1/bias*
shape:
�
6mio_variable/text_comment_layers/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" text_comment_layers/dense_1/bias*
shape:
F
Initializer_207/zerosConst*
valueB*    *
dtype0
�

Assign_207Assign6mio_variable/text_comment_layers/dense_1/bias/gradientInitializer_207/zeros*
validate_shape(*
use_locking(*
T0*I
_class?
=;loc:@mio_variable/text_comment_layers/dense_1/bias/gradient
�
(model/text_comment_layers/dense_1/MatMulMatMul)model/text_comment_layers/dense/LeakyRelu8mio_variable/text_comment_layers/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
)model/text_comment_layers/dense_1/BiasAddBiasAdd(model/text_comment_layers/dense_1/MatMul6mio_variable/text_comment_layers/dense_1/bias/variable*
data_formatNHWC*
T0
�
7mio_variable/video_comment_layers/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�@*0
	container#!video_comment_layers/dense/kernel
�
7mio_variable/video_comment_layers/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!video_comment_layers/dense/kernel*
shape:	�@
Y
$Initializer_208/random_uniform/shapeConst*
valueB"�   @   *
dtype0
O
"Initializer_208/random_uniform/minConst*
dtype0*
valueB
 *�5�
O
"Initializer_208/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_208/random_uniform/RandomUniformRandomUniform$Initializer_208/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_208/random_uniform/subSub"Initializer_208/random_uniform/max"Initializer_208/random_uniform/min*
T0
�
"Initializer_208/random_uniform/mulMul,Initializer_208/random_uniform/RandomUniform"Initializer_208/random_uniform/sub*
T0
v
Initializer_208/random_uniformAdd"Initializer_208/random_uniform/mul"Initializer_208/random_uniform/min*
T0
�

Assign_208Assign7mio_variable/video_comment_layers/dense/kernel/gradientInitializer_208/random_uniform*
use_locking(*
T0*J
_class@
><loc:@mio_variable/video_comment_layers/dense/kernel/gradient*
validate_shape(
�
5mio_variable/video_comment_layers/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!video_comment_layers/dense/bias*
shape:@
�
5mio_variable/video_comment_layers/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!video_comment_layers/dense/bias*
shape:@
F
Initializer_209/zerosConst*
valueB@*    *
dtype0
�

Assign_209Assign5mio_variable/video_comment_layers/dense/bias/gradientInitializer_209/zeros*
T0*H
_class>
<:loc:@mio_variable/video_comment_layers/dense/bias/gradient*
validate_shape(*
use_locking(
�
'model/video_comment_layers/dense/MatMulMatMul,model/comment_genre_layers/dense_1/LeakyRelu7mio_variable/video_comment_layers/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
(model/video_comment_layers/dense/BiasAddBiasAdd'model/video_comment_layers/dense/MatMul5mio_variable/video_comment_layers/dense/bias/variable*
data_formatNHWC*
T0
]
0model/video_comment_layers/dense/LeakyRelu/alphaConst*
dtype0*
valueB
 *��L>
�
.model/video_comment_layers/dense/LeakyRelu/mulMul0model/video_comment_layers/dense/LeakyRelu/alpha(model/video_comment_layers/dense/BiasAdd*
T0
�
*model/video_comment_layers/dense/LeakyReluMaximum.model/video_comment_layers/dense/LeakyRelu/mul(model/video_comment_layers/dense/BiasAdd*
T0
�
9mio_variable/video_comment_layers/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*2
	container%#video_comment_layers/dense_1/kernel
�
9mio_variable/video_comment_layers/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#video_comment_layers/dense_1/kernel*
shape
:@
Y
$Initializer_210/random_uniform/shapeConst*
dtype0*
valueB"@      
O
"Initializer_210/random_uniform/minConst*
dtype0*
valueB
 *����
O
"Initializer_210/random_uniform/maxConst*
dtype0*
valueB
 *���>
�
,Initializer_210/random_uniform/RandomUniformRandomUniform$Initializer_210/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_210/random_uniform/subSub"Initializer_210/random_uniform/max"Initializer_210/random_uniform/min*
T0
�
"Initializer_210/random_uniform/mulMul,Initializer_210/random_uniform/RandomUniform"Initializer_210/random_uniform/sub*
T0
v
Initializer_210/random_uniformAdd"Initializer_210/random_uniform/mul"Initializer_210/random_uniform/min*
T0
�

Assign_210Assign9mio_variable/video_comment_layers/dense_1/kernel/gradientInitializer_210/random_uniform*
T0*L
_classB
@>loc:@mio_variable/video_comment_layers/dense_1/kernel/gradient*
validate_shape(*
use_locking(
�
7mio_variable/video_comment_layers/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*0
	container#!video_comment_layers/dense_1/bias*
shape:
�
7mio_variable/video_comment_layers/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*0
	container#!video_comment_layers/dense_1/bias
F
Initializer_211/zerosConst*
valueB*    *
dtype0
�

Assign_211Assign7mio_variable/video_comment_layers/dense_1/bias/gradientInitializer_211/zeros*
T0*J
_class@
><loc:@mio_variable/video_comment_layers/dense_1/bias/gradient*
validate_shape(*
use_locking(
�
)model/video_comment_layers/dense_1/MatMulMatMul*model/video_comment_layers/dense/LeakyRelu9mio_variable/video_comment_layers/dense_1/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
*model/video_comment_layers/dense_1/BiasAddBiasAdd)model/video_comment_layers/dense_1/MatMul7mio_variable/video_comment_layers/dense_1/bias/variable*
data_formatNHWC*
T0
;
model/concat/axisConst*
dtype0*
value	B :
�	
model/concatConcatV2#model/follow_layers/dense_3/BiasAdd+model/forward_inside_layers/dense_3/BiasAdd%model/interact_layers/dense_3/BiasAdd*model/click_comment_layers/dense_3/BiasAdd)model/comment_time_layers/dense_3/BiasAdd2model/comment_unfold_score_logit/dense_1/LeakyRelu0model/comment_like_score_logit/dense_1/LeakyRelu<model/comment_content_copyward_score_logit/dense_1/LeakyRelu:model/comment_effective_read_score_logit/dense_1/LeakyRelu(model/sub_comment_layers/dense_1/BiasAdd*model/emoji_comment_layers/dense_1/BiasAdd(model/gif_comment_layers/dense_1/BiasAdd'model/at_comment_layers/dense_1/BiasAdd*model/image_comment_layers/dense_1/BiasAdd)model/text_comment_layers/dense_1/BiasAdd*model/video_comment_layers/dense_1/BiasAdd9model/uplift_comment_consume_depth_layers/dense_1/BiasAdd6model/comment_slide_down_score_logit/dense_1/LeakyRelu9model/uplift_comment_stay_duration_layers/dense_1/BiasAdd?model/effective_read_comment_fresh_label_layers/dense_1/BiasAdd#model/eft_click_cmt/dense_1/BiasAdd#model/eft_write_cmt/dense_1/BiasAdd.model/long_view_wiz_cmt_layers/dense_3/BiasAdd1model/long_view_wiz_no_cmt_layers/dense_3/BiasAddmodel/concat/axis*
T0*
N*

Tidx0
�
>mio_variable/context_aware_logits_layers/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(context_aware_logits_layers/dense/kernel*
shape
:@
�
>mio_variable/context_aware_logits_layers/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(context_aware_logits_layers/dense/kernel*
shape
:@
Y
$Initializer_212/random_uniform/shapeConst*
valueB"   @   *
dtype0
O
"Initializer_212/random_uniform/minConst*
valueB
 *���*
dtype0
O
"Initializer_212/random_uniform/maxConst*
dtype0*
valueB
 *��>
�
,Initializer_212/random_uniform/RandomUniformRandomUniform$Initializer_212/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_212/random_uniform/subSub"Initializer_212/random_uniform/max"Initializer_212/random_uniform/min*
T0
�
"Initializer_212/random_uniform/mulMul,Initializer_212/random_uniform/RandomUniform"Initializer_212/random_uniform/sub*
T0
v
Initializer_212/random_uniformAdd"Initializer_212/random_uniform/mul"Initializer_212/random_uniform/min*
T0
�

Assign_212Assign>mio_variable/context_aware_logits_layers/dense/kernel/gradientInitializer_212/random_uniform*
validate_shape(*
use_locking(*
T0*Q
_classG
ECloc:@mio_variable/context_aware_logits_layers/dense/kernel/gradient
�
<mio_variable/context_aware_logits_layers/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&context_aware_logits_layers/dense/bias*
shape:@
�
<mio_variable/context_aware_logits_layers/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&context_aware_logits_layers/dense/bias*
shape:@
F
Initializer_213/zerosConst*
dtype0*
valueB@*    
�

Assign_213Assign<mio_variable/context_aware_logits_layers/dense/bias/gradientInitializer_213/zeros*
use_locking(*
T0*O
_classE
CAloc:@mio_variable/context_aware_logits_layers/dense/bias/gradient*
validate_shape(
�
.model/context_aware_logits_layers/dense/MatMulMatMulmodel/concat>mio_variable/context_aware_logits_layers/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
/model/context_aware_logits_layers/dense/BiasAddBiasAdd.model/context_aware_logits_layers/dense/MatMul<mio_variable/context_aware_logits_layers/dense/bias/variable*
T0*
data_formatNHWC
n
,model/context_aware_logits_layers/dense/ReluRelu/model/context_aware_logits_layers/dense/BiasAdd*
T0
�
@mio_variable/context_aware_logits_layers/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*context_aware_logits_layers/dense_1/kernel*
shape
:@
�
@mio_variable/context_aware_logits_layers/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*9
	container,*context_aware_logits_layers/dense_1/kernel
Y
$Initializer_214/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_214/random_uniform/minConst*
dtype0*
valueB
 *���
O
"Initializer_214/random_uniform/maxConst*
valueB
 *��>*
dtype0
�
,Initializer_214/random_uniform/RandomUniformRandomUniform$Initializer_214/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_214/random_uniform/subSub"Initializer_214/random_uniform/max"Initializer_214/random_uniform/min*
T0
�
"Initializer_214/random_uniform/mulMul,Initializer_214/random_uniform/RandomUniform"Initializer_214/random_uniform/sub*
T0
v
Initializer_214/random_uniformAdd"Initializer_214/random_uniform/mul"Initializer_214/random_uniform/min*
T0
�

Assign_214Assign@mio_variable/context_aware_logits_layers/dense_1/kernel/gradientInitializer_214/random_uniform*
use_locking(*
T0*S
_classI
GEloc:@mio_variable/context_aware_logits_layers/dense_1/kernel/gradient*
validate_shape(
�
>mio_variable/context_aware_logits_layers/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(context_aware_logits_layers/dense_1/bias*
shape:
�
>mio_variable/context_aware_logits_layers/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(context_aware_logits_layers/dense_1/bias*
shape:
F
Initializer_215/zerosConst*
dtype0*
valueB*    
�

Assign_215Assign>mio_variable/context_aware_logits_layers/dense_1/bias/gradientInitializer_215/zeros*
T0*Q
_classG
ECloc:@mio_variable/context_aware_logits_layers/dense_1/bias/gradient*
validate_shape(*
use_locking(
�
0model/context_aware_logits_layers/dense_1/MatMulMatMul,model/context_aware_logits_layers/dense/Relu@mio_variable/context_aware_logits_layers/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
�
1model/context_aware_logits_layers/dense_1/BiasAddBiasAdd0model/context_aware_logits_layers/dense_1/MatMul>mio_variable/context_aware_logits_layers/dense_1/bias/variable*
T0*
data_formatNHWC
Z
	model/addAddmodel/concat1model/context_aware_logits_layers/dense_1/BiasAdd*
T0
,
model/SigmoidSigmoid	model/add*
T0
1
model/Sigmoid_1Sigmoidmodel/concat*
T0
5
model/ConstConst*
value	B :*
dtype0
?
model/split/split_dimConst*
dtype0*
value	B :
V
model/splitSplitmodel/split/split_dimmodel/Sigmoid_1*
T0*
	num_split
X
Const_1Const*9
value0B."     �Q�?��8@�w@�¥@��@
�A
�A*
dtype0
4
ShapeShapemodel/split*
T0*
out_type0
C
strided_slice_6/stackConst*
valueB: *
dtype0
E
strided_slice_6/stack_1Const*
valueB:*
dtype0
E
strided_slice_6/stack_2Const*
valueB:*
dtype0
�
strided_slice_6StridedSliceShapestrided_slice_6/stackstrided_slice_6/stack_1strided_slice_6/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
:
Tile/multiples/1Const*
value	B :*
dtype0
W
Tile/multiplesPackstrided_slice_6Tile/multiples/1*
T0*

axis *
N
@
TileTileConst_1Tile/multiples*

Tmultiples0*
T0
X
Const_2Const*9
value0B." �Q�?��8@�w@�¥@��@
�A
�A
�C*
dtype0
6
Shape_1Shapemodel/split*
T0*
out_type0
C
strided_slice_7/stackConst*
dtype0*
valueB: 
E
strided_slice_7/stack_1Const*
valueB:*
dtype0
E
strided_slice_7/stack_2Const*
valueB:*
dtype0
�
strided_slice_7StridedSliceShape_1strided_slice_7/stackstrided_slice_7/stack_1strided_slice_7/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
<
Tile_1/multiples/1Const*
dtype0*
value	B :
[
Tile_1/multiplesPackstrided_slice_7Tile_1/multiples/1*
T0*

axis *
N
D
Tile_1TileConst_2Tile_1/multiples*

Tmultiples0*
T0
�
Amio_variable/tpm_comment_consume_depth_pred/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	�@*:
	container-+tpm_comment_consume_depth_pred/dense/kernel
�
Amio_variable/tpm_comment_consume_depth_pred/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+tpm_comment_consume_depth_pred/dense/kernel*
shape:	�@
Y
$Initializer_216/random_uniform/shapeConst*
valueB"�   @   *
dtype0
O
"Initializer_216/random_uniform/minConst*
dtype0*
valueB
 *�5�
O
"Initializer_216/random_uniform/maxConst*
valueB
 *�5>*
dtype0
�
,Initializer_216/random_uniform/RandomUniformRandomUniform$Initializer_216/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_216/random_uniform/subSub"Initializer_216/random_uniform/max"Initializer_216/random_uniform/min*
T0
�
"Initializer_216/random_uniform/mulMul,Initializer_216/random_uniform/RandomUniform"Initializer_216/random_uniform/sub*
T0
v
Initializer_216/random_uniformAdd"Initializer_216/random_uniform/mul"Initializer_216/random_uniform/min*
T0
�

Assign_216AssignAmio_variable/tpm_comment_consume_depth_pred/dense/kernel/gradientInitializer_216/random_uniform*
use_locking(*
T0*T
_classJ
HFloc:@mio_variable/tpm_comment_consume_depth_pred/dense/kernel/gradient*
validate_shape(
�
?mio_variable/tpm_comment_consume_depth_pred/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*8
	container+)tpm_comment_consume_depth_pred/dense/bias*
shape:@
�
?mio_variable/tpm_comment_consume_depth_pred/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*8
	container+)tpm_comment_consume_depth_pred/dense/bias*
shape:@
F
Initializer_217/zerosConst*
valueB@*    *
dtype0
�

Assign_217Assign?mio_variable/tpm_comment_consume_depth_pred/dense/bias/gradientInitializer_217/zeros*
use_locking(*
T0*R
_classH
FDloc:@mio_variable/tpm_comment_consume_depth_pred/dense/bias/gradient*
validate_shape(
�
+tpm_comment_consume_depth_pred/dense/MatMulMatMul'model/comment_top_net/dense_1/LeakyReluAmio_variable/tpm_comment_consume_depth_pred/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
,tpm_comment_consume_depth_pred/dense/BiasAddBiasAdd+tpm_comment_consume_depth_pred/dense/MatMul?mio_variable/tpm_comment_consume_depth_pred/dense/bias/variable*
T0*
data_formatNHWC
b
#tpm_comment_consume_depth_pred/ReluRelu,tpm_comment_consume_depth_pred/dense/BiasAdd*
T0
e
8tpm_comment_consume_depth_pred/dropout/dropout/keep_probConst*
valueB
 *  �?*
dtype0
�
Cmio_variable/tpm_comment_consume_depth_pred/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*<
	container/-tpm_comment_consume_depth_pred/dense_1/kernel*
shape
:@ 
�
Cmio_variable/tpm_comment_consume_depth_pred/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*<
	container/-tpm_comment_consume_depth_pred/dense_1/kernel*
shape
:@ 
Y
$Initializer_218/random_uniform/shapeConst*
valueB"@       *
dtype0
O
"Initializer_218/random_uniform/minConst*
valueB
 *  ��*
dtype0
O
"Initializer_218/random_uniform/maxConst*
valueB
 *  �>*
dtype0
�
,Initializer_218/random_uniform/RandomUniformRandomUniform$Initializer_218/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_218/random_uniform/subSub"Initializer_218/random_uniform/max"Initializer_218/random_uniform/min*
T0
�
"Initializer_218/random_uniform/mulMul,Initializer_218/random_uniform/RandomUniform"Initializer_218/random_uniform/sub*
T0
v
Initializer_218/random_uniformAdd"Initializer_218/random_uniform/mul"Initializer_218/random_uniform/min*
T0
�

Assign_218AssignCmio_variable/tpm_comment_consume_depth_pred/dense_1/kernel/gradientInitializer_218/random_uniform*
T0*V
_classL
JHloc:@mio_variable/tpm_comment_consume_depth_pred/dense_1/kernel/gradient*
validate_shape(*
use_locking(
�
Amio_variable/tpm_comment_consume_depth_pred/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape: *:
	container-+tpm_comment_consume_depth_pred/dense_1/bias
�
Amio_variable/tpm_comment_consume_depth_pred/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+tpm_comment_consume_depth_pred/dense_1/bias*
shape: 
F
Initializer_219/zerosConst*
dtype0*
valueB *    
�

Assign_219AssignAmio_variable/tpm_comment_consume_depth_pred/dense_1/bias/gradientInitializer_219/zeros*
T0*T
_classJ
HFloc:@mio_variable/tpm_comment_consume_depth_pred/dense_1/bias/gradient*
validate_shape(*
use_locking(
�
-tpm_comment_consume_depth_pred/dense_1/MatMulMatMul#tpm_comment_consume_depth_pred/ReluCmio_variable/tpm_comment_consume_depth_pred/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0
�
.tpm_comment_consume_depth_pred/dense_1/BiasAddBiasAdd-tpm_comment_consume_depth_pred/dense_1/MatMulAmio_variable/tpm_comment_consume_depth_pred/dense_1/bias/variable*
T0*
data_formatNHWC
f
%tpm_comment_consume_depth_pred/Relu_1Relu.tpm_comment_consume_depth_pred/dense_1/BiasAdd*
T0
�
Cmio_variable/tpm_comment_consume_depth_pred/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*<
	container/-tpm_comment_consume_depth_pred/dense_2/kernel*
shape
: 
�
Cmio_variable/tpm_comment_consume_depth_pred/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*<
	container/-tpm_comment_consume_depth_pred/dense_2/kernel*
shape
: 
Y
$Initializer_220/random_uniform/shapeConst*
dtype0*
valueB"       
O
"Initializer_220/random_uniform/minConst*
valueB
 *��Ⱦ*
dtype0
O
"Initializer_220/random_uniform/maxConst*
valueB
 *���>*
dtype0
�
,Initializer_220/random_uniform/RandomUniformRandomUniform$Initializer_220/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_220/random_uniform/subSub"Initializer_220/random_uniform/max"Initializer_220/random_uniform/min*
T0
�
"Initializer_220/random_uniform/mulMul,Initializer_220/random_uniform/RandomUniform"Initializer_220/random_uniform/sub*
T0
v
Initializer_220/random_uniformAdd"Initializer_220/random_uniform/mul"Initializer_220/random_uniform/min*
T0
�

Assign_220AssignCmio_variable/tpm_comment_consume_depth_pred/dense_2/kernel/gradientInitializer_220/random_uniform*
validate_shape(*
use_locking(*
T0*V
_classL
JHloc:@mio_variable/tpm_comment_consume_depth_pred/dense_2/kernel/gradient
�
Amio_variable/tpm_comment_consume_depth_pred/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+tpm_comment_consume_depth_pred/dense_2/bias*
shape:
�
Amio_variable/tpm_comment_consume_depth_pred/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*:
	container-+tpm_comment_consume_depth_pred/dense_2/bias
F
Initializer_221/zerosConst*
valueB*    *
dtype0
�

Assign_221AssignAmio_variable/tpm_comment_consume_depth_pred/dense_2/bias/gradientInitializer_221/zeros*
validate_shape(*
use_locking(*
T0*T
_classJ
HFloc:@mio_variable/tpm_comment_consume_depth_pred/dense_2/bias/gradient
�
-tpm_comment_consume_depth_pred/dense_2/MatMulMatMul%tpm_comment_consume_depth_pred/Relu_1Cmio_variable/tpm_comment_consume_depth_pred/dense_2/kernel/variable*
transpose_b( *
T0*
transpose_a( 
�
.tpm_comment_consume_depth_pred/dense_2/BiasAddBiasAdd-tpm_comment_consume_depth_pred/dense_2/MatMulAmio_variable/tpm_comment_consume_depth_pred/dense_2/bias/variable*
T0*
data_formatNHWC
j
&tpm_comment_consume_depth_pred/SigmoidSigmoid.tpm_comment_consume_depth_pred/dense_2/BiasAdd*
T0
!
addAddTileTile_1*
T0
6
	truediv/yConst*
valueB
 *   @*
dtype0
+
truedivRealDivadd	truediv/y*
T0
J
strided_slice_8/stackConst*
valueB"       *
dtype0
L
strided_slice_8/stack_1Const*
valueB"       *
dtype0
L
strided_slice_8/stack_2Const*
valueB"      *
dtype0
�
strided_slice_8StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_8/stackstrided_slice_8/stack_1strided_slice_8/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
T0*
Index0
2
sub/xConst*
valueB
 *  �?*
dtype0
+
subSubsub/xstrided_slice_8*
T0
4
add_1/yConst*
valueB
 *��'7*
dtype0
#
add_1Addsubadd_1/y*
T0

LogLogadd_1*
T0
4
add_2/xConst*
valueB
 *    *
dtype0
#
add_2Addadd_2/xLog*
T0
J
strided_slice_9/stackConst*
valueB"       *
dtype0
L
strided_slice_9/stack_1Const*
dtype0*
valueB"       
L
strided_slice_9/stack_2Const*
dtype0*
valueB"      
�
strided_slice_9StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_9/stackstrided_slice_9/stack_1strided_slice_9/stack_2*
end_mask*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask 
4
sub_1/xConst*
valueB
 *  �?*
dtype0
/
sub_1Subsub_1/xstrided_slice_9*
T0
4
add_3/yConst*
dtype0*
valueB
 *��'7
%
add_3Addsub_1add_3/y*
T0

Log_1Logadd_3*
T0
#
add_4Addadd_2Log_1*
T0
K
strided_slice_10/stackConst*
valueB"        *
dtype0
M
strided_slice_10/stack_1Const*
valueB"       *
dtype0
M
strided_slice_10/stack_2Const*
valueB"      *
dtype0
�
strided_slice_10StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_10/stackstrided_slice_10/stack_1strided_slice_10/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
Index0*
T0
4
sub_2/xConst*
valueB
 *  �?*
dtype0
0
sub_2Subsub_2/xstrided_slice_10*
T0
4
add_5/yConst*
valueB
 *��'7*
dtype0
%
add_5Addsub_2add_5/y*
T0

Log_2Logadd_5*
T0
#
add_6Addadd_4Log_2*
T0
K
strided_slice_11/stackConst*
valueB"       *
dtype0
M
strided_slice_11/stack_1Const*
valueB"       *
dtype0
M
strided_slice_11/stack_2Const*
valueB"      *
dtype0
�
strided_slice_11StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_11/stackstrided_slice_11/stack_1strided_slice_11/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
4
add_7/yConst*
valueB
 *��'7*
dtype0
0
add_7Addstrided_slice_11add_7/y*
T0

Log_3Logadd_7*
T0
4
add_8/xConst*
valueB
 *    *
dtype0
%
add_8Addadd_8/xLog_3*
T0
K
strided_slice_12/stackConst*
valueB"       *
dtype0
M
strided_slice_12/stack_1Const*
valueB"       *
dtype0
M
strided_slice_12/stack_2Const*
valueB"      *
dtype0
�
strided_slice_12StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_12/stackstrided_slice_12/stack_1strided_slice_12/stack_2*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
Index0*
T0*
shrink_axis_mask
4
sub_3/xConst*
dtype0*
valueB
 *  �?
0
sub_3Subsub_3/xstrided_slice_12*
T0
4
add_9/yConst*
dtype0*
valueB
 *��'7
%
add_9Addsub_3add_9/y*
T0

Log_4Logadd_9*
T0
$
add_10Addadd_8Log_4*
T0
K
strided_slice_13/stackConst*
dtype0*
valueB"        
M
strided_slice_13/stack_1Const*
valueB"       *
dtype0
M
strided_slice_13/stack_2Const*
valueB"      *
dtype0
�
strided_slice_13StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_13/stackstrided_slice_13/stack_1strided_slice_13/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
4
sub_4/xConst*
valueB
 *  �?*
dtype0
0
sub_4Subsub_4/xstrided_slice_13*
T0
5
add_11/yConst*
dtype0*
valueB
 *��'7
'
add_11Addsub_4add_11/y*
T0

Log_5Logadd_11*
T0
%
add_12Addadd_10Log_5*
T0
K
strided_slice_14/stackConst*
valueB"       *
dtype0
M
strided_slice_14/stack_1Const*
valueB"       *
dtype0
M
strided_slice_14/stack_2Const*
dtype0*
valueB"      
�
strided_slice_14StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_14/stackstrided_slice_14/stack_1strided_slice_14/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0*
shrink_axis_mask
4
sub_5/xConst*
dtype0*
valueB
 *  �?
0
sub_5Subsub_5/xstrided_slice_14*
T0
5
add_13/yConst*
valueB
 *��'7*
dtype0
'
add_13Addsub_5add_13/y*
T0

Log_6Logadd_13*
T0
5
add_14/xConst*
valueB
 *    *
dtype0
'
add_14Addadd_14/xLog_6*
T0
K
strided_slice_15/stackConst*
dtype0*
valueB"       
M
strided_slice_15/stack_1Const*
valueB"       *
dtype0
M
strided_slice_15/stack_2Const*
valueB"      *
dtype0
�
strided_slice_15StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_15/stackstrided_slice_15/stack_1strided_slice_15/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
5
add_15/yConst*
valueB
 *��'7*
dtype0
2
add_15Addstrided_slice_15add_15/y*
T0

Log_7Logadd_15*
T0
%
add_16Addadd_14Log_7*
T0
K
strided_slice_16/stackConst*
valueB"        *
dtype0
M
strided_slice_16/stack_1Const*
valueB"       *
dtype0
M
strided_slice_16/stack_2Const*
dtype0*
valueB"      
�
strided_slice_16StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_16/stackstrided_slice_16/stack_1strided_slice_16/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
4
sub_6/xConst*
valueB
 *  �?*
dtype0
0
sub_6Subsub_6/xstrided_slice_16*
T0
5
add_17/yConst*
valueB
 *��'7*
dtype0
'
add_17Addsub_6add_17/y*
T0

Log_8Logadd_17*
T0
%
add_18Addadd_16Log_8*
T0
K
strided_slice_17/stackConst*
dtype0*
valueB"       
M
strided_slice_17/stack_1Const*
valueB"       *
dtype0
M
strided_slice_17/stack_2Const*
valueB"      *
dtype0
�
strided_slice_17StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_17/stackstrided_slice_17/stack_1strided_slice_17/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0*
shrink_axis_mask
5
add_19/yConst*
valueB
 *��'7*
dtype0
2
add_19Addstrided_slice_17add_19/y*
T0

Log_9Logadd_19*
T0
5
add_20/xConst*
valueB
 *    *
dtype0
'
add_20Addadd_20/xLog_9*
T0
K
strided_slice_18/stackConst*
valueB"       *
dtype0
M
strided_slice_18/stack_1Const*
dtype0*
valueB"       
M
strided_slice_18/stack_2Const*
valueB"      *
dtype0
�
strided_slice_18StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_18/stackstrided_slice_18/stack_1strided_slice_18/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
Index0*
T0
5
add_21/yConst*
dtype0*
valueB
 *��'7
2
add_21Addstrided_slice_18add_21/y*
T0

Log_10Logadd_21*
T0
&
add_22Addadd_20Log_10*
T0
K
strided_slice_19/stackConst*
dtype0*
valueB"        
M
strided_slice_19/stack_1Const*
valueB"       *
dtype0
M
strided_slice_19/stack_2Const*
valueB"      *
dtype0
�
strided_slice_19StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_19/stackstrided_slice_19/stack_1strided_slice_19/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
4
sub_7/xConst*
valueB
 *  �?*
dtype0
0
sub_7Subsub_7/xstrided_slice_19*
T0
5
add_23/yConst*
dtype0*
valueB
 *��'7
'
add_23Addsub_7add_23/y*
T0

Log_11Logadd_23*
T0
&
add_24Addadd_22Log_11*
T0
K
strided_slice_20/stackConst*
valueB"       *
dtype0
M
strided_slice_20/stack_1Const*
valueB"       *
dtype0
M
strided_slice_20/stack_2Const*
valueB"      *
dtype0
�
strided_slice_20StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_20/stackstrided_slice_20/stack_1strided_slice_20/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
4
sub_8/xConst*
valueB
 *  �?*
dtype0
0
sub_8Subsub_8/xstrided_slice_20*
T0
5
add_25/yConst*
valueB
 *��'7*
dtype0
'
add_25Addsub_8add_25/y*
T0

Log_12Logadd_25*
T0
5
add_26/xConst*
valueB
 *    *
dtype0
(
add_26Addadd_26/xLog_12*
T0
K
strided_slice_21/stackConst*
valueB"       *
dtype0
M
strided_slice_21/stack_1Const*
valueB"       *
dtype0
M
strided_slice_21/stack_2Const*
valueB"      *
dtype0
�
strided_slice_21StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_21/stackstrided_slice_21/stack_1strided_slice_21/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
4
sub_9/xConst*
valueB
 *  �?*
dtype0
0
sub_9Subsub_9/xstrided_slice_21*
T0
5
add_27/yConst*
valueB
 *��'7*
dtype0
'
add_27Addsub_9add_27/y*
T0

Log_13Logadd_27*
T0
&
add_28Addadd_26Log_13*
T0
K
strided_slice_22/stackConst*
valueB"        *
dtype0
M
strided_slice_22/stack_1Const*
valueB"       *
dtype0
M
strided_slice_22/stack_2Const*
valueB"      *
dtype0
�
strided_slice_22StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_22/stackstrided_slice_22/stack_1strided_slice_22/stack_2*
end_mask*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
5
add_29/yConst*
dtype0*
valueB
 *��'7
2
add_29Addstrided_slice_22add_29/y*
T0

Log_14Logadd_29*
T0
&
add_30Addadd_28Log_14*
T0
K
strided_slice_23/stackConst*
valueB"       *
dtype0
M
strided_slice_23/stack_1Const*
dtype0*
valueB"       
M
strided_slice_23/stack_2Const*
valueB"      *
dtype0
�
strided_slice_23StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_23/stackstrided_slice_23/stack_1strided_slice_23/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0
5
add_31/yConst*
dtype0*
valueB
 *��'7
2
add_31Addstrided_slice_23add_31/y*
T0

Log_15Logadd_31*
T0
5
add_32/xConst*
valueB
 *    *
dtype0
(
add_32Addadd_32/xLog_15*
T0
K
strided_slice_24/stackConst*
valueB"       *
dtype0
M
strided_slice_24/stack_1Const*
dtype0*
valueB"       
M
strided_slice_24/stack_2Const*
valueB"      *
dtype0
�
strided_slice_24StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_24/stackstrided_slice_24/stack_1strided_slice_24/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
5
sub_10/xConst*
valueB
 *  �?*
dtype0
2
sub_10Subsub_10/xstrided_slice_24*
T0
5
add_33/yConst*
valueB
 *��'7*
dtype0
(
add_33Addsub_10add_33/y*
T0

Log_16Logadd_33*
T0
&
add_34Addadd_32Log_16*
T0
K
strided_slice_25/stackConst*
dtype0*
valueB"        
M
strided_slice_25/stack_1Const*
valueB"       *
dtype0
M
strided_slice_25/stack_2Const*
valueB"      *
dtype0
�
strided_slice_25StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_25/stackstrided_slice_25/stack_1strided_slice_25/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
5
add_35/yConst*
valueB
 *��'7*
dtype0
2
add_35Addstrided_slice_25add_35/y*
T0

Log_17Logadd_35*
T0
&
add_36Addadd_34Log_17*
T0
K
strided_slice_26/stackConst*
dtype0*
valueB"       
M
strided_slice_26/stack_1Const*
valueB"       *
dtype0
M
strided_slice_26/stack_2Const*
valueB"      *
dtype0
�
strided_slice_26StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_26/stackstrided_slice_26/stack_1strided_slice_26/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
5
sub_11/xConst*
valueB
 *  �?*
dtype0
2
sub_11Subsub_11/xstrided_slice_26*
T0
5
add_37/yConst*
valueB
 *��'7*
dtype0
(
add_37Addsub_11add_37/y*
T0

Log_18Logadd_37*
T0
5
add_38/xConst*
valueB
 *    *
dtype0
(
add_38Addadd_38/xLog_18*
T0
K
strided_slice_27/stackConst*
valueB"       *
dtype0
M
strided_slice_27/stack_1Const*
valueB"       *
dtype0
M
strided_slice_27/stack_2Const*
valueB"      *
dtype0
�
strided_slice_27StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_27/stackstrided_slice_27/stack_1strided_slice_27/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
5
add_39/yConst*
valueB
 *��'7*
dtype0
2
add_39Addstrided_slice_27add_39/y*
T0

Log_19Logadd_39*
T0
&
add_40Addadd_38Log_19*
T0
K
strided_slice_28/stackConst*
valueB"        *
dtype0
M
strided_slice_28/stack_1Const*
valueB"       *
dtype0
M
strided_slice_28/stack_2Const*
valueB"      *
dtype0
�
strided_slice_28StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_28/stackstrided_slice_28/stack_1strided_slice_28/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
5
add_41/yConst*
valueB
 *��'7*
dtype0
2
add_41Addstrided_slice_28add_41/y*
T0

Log_20Logadd_41*
T0
&
add_42Addadd_40Log_20*
T0
K
strided_slice_29/stackConst*
valueB"       *
dtype0
M
strided_slice_29/stack_1Const*
dtype0*
valueB"       
M
strided_slice_29/stack_2Const*
valueB"      *
dtype0
�
strided_slice_29StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_29/stackstrided_slice_29/stack_1strided_slice_29/stack_2*
end_mask*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
5
add_43/yConst*
valueB
 *��'7*
dtype0
2
add_43Addstrided_slice_29add_43/y*
T0

Log_21Logadd_43*
T0
5
add_44/xConst*
dtype0*
valueB
 *    
(
add_44Addadd_44/xLog_21*
T0
K
strided_slice_30/stackConst*
valueB"       *
dtype0
M
strided_slice_30/stack_1Const*
valueB"       *
dtype0
M
strided_slice_30/stack_2Const*
dtype0*
valueB"      
�
strided_slice_30StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_30/stackstrided_slice_30/stack_1strided_slice_30/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
5
add_45/yConst*
valueB
 *��'7*
dtype0
2
add_45Addstrided_slice_30add_45/y*
T0

Log_22Logadd_45*
T0
&
add_46Addadd_44Log_22*
T0
K
strided_slice_31/stackConst*
valueB"        *
dtype0
M
strided_slice_31/stack_1Const*
valueB"       *
dtype0
M
strided_slice_31/stack_2Const*
valueB"      *
dtype0
�
strided_slice_31StridedSlice&tpm_comment_consume_depth_pred/Sigmoidstrided_slice_31/stackstrided_slice_31/stack_1strided_slice_31/stack_2*
end_mask*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
5
add_47/yConst*
valueB
 *��'7*
dtype0
2
add_47Addstrided_slice_31add_47/y*
T0

Log_23Logadd_47*
T0
&
add_48Addadd_46Log_23*
T0
j
stackPackadd_6add_12add_18add_24add_30add_36add_42add_48*
N*
T0*

axis

ExpExpstack*
T0
!
mulMultruedivExp*
T0
J
Sum_1/reduction_indicesConst*
valueB :
���������*
dtype0
P
Sum_1SummulSum_1/reduction_indices*

Tidx0*
	keep_dims(*
T0
 
SquareSquareSum_1*
T0
"
mul_1MulSquareExp*
T0
J
Sum_2/reduction_indicesConst*
valueB :
���������*
dtype0
R
Sum_2Summul_1Sum_2/reduction_indices*

Tidx0*
	keep_dims(*
T0
"
Square_1SquareSum_1*
T0
'
sub_12SubSum_2Square_1*
T0

SqrtSqrtsub_12*
T0
<
Const_3Const*
valueB"       *
dtype0
A
Sum_3SumSqrtConst_3*
T0*

Tidx0*
	keep_dims( 
3
mul_2Mulmodel/split:5model/split:3*
T0
3
mul_3Mulmodel/split:6model/split:3*
T0
3
mul_4Mulmodel/split:7model/split:3*
T0
3
mul_5Mulmodel/split:8model/split:3*
T0
+
mul_6MulSum_1model/split:3*
T0
4
mul_7Mulmodel/split:17model/split:3*
T0
4
mul_8Mulmodel/split:16model/split:3*
T0
4
mul_9Mulmodel/split:18model/split:3*
T0"