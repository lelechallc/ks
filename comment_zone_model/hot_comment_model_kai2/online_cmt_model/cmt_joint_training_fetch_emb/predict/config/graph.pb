
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
dtype0*
shape: 
K
MIO_TABLE_ADDRESSConst"/device:CPU:0*
value
B  *
dtype0
Ĩ
2mio_compress_indices/COMPRESS_INDEX__USER/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:˙˙˙˙˙˙˙˙˙*#
	containerCOMPRESS_INDEX__USER
Ĩ
2mio_compress_indices/COMPRESS_INDEX__USER/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:˙˙˙˙˙˙˙˙˙*#
	containerCOMPRESS_INDEX__USER
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

%mio_embeddings/pid_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:˙˙˙˙˙˙˙˙˙@*
	containerpid_embedding

%mio_embeddings/pid_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:˙˙˙˙˙˙˙˙˙@*
	containerpid_embedding

%mio_embeddings/aid_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containeraid_embedding*
shape:˙˙˙˙˙˙˙˙˙@

%mio_embeddings/aid_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containeraid_embedding*
shape:˙˙˙˙˙˙˙˙˙@

%mio_embeddings/uid_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containeruid_embedding*
shape:˙˙˙˙˙˙˙˙˙@

%mio_embeddings/uid_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containeruid_embedding*
shape:˙˙˙˙˙˙˙˙˙@

%mio_embeddings/did_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerdid_embedding*
shape:˙˙˙˙˙˙˙˙˙@

%mio_embeddings/did_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containerdid_embedding*
shape:˙˙˙˙˙˙˙˙˙@

)mio_embeddings/context_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS* 
	containercontext_embedding*
shape:˙˙˙˙˙˙˙˙˙@

)mio_embeddings/context_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS* 
	containercontext_embedding*
shape:˙˙˙˙˙˙˙˙˙@
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
*mio_embeddings/position_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:˙˙˙˙˙˙˙˙˙*!
	containerposition_embedding
Š
/mio_embeddings/comment_genre_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containercomment_genre_embedding*
shape:˙˙˙˙˙˙˙˙˙
Š
/mio_embeddings/comment_genre_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containercomment_genre_embedding*
shape:˙˙˙˙˙˙˙˙˙
Ģ
0mio_embeddings/comment_length_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containercomment_length_embedding*
shape:˙˙˙˙˙˙˙˙˙ 
Ģ
0mio_embeddings/comment_length_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:˙˙˙˙˙˙˙˙˙ *'
	containercomment_length_embedding

(mio_extra_param/token_input_ids/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containertoken_input_ids*
shape:˙˙˙˙˙˙˙˙˙

(mio_extra_param/token_input_ids/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containertoken_input_ids*
shape:˙˙˙˙˙˙˙˙˙

)mio_extra_param/token_input_mask/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containertoken_input_mask*
shape:˙˙˙˙˙˙˙˙˙

)mio_extra_param/token_input_mask/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containertoken_input_mask*
shape:˙˙˙˙˙˙˙˙˙

&mio_extra_param/token_sep_ids/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containertoken_sep_ids*
shape:˙˙˙˙˙˙˙˙˙

&mio_extra_param/token_sep_ids/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
	containertoken_sep_ids*
shape:˙˙˙˙˙˙˙˙˙
Š
/mio_extra_param/bert_first5_layers_emb/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerbert_first5_layers_emb*
shape:˙˙˙˙˙˙˙˙˙6
Š
/mio_extra_param/bert_first5_layers_emb/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerbert_first5_layers_emb*
shape:˙˙˙˙˙˙˙˙˙6
`
Cast_1Cast(mio_extra_param/token_input_ids/variable*

SrcT0*
Truncate( *

DstT0
a
Cast_2Cast)mio_extra_param/token_input_mask/variable*

SrcT0*
Truncate( *

DstT0
^
Cast_3Cast&mio_extra_param/token_sep_ids/variable*

SrcT0*
Truncate( *

DstT0
F
Reshape/shapeConst*!
valueB"˙˙˙˙      *
dtype0
i
ReshapeReshape/mio_extra_param/bert_first5_layers_emb/variableReshape/shape*
T0*
Tshape0
/
ShapeShapeCast_1*
T0*
out_type0
A
strided_slice/stackConst*
valueB: *
dtype0
C
strided_slice/stack_1Const*
valueB:*
dtype0
C
strided_slice/stack_2Const*
valueB:*
dtype0
á
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
end_mask *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
<
bert/encoder/ShapeShapeCast_1*
T0*
out_type0
N
 bert/encoder/strided_slice/stackConst*
valueB: *
dtype0
P
"bert/encoder/strided_slice/stack_1Const*
dtype0*
valueB:
P
"bert/encoder/strided_slice/stack_2Const*
valueB:*
dtype0
ĸ
bert/encoder/strided_sliceStridedSlicebert/encoder/Shape bert/encoder/strided_slice/stack"bert/encoder/strided_slice/stack_1"bert/encoder/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
>
bert/encoder/Shape_1ShapeCast_2*
T0*
out_type0
P
"bert/encoder/strided_slice_1/stackConst*
valueB: *
dtype0
R
$bert/encoder/strided_slice_1/stack_1Const*
dtype0*
valueB:
R
$bert/encoder/strided_slice_1/stack_2Const*
valueB:*
dtype0
Ŧ
bert/encoder/strided_slice_1StridedSlicebert/encoder/Shape_1"bert/encoder/strided_slice_1/stack$bert/encoder/strided_slice_1/stack_1$bert/encoder/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
F
bert/encoder/Reshape/shape/1Const*
value	B :*
dtype0
F
bert/encoder/Reshape/shape/2Const*
value	B :*
dtype0

bert/encoder/Reshape/shapePackbert/encoder/strided_slicebert/encoder/Reshape/shape/1bert/encoder/Reshape/shape/2*
T0*

axis *
N
Z
bert/encoder/ReshapeReshapeCast_2bert/encoder/Reshape/shape*
T0*
Tshape0
W
bert/encoder/CastCastbert/encoder/Reshape*
Truncate( *

DstT0*

SrcT0
A
bert/encoder/ones/mul/yConst*
value	B :*
dtype0
Z
bert/encoder/ones/mulMulbert/encoder/strided_slicebert/encoder/ones/mul/y*
T0
C
bert/encoder/ones/mul_1/yConst*
value	B :*
dtype0
Y
bert/encoder/ones/mul_1Mulbert/encoder/ones/mulbert/encoder/ones/mul_1/y*
T0
C
bert/encoder/ones/Less/yConst*
dtype0*
value
B :č
Z
bert/encoder/ones/LessLessbert/encoder/ones/mul_1bert/encoder/ones/Less/y*
T0
D
bert/encoder/ones/packed/1Const*
value	B :*
dtype0
D
bert/encoder/ones/packed/2Const*
value	B :*
dtype0

bert/encoder/ones/packedPackbert/encoder/strided_slicebert/encoder/ones/packed/1bert/encoder/ones/packed/2*
T0*

axis *
N
D
bert/encoder/ones/ConstConst*
valueB
 *  ?*
dtype0
g
bert/encoder/onesFillbert/encoder/ones/packedbert/encoder/ones/Const*
T0*

index_type0
F
bert/encoder/mulMulbert/encoder/onesbert/encoder/Cast*
T0
?
bert/encoder/Shape_2ShapeReshape*
T0*
out_type0
P
"bert/encoder/strided_slice_2/stackConst*
valueB: *
dtype0
R
$bert/encoder/strided_slice_2/stack_1Const*
valueB:*
dtype0
R
$bert/encoder/strided_slice_2/stack_2Const*
valueB:*
dtype0
Ŧ
bert/encoder/strided_slice_2StridedSlicebert/encoder/Shape_2"bert/encoder/strided_slice_2/stack$bert/encoder/strided_slice_2/stack_1$bert/encoder/strided_slice_2/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
Q
bert/encoder/Reshape_1/shapeConst*
valueB"˙˙˙˙   *
dtype0
_
bert/encoder/Reshape_1ReshapeReshapebert/encoder/Reshape_1/shape*
T0*
Tshape0
c
)bert/encoder/layer_5/attention/self/ShapeShapebert/encoder/Reshape_1*
T0*
out_type0
e
7bert/encoder/layer_5/attention/self/strided_slice/stackConst*
valueB: *
dtype0
g
9bert/encoder/layer_5/attention/self/strided_slice/stack_1Const*
valueB:*
dtype0
g
9bert/encoder/layer_5/attention/self/strided_slice/stack_2Const*
valueB:*
dtype0

1bert/encoder/layer_5/attention/self/strided_sliceStridedSlice)bert/encoder/layer_5/attention/self/Shape7bert/encoder/layer_5/attention/self/strided_slice/stack9bert/encoder/layer_5/attention/self/strided_slice/stack_19bert/encoder/layer_5/attention/self/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
e
+bert/encoder/layer_5/attention/self/Shape_1Shapebert/encoder/Reshape_1*
T0*
out_type0
g
9bert/encoder/layer_5/attention/self/strided_slice_1/stackConst*
valueB: *
dtype0
i
;bert/encoder/layer_5/attention/self/strided_slice_1/stack_1Const*
valueB:*
dtype0
i
;bert/encoder/layer_5/attention/self/strided_slice_1/stack_2Const*
valueB:*
dtype0

3bert/encoder/layer_5/attention/self/strided_slice_1StridedSlice+bert/encoder/layer_5/attention/self/Shape_19bert/encoder/layer_5/attention/self/strided_slice_1/stack;bert/encoder/layer_5/attention/self/strided_slice_1/stack_1;bert/encoder/layer_5/attention/self/strided_slice_1/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0
Ō
Fmio_variable/bert/encoder/layer_5/attention/self/query/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*?
	container20bert/encoder/layer_5/attention/self/query/kernel
Ō
Fmio_variable/bert/encoder/layer_5/attention/self/query/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_5/attention/self/query/kernel*
shape:

W
"Initializer/truncated_normal/shapeConst*
valueB"      *
dtype0
N
!Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0
P
#Initializer/truncated_normal/stddevConst*
dtype0*
valueB
 *
×Ŗ<

,Initializer/truncated_normal/TruncatedNormalTruncatedNormal"Initializer/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0

 Initializer/truncated_normal/mulMul,Initializer/truncated_normal/TruncatedNormal#Initializer/truncated_normal/stddev*
T0
q
Initializer/truncated_normalAdd Initializer/truncated_normal/mul!Initializer/truncated_normal/mean*
T0

AssignAssignFmio_variable/bert/encoder/layer_5/attention/self/query/kernel/gradientInitializer/truncated_normal*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_5/attention/self/query/kernel/gradient*
validate_shape(
É
Dmio_variable/bert/encoder/layer_5/attention/self/query/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_5/attention/self/query/bias*
shape:
É
Dmio_variable/bert/encoder/layer_5/attention/self/query/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*=
	container0.bert/encoder/layer_5/attention/self/query/bias
E
Initializer_1/zerosConst*
valueB*    *
dtype0
ø
Assign_1AssignDmio_variable/bert/encoder/layer_5/attention/self/query/bias/gradientInitializer_1/zeros*W
_classM
KIloc:@mio_variable/bert/encoder/layer_5/attention/self/query/bias/gradient*
validate_shape(*
use_locking(*
T0
É
0bert/encoder/layer_5/attention/self/query/MatMulMatMulbert/encoder/Reshape_1Fmio_variable/bert/encoder/layer_5/attention/self/query/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Ô
1bert/encoder/layer_5/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_5/attention/self/query/MatMulDmio_variable/bert/encoder/layer_5/attention/self/query/bias/variable*
T0*
data_formatNHWC
Î
Dmio_variable/bert/encoder/layer_5/attention/self/key/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_5/attention/self/key/kernel*
shape:

Î
Dmio_variable/bert/encoder/layer_5/attention/self/key/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_5/attention/self/key/kernel*
shape:

Y
$Initializer_2/truncated_normal/shapeConst*
valueB"      *
dtype0
P
#Initializer_2/truncated_normal/meanConst*
valueB
 *    *
dtype0
R
%Initializer_2/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

.Initializer_2/truncated_normal/TruncatedNormalTruncatedNormal$Initializer_2/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 

"Initializer_2/truncated_normal/mulMul.Initializer_2/truncated_normal/TruncatedNormal%Initializer_2/truncated_normal/stddev*
T0
w
Initializer_2/truncated_normalAdd"Initializer_2/truncated_normal/mul#Initializer_2/truncated_normal/mean*
T0

Assign_2AssignDmio_variable/bert/encoder/layer_5/attention/self/key/kernel/gradientInitializer_2/truncated_normal*W
_classM
KIloc:@mio_variable/bert/encoder/layer_5/attention/self/key/kernel/gradient*
validate_shape(*
use_locking(*
T0
Å
Bmio_variable/bert/encoder/layer_5/attention/self/key/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_5/attention/self/key/bias*
shape:
Å
Bmio_variable/bert/encoder/layer_5/attention/self/key/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_5/attention/self/key/bias*
shape:
E
Initializer_3/zerosConst*
valueB*    *
dtype0
ô
Assign_3AssignBmio_variable/bert/encoder/layer_5/attention/self/key/bias/gradientInitializer_3/zeros*
use_locking(*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_5/attention/self/key/bias/gradient*
validate_shape(
Å
.bert/encoder/layer_5/attention/self/key/MatMulMatMulbert/encoder/Reshape_1Dmio_variable/bert/encoder/layer_5/attention/self/key/kernel/variable*
transpose_b( *
T0*
transpose_a( 
Î
/bert/encoder/layer_5/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_5/attention/self/key/MatMulBmio_variable/bert/encoder/layer_5/attention/self/key/bias/variable*
T0*
data_formatNHWC
Ō
Fmio_variable/bert/encoder/layer_5/attention/self/value/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_5/attention/self/value/kernel*
shape:

Ō
Fmio_variable/bert/encoder/layer_5/attention/self/value/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*?
	container20bert/encoder/layer_5/attention/self/value/kernel
Y
$Initializer_4/truncated_normal/shapeConst*
dtype0*
valueB"      
P
#Initializer_4/truncated_normal/meanConst*
valueB
 *    *
dtype0
R
%Initializer_4/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

.Initializer_4/truncated_normal/TruncatedNormalTruncatedNormal$Initializer_4/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 

"Initializer_4/truncated_normal/mulMul.Initializer_4/truncated_normal/TruncatedNormal%Initializer_4/truncated_normal/stddev*
T0
w
Initializer_4/truncated_normalAdd"Initializer_4/truncated_normal/mul#Initializer_4/truncated_normal/mean*
T0

Assign_4AssignFmio_variable/bert/encoder/layer_5/attention/self/value/kernel/gradientInitializer_4/truncated_normal*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_5/attention/self/value/kernel/gradient*
validate_shape(
É
Dmio_variable/bert/encoder/layer_5/attention/self/value/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_5/attention/self/value/bias*
shape:
É
Dmio_variable/bert/encoder/layer_5/attention/self/value/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_5/attention/self/value/bias*
shape:
E
Initializer_5/zerosConst*
valueB*    *
dtype0
ø
Assign_5AssignDmio_variable/bert/encoder/layer_5/attention/self/value/bias/gradientInitializer_5/zeros*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_5/attention/self/value/bias/gradient*
validate_shape(
É
0bert/encoder/layer_5/attention/self/value/MatMulMatMulbert/encoder/Reshape_1Fmio_variable/bert/encoder/layer_5/attention/self/value/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Ô
1bert/encoder/layer_5/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_5/attention/self/value/MatMulDmio_variable/bert/encoder/layer_5/attention/self/value/bias/variable*
data_formatNHWC*
T0
]
3bert/encoder/layer_5/attention/self/Reshape/shape/1Const*
value	B :*
dtype0
]
3bert/encoder/layer_5/attention/self/Reshape/shape/2Const*
dtype0*
value	B :
]
3bert/encoder/layer_5/attention/self/Reshape/shape/3Const*
value	B : *
dtype0

1bert/encoder/layer_5/attention/self/Reshape/shapePackbert/encoder/strided_slice_23bert/encoder/layer_5/attention/self/Reshape/shape/13bert/encoder/layer_5/attention/self/Reshape/shape/23bert/encoder/layer_5/attention/self/Reshape/shape/3*
T0*

axis *
N
ŗ
+bert/encoder/layer_5/attention/self/ReshapeReshape1bert/encoder/layer_5/attention/self/query/BiasAdd1bert/encoder/layer_5/attention/self/Reshape/shape*
T0*
Tshape0
o
2bert/encoder/layer_5/attention/self/transpose/permConst*
dtype0*%
valueB"             
ą
-bert/encoder/layer_5/attention/self/transpose	Transpose+bert/encoder/layer_5/attention/self/Reshape2bert/encoder/layer_5/attention/self/transpose/perm*
Tperm0*
T0
_
5bert/encoder/layer_5/attention/self/Reshape_1/shape/1Const*
value	B :*
dtype0
_
5bert/encoder/layer_5/attention/self/Reshape_1/shape/2Const*
value	B :*
dtype0
_
5bert/encoder/layer_5/attention/self/Reshape_1/shape/3Const*
value	B : *
dtype0

3bert/encoder/layer_5/attention/self/Reshape_1/shapePackbert/encoder/strided_slice_25bert/encoder/layer_5/attention/self/Reshape_1/shape/15bert/encoder/layer_5/attention/self/Reshape_1/shape/25bert/encoder/layer_5/attention/self/Reshape_1/shape/3*
T0*

axis *
N
ĩ
-bert/encoder/layer_5/attention/self/Reshape_1Reshape/bert/encoder/layer_5/attention/self/key/BiasAdd3bert/encoder/layer_5/attention/self/Reshape_1/shape*
T0*
Tshape0
q
4bert/encoder/layer_5/attention/self/transpose_1/permConst*%
valueB"             *
dtype0
ˇ
/bert/encoder/layer_5/attention/self/transpose_1	Transpose-bert/encoder/layer_5/attention/self/Reshape_14bert/encoder/layer_5/attention/self/transpose_1/perm*
T0*
Tperm0
ŧ
*bert/encoder/layer_5/attention/self/MatMulBatchMatMul-bert/encoder/layer_5/attention/self/transpose/bert/encoder/layer_5/attention/self/transpose_1*
adj_x( *
adj_y(*
T0
V
)bert/encoder/layer_5/attention/self/Mul/yConst*
valueB
 *ķ5>*
dtype0

'bert/encoder/layer_5/attention/self/MulMul*bert/encoder/layer_5/attention/self/MatMul)bert/encoder/layer_5/attention/self/Mul/y*
T0
`
2bert/encoder/layer_5/attention/self/ExpandDims/dimConst*
valueB:*
dtype0

.bert/encoder/layer_5/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_5/attention/self/ExpandDims/dim*

Tdim0*
T0
V
)bert/encoder/layer_5/attention/self/sub/xConst*
valueB
 *  ?*
dtype0

'bert/encoder/layer_5/attention/self/subSub)bert/encoder/layer_5/attention/self/sub/x.bert/encoder/layer_5/attention/self/ExpandDims*
T0
X
+bert/encoder/layer_5/attention/self/mul_1/yConst*
valueB
 * @Æ*
dtype0

)bert/encoder/layer_5/attention/self/mul_1Mul'bert/encoder/layer_5/attention/self/sub+bert/encoder/layer_5/attention/self/mul_1/y*
T0

'bert/encoder/layer_5/attention/self/addAdd'bert/encoder/layer_5/attention/self/Mul)bert/encoder/layer_5/attention/self/mul_1*
T0
h
+bert/encoder/layer_5/attention/self/SoftmaxSoftmax'bert/encoder/layer_5/attention/self/add*
T0
b
5bert/encoder/layer_5/attention/self/dropout/keep_probConst*
valueB
 *fff?*
dtype0

1bert/encoder/layer_5/attention/self/dropout/ShapeShape+bert/encoder/layer_5/attention/self/Softmax*
T0*
out_type0
k
>bert/encoder/layer_5/attention/self/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
k
>bert/encoder/layer_5/attention/self/dropout/random_uniform/maxConst*
dtype0*
valueB
 *  ?
ģ
Hbert/encoder/layer_5/attention/self/dropout/random_uniform/RandomUniformRandomUniform1bert/encoder/layer_5/attention/self/dropout/Shape*

seed *
T0*
dtype0*
seed2 
Î
>bert/encoder/layer_5/attention/self/dropout/random_uniform/subSub>bert/encoder/layer_5/attention/self/dropout/random_uniform/max>bert/encoder/layer_5/attention/self/dropout/random_uniform/min*
T0
Ø
>bert/encoder/layer_5/attention/self/dropout/random_uniform/mulMulHbert/encoder/layer_5/attention/self/dropout/random_uniform/RandomUniform>bert/encoder/layer_5/attention/self/dropout/random_uniform/sub*
T0
Ę
:bert/encoder/layer_5/attention/self/dropout/random_uniformAdd>bert/encoder/layer_5/attention/self/dropout/random_uniform/mul>bert/encoder/layer_5/attention/self/dropout/random_uniform/min*
T0
˛
/bert/encoder/layer_5/attention/self/dropout/addAdd5bert/encoder/layer_5/attention/self/dropout/keep_prob:bert/encoder/layer_5/attention/self/dropout/random_uniform*
T0
t
1bert/encoder/layer_5/attention/self/dropout/FloorFloor/bert/encoder/layer_5/attention/self/dropout/add*
T0
§
/bert/encoder/layer_5/attention/self/dropout/divRealDiv+bert/encoder/layer_5/attention/self/Softmax5bert/encoder/layer_5/attention/self/dropout/keep_prob*
T0
Ŗ
/bert/encoder/layer_5/attention/self/dropout/mulMul/bert/encoder/layer_5/attention/self/dropout/div1bert/encoder/layer_5/attention/self/dropout/Floor*
T0
_
5bert/encoder/layer_5/attention/self/Reshape_2/shape/1Const*
value	B :*
dtype0
_
5bert/encoder/layer_5/attention/self/Reshape_2/shape/2Const*
value	B :*
dtype0
_
5bert/encoder/layer_5/attention/self/Reshape_2/shape/3Const*
value	B : *
dtype0

3bert/encoder/layer_5/attention/self/Reshape_2/shapePackbert/encoder/strided_slice_25bert/encoder/layer_5/attention/self/Reshape_2/shape/15bert/encoder/layer_5/attention/self/Reshape_2/shape/25bert/encoder/layer_5/attention/self/Reshape_2/shape/3*
T0*

axis *
N
ˇ
-bert/encoder/layer_5/attention/self/Reshape_2Reshape1bert/encoder/layer_5/attention/self/value/BiasAdd3bert/encoder/layer_5/attention/self/Reshape_2/shape*
T0*
Tshape0
q
4bert/encoder/layer_5/attention/self/transpose_2/permConst*%
valueB"             *
dtype0
ˇ
/bert/encoder/layer_5/attention/self/transpose_2	Transpose-bert/encoder/layer_5/attention/self/Reshape_24bert/encoder/layer_5/attention/self/transpose_2/perm*
T0*
Tperm0
Ā
,bert/encoder/layer_5/attention/self/MatMul_1BatchMatMul/bert/encoder/layer_5/attention/self/dropout/mul/bert/encoder/layer_5/attention/self/transpose_2*
adj_x( *
adj_y( *
T0
q
4bert/encoder/layer_5/attention/self/transpose_3/permConst*%
valueB"             *
dtype0
ļ
/bert/encoder/layer_5/attention/self/transpose_3	Transpose,bert/encoder/layer_5/attention/self/MatMul_14bert/encoder/layer_5/attention/self/transpose_3/perm*
Tperm0*
T0
U
+bert/encoder/layer_5/attention/self/mul_2/yConst*
value	B :*
dtype0

)bert/encoder/layer_5/attention/self/mul_2Mulbert/encoder/strided_slice_2+bert/encoder/layer_5/attention/self/mul_2/y*
T0
`
5bert/encoder/layer_5/attention/self/Reshape_3/shape/1Const*
value
B :*
dtype0
ģ
3bert/encoder/layer_5/attention/self/Reshape_3/shapePack)bert/encoder/layer_5/attention/self/mul_25bert/encoder/layer_5/attention/self/Reshape_3/shape/1*
T0*

axis *
N
ĩ
-bert/encoder/layer_5/attention/self/Reshape_3Reshape/bert/encoder/layer_5/attention/self/transpose_33bert/encoder/layer_5/attention/self/Reshape_3/shape*
T0*
Tshape0
Ö
Hmio_variable/bert/encoder/layer_5/attention/output/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42bert/encoder/layer_5/attention/output/dense/kernel*
shape:

Ö
Hmio_variable/bert/encoder/layer_5/attention/output/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42bert/encoder/layer_5/attention/output/dense/kernel*
shape:

Y
$Initializer_6/truncated_normal/shapeConst*
valueB"      *
dtype0
P
#Initializer_6/truncated_normal/meanConst*
valueB
 *    *
dtype0
R
%Initializer_6/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

.Initializer_6/truncated_normal/TruncatedNormalTruncatedNormal$Initializer_6/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0

"Initializer_6/truncated_normal/mulMul.Initializer_6/truncated_normal/TruncatedNormal%Initializer_6/truncated_normal/stddev*
T0
w
Initializer_6/truncated_normalAdd"Initializer_6/truncated_normal/mul#Initializer_6/truncated_normal/mean*
T0

Assign_6AssignHmio_variable/bert/encoder/layer_5/attention/output/dense/kernel/gradientInitializer_6/truncated_normal*
use_locking(*
T0*[
_classQ
OMloc:@mio_variable/bert/encoder/layer_5/attention/output/dense/kernel/gradient*
validate_shape(
Í
Fmio_variable/bert/encoder/layer_5/attention/output/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_5/attention/output/dense/bias*
shape:
Í
Fmio_variable/bert/encoder/layer_5/attention/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_5/attention/output/dense/bias*
shape:
E
Initializer_7/zerosConst*
valueB*    *
dtype0
ü
Assign_7AssignFmio_variable/bert/encoder/layer_5/attention/output/dense/bias/gradientInitializer_7/zeros*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_5/attention/output/dense/bias/gradient*
validate_shape(
ä
2bert/encoder/layer_5/attention/output/dense/MatMulMatMul-bert/encoder/layer_5/attention/self/Reshape_3Hmio_variable/bert/encoder/layer_5/attention/output/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Ú
3bert/encoder/layer_5/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_5/attention/output/dense/MatMulFmio_variable/bert/encoder/layer_5/attention/output/dense/bias/variable*
data_formatNHWC*
T0
d
7bert/encoder/layer_5/attention/output/dropout/keep_probConst*
valueB
 *fff?*
dtype0

3bert/encoder/layer_5/attention/output/dropout/ShapeShape3bert/encoder/layer_5/attention/output/dense/BiasAdd*
T0*
out_type0
m
@bert/encoder/layer_5/attention/output/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
m
@bert/encoder/layer_5/attention/output/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0
ŋ
Jbert/encoder/layer_5/attention/output/dropout/random_uniform/RandomUniformRandomUniform3bert/encoder/layer_5/attention/output/dropout/Shape*

seed *
T0*
dtype0*
seed2 
Ô
@bert/encoder/layer_5/attention/output/dropout/random_uniform/subSub@bert/encoder/layer_5/attention/output/dropout/random_uniform/max@bert/encoder/layer_5/attention/output/dropout/random_uniform/min*
T0
Ū
@bert/encoder/layer_5/attention/output/dropout/random_uniform/mulMulJbert/encoder/layer_5/attention/output/dropout/random_uniform/RandomUniform@bert/encoder/layer_5/attention/output/dropout/random_uniform/sub*
T0
Đ
<bert/encoder/layer_5/attention/output/dropout/random_uniformAdd@bert/encoder/layer_5/attention/output/dropout/random_uniform/mul@bert/encoder/layer_5/attention/output/dropout/random_uniform/min*
T0
¸
1bert/encoder/layer_5/attention/output/dropout/addAdd7bert/encoder/layer_5/attention/output/dropout/keep_prob<bert/encoder/layer_5/attention/output/dropout/random_uniform*
T0
x
3bert/encoder/layer_5/attention/output/dropout/FloorFloor1bert/encoder/layer_5/attention/output/dropout/add*
T0
ŗ
1bert/encoder/layer_5/attention/output/dropout/divRealDiv3bert/encoder/layer_5/attention/output/dense/BiasAdd7bert/encoder/layer_5/attention/output/dropout/keep_prob*
T0
Š
1bert/encoder/layer_5/attention/output/dropout/mulMul1bert/encoder/layer_5/attention/output/dropout/div3bert/encoder/layer_5/attention/output/dropout/Floor*
T0

)bert/encoder/layer_5/attention/output/addAdd1bert/encoder/layer_5/attention/output/dropout/mulbert/encoder/Reshape_1*
T0
Õ
Jmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64bert/encoder/layer_5/attention/output/LayerNorm/beta*
shape:
Õ
Jmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*C
	container64bert/encoder/layer_5/attention/output/LayerNorm/beta
E
Initializer_8/zerosConst*
valueB*    *
dtype0

Assign_8AssignJmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/beta/gradientInitializer_8/zeros*
use_locking(*
T0*]
_classS
QOloc:@mio_variable/bert/encoder/layer_5/attention/output/LayerNorm/beta/gradient*
validate_shape(
×
Kmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*D
	container75bert/encoder/layer_5/attention/output/LayerNorm/gamma*
shape:
×
Kmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*D
	container75bert/encoder/layer_5/attention/output/LayerNorm/gamma
D
Initializer_9/onesConst*
valueB*  ?*
dtype0

Assign_9AssignKmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/gamma/gradientInitializer_9/ones*
use_locking(*
T0*^
_classT
RPloc:@mio_variable/bert/encoder/layer_5/attention/output/LayerNorm/gamma/gradient*
validate_shape(
|
Nbert/encoder/layer_5/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0
å
<bert/encoder/layer_5/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_5/attention/output/addNbert/encoder/layer_5/attention/output/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0

Dbert/encoder/layer_5/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_5/attention/output/LayerNorm/moments/mean*
T0
Ø
Ibert/encoder/layer_5/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_5/attention/output/addDbert/encoder/layer_5/attention/output/LayerNorm/moments/StopGradient*
T0

Rbert/encoder/layer_5/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0

@bert/encoder/layer_5/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_5/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_5/attention/output/LayerNorm/moments/variance/reduction_indices*
T0*

Tidx0*
	keep_dims(
l
?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ėŧ+*
dtype0
Đ
=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_5/attention/output/LayerNorm/moments/variance?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add/y*
T0

?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add*
T0
Û
=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/RsqrtKmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/gamma/variable*
T0
š
?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_5/attention/output/add=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul*
T0
Ė
?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_5/attention/output/LayerNorm/moments/mean=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul*
T0
Ú
=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/subSubJmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/beta/variable?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul_2*
T0
Ī
?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/sub*
T0
Î
Dmio_variable/bert/encoder/layer_5/intermediate/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*=
	container0.bert/encoder/layer_5/intermediate/dense/kernel
Î
Dmio_variable/bert/encoder/layer_5/intermediate/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_5/intermediate/dense/kernel*
shape:

Z
%Initializer_10/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_10/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_10/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_10/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_10/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

#Initializer_10/truncated_normal/mulMul/Initializer_10/truncated_normal/TruncatedNormal&Initializer_10/truncated_normal/stddev*
T0
z
Initializer_10/truncated_normalAdd#Initializer_10/truncated_normal/mul$Initializer_10/truncated_normal/mean*
T0

	Assign_10AssignDmio_variable/bert/encoder/layer_5/intermediate/dense/kernel/gradientInitializer_10/truncated_normal*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_5/intermediate/dense/kernel/gradient*
validate_shape(
Å
Bmio_variable/bert/encoder/layer_5/intermediate/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_5/intermediate/dense/bias*
shape:
Å
Bmio_variable/bert/encoder/layer_5/intermediate/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_5/intermediate/dense/bias*
shape:
S
$Initializer_11/zeros/shape_as_tensorConst*
dtype0*
valueB:
G
Initializer_11/zeros/ConstConst*
valueB
 *    *
dtype0
y
Initializer_11/zerosFill$Initializer_11/zeros/shape_as_tensorInitializer_11/zeros/Const*
T0*

index_type0
ö
	Assign_11AssignBmio_variable/bert/encoder/layer_5/intermediate/dense/bias/gradientInitializer_11/zeros*
use_locking(*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_5/intermediate/dense/bias/gradient*
validate_shape(
î
.bert/encoder/layer_5/intermediate/dense/MatMulMatMul?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add_1Dmio_variable/bert/encoder/layer_5/intermediate/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
Î
/bert/encoder/layer_5/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_5/intermediate/dense/MatMulBmio_variable/bert/encoder/layer_5/intermediate/dense/bias/variable*
T0*
data_formatNHWC
Z
-bert/encoder/layer_5/intermediate/dense/Pow/yConst*
valueB
 *  @@*
dtype0

+bert/encoder/layer_5/intermediate/dense/PowPow/bert/encoder/layer_5/intermediate/dense/BiasAdd-bert/encoder/layer_5/intermediate/dense/Pow/y*
T0
Z
-bert/encoder/layer_5/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0

+bert/encoder/layer_5/intermediate/dense/mulMul-bert/encoder/layer_5/intermediate/dense/mul/x+bert/encoder/layer_5/intermediate/dense/Pow*
T0

+bert/encoder/layer_5/intermediate/dense/addAdd/bert/encoder/layer_5/intermediate/dense/BiasAdd+bert/encoder/layer_5/intermediate/dense/mul*
T0
\
/bert/encoder/layer_5/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0

-bert/encoder/layer_5/intermediate/dense/mul_1Mul/bert/encoder/layer_5/intermediate/dense/mul_1/x+bert/encoder/layer_5/intermediate/dense/add*
T0
l
,bert/encoder/layer_5/intermediate/dense/TanhTanh-bert/encoder/layer_5/intermediate/dense/mul_1*
T0
\
/bert/encoder/layer_5/intermediate/dense/add_1/xConst*
valueB
 *  ?*
dtype0

-bert/encoder/layer_5/intermediate/dense/add_1Add/bert/encoder/layer_5/intermediate/dense/add_1/x,bert/encoder/layer_5/intermediate/dense/Tanh*
T0
\
/bert/encoder/layer_5/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0

-bert/encoder/layer_5/intermediate/dense/mul_2Mul/bert/encoder/layer_5/intermediate/dense/mul_2/x-bert/encoder/layer_5/intermediate/dense/add_1*
T0

-bert/encoder/layer_5/intermediate/dense/mul_3Mul/bert/encoder/layer_5/intermediate/dense/BiasAdd-bert/encoder/layer_5/intermediate/dense/mul_2*
T0
Â
>mio_variable/bert/encoder/layer_5/output/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(bert/encoder/layer_5/output/dense/kernel*
shape:

Â
>mio_variable/bert/encoder/layer_5/output/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*7
	container*(bert/encoder/layer_5/output/dense/kernel
Z
%Initializer_12/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_12/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_12/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_12/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_12/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0

#Initializer_12/truncated_normal/mulMul/Initializer_12/truncated_normal/TruncatedNormal&Initializer_12/truncated_normal/stddev*
T0
z
Initializer_12/truncated_normalAdd#Initializer_12/truncated_normal/mul$Initializer_12/truncated_normal/mean*
T0
ų
	Assign_12Assign>mio_variable/bert/encoder/layer_5/output/dense/kernel/gradientInitializer_12/truncated_normal*
use_locking(*
T0*Q
_classG
ECloc:@mio_variable/bert/encoder/layer_5/output/dense/kernel/gradient*
validate_shape(
š
<mio_variable/bert/encoder/layer_5/output/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&bert/encoder/layer_5/output/dense/bias*
shape:
š
<mio_variable/bert/encoder/layer_5/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*5
	container(&bert/encoder/layer_5/output/dense/bias
F
Initializer_13/zerosConst*
valueB*    *
dtype0
ę
	Assign_13Assign<mio_variable/bert/encoder/layer_5/output/dense/bias/gradientInitializer_13/zeros*
validate_shape(*
use_locking(*
T0*O
_classE
CAloc:@mio_variable/bert/encoder/layer_5/output/dense/bias/gradient
Đ
(bert/encoder/layer_5/output/dense/MatMulMatMul-bert/encoder/layer_5/intermediate/dense/mul_3>mio_variable/bert/encoder/layer_5/output/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 
ŧ
)bert/encoder/layer_5/output/dense/BiasAddBiasAdd(bert/encoder/layer_5/output/dense/MatMul<mio_variable/bert/encoder/layer_5/output/dense/bias/variable*
data_formatNHWC*
T0
Z
-bert/encoder/layer_5/output/dropout/keep_probConst*
valueB
 *fff?*
dtype0
v
)bert/encoder/layer_5/output/dropout/ShapeShape)bert/encoder/layer_5/output/dense/BiasAdd*
T0*
out_type0
c
6bert/encoder/layer_5/output/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
c
6bert/encoder/layer_5/output/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0
Ģ
@bert/encoder/layer_5/output/dropout/random_uniform/RandomUniformRandomUniform)bert/encoder/layer_5/output/dropout/Shape*
seed2 *

seed *
T0*
dtype0
ļ
6bert/encoder/layer_5/output/dropout/random_uniform/subSub6bert/encoder/layer_5/output/dropout/random_uniform/max6bert/encoder/layer_5/output/dropout/random_uniform/min*
T0
Ā
6bert/encoder/layer_5/output/dropout/random_uniform/mulMul@bert/encoder/layer_5/output/dropout/random_uniform/RandomUniform6bert/encoder/layer_5/output/dropout/random_uniform/sub*
T0
˛
2bert/encoder/layer_5/output/dropout/random_uniformAdd6bert/encoder/layer_5/output/dropout/random_uniform/mul6bert/encoder/layer_5/output/dropout/random_uniform/min*
T0

'bert/encoder/layer_5/output/dropout/addAdd-bert/encoder/layer_5/output/dropout/keep_prob2bert/encoder/layer_5/output/dropout/random_uniform*
T0
d
)bert/encoder/layer_5/output/dropout/FloorFloor'bert/encoder/layer_5/output/dropout/add*
T0

'bert/encoder/layer_5/output/dropout/divRealDiv)bert/encoder/layer_5/output/dense/BiasAdd-bert/encoder/layer_5/output/dropout/keep_prob*
T0

'bert/encoder/layer_5/output/dropout/mulMul'bert/encoder/layer_5/output/dropout/div)bert/encoder/layer_5/output/dropout/Floor*
T0

bert/encoder/layer_5/output/addAdd'bert/encoder/layer_5/output/dropout/mul?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add_1*
T0
Á
@mio_variable/bert/encoder/layer_5/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*9
	container,*bert/encoder/layer_5/output/LayerNorm/beta
Á
@mio_variable/bert/encoder/layer_5/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*bert/encoder/layer_5/output/LayerNorm/beta*
shape:
F
Initializer_14/zerosConst*
valueB*    *
dtype0
ō
	Assign_14Assign@mio_variable/bert/encoder/layer_5/output/LayerNorm/beta/gradientInitializer_14/zeros*S
_classI
GEloc:@mio_variable/bert/encoder/layer_5/output/LayerNorm/beta/gradient*
validate_shape(*
use_locking(*
T0
Ã
Amio_variable/bert/encoder/layer_5/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+bert/encoder/layer_5/output/LayerNorm/gamma*
shape:
Ã
Amio_variable/bert/encoder/layer_5/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+bert/encoder/layer_5/output/LayerNorm/gamma*
shape:
E
Initializer_15/onesConst*
valueB*  ?*
dtype0
ķ
	Assign_15AssignAmio_variable/bert/encoder/layer_5/output/LayerNorm/gamma/gradientInitializer_15/ones*
use_locking(*
T0*T
_classJ
HFloc:@mio_variable/bert/encoder/layer_5/output/LayerNorm/gamma/gradient*
validate_shape(
r
Dbert/encoder/layer_5/output/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
valueB:
Į
2bert/encoder/layer_5/output/LayerNorm/moments/meanMeanbert/encoder/layer_5/output/addDbert/encoder/layer_5/output/LayerNorm/moments/mean/reduction_indices*
T0*

Tidx0*
	keep_dims(

:bert/encoder/layer_5/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_5/output/LayerNorm/moments/mean*
T0
ē
?bert/encoder/layer_5/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_5/output/add:bert/encoder/layer_5/output/LayerNorm/moments/StopGradient*
T0
v
Hbert/encoder/layer_5/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0
ī
6bert/encoder/layer_5/output/LayerNorm/moments/varianceMean?bert/encoder/layer_5/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_5/output/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0
b
5bert/encoder/layer_5/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ėŧ+*
dtype0
˛
3bert/encoder/layer_5/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_5/output/LayerNorm/moments/variance5bert/encoder/layer_5/output/LayerNorm/batchnorm/add/y*
T0
|
5bert/encoder/layer_5/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_5/output/LayerNorm/batchnorm/add*
T0
Ŋ
3bert/encoder/layer_5/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_5/output/LayerNorm/batchnorm/RsqrtAmio_variable/bert/encoder/layer_5/output/LayerNorm/gamma/variable*
T0

5bert/encoder/layer_5/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_5/output/add3bert/encoder/layer_5/output/LayerNorm/batchnorm/mul*
T0
Ž
5bert/encoder/layer_5/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_5/output/LayerNorm/moments/mean3bert/encoder/layer_5/output/LayerNorm/batchnorm/mul*
T0
ŧ
3bert/encoder/layer_5/output/LayerNorm/batchnorm/subSub@mio_variable/bert/encoder/layer_5/output/LayerNorm/beta/variable5bert/encoder/layer_5/output/LayerNorm/batchnorm/mul_2*
T0
ą
5bert/encoder/layer_5/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_5/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_5/output/LayerNorm/batchnorm/sub*
T0
m
bert/encoder/Shape_3Shape5bert/encoder/layer_5/output/LayerNorm/batchnorm/add_1*
out_type0*
T0
P
"bert/encoder/strided_slice_3/stackConst*
valueB: *
dtype0
R
$bert/encoder/strided_slice_3/stack_1Const*
valueB:*
dtype0
R
$bert/encoder/strided_slice_3/stack_2Const*
valueB:*
dtype0
Ŧ
bert/encoder/strided_slice_3StridedSlicebert/encoder/Shape_3"bert/encoder/strided_slice_3/stack$bert/encoder/strided_slice_3/stack_1$bert/encoder/strided_slice_3/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask
H
bert/encoder/Reshape_2/shape/1Const*
value	B :*
dtype0
I
bert/encoder/Reshape_2/shape/2Const*
value
B :*
dtype0
 
bert/encoder/Reshape_2/shapePackbert/encoder/strided_slice_2bert/encoder/Reshape_2/shape/1bert/encoder/Reshape_2/shape/2*
T0*

axis *
N

bert/encoder/Reshape_2Reshape5bert/encoder/layer_5/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_2/shape*
Tshape0*
T0
X
bert/pooler/strided_slice/stackConst*!
valueB"            *
dtype0
Z
!bert/pooler/strided_slice/stack_1Const*!
valueB"           *
dtype0
Z
!bert/pooler/strided_slice/stack_2Const*!
valueB"         *
dtype0
ĸ
bert/pooler/strided_sliceStridedSlicebert/encoder/Reshape_2bert/pooler/strided_slice/stack!bert/pooler/strided_slice/stack_1!bert/pooler/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
Y
bert/pooler/SqueezeSqueezebert/pooler/strided_slice*
squeeze_dims
*
T0
ĸ
.mio_variable/bert/pooler/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerbert/pooler/dense/kernel*
shape:

ĸ
.mio_variable/bert/pooler/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*'
	containerbert/pooler/dense/kernel
Z
%Initializer_16/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_16/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_16/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_16/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_16/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 

#Initializer_16/truncated_normal/mulMul/Initializer_16/truncated_normal/TruncatedNormal&Initializer_16/truncated_normal/stddev*
T0
z
Initializer_16/truncated_normalAdd#Initializer_16/truncated_normal/mul$Initializer_16/truncated_normal/mean*
T0
Ų
	Assign_16Assign.mio_variable/bert/pooler/dense/kernel/gradientInitializer_16/truncated_normal*
validate_shape(*
use_locking(*
T0*A
_class7
53loc:@mio_variable/bert/pooler/dense/kernel/gradient

,mio_variable/bert/pooler/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerbert/pooler/dense/bias*
shape:

,mio_variable/bert/pooler/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerbert/pooler/dense/bias*
shape:
F
Initializer_17/zerosConst*
valueB*    *
dtype0
Ę
	Assign_17Assign,mio_variable/bert/pooler/dense/bias/gradientInitializer_17/zeros*
T0*?
_class5
31loc:@mio_variable/bert/pooler/dense/bias/gradient*
validate_shape(*
use_locking(

bert/pooler/dense/MatMulMatMulbert/pooler/Squeeze.mio_variable/bert/pooler/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0

bert/pooler/dense/BiasAddBiasAddbert/pooler/dense/MatMul,mio_variable/bert/pooler/dense/bias/variable*
T0*
data_formatNHWC
B
bert/pooler/dense/TanhTanhbert/pooler/dense/BiasAdd*
T0
N
strided_slice_1/stackConst*!
valueB"            *
dtype0
P
strided_slice_1/stack_1Const*!
valueB"           *
dtype0
P
strided_slice_1/stack_2Const*!
valueB"         *
dtype0
ú
strided_slice_1StridedSlicebert/encoder/Reshape_2strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
end_mask*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask 
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
Tparams0*
Taxis0*
Tindices0
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
concat/values_7GatherV2)mio_embeddings/context_embedding/variableCastconcat/values_7/axis*
Tindices0*
Tparams0*
Taxis0
>
concat/axisConst*
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
Ø
concatConcatV2concat/values_0&mio_embeddings/c_id_embedding/variable(mio_embeddings/c_info_embedding/variableconcat/values_3concat/values_4concat/values_5concat/values_6concat/values_7/mio_embeddings/comment_genre_embedding/variable0mio_embeddings/comment_length_embedding/variableconcat/axis*
T0*
N
*

Tidx0
@
concat_1/axisConst*
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
Z
concat_1ConcatV2concatstrided_slice_1concat_1/axis*

Tidx0*
T0*
N
@
concat_2/values_0/axisConst*
value	B : *
dtype0

concat_2/values_0GatherV2%mio_embeddings/did_embedding/variableCastconcat_2/values_0/axis*
Taxis0*
Tindices0*
Tparams0
@
concat_2/values_2/axisConst*
dtype0*
value	B : 

concat_2/values_2GatherV2)mio_embeddings/context_embedding/variableCastconcat_2/values_2/axis*
Taxis0*
Tindices0*
Tparams0
@
concat_2/axisConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0

concat_2ConcatV2concat_2/values_0*mio_embeddings/position_embedding/variableconcat_2/values_2concat_2/axis*

Tidx0*
T0*
N
 
-mio_variable/expand_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerexpand_xtr/dense/kernel*
shape:
°
 
-mio_variable/expand_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerexpand_xtr/dense/kernel*
shape:
°
X
#Initializer_18/random_uniform/shapeConst*
valueB"°     *
dtype0
N
!Initializer_18/random_uniform/minConst*
valueB
 *ÃĐŊ*
dtype0
N
!Initializer_18/random_uniform/maxConst*
valueB
 *ÃĐ=*
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
Õ
	Assign_18Assign-mio_variable/expand_xtr/dense/kernel/gradientInitializer_18/random_uniform*@
_class6
42loc:@mio_variable/expand_xtr/dense/kernel/gradient*
validate_shape(*
use_locking(*
T0

+mio_variable/expand_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerexpand_xtr/dense/bias*
shape:

+mio_variable/expand_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containerexpand_xtr/dense/bias
F
Initializer_19/zerosConst*
valueB*    *
dtype0
Č
	Assign_19Assign+mio_variable/expand_xtr/dense/bias/gradientInitializer_19/zeros*
use_locking(*
T0*>
_class4
20loc:@mio_variable/expand_xtr/dense/bias/gradient*
validate_shape(

expand_xtr/dense/MatMulMatMulconcat_1-mio_variable/expand_xtr/dense/kernel/variable*
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
X
#Initializer_20/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_20/random_uniform/minConst*
valueB
 *   ž*
dtype0
N
!Initializer_20/random_uniform/maxConst*
valueB
 *   >*
dtype0

+Initializer_20/random_uniform/RandomUniformRandomUniform#Initializer_20/random_uniform/shape*
seed2 *

seed *
T0*
dtype0
w
!Initializer_20/random_uniform/subSub!Initializer_20/random_uniform/max!Initializer_20/random_uniform/min*
T0

!Initializer_20/random_uniform/mulMul+Initializer_20/random_uniform/RandomUniform!Initializer_20/random_uniform/sub*
T0
s
Initializer_20/random_uniformAdd!Initializer_20/random_uniform/mul!Initializer_20/random_uniform/min*
T0
Ų
	Assign_20Assign/mio_variable/expand_xtr/dense_1/kernel/gradientInitializer_20/random_uniform*
use_locking(*
T0*B
_class8
64loc:@mio_variable/expand_xtr/dense_1/kernel/gradient*
validate_shape(

-mio_variable/expand_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerexpand_xtr/dense_1/bias*
shape:

-mio_variable/expand_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerexpand_xtr/dense_1/bias*
shape:
F
Initializer_21/zerosConst*
valueB*    *
dtype0
Ė
	Assign_21Assign-mio_variable/expand_xtr/dense_1/bias/gradientInitializer_21/zeros*
validate_shape(*
use_locking(*
T0*@
_class6
42loc:@mio_variable/expand_xtr/dense_1/bias/gradient
 
expand_xtr/dense_1/MatMulMatMulexpand_xtr/dropout/Identity/mio_variable/expand_xtr/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
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
P
expand_xtr/dropout_1/IdentityIdentityexpand_xtr/dense_1/LeakyRelu*
T0
Ŗ
/mio_variable/expand_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerexpand_xtr/dense_2/kernel*
shape:	@
Ŗ
/mio_variable/expand_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerexpand_xtr/dense_2/kernel*
shape:	@
X
#Initializer_22/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_22/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
N
!Initializer_22/random_uniform/maxConst*
dtype0*
valueB
 *ķ5>
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
Ų
	Assign_22Assign/mio_variable/expand_xtr/dense_2/kernel/gradientInitializer_22/random_uniform*
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
-mio_variable/expand_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerexpand_xtr/dense_2/bias*
shape:@
E
Initializer_23/zerosConst*
dtype0*
valueB@*    
Ė
	Assign_23Assign-mio_variable/expand_xtr/dense_2/bias/gradientInitializer_23/zeros*
use_locking(*
T0*@
_class6
42loc:@mio_variable/expand_xtr/dense_2/bias/gradient*
validate_shape(
ĸ
expand_xtr/dense_2/MatMulMatMulexpand_xtr/dropout_1/Identity/mio_variable/expand_xtr/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 
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
ĸ
/mio_variable/expand_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*(
	containerexpand_xtr/dense_3/kernel
ĸ
/mio_variable/expand_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerexpand_xtr/dense_3/kernel*
shape
:@
X
#Initializer_24/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_24/random_uniform/minConst*
dtype0*
valueB
 *ž
N
!Initializer_24/random_uniform/maxConst*
valueB
 *>*
dtype0

+Initializer_24/random_uniform/RandomUniformRandomUniform#Initializer_24/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_24/random_uniform/subSub!Initializer_24/random_uniform/max!Initializer_24/random_uniform/min*
T0

!Initializer_24/random_uniform/mulMul+Initializer_24/random_uniform/RandomUniform!Initializer_24/random_uniform/sub*
T0
s
Initializer_24/random_uniformAdd!Initializer_24/random_uniform/mul!Initializer_24/random_uniform/min*
T0
Ų
	Assign_24Assign/mio_variable/expand_xtr/dense_3/kernel/gradientInitializer_24/random_uniform*
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
E
Initializer_25/zerosConst*
valueB*    *
dtype0
Ė
	Assign_25Assign-mio_variable/expand_xtr/dense_3/bias/gradientInitializer_25/zeros*
use_locking(*
T0*@
_class6
42loc:@mio_variable/expand_xtr/dense_3/bias/gradient*
validate_shape(
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
°

+mio_variable/like_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*$
	containerlike_xtr/dense/kernel
X
#Initializer_26/random_uniform/shapeConst*
valueB"°     *
dtype0
N
!Initializer_26/random_uniform/minConst*
valueB
 *ÃĐŊ*
dtype0
N
!Initializer_26/random_uniform/maxConst*
valueB
 *ÃĐ=*
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
Ņ
	Assign_26Assign+mio_variable/like_xtr/dense/kernel/gradientInitializer_26/random_uniform*
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
)mio_variable/like_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*"
	containerlike_xtr/dense/bias*
shape:
F
Initializer_27/zerosConst*
valueB*    *
dtype0
Ä
	Assign_27Assign)mio_variable/like_xtr/dense/bias/gradientInitializer_27/zeros*
use_locking(*
T0*<
_class2
0.loc:@mio_variable/like_xtr/dense/bias/gradient*
validate_shape(

like_xtr/dense/MatMulMatMulconcat_1+mio_variable/like_xtr/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
H
like_xtr/dropout/IdentityIdentitylike_xtr/dense/LeakyRelu*
T0
 
-mio_variable/like_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*&
	containerlike_xtr/dense_1/kernel
 
-mio_variable/like_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerlike_xtr/dense_1/kernel*
shape:

X
#Initializer_28/random_uniform/shapeConst*
dtype0*
valueB"      
N
!Initializer_28/random_uniform/minConst*
valueB
 *   ž*
dtype0
N
!Initializer_28/random_uniform/maxConst*
valueB
 *   >*
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
	Assign_28Assign-mio_variable/like_xtr/dense_1/kernel/gradientInitializer_28/random_uniform*
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
Initializer_29/zerosConst*
valueB*    *
dtype0
Č
	Assign_29Assign+mio_variable/like_xtr/dense_1/bias/gradientInitializer_29/zeros*
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
#Initializer_30/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_30/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
N
!Initializer_30/random_uniform/maxConst*
valueB
 *ķ5>*
dtype0

+Initializer_30/random_uniform/RandomUniformRandomUniform#Initializer_30/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
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
	Assign_30Assign-mio_variable/like_xtr/dense_2/kernel/gradientInitializer_30/random_uniform*
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
Initializer_31/zerosConst*
valueB@*    *
dtype0
Č
	Assign_31Assign+mio_variable/like_xtr/dense_2/bias/gradientInitializer_31/zeros*>
_class4
20loc:@mio_variable/like_xtr/dense_2/bias/gradient*
validate_shape(*
use_locking(*
T0

like_xtr/dense_2/MatMulMatMullike_xtr/dropout_1/Identity-mio_variable/like_xtr/dense_2/kernel/variable*
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
#Initializer_32/random_uniform/shapeConst*
dtype0*
valueB"@      
N
!Initializer_32/random_uniform/minConst*
valueB
 *ž*
dtype0
N
!Initializer_32/random_uniform/maxConst*
valueB
 *>*
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
Õ
	Assign_32Assign-mio_variable/like_xtr/dense_3/kernel/gradientInitializer_32/random_uniform*
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
+mio_variable/like_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containerlike_xtr/dense_3/bias
E
Initializer_33/zerosConst*
valueB*    *
dtype0
Č
	Assign_33Assign+mio_variable/like_xtr/dense_3/bias/gradientInitializer_33/zeros*
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
like_xtr/dense_3/BiasAddBiasAddlike_xtr/dense_3/MatMul+mio_variable/like_xtr/dense_3/bias/variable*
T0*
data_formatNHWC
F
like_xtr/dense_3/SigmoidSigmoidlike_xtr/dense_3/BiasAdd*
T0

,mio_variable/reply_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense/kernel*
shape:
°

,mio_variable/reply_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense/kernel*
shape:
°
X
#Initializer_34/random_uniform/shapeConst*
valueB"°     *
dtype0
N
!Initializer_34/random_uniform/minConst*
valueB
 *ÃĐŊ*
dtype0
N
!Initializer_34/random_uniform/maxConst*
valueB
 *ÃĐ=*
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
Ķ
	Assign_34Assign,mio_variable/reply_xtr/dense/kernel/gradientInitializer_34/random_uniform*
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
Initializer_35/zerosConst*
dtype0*
valueB*    
Æ
	Assign_35Assign*mio_variable/reply_xtr/dense/bias/gradientInitializer_35/zeros*
use_locking(*
T0*=
_class3
1/loc:@mio_variable/reply_xtr/dense/bias/gradient*
validate_shape(

reply_xtr/dense/MatMulMatMulconcat_1,mio_variable/reply_xtr/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
J
reply_xtr/dropout/IdentityIdentityreply_xtr/dense/LeakyRelu*
T0
ĸ
.mio_variable/reply_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerreply_xtr/dense_1/kernel*
shape:

ĸ
.mio_variable/reply_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*'
	containerreply_xtr/dense_1/kernel
X
#Initializer_36/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_36/random_uniform/minConst*
valueB
 *   ž*
dtype0
N
!Initializer_36/random_uniform/maxConst*
valueB
 *   >*
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
	Assign_36Assign.mio_variable/reply_xtr/dense_1/kernel/gradientInitializer_36/random_uniform*A
_class7
53loc:@mio_variable/reply_xtr/dense_1/kernel/gradient*
validate_shape(*
use_locking(*
T0

,mio_variable/reply_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense_1/bias*
shape:

,mio_variable/reply_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense_1/bias*
shape:
F
Initializer_37/zerosConst*
valueB*    *
dtype0
Ę
	Assign_37Assign,mio_variable/reply_xtr/dense_1/bias/gradientInitializer_37/zeros*?
_class5
31loc:@mio_variable/reply_xtr/dense_1/bias/gradient*
validate_shape(*
use_locking(*
T0

reply_xtr/dense_1/MatMulMatMulreply_xtr/dropout/Identity.mio_variable/reply_xtr/dense_1/kernel/variable*
transpose_b( *
T0*
transpose_a( 

reply_xtr/dense_1/BiasAddBiasAddreply_xtr/dense_1/MatMul,mio_variable/reply_xtr/dense_1/bias/variable*
T0*
data_formatNHWC
N
!reply_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
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
Ą
.mio_variable/reply_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerreply_xtr/dense_2/kernel*
shape:	@
Ą
.mio_variable/reply_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerreply_xtr/dense_2/kernel*
shape:	@
X
#Initializer_38/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_38/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
N
!Initializer_38/random_uniform/maxConst*
valueB
 *ķ5>*
dtype0

+Initializer_38/random_uniform/RandomUniformRandomUniform#Initializer_38/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
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
	Assign_38Assign.mio_variable/reply_xtr/dense_2/kernel/gradientInitializer_38/random_uniform*A
_class7
53loc:@mio_variable/reply_xtr/dense_2/kernel/gradient*
validate_shape(*
use_locking(*
T0

,mio_variable/reply_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*%
	containerreply_xtr/dense_2/bias

,mio_variable/reply_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*%
	containerreply_xtr/dense_2/bias
E
Initializer_39/zerosConst*
valueB@*    *
dtype0
Ę
	Assign_39Assign,mio_variable/reply_xtr/dense_2/bias/gradientInitializer_39/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/reply_xtr/dense_2/bias/gradient*
validate_shape(

reply_xtr/dense_2/MatMulMatMulreply_xtr/dropout_1/Identity.mio_variable/reply_xtr/dense_2/kernel/variable*
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
.mio_variable/reply_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerreply_xtr/dense_3/kernel*
shape
:@
 
.mio_variable/reply_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*'
	containerreply_xtr/dense_3/kernel
X
#Initializer_40/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_40/random_uniform/minConst*
valueB
 *ž*
dtype0
N
!Initializer_40/random_uniform/maxConst*
valueB
 *>*
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
×
	Assign_40Assign.mio_variable/reply_xtr/dense_3/kernel/gradientInitializer_40/random_uniform*
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
Initializer_41/zerosConst*
valueB*    *
dtype0
Ę
	Assign_41Assign,mio_variable/reply_xtr/dense_3/bias/gradientInitializer_41/zeros*
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
°

+mio_variable/copy_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense/kernel*
shape:
°
X
#Initializer_42/random_uniform/shapeConst*
valueB"°     *
dtype0
N
!Initializer_42/random_uniform/minConst*
valueB
 *ÃĐŊ*
dtype0
N
!Initializer_42/random_uniform/maxConst*
valueB
 *ÃĐ=*
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
Ņ
	Assign_42Assign+mio_variable/copy_xtr/dense/kernel/gradientInitializer_42/random_uniform*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@mio_variable/copy_xtr/dense/kernel/gradient

)mio_variable/copy_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*"
	containercopy_xtr/dense/bias

)mio_variable/copy_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*"
	containercopy_xtr/dense/bias*
shape:
F
Initializer_43/zerosConst*
valueB*    *
dtype0
Ä
	Assign_43Assign)mio_variable/copy_xtr/dense/bias/gradientInitializer_43/zeros*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@mio_variable/copy_xtr/dense/bias/gradient

copy_xtr/dense/MatMulMatMulconcat_1+mio_variable/copy_xtr/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
#Initializer_44/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_44/random_uniform/minConst*
valueB
 *   ž*
dtype0
N
!Initializer_44/random_uniform/maxConst*
valueB
 *   >*
dtype0

+Initializer_44/random_uniform/RandomUniformRandomUniform#Initializer_44/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_44/random_uniform/subSub!Initializer_44/random_uniform/max!Initializer_44/random_uniform/min*
T0

!Initializer_44/random_uniform/mulMul+Initializer_44/random_uniform/RandomUniform!Initializer_44/random_uniform/sub*
T0
s
Initializer_44/random_uniformAdd!Initializer_44/random_uniform/mul!Initializer_44/random_uniform/min*
T0
Õ
	Assign_44Assign-mio_variable/copy_xtr/dense_1/kernel/gradientInitializer_44/random_uniform*
T0*@
_class6
42loc:@mio_variable/copy_xtr/dense_1/kernel/gradient*
validate_shape(*
use_locking(

+mio_variable/copy_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense_1/bias*
shape:

+mio_variable/copy_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containercopy_xtr/dense_1/bias
F
Initializer_45/zerosConst*
valueB*    *
dtype0
Č
	Assign_45Assign+mio_variable/copy_xtr/dense_1/bias/gradientInitializer_45/zeros*
T0*>
_class4
20loc:@mio_variable/copy_xtr/dense_1/bias/gradient*
validate_shape(*
use_locking(
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
 *ÍĖL>*
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
#Initializer_46/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_46/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
N
!Initializer_46/random_uniform/maxConst*
valueB
 *ķ5>*
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
Õ
	Assign_46Assign-mio_variable/copy_xtr/dense_2/kernel/gradientInitializer_46/random_uniform*
use_locking(*
T0*@
_class6
42loc:@mio_variable/copy_xtr/dense_2/kernel/gradient*
validate_shape(

+mio_variable/copy_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense_2/bias*
shape:@

+mio_variable/copy_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense_2/bias*
shape:@
E
Initializer_47/zerosConst*
dtype0*
valueB@*    
Č
	Assign_47Assign+mio_variable/copy_xtr/dense_2/bias/gradientInitializer_47/zeros*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@mio_variable/copy_xtr/dense_2/bias/gradient
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
 copy_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
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
#Initializer_48/random_uniform/shapeConst*
dtype0*
valueB"@      
N
!Initializer_48/random_uniform/minConst*
valueB
 *ž*
dtype0
N
!Initializer_48/random_uniform/maxConst*
valueB
 *>*
dtype0

+Initializer_48/random_uniform/RandomUniformRandomUniform#Initializer_48/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_48/random_uniform/subSub!Initializer_48/random_uniform/max!Initializer_48/random_uniform/min*
T0

!Initializer_48/random_uniform/mulMul+Initializer_48/random_uniform/RandomUniform!Initializer_48/random_uniform/sub*
T0
s
Initializer_48/random_uniformAdd!Initializer_48/random_uniform/mul!Initializer_48/random_uniform/min*
T0
Õ
	Assign_48Assign-mio_variable/copy_xtr/dense_3/kernel/gradientInitializer_48/random_uniform*
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
+mio_variable/copy_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containercopy_xtr/dense_3/bias
E
Initializer_49/zerosConst*
valueB*    *
dtype0
Č
	Assign_49Assign+mio_variable/copy_xtr/dense_3/bias/gradientInitializer_49/zeros*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@mio_variable/copy_xtr/dense_3/bias/gradient
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
°

,mio_variable/share_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*%
	containershare_xtr/dense/kernel
X
#Initializer_50/random_uniform/shapeConst*
valueB"°     *
dtype0
N
!Initializer_50/random_uniform/minConst*
valueB
 *ÃĐŊ*
dtype0
N
!Initializer_50/random_uniform/maxConst*
valueB
 *ÃĐ=*
dtype0

+Initializer_50/random_uniform/RandomUniformRandomUniform#Initializer_50/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_50/random_uniform/subSub!Initializer_50/random_uniform/max!Initializer_50/random_uniform/min*
T0

!Initializer_50/random_uniform/mulMul+Initializer_50/random_uniform/RandomUniform!Initializer_50/random_uniform/sub*
T0
s
Initializer_50/random_uniformAdd!Initializer_50/random_uniform/mul!Initializer_50/random_uniform/min*
T0
Ķ
	Assign_50Assign,mio_variable/share_xtr/dense/kernel/gradientInitializer_50/random_uniform*
validate_shape(*
use_locking(*
T0*?
_class5
31loc:@mio_variable/share_xtr/dense/kernel/gradient

*mio_variable/share_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*#
	containershare_xtr/dense/bias

*mio_variable/share_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*#
	containershare_xtr/dense/bias*
shape:
F
Initializer_51/zerosConst*
valueB*    *
dtype0
Æ
	Assign_51Assign*mio_variable/share_xtr/dense/bias/gradientInitializer_51/zeros*
use_locking(*
T0*=
_class3
1/loc:@mio_variable/share_xtr/dense/bias/gradient*
validate_shape(

share_xtr/dense/MatMulMatMulconcat_1,mio_variable/share_xtr/dense/kernel/variable*
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
 *ÍĖL>*
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
ĸ
.mio_variable/share_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containershare_xtr/dense_1/kernel*
shape:

ĸ
.mio_variable/share_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containershare_xtr/dense_1/kernel*
shape:

X
#Initializer_52/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_52/random_uniform/minConst*
valueB
 *   ž*
dtype0
N
!Initializer_52/random_uniform/maxConst*
valueB
 *   >*
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
×
	Assign_52Assign.mio_variable/share_xtr/dense_1/kernel/gradientInitializer_52/random_uniform*
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
Initializer_53/zerosConst*
valueB*    *
dtype0
Ę
	Assign_53Assign,mio_variable/share_xtr/dense_1/bias/gradientInitializer_53/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/share_xtr/dense_1/bias/gradient*
validate_shape(

share_xtr/dense_1/MatMulMatMulshare_xtr/dropout/Identity.mio_variable/share_xtr/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
N
share_xtr/dropout_1/IdentityIdentityshare_xtr/dense_1/LeakyRelu*
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
#Initializer_54/random_uniform/shapeConst*
dtype0*
valueB"   @   
N
!Initializer_54/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
N
!Initializer_54/random_uniform/maxConst*
valueB
 *ķ5>*
dtype0

+Initializer_54/random_uniform/RandomUniformRandomUniform#Initializer_54/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
w
!Initializer_54/random_uniform/subSub!Initializer_54/random_uniform/max!Initializer_54/random_uniform/min*
T0

!Initializer_54/random_uniform/mulMul+Initializer_54/random_uniform/RandomUniform!Initializer_54/random_uniform/sub*
T0
s
Initializer_54/random_uniformAdd!Initializer_54/random_uniform/mul!Initializer_54/random_uniform/min*
T0
×
	Assign_54Assign.mio_variable/share_xtr/dense_2/kernel/gradientInitializer_54/random_uniform*
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
Initializer_55/zerosConst*
valueB@*    *
dtype0
Ę
	Assign_55Assign,mio_variable/share_xtr/dense_2/bias/gradientInitializer_55/zeros*
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
.mio_variable/share_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containershare_xtr/dense_3/kernel*
shape
:@
X
#Initializer_56/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_56/random_uniform/minConst*
valueB
 *ž*
dtype0
N
!Initializer_56/random_uniform/maxConst*
valueB
 *>*
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
×
	Assign_56Assign.mio_variable/share_xtr/dense_3/kernel/gradientInitializer_56/random_uniform*
validate_shape(*
use_locking(*
T0*A
_class7
53loc:@mio_variable/share_xtr/dense_3/kernel/gradient

,mio_variable/share_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containershare_xtr/dense_3/bias*
shape:

,mio_variable/share_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containershare_xtr/dense_3/bias*
shape:
E
Initializer_57/zerosConst*
valueB*    *
dtype0
Ę
	Assign_57Assign,mio_variable/share_xtr/dense_3/bias/gradientInitializer_57/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/share_xtr/dense_3/bias/gradient*
validate_shape(

share_xtr/dense_3/MatMulMatMulshare_xtr/dense_2/LeakyRelu.mio_variable/share_xtr/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 

share_xtr/dense_3/BiasAddBiasAddshare_xtr/dense_3/MatMul,mio_variable/share_xtr/dense_3/bias/variable*
data_formatNHWC*
T0
H
share_xtr/dense_3/SigmoidSigmoidshare_xtr/dense_3/BiasAdd*
T0
¤
/mio_variable/audience_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeraudience_xtr/dense/kernel*
shape:
°
¤
/mio_variable/audience_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeraudience_xtr/dense/kernel*
shape:
°
X
#Initializer_58/random_uniform/shapeConst*
dtype0*
valueB"°     
N
!Initializer_58/random_uniform/minConst*
valueB
 *ÃĐŊ*
dtype0
N
!Initializer_58/random_uniform/maxConst*
valueB
 *ÃĐ=*
dtype0
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
Ų
	Assign_58Assign/mio_variable/audience_xtr/dense/kernel/gradientInitializer_58/random_uniform*
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
-mio_variable/audience_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containeraudience_xtr/dense/bias*
shape:
F
Initializer_59/zerosConst*
valueB*    *
dtype0
Ė
	Assign_59Assign-mio_variable/audience_xtr/dense/bias/gradientInitializer_59/zeros*
use_locking(*
T0*@
_class6
42loc:@mio_variable/audience_xtr/dense/bias/gradient*
validate_shape(

audience_xtr/dense/MatMulMatMulconcat_1/mio_variable/audience_xtr/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 

audience_xtr/dense/BiasAddBiasAddaudience_xtr/dense/MatMul-mio_variable/audience_xtr/dense/bias/variable*
data_formatNHWC*
T0
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
P
audience_xtr/dropout/IdentityIdentityaudience_xtr/dense/LeakyRelu*
T0
¨
1mio_variable/audience_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS**
	containeraudience_xtr/dense_1/kernel*
shape:

¨
1mio_variable/audience_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
**
	containeraudience_xtr/dense_1/kernel
X
#Initializer_60/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_60/random_uniform/minConst*
valueB
 *   ž*
dtype0
N
!Initializer_60/random_uniform/maxConst*
valueB
 *   >*
dtype0
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
Ũ
	Assign_60Assign1mio_variable/audience_xtr/dense_1/kernel/gradientInitializer_60/random_uniform*
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
Initializer_61/zerosConst*
valueB*    *
dtype0
Đ
	Assign_61Assign/mio_variable/audience_xtr/dense_1/bias/gradientInitializer_61/zeros*
validate_shape(*
use_locking(*
T0*B
_class8
64loc:@mio_variable/audience_xtr/dense_1/bias/gradient
Ļ
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
 *ÍĖL>*
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
1mio_variable/audience_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@**
	containeraudience_xtr/dense_2/kernel
§
1mio_variable/audience_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@**
	containeraudience_xtr/dense_2/kernel
X
#Initializer_62/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_62/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
N
!Initializer_62/random_uniform/maxConst*
dtype0*
valueB
 *ķ5>

+Initializer_62/random_uniform/RandomUniformRandomUniform#Initializer_62/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_62/random_uniform/subSub!Initializer_62/random_uniform/max!Initializer_62/random_uniform/min*
T0

!Initializer_62/random_uniform/mulMul+Initializer_62/random_uniform/RandomUniform!Initializer_62/random_uniform/sub*
T0
s
Initializer_62/random_uniformAdd!Initializer_62/random_uniform/mul!Initializer_62/random_uniform/min*
T0
Ũ
	Assign_62Assign1mio_variable/audience_xtr/dense_2/kernel/gradientInitializer_62/random_uniform*
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
Initializer_63/zerosConst*
valueB@*    *
dtype0
Đ
	Assign_63Assign/mio_variable/audience_xtr/dense_2/bias/gradientInitializer_63/zeros*
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
 *ÍĖL>*
dtype0
v
"audience_xtr/dense_2/LeakyRelu/mulMul$audience_xtr/dense_2/LeakyRelu/alphaaudience_xtr/dense_2/BiasAdd*
T0
t
audience_xtr/dense_2/LeakyReluMaximum"audience_xtr/dense_2/LeakyRelu/mulaudience_xtr/dense_2/BiasAdd*
T0
Ļ
1mio_variable/audience_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@**
	containeraudience_xtr/dense_3/kernel
Ļ
1mio_variable/audience_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@**
	containeraudience_xtr/dense_3/kernel
X
#Initializer_64/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_64/random_uniform/minConst*
valueB
 *ž*
dtype0
N
!Initializer_64/random_uniform/maxConst*
valueB
 *>*
dtype0
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
Ũ
	Assign_64Assign1mio_variable/audience_xtr/dense_3/kernel/gradientInitializer_64/random_uniform*
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
/mio_variable/audience_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*(
	containeraudience_xtr/dense_3/bias
E
Initializer_65/zerosConst*
valueB*    *
dtype0
Đ
	Assign_65Assign/mio_variable/audience_xtr/dense_3/bias/gradientInitializer_65/zeros*
validate_shape(*
use_locking(*
T0*B
_class8
64loc:@mio_variable/audience_xtr/dense_3/bias/gradient
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
ļ
8mio_variable/continuous_expand_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*1
	container$"continuous_expand_xtr/dense/kernel
ļ
8mio_variable/continuous_expand_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"continuous_expand_xtr/dense/kernel*
shape:
°
X
#Initializer_66/random_uniform/shapeConst*
valueB"°     *
dtype0
N
!Initializer_66/random_uniform/minConst*
valueB
 *ÃĐŊ*
dtype0
N
!Initializer_66/random_uniform/maxConst*
dtype0*
valueB
 *ÃĐ=
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
ë
	Assign_66Assign8mio_variable/continuous_expand_xtr/dense/kernel/gradientInitializer_66/random_uniform*K
_classA
?=loc:@mio_variable/continuous_expand_xtr/dense/kernel/gradient*
validate_shape(*
use_locking(*
T0
­
6mio_variable/continuous_expand_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" continuous_expand_xtr/dense/bias*
shape:
­
6mio_variable/continuous_expand_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*/
	container" continuous_expand_xtr/dense/bias*
shape:
F
Initializer_67/zerosConst*
valueB*    *
dtype0
Ū
	Assign_67Assign6mio_variable/continuous_expand_xtr/dense/bias/gradientInitializer_67/zeros*
T0*I
_class?
=;loc:@mio_variable/continuous_expand_xtr/dense/bias/gradient*
validate_shape(*
use_locking(

"continuous_expand_xtr/dense/MatMulMatMulconcat_18mio_variable/continuous_expand_xtr/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Ē
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
b
&continuous_expand_xtr/dropout/IdentityIdentity%continuous_expand_xtr/dense/LeakyRelu*
T0
ē
:mio_variable/continuous_expand_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$continuous_expand_xtr/dense_1/kernel*
shape:

ē
:mio_variable/continuous_expand_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$continuous_expand_xtr/dense_1/kernel*
shape:

X
#Initializer_68/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_68/random_uniform/minConst*
valueB
 *   ž*
dtype0
N
!Initializer_68/random_uniform/maxConst*
valueB
 *   >*
dtype0

+Initializer_68/random_uniform/RandomUniformRandomUniform#Initializer_68/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_68/random_uniform/subSub!Initializer_68/random_uniform/max!Initializer_68/random_uniform/min*
T0

!Initializer_68/random_uniform/mulMul+Initializer_68/random_uniform/RandomUniform!Initializer_68/random_uniform/sub*
T0
s
Initializer_68/random_uniformAdd!Initializer_68/random_uniform/mul!Initializer_68/random_uniform/min*
T0
ī
	Assign_68Assign:mio_variable/continuous_expand_xtr/dense_1/kernel/gradientInitializer_68/random_uniform*
use_locking(*
T0*M
_classC
A?loc:@mio_variable/continuous_expand_xtr/dense_1/kernel/gradient*
validate_shape(
ą
8mio_variable/continuous_expand_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"continuous_expand_xtr/dense_1/bias*
shape:
ą
8mio_variable/continuous_expand_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"continuous_expand_xtr/dense_1/bias*
shape:
F
Initializer_69/zerosConst*
valueB*    *
dtype0
â
	Assign_69Assign8mio_variable/continuous_expand_xtr/dense_1/bias/gradientInitializer_69/zeros*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/continuous_expand_xtr/dense_1/bias/gradient*
validate_shape(
Á
$continuous_expand_xtr/dense_1/MatMulMatMul&continuous_expand_xtr/dropout/Identity:mio_variable/continuous_expand_xtr/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
f
(continuous_expand_xtr/dropout_1/IdentityIdentity'continuous_expand_xtr/dense_1/LeakyRelu*
T0
š
:mio_variable/continuous_expand_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$continuous_expand_xtr/dense_2/kernel*
shape:	@
š
:mio_variable/continuous_expand_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*3
	container&$continuous_expand_xtr/dense_2/kernel
X
#Initializer_70/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_70/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
N
!Initializer_70/random_uniform/maxConst*
valueB
 *ķ5>*
dtype0

+Initializer_70/random_uniform/RandomUniformRandomUniform#Initializer_70/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_70/random_uniform/subSub!Initializer_70/random_uniform/max!Initializer_70/random_uniform/min*
T0

!Initializer_70/random_uniform/mulMul+Initializer_70/random_uniform/RandomUniform!Initializer_70/random_uniform/sub*
T0
s
Initializer_70/random_uniformAdd!Initializer_70/random_uniform/mul!Initializer_70/random_uniform/min*
T0
ī
	Assign_70Assign:mio_variable/continuous_expand_xtr/dense_2/kernel/gradientInitializer_70/random_uniform*
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
8mio_variable/continuous_expand_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*1
	container$"continuous_expand_xtr/dense_2/bias*
shape:@
E
Initializer_71/zerosConst*
valueB@*    *
dtype0
â
	Assign_71Assign8mio_variable/continuous_expand_xtr/dense_2/bias/gradientInitializer_71/zeros*K
_classA
?=loc:@mio_variable/continuous_expand_xtr/dense_2/bias/gradient*
validate_shape(*
use_locking(*
T0
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
 *ÍĖL>*
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
:mio_variable/continuous_expand_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$continuous_expand_xtr/dense_3/kernel*
shape
:@
X
#Initializer_72/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_72/random_uniform/minConst*
valueB
 *ž*
dtype0
N
!Initializer_72/random_uniform/maxConst*
valueB
 *>*
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
ī
	Assign_72Assign:mio_variable/continuous_expand_xtr/dense_3/kernel/gradientInitializer_72/random_uniform*
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
Initializer_73/zerosConst*
dtype0*
valueB*    
â
	Assign_73Assign8mio_variable/continuous_expand_xtr/dense_3/bias/gradientInitializer_73/zeros*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/continuous_expand_xtr/dense_3/bias/gradient*
validate_shape(
Â
$continuous_expand_xtr/dense_3/MatMulMatMul'continuous_expand_xtr/dense_2/LeakyRelu:mio_variable/continuous_expand_xtr/dense_3/kernel/variable*
transpose_a( *
transpose_b( *
T0
°
%continuous_expand_xtr/dense_3/BiasAddBiasAdd$continuous_expand_xtr/dense_3/MatMul8mio_variable/continuous_expand_xtr/dense_3/bias/variable*
T0*
data_formatNHWC
`
%continuous_expand_xtr/dense_3/SigmoidSigmoid%continuous_expand_xtr/dense_3/BiasAdd*
T0
Ŧ
3mio_variable/duration_predict/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerduration_predict/dense/kernel*
shape:
°
Ŧ
3mio_variable/duration_predict/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*,
	containerduration_predict/dense/kernel
X
#Initializer_74/random_uniform/shapeConst*
valueB"°     *
dtype0
N
!Initializer_74/random_uniform/minConst*
valueB
 *ÃĐŊ*
dtype0
N
!Initializer_74/random_uniform/maxConst*
valueB
 *ÃĐ=*
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
á
	Assign_74Assign3mio_variable/duration_predict/dense/kernel/gradientInitializer_74/random_uniform*
use_locking(*
T0*F
_class<
:8loc:@mio_variable/duration_predict/dense/kernel/gradient*
validate_shape(
Ŗ
1mio_variable/duration_predict/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS**
	containerduration_predict/dense/bias*
shape:
Ŗ
1mio_variable/duration_predict/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS**
	containerduration_predict/dense/bias*
shape:
F
Initializer_75/zerosConst*
dtype0*
valueB*    
Ô
	Assign_75Assign1mio_variable/duration_predict/dense/bias/gradientInitializer_75/zeros*
use_locking(*
T0*D
_class:
86loc:@mio_variable/duration_predict/dense/bias/gradient*
validate_shape(

duration_predict/dense/MatMulMatMulconcat_13mio_variable/duration_predict/dense/kernel/variable*
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
 *ÍĖL>*
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
5mio_variable/duration_predict/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*.
	container!duration_predict/dense_1/kernel
X
#Initializer_76/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_76/random_uniform/minConst*
valueB
 *   ž*
dtype0
N
!Initializer_76/random_uniform/maxConst*
valueB
 *   >*
dtype0

+Initializer_76/random_uniform/RandomUniformRandomUniform#Initializer_76/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
w
!Initializer_76/random_uniform/subSub!Initializer_76/random_uniform/max!Initializer_76/random_uniform/min*
T0

!Initializer_76/random_uniform/mulMul+Initializer_76/random_uniform/RandomUniform!Initializer_76/random_uniform/sub*
T0
s
Initializer_76/random_uniformAdd!Initializer_76/random_uniform/mul!Initializer_76/random_uniform/min*
T0
å
	Assign_76Assign5mio_variable/duration_predict/dense_1/kernel/gradientInitializer_76/random_uniform*H
_class>
<:loc:@mio_variable/duration_predict/dense_1/kernel/gradient*
validate_shape(*
use_locking(*
T0
§
3mio_variable/duration_predict/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerduration_predict/dense_1/bias*
shape:
§
3mio_variable/duration_predict/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*,
	containerduration_predict/dense_1/bias
F
Initializer_77/zerosConst*
valueB*    *
dtype0
Ø
	Assign_77Assign3mio_variable/duration_predict/dense_1/bias/gradientInitializer_77/zeros*
use_locking(*
T0*F
_class<
:8loc:@mio_variable/duration_predict/dense_1/bias/gradient*
validate_shape(
˛
duration_predict/dense_1/MatMulMatMul!duration_predict/dropout/Identity5mio_variable/duration_predict/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Ą
 duration_predict/dense_1/BiasAddBiasAddduration_predict/dense_1/MatMul3mio_variable/duration_predict/dense_1/bias/variable*
data_formatNHWC*
T0
U
(duration_predict/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
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
5mio_variable/duration_predict/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!duration_predict/dense_2/kernel*
shape:	@
¯
5mio_variable/duration_predict/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!duration_predict/dense_2/kernel*
shape:	@
X
#Initializer_78/random_uniform/shapeConst*
dtype0*
valueB"   @   
N
!Initializer_78/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
N
!Initializer_78/random_uniform/maxConst*
valueB
 *ķ5>*
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
å
	Assign_78Assign5mio_variable/duration_predict/dense_2/kernel/gradientInitializer_78/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/duration_predict/dense_2/kernel/gradient*
validate_shape(
Ļ
3mio_variable/duration_predict/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerduration_predict/dense_2/bias*
shape:@
Ļ
3mio_variable/duration_predict/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerduration_predict/dense_2/bias*
shape:@
E
Initializer_79/zerosConst*
dtype0*
valueB@*    
Ø
	Assign_79Assign3mio_variable/duration_predict/dense_2/bias/gradientInitializer_79/zeros*F
_class<
:8loc:@mio_variable/duration_predict/dense_2/bias/gradient*
validate_shape(*
use_locking(*
T0
´
duration_predict/dense_2/MatMulMatMul#duration_predict/dropout_1/Identity5mio_variable/duration_predict/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0
Ą
 duration_predict/dense_2/BiasAddBiasAddduration_predict/dense_2/MatMul3mio_variable/duration_predict/dense_2/bias/variable*
T0*
data_formatNHWC
U
(duration_predict/dense_2/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍĖL>

&duration_predict/dense_2/LeakyRelu/mulMul(duration_predict/dense_2/LeakyRelu/alpha duration_predict/dense_2/BiasAdd*
T0

"duration_predict/dense_2/LeakyReluMaximum&duration_predict/dense_2/LeakyRelu/mul duration_predict/dense_2/BiasAdd*
T0
Ž
5mio_variable/duration_predict/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!duration_predict/dense_3/kernel*
shape
:@
Ž
5mio_variable/duration_predict/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*.
	container!duration_predict/dense_3/kernel
X
#Initializer_80/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_80/random_uniform/minConst*
valueB
 *ž*
dtype0
N
!Initializer_80/random_uniform/maxConst*
valueB
 *>*
dtype0

+Initializer_80/random_uniform/RandomUniformRandomUniform#Initializer_80/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
w
!Initializer_80/random_uniform/subSub!Initializer_80/random_uniform/max!Initializer_80/random_uniform/min*
T0

!Initializer_80/random_uniform/mulMul+Initializer_80/random_uniform/RandomUniform!Initializer_80/random_uniform/sub*
T0
s
Initializer_80/random_uniformAdd!Initializer_80/random_uniform/mul!Initializer_80/random_uniform/min*
T0
å
	Assign_80Assign5mio_variable/duration_predict/dense_3/kernel/gradientInitializer_80/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/duration_predict/dense_3/kernel/gradient*
validate_shape(
Ļ
3mio_variable/duration_predict/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerduration_predict/dense_3/bias*
shape:
Ļ
3mio_variable/duration_predict/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerduration_predict/dense_3/bias*
shape:
E
Initializer_81/zerosConst*
valueB*    *
dtype0
Ø
	Assign_81Assign3mio_variable/duration_predict/dense_3/bias/gradientInitializer_81/zeros*
use_locking(*
T0*F
_class<
:8loc:@mio_variable/duration_predict/dense_3/bias/gradient*
validate_shape(
ŗ
duration_predict/dense_3/MatMulMatMul"duration_predict/dense_2/LeakyRelu5mio_variable/duration_predict/dense_3/kernel/variable*
transpose_b( *
T0*
transpose_a( 
Ą
 duration_predict/dense_3/BiasAddBiasAddduration_predict/dense_3/MatMul3mio_variable/duration_predict/dense_3/bias/variable*
T0*
data_formatNHWC
P
duration_predict/dense_3/ReluRelu duration_predict/dense_3/BiasAdd*
T0
ž
<mio_variable/duration_pos_bias_predict/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*5
	container(&duration_pos_bias_predict/dense/kernel
ž
<mio_variable/duration_pos_bias_predict/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&duration_pos_bias_predict/dense/kernel*
shape:

X
#Initializer_82/random_uniform/shapeConst*
valueB"      *
dtype0
N
!Initializer_82/random_uniform/minConst*
valueB
 *˛_ž*
dtype0
N
!Initializer_82/random_uniform/maxConst*
valueB
 *˛_>*
dtype0
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
ķ
	Assign_82Assign<mio_variable/duration_pos_bias_predict/dense/kernel/gradientInitializer_82/random_uniform*
use_locking(*
T0*O
_classE
CAloc:@mio_variable/duration_pos_bias_predict/dense/kernel/gradient*
validate_shape(
ĩ
:mio_variable/duration_pos_bias_predict/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$duration_pos_bias_predict/dense/bias*
shape:
ĩ
:mio_variable/duration_pos_bias_predict/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*3
	container&$duration_pos_bias_predict/dense/bias
F
Initializer_83/zerosConst*
valueB*    *
dtype0
æ
	Assign_83Assign:mio_variable/duration_pos_bias_predict/dense/bias/gradientInitializer_83/zeros*
use_locking(*
T0*M
_classC
A?loc:@mio_variable/duration_pos_bias_predict/dense/bias/gradient*
validate_shape(
§
&duration_pos_bias_predict/dense/MatMulMatMulconcat_2<mio_variable/duration_pos_bias_predict/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
ļ
'duration_pos_bias_predict/dense/BiasAddBiasAdd&duration_pos_bias_predict/dense/MatMul:mio_variable/duration_pos_bias_predict/dense/bias/variable*
T0*
data_formatNHWC
\
/duration_pos_bias_predict/dense/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
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
>mio_variable/duration_pos_bias_predict/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*7
	container*(duration_pos_bias_predict/dense_1/kernel
X
#Initializer_84/random_uniform/shapeConst*
valueB"   @   *
dtype0
N
!Initializer_84/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
N
!Initializer_84/random_uniform/maxConst*
valueB
 *ķ5>*
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
÷
	Assign_84Assign>mio_variable/duration_pos_bias_predict/dense_1/kernel/gradientInitializer_84/random_uniform*
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
Initializer_85/zerosConst*
valueB@*    *
dtype0
ę
	Assign_85Assign<mio_variable/duration_pos_bias_predict/dense_1/bias/gradientInitializer_85/zeros*
T0*O
_classE
CAloc:@mio_variable/duration_pos_bias_predict/dense_1/bias/gradient*
validate_shape(*
use_locking(
Í
(duration_pos_bias_predict/dense_1/MatMulMatMul*duration_pos_bias_predict/dropout/Identity>mio_variable/duration_pos_bias_predict/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0
ŧ
)duration_pos_bias_predict/dense_1/BiasAddBiasAdd(duration_pos_bias_predict/dense_1/MatMul<mio_variable/duration_pos_bias_predict/dense_1/bias/variable*
T0*
data_formatNHWC
^
1duration_pos_bias_predict/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0

/duration_pos_bias_predict/dense_1/LeakyRelu/mulMul1duration_pos_bias_predict/dense_1/LeakyRelu/alpha)duration_pos_bias_predict/dense_1/BiasAdd*
T0

+duration_pos_bias_predict/dense_1/LeakyReluMaximum/duration_pos_bias_predict/dense_1/LeakyRelu/mul)duration_pos_bias_predict/dense_1/BiasAdd*
T0
Ā
>mio_variable/duration_pos_bias_predict/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*7
	container*(duration_pos_bias_predict/dense_2/kernel
Ā
>mio_variable/duration_pos_bias_predict/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*7
	container*(duration_pos_bias_predict/dense_2/kernel
X
#Initializer_86/random_uniform/shapeConst*
valueB"@      *
dtype0
N
!Initializer_86/random_uniform/minConst*
valueB
 *ž*
dtype0
N
!Initializer_86/random_uniform/maxConst*
valueB
 *>*
dtype0

+Initializer_86/random_uniform/RandomUniformRandomUniform#Initializer_86/random_uniform/shape*
seed2 *

seed *
T0*
dtype0
w
!Initializer_86/random_uniform/subSub!Initializer_86/random_uniform/max!Initializer_86/random_uniform/min*
T0

!Initializer_86/random_uniform/mulMul+Initializer_86/random_uniform/RandomUniform!Initializer_86/random_uniform/sub*
T0
s
Initializer_86/random_uniformAdd!Initializer_86/random_uniform/mul!Initializer_86/random_uniform/min*
T0
÷
	Assign_86Assign>mio_variable/duration_pos_bias_predict/dense_2/kernel/gradientInitializer_86/random_uniform*
use_locking(*
T0*Q
_classG
ECloc:@mio_variable/duration_pos_bias_predict/dense_2/kernel/gradient*
validate_shape(
¸
<mio_variable/duration_pos_bias_predict/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*5
	container(&duration_pos_bias_predict/dense_2/bias
¸
<mio_variable/duration_pos_bias_predict/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&duration_pos_bias_predict/dense_2/bias*
shape:
E
Initializer_87/zerosConst*
valueB*    *
dtype0
ę
	Assign_87Assign<mio_variable/duration_pos_bias_predict/dense_2/bias/gradientInitializer_87/zeros*
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
ŧ
)duration_pos_bias_predict/dense_2/BiasAddBiasAdd(duration_pos_bias_predict/dense_2/MatMul<mio_variable/duration_pos_bias_predict/dense_2/bias/variable*
T0*
data_formatNHWC
b
&duration_pos_bias_predict/dense_2/ReluRelu)duration_pos_bias_predict/dense_2/BiasAdd*
T0"