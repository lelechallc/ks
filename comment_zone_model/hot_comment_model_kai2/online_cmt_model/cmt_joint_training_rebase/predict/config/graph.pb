
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
	containerpid_embedding*
shape:˙˙˙˙˙˙˙˙˙@

%mio_embeddings/pid_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:˙˙˙˙˙˙˙˙˙@*
	containerpid_embedding

%mio_embeddings/aid_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:˙˙˙˙˙˙˙˙˙@*
	containeraid_embedding
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
shape:˙˙˙˙˙˙˙˙˙@*
	containerdid_embedding

%mio_embeddings/did_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:˙˙˙˙˙˙˙˙˙@*
	containerdid_embedding

)mio_embeddings/context_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS* 
	containercontext_embedding*
shape:˙˙˙˙˙˙˙˙˙@

)mio_embeddings/context_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:˙˙˙˙˙˙˙˙˙@* 
	containercontext_embedding
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
	containerc_info_embedding*
shape:˙˙˙˙˙˙˙˙˙Ā
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
Š
/mio_embeddings/comment_genre_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:˙˙˙˙˙˙˙˙˙*&
	containercomment_genre_embedding
Š
/mio_embeddings/comment_genre_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containercomment_genre_embedding*
shape:˙˙˙˙˙˙˙˙˙
Ģ
0mio_embeddings/comment_length_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containercomment_length_embedding*
shape:˙˙˙˙˙˙˙˙˙ 
Ģ
0mio_embeddings/comment_length_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containercomment_length_embedding*
shape:˙˙˙˙˙˙˙˙˙ 

(mio_extra_param/token_input_ids/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containertoken_input_ids*
shape:˙˙˙˙˙˙˙˙˙

(mio_extra_param/token_input_ids/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:˙˙˙˙˙˙˙˙˙*
	containertoken_input_ids
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
shape:˙˙˙˙˙˙˙˙˙*
	containertoken_sep_ids
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
strided_slice/stack_2Const*
dtype0*
valueB:
á
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
U
bert/embeddings/ExpandDims/dimConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
e
bert/embeddings/ExpandDims
ExpandDimsCast_1bert/embeddings/ExpandDims/dim*

Tdim0*
T0
ą
5mio_variable/bert/embeddings/word_embeddings/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!bert/embeddings/word_embeddings*
shape:Ĩ
ą
5mio_variable/bert/embeddings/word_embeddings/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!bert/embeddings/word_embeddings*
shape:Ĩ
W
"Initializer/truncated_normal/shapeConst*
valueB"R     *
dtype0
N
!Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0
P
#Initializer/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

,Initializer/truncated_normal/TruncatedNormalTruncatedNormal"Initializer/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 

 Initializer/truncated_normal/mulMul,Initializer/truncated_normal/TruncatedNormal#Initializer/truncated_normal/stddev*
T0
q
Initializer/truncated_normalAdd Initializer/truncated_normal/mul!Initializer/truncated_normal/mean*
T0
á
AssignAssign5mio_variable/bert/embeddings/word_embeddings/gradientInitializer/truncated_normal*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/bert/embeddings/word_embeddings/gradient*
validate_shape(
T
bert/embeddings/Reshape/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
t
bert/embeddings/ReshapeReshapebert/embeddings/ExpandDimsbert/embeddings/Reshape/shape*
T0*
Tshape0
G
bert/embeddings/GatherV2/axisConst*
value	B : *
dtype0
ŋ
bert/embeddings/GatherV2GatherV25mio_variable/bert/embeddings/word_embeddings/variablebert/embeddings/Reshapebert/embeddings/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0
S
bert/embeddings/ShapeShapebert/embeddings/ExpandDims*
T0*
out_type0
Q
#bert/embeddings/strided_slice/stackConst*
valueB: *
dtype0
S
%bert/embeddings/strided_slice/stack_1Const*
valueB:*
dtype0
S
%bert/embeddings/strided_slice/stack_2Const*
valueB:*
dtype0
ą
bert/embeddings/strided_sliceStridedSlicebert/embeddings/Shape#bert/embeddings/strided_slice/stack%bert/embeddings/strided_slice/stack_1%bert/embeddings/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0
K
!bert/embeddings/Reshape_1/shape/1Const*
value	B :*
dtype0
L
!bert/embeddings/Reshape_1/shape/2Const*
value
B :*
dtype0
Ē
bert/embeddings/Reshape_1/shapePackbert/embeddings/strided_slice!bert/embeddings/Reshape_1/shape/1!bert/embeddings/Reshape_1/shape/2*
T0*

axis *
N
v
bert/embeddings/Reshape_1Reshapebert/embeddings/GatherV2bert/embeddings/Reshape_1/shape*
T0*
Tshape0
T
bert/embeddings/Shape_1Shapebert/embeddings/Reshape_1*
T0*
out_type0
S
%bert/embeddings/strided_slice_1/stackConst*
valueB: *
dtype0
U
'bert/embeddings/strided_slice_1/stack_1Const*
valueB:*
dtype0
U
'bert/embeddings/strided_slice_1/stack_2Const*
valueB:*
dtype0
ģ
bert/embeddings/strided_slice_1StridedSlicebert/embeddings/Shape_1%bert/embeddings/strided_slice_1/stack'bert/embeddings/strided_slice_1/stack_1'bert/embeddings/strided_slice_1/stack_2*
end_mask *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
ģ
;mio_variable/bert/embeddings/token_type_embeddings/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	*4
	container'%bert/embeddings/token_type_embeddings
ģ
;mio_variable/bert/embeddings/token_type_embeddings/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*4
	container'%bert/embeddings/token_type_embeddings*
shape:	
Y
$Initializer_1/truncated_normal/shapeConst*
valueB"      *
dtype0
P
#Initializer_1/truncated_normal/meanConst*
dtype0*
valueB
 *    
R
%Initializer_1/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

.Initializer_1/truncated_normal/TruncatedNormalTruncatedNormal$Initializer_1/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 

"Initializer_1/truncated_normal/mulMul.Initializer_1/truncated_normal/TruncatedNormal%Initializer_1/truncated_normal/stddev*
T0
w
Initializer_1/truncated_normalAdd"Initializer_1/truncated_normal/mul#Initializer_1/truncated_normal/mean*
T0
ņ
Assign_1Assign;mio_variable/bert/embeddings/token_type_embeddings/gradientInitializer_1/truncated_normal*
use_locking(*
T0*N
_classD
B@loc:@mio_variable/bert/embeddings/token_type_embeddings/gradient*
validate_shape(
V
bert/embeddings/Reshape_2/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
d
bert/embeddings/Reshape_2ReshapeCast_3bert/embeddings/Reshape_2/shape*
T0*
Tshape0
M
 bert/embeddings/one_hot/on_valueConst*
valueB
 *  ?*
dtype0
N
!bert/embeddings/one_hot/off_valueConst*
valueB
 *    *
dtype0
G
bert/embeddings/one_hot/depthConst*
value	B :*
dtype0
Č
bert/embeddings/one_hotOneHotbert/embeddings/Reshape_2bert/embeddings/one_hot/depth bert/embeddings/one_hot/on_value!bert/embeddings/one_hot/off_value*
T0*
TI0*
axis˙˙˙˙˙˙˙˙˙
Ĩ
bert/embeddings/MatMulMatMulbert/embeddings/one_hot;mio_variable/bert/embeddings/token_type_embeddings/variable*
transpose_a( *
transpose_b( *
T0
K
!bert/embeddings/Reshape_3/shape/1Const*
value	B :*
dtype0
L
!bert/embeddings/Reshape_3/shape/2Const*
dtype0*
value
B :
Ŧ
bert/embeddings/Reshape_3/shapePackbert/embeddings/strided_slice_1!bert/embeddings/Reshape_3/shape/1!bert/embeddings/Reshape_3/shape/2*

axis *
N*
T0
t
bert/embeddings/Reshape_3Reshapebert/embeddings/MatMulbert/embeddings/Reshape_3/shape*
T0*
Tshape0
Y
bert/embeddings/addAddbert/embeddings/Reshape_1bert/embeddings/Reshape_3*
T0
M
#bert/embeddings/assert_less_equal/xConst*
value	B :*
dtype0
N
#bert/embeddings/assert_less_equal/yConst*
value
B :*
dtype0

+bert/embeddings/assert_less_equal/LessEqual	LessEqual#bert/embeddings/assert_less_equal/x#bert/embeddings/assert_less_equal/y*
T0
P
'bert/embeddings/assert_less_equal/ConstConst*
valueB *
dtype0

%bert/embeddings/assert_less_equal/AllAll+bert/embeddings/assert_less_equal/LessEqual'bert/embeddings/assert_less_equal/Const*

Tidx0*
	keep_dims( 
W
.bert/embeddings/assert_less_equal/Assert/ConstConst*
valueB B *
dtype0
°
0bert/embeddings/assert_less_equal/Assert/Const_1Const*h
value_B] BWCondition x <= y did not hold element-wise:x (bert/embeddings/assert_less_equal/x:0) = *
dtype0

0bert/embeddings/assert_less_equal/Assert/Const_2Const*=
value4B2 B,y (bert/embeddings/assert_less_equal/y:0) = *
dtype0
_
6bert/embeddings/assert_less_equal/Assert/Assert/data_0Const*
valueB B *
dtype0
ļ
6bert/embeddings/assert_less_equal/Assert/Assert/data_1Const*h
value_B] BWCondition x <= y did not hold element-wise:x (bert/embeddings/assert_less_equal/x:0) = *
dtype0

6bert/embeddings/assert_less_equal/Assert/Assert/data_3Const*=
value4B2 B,y (bert/embeddings/assert_less_equal/y:0) = *
dtype0
ķ
/bert/embeddings/assert_less_equal/Assert/AssertAssert%bert/embeddings/assert_less_equal/All6bert/embeddings/assert_less_equal/Assert/Assert/data_06bert/embeddings/assert_less_equal/Assert/Assert/data_1#bert/embeddings/assert_less_equal/x6bert/embeddings/assert_less_equal/Assert/Assert/data_3#bert/embeddings/assert_less_equal/y*
T	
2*
	summarize
¸
9mio_variable/bert/embeddings/position_embeddings/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#bert/embeddings/position_embeddings*
shape:

¸
9mio_variable/bert/embeddings/position_embeddings/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*2
	container%#bert/embeddings/position_embeddings*
shape:

Y
$Initializer_2/truncated_normal/shapeConst*
valueB"      *
dtype0
P
#Initializer_2/truncated_normal/meanConst*
dtype0*
valueB
 *    
R
%Initializer_2/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

.Initializer_2/truncated_normal/TruncatedNormalTruncatedNormal$Initializer_2/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0

"Initializer_2/truncated_normal/mulMul.Initializer_2/truncated_normal/TruncatedNormal%Initializer_2/truncated_normal/stddev*
T0
w
Initializer_2/truncated_normalAdd"Initializer_2/truncated_normal/mul#Initializer_2/truncated_normal/mean*
T0
í
Assign_2Assign9mio_variable/bert/embeddings/position_embeddings/gradientInitializer_2/truncated_normal*
use_locking(*
T0*L
_classB
@>loc:@mio_variable/bert/embeddings/position_embeddings/gradient*
validate_shape(

bert/embeddings/Slice/beginConst0^bert/embeddings/assert_less_equal/Assert/Assert*
valueB"        *
dtype0

bert/embeddings/Slice/sizeConst0^bert/embeddings/assert_less_equal/Assert/Assert*
valueB"   ˙˙˙˙*
dtype0
¨
bert/embeddings/SliceSlice9mio_variable/bert/embeddings/position_embeddings/variablebert/embeddings/Slice/beginbert/embeddings/Slice/size*
T0*
Index0

bert/embeddings/Reshape_4/shapeConst0^bert/embeddings/assert_less_equal/Assert/Assert*!
valueB"         *
dtype0
s
bert/embeddings/Reshape_4Reshapebert/embeddings/Slicebert/embeddings/Reshape_4/shape*
T0*
Tshape0
U
bert/embeddings/add_1Addbert/embeddings/addbert/embeddings/Reshape_4*
T0
Š
4mio_variable/bert/embeddings/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*-
	container bert/embeddings/LayerNorm/beta*
shape:
Š
4mio_variable/bert/embeddings/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*-
	container bert/embeddings/LayerNorm/beta
E
Initializer_3/zerosConst*
valueB*    *
dtype0
Ø
Assign_3Assign4mio_variable/bert/embeddings/LayerNorm/beta/gradientInitializer_3/zeros*
validate_shape(*
use_locking(*
T0*G
_class=
;9loc:@mio_variable/bert/embeddings/LayerNorm/beta/gradient
Ģ
5mio_variable/bert/embeddings/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!bert/embeddings/LayerNorm/gamma*
shape:
Ģ
5mio_variable/bert/embeddings/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!bert/embeddings/LayerNorm/gamma*
shape:
D
Initializer_4/onesConst*
valueB*  ?*
dtype0
Ų
Assign_4Assign5mio_variable/bert/embeddings/LayerNorm/gamma/gradientInitializer_4/ones*
T0*H
_class>
<:loc:@mio_variable/bert/embeddings/LayerNorm/gamma/gradient*
validate_shape(*
use_locking(
f
8bert/embeddings/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0
Ĩ
&bert/embeddings/LayerNorm/moments/meanMeanbert/embeddings/add_18bert/embeddings/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
o
.bert/embeddings/LayerNorm/moments/StopGradientStopGradient&bert/embeddings/LayerNorm/moments/mean*
T0

3bert/embeddings/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/embeddings/add_1.bert/embeddings/LayerNorm/moments/StopGradient*
T0
j
<bert/embeddings/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0
Ë
*bert/embeddings/LayerNorm/moments/varianceMean3bert/embeddings/LayerNorm/moments/SquaredDifference<bert/embeddings/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0
V
)bert/embeddings/LayerNorm/batchnorm/add/yConst*
valueB
 *Ėŧ+*
dtype0

'bert/embeddings/LayerNorm/batchnorm/addAdd*bert/embeddings/LayerNorm/moments/variance)bert/embeddings/LayerNorm/batchnorm/add/y*
T0
d
)bert/embeddings/LayerNorm/batchnorm/RsqrtRsqrt'bert/embeddings/LayerNorm/batchnorm/add*
T0

'bert/embeddings/LayerNorm/batchnorm/mulMul)bert/embeddings/LayerNorm/batchnorm/Rsqrt5mio_variable/bert/embeddings/LayerNorm/gamma/variable*
T0
y
)bert/embeddings/LayerNorm/batchnorm/mul_1Mulbert/embeddings/add_1'bert/embeddings/LayerNorm/batchnorm/mul*
T0

)bert/embeddings/LayerNorm/batchnorm/mul_2Mul&bert/embeddings/LayerNorm/moments/mean'bert/embeddings/LayerNorm/batchnorm/mul*
T0

'bert/embeddings/LayerNorm/batchnorm/subSub4mio_variable/bert/embeddings/LayerNorm/beta/variable)bert/embeddings/LayerNorm/batchnorm/mul_2*
T0

)bert/embeddings/LayerNorm/batchnorm/add_1Add)bert/embeddings/LayerNorm/batchnorm/mul_1'bert/embeddings/LayerNorm/batchnorm/sub*
T0
N
!bert/embeddings/dropout/keep_probConst*
dtype0*
valueB
 *fff?
j
bert/embeddings/dropout/ShapeShape)bert/embeddings/LayerNorm/batchnorm/add_1*
T0*
out_type0
W
*bert/embeddings/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
W
*bert/embeddings/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0

4bert/embeddings/dropout/random_uniform/RandomUniformRandomUniformbert/embeddings/dropout/Shape*
T0*
dtype0*
seed2 *

seed 

*bert/embeddings/dropout/random_uniform/subSub*bert/embeddings/dropout/random_uniform/max*bert/embeddings/dropout/random_uniform/min*
T0

*bert/embeddings/dropout/random_uniform/mulMul4bert/embeddings/dropout/random_uniform/RandomUniform*bert/embeddings/dropout/random_uniform/sub*
T0

&bert/embeddings/dropout/random_uniformAdd*bert/embeddings/dropout/random_uniform/mul*bert/embeddings/dropout/random_uniform/min*
T0
v
bert/embeddings/dropout/addAdd!bert/embeddings/dropout/keep_prob&bert/embeddings/dropout/random_uniform*
T0
L
bert/embeddings/dropout/FloorFloorbert/embeddings/dropout/add*
T0
}
bert/embeddings/dropout/divRealDiv)bert/embeddings/LayerNorm/batchnorm/add_1!bert/embeddings/dropout/keep_prob*
T0
g
bert/embeddings/dropout/mulMulbert/embeddings/dropout/divbert/embeddings/dropout/Floor*
T0
<
bert/encoder/ShapeShapeCast_1*
T0*
out_type0
N
 bert/encoder/strided_slice/stackConst*
valueB: *
dtype0
P
"bert/encoder/strided_slice/stack_1Const*
valueB:*
dtype0
P
"bert/encoder/strided_slice/stack_2Const*
valueB:*
dtype0
ĸ
bert/encoder/strided_sliceStridedSlicebert/encoder/Shape bert/encoder/strided_slice/stack"bert/encoder/strided_slice/stack_1"bert/encoder/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
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
$bert/encoder/strided_slice_1/stack_1Const*
valueB:*
dtype0
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
end_mask *
Index0*
T0
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
bert/encoder/CastCastbert/encoder/Reshape*

SrcT0*
Truncate( *

DstT0
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
bert/encoder/ones/Less/yConst*
value
B :č*
dtype0
Z
bert/encoder/ones/LessLessbert/encoder/ones/mul_1bert/encoder/ones/Less/y*
T0
D
bert/encoder/ones/packed/1Const*
dtype0*
value	B :
D
bert/encoder/ones/packed/2Const*
value	B :*
dtype0

bert/encoder/ones/packedPackbert/encoder/strided_slicebert/encoder/ones/packed/1bert/encoder/ones/packed/2*
N*
T0*

axis 
D
bert/encoder/ones/ConstConst*
dtype0*
valueB
 *  ?
g
bert/encoder/onesFillbert/encoder/ones/packedbert/encoder/ones/Const*
T0*

index_type0
F
bert/encoder/mulMulbert/encoder/onesbert/encoder/Cast*
T0
S
bert/encoder/Shape_2Shapebert/embeddings/dropout/mul*
T0*
out_type0
P
"bert/encoder/strided_slice_2/stackConst*
dtype0*
valueB: 
R
$bert/encoder/strided_slice_2/stack_1Const*
valueB:*
dtype0
R
$bert/encoder/strided_slice_2/stack_2Const*
valueB:*
dtype0
Ŧ
bert/encoder/strided_slice_2StridedSlicebert/encoder/Shape_2"bert/encoder/strided_slice_2/stack$bert/encoder/strided_slice_2/stack_1$bert/encoder/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
Q
bert/encoder/Reshape_1/shapeConst*
dtype0*
valueB"˙˙˙˙   
s
bert/encoder/Reshape_1Reshapebert/embeddings/dropout/mulbert/encoder/Reshape_1/shape*
T0*
Tshape0
c
)bert/encoder/layer_0/attention/self/ShapeShapebert/encoder/Reshape_1*
T0*
out_type0
e
7bert/encoder/layer_0/attention/self/strided_slice/stackConst*
dtype0*
valueB: 
g
9bert/encoder/layer_0/attention/self/strided_slice/stack_1Const*
valueB:*
dtype0
g
9bert/encoder/layer_0/attention/self/strided_slice/stack_2Const*
valueB:*
dtype0

1bert/encoder/layer_0/attention/self/strided_sliceStridedSlice)bert/encoder/layer_0/attention/self/Shape7bert/encoder/layer_0/attention/self/strided_slice/stack9bert/encoder/layer_0/attention/self/strided_slice/stack_19bert/encoder/layer_0/attention/self/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0
e
+bert/encoder/layer_0/attention/self/Shape_1Shapebert/encoder/Reshape_1*
T0*
out_type0
g
9bert/encoder/layer_0/attention/self/strided_slice_1/stackConst*
valueB: *
dtype0
i
;bert/encoder/layer_0/attention/self/strided_slice_1/stack_1Const*
dtype0*
valueB:
i
;bert/encoder/layer_0/attention/self/strided_slice_1/stack_2Const*
valueB:*
dtype0

3bert/encoder/layer_0/attention/self/strided_slice_1StridedSlice+bert/encoder/layer_0/attention/self/Shape_19bert/encoder/layer_0/attention/self/strided_slice_1/stack;bert/encoder/layer_0/attention/self/strided_slice_1/stack_1;bert/encoder/layer_0/attention/self/strided_slice_1/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
Ō
Fmio_variable/bert/encoder/layer_0/attention/self/query/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_0/attention/self/query/kernel*
shape:

Ō
Fmio_variable/bert/encoder/layer_0/attention/self/query/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_0/attention/self/query/kernel*
shape:

Y
$Initializer_5/truncated_normal/shapeConst*
valueB"      *
dtype0
P
#Initializer_5/truncated_normal/meanConst*
valueB
 *    *
dtype0
R
%Initializer_5/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

.Initializer_5/truncated_normal/TruncatedNormalTruncatedNormal$Initializer_5/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

"Initializer_5/truncated_normal/mulMul.Initializer_5/truncated_normal/TruncatedNormal%Initializer_5/truncated_normal/stddev*
T0
w
Initializer_5/truncated_normalAdd"Initializer_5/truncated_normal/mul#Initializer_5/truncated_normal/mean*
T0

Assign_5AssignFmio_variable/bert/encoder/layer_0/attention/self/query/kernel/gradientInitializer_5/truncated_normal*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_0/attention/self/query/kernel/gradient*
validate_shape(
É
Dmio_variable/bert/encoder/layer_0/attention/self/query/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_0/attention/self/query/bias*
shape:
É
Dmio_variable/bert/encoder/layer_0/attention/self/query/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_0/attention/self/query/bias*
shape:
E
Initializer_6/zerosConst*
valueB*    *
dtype0
ø
Assign_6AssignDmio_variable/bert/encoder/layer_0/attention/self/query/bias/gradientInitializer_6/zeros*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_0/attention/self/query/bias/gradient*
validate_shape(*
use_locking(
É
0bert/encoder/layer_0/attention/self/query/MatMulMatMulbert/encoder/Reshape_1Fmio_variable/bert/encoder/layer_0/attention/self/query/kernel/variable*
transpose_a( *
transpose_b( *
T0
Ô
1bert/encoder/layer_0/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_0/attention/self/query/MatMulDmio_variable/bert/encoder/layer_0/attention/self/query/bias/variable*
T0*
data_formatNHWC
Î
Dmio_variable/bert/encoder/layer_0/attention/self/key/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*=
	container0.bert/encoder/layer_0/attention/self/key/kernel
Î
Dmio_variable/bert/encoder/layer_0/attention/self/key/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_0/attention/self/key/kernel*
shape:

Y
$Initializer_7/truncated_normal/shapeConst*
valueB"      *
dtype0
P
#Initializer_7/truncated_normal/meanConst*
valueB
 *    *
dtype0
R
%Initializer_7/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

.Initializer_7/truncated_normal/TruncatedNormalTruncatedNormal$Initializer_7/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0

"Initializer_7/truncated_normal/mulMul.Initializer_7/truncated_normal/TruncatedNormal%Initializer_7/truncated_normal/stddev*
T0
w
Initializer_7/truncated_normalAdd"Initializer_7/truncated_normal/mul#Initializer_7/truncated_normal/mean*
T0

Assign_7AssignDmio_variable/bert/encoder/layer_0/attention/self/key/kernel/gradientInitializer_7/truncated_normal*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_0/attention/self/key/kernel/gradient*
validate_shape(*
use_locking(
Å
Bmio_variable/bert/encoder/layer_0/attention/self/key/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_0/attention/self/key/bias*
shape:
Å
Bmio_variable/bert/encoder/layer_0/attention/self/key/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_0/attention/self/key/bias*
shape:
E
Initializer_8/zerosConst*
valueB*    *
dtype0
ô
Assign_8AssignBmio_variable/bert/encoder/layer_0/attention/self/key/bias/gradientInitializer_8/zeros*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_0/attention/self/key/bias/gradient*
validate_shape(*
use_locking(
Å
.bert/encoder/layer_0/attention/self/key/MatMulMatMulbert/encoder/Reshape_1Dmio_variable/bert/encoder/layer_0/attention/self/key/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Î
/bert/encoder/layer_0/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_0/attention/self/key/MatMulBmio_variable/bert/encoder/layer_0/attention/self/key/bias/variable*
data_formatNHWC*
T0
Ō
Fmio_variable/bert/encoder/layer_0/attention/self/value/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*?
	container20bert/encoder/layer_0/attention/self/value/kernel
Ō
Fmio_variable/bert/encoder/layer_0/attention/self/value/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*?
	container20bert/encoder/layer_0/attention/self/value/kernel
Y
$Initializer_9/truncated_normal/shapeConst*
valueB"      *
dtype0
P
#Initializer_9/truncated_normal/meanConst*
valueB
 *    *
dtype0
R
%Initializer_9/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

.Initializer_9/truncated_normal/TruncatedNormalTruncatedNormal$Initializer_9/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0

"Initializer_9/truncated_normal/mulMul.Initializer_9/truncated_normal/TruncatedNormal%Initializer_9/truncated_normal/stddev*
T0
w
Initializer_9/truncated_normalAdd"Initializer_9/truncated_normal/mul#Initializer_9/truncated_normal/mean*
T0

Assign_9AssignFmio_variable/bert/encoder/layer_0/attention/self/value/kernel/gradientInitializer_9/truncated_normal*
validate_shape(*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_0/attention/self/value/kernel/gradient
É
Dmio_variable/bert/encoder/layer_0/attention/self/value/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_0/attention/self/value/bias*
shape:
É
Dmio_variable/bert/encoder/layer_0/attention/self/value/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*=
	container0.bert/encoder/layer_0/attention/self/value/bias
F
Initializer_10/zerosConst*
dtype0*
valueB*    
ú
	Assign_10AssignDmio_variable/bert/encoder/layer_0/attention/self/value/bias/gradientInitializer_10/zeros*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_0/attention/self/value/bias/gradient*
validate_shape(
É
0bert/encoder/layer_0/attention/self/value/MatMulMatMulbert/encoder/Reshape_1Fmio_variable/bert/encoder/layer_0/attention/self/value/kernel/variable*
transpose_a( *
transpose_b( *
T0
Ô
1bert/encoder/layer_0/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_0/attention/self/value/MatMulDmio_variable/bert/encoder/layer_0/attention/self/value/bias/variable*
T0*
data_formatNHWC
]
3bert/encoder/layer_0/attention/self/Reshape/shape/1Const*
value	B :*
dtype0
]
3bert/encoder/layer_0/attention/self/Reshape/shape/2Const*
value	B :*
dtype0
]
3bert/encoder/layer_0/attention/self/Reshape/shape/3Const*
value	B : *
dtype0

1bert/encoder/layer_0/attention/self/Reshape/shapePackbert/encoder/strided_slice_23bert/encoder/layer_0/attention/self/Reshape/shape/13bert/encoder/layer_0/attention/self/Reshape/shape/23bert/encoder/layer_0/attention/self/Reshape/shape/3*
N*
T0*

axis 
ŗ
+bert/encoder/layer_0/attention/self/ReshapeReshape1bert/encoder/layer_0/attention/self/query/BiasAdd1bert/encoder/layer_0/attention/self/Reshape/shape*
T0*
Tshape0
o
2bert/encoder/layer_0/attention/self/transpose/permConst*%
valueB"             *
dtype0
ą
-bert/encoder/layer_0/attention/self/transpose	Transpose+bert/encoder/layer_0/attention/self/Reshape2bert/encoder/layer_0/attention/self/transpose/perm*
Tperm0*
T0
_
5bert/encoder/layer_0/attention/self/Reshape_1/shape/1Const*
value	B :*
dtype0
_
5bert/encoder/layer_0/attention/self/Reshape_1/shape/2Const*
dtype0*
value	B :
_
5bert/encoder/layer_0/attention/self/Reshape_1/shape/3Const*
value	B : *
dtype0

3bert/encoder/layer_0/attention/self/Reshape_1/shapePackbert/encoder/strided_slice_25bert/encoder/layer_0/attention/self/Reshape_1/shape/15bert/encoder/layer_0/attention/self/Reshape_1/shape/25bert/encoder/layer_0/attention/self/Reshape_1/shape/3*
T0*

axis *
N
ĩ
-bert/encoder/layer_0/attention/self/Reshape_1Reshape/bert/encoder/layer_0/attention/self/key/BiasAdd3bert/encoder/layer_0/attention/self/Reshape_1/shape*
T0*
Tshape0
q
4bert/encoder/layer_0/attention/self/transpose_1/permConst*%
valueB"             *
dtype0
ˇ
/bert/encoder/layer_0/attention/self/transpose_1	Transpose-bert/encoder/layer_0/attention/self/Reshape_14bert/encoder/layer_0/attention/self/transpose_1/perm*
Tperm0*
T0
ŧ
*bert/encoder/layer_0/attention/self/MatMulBatchMatMul-bert/encoder/layer_0/attention/self/transpose/bert/encoder/layer_0/attention/self/transpose_1*
adj_y(*
T0*
adj_x( 
V
)bert/encoder/layer_0/attention/self/Mul/yConst*
valueB
 *ķ5>*
dtype0

'bert/encoder/layer_0/attention/self/MulMul*bert/encoder/layer_0/attention/self/MatMul)bert/encoder/layer_0/attention/self/Mul/y*
T0
`
2bert/encoder/layer_0/attention/self/ExpandDims/dimConst*
valueB:*
dtype0

.bert/encoder/layer_0/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_0/attention/self/ExpandDims/dim*
T0*

Tdim0
V
)bert/encoder/layer_0/attention/self/sub/xConst*
dtype0*
valueB
 *  ?

'bert/encoder/layer_0/attention/self/subSub)bert/encoder/layer_0/attention/self/sub/x.bert/encoder/layer_0/attention/self/ExpandDims*
T0
X
+bert/encoder/layer_0/attention/self/mul_1/yConst*
valueB
 * @Æ*
dtype0

)bert/encoder/layer_0/attention/self/mul_1Mul'bert/encoder/layer_0/attention/self/sub+bert/encoder/layer_0/attention/self/mul_1/y*
T0

'bert/encoder/layer_0/attention/self/addAdd'bert/encoder/layer_0/attention/self/Mul)bert/encoder/layer_0/attention/self/mul_1*
T0
h
+bert/encoder/layer_0/attention/self/SoftmaxSoftmax'bert/encoder/layer_0/attention/self/add*
T0
b
5bert/encoder/layer_0/attention/self/dropout/keep_probConst*
valueB
 *fff?*
dtype0

1bert/encoder/layer_0/attention/self/dropout/ShapeShape+bert/encoder/layer_0/attention/self/Softmax*
T0*
out_type0
k
>bert/encoder/layer_0/attention/self/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
k
>bert/encoder/layer_0/attention/self/dropout/random_uniform/maxConst*
dtype0*
valueB
 *  ?
ģ
Hbert/encoder/layer_0/attention/self/dropout/random_uniform/RandomUniformRandomUniform1bert/encoder/layer_0/attention/self/dropout/Shape*
T0*
dtype0*
seed2 *

seed 
Î
>bert/encoder/layer_0/attention/self/dropout/random_uniform/subSub>bert/encoder/layer_0/attention/self/dropout/random_uniform/max>bert/encoder/layer_0/attention/self/dropout/random_uniform/min*
T0
Ø
>bert/encoder/layer_0/attention/self/dropout/random_uniform/mulMulHbert/encoder/layer_0/attention/self/dropout/random_uniform/RandomUniform>bert/encoder/layer_0/attention/self/dropout/random_uniform/sub*
T0
Ę
:bert/encoder/layer_0/attention/self/dropout/random_uniformAdd>bert/encoder/layer_0/attention/self/dropout/random_uniform/mul>bert/encoder/layer_0/attention/self/dropout/random_uniform/min*
T0
˛
/bert/encoder/layer_0/attention/self/dropout/addAdd5bert/encoder/layer_0/attention/self/dropout/keep_prob:bert/encoder/layer_0/attention/self/dropout/random_uniform*
T0
t
1bert/encoder/layer_0/attention/self/dropout/FloorFloor/bert/encoder/layer_0/attention/self/dropout/add*
T0
§
/bert/encoder/layer_0/attention/self/dropout/divRealDiv+bert/encoder/layer_0/attention/self/Softmax5bert/encoder/layer_0/attention/self/dropout/keep_prob*
T0
Ŗ
/bert/encoder/layer_0/attention/self/dropout/mulMul/bert/encoder/layer_0/attention/self/dropout/div1bert/encoder/layer_0/attention/self/dropout/Floor*
T0
_
5bert/encoder/layer_0/attention/self/Reshape_2/shape/1Const*
value	B :*
dtype0
_
5bert/encoder/layer_0/attention/self/Reshape_2/shape/2Const*
dtype0*
value	B :
_
5bert/encoder/layer_0/attention/self/Reshape_2/shape/3Const*
dtype0*
value	B : 

3bert/encoder/layer_0/attention/self/Reshape_2/shapePackbert/encoder/strided_slice_25bert/encoder/layer_0/attention/self/Reshape_2/shape/15bert/encoder/layer_0/attention/self/Reshape_2/shape/25bert/encoder/layer_0/attention/self/Reshape_2/shape/3*

axis *
N*
T0
ˇ
-bert/encoder/layer_0/attention/self/Reshape_2Reshape1bert/encoder/layer_0/attention/self/value/BiasAdd3bert/encoder/layer_0/attention/self/Reshape_2/shape*
T0*
Tshape0
q
4bert/encoder/layer_0/attention/self/transpose_2/permConst*
dtype0*%
valueB"             
ˇ
/bert/encoder/layer_0/attention/self/transpose_2	Transpose-bert/encoder/layer_0/attention/self/Reshape_24bert/encoder/layer_0/attention/self/transpose_2/perm*
Tperm0*
T0
Ā
,bert/encoder/layer_0/attention/self/MatMul_1BatchMatMul/bert/encoder/layer_0/attention/self/dropout/mul/bert/encoder/layer_0/attention/self/transpose_2*
adj_x( *
adj_y( *
T0
q
4bert/encoder/layer_0/attention/self/transpose_3/permConst*
dtype0*%
valueB"             
ļ
/bert/encoder/layer_0/attention/self/transpose_3	Transpose,bert/encoder/layer_0/attention/self/MatMul_14bert/encoder/layer_0/attention/self/transpose_3/perm*
Tperm0*
T0
U
+bert/encoder/layer_0/attention/self/mul_2/yConst*
value	B :*
dtype0

)bert/encoder/layer_0/attention/self/mul_2Mulbert/encoder/strided_slice_2+bert/encoder/layer_0/attention/self/mul_2/y*
T0
`
5bert/encoder/layer_0/attention/self/Reshape_3/shape/1Const*
value
B :*
dtype0
ģ
3bert/encoder/layer_0/attention/self/Reshape_3/shapePack)bert/encoder/layer_0/attention/self/mul_25bert/encoder/layer_0/attention/self/Reshape_3/shape/1*
T0*

axis *
N
ĩ
-bert/encoder/layer_0/attention/self/Reshape_3Reshape/bert/encoder/layer_0/attention/self/transpose_33bert/encoder/layer_0/attention/self/Reshape_3/shape*
T0*
Tshape0
Ö
Hmio_variable/bert/encoder/layer_0/attention/output/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42bert/encoder/layer_0/attention/output/dense/kernel*
shape:

Ö
Hmio_variable/bert/encoder/layer_0/attention/output/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42bert/encoder/layer_0/attention/output/dense/kernel*
shape:

Z
%Initializer_11/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_11/truncated_normal/meanConst*
dtype0*
valueB
 *    
S
&Initializer_11/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_11/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_11/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

#Initializer_11/truncated_normal/mulMul/Initializer_11/truncated_normal/TruncatedNormal&Initializer_11/truncated_normal/stddev*
T0
z
Initializer_11/truncated_normalAdd#Initializer_11/truncated_normal/mul$Initializer_11/truncated_normal/mean*
T0

	Assign_11AssignHmio_variable/bert/encoder/layer_0/attention/output/dense/kernel/gradientInitializer_11/truncated_normal*
use_locking(*
T0*[
_classQ
OMloc:@mio_variable/bert/encoder/layer_0/attention/output/dense/kernel/gradient*
validate_shape(
Í
Fmio_variable/bert/encoder/layer_0/attention/output/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*?
	container20bert/encoder/layer_0/attention/output/dense/bias
Í
Fmio_variable/bert/encoder/layer_0/attention/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*?
	container20bert/encoder/layer_0/attention/output/dense/bias
F
Initializer_12/zerosConst*
valueB*    *
dtype0
ū
	Assign_12AssignFmio_variable/bert/encoder/layer_0/attention/output/dense/bias/gradientInitializer_12/zeros*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_0/attention/output/dense/bias/gradient*
validate_shape(
ä
2bert/encoder/layer_0/attention/output/dense/MatMulMatMul-bert/encoder/layer_0/attention/self/Reshape_3Hmio_variable/bert/encoder/layer_0/attention/output/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
Ú
3bert/encoder/layer_0/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_0/attention/output/dense/MatMulFmio_variable/bert/encoder/layer_0/attention/output/dense/bias/variable*
T0*
data_formatNHWC
d
7bert/encoder/layer_0/attention/output/dropout/keep_probConst*
valueB
 *fff?*
dtype0

3bert/encoder/layer_0/attention/output/dropout/ShapeShape3bert/encoder/layer_0/attention/output/dense/BiasAdd*
T0*
out_type0
m
@bert/encoder/layer_0/attention/output/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
m
@bert/encoder/layer_0/attention/output/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0
ŋ
Jbert/encoder/layer_0/attention/output/dropout/random_uniform/RandomUniformRandomUniform3bert/encoder/layer_0/attention/output/dropout/Shape*

seed *
T0*
dtype0*
seed2 
Ô
@bert/encoder/layer_0/attention/output/dropout/random_uniform/subSub@bert/encoder/layer_0/attention/output/dropout/random_uniform/max@bert/encoder/layer_0/attention/output/dropout/random_uniform/min*
T0
Ū
@bert/encoder/layer_0/attention/output/dropout/random_uniform/mulMulJbert/encoder/layer_0/attention/output/dropout/random_uniform/RandomUniform@bert/encoder/layer_0/attention/output/dropout/random_uniform/sub*
T0
Đ
<bert/encoder/layer_0/attention/output/dropout/random_uniformAdd@bert/encoder/layer_0/attention/output/dropout/random_uniform/mul@bert/encoder/layer_0/attention/output/dropout/random_uniform/min*
T0
¸
1bert/encoder/layer_0/attention/output/dropout/addAdd7bert/encoder/layer_0/attention/output/dropout/keep_prob<bert/encoder/layer_0/attention/output/dropout/random_uniform*
T0
x
3bert/encoder/layer_0/attention/output/dropout/FloorFloor1bert/encoder/layer_0/attention/output/dropout/add*
T0
ŗ
1bert/encoder/layer_0/attention/output/dropout/divRealDiv3bert/encoder/layer_0/attention/output/dense/BiasAdd7bert/encoder/layer_0/attention/output/dropout/keep_prob*
T0
Š
1bert/encoder/layer_0/attention/output/dropout/mulMul1bert/encoder/layer_0/attention/output/dropout/div3bert/encoder/layer_0/attention/output/dropout/Floor*
T0

)bert/encoder/layer_0/attention/output/addAdd1bert/encoder/layer_0/attention/output/dropout/mulbert/encoder/Reshape_1*
T0
Õ
Jmio_variable/bert/encoder/layer_0/attention/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64bert/encoder/layer_0/attention/output/LayerNorm/beta*
shape:
Õ
Jmio_variable/bert/encoder/layer_0/attention/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*C
	container64bert/encoder/layer_0/attention/output/LayerNorm/beta
F
Initializer_13/zerosConst*
valueB*    *
dtype0

	Assign_13AssignJmio_variable/bert/encoder/layer_0/attention/output/LayerNorm/beta/gradientInitializer_13/zeros*
use_locking(*
T0*]
_classS
QOloc:@mio_variable/bert/encoder/layer_0/attention/output/LayerNorm/beta/gradient*
validate_shape(
×
Kmio_variable/bert/encoder/layer_0/attention/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*D
	container75bert/encoder/layer_0/attention/output/LayerNorm/gamma*
shape:
×
Kmio_variable/bert/encoder/layer_0/attention/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*D
	container75bert/encoder/layer_0/attention/output/LayerNorm/gamma*
shape:
E
Initializer_14/onesConst*
valueB*  ?*
dtype0

	Assign_14AssignKmio_variable/bert/encoder/layer_0/attention/output/LayerNorm/gamma/gradientInitializer_14/ones*^
_classT
RPloc:@mio_variable/bert/encoder/layer_0/attention/output/LayerNorm/gamma/gradient*
validate_shape(*
use_locking(*
T0
|
Nbert/encoder/layer_0/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0
å
<bert/encoder/layer_0/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_0/attention/output/addNbert/encoder/layer_0/attention/output/LayerNorm/moments/mean/reduction_indices*
T0*

Tidx0*
	keep_dims(

Dbert/encoder/layer_0/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_0/attention/output/LayerNorm/moments/mean*
T0
Ø
Ibert/encoder/layer_0/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_0/attention/output/addDbert/encoder/layer_0/attention/output/LayerNorm/moments/StopGradient*
T0

Rbert/encoder/layer_0/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
dtype0*
valueB:

@bert/encoder/layer_0/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_0/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_0/attention/output/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0
l
?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ėŧ+*
dtype0
Đ
=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_0/attention/output/LayerNorm/moments/variance?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add/y*
T0

?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add*
T0
Û
=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/RsqrtKmio_variable/bert/encoder/layer_0/attention/output/LayerNorm/gamma/variable*
T0
š
?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_0/attention/output/add=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul*
T0
Ė
?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_0/attention/output/LayerNorm/moments/mean=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul*
T0
Ú
=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/subSubJmio_variable/bert/encoder/layer_0/attention/output/LayerNorm/beta/variable?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_2*
T0
Ī
?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/sub*
T0
Î
Dmio_variable/bert/encoder/layer_0/intermediate/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_0/intermediate/dense/kernel*
shape:

Î
Dmio_variable/bert/encoder/layer_0/intermediate/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_0/intermediate/dense/kernel*
shape:

Z
%Initializer_15/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_15/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_15/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_15/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_15/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0

#Initializer_15/truncated_normal/mulMul/Initializer_15/truncated_normal/TruncatedNormal&Initializer_15/truncated_normal/stddev*
T0
z
Initializer_15/truncated_normalAdd#Initializer_15/truncated_normal/mul$Initializer_15/truncated_normal/mean*
T0

	Assign_15AssignDmio_variable/bert/encoder/layer_0/intermediate/dense/kernel/gradientInitializer_15/truncated_normal*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_0/intermediate/dense/kernel/gradient*
validate_shape(
Å
Bmio_variable/bert/encoder/layer_0/intermediate/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_0/intermediate/dense/bias*
shape:
Å
Bmio_variable/bert/encoder/layer_0/intermediate/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_0/intermediate/dense/bias*
shape:
S
$Initializer_16/zeros/shape_as_tensorConst*
valueB:*
dtype0
G
Initializer_16/zeros/ConstConst*
valueB
 *    *
dtype0
y
Initializer_16/zerosFill$Initializer_16/zeros/shape_as_tensorInitializer_16/zeros/Const*
T0*

index_type0
ö
	Assign_16AssignBmio_variable/bert/encoder/layer_0/intermediate/dense/bias/gradientInitializer_16/zeros*
use_locking(*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_0/intermediate/dense/bias/gradient*
validate_shape(
î
.bert/encoder/layer_0/intermediate/dense/MatMulMatMul?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1Dmio_variable/bert/encoder/layer_0/intermediate/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Î
/bert/encoder/layer_0/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_0/intermediate/dense/MatMulBmio_variable/bert/encoder/layer_0/intermediate/dense/bias/variable*
T0*
data_formatNHWC
Z
-bert/encoder/layer_0/intermediate/dense/Pow/yConst*
valueB
 *  @@*
dtype0

+bert/encoder/layer_0/intermediate/dense/PowPow/bert/encoder/layer_0/intermediate/dense/BiasAdd-bert/encoder/layer_0/intermediate/dense/Pow/y*
T0
Z
-bert/encoder/layer_0/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0

+bert/encoder/layer_0/intermediate/dense/mulMul-bert/encoder/layer_0/intermediate/dense/mul/x+bert/encoder/layer_0/intermediate/dense/Pow*
T0

+bert/encoder/layer_0/intermediate/dense/addAdd/bert/encoder/layer_0/intermediate/dense/BiasAdd+bert/encoder/layer_0/intermediate/dense/mul*
T0
\
/bert/encoder/layer_0/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0

-bert/encoder/layer_0/intermediate/dense/mul_1Mul/bert/encoder/layer_0/intermediate/dense/mul_1/x+bert/encoder/layer_0/intermediate/dense/add*
T0
l
,bert/encoder/layer_0/intermediate/dense/TanhTanh-bert/encoder/layer_0/intermediate/dense/mul_1*
T0
\
/bert/encoder/layer_0/intermediate/dense/add_1/xConst*
valueB
 *  ?*
dtype0

-bert/encoder/layer_0/intermediate/dense/add_1Add/bert/encoder/layer_0/intermediate/dense/add_1/x,bert/encoder/layer_0/intermediate/dense/Tanh*
T0
\
/bert/encoder/layer_0/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0

-bert/encoder/layer_0/intermediate/dense/mul_2Mul/bert/encoder/layer_0/intermediate/dense/mul_2/x-bert/encoder/layer_0/intermediate/dense/add_1*
T0

-bert/encoder/layer_0/intermediate/dense/mul_3Mul/bert/encoder/layer_0/intermediate/dense/BiasAdd-bert/encoder/layer_0/intermediate/dense/mul_2*
T0
Â
>mio_variable/bert/encoder/layer_0/output/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(bert/encoder/layer_0/output/dense/kernel*
shape:

Â
>mio_variable/bert/encoder/layer_0/output/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(bert/encoder/layer_0/output/dense/kernel*
shape:

Z
%Initializer_17/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_17/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_17/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_17/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_17/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0

#Initializer_17/truncated_normal/mulMul/Initializer_17/truncated_normal/TruncatedNormal&Initializer_17/truncated_normal/stddev*
T0
z
Initializer_17/truncated_normalAdd#Initializer_17/truncated_normal/mul$Initializer_17/truncated_normal/mean*
T0
ų
	Assign_17Assign>mio_variable/bert/encoder/layer_0/output/dense/kernel/gradientInitializer_17/truncated_normal*
use_locking(*
T0*Q
_classG
ECloc:@mio_variable/bert/encoder/layer_0/output/dense/kernel/gradient*
validate_shape(
š
<mio_variable/bert/encoder/layer_0/output/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&bert/encoder/layer_0/output/dense/bias*
shape:
š
<mio_variable/bert/encoder/layer_0/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&bert/encoder/layer_0/output/dense/bias*
shape:
F
Initializer_18/zerosConst*
valueB*    *
dtype0
ę
	Assign_18Assign<mio_variable/bert/encoder/layer_0/output/dense/bias/gradientInitializer_18/zeros*
use_locking(*
T0*O
_classE
CAloc:@mio_variable/bert/encoder/layer_0/output/dense/bias/gradient*
validate_shape(
Đ
(bert/encoder/layer_0/output/dense/MatMulMatMul-bert/encoder/layer_0/intermediate/dense/mul_3>mio_variable/bert/encoder/layer_0/output/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
ŧ
)bert/encoder/layer_0/output/dense/BiasAddBiasAdd(bert/encoder/layer_0/output/dense/MatMul<mio_variable/bert/encoder/layer_0/output/dense/bias/variable*
T0*
data_formatNHWC
Z
-bert/encoder/layer_0/output/dropout/keep_probConst*
dtype0*
valueB
 *fff?
v
)bert/encoder/layer_0/output/dropout/ShapeShape)bert/encoder/layer_0/output/dense/BiasAdd*
T0*
out_type0
c
6bert/encoder/layer_0/output/dropout/random_uniform/minConst*
dtype0*
valueB
 *    
c
6bert/encoder/layer_0/output/dropout/random_uniform/maxConst*
dtype0*
valueB
 *  ?
Ģ
@bert/encoder/layer_0/output/dropout/random_uniform/RandomUniformRandomUniform)bert/encoder/layer_0/output/dropout/Shape*

seed *
T0*
dtype0*
seed2 
ļ
6bert/encoder/layer_0/output/dropout/random_uniform/subSub6bert/encoder/layer_0/output/dropout/random_uniform/max6bert/encoder/layer_0/output/dropout/random_uniform/min*
T0
Ā
6bert/encoder/layer_0/output/dropout/random_uniform/mulMul@bert/encoder/layer_0/output/dropout/random_uniform/RandomUniform6bert/encoder/layer_0/output/dropout/random_uniform/sub*
T0
˛
2bert/encoder/layer_0/output/dropout/random_uniformAdd6bert/encoder/layer_0/output/dropout/random_uniform/mul6bert/encoder/layer_0/output/dropout/random_uniform/min*
T0

'bert/encoder/layer_0/output/dropout/addAdd-bert/encoder/layer_0/output/dropout/keep_prob2bert/encoder/layer_0/output/dropout/random_uniform*
T0
d
)bert/encoder/layer_0/output/dropout/FloorFloor'bert/encoder/layer_0/output/dropout/add*
T0

'bert/encoder/layer_0/output/dropout/divRealDiv)bert/encoder/layer_0/output/dense/BiasAdd-bert/encoder/layer_0/output/dropout/keep_prob*
T0

'bert/encoder/layer_0/output/dropout/mulMul'bert/encoder/layer_0/output/dropout/div)bert/encoder/layer_0/output/dropout/Floor*
T0

bert/encoder/layer_0/output/addAdd'bert/encoder/layer_0/output/dropout/mul?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1*
T0
Á
@mio_variable/bert/encoder/layer_0/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*bert/encoder/layer_0/output/LayerNorm/beta*
shape:
Á
@mio_variable/bert/encoder/layer_0/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*9
	container,*bert/encoder/layer_0/output/LayerNorm/beta
F
Initializer_19/zerosConst*
dtype0*
valueB*    
ō
	Assign_19Assign@mio_variable/bert/encoder/layer_0/output/LayerNorm/beta/gradientInitializer_19/zeros*
T0*S
_classI
GEloc:@mio_variable/bert/encoder/layer_0/output/LayerNorm/beta/gradient*
validate_shape(*
use_locking(
Ã
Amio_variable/bert/encoder/layer_0/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+bert/encoder/layer_0/output/LayerNorm/gamma*
shape:
Ã
Amio_variable/bert/encoder/layer_0/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+bert/encoder/layer_0/output/LayerNorm/gamma*
shape:
E
Initializer_20/onesConst*
valueB*  ?*
dtype0
ķ
	Assign_20AssignAmio_variable/bert/encoder/layer_0/output/LayerNorm/gamma/gradientInitializer_20/ones*
use_locking(*
T0*T
_classJ
HFloc:@mio_variable/bert/encoder/layer_0/output/LayerNorm/gamma/gradient*
validate_shape(
r
Dbert/encoder/layer_0/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0
Į
2bert/encoder/layer_0/output/LayerNorm/moments/meanMeanbert/encoder/layer_0/output/addDbert/encoder/layer_0/output/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0

:bert/encoder/layer_0/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_0/output/LayerNorm/moments/mean*
T0
ē
?bert/encoder/layer_0/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_0/output/add:bert/encoder/layer_0/output/LayerNorm/moments/StopGradient*
T0
v
Hbert/encoder/layer_0/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0
ī
6bert/encoder/layer_0/output/LayerNorm/moments/varianceMean?bert/encoder/layer_0/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_0/output/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0
b
5bert/encoder/layer_0/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ėŧ+*
dtype0
˛
3bert/encoder/layer_0/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_0/output/LayerNorm/moments/variance5bert/encoder/layer_0/output/LayerNorm/batchnorm/add/y*
T0
|
5bert/encoder/layer_0/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_0/output/LayerNorm/batchnorm/add*
T0
Ŋ
3bert/encoder/layer_0/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_0/output/LayerNorm/batchnorm/RsqrtAmio_variable/bert/encoder/layer_0/output/LayerNorm/gamma/variable*
T0

5bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_0/output/add3bert/encoder/layer_0/output/LayerNorm/batchnorm/mul*
T0
Ž
5bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_0/output/LayerNorm/moments/mean3bert/encoder/layer_0/output/LayerNorm/batchnorm/mul*
T0
ŧ
3bert/encoder/layer_0/output/LayerNorm/batchnorm/subSub@mio_variable/bert/encoder/layer_0/output/LayerNorm/beta/variable5bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_2*
T0
ą
5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_0/output/LayerNorm/batchnorm/sub*
T0

)bert/encoder/layer_1/attention/self/ShapeShape5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
e
7bert/encoder/layer_1/attention/self/strided_slice/stackConst*
valueB: *
dtype0
g
9bert/encoder/layer_1/attention/self/strided_slice/stack_1Const*
valueB:*
dtype0
g
9bert/encoder/layer_1/attention/self/strided_slice/stack_2Const*
valueB:*
dtype0

1bert/encoder/layer_1/attention/self/strided_sliceStridedSlice)bert/encoder/layer_1/attention/self/Shape7bert/encoder/layer_1/attention/self/strided_slice/stack9bert/encoder/layer_1/attention/self/strided_slice/stack_19bert/encoder/layer_1/attention/self/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0

+bert/encoder/layer_1/attention/self/Shape_1Shape5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
g
9bert/encoder/layer_1/attention/self/strided_slice_1/stackConst*
valueB: *
dtype0
i
;bert/encoder/layer_1/attention/self/strided_slice_1/stack_1Const*
valueB:*
dtype0
i
;bert/encoder/layer_1/attention/self/strided_slice_1/stack_2Const*
valueB:*
dtype0

3bert/encoder/layer_1/attention/self/strided_slice_1StridedSlice+bert/encoder/layer_1/attention/self/Shape_19bert/encoder/layer_1/attention/self/strided_slice_1/stack;bert/encoder/layer_1/attention/self/strided_slice_1/stack_1;bert/encoder/layer_1/attention/self/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
Ō
Fmio_variable/bert/encoder/layer_1/attention/self/query/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_1/attention/self/query/kernel*
shape:

Ō
Fmio_variable/bert/encoder/layer_1/attention/self/query/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_1/attention/self/query/kernel*
shape:

Z
%Initializer_21/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_21/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_21/truncated_normal/stddevConst*
dtype0*
valueB
 *
×Ŗ<

/Initializer_21/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_21/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

#Initializer_21/truncated_normal/mulMul/Initializer_21/truncated_normal/TruncatedNormal&Initializer_21/truncated_normal/stddev*
T0
z
Initializer_21/truncated_normalAdd#Initializer_21/truncated_normal/mul$Initializer_21/truncated_normal/mean*
T0

	Assign_21AssignFmio_variable/bert/encoder/layer_1/attention/self/query/kernel/gradientInitializer_21/truncated_normal*
validate_shape(*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_1/attention/self/query/kernel/gradient
É
Dmio_variable/bert/encoder/layer_1/attention/self/query/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_1/attention/self/query/bias*
shape:
É
Dmio_variable/bert/encoder/layer_1/attention/self/query/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*=
	container0.bert/encoder/layer_1/attention/self/query/bias
F
Initializer_22/zerosConst*
valueB*    *
dtype0
ú
	Assign_22AssignDmio_variable/bert/encoder/layer_1/attention/self/query/bias/gradientInitializer_22/zeros*
validate_shape(*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_1/attention/self/query/bias/gradient
č
0bert/encoder/layer_1/attention/self/query/MatMulMatMul5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1Fmio_variable/bert/encoder/layer_1/attention/self/query/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Ô
1bert/encoder/layer_1/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_1/attention/self/query/MatMulDmio_variable/bert/encoder/layer_1/attention/self/query/bias/variable*
data_formatNHWC*
T0
Î
Dmio_variable/bert/encoder/layer_1/attention/self/key/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_1/attention/self/key/kernel*
shape:

Î
Dmio_variable/bert/encoder/layer_1/attention/self/key/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_1/attention/self/key/kernel*
shape:

Z
%Initializer_23/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_23/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_23/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_23/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_23/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

#Initializer_23/truncated_normal/mulMul/Initializer_23/truncated_normal/TruncatedNormal&Initializer_23/truncated_normal/stddev*
T0
z
Initializer_23/truncated_normalAdd#Initializer_23/truncated_normal/mul$Initializer_23/truncated_normal/mean*
T0

	Assign_23AssignDmio_variable/bert/encoder/layer_1/attention/self/key/kernel/gradientInitializer_23/truncated_normal*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_1/attention/self/key/kernel/gradient*
validate_shape(
Å
Bmio_variable/bert/encoder/layer_1/attention/self/key/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_1/attention/self/key/bias*
shape:
Å
Bmio_variable/bert/encoder/layer_1/attention/self/key/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*;
	container.,bert/encoder/layer_1/attention/self/key/bias
F
Initializer_24/zerosConst*
valueB*    *
dtype0
ö
	Assign_24AssignBmio_variable/bert/encoder/layer_1/attention/self/key/bias/gradientInitializer_24/zeros*
use_locking(*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_1/attention/self/key/bias/gradient*
validate_shape(
ä
.bert/encoder/layer_1/attention/self/key/MatMulMatMul5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1Dmio_variable/bert/encoder/layer_1/attention/self/key/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Î
/bert/encoder/layer_1/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_1/attention/self/key/MatMulBmio_variable/bert/encoder/layer_1/attention/self/key/bias/variable*
data_formatNHWC*
T0
Ō
Fmio_variable/bert/encoder/layer_1/attention/self/value/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_1/attention/self/value/kernel*
shape:

Ō
Fmio_variable/bert/encoder/layer_1/attention/self/value/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_1/attention/self/value/kernel*
shape:

Z
%Initializer_25/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_25/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_25/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_25/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_25/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 

#Initializer_25/truncated_normal/mulMul/Initializer_25/truncated_normal/TruncatedNormal&Initializer_25/truncated_normal/stddev*
T0
z
Initializer_25/truncated_normalAdd#Initializer_25/truncated_normal/mul$Initializer_25/truncated_normal/mean*
T0

	Assign_25AssignFmio_variable/bert/encoder/layer_1/attention/self/value/kernel/gradientInitializer_25/truncated_normal*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_1/attention/self/value/kernel/gradient*
validate_shape(
É
Dmio_variable/bert/encoder/layer_1/attention/self/value/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_1/attention/self/value/bias*
shape:
É
Dmio_variable/bert/encoder/layer_1/attention/self/value/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_1/attention/self/value/bias*
shape:
F
Initializer_26/zerosConst*
valueB*    *
dtype0
ú
	Assign_26AssignDmio_variable/bert/encoder/layer_1/attention/self/value/bias/gradientInitializer_26/zeros*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_1/attention/self/value/bias/gradient*
validate_shape(
č
0bert/encoder/layer_1/attention/self/value/MatMulMatMul5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1Fmio_variable/bert/encoder/layer_1/attention/self/value/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Ô
1bert/encoder/layer_1/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_1/attention/self/value/MatMulDmio_variable/bert/encoder/layer_1/attention/self/value/bias/variable*
T0*
data_formatNHWC
]
3bert/encoder/layer_1/attention/self/Reshape/shape/1Const*
value	B :*
dtype0
]
3bert/encoder/layer_1/attention/self/Reshape/shape/2Const*
value	B :*
dtype0
]
3bert/encoder/layer_1/attention/self/Reshape/shape/3Const*
dtype0*
value	B : 

1bert/encoder/layer_1/attention/self/Reshape/shapePackbert/encoder/strided_slice_23bert/encoder/layer_1/attention/self/Reshape/shape/13bert/encoder/layer_1/attention/self/Reshape/shape/23bert/encoder/layer_1/attention/self/Reshape/shape/3*
T0*

axis *
N
ŗ
+bert/encoder/layer_1/attention/self/ReshapeReshape1bert/encoder/layer_1/attention/self/query/BiasAdd1bert/encoder/layer_1/attention/self/Reshape/shape*
T0*
Tshape0
o
2bert/encoder/layer_1/attention/self/transpose/permConst*%
valueB"             *
dtype0
ą
-bert/encoder/layer_1/attention/self/transpose	Transpose+bert/encoder/layer_1/attention/self/Reshape2bert/encoder/layer_1/attention/self/transpose/perm*
T0*
Tperm0
_
5bert/encoder/layer_1/attention/self/Reshape_1/shape/1Const*
dtype0*
value	B :
_
5bert/encoder/layer_1/attention/self/Reshape_1/shape/2Const*
value	B :*
dtype0
_
5bert/encoder/layer_1/attention/self/Reshape_1/shape/3Const*
dtype0*
value	B : 

3bert/encoder/layer_1/attention/self/Reshape_1/shapePackbert/encoder/strided_slice_25bert/encoder/layer_1/attention/self/Reshape_1/shape/15bert/encoder/layer_1/attention/self/Reshape_1/shape/25bert/encoder/layer_1/attention/self/Reshape_1/shape/3*
T0*

axis *
N
ĩ
-bert/encoder/layer_1/attention/self/Reshape_1Reshape/bert/encoder/layer_1/attention/self/key/BiasAdd3bert/encoder/layer_1/attention/self/Reshape_1/shape*
T0*
Tshape0
q
4bert/encoder/layer_1/attention/self/transpose_1/permConst*%
valueB"             *
dtype0
ˇ
/bert/encoder/layer_1/attention/self/transpose_1	Transpose-bert/encoder/layer_1/attention/self/Reshape_14bert/encoder/layer_1/attention/self/transpose_1/perm*
Tperm0*
T0
ŧ
*bert/encoder/layer_1/attention/self/MatMulBatchMatMul-bert/encoder/layer_1/attention/self/transpose/bert/encoder/layer_1/attention/self/transpose_1*
adj_x( *
adj_y(*
T0
V
)bert/encoder/layer_1/attention/self/Mul/yConst*
valueB
 *ķ5>*
dtype0

'bert/encoder/layer_1/attention/self/MulMul*bert/encoder/layer_1/attention/self/MatMul)bert/encoder/layer_1/attention/self/Mul/y*
T0
`
2bert/encoder/layer_1/attention/self/ExpandDims/dimConst*
valueB:*
dtype0

.bert/encoder/layer_1/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_1/attention/self/ExpandDims/dim*

Tdim0*
T0
V
)bert/encoder/layer_1/attention/self/sub/xConst*
valueB
 *  ?*
dtype0

'bert/encoder/layer_1/attention/self/subSub)bert/encoder/layer_1/attention/self/sub/x.bert/encoder/layer_1/attention/self/ExpandDims*
T0
X
+bert/encoder/layer_1/attention/self/mul_1/yConst*
valueB
 * @Æ*
dtype0

)bert/encoder/layer_1/attention/self/mul_1Mul'bert/encoder/layer_1/attention/self/sub+bert/encoder/layer_1/attention/self/mul_1/y*
T0

'bert/encoder/layer_1/attention/self/addAdd'bert/encoder/layer_1/attention/self/Mul)bert/encoder/layer_1/attention/self/mul_1*
T0
h
+bert/encoder/layer_1/attention/self/SoftmaxSoftmax'bert/encoder/layer_1/attention/self/add*
T0
b
5bert/encoder/layer_1/attention/self/dropout/keep_probConst*
valueB
 *fff?*
dtype0

1bert/encoder/layer_1/attention/self/dropout/ShapeShape+bert/encoder/layer_1/attention/self/Softmax*
T0*
out_type0
k
>bert/encoder/layer_1/attention/self/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
k
>bert/encoder/layer_1/attention/self/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0
ģ
Hbert/encoder/layer_1/attention/self/dropout/random_uniform/RandomUniformRandomUniform1bert/encoder/layer_1/attention/self/dropout/Shape*
T0*
dtype0*
seed2 *

seed 
Î
>bert/encoder/layer_1/attention/self/dropout/random_uniform/subSub>bert/encoder/layer_1/attention/self/dropout/random_uniform/max>bert/encoder/layer_1/attention/self/dropout/random_uniform/min*
T0
Ø
>bert/encoder/layer_1/attention/self/dropout/random_uniform/mulMulHbert/encoder/layer_1/attention/self/dropout/random_uniform/RandomUniform>bert/encoder/layer_1/attention/self/dropout/random_uniform/sub*
T0
Ę
:bert/encoder/layer_1/attention/self/dropout/random_uniformAdd>bert/encoder/layer_1/attention/self/dropout/random_uniform/mul>bert/encoder/layer_1/attention/self/dropout/random_uniform/min*
T0
˛
/bert/encoder/layer_1/attention/self/dropout/addAdd5bert/encoder/layer_1/attention/self/dropout/keep_prob:bert/encoder/layer_1/attention/self/dropout/random_uniform*
T0
t
1bert/encoder/layer_1/attention/self/dropout/FloorFloor/bert/encoder/layer_1/attention/self/dropout/add*
T0
§
/bert/encoder/layer_1/attention/self/dropout/divRealDiv+bert/encoder/layer_1/attention/self/Softmax5bert/encoder/layer_1/attention/self/dropout/keep_prob*
T0
Ŗ
/bert/encoder/layer_1/attention/self/dropout/mulMul/bert/encoder/layer_1/attention/self/dropout/div1bert/encoder/layer_1/attention/self/dropout/Floor*
T0
_
5bert/encoder/layer_1/attention/self/Reshape_2/shape/1Const*
dtype0*
value	B :
_
5bert/encoder/layer_1/attention/self/Reshape_2/shape/2Const*
value	B :*
dtype0
_
5bert/encoder/layer_1/attention/self/Reshape_2/shape/3Const*
value	B : *
dtype0

3bert/encoder/layer_1/attention/self/Reshape_2/shapePackbert/encoder/strided_slice_25bert/encoder/layer_1/attention/self/Reshape_2/shape/15bert/encoder/layer_1/attention/self/Reshape_2/shape/25bert/encoder/layer_1/attention/self/Reshape_2/shape/3*
T0*

axis *
N
ˇ
-bert/encoder/layer_1/attention/self/Reshape_2Reshape1bert/encoder/layer_1/attention/self/value/BiasAdd3bert/encoder/layer_1/attention/self/Reshape_2/shape*
T0*
Tshape0
q
4bert/encoder/layer_1/attention/self/transpose_2/permConst*%
valueB"             *
dtype0
ˇ
/bert/encoder/layer_1/attention/self/transpose_2	Transpose-bert/encoder/layer_1/attention/self/Reshape_24bert/encoder/layer_1/attention/self/transpose_2/perm*
T0*
Tperm0
Ā
,bert/encoder/layer_1/attention/self/MatMul_1BatchMatMul/bert/encoder/layer_1/attention/self/dropout/mul/bert/encoder/layer_1/attention/self/transpose_2*
T0*
adj_x( *
adj_y( 
q
4bert/encoder/layer_1/attention/self/transpose_3/permConst*%
valueB"             *
dtype0
ļ
/bert/encoder/layer_1/attention/self/transpose_3	Transpose,bert/encoder/layer_1/attention/self/MatMul_14bert/encoder/layer_1/attention/self/transpose_3/perm*
Tperm0*
T0
U
+bert/encoder/layer_1/attention/self/mul_2/yConst*
value	B :*
dtype0

)bert/encoder/layer_1/attention/self/mul_2Mulbert/encoder/strided_slice_2+bert/encoder/layer_1/attention/self/mul_2/y*
T0
`
5bert/encoder/layer_1/attention/self/Reshape_3/shape/1Const*
value
B :*
dtype0
ģ
3bert/encoder/layer_1/attention/self/Reshape_3/shapePack)bert/encoder/layer_1/attention/self/mul_25bert/encoder/layer_1/attention/self/Reshape_3/shape/1*
T0*

axis *
N
ĩ
-bert/encoder/layer_1/attention/self/Reshape_3Reshape/bert/encoder/layer_1/attention/self/transpose_33bert/encoder/layer_1/attention/self/Reshape_3/shape*
Tshape0*
T0
Ö
Hmio_variable/bert/encoder/layer_1/attention/output/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*A
	container42bert/encoder/layer_1/attention/output/dense/kernel
Ö
Hmio_variable/bert/encoder/layer_1/attention/output/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42bert/encoder/layer_1/attention/output/dense/kernel*
shape:

Z
%Initializer_27/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_27/truncated_normal/meanConst*
dtype0*
valueB
 *    
S
&Initializer_27/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_27/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_27/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

#Initializer_27/truncated_normal/mulMul/Initializer_27/truncated_normal/TruncatedNormal&Initializer_27/truncated_normal/stddev*
T0
z
Initializer_27/truncated_normalAdd#Initializer_27/truncated_normal/mul$Initializer_27/truncated_normal/mean*
T0

	Assign_27AssignHmio_variable/bert/encoder/layer_1/attention/output/dense/kernel/gradientInitializer_27/truncated_normal*
use_locking(*
T0*[
_classQ
OMloc:@mio_variable/bert/encoder/layer_1/attention/output/dense/kernel/gradient*
validate_shape(
Í
Fmio_variable/bert/encoder/layer_1/attention/output/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_1/attention/output/dense/bias*
shape:
Í
Fmio_variable/bert/encoder/layer_1/attention/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_1/attention/output/dense/bias*
shape:
F
Initializer_28/zerosConst*
valueB*    *
dtype0
ū
	Assign_28AssignFmio_variable/bert/encoder/layer_1/attention/output/dense/bias/gradientInitializer_28/zeros*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_1/attention/output/dense/bias/gradient*
validate_shape(
ä
2bert/encoder/layer_1/attention/output/dense/MatMulMatMul-bert/encoder/layer_1/attention/self/Reshape_3Hmio_variable/bert/encoder/layer_1/attention/output/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
Ú
3bert/encoder/layer_1/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_1/attention/output/dense/MatMulFmio_variable/bert/encoder/layer_1/attention/output/dense/bias/variable*
T0*
data_formatNHWC
d
7bert/encoder/layer_1/attention/output/dropout/keep_probConst*
dtype0*
valueB
 *fff?

3bert/encoder/layer_1/attention/output/dropout/ShapeShape3bert/encoder/layer_1/attention/output/dense/BiasAdd*
out_type0*
T0
m
@bert/encoder/layer_1/attention/output/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
m
@bert/encoder/layer_1/attention/output/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0
ŋ
Jbert/encoder/layer_1/attention/output/dropout/random_uniform/RandomUniformRandomUniform3bert/encoder/layer_1/attention/output/dropout/Shape*
T0*
dtype0*
seed2 *

seed 
Ô
@bert/encoder/layer_1/attention/output/dropout/random_uniform/subSub@bert/encoder/layer_1/attention/output/dropout/random_uniform/max@bert/encoder/layer_1/attention/output/dropout/random_uniform/min*
T0
Ū
@bert/encoder/layer_1/attention/output/dropout/random_uniform/mulMulJbert/encoder/layer_1/attention/output/dropout/random_uniform/RandomUniform@bert/encoder/layer_1/attention/output/dropout/random_uniform/sub*
T0
Đ
<bert/encoder/layer_1/attention/output/dropout/random_uniformAdd@bert/encoder/layer_1/attention/output/dropout/random_uniform/mul@bert/encoder/layer_1/attention/output/dropout/random_uniform/min*
T0
¸
1bert/encoder/layer_1/attention/output/dropout/addAdd7bert/encoder/layer_1/attention/output/dropout/keep_prob<bert/encoder/layer_1/attention/output/dropout/random_uniform*
T0
x
3bert/encoder/layer_1/attention/output/dropout/FloorFloor1bert/encoder/layer_1/attention/output/dropout/add*
T0
ŗ
1bert/encoder/layer_1/attention/output/dropout/divRealDiv3bert/encoder/layer_1/attention/output/dense/BiasAdd7bert/encoder/layer_1/attention/output/dropout/keep_prob*
T0
Š
1bert/encoder/layer_1/attention/output/dropout/mulMul1bert/encoder/layer_1/attention/output/dropout/div3bert/encoder/layer_1/attention/output/dropout/Floor*
T0
Ŗ
)bert/encoder/layer_1/attention/output/addAdd1bert/encoder/layer_1/attention/output/dropout/mul5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1*
T0
Õ
Jmio_variable/bert/encoder/layer_1/attention/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64bert/encoder/layer_1/attention/output/LayerNorm/beta*
shape:
Õ
Jmio_variable/bert/encoder/layer_1/attention/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*C
	container64bert/encoder/layer_1/attention/output/LayerNorm/beta
F
Initializer_29/zerosConst*
dtype0*
valueB*    

	Assign_29AssignJmio_variable/bert/encoder/layer_1/attention/output/LayerNorm/beta/gradientInitializer_29/zeros*
use_locking(*
T0*]
_classS
QOloc:@mio_variable/bert/encoder/layer_1/attention/output/LayerNorm/beta/gradient*
validate_shape(
×
Kmio_variable/bert/encoder/layer_1/attention/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*D
	container75bert/encoder/layer_1/attention/output/LayerNorm/gamma*
shape:
×
Kmio_variable/bert/encoder/layer_1/attention/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*D
	container75bert/encoder/layer_1/attention/output/LayerNorm/gamma
E
Initializer_30/onesConst*
valueB*  ?*
dtype0

	Assign_30AssignKmio_variable/bert/encoder/layer_1/attention/output/LayerNorm/gamma/gradientInitializer_30/ones*
use_locking(*
T0*^
_classT
RPloc:@mio_variable/bert/encoder/layer_1/attention/output/LayerNorm/gamma/gradient*
validate_shape(
|
Nbert/encoder/layer_1/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
valueB:
å
<bert/encoder/layer_1/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_1/attention/output/addNbert/encoder/layer_1/attention/output/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0

Dbert/encoder/layer_1/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_1/attention/output/LayerNorm/moments/mean*
T0
Ø
Ibert/encoder/layer_1/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_1/attention/output/addDbert/encoder/layer_1/attention/output/LayerNorm/moments/StopGradient*
T0

Rbert/encoder/layer_1/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0

@bert/encoder/layer_1/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_1/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_1/attention/output/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0
l
?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add/yConst*
dtype0*
valueB
 *Ėŧ+
Đ
=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_1/attention/output/LayerNorm/moments/variance?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add/y*
T0

?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add*
T0
Û
=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/RsqrtKmio_variable/bert/encoder/layer_1/attention/output/LayerNorm/gamma/variable*
T0
š
?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_1/attention/output/add=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul*
T0
Ė
?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_1/attention/output/LayerNorm/moments/mean=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul*
T0
Ú
=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/subSubJmio_variable/bert/encoder/layer_1/attention/output/LayerNorm/beta/variable?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_2*
T0
Ī
?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/sub*
T0
Î
Dmio_variable/bert/encoder/layer_1/intermediate/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_1/intermediate/dense/kernel*
shape:

Î
Dmio_variable/bert/encoder/layer_1/intermediate/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_1/intermediate/dense/kernel*
shape:

Z
%Initializer_31/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_31/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_31/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_31/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_31/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

#Initializer_31/truncated_normal/mulMul/Initializer_31/truncated_normal/TruncatedNormal&Initializer_31/truncated_normal/stddev*
T0
z
Initializer_31/truncated_normalAdd#Initializer_31/truncated_normal/mul$Initializer_31/truncated_normal/mean*
T0

	Assign_31AssignDmio_variable/bert/encoder/layer_1/intermediate/dense/kernel/gradientInitializer_31/truncated_normal*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_1/intermediate/dense/kernel/gradient*
validate_shape(
Å
Bmio_variable/bert/encoder/layer_1/intermediate/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_1/intermediate/dense/bias*
shape:
Å
Bmio_variable/bert/encoder/layer_1/intermediate/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_1/intermediate/dense/bias*
shape:
S
$Initializer_32/zeros/shape_as_tensorConst*
valueB:*
dtype0
G
Initializer_32/zeros/ConstConst*
valueB
 *    *
dtype0
y
Initializer_32/zerosFill$Initializer_32/zeros/shape_as_tensorInitializer_32/zeros/Const*
T0*

index_type0
ö
	Assign_32AssignBmio_variable/bert/encoder/layer_1/intermediate/dense/bias/gradientInitializer_32/zeros*
use_locking(*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_1/intermediate/dense/bias/gradient*
validate_shape(
î
.bert/encoder/layer_1/intermediate/dense/MatMulMatMul?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add_1Dmio_variable/bert/encoder/layer_1/intermediate/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Î
/bert/encoder/layer_1/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_1/intermediate/dense/MatMulBmio_variable/bert/encoder/layer_1/intermediate/dense/bias/variable*
T0*
data_formatNHWC
Z
-bert/encoder/layer_1/intermediate/dense/Pow/yConst*
dtype0*
valueB
 *  @@

+bert/encoder/layer_1/intermediate/dense/PowPow/bert/encoder/layer_1/intermediate/dense/BiasAdd-bert/encoder/layer_1/intermediate/dense/Pow/y*
T0
Z
-bert/encoder/layer_1/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0

+bert/encoder/layer_1/intermediate/dense/mulMul-bert/encoder/layer_1/intermediate/dense/mul/x+bert/encoder/layer_1/intermediate/dense/Pow*
T0

+bert/encoder/layer_1/intermediate/dense/addAdd/bert/encoder/layer_1/intermediate/dense/BiasAdd+bert/encoder/layer_1/intermediate/dense/mul*
T0
\
/bert/encoder/layer_1/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0

-bert/encoder/layer_1/intermediate/dense/mul_1Mul/bert/encoder/layer_1/intermediate/dense/mul_1/x+bert/encoder/layer_1/intermediate/dense/add*
T0
l
,bert/encoder/layer_1/intermediate/dense/TanhTanh-bert/encoder/layer_1/intermediate/dense/mul_1*
T0
\
/bert/encoder/layer_1/intermediate/dense/add_1/xConst*
valueB
 *  ?*
dtype0

-bert/encoder/layer_1/intermediate/dense/add_1Add/bert/encoder/layer_1/intermediate/dense/add_1/x,bert/encoder/layer_1/intermediate/dense/Tanh*
T0
\
/bert/encoder/layer_1/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0

-bert/encoder/layer_1/intermediate/dense/mul_2Mul/bert/encoder/layer_1/intermediate/dense/mul_2/x-bert/encoder/layer_1/intermediate/dense/add_1*
T0

-bert/encoder/layer_1/intermediate/dense/mul_3Mul/bert/encoder/layer_1/intermediate/dense/BiasAdd-bert/encoder/layer_1/intermediate/dense/mul_2*
T0
Â
>mio_variable/bert/encoder/layer_1/output/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*7
	container*(bert/encoder/layer_1/output/dense/kernel
Â
>mio_variable/bert/encoder/layer_1/output/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(bert/encoder/layer_1/output/dense/kernel*
shape:

Z
%Initializer_33/truncated_normal/shapeConst*
dtype0*
valueB"      
Q
$Initializer_33/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_33/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_33/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_33/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 

#Initializer_33/truncated_normal/mulMul/Initializer_33/truncated_normal/TruncatedNormal&Initializer_33/truncated_normal/stddev*
T0
z
Initializer_33/truncated_normalAdd#Initializer_33/truncated_normal/mul$Initializer_33/truncated_normal/mean*
T0
ų
	Assign_33Assign>mio_variable/bert/encoder/layer_1/output/dense/kernel/gradientInitializer_33/truncated_normal*
T0*Q
_classG
ECloc:@mio_variable/bert/encoder/layer_1/output/dense/kernel/gradient*
validate_shape(*
use_locking(
š
<mio_variable/bert/encoder/layer_1/output/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*5
	container(&bert/encoder/layer_1/output/dense/bias
š
<mio_variable/bert/encoder/layer_1/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&bert/encoder/layer_1/output/dense/bias*
shape:
F
Initializer_34/zerosConst*
valueB*    *
dtype0
ę
	Assign_34Assign<mio_variable/bert/encoder/layer_1/output/dense/bias/gradientInitializer_34/zeros*
T0*O
_classE
CAloc:@mio_variable/bert/encoder/layer_1/output/dense/bias/gradient*
validate_shape(*
use_locking(
Đ
(bert/encoder/layer_1/output/dense/MatMulMatMul-bert/encoder/layer_1/intermediate/dense/mul_3>mio_variable/bert/encoder/layer_1/output/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
ŧ
)bert/encoder/layer_1/output/dense/BiasAddBiasAdd(bert/encoder/layer_1/output/dense/MatMul<mio_variable/bert/encoder/layer_1/output/dense/bias/variable*
T0*
data_formatNHWC
Z
-bert/encoder/layer_1/output/dropout/keep_probConst*
dtype0*
valueB
 *fff?
v
)bert/encoder/layer_1/output/dropout/ShapeShape)bert/encoder/layer_1/output/dense/BiasAdd*
out_type0*
T0
c
6bert/encoder/layer_1/output/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
c
6bert/encoder/layer_1/output/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0
Ģ
@bert/encoder/layer_1/output/dropout/random_uniform/RandomUniformRandomUniform)bert/encoder/layer_1/output/dropout/Shape*
dtype0*
seed2 *

seed *
T0
ļ
6bert/encoder/layer_1/output/dropout/random_uniform/subSub6bert/encoder/layer_1/output/dropout/random_uniform/max6bert/encoder/layer_1/output/dropout/random_uniform/min*
T0
Ā
6bert/encoder/layer_1/output/dropout/random_uniform/mulMul@bert/encoder/layer_1/output/dropout/random_uniform/RandomUniform6bert/encoder/layer_1/output/dropout/random_uniform/sub*
T0
˛
2bert/encoder/layer_1/output/dropout/random_uniformAdd6bert/encoder/layer_1/output/dropout/random_uniform/mul6bert/encoder/layer_1/output/dropout/random_uniform/min*
T0

'bert/encoder/layer_1/output/dropout/addAdd-bert/encoder/layer_1/output/dropout/keep_prob2bert/encoder/layer_1/output/dropout/random_uniform*
T0
d
)bert/encoder/layer_1/output/dropout/FloorFloor'bert/encoder/layer_1/output/dropout/add*
T0

'bert/encoder/layer_1/output/dropout/divRealDiv)bert/encoder/layer_1/output/dense/BiasAdd-bert/encoder/layer_1/output/dropout/keep_prob*
T0

'bert/encoder/layer_1/output/dropout/mulMul'bert/encoder/layer_1/output/dropout/div)bert/encoder/layer_1/output/dropout/Floor*
T0

bert/encoder/layer_1/output/addAdd'bert/encoder/layer_1/output/dropout/mul?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add_1*
T0
Á
@mio_variable/bert/encoder/layer_1/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*9
	container,*bert/encoder/layer_1/output/LayerNorm/beta
Á
@mio_variable/bert/encoder/layer_1/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*bert/encoder/layer_1/output/LayerNorm/beta*
shape:
F
Initializer_35/zerosConst*
valueB*    *
dtype0
ō
	Assign_35Assign@mio_variable/bert/encoder/layer_1/output/LayerNorm/beta/gradientInitializer_35/zeros*
use_locking(*
T0*S
_classI
GEloc:@mio_variable/bert/encoder/layer_1/output/LayerNorm/beta/gradient*
validate_shape(
Ã
Amio_variable/bert/encoder/layer_1/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+bert/encoder/layer_1/output/LayerNorm/gamma*
shape:
Ã
Amio_variable/bert/encoder/layer_1/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+bert/encoder/layer_1/output/LayerNorm/gamma*
shape:
E
Initializer_36/onesConst*
valueB*  ?*
dtype0
ķ
	Assign_36AssignAmio_variable/bert/encoder/layer_1/output/LayerNorm/gamma/gradientInitializer_36/ones*
use_locking(*
T0*T
_classJ
HFloc:@mio_variable/bert/encoder/layer_1/output/LayerNorm/gamma/gradient*
validate_shape(
r
Dbert/encoder/layer_1/output/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
valueB:
Į
2bert/encoder/layer_1/output/LayerNorm/moments/meanMeanbert/encoder/layer_1/output/addDbert/encoder/layer_1/output/LayerNorm/moments/mean/reduction_indices*
T0*

Tidx0*
	keep_dims(

:bert/encoder/layer_1/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_1/output/LayerNorm/moments/mean*
T0
ē
?bert/encoder/layer_1/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_1/output/add:bert/encoder/layer_1/output/LayerNorm/moments/StopGradient*
T0
v
Hbert/encoder/layer_1/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0
ī
6bert/encoder/layer_1/output/LayerNorm/moments/varianceMean?bert/encoder/layer_1/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_1/output/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0
b
5bert/encoder/layer_1/output/LayerNorm/batchnorm/add/yConst*
dtype0*
valueB
 *Ėŧ+
˛
3bert/encoder/layer_1/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_1/output/LayerNorm/moments/variance5bert/encoder/layer_1/output/LayerNorm/batchnorm/add/y*
T0
|
5bert/encoder/layer_1/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_1/output/LayerNorm/batchnorm/add*
T0
Ŋ
3bert/encoder/layer_1/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_1/output/LayerNorm/batchnorm/RsqrtAmio_variable/bert/encoder/layer_1/output/LayerNorm/gamma/variable*
T0

5bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_1/output/add3bert/encoder/layer_1/output/LayerNorm/batchnorm/mul*
T0
Ž
5bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_1/output/LayerNorm/moments/mean3bert/encoder/layer_1/output/LayerNorm/batchnorm/mul*
T0
ŧ
3bert/encoder/layer_1/output/LayerNorm/batchnorm/subSub@mio_variable/bert/encoder/layer_1/output/LayerNorm/beta/variable5bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_2*
T0
ą
5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_1/output/LayerNorm/batchnorm/sub*
T0

)bert/encoder/layer_2/attention/self/ShapeShape5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
e
7bert/encoder/layer_2/attention/self/strided_slice/stackConst*
valueB: *
dtype0
g
9bert/encoder/layer_2/attention/self/strided_slice/stack_1Const*
valueB:*
dtype0
g
9bert/encoder/layer_2/attention/self/strided_slice/stack_2Const*
dtype0*
valueB:

1bert/encoder/layer_2/attention/self/strided_sliceStridedSlice)bert/encoder/layer_2/attention/self/Shape7bert/encoder/layer_2/attention/self/strided_slice/stack9bert/encoder/layer_2/attention/self/strided_slice/stack_19bert/encoder/layer_2/attention/self/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 

+bert/encoder/layer_2/attention/self/Shape_1Shape5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
g
9bert/encoder/layer_2/attention/self/strided_slice_1/stackConst*
valueB: *
dtype0
i
;bert/encoder/layer_2/attention/self/strided_slice_1/stack_1Const*
valueB:*
dtype0
i
;bert/encoder/layer_2/attention/self/strided_slice_1/stack_2Const*
valueB:*
dtype0

3bert/encoder/layer_2/attention/self/strided_slice_1StridedSlice+bert/encoder/layer_2/attention/self/Shape_19bert/encoder/layer_2/attention/self/strided_slice_1/stack;bert/encoder/layer_2/attention/self/strided_slice_1/stack_1;bert/encoder/layer_2/attention/self/strided_slice_1/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask
Ō
Fmio_variable/bert/encoder/layer_2/attention/self/query/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*?
	container20bert/encoder/layer_2/attention/self/query/kernel
Ō
Fmio_variable/bert/encoder/layer_2/attention/self/query/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_2/attention/self/query/kernel*
shape:

Z
%Initializer_37/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_37/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_37/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_37/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_37/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 

#Initializer_37/truncated_normal/mulMul/Initializer_37/truncated_normal/TruncatedNormal&Initializer_37/truncated_normal/stddev*
T0
z
Initializer_37/truncated_normalAdd#Initializer_37/truncated_normal/mul$Initializer_37/truncated_normal/mean*
T0

	Assign_37AssignFmio_variable/bert/encoder/layer_2/attention/self/query/kernel/gradientInitializer_37/truncated_normal*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_2/attention/self/query/kernel/gradient*
validate_shape(*
use_locking(
É
Dmio_variable/bert/encoder/layer_2/attention/self/query/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_2/attention/self/query/bias*
shape:
É
Dmio_variable/bert/encoder/layer_2/attention/self/query/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_2/attention/self/query/bias*
shape:
F
Initializer_38/zerosConst*
valueB*    *
dtype0
ú
	Assign_38AssignDmio_variable/bert/encoder/layer_2/attention/self/query/bias/gradientInitializer_38/zeros*W
_classM
KIloc:@mio_variable/bert/encoder/layer_2/attention/self/query/bias/gradient*
validate_shape(*
use_locking(*
T0
č
0bert/encoder/layer_2/attention/self/query/MatMulMatMul5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1Fmio_variable/bert/encoder/layer_2/attention/self/query/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Ô
1bert/encoder/layer_2/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_2/attention/self/query/MatMulDmio_variable/bert/encoder/layer_2/attention/self/query/bias/variable*
T0*
data_formatNHWC
Î
Dmio_variable/bert/encoder/layer_2/attention/self/key/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_2/attention/self/key/kernel*
shape:

Î
Dmio_variable/bert/encoder/layer_2/attention/self/key/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*=
	container0.bert/encoder/layer_2/attention/self/key/kernel
Z
%Initializer_39/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_39/truncated_normal/meanConst*
dtype0*
valueB
 *    
S
&Initializer_39/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_39/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_39/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0

#Initializer_39/truncated_normal/mulMul/Initializer_39/truncated_normal/TruncatedNormal&Initializer_39/truncated_normal/stddev*
T0
z
Initializer_39/truncated_normalAdd#Initializer_39/truncated_normal/mul$Initializer_39/truncated_normal/mean*
T0

	Assign_39AssignDmio_variable/bert/encoder/layer_2/attention/self/key/kernel/gradientInitializer_39/truncated_normal*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_2/attention/self/key/kernel/gradient*
validate_shape(
Å
Bmio_variable/bert/encoder/layer_2/attention/self/key/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_2/attention/self/key/bias*
shape:
Å
Bmio_variable/bert/encoder/layer_2/attention/self/key/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_2/attention/self/key/bias*
shape:
F
Initializer_40/zerosConst*
valueB*    *
dtype0
ö
	Assign_40AssignBmio_variable/bert/encoder/layer_2/attention/self/key/bias/gradientInitializer_40/zeros*
use_locking(*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_2/attention/self/key/bias/gradient*
validate_shape(
ä
.bert/encoder/layer_2/attention/self/key/MatMulMatMul5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1Dmio_variable/bert/encoder/layer_2/attention/self/key/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Î
/bert/encoder/layer_2/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_2/attention/self/key/MatMulBmio_variable/bert/encoder/layer_2/attention/self/key/bias/variable*
T0*
data_formatNHWC
Ō
Fmio_variable/bert/encoder/layer_2/attention/self/value/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*?
	container20bert/encoder/layer_2/attention/self/value/kernel
Ō
Fmio_variable/bert/encoder/layer_2/attention/self/value/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_2/attention/self/value/kernel*
shape:

Z
%Initializer_41/truncated_normal/shapeConst*
dtype0*
valueB"      
Q
$Initializer_41/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_41/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_41/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_41/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 

#Initializer_41/truncated_normal/mulMul/Initializer_41/truncated_normal/TruncatedNormal&Initializer_41/truncated_normal/stddev*
T0
z
Initializer_41/truncated_normalAdd#Initializer_41/truncated_normal/mul$Initializer_41/truncated_normal/mean*
T0

	Assign_41AssignFmio_variable/bert/encoder/layer_2/attention/self/value/kernel/gradientInitializer_41/truncated_normal*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_2/attention/self/value/kernel/gradient*
validate_shape(
É
Dmio_variable/bert/encoder/layer_2/attention/self/value/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_2/attention/self/value/bias*
shape:
É
Dmio_variable/bert/encoder/layer_2/attention/self/value/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_2/attention/self/value/bias*
shape:
F
Initializer_42/zerosConst*
valueB*    *
dtype0
ú
	Assign_42AssignDmio_variable/bert/encoder/layer_2/attention/self/value/bias/gradientInitializer_42/zeros*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_2/attention/self/value/bias/gradient*
validate_shape(
č
0bert/encoder/layer_2/attention/self/value/MatMulMatMul5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1Fmio_variable/bert/encoder/layer_2/attention/self/value/kernel/variable*
transpose_a( *
transpose_b( *
T0
Ô
1bert/encoder/layer_2/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_2/attention/self/value/MatMulDmio_variable/bert/encoder/layer_2/attention/self/value/bias/variable*
T0*
data_formatNHWC
]
3bert/encoder/layer_2/attention/self/Reshape/shape/1Const*
value	B :*
dtype0
]
3bert/encoder/layer_2/attention/self/Reshape/shape/2Const*
value	B :*
dtype0
]
3bert/encoder/layer_2/attention/self/Reshape/shape/3Const*
value	B : *
dtype0

1bert/encoder/layer_2/attention/self/Reshape/shapePackbert/encoder/strided_slice_23bert/encoder/layer_2/attention/self/Reshape/shape/13bert/encoder/layer_2/attention/self/Reshape/shape/23bert/encoder/layer_2/attention/self/Reshape/shape/3*

axis *
N*
T0
ŗ
+bert/encoder/layer_2/attention/self/ReshapeReshape1bert/encoder/layer_2/attention/self/query/BiasAdd1bert/encoder/layer_2/attention/self/Reshape/shape*
T0*
Tshape0
o
2bert/encoder/layer_2/attention/self/transpose/permConst*
dtype0*%
valueB"             
ą
-bert/encoder/layer_2/attention/self/transpose	Transpose+bert/encoder/layer_2/attention/self/Reshape2bert/encoder/layer_2/attention/self/transpose/perm*
Tperm0*
T0
_
5bert/encoder/layer_2/attention/self/Reshape_1/shape/1Const*
value	B :*
dtype0
_
5bert/encoder/layer_2/attention/self/Reshape_1/shape/2Const*
value	B :*
dtype0
_
5bert/encoder/layer_2/attention/self/Reshape_1/shape/3Const*
value	B : *
dtype0

3bert/encoder/layer_2/attention/self/Reshape_1/shapePackbert/encoder/strided_slice_25bert/encoder/layer_2/attention/self/Reshape_1/shape/15bert/encoder/layer_2/attention/self/Reshape_1/shape/25bert/encoder/layer_2/attention/self/Reshape_1/shape/3*
N*
T0*

axis 
ĩ
-bert/encoder/layer_2/attention/self/Reshape_1Reshape/bert/encoder/layer_2/attention/self/key/BiasAdd3bert/encoder/layer_2/attention/self/Reshape_1/shape*
T0*
Tshape0
q
4bert/encoder/layer_2/attention/self/transpose_1/permConst*%
valueB"             *
dtype0
ˇ
/bert/encoder/layer_2/attention/self/transpose_1	Transpose-bert/encoder/layer_2/attention/self/Reshape_14bert/encoder/layer_2/attention/self/transpose_1/perm*
Tperm0*
T0
ŧ
*bert/encoder/layer_2/attention/self/MatMulBatchMatMul-bert/encoder/layer_2/attention/self/transpose/bert/encoder/layer_2/attention/self/transpose_1*
T0*
adj_x( *
adj_y(
V
)bert/encoder/layer_2/attention/self/Mul/yConst*
dtype0*
valueB
 *ķ5>

'bert/encoder/layer_2/attention/self/MulMul*bert/encoder/layer_2/attention/self/MatMul)bert/encoder/layer_2/attention/self/Mul/y*
T0
`
2bert/encoder/layer_2/attention/self/ExpandDims/dimConst*
dtype0*
valueB:

.bert/encoder/layer_2/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_2/attention/self/ExpandDims/dim*

Tdim0*
T0
V
)bert/encoder/layer_2/attention/self/sub/xConst*
valueB
 *  ?*
dtype0

'bert/encoder/layer_2/attention/self/subSub)bert/encoder/layer_2/attention/self/sub/x.bert/encoder/layer_2/attention/self/ExpandDims*
T0
X
+bert/encoder/layer_2/attention/self/mul_1/yConst*
dtype0*
valueB
 * @Æ

)bert/encoder/layer_2/attention/self/mul_1Mul'bert/encoder/layer_2/attention/self/sub+bert/encoder/layer_2/attention/self/mul_1/y*
T0

'bert/encoder/layer_2/attention/self/addAdd'bert/encoder/layer_2/attention/self/Mul)bert/encoder/layer_2/attention/self/mul_1*
T0
h
+bert/encoder/layer_2/attention/self/SoftmaxSoftmax'bert/encoder/layer_2/attention/self/add*
T0
b
5bert/encoder/layer_2/attention/self/dropout/keep_probConst*
dtype0*
valueB
 *fff?

1bert/encoder/layer_2/attention/self/dropout/ShapeShape+bert/encoder/layer_2/attention/self/Softmax*
T0*
out_type0
k
>bert/encoder/layer_2/attention/self/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
k
>bert/encoder/layer_2/attention/self/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0
ģ
Hbert/encoder/layer_2/attention/self/dropout/random_uniform/RandomUniformRandomUniform1bert/encoder/layer_2/attention/self/dropout/Shape*

seed *
T0*
dtype0*
seed2 
Î
>bert/encoder/layer_2/attention/self/dropout/random_uniform/subSub>bert/encoder/layer_2/attention/self/dropout/random_uniform/max>bert/encoder/layer_2/attention/self/dropout/random_uniform/min*
T0
Ø
>bert/encoder/layer_2/attention/self/dropout/random_uniform/mulMulHbert/encoder/layer_2/attention/self/dropout/random_uniform/RandomUniform>bert/encoder/layer_2/attention/self/dropout/random_uniform/sub*
T0
Ę
:bert/encoder/layer_2/attention/self/dropout/random_uniformAdd>bert/encoder/layer_2/attention/self/dropout/random_uniform/mul>bert/encoder/layer_2/attention/self/dropout/random_uniform/min*
T0
˛
/bert/encoder/layer_2/attention/self/dropout/addAdd5bert/encoder/layer_2/attention/self/dropout/keep_prob:bert/encoder/layer_2/attention/self/dropout/random_uniform*
T0
t
1bert/encoder/layer_2/attention/self/dropout/FloorFloor/bert/encoder/layer_2/attention/self/dropout/add*
T0
§
/bert/encoder/layer_2/attention/self/dropout/divRealDiv+bert/encoder/layer_2/attention/self/Softmax5bert/encoder/layer_2/attention/self/dropout/keep_prob*
T0
Ŗ
/bert/encoder/layer_2/attention/self/dropout/mulMul/bert/encoder/layer_2/attention/self/dropout/div1bert/encoder/layer_2/attention/self/dropout/Floor*
T0
_
5bert/encoder/layer_2/attention/self/Reshape_2/shape/1Const*
value	B :*
dtype0
_
5bert/encoder/layer_2/attention/self/Reshape_2/shape/2Const*
value	B :*
dtype0
_
5bert/encoder/layer_2/attention/self/Reshape_2/shape/3Const*
value	B : *
dtype0

3bert/encoder/layer_2/attention/self/Reshape_2/shapePackbert/encoder/strided_slice_25bert/encoder/layer_2/attention/self/Reshape_2/shape/15bert/encoder/layer_2/attention/self/Reshape_2/shape/25bert/encoder/layer_2/attention/self/Reshape_2/shape/3*
T0*

axis *
N
ˇ
-bert/encoder/layer_2/attention/self/Reshape_2Reshape1bert/encoder/layer_2/attention/self/value/BiasAdd3bert/encoder/layer_2/attention/self/Reshape_2/shape*
T0*
Tshape0
q
4bert/encoder/layer_2/attention/self/transpose_2/permConst*
dtype0*%
valueB"             
ˇ
/bert/encoder/layer_2/attention/self/transpose_2	Transpose-bert/encoder/layer_2/attention/self/Reshape_24bert/encoder/layer_2/attention/self/transpose_2/perm*
Tperm0*
T0
Ā
,bert/encoder/layer_2/attention/self/MatMul_1BatchMatMul/bert/encoder/layer_2/attention/self/dropout/mul/bert/encoder/layer_2/attention/self/transpose_2*
T0*
adj_x( *
adj_y( 
q
4bert/encoder/layer_2/attention/self/transpose_3/permConst*%
valueB"             *
dtype0
ļ
/bert/encoder/layer_2/attention/self/transpose_3	Transpose,bert/encoder/layer_2/attention/self/MatMul_14bert/encoder/layer_2/attention/self/transpose_3/perm*
T0*
Tperm0
U
+bert/encoder/layer_2/attention/self/mul_2/yConst*
value	B :*
dtype0

)bert/encoder/layer_2/attention/self/mul_2Mulbert/encoder/strided_slice_2+bert/encoder/layer_2/attention/self/mul_2/y*
T0
`
5bert/encoder/layer_2/attention/self/Reshape_3/shape/1Const*
value
B :*
dtype0
ģ
3bert/encoder/layer_2/attention/self/Reshape_3/shapePack)bert/encoder/layer_2/attention/self/mul_25bert/encoder/layer_2/attention/self/Reshape_3/shape/1*
T0*

axis *
N
ĩ
-bert/encoder/layer_2/attention/self/Reshape_3Reshape/bert/encoder/layer_2/attention/self/transpose_33bert/encoder/layer_2/attention/self/Reshape_3/shape*
Tshape0*
T0
Ö
Hmio_variable/bert/encoder/layer_2/attention/output/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42bert/encoder/layer_2/attention/output/dense/kernel*
shape:

Ö
Hmio_variable/bert/encoder/layer_2/attention/output/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*A
	container42bert/encoder/layer_2/attention/output/dense/kernel
Z
%Initializer_43/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_43/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_43/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_43/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_43/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 

#Initializer_43/truncated_normal/mulMul/Initializer_43/truncated_normal/TruncatedNormal&Initializer_43/truncated_normal/stddev*
T0
z
Initializer_43/truncated_normalAdd#Initializer_43/truncated_normal/mul$Initializer_43/truncated_normal/mean*
T0

	Assign_43AssignHmio_variable/bert/encoder/layer_2/attention/output/dense/kernel/gradientInitializer_43/truncated_normal*[
_classQ
OMloc:@mio_variable/bert/encoder/layer_2/attention/output/dense/kernel/gradient*
validate_shape(*
use_locking(*
T0
Í
Fmio_variable/bert/encoder/layer_2/attention/output/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*?
	container20bert/encoder/layer_2/attention/output/dense/bias
Í
Fmio_variable/bert/encoder/layer_2/attention/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_2/attention/output/dense/bias*
shape:
F
Initializer_44/zerosConst*
valueB*    *
dtype0
ū
	Assign_44AssignFmio_variable/bert/encoder/layer_2/attention/output/dense/bias/gradientInitializer_44/zeros*
validate_shape(*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_2/attention/output/dense/bias/gradient
ä
2bert/encoder/layer_2/attention/output/dense/MatMulMatMul-bert/encoder/layer_2/attention/self/Reshape_3Hmio_variable/bert/encoder/layer_2/attention/output/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Ú
3bert/encoder/layer_2/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_2/attention/output/dense/MatMulFmio_variable/bert/encoder/layer_2/attention/output/dense/bias/variable*
T0*
data_formatNHWC
d
7bert/encoder/layer_2/attention/output/dropout/keep_probConst*
dtype0*
valueB
 *fff?

3bert/encoder/layer_2/attention/output/dropout/ShapeShape3bert/encoder/layer_2/attention/output/dense/BiasAdd*
T0*
out_type0
m
@bert/encoder/layer_2/attention/output/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
m
@bert/encoder/layer_2/attention/output/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0
ŋ
Jbert/encoder/layer_2/attention/output/dropout/random_uniform/RandomUniformRandomUniform3bert/encoder/layer_2/attention/output/dropout/Shape*
T0*
dtype0*
seed2 *

seed 
Ô
@bert/encoder/layer_2/attention/output/dropout/random_uniform/subSub@bert/encoder/layer_2/attention/output/dropout/random_uniform/max@bert/encoder/layer_2/attention/output/dropout/random_uniform/min*
T0
Ū
@bert/encoder/layer_2/attention/output/dropout/random_uniform/mulMulJbert/encoder/layer_2/attention/output/dropout/random_uniform/RandomUniform@bert/encoder/layer_2/attention/output/dropout/random_uniform/sub*
T0
Đ
<bert/encoder/layer_2/attention/output/dropout/random_uniformAdd@bert/encoder/layer_2/attention/output/dropout/random_uniform/mul@bert/encoder/layer_2/attention/output/dropout/random_uniform/min*
T0
¸
1bert/encoder/layer_2/attention/output/dropout/addAdd7bert/encoder/layer_2/attention/output/dropout/keep_prob<bert/encoder/layer_2/attention/output/dropout/random_uniform*
T0
x
3bert/encoder/layer_2/attention/output/dropout/FloorFloor1bert/encoder/layer_2/attention/output/dropout/add*
T0
ŗ
1bert/encoder/layer_2/attention/output/dropout/divRealDiv3bert/encoder/layer_2/attention/output/dense/BiasAdd7bert/encoder/layer_2/attention/output/dropout/keep_prob*
T0
Š
1bert/encoder/layer_2/attention/output/dropout/mulMul1bert/encoder/layer_2/attention/output/dropout/div3bert/encoder/layer_2/attention/output/dropout/Floor*
T0
Ŗ
)bert/encoder/layer_2/attention/output/addAdd1bert/encoder/layer_2/attention/output/dropout/mul5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1*
T0
Õ
Jmio_variable/bert/encoder/layer_2/attention/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*C
	container64bert/encoder/layer_2/attention/output/LayerNorm/beta
Õ
Jmio_variable/bert/encoder/layer_2/attention/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64bert/encoder/layer_2/attention/output/LayerNorm/beta*
shape:
F
Initializer_45/zerosConst*
valueB*    *
dtype0

	Assign_45AssignJmio_variable/bert/encoder/layer_2/attention/output/LayerNorm/beta/gradientInitializer_45/zeros*
use_locking(*
T0*]
_classS
QOloc:@mio_variable/bert/encoder/layer_2/attention/output/LayerNorm/beta/gradient*
validate_shape(
×
Kmio_variable/bert/encoder/layer_2/attention/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*D
	container75bert/encoder/layer_2/attention/output/LayerNorm/gamma*
shape:
×
Kmio_variable/bert/encoder/layer_2/attention/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*D
	container75bert/encoder/layer_2/attention/output/LayerNorm/gamma
E
Initializer_46/onesConst*
valueB*  ?*
dtype0

	Assign_46AssignKmio_variable/bert/encoder/layer_2/attention/output/LayerNorm/gamma/gradientInitializer_46/ones*
use_locking(*
T0*^
_classT
RPloc:@mio_variable/bert/encoder/layer_2/attention/output/LayerNorm/gamma/gradient*
validate_shape(
|
Nbert/encoder/layer_2/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0
å
<bert/encoder/layer_2/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_2/attention/output/addNbert/encoder/layer_2/attention/output/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0

Dbert/encoder/layer_2/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_2/attention/output/LayerNorm/moments/mean*
T0
Ø
Ibert/encoder/layer_2/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_2/attention/output/addDbert/encoder/layer_2/attention/output/LayerNorm/moments/StopGradient*
T0

Rbert/encoder/layer_2/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0

@bert/encoder/layer_2/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_2/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_2/attention/output/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0
l
?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ėŧ+*
dtype0
Đ
=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_2/attention/output/LayerNorm/moments/variance?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add/y*
T0

?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add*
T0
Û
=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/RsqrtKmio_variable/bert/encoder/layer_2/attention/output/LayerNorm/gamma/variable*
T0
š
?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_2/attention/output/add=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul*
T0
Ė
?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_2/attention/output/LayerNorm/moments/mean=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul*
T0
Ú
=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/subSubJmio_variable/bert/encoder/layer_2/attention/output/LayerNorm/beta/variable?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul_2*
T0
Ī
?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/sub*
T0
Î
Dmio_variable/bert/encoder/layer_2/intermediate/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*=
	container0.bert/encoder/layer_2/intermediate/dense/kernel
Î
Dmio_variable/bert/encoder/layer_2/intermediate/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_2/intermediate/dense/kernel*
shape:

Z
%Initializer_47/truncated_normal/shapeConst*
dtype0*
valueB"      
Q
$Initializer_47/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_47/truncated_normal/stddevConst*
dtype0*
valueB
 *
×Ŗ<

/Initializer_47/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_47/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

#Initializer_47/truncated_normal/mulMul/Initializer_47/truncated_normal/TruncatedNormal&Initializer_47/truncated_normal/stddev*
T0
z
Initializer_47/truncated_normalAdd#Initializer_47/truncated_normal/mul$Initializer_47/truncated_normal/mean*
T0

	Assign_47AssignDmio_variable/bert/encoder/layer_2/intermediate/dense/kernel/gradientInitializer_47/truncated_normal*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_2/intermediate/dense/kernel/gradient*
validate_shape(
Å
Bmio_variable/bert/encoder/layer_2/intermediate/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_2/intermediate/dense/bias*
shape:
Å
Bmio_variable/bert/encoder/layer_2/intermediate/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*;
	container.,bert/encoder/layer_2/intermediate/dense/bias
S
$Initializer_48/zeros/shape_as_tensorConst*
valueB:*
dtype0
G
Initializer_48/zeros/ConstConst*
dtype0*
valueB
 *    
y
Initializer_48/zerosFill$Initializer_48/zeros/shape_as_tensorInitializer_48/zeros/Const*
T0*

index_type0
ö
	Assign_48AssignBmio_variable/bert/encoder/layer_2/intermediate/dense/bias/gradientInitializer_48/zeros*U
_classK
IGloc:@mio_variable/bert/encoder/layer_2/intermediate/dense/bias/gradient*
validate_shape(*
use_locking(*
T0
î
.bert/encoder/layer_2/intermediate/dense/MatMulMatMul?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add_1Dmio_variable/bert/encoder/layer_2/intermediate/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Î
/bert/encoder/layer_2/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_2/intermediate/dense/MatMulBmio_variable/bert/encoder/layer_2/intermediate/dense/bias/variable*
data_formatNHWC*
T0
Z
-bert/encoder/layer_2/intermediate/dense/Pow/yConst*
valueB
 *  @@*
dtype0

+bert/encoder/layer_2/intermediate/dense/PowPow/bert/encoder/layer_2/intermediate/dense/BiasAdd-bert/encoder/layer_2/intermediate/dense/Pow/y*
T0
Z
-bert/encoder/layer_2/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0

+bert/encoder/layer_2/intermediate/dense/mulMul-bert/encoder/layer_2/intermediate/dense/mul/x+bert/encoder/layer_2/intermediate/dense/Pow*
T0

+bert/encoder/layer_2/intermediate/dense/addAdd/bert/encoder/layer_2/intermediate/dense/BiasAdd+bert/encoder/layer_2/intermediate/dense/mul*
T0
\
/bert/encoder/layer_2/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0

-bert/encoder/layer_2/intermediate/dense/mul_1Mul/bert/encoder/layer_2/intermediate/dense/mul_1/x+bert/encoder/layer_2/intermediate/dense/add*
T0
l
,bert/encoder/layer_2/intermediate/dense/TanhTanh-bert/encoder/layer_2/intermediate/dense/mul_1*
T0
\
/bert/encoder/layer_2/intermediate/dense/add_1/xConst*
dtype0*
valueB
 *  ?

-bert/encoder/layer_2/intermediate/dense/add_1Add/bert/encoder/layer_2/intermediate/dense/add_1/x,bert/encoder/layer_2/intermediate/dense/Tanh*
T0
\
/bert/encoder/layer_2/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0

-bert/encoder/layer_2/intermediate/dense/mul_2Mul/bert/encoder/layer_2/intermediate/dense/mul_2/x-bert/encoder/layer_2/intermediate/dense/add_1*
T0

-bert/encoder/layer_2/intermediate/dense/mul_3Mul/bert/encoder/layer_2/intermediate/dense/BiasAdd-bert/encoder/layer_2/intermediate/dense/mul_2*
T0
Â
>mio_variable/bert/encoder/layer_2/output/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(bert/encoder/layer_2/output/dense/kernel*
shape:

Â
>mio_variable/bert/encoder/layer_2/output/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*7
	container*(bert/encoder/layer_2/output/dense/kernel
Z
%Initializer_49/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_49/truncated_normal/meanConst*
dtype0*
valueB
 *    
S
&Initializer_49/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_49/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_49/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 

#Initializer_49/truncated_normal/mulMul/Initializer_49/truncated_normal/TruncatedNormal&Initializer_49/truncated_normal/stddev*
T0
z
Initializer_49/truncated_normalAdd#Initializer_49/truncated_normal/mul$Initializer_49/truncated_normal/mean*
T0
ų
	Assign_49Assign>mio_variable/bert/encoder/layer_2/output/dense/kernel/gradientInitializer_49/truncated_normal*
use_locking(*
T0*Q
_classG
ECloc:@mio_variable/bert/encoder/layer_2/output/dense/kernel/gradient*
validate_shape(
š
<mio_variable/bert/encoder/layer_2/output/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&bert/encoder/layer_2/output/dense/bias*
shape:
š
<mio_variable/bert/encoder/layer_2/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&bert/encoder/layer_2/output/dense/bias*
shape:
F
Initializer_50/zerosConst*
dtype0*
valueB*    
ę
	Assign_50Assign<mio_variable/bert/encoder/layer_2/output/dense/bias/gradientInitializer_50/zeros*
validate_shape(*
use_locking(*
T0*O
_classE
CAloc:@mio_variable/bert/encoder/layer_2/output/dense/bias/gradient
Đ
(bert/encoder/layer_2/output/dense/MatMulMatMul-bert/encoder/layer_2/intermediate/dense/mul_3>mio_variable/bert/encoder/layer_2/output/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
ŧ
)bert/encoder/layer_2/output/dense/BiasAddBiasAdd(bert/encoder/layer_2/output/dense/MatMul<mio_variable/bert/encoder/layer_2/output/dense/bias/variable*
T0*
data_formatNHWC
Z
-bert/encoder/layer_2/output/dropout/keep_probConst*
valueB
 *fff?*
dtype0
v
)bert/encoder/layer_2/output/dropout/ShapeShape)bert/encoder/layer_2/output/dense/BiasAdd*
T0*
out_type0
c
6bert/encoder/layer_2/output/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
c
6bert/encoder/layer_2/output/dropout/random_uniform/maxConst*
dtype0*
valueB
 *  ?
Ģ
@bert/encoder/layer_2/output/dropout/random_uniform/RandomUniformRandomUniform)bert/encoder/layer_2/output/dropout/Shape*
dtype0*
seed2 *

seed *
T0
ļ
6bert/encoder/layer_2/output/dropout/random_uniform/subSub6bert/encoder/layer_2/output/dropout/random_uniform/max6bert/encoder/layer_2/output/dropout/random_uniform/min*
T0
Ā
6bert/encoder/layer_2/output/dropout/random_uniform/mulMul@bert/encoder/layer_2/output/dropout/random_uniform/RandomUniform6bert/encoder/layer_2/output/dropout/random_uniform/sub*
T0
˛
2bert/encoder/layer_2/output/dropout/random_uniformAdd6bert/encoder/layer_2/output/dropout/random_uniform/mul6bert/encoder/layer_2/output/dropout/random_uniform/min*
T0

'bert/encoder/layer_2/output/dropout/addAdd-bert/encoder/layer_2/output/dropout/keep_prob2bert/encoder/layer_2/output/dropout/random_uniform*
T0
d
)bert/encoder/layer_2/output/dropout/FloorFloor'bert/encoder/layer_2/output/dropout/add*
T0

'bert/encoder/layer_2/output/dropout/divRealDiv)bert/encoder/layer_2/output/dense/BiasAdd-bert/encoder/layer_2/output/dropout/keep_prob*
T0

'bert/encoder/layer_2/output/dropout/mulMul'bert/encoder/layer_2/output/dropout/div)bert/encoder/layer_2/output/dropout/Floor*
T0

bert/encoder/layer_2/output/addAdd'bert/encoder/layer_2/output/dropout/mul?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add_1*
T0
Á
@mio_variable/bert/encoder/layer_2/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*9
	container,*bert/encoder/layer_2/output/LayerNorm/beta
Á
@mio_variable/bert/encoder/layer_2/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*bert/encoder/layer_2/output/LayerNorm/beta*
shape:
F
Initializer_51/zerosConst*
valueB*    *
dtype0
ō
	Assign_51Assign@mio_variable/bert/encoder/layer_2/output/LayerNorm/beta/gradientInitializer_51/zeros*
use_locking(*
T0*S
_classI
GEloc:@mio_variable/bert/encoder/layer_2/output/LayerNorm/beta/gradient*
validate_shape(
Ã
Amio_variable/bert/encoder/layer_2/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*:
	container-+bert/encoder/layer_2/output/LayerNorm/gamma
Ã
Amio_variable/bert/encoder/layer_2/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*:
	container-+bert/encoder/layer_2/output/LayerNorm/gamma
E
Initializer_52/onesConst*
valueB*  ?*
dtype0
ķ
	Assign_52AssignAmio_variable/bert/encoder/layer_2/output/LayerNorm/gamma/gradientInitializer_52/ones*
validate_shape(*
use_locking(*
T0*T
_classJ
HFloc:@mio_variable/bert/encoder/layer_2/output/LayerNorm/gamma/gradient
r
Dbert/encoder/layer_2/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0
Į
2bert/encoder/layer_2/output/LayerNorm/moments/meanMeanbert/encoder/layer_2/output/addDbert/encoder/layer_2/output/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0

:bert/encoder/layer_2/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_2/output/LayerNorm/moments/mean*
T0
ē
?bert/encoder/layer_2/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_2/output/add:bert/encoder/layer_2/output/LayerNorm/moments/StopGradient*
T0
v
Hbert/encoder/layer_2/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0
ī
6bert/encoder/layer_2/output/LayerNorm/moments/varianceMean?bert/encoder/layer_2/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_2/output/LayerNorm/moments/variance/reduction_indices*
T0*

Tidx0*
	keep_dims(
b
5bert/encoder/layer_2/output/LayerNorm/batchnorm/add/yConst*
dtype0*
valueB
 *Ėŧ+
˛
3bert/encoder/layer_2/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_2/output/LayerNorm/moments/variance5bert/encoder/layer_2/output/LayerNorm/batchnorm/add/y*
T0
|
5bert/encoder/layer_2/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_2/output/LayerNorm/batchnorm/add*
T0
Ŋ
3bert/encoder/layer_2/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_2/output/LayerNorm/batchnorm/RsqrtAmio_variable/bert/encoder/layer_2/output/LayerNorm/gamma/variable*
T0

5bert/encoder/layer_2/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_2/output/add3bert/encoder/layer_2/output/LayerNorm/batchnorm/mul*
T0
Ž
5bert/encoder/layer_2/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_2/output/LayerNorm/moments/mean3bert/encoder/layer_2/output/LayerNorm/batchnorm/mul*
T0
ŧ
3bert/encoder/layer_2/output/LayerNorm/batchnorm/subSub@mio_variable/bert/encoder/layer_2/output/LayerNorm/beta/variable5bert/encoder/layer_2/output/LayerNorm/batchnorm/mul_2*
T0
ą
5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_2/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_2/output/LayerNorm/batchnorm/sub*
T0

)bert/encoder/layer_3/attention/self/ShapeShape5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
e
7bert/encoder/layer_3/attention/self/strided_slice/stackConst*
valueB: *
dtype0
g
9bert/encoder/layer_3/attention/self/strided_slice/stack_1Const*
valueB:*
dtype0
g
9bert/encoder/layer_3/attention/self/strided_slice/stack_2Const*
valueB:*
dtype0

1bert/encoder/layer_3/attention/self/strided_sliceStridedSlice)bert/encoder/layer_3/attention/self/Shape7bert/encoder/layer_3/attention/self/strided_slice/stack9bert/encoder/layer_3/attention/self/strided_slice/stack_19bert/encoder/layer_3/attention/self/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0

+bert/encoder/layer_3/attention/self/Shape_1Shape5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
g
9bert/encoder/layer_3/attention/self/strided_slice_1/stackConst*
valueB: *
dtype0
i
;bert/encoder/layer_3/attention/self/strided_slice_1/stack_1Const*
dtype0*
valueB:
i
;bert/encoder/layer_3/attention/self/strided_slice_1/stack_2Const*
valueB:*
dtype0

3bert/encoder/layer_3/attention/self/strided_slice_1StridedSlice+bert/encoder/layer_3/attention/self/Shape_19bert/encoder/layer_3/attention/self/strided_slice_1/stack;bert/encoder/layer_3/attention/self/strided_slice_1/stack_1;bert/encoder/layer_3/attention/self/strided_slice_1/stack_2*
end_mask *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
Ō
Fmio_variable/bert/encoder/layer_3/attention/self/query/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_3/attention/self/query/kernel*
shape:

Ō
Fmio_variable/bert/encoder/layer_3/attention/self/query/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_3/attention/self/query/kernel*
shape:

Z
%Initializer_53/truncated_normal/shapeConst*
dtype0*
valueB"      
Q
$Initializer_53/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_53/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_53/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_53/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 

#Initializer_53/truncated_normal/mulMul/Initializer_53/truncated_normal/TruncatedNormal&Initializer_53/truncated_normal/stddev*
T0
z
Initializer_53/truncated_normalAdd#Initializer_53/truncated_normal/mul$Initializer_53/truncated_normal/mean*
T0

	Assign_53AssignFmio_variable/bert/encoder/layer_3/attention/self/query/kernel/gradientInitializer_53/truncated_normal*
validate_shape(*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_3/attention/self/query/kernel/gradient
É
Dmio_variable/bert/encoder/layer_3/attention/self/query/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_3/attention/self/query/bias*
shape:
É
Dmio_variable/bert/encoder/layer_3/attention/self/query/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_3/attention/self/query/bias*
shape:
F
Initializer_54/zerosConst*
valueB*    *
dtype0
ú
	Assign_54AssignDmio_variable/bert/encoder/layer_3/attention/self/query/bias/gradientInitializer_54/zeros*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_3/attention/self/query/bias/gradient*
validate_shape(
č
0bert/encoder/layer_3/attention/self/query/MatMulMatMul5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1Fmio_variable/bert/encoder/layer_3/attention/self/query/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Ô
1bert/encoder/layer_3/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_3/attention/self/query/MatMulDmio_variable/bert/encoder/layer_3/attention/self/query/bias/variable*
T0*
data_formatNHWC
Î
Dmio_variable/bert/encoder/layer_3/attention/self/key/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_3/attention/self/key/kernel*
shape:

Î
Dmio_variable/bert/encoder/layer_3/attention/self/key/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_3/attention/self/key/kernel*
shape:

Z
%Initializer_55/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_55/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_55/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_55/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_55/truncated_normal/shape*
seed2 *

seed *
T0*
dtype0

#Initializer_55/truncated_normal/mulMul/Initializer_55/truncated_normal/TruncatedNormal&Initializer_55/truncated_normal/stddev*
T0
z
Initializer_55/truncated_normalAdd#Initializer_55/truncated_normal/mul$Initializer_55/truncated_normal/mean*
T0

	Assign_55AssignDmio_variable/bert/encoder/layer_3/attention/self/key/kernel/gradientInitializer_55/truncated_normal*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_3/attention/self/key/kernel/gradient*
validate_shape(
Å
Bmio_variable/bert/encoder/layer_3/attention/self/key/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_3/attention/self/key/bias*
shape:
Å
Bmio_variable/bert/encoder/layer_3/attention/self/key/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_3/attention/self/key/bias*
shape:
F
Initializer_56/zerosConst*
valueB*    *
dtype0
ö
	Assign_56AssignBmio_variable/bert/encoder/layer_3/attention/self/key/bias/gradientInitializer_56/zeros*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_3/attention/self/key/bias/gradient*
validate_shape(*
use_locking(
ä
.bert/encoder/layer_3/attention/self/key/MatMulMatMul5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1Dmio_variable/bert/encoder/layer_3/attention/self/key/kernel/variable*
transpose_a( *
transpose_b( *
T0
Î
/bert/encoder/layer_3/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_3/attention/self/key/MatMulBmio_variable/bert/encoder/layer_3/attention/self/key/bias/variable*
T0*
data_formatNHWC
Ō
Fmio_variable/bert/encoder/layer_3/attention/self/value/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_3/attention/self/value/kernel*
shape:

Ō
Fmio_variable/bert/encoder/layer_3/attention/self/value/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_3/attention/self/value/kernel*
shape:

Z
%Initializer_57/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_57/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_57/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_57/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_57/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

#Initializer_57/truncated_normal/mulMul/Initializer_57/truncated_normal/TruncatedNormal&Initializer_57/truncated_normal/stddev*
T0
z
Initializer_57/truncated_normalAdd#Initializer_57/truncated_normal/mul$Initializer_57/truncated_normal/mean*
T0

	Assign_57AssignFmio_variable/bert/encoder/layer_3/attention/self/value/kernel/gradientInitializer_57/truncated_normal*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_3/attention/self/value/kernel/gradient*
validate_shape(
É
Dmio_variable/bert/encoder/layer_3/attention/self/value/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_3/attention/self/value/bias*
shape:
É
Dmio_variable/bert/encoder/layer_3/attention/self/value/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_3/attention/self/value/bias*
shape:
F
Initializer_58/zerosConst*
valueB*    *
dtype0
ú
	Assign_58AssignDmio_variable/bert/encoder/layer_3/attention/self/value/bias/gradientInitializer_58/zeros*
validate_shape(*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_3/attention/self/value/bias/gradient
č
0bert/encoder/layer_3/attention/self/value/MatMulMatMul5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1Fmio_variable/bert/encoder/layer_3/attention/self/value/kernel/variable*
transpose_a( *
transpose_b( *
T0
Ô
1bert/encoder/layer_3/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_3/attention/self/value/MatMulDmio_variable/bert/encoder/layer_3/attention/self/value/bias/variable*
T0*
data_formatNHWC
]
3bert/encoder/layer_3/attention/self/Reshape/shape/1Const*
value	B :*
dtype0
]
3bert/encoder/layer_3/attention/self/Reshape/shape/2Const*
value	B :*
dtype0
]
3bert/encoder/layer_3/attention/self/Reshape/shape/3Const*
dtype0*
value	B : 

1bert/encoder/layer_3/attention/self/Reshape/shapePackbert/encoder/strided_slice_23bert/encoder/layer_3/attention/self/Reshape/shape/13bert/encoder/layer_3/attention/self/Reshape/shape/23bert/encoder/layer_3/attention/self/Reshape/shape/3*
T0*

axis *
N
ŗ
+bert/encoder/layer_3/attention/self/ReshapeReshape1bert/encoder/layer_3/attention/self/query/BiasAdd1bert/encoder/layer_3/attention/self/Reshape/shape*
T0*
Tshape0
o
2bert/encoder/layer_3/attention/self/transpose/permConst*%
valueB"             *
dtype0
ą
-bert/encoder/layer_3/attention/self/transpose	Transpose+bert/encoder/layer_3/attention/self/Reshape2bert/encoder/layer_3/attention/self/transpose/perm*
T0*
Tperm0
_
5bert/encoder/layer_3/attention/self/Reshape_1/shape/1Const*
value	B :*
dtype0
_
5bert/encoder/layer_3/attention/self/Reshape_1/shape/2Const*
value	B :*
dtype0
_
5bert/encoder/layer_3/attention/self/Reshape_1/shape/3Const*
value	B : *
dtype0

3bert/encoder/layer_3/attention/self/Reshape_1/shapePackbert/encoder/strided_slice_25bert/encoder/layer_3/attention/self/Reshape_1/shape/15bert/encoder/layer_3/attention/self/Reshape_1/shape/25bert/encoder/layer_3/attention/self/Reshape_1/shape/3*
T0*

axis *
N
ĩ
-bert/encoder/layer_3/attention/self/Reshape_1Reshape/bert/encoder/layer_3/attention/self/key/BiasAdd3bert/encoder/layer_3/attention/self/Reshape_1/shape*
T0*
Tshape0
q
4bert/encoder/layer_3/attention/self/transpose_1/permConst*%
valueB"             *
dtype0
ˇ
/bert/encoder/layer_3/attention/self/transpose_1	Transpose-bert/encoder/layer_3/attention/self/Reshape_14bert/encoder/layer_3/attention/self/transpose_1/perm*
Tperm0*
T0
ŧ
*bert/encoder/layer_3/attention/self/MatMulBatchMatMul-bert/encoder/layer_3/attention/self/transpose/bert/encoder/layer_3/attention/self/transpose_1*
T0*
adj_x( *
adj_y(
V
)bert/encoder/layer_3/attention/self/Mul/yConst*
dtype0*
valueB
 *ķ5>

'bert/encoder/layer_3/attention/self/MulMul*bert/encoder/layer_3/attention/self/MatMul)bert/encoder/layer_3/attention/self/Mul/y*
T0
`
2bert/encoder/layer_3/attention/self/ExpandDims/dimConst*
dtype0*
valueB:

.bert/encoder/layer_3/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_3/attention/self/ExpandDims/dim*

Tdim0*
T0
V
)bert/encoder/layer_3/attention/self/sub/xConst*
valueB
 *  ?*
dtype0

'bert/encoder/layer_3/attention/self/subSub)bert/encoder/layer_3/attention/self/sub/x.bert/encoder/layer_3/attention/self/ExpandDims*
T0
X
+bert/encoder/layer_3/attention/self/mul_1/yConst*
valueB
 * @Æ*
dtype0

)bert/encoder/layer_3/attention/self/mul_1Mul'bert/encoder/layer_3/attention/self/sub+bert/encoder/layer_3/attention/self/mul_1/y*
T0

'bert/encoder/layer_3/attention/self/addAdd'bert/encoder/layer_3/attention/self/Mul)bert/encoder/layer_3/attention/self/mul_1*
T0
h
+bert/encoder/layer_3/attention/self/SoftmaxSoftmax'bert/encoder/layer_3/attention/self/add*
T0
b
5bert/encoder/layer_3/attention/self/dropout/keep_probConst*
valueB
 *fff?*
dtype0

1bert/encoder/layer_3/attention/self/dropout/ShapeShape+bert/encoder/layer_3/attention/self/Softmax*
T0*
out_type0
k
>bert/encoder/layer_3/attention/self/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
k
>bert/encoder/layer_3/attention/self/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0
ģ
Hbert/encoder/layer_3/attention/self/dropout/random_uniform/RandomUniformRandomUniform1bert/encoder/layer_3/attention/self/dropout/Shape*
T0*
dtype0*
seed2 *

seed 
Î
>bert/encoder/layer_3/attention/self/dropout/random_uniform/subSub>bert/encoder/layer_3/attention/self/dropout/random_uniform/max>bert/encoder/layer_3/attention/self/dropout/random_uniform/min*
T0
Ø
>bert/encoder/layer_3/attention/self/dropout/random_uniform/mulMulHbert/encoder/layer_3/attention/self/dropout/random_uniform/RandomUniform>bert/encoder/layer_3/attention/self/dropout/random_uniform/sub*
T0
Ę
:bert/encoder/layer_3/attention/self/dropout/random_uniformAdd>bert/encoder/layer_3/attention/self/dropout/random_uniform/mul>bert/encoder/layer_3/attention/self/dropout/random_uniform/min*
T0
˛
/bert/encoder/layer_3/attention/self/dropout/addAdd5bert/encoder/layer_3/attention/self/dropout/keep_prob:bert/encoder/layer_3/attention/self/dropout/random_uniform*
T0
t
1bert/encoder/layer_3/attention/self/dropout/FloorFloor/bert/encoder/layer_3/attention/self/dropout/add*
T0
§
/bert/encoder/layer_3/attention/self/dropout/divRealDiv+bert/encoder/layer_3/attention/self/Softmax5bert/encoder/layer_3/attention/self/dropout/keep_prob*
T0
Ŗ
/bert/encoder/layer_3/attention/self/dropout/mulMul/bert/encoder/layer_3/attention/self/dropout/div1bert/encoder/layer_3/attention/self/dropout/Floor*
T0
_
5bert/encoder/layer_3/attention/self/Reshape_2/shape/1Const*
dtype0*
value	B :
_
5bert/encoder/layer_3/attention/self/Reshape_2/shape/2Const*
value	B :*
dtype0
_
5bert/encoder/layer_3/attention/self/Reshape_2/shape/3Const*
value	B : *
dtype0

3bert/encoder/layer_3/attention/self/Reshape_2/shapePackbert/encoder/strided_slice_25bert/encoder/layer_3/attention/self/Reshape_2/shape/15bert/encoder/layer_3/attention/self/Reshape_2/shape/25bert/encoder/layer_3/attention/self/Reshape_2/shape/3*
T0*

axis *
N
ˇ
-bert/encoder/layer_3/attention/self/Reshape_2Reshape1bert/encoder/layer_3/attention/self/value/BiasAdd3bert/encoder/layer_3/attention/self/Reshape_2/shape*
Tshape0*
T0
q
4bert/encoder/layer_3/attention/self/transpose_2/permConst*%
valueB"             *
dtype0
ˇ
/bert/encoder/layer_3/attention/self/transpose_2	Transpose-bert/encoder/layer_3/attention/self/Reshape_24bert/encoder/layer_3/attention/self/transpose_2/perm*
Tperm0*
T0
Ā
,bert/encoder/layer_3/attention/self/MatMul_1BatchMatMul/bert/encoder/layer_3/attention/self/dropout/mul/bert/encoder/layer_3/attention/self/transpose_2*
adj_x( *
adj_y( *
T0
q
4bert/encoder/layer_3/attention/self/transpose_3/permConst*%
valueB"             *
dtype0
ļ
/bert/encoder/layer_3/attention/self/transpose_3	Transpose,bert/encoder/layer_3/attention/self/MatMul_14bert/encoder/layer_3/attention/self/transpose_3/perm*
T0*
Tperm0
U
+bert/encoder/layer_3/attention/self/mul_2/yConst*
value	B :*
dtype0

)bert/encoder/layer_3/attention/self/mul_2Mulbert/encoder/strided_slice_2+bert/encoder/layer_3/attention/self/mul_2/y*
T0
`
5bert/encoder/layer_3/attention/self/Reshape_3/shape/1Const*
value
B :*
dtype0
ģ
3bert/encoder/layer_3/attention/self/Reshape_3/shapePack)bert/encoder/layer_3/attention/self/mul_25bert/encoder/layer_3/attention/self/Reshape_3/shape/1*
T0*

axis *
N
ĩ
-bert/encoder/layer_3/attention/self/Reshape_3Reshape/bert/encoder/layer_3/attention/self/transpose_33bert/encoder/layer_3/attention/self/Reshape_3/shape*
T0*
Tshape0
Ö
Hmio_variable/bert/encoder/layer_3/attention/output/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42bert/encoder/layer_3/attention/output/dense/kernel*
shape:

Ö
Hmio_variable/bert/encoder/layer_3/attention/output/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*A
	container42bert/encoder/layer_3/attention/output/dense/kernel
Z
%Initializer_59/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_59/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_59/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_59/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_59/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 

#Initializer_59/truncated_normal/mulMul/Initializer_59/truncated_normal/TruncatedNormal&Initializer_59/truncated_normal/stddev*
T0
z
Initializer_59/truncated_normalAdd#Initializer_59/truncated_normal/mul$Initializer_59/truncated_normal/mean*
T0

	Assign_59AssignHmio_variable/bert/encoder/layer_3/attention/output/dense/kernel/gradientInitializer_59/truncated_normal*
use_locking(*
T0*[
_classQ
OMloc:@mio_variable/bert/encoder/layer_3/attention/output/dense/kernel/gradient*
validate_shape(
Í
Fmio_variable/bert/encoder/layer_3/attention/output/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*?
	container20bert/encoder/layer_3/attention/output/dense/bias
Í
Fmio_variable/bert/encoder/layer_3/attention/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_3/attention/output/dense/bias*
shape:
F
Initializer_60/zerosConst*
valueB*    *
dtype0
ū
	Assign_60AssignFmio_variable/bert/encoder/layer_3/attention/output/dense/bias/gradientInitializer_60/zeros*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_3/attention/output/dense/bias/gradient*
validate_shape(
ä
2bert/encoder/layer_3/attention/output/dense/MatMulMatMul-bert/encoder/layer_3/attention/self/Reshape_3Hmio_variable/bert/encoder/layer_3/attention/output/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Ú
3bert/encoder/layer_3/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_3/attention/output/dense/MatMulFmio_variable/bert/encoder/layer_3/attention/output/dense/bias/variable*
T0*
data_formatNHWC
d
7bert/encoder/layer_3/attention/output/dropout/keep_probConst*
valueB
 *fff?*
dtype0

3bert/encoder/layer_3/attention/output/dropout/ShapeShape3bert/encoder/layer_3/attention/output/dense/BiasAdd*
T0*
out_type0
m
@bert/encoder/layer_3/attention/output/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
m
@bert/encoder/layer_3/attention/output/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0
ŋ
Jbert/encoder/layer_3/attention/output/dropout/random_uniform/RandomUniformRandomUniform3bert/encoder/layer_3/attention/output/dropout/Shape*
T0*
dtype0*
seed2 *

seed 
Ô
@bert/encoder/layer_3/attention/output/dropout/random_uniform/subSub@bert/encoder/layer_3/attention/output/dropout/random_uniform/max@bert/encoder/layer_3/attention/output/dropout/random_uniform/min*
T0
Ū
@bert/encoder/layer_3/attention/output/dropout/random_uniform/mulMulJbert/encoder/layer_3/attention/output/dropout/random_uniform/RandomUniform@bert/encoder/layer_3/attention/output/dropout/random_uniform/sub*
T0
Đ
<bert/encoder/layer_3/attention/output/dropout/random_uniformAdd@bert/encoder/layer_3/attention/output/dropout/random_uniform/mul@bert/encoder/layer_3/attention/output/dropout/random_uniform/min*
T0
¸
1bert/encoder/layer_3/attention/output/dropout/addAdd7bert/encoder/layer_3/attention/output/dropout/keep_prob<bert/encoder/layer_3/attention/output/dropout/random_uniform*
T0
x
3bert/encoder/layer_3/attention/output/dropout/FloorFloor1bert/encoder/layer_3/attention/output/dropout/add*
T0
ŗ
1bert/encoder/layer_3/attention/output/dropout/divRealDiv3bert/encoder/layer_3/attention/output/dense/BiasAdd7bert/encoder/layer_3/attention/output/dropout/keep_prob*
T0
Š
1bert/encoder/layer_3/attention/output/dropout/mulMul1bert/encoder/layer_3/attention/output/dropout/div3bert/encoder/layer_3/attention/output/dropout/Floor*
T0
Ŗ
)bert/encoder/layer_3/attention/output/addAdd1bert/encoder/layer_3/attention/output/dropout/mul5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1*
T0
Õ
Jmio_variable/bert/encoder/layer_3/attention/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64bert/encoder/layer_3/attention/output/LayerNorm/beta*
shape:
Õ
Jmio_variable/bert/encoder/layer_3/attention/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64bert/encoder/layer_3/attention/output/LayerNorm/beta*
shape:
F
Initializer_61/zerosConst*
valueB*    *
dtype0

	Assign_61AssignJmio_variable/bert/encoder/layer_3/attention/output/LayerNorm/beta/gradientInitializer_61/zeros*
use_locking(*
T0*]
_classS
QOloc:@mio_variable/bert/encoder/layer_3/attention/output/LayerNorm/beta/gradient*
validate_shape(
×
Kmio_variable/bert/encoder/layer_3/attention/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*D
	container75bert/encoder/layer_3/attention/output/LayerNorm/gamma
×
Kmio_variable/bert/encoder/layer_3/attention/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*D
	container75bert/encoder/layer_3/attention/output/LayerNorm/gamma*
shape:
E
Initializer_62/onesConst*
valueB*  ?*
dtype0

	Assign_62AssignKmio_variable/bert/encoder/layer_3/attention/output/LayerNorm/gamma/gradientInitializer_62/ones*
use_locking(*
T0*^
_classT
RPloc:@mio_variable/bert/encoder/layer_3/attention/output/LayerNorm/gamma/gradient*
validate_shape(
|
Nbert/encoder/layer_3/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0
å
<bert/encoder/layer_3/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_3/attention/output/addNbert/encoder/layer_3/attention/output/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0

Dbert/encoder/layer_3/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_3/attention/output/LayerNorm/moments/mean*
T0
Ø
Ibert/encoder/layer_3/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_3/attention/output/addDbert/encoder/layer_3/attention/output/LayerNorm/moments/StopGradient*
T0

Rbert/encoder/layer_3/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0

@bert/encoder/layer_3/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_3/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_3/attention/output/LayerNorm/moments/variance/reduction_indices*
T0*

Tidx0*
	keep_dims(
l
?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ėŧ+*
dtype0
Đ
=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_3/attention/output/LayerNorm/moments/variance?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add/y*
T0

?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add*
T0
Û
=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/RsqrtKmio_variable/bert/encoder/layer_3/attention/output/LayerNorm/gamma/variable*
T0
š
?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_3/attention/output/add=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul*
T0
Ė
?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_3/attention/output/LayerNorm/moments/mean=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul*
T0
Ú
=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/subSubJmio_variable/bert/encoder/layer_3/attention/output/LayerNorm/beta/variable?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul_2*
T0
Ī
?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/sub*
T0
Î
Dmio_variable/bert/encoder/layer_3/intermediate/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_3/intermediate/dense/kernel*
shape:

Î
Dmio_variable/bert/encoder/layer_3/intermediate/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*=
	container0.bert/encoder/layer_3/intermediate/dense/kernel
Z
%Initializer_63/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_63/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_63/truncated_normal/stddevConst*
dtype0*
valueB
 *
×Ŗ<

/Initializer_63/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_63/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0

#Initializer_63/truncated_normal/mulMul/Initializer_63/truncated_normal/TruncatedNormal&Initializer_63/truncated_normal/stddev*
T0
z
Initializer_63/truncated_normalAdd#Initializer_63/truncated_normal/mul$Initializer_63/truncated_normal/mean*
T0

	Assign_63AssignDmio_variable/bert/encoder/layer_3/intermediate/dense/kernel/gradientInitializer_63/truncated_normal*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_3/intermediate/dense/kernel/gradient*
validate_shape(
Å
Bmio_variable/bert/encoder/layer_3/intermediate/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*;
	container.,bert/encoder/layer_3/intermediate/dense/bias
Å
Bmio_variable/bert/encoder/layer_3/intermediate/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_3/intermediate/dense/bias*
shape:
S
$Initializer_64/zeros/shape_as_tensorConst*
valueB:*
dtype0
G
Initializer_64/zeros/ConstConst*
valueB
 *    *
dtype0
y
Initializer_64/zerosFill$Initializer_64/zeros/shape_as_tensorInitializer_64/zeros/Const*
T0*

index_type0
ö
	Assign_64AssignBmio_variable/bert/encoder/layer_3/intermediate/dense/bias/gradientInitializer_64/zeros*U
_classK
IGloc:@mio_variable/bert/encoder/layer_3/intermediate/dense/bias/gradient*
validate_shape(*
use_locking(*
T0
î
.bert/encoder/layer_3/intermediate/dense/MatMulMatMul?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add_1Dmio_variable/bert/encoder/layer_3/intermediate/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Î
/bert/encoder/layer_3/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_3/intermediate/dense/MatMulBmio_variable/bert/encoder/layer_3/intermediate/dense/bias/variable*
data_formatNHWC*
T0
Z
-bert/encoder/layer_3/intermediate/dense/Pow/yConst*
valueB
 *  @@*
dtype0

+bert/encoder/layer_3/intermediate/dense/PowPow/bert/encoder/layer_3/intermediate/dense/BiasAdd-bert/encoder/layer_3/intermediate/dense/Pow/y*
T0
Z
-bert/encoder/layer_3/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0

+bert/encoder/layer_3/intermediate/dense/mulMul-bert/encoder/layer_3/intermediate/dense/mul/x+bert/encoder/layer_3/intermediate/dense/Pow*
T0

+bert/encoder/layer_3/intermediate/dense/addAdd/bert/encoder/layer_3/intermediate/dense/BiasAdd+bert/encoder/layer_3/intermediate/dense/mul*
T0
\
/bert/encoder/layer_3/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0

-bert/encoder/layer_3/intermediate/dense/mul_1Mul/bert/encoder/layer_3/intermediate/dense/mul_1/x+bert/encoder/layer_3/intermediate/dense/add*
T0
l
,bert/encoder/layer_3/intermediate/dense/TanhTanh-bert/encoder/layer_3/intermediate/dense/mul_1*
T0
\
/bert/encoder/layer_3/intermediate/dense/add_1/xConst*
valueB
 *  ?*
dtype0

-bert/encoder/layer_3/intermediate/dense/add_1Add/bert/encoder/layer_3/intermediate/dense/add_1/x,bert/encoder/layer_3/intermediate/dense/Tanh*
T0
\
/bert/encoder/layer_3/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0

-bert/encoder/layer_3/intermediate/dense/mul_2Mul/bert/encoder/layer_3/intermediate/dense/mul_2/x-bert/encoder/layer_3/intermediate/dense/add_1*
T0

-bert/encoder/layer_3/intermediate/dense/mul_3Mul/bert/encoder/layer_3/intermediate/dense/BiasAdd-bert/encoder/layer_3/intermediate/dense/mul_2*
T0
Â
>mio_variable/bert/encoder/layer_3/output/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(bert/encoder/layer_3/output/dense/kernel*
shape:

Â
>mio_variable/bert/encoder/layer_3/output/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(bert/encoder/layer_3/output/dense/kernel*
shape:

Z
%Initializer_65/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_65/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_65/truncated_normal/stddevConst*
dtype0*
valueB
 *
×Ŗ<

/Initializer_65/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_65/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

#Initializer_65/truncated_normal/mulMul/Initializer_65/truncated_normal/TruncatedNormal&Initializer_65/truncated_normal/stddev*
T0
z
Initializer_65/truncated_normalAdd#Initializer_65/truncated_normal/mul$Initializer_65/truncated_normal/mean*
T0
ų
	Assign_65Assign>mio_variable/bert/encoder/layer_3/output/dense/kernel/gradientInitializer_65/truncated_normal*
use_locking(*
T0*Q
_classG
ECloc:@mio_variable/bert/encoder/layer_3/output/dense/kernel/gradient*
validate_shape(
š
<mio_variable/bert/encoder/layer_3/output/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&bert/encoder/layer_3/output/dense/bias*
shape:
š
<mio_variable/bert/encoder/layer_3/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*5
	container(&bert/encoder/layer_3/output/dense/bias
F
Initializer_66/zerosConst*
valueB*    *
dtype0
ę
	Assign_66Assign<mio_variable/bert/encoder/layer_3/output/dense/bias/gradientInitializer_66/zeros*
use_locking(*
T0*O
_classE
CAloc:@mio_variable/bert/encoder/layer_3/output/dense/bias/gradient*
validate_shape(
Đ
(bert/encoder/layer_3/output/dense/MatMulMatMul-bert/encoder/layer_3/intermediate/dense/mul_3>mio_variable/bert/encoder/layer_3/output/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
ŧ
)bert/encoder/layer_3/output/dense/BiasAddBiasAdd(bert/encoder/layer_3/output/dense/MatMul<mio_variable/bert/encoder/layer_3/output/dense/bias/variable*
data_formatNHWC*
T0
Z
-bert/encoder/layer_3/output/dropout/keep_probConst*
valueB
 *fff?*
dtype0
v
)bert/encoder/layer_3/output/dropout/ShapeShape)bert/encoder/layer_3/output/dense/BiasAdd*
T0*
out_type0
c
6bert/encoder/layer_3/output/dropout/random_uniform/minConst*
dtype0*
valueB
 *    
c
6bert/encoder/layer_3/output/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0
Ģ
@bert/encoder/layer_3/output/dropout/random_uniform/RandomUniformRandomUniform)bert/encoder/layer_3/output/dropout/Shape*
T0*
dtype0*
seed2 *

seed 
ļ
6bert/encoder/layer_3/output/dropout/random_uniform/subSub6bert/encoder/layer_3/output/dropout/random_uniform/max6bert/encoder/layer_3/output/dropout/random_uniform/min*
T0
Ā
6bert/encoder/layer_3/output/dropout/random_uniform/mulMul@bert/encoder/layer_3/output/dropout/random_uniform/RandomUniform6bert/encoder/layer_3/output/dropout/random_uniform/sub*
T0
˛
2bert/encoder/layer_3/output/dropout/random_uniformAdd6bert/encoder/layer_3/output/dropout/random_uniform/mul6bert/encoder/layer_3/output/dropout/random_uniform/min*
T0

'bert/encoder/layer_3/output/dropout/addAdd-bert/encoder/layer_3/output/dropout/keep_prob2bert/encoder/layer_3/output/dropout/random_uniform*
T0
d
)bert/encoder/layer_3/output/dropout/FloorFloor'bert/encoder/layer_3/output/dropout/add*
T0

'bert/encoder/layer_3/output/dropout/divRealDiv)bert/encoder/layer_3/output/dense/BiasAdd-bert/encoder/layer_3/output/dropout/keep_prob*
T0

'bert/encoder/layer_3/output/dropout/mulMul'bert/encoder/layer_3/output/dropout/div)bert/encoder/layer_3/output/dropout/Floor*
T0

bert/encoder/layer_3/output/addAdd'bert/encoder/layer_3/output/dropout/mul?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add_1*
T0
Á
@mio_variable/bert/encoder/layer_3/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*bert/encoder/layer_3/output/LayerNorm/beta*
shape:
Á
@mio_variable/bert/encoder/layer_3/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*9
	container,*bert/encoder/layer_3/output/LayerNorm/beta
F
Initializer_67/zerosConst*
valueB*    *
dtype0
ō
	Assign_67Assign@mio_variable/bert/encoder/layer_3/output/LayerNorm/beta/gradientInitializer_67/zeros*
use_locking(*
T0*S
_classI
GEloc:@mio_variable/bert/encoder/layer_3/output/LayerNorm/beta/gradient*
validate_shape(
Ã
Amio_variable/bert/encoder/layer_3/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+bert/encoder/layer_3/output/LayerNorm/gamma*
shape:
Ã
Amio_variable/bert/encoder/layer_3/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*:
	container-+bert/encoder/layer_3/output/LayerNorm/gamma
E
Initializer_68/onesConst*
dtype0*
valueB*  ?
ķ
	Assign_68AssignAmio_variable/bert/encoder/layer_3/output/LayerNorm/gamma/gradientInitializer_68/ones*T
_classJ
HFloc:@mio_variable/bert/encoder/layer_3/output/LayerNorm/gamma/gradient*
validate_shape(*
use_locking(*
T0
r
Dbert/encoder/layer_3/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0
Į
2bert/encoder/layer_3/output/LayerNorm/moments/meanMeanbert/encoder/layer_3/output/addDbert/encoder/layer_3/output/LayerNorm/moments/mean/reduction_indices*
T0*

Tidx0*
	keep_dims(

:bert/encoder/layer_3/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_3/output/LayerNorm/moments/mean*
T0
ē
?bert/encoder/layer_3/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_3/output/add:bert/encoder/layer_3/output/LayerNorm/moments/StopGradient*
T0
v
Hbert/encoder/layer_3/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0
ī
6bert/encoder/layer_3/output/LayerNorm/moments/varianceMean?bert/encoder/layer_3/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_3/output/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0
b
5bert/encoder/layer_3/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ėŧ+*
dtype0
˛
3bert/encoder/layer_3/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_3/output/LayerNorm/moments/variance5bert/encoder/layer_3/output/LayerNorm/batchnorm/add/y*
T0
|
5bert/encoder/layer_3/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_3/output/LayerNorm/batchnorm/add*
T0
Ŋ
3bert/encoder/layer_3/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_3/output/LayerNorm/batchnorm/RsqrtAmio_variable/bert/encoder/layer_3/output/LayerNorm/gamma/variable*
T0

5bert/encoder/layer_3/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_3/output/add3bert/encoder/layer_3/output/LayerNorm/batchnorm/mul*
T0
Ž
5bert/encoder/layer_3/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_3/output/LayerNorm/moments/mean3bert/encoder/layer_3/output/LayerNorm/batchnorm/mul*
T0
ŧ
3bert/encoder/layer_3/output/LayerNorm/batchnorm/subSub@mio_variable/bert/encoder/layer_3/output/LayerNorm/beta/variable5bert/encoder/layer_3/output/LayerNorm/batchnorm/mul_2*
T0
ą
5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_3/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_3/output/LayerNorm/batchnorm/sub*
T0

)bert/encoder/layer_4/attention/self/ShapeShape5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
e
7bert/encoder/layer_4/attention/self/strided_slice/stackConst*
valueB: *
dtype0
g
9bert/encoder/layer_4/attention/self/strided_slice/stack_1Const*
valueB:*
dtype0
g
9bert/encoder/layer_4/attention/self/strided_slice/stack_2Const*
valueB:*
dtype0

1bert/encoder/layer_4/attention/self/strided_sliceStridedSlice)bert/encoder/layer_4/attention/self/Shape7bert/encoder/layer_4/attention/self/strided_slice/stack9bert/encoder/layer_4/attention/self/strided_slice/stack_19bert/encoder/layer_4/attention/self/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0

+bert/encoder/layer_4/attention/self/Shape_1Shape5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
g
9bert/encoder/layer_4/attention/self/strided_slice_1/stackConst*
valueB: *
dtype0
i
;bert/encoder/layer_4/attention/self/strided_slice_1/stack_1Const*
valueB:*
dtype0
i
;bert/encoder/layer_4/attention/self/strided_slice_1/stack_2Const*
valueB:*
dtype0

3bert/encoder/layer_4/attention/self/strided_slice_1StridedSlice+bert/encoder/layer_4/attention/self/Shape_19bert/encoder/layer_4/attention/self/strided_slice_1/stack;bert/encoder/layer_4/attention/self/strided_slice_1/stack_1;bert/encoder/layer_4/attention/self/strided_slice_1/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
Ō
Fmio_variable/bert/encoder/layer_4/attention/self/query/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_4/attention/self/query/kernel*
shape:

Ō
Fmio_variable/bert/encoder/layer_4/attention/self/query/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_4/attention/self/query/kernel*
shape:

Z
%Initializer_69/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_69/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_69/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_69/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_69/truncated_normal/shape*
seed2 *

seed *
T0*
dtype0

#Initializer_69/truncated_normal/mulMul/Initializer_69/truncated_normal/TruncatedNormal&Initializer_69/truncated_normal/stddev*
T0
z
Initializer_69/truncated_normalAdd#Initializer_69/truncated_normal/mul$Initializer_69/truncated_normal/mean*
T0

	Assign_69AssignFmio_variable/bert/encoder/layer_4/attention/self/query/kernel/gradientInitializer_69/truncated_normal*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_4/attention/self/query/kernel/gradient*
validate_shape(
É
Dmio_variable/bert/encoder/layer_4/attention/self/query/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_4/attention/self/query/bias*
shape:
É
Dmio_variable/bert/encoder/layer_4/attention/self/query/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_4/attention/self/query/bias*
shape:
F
Initializer_70/zerosConst*
valueB*    *
dtype0
ú
	Assign_70AssignDmio_variable/bert/encoder/layer_4/attention/self/query/bias/gradientInitializer_70/zeros*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_4/attention/self/query/bias/gradient*
validate_shape(
č
0bert/encoder/layer_4/attention/self/query/MatMulMatMul5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1Fmio_variable/bert/encoder/layer_4/attention/self/query/kernel/variable*
transpose_a( *
transpose_b( *
T0
Ô
1bert/encoder/layer_4/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_4/attention/self/query/MatMulDmio_variable/bert/encoder/layer_4/attention/self/query/bias/variable*
T0*
data_formatNHWC
Î
Dmio_variable/bert/encoder/layer_4/attention/self/key/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_4/attention/self/key/kernel*
shape:

Î
Dmio_variable/bert/encoder/layer_4/attention/self/key/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_4/attention/self/key/kernel*
shape:

Z
%Initializer_71/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_71/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_71/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_71/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_71/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 

#Initializer_71/truncated_normal/mulMul/Initializer_71/truncated_normal/TruncatedNormal&Initializer_71/truncated_normal/stddev*
T0
z
Initializer_71/truncated_normalAdd#Initializer_71/truncated_normal/mul$Initializer_71/truncated_normal/mean*
T0

	Assign_71AssignDmio_variable/bert/encoder/layer_4/attention/self/key/kernel/gradientInitializer_71/truncated_normal*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_4/attention/self/key/kernel/gradient*
validate_shape(
Å
Bmio_variable/bert/encoder/layer_4/attention/self/key/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_4/attention/self/key/bias*
shape:
Å
Bmio_variable/bert/encoder/layer_4/attention/self/key/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_4/attention/self/key/bias*
shape:
F
Initializer_72/zerosConst*
valueB*    *
dtype0
ö
	Assign_72AssignBmio_variable/bert/encoder/layer_4/attention/self/key/bias/gradientInitializer_72/zeros*
use_locking(*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_4/attention/self/key/bias/gradient*
validate_shape(
ä
.bert/encoder/layer_4/attention/self/key/MatMulMatMul5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1Dmio_variable/bert/encoder/layer_4/attention/self/key/kernel/variable*
transpose_a( *
transpose_b( *
T0
Î
/bert/encoder/layer_4/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_4/attention/self/key/MatMulBmio_variable/bert/encoder/layer_4/attention/self/key/bias/variable*
T0*
data_formatNHWC
Ō
Fmio_variable/bert/encoder/layer_4/attention/self/value/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_4/attention/self/value/kernel*
shape:

Ō
Fmio_variable/bert/encoder/layer_4/attention/self/value/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_4/attention/self/value/kernel*
shape:

Z
%Initializer_73/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_73/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_73/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_73/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_73/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0

#Initializer_73/truncated_normal/mulMul/Initializer_73/truncated_normal/TruncatedNormal&Initializer_73/truncated_normal/stddev*
T0
z
Initializer_73/truncated_normalAdd#Initializer_73/truncated_normal/mul$Initializer_73/truncated_normal/mean*
T0

	Assign_73AssignFmio_variable/bert/encoder/layer_4/attention/self/value/kernel/gradientInitializer_73/truncated_normal*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_4/attention/self/value/kernel/gradient*
validate_shape(
É
Dmio_variable/bert/encoder/layer_4/attention/self/value/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*=
	container0.bert/encoder/layer_4/attention/self/value/bias
É
Dmio_variable/bert/encoder/layer_4/attention/self/value/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*=
	container0.bert/encoder/layer_4/attention/self/value/bias
F
Initializer_74/zerosConst*
dtype0*
valueB*    
ú
	Assign_74AssignDmio_variable/bert/encoder/layer_4/attention/self/value/bias/gradientInitializer_74/zeros*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_4/attention/self/value/bias/gradient*
validate_shape(
č
0bert/encoder/layer_4/attention/self/value/MatMulMatMul5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1Fmio_variable/bert/encoder/layer_4/attention/self/value/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Ô
1bert/encoder/layer_4/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_4/attention/self/value/MatMulDmio_variable/bert/encoder/layer_4/attention/self/value/bias/variable*
T0*
data_formatNHWC
]
3bert/encoder/layer_4/attention/self/Reshape/shape/1Const*
dtype0*
value	B :
]
3bert/encoder/layer_4/attention/self/Reshape/shape/2Const*
value	B :*
dtype0
]
3bert/encoder/layer_4/attention/self/Reshape/shape/3Const*
value	B : *
dtype0

1bert/encoder/layer_4/attention/self/Reshape/shapePackbert/encoder/strided_slice_23bert/encoder/layer_4/attention/self/Reshape/shape/13bert/encoder/layer_4/attention/self/Reshape/shape/23bert/encoder/layer_4/attention/self/Reshape/shape/3*
T0*

axis *
N
ŗ
+bert/encoder/layer_4/attention/self/ReshapeReshape1bert/encoder/layer_4/attention/self/query/BiasAdd1bert/encoder/layer_4/attention/self/Reshape/shape*
T0*
Tshape0
o
2bert/encoder/layer_4/attention/self/transpose/permConst*%
valueB"             *
dtype0
ą
-bert/encoder/layer_4/attention/self/transpose	Transpose+bert/encoder/layer_4/attention/self/Reshape2bert/encoder/layer_4/attention/self/transpose/perm*
T0*
Tperm0
_
5bert/encoder/layer_4/attention/self/Reshape_1/shape/1Const*
value	B :*
dtype0
_
5bert/encoder/layer_4/attention/self/Reshape_1/shape/2Const*
value	B :*
dtype0
_
5bert/encoder/layer_4/attention/self/Reshape_1/shape/3Const*
value	B : *
dtype0

3bert/encoder/layer_4/attention/self/Reshape_1/shapePackbert/encoder/strided_slice_25bert/encoder/layer_4/attention/self/Reshape_1/shape/15bert/encoder/layer_4/attention/self/Reshape_1/shape/25bert/encoder/layer_4/attention/self/Reshape_1/shape/3*
T0*

axis *
N
ĩ
-bert/encoder/layer_4/attention/self/Reshape_1Reshape/bert/encoder/layer_4/attention/self/key/BiasAdd3bert/encoder/layer_4/attention/self/Reshape_1/shape*
T0*
Tshape0
q
4bert/encoder/layer_4/attention/self/transpose_1/permConst*%
valueB"             *
dtype0
ˇ
/bert/encoder/layer_4/attention/self/transpose_1	Transpose-bert/encoder/layer_4/attention/self/Reshape_14bert/encoder/layer_4/attention/self/transpose_1/perm*
T0*
Tperm0
ŧ
*bert/encoder/layer_4/attention/self/MatMulBatchMatMul-bert/encoder/layer_4/attention/self/transpose/bert/encoder/layer_4/attention/self/transpose_1*
adj_x( *
adj_y(*
T0
V
)bert/encoder/layer_4/attention/self/Mul/yConst*
valueB
 *ķ5>*
dtype0

'bert/encoder/layer_4/attention/self/MulMul*bert/encoder/layer_4/attention/self/MatMul)bert/encoder/layer_4/attention/self/Mul/y*
T0
`
2bert/encoder/layer_4/attention/self/ExpandDims/dimConst*
dtype0*
valueB:

.bert/encoder/layer_4/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_4/attention/self/ExpandDims/dim*

Tdim0*
T0
V
)bert/encoder/layer_4/attention/self/sub/xConst*
valueB
 *  ?*
dtype0

'bert/encoder/layer_4/attention/self/subSub)bert/encoder/layer_4/attention/self/sub/x.bert/encoder/layer_4/attention/self/ExpandDims*
T0
X
+bert/encoder/layer_4/attention/self/mul_1/yConst*
dtype0*
valueB
 * @Æ

)bert/encoder/layer_4/attention/self/mul_1Mul'bert/encoder/layer_4/attention/self/sub+bert/encoder/layer_4/attention/self/mul_1/y*
T0

'bert/encoder/layer_4/attention/self/addAdd'bert/encoder/layer_4/attention/self/Mul)bert/encoder/layer_4/attention/self/mul_1*
T0
h
+bert/encoder/layer_4/attention/self/SoftmaxSoftmax'bert/encoder/layer_4/attention/self/add*
T0
b
5bert/encoder/layer_4/attention/self/dropout/keep_probConst*
dtype0*
valueB
 *fff?

1bert/encoder/layer_4/attention/self/dropout/ShapeShape+bert/encoder/layer_4/attention/self/Softmax*
T0*
out_type0
k
>bert/encoder/layer_4/attention/self/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
k
>bert/encoder/layer_4/attention/self/dropout/random_uniform/maxConst*
dtype0*
valueB
 *  ?
ģ
Hbert/encoder/layer_4/attention/self/dropout/random_uniform/RandomUniformRandomUniform1bert/encoder/layer_4/attention/self/dropout/Shape*
T0*
dtype0*
seed2 *

seed 
Î
>bert/encoder/layer_4/attention/self/dropout/random_uniform/subSub>bert/encoder/layer_4/attention/self/dropout/random_uniform/max>bert/encoder/layer_4/attention/self/dropout/random_uniform/min*
T0
Ø
>bert/encoder/layer_4/attention/self/dropout/random_uniform/mulMulHbert/encoder/layer_4/attention/self/dropout/random_uniform/RandomUniform>bert/encoder/layer_4/attention/self/dropout/random_uniform/sub*
T0
Ę
:bert/encoder/layer_4/attention/self/dropout/random_uniformAdd>bert/encoder/layer_4/attention/self/dropout/random_uniform/mul>bert/encoder/layer_4/attention/self/dropout/random_uniform/min*
T0
˛
/bert/encoder/layer_4/attention/self/dropout/addAdd5bert/encoder/layer_4/attention/self/dropout/keep_prob:bert/encoder/layer_4/attention/self/dropout/random_uniform*
T0
t
1bert/encoder/layer_4/attention/self/dropout/FloorFloor/bert/encoder/layer_4/attention/self/dropout/add*
T0
§
/bert/encoder/layer_4/attention/self/dropout/divRealDiv+bert/encoder/layer_4/attention/self/Softmax5bert/encoder/layer_4/attention/self/dropout/keep_prob*
T0
Ŗ
/bert/encoder/layer_4/attention/self/dropout/mulMul/bert/encoder/layer_4/attention/self/dropout/div1bert/encoder/layer_4/attention/self/dropout/Floor*
T0
_
5bert/encoder/layer_4/attention/self/Reshape_2/shape/1Const*
value	B :*
dtype0
_
5bert/encoder/layer_4/attention/self/Reshape_2/shape/2Const*
value	B :*
dtype0
_
5bert/encoder/layer_4/attention/self/Reshape_2/shape/3Const*
value	B : *
dtype0

3bert/encoder/layer_4/attention/self/Reshape_2/shapePackbert/encoder/strided_slice_25bert/encoder/layer_4/attention/self/Reshape_2/shape/15bert/encoder/layer_4/attention/self/Reshape_2/shape/25bert/encoder/layer_4/attention/self/Reshape_2/shape/3*
T0*

axis *
N
ˇ
-bert/encoder/layer_4/attention/self/Reshape_2Reshape1bert/encoder/layer_4/attention/self/value/BiasAdd3bert/encoder/layer_4/attention/self/Reshape_2/shape*
T0*
Tshape0
q
4bert/encoder/layer_4/attention/self/transpose_2/permConst*%
valueB"             *
dtype0
ˇ
/bert/encoder/layer_4/attention/self/transpose_2	Transpose-bert/encoder/layer_4/attention/self/Reshape_24bert/encoder/layer_4/attention/self/transpose_2/perm*
Tperm0*
T0
Ā
,bert/encoder/layer_4/attention/self/MatMul_1BatchMatMul/bert/encoder/layer_4/attention/self/dropout/mul/bert/encoder/layer_4/attention/self/transpose_2*
T0*
adj_x( *
adj_y( 
q
4bert/encoder/layer_4/attention/self/transpose_3/permConst*%
valueB"             *
dtype0
ļ
/bert/encoder/layer_4/attention/self/transpose_3	Transpose,bert/encoder/layer_4/attention/self/MatMul_14bert/encoder/layer_4/attention/self/transpose_3/perm*
T0*
Tperm0
U
+bert/encoder/layer_4/attention/self/mul_2/yConst*
dtype0*
value	B :

)bert/encoder/layer_4/attention/self/mul_2Mulbert/encoder/strided_slice_2+bert/encoder/layer_4/attention/self/mul_2/y*
T0
`
5bert/encoder/layer_4/attention/self/Reshape_3/shape/1Const*
value
B :*
dtype0
ģ
3bert/encoder/layer_4/attention/self/Reshape_3/shapePack)bert/encoder/layer_4/attention/self/mul_25bert/encoder/layer_4/attention/self/Reshape_3/shape/1*
T0*

axis *
N
ĩ
-bert/encoder/layer_4/attention/self/Reshape_3Reshape/bert/encoder/layer_4/attention/self/transpose_33bert/encoder/layer_4/attention/self/Reshape_3/shape*
T0*
Tshape0
Ö
Hmio_variable/bert/encoder/layer_4/attention/output/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42bert/encoder/layer_4/attention/output/dense/kernel*
shape:

Ö
Hmio_variable/bert/encoder/layer_4/attention/output/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42bert/encoder/layer_4/attention/output/dense/kernel*
shape:

Z
%Initializer_75/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_75/truncated_normal/meanConst*
dtype0*
valueB
 *    
S
&Initializer_75/truncated_normal/stddevConst*
dtype0*
valueB
 *
×Ŗ<

/Initializer_75/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_75/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

#Initializer_75/truncated_normal/mulMul/Initializer_75/truncated_normal/TruncatedNormal&Initializer_75/truncated_normal/stddev*
T0
z
Initializer_75/truncated_normalAdd#Initializer_75/truncated_normal/mul$Initializer_75/truncated_normal/mean*
T0

	Assign_75AssignHmio_variable/bert/encoder/layer_4/attention/output/dense/kernel/gradientInitializer_75/truncated_normal*
use_locking(*
T0*[
_classQ
OMloc:@mio_variable/bert/encoder/layer_4/attention/output/dense/kernel/gradient*
validate_shape(
Í
Fmio_variable/bert/encoder/layer_4/attention/output/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_4/attention/output/dense/bias*
shape:
Í
Fmio_variable/bert/encoder/layer_4/attention/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*?
	container20bert/encoder/layer_4/attention/output/dense/bias
F
Initializer_76/zerosConst*
valueB*    *
dtype0
ū
	Assign_76AssignFmio_variable/bert/encoder/layer_4/attention/output/dense/bias/gradientInitializer_76/zeros*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_4/attention/output/dense/bias/gradient*
validate_shape(
ä
2bert/encoder/layer_4/attention/output/dense/MatMulMatMul-bert/encoder/layer_4/attention/self/Reshape_3Hmio_variable/bert/encoder/layer_4/attention/output/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Ú
3bert/encoder/layer_4/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_4/attention/output/dense/MatMulFmio_variable/bert/encoder/layer_4/attention/output/dense/bias/variable*
T0*
data_formatNHWC
d
7bert/encoder/layer_4/attention/output/dropout/keep_probConst*
valueB
 *fff?*
dtype0

3bert/encoder/layer_4/attention/output/dropout/ShapeShape3bert/encoder/layer_4/attention/output/dense/BiasAdd*
T0*
out_type0
m
@bert/encoder/layer_4/attention/output/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
m
@bert/encoder/layer_4/attention/output/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0
ŋ
Jbert/encoder/layer_4/attention/output/dropout/random_uniform/RandomUniformRandomUniform3bert/encoder/layer_4/attention/output/dropout/Shape*
dtype0*
seed2 *

seed *
T0
Ô
@bert/encoder/layer_4/attention/output/dropout/random_uniform/subSub@bert/encoder/layer_4/attention/output/dropout/random_uniform/max@bert/encoder/layer_4/attention/output/dropout/random_uniform/min*
T0
Ū
@bert/encoder/layer_4/attention/output/dropout/random_uniform/mulMulJbert/encoder/layer_4/attention/output/dropout/random_uniform/RandomUniform@bert/encoder/layer_4/attention/output/dropout/random_uniform/sub*
T0
Đ
<bert/encoder/layer_4/attention/output/dropout/random_uniformAdd@bert/encoder/layer_4/attention/output/dropout/random_uniform/mul@bert/encoder/layer_4/attention/output/dropout/random_uniform/min*
T0
¸
1bert/encoder/layer_4/attention/output/dropout/addAdd7bert/encoder/layer_4/attention/output/dropout/keep_prob<bert/encoder/layer_4/attention/output/dropout/random_uniform*
T0
x
3bert/encoder/layer_4/attention/output/dropout/FloorFloor1bert/encoder/layer_4/attention/output/dropout/add*
T0
ŗ
1bert/encoder/layer_4/attention/output/dropout/divRealDiv3bert/encoder/layer_4/attention/output/dense/BiasAdd7bert/encoder/layer_4/attention/output/dropout/keep_prob*
T0
Š
1bert/encoder/layer_4/attention/output/dropout/mulMul1bert/encoder/layer_4/attention/output/dropout/div3bert/encoder/layer_4/attention/output/dropout/Floor*
T0
Ŗ
)bert/encoder/layer_4/attention/output/addAdd1bert/encoder/layer_4/attention/output/dropout/mul5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1*
T0
Õ
Jmio_variable/bert/encoder/layer_4/attention/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64bert/encoder/layer_4/attention/output/LayerNorm/beta*
shape:
Õ
Jmio_variable/bert/encoder/layer_4/attention/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64bert/encoder/layer_4/attention/output/LayerNorm/beta*
shape:
F
Initializer_77/zerosConst*
valueB*    *
dtype0

	Assign_77AssignJmio_variable/bert/encoder/layer_4/attention/output/LayerNorm/beta/gradientInitializer_77/zeros*
T0*]
_classS
QOloc:@mio_variable/bert/encoder/layer_4/attention/output/LayerNorm/beta/gradient*
validate_shape(*
use_locking(
×
Kmio_variable/bert/encoder/layer_4/attention/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*D
	container75bert/encoder/layer_4/attention/output/LayerNorm/gamma*
shape:
×
Kmio_variable/bert/encoder/layer_4/attention/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*D
	container75bert/encoder/layer_4/attention/output/LayerNorm/gamma*
shape:
E
Initializer_78/onesConst*
valueB*  ?*
dtype0

	Assign_78AssignKmio_variable/bert/encoder/layer_4/attention/output/LayerNorm/gamma/gradientInitializer_78/ones*
use_locking(*
T0*^
_classT
RPloc:@mio_variable/bert/encoder/layer_4/attention/output/LayerNorm/gamma/gradient*
validate_shape(
|
Nbert/encoder/layer_4/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0
å
<bert/encoder/layer_4/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_4/attention/output/addNbert/encoder/layer_4/attention/output/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0

Dbert/encoder/layer_4/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_4/attention/output/LayerNorm/moments/mean*
T0
Ø
Ibert/encoder/layer_4/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_4/attention/output/addDbert/encoder/layer_4/attention/output/LayerNorm/moments/StopGradient*
T0

Rbert/encoder/layer_4/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0

@bert/encoder/layer_4/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_4/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_4/attention/output/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0
l
?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ėŧ+*
dtype0
Đ
=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_4/attention/output/LayerNorm/moments/variance?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add/y*
T0

?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add*
T0
Û
=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/RsqrtKmio_variable/bert/encoder/layer_4/attention/output/LayerNorm/gamma/variable*
T0
š
?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_4/attention/output/add=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul*
T0
Ė
?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_4/attention/output/LayerNorm/moments/mean=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul*
T0
Ú
=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/subSubJmio_variable/bert/encoder/layer_4/attention/output/LayerNorm/beta/variable?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul_2*
T0
Ī
?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/sub*
T0
Î
Dmio_variable/bert/encoder/layer_4/intermediate/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*=
	container0.bert/encoder/layer_4/intermediate/dense/kernel
Î
Dmio_variable/bert/encoder/layer_4/intermediate/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_4/intermediate/dense/kernel*
shape:

Z
%Initializer_79/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_79/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_79/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_79/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_79/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0

#Initializer_79/truncated_normal/mulMul/Initializer_79/truncated_normal/TruncatedNormal&Initializer_79/truncated_normal/stddev*
T0
z
Initializer_79/truncated_normalAdd#Initializer_79/truncated_normal/mul$Initializer_79/truncated_normal/mean*
T0

	Assign_79AssignDmio_variable/bert/encoder/layer_4/intermediate/dense/kernel/gradientInitializer_79/truncated_normal*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_4/intermediate/dense/kernel/gradient*
validate_shape(
Å
Bmio_variable/bert/encoder/layer_4/intermediate/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_4/intermediate/dense/bias*
shape:
Å
Bmio_variable/bert/encoder/layer_4/intermediate/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_4/intermediate/dense/bias*
shape:
S
$Initializer_80/zeros/shape_as_tensorConst*
valueB:*
dtype0
G
Initializer_80/zeros/ConstConst*
valueB
 *    *
dtype0
y
Initializer_80/zerosFill$Initializer_80/zeros/shape_as_tensorInitializer_80/zeros/Const*

index_type0*
T0
ö
	Assign_80AssignBmio_variable/bert/encoder/layer_4/intermediate/dense/bias/gradientInitializer_80/zeros*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_4/intermediate/dense/bias/gradient*
validate_shape(*
use_locking(
î
.bert/encoder/layer_4/intermediate/dense/MatMulMatMul?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add_1Dmio_variable/bert/encoder/layer_4/intermediate/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Î
/bert/encoder/layer_4/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_4/intermediate/dense/MatMulBmio_variable/bert/encoder/layer_4/intermediate/dense/bias/variable*
T0*
data_formatNHWC
Z
-bert/encoder/layer_4/intermediate/dense/Pow/yConst*
dtype0*
valueB
 *  @@

+bert/encoder/layer_4/intermediate/dense/PowPow/bert/encoder/layer_4/intermediate/dense/BiasAdd-bert/encoder/layer_4/intermediate/dense/Pow/y*
T0
Z
-bert/encoder/layer_4/intermediate/dense/mul/xConst*
dtype0*
valueB
 *'7=

+bert/encoder/layer_4/intermediate/dense/mulMul-bert/encoder/layer_4/intermediate/dense/mul/x+bert/encoder/layer_4/intermediate/dense/Pow*
T0

+bert/encoder/layer_4/intermediate/dense/addAdd/bert/encoder/layer_4/intermediate/dense/BiasAdd+bert/encoder/layer_4/intermediate/dense/mul*
T0
\
/bert/encoder/layer_4/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0

-bert/encoder/layer_4/intermediate/dense/mul_1Mul/bert/encoder/layer_4/intermediate/dense/mul_1/x+bert/encoder/layer_4/intermediate/dense/add*
T0
l
,bert/encoder/layer_4/intermediate/dense/TanhTanh-bert/encoder/layer_4/intermediate/dense/mul_1*
T0
\
/bert/encoder/layer_4/intermediate/dense/add_1/xConst*
valueB
 *  ?*
dtype0

-bert/encoder/layer_4/intermediate/dense/add_1Add/bert/encoder/layer_4/intermediate/dense/add_1/x,bert/encoder/layer_4/intermediate/dense/Tanh*
T0
\
/bert/encoder/layer_4/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0

-bert/encoder/layer_4/intermediate/dense/mul_2Mul/bert/encoder/layer_4/intermediate/dense/mul_2/x-bert/encoder/layer_4/intermediate/dense/add_1*
T0

-bert/encoder/layer_4/intermediate/dense/mul_3Mul/bert/encoder/layer_4/intermediate/dense/BiasAdd-bert/encoder/layer_4/intermediate/dense/mul_2*
T0
Â
>mio_variable/bert/encoder/layer_4/output/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*7
	container*(bert/encoder/layer_4/output/dense/kernel
Â
>mio_variable/bert/encoder/layer_4/output/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(bert/encoder/layer_4/output/dense/kernel*
shape:

Z
%Initializer_81/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_81/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_81/truncated_normal/stddevConst*
dtype0*
valueB
 *
×Ŗ<

/Initializer_81/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_81/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

#Initializer_81/truncated_normal/mulMul/Initializer_81/truncated_normal/TruncatedNormal&Initializer_81/truncated_normal/stddev*
T0
z
Initializer_81/truncated_normalAdd#Initializer_81/truncated_normal/mul$Initializer_81/truncated_normal/mean*
T0
ų
	Assign_81Assign>mio_variable/bert/encoder/layer_4/output/dense/kernel/gradientInitializer_81/truncated_normal*Q
_classG
ECloc:@mio_variable/bert/encoder/layer_4/output/dense/kernel/gradient*
validate_shape(*
use_locking(*
T0
š
<mio_variable/bert/encoder/layer_4/output/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&bert/encoder/layer_4/output/dense/bias*
shape:
š
<mio_variable/bert/encoder/layer_4/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*5
	container(&bert/encoder/layer_4/output/dense/bias
F
Initializer_82/zerosConst*
valueB*    *
dtype0
ę
	Assign_82Assign<mio_variable/bert/encoder/layer_4/output/dense/bias/gradientInitializer_82/zeros*
use_locking(*
T0*O
_classE
CAloc:@mio_variable/bert/encoder/layer_4/output/dense/bias/gradient*
validate_shape(
Đ
(bert/encoder/layer_4/output/dense/MatMulMatMul-bert/encoder/layer_4/intermediate/dense/mul_3>mio_variable/bert/encoder/layer_4/output/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
ŧ
)bert/encoder/layer_4/output/dense/BiasAddBiasAdd(bert/encoder/layer_4/output/dense/MatMul<mio_variable/bert/encoder/layer_4/output/dense/bias/variable*
data_formatNHWC*
T0
Z
-bert/encoder/layer_4/output/dropout/keep_probConst*
dtype0*
valueB
 *fff?
v
)bert/encoder/layer_4/output/dropout/ShapeShape)bert/encoder/layer_4/output/dense/BiasAdd*
T0*
out_type0
c
6bert/encoder/layer_4/output/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
c
6bert/encoder/layer_4/output/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0
Ģ
@bert/encoder/layer_4/output/dropout/random_uniform/RandomUniformRandomUniform)bert/encoder/layer_4/output/dropout/Shape*
dtype0*
seed2 *

seed *
T0
ļ
6bert/encoder/layer_4/output/dropout/random_uniform/subSub6bert/encoder/layer_4/output/dropout/random_uniform/max6bert/encoder/layer_4/output/dropout/random_uniform/min*
T0
Ā
6bert/encoder/layer_4/output/dropout/random_uniform/mulMul@bert/encoder/layer_4/output/dropout/random_uniform/RandomUniform6bert/encoder/layer_4/output/dropout/random_uniform/sub*
T0
˛
2bert/encoder/layer_4/output/dropout/random_uniformAdd6bert/encoder/layer_4/output/dropout/random_uniform/mul6bert/encoder/layer_4/output/dropout/random_uniform/min*
T0

'bert/encoder/layer_4/output/dropout/addAdd-bert/encoder/layer_4/output/dropout/keep_prob2bert/encoder/layer_4/output/dropout/random_uniform*
T0
d
)bert/encoder/layer_4/output/dropout/FloorFloor'bert/encoder/layer_4/output/dropout/add*
T0

'bert/encoder/layer_4/output/dropout/divRealDiv)bert/encoder/layer_4/output/dense/BiasAdd-bert/encoder/layer_4/output/dropout/keep_prob*
T0

'bert/encoder/layer_4/output/dropout/mulMul'bert/encoder/layer_4/output/dropout/div)bert/encoder/layer_4/output/dropout/Floor*
T0

bert/encoder/layer_4/output/addAdd'bert/encoder/layer_4/output/dropout/mul?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add_1*
T0
Á
@mio_variable/bert/encoder/layer_4/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*bert/encoder/layer_4/output/LayerNorm/beta*
shape:
Á
@mio_variable/bert/encoder/layer_4/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*bert/encoder/layer_4/output/LayerNorm/beta*
shape:
F
Initializer_83/zerosConst*
valueB*    *
dtype0
ō
	Assign_83Assign@mio_variable/bert/encoder/layer_4/output/LayerNorm/beta/gradientInitializer_83/zeros*
validate_shape(*
use_locking(*
T0*S
_classI
GEloc:@mio_variable/bert/encoder/layer_4/output/LayerNorm/beta/gradient
Ã
Amio_variable/bert/encoder/layer_4/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*:
	container-+bert/encoder/layer_4/output/LayerNorm/gamma
Ã
Amio_variable/bert/encoder/layer_4/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*:
	container-+bert/encoder/layer_4/output/LayerNorm/gamma
E
Initializer_84/onesConst*
valueB*  ?*
dtype0
ķ
	Assign_84AssignAmio_variable/bert/encoder/layer_4/output/LayerNorm/gamma/gradientInitializer_84/ones*
use_locking(*
T0*T
_classJ
HFloc:@mio_variable/bert/encoder/layer_4/output/LayerNorm/gamma/gradient*
validate_shape(
r
Dbert/encoder/layer_4/output/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
valueB:
Į
2bert/encoder/layer_4/output/LayerNorm/moments/meanMeanbert/encoder/layer_4/output/addDbert/encoder/layer_4/output/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0

:bert/encoder/layer_4/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_4/output/LayerNorm/moments/mean*
T0
ē
?bert/encoder/layer_4/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_4/output/add:bert/encoder/layer_4/output/LayerNorm/moments/StopGradient*
T0
v
Hbert/encoder/layer_4/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0
ī
6bert/encoder/layer_4/output/LayerNorm/moments/varianceMean?bert/encoder/layer_4/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_4/output/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0
b
5bert/encoder/layer_4/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ėŧ+*
dtype0
˛
3bert/encoder/layer_4/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_4/output/LayerNorm/moments/variance5bert/encoder/layer_4/output/LayerNorm/batchnorm/add/y*
T0
|
5bert/encoder/layer_4/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_4/output/LayerNorm/batchnorm/add*
T0
Ŋ
3bert/encoder/layer_4/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_4/output/LayerNorm/batchnorm/RsqrtAmio_variable/bert/encoder/layer_4/output/LayerNorm/gamma/variable*
T0

5bert/encoder/layer_4/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_4/output/add3bert/encoder/layer_4/output/LayerNorm/batchnorm/mul*
T0
Ž
5bert/encoder/layer_4/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_4/output/LayerNorm/moments/mean3bert/encoder/layer_4/output/LayerNorm/batchnorm/mul*
T0
ŧ
3bert/encoder/layer_4/output/LayerNorm/batchnorm/subSub@mio_variable/bert/encoder/layer_4/output/LayerNorm/beta/variable5bert/encoder/layer_4/output/LayerNorm/batchnorm/mul_2*
T0
ą
5bert/encoder/layer_4/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_4/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_4/output/LayerNorm/batchnorm/sub*
T0
x
(bert/encoder/layer_4/output/StopGradientStopGradient5bert/encoder/layer_4/output/LayerNorm/batchnorm/add_1*
T0
u
)bert/encoder/layer_5/attention/self/ShapeShape(bert/encoder/layer_4/output/StopGradient*
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
1bert/encoder/layer_5/attention/self/strided_sliceStridedSlice)bert/encoder/layer_5/attention/self/Shape7bert/encoder/layer_5/attention/self/strided_slice/stack9bert/encoder/layer_5/attention/self/strided_slice/stack_19bert/encoder/layer_5/attention/self/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0
w
+bert/encoder/layer_5/attention/self/Shape_1Shape(bert/encoder/layer_4/output/StopGradient*
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
;bert/encoder/layer_5/attention/self/strided_slice_1/stack_2Const*
dtype0*
valueB:

3bert/encoder/layer_5/attention/self/strided_slice_1StridedSlice+bert/encoder/layer_5/attention/self/Shape_19bert/encoder/layer_5/attention/self/strided_slice_1/stack;bert/encoder/layer_5/attention/self/strided_slice_1/stack_1;bert/encoder/layer_5/attention/self/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
Ō
Fmio_variable/bert/encoder/layer_5/attention/self/query/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_5/attention/self/query/kernel*
shape:

Ō
Fmio_variable/bert/encoder/layer_5/attention/self/query/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_5/attention/self/query/kernel*
shape:

Z
%Initializer_85/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_85/truncated_normal/meanConst*
dtype0*
valueB
 *    
S
&Initializer_85/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_85/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_85/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

#Initializer_85/truncated_normal/mulMul/Initializer_85/truncated_normal/TruncatedNormal&Initializer_85/truncated_normal/stddev*
T0
z
Initializer_85/truncated_normalAdd#Initializer_85/truncated_normal/mul$Initializer_85/truncated_normal/mean*
T0

	Assign_85AssignFmio_variable/bert/encoder/layer_5/attention/self/query/kernel/gradientInitializer_85/truncated_normal*
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
Dmio_variable/bert/encoder/layer_5/attention/self/query/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_5/attention/self/query/bias*
shape:
F
Initializer_86/zerosConst*
dtype0*
valueB*    
ú
	Assign_86AssignDmio_variable/bert/encoder/layer_5/attention/self/query/bias/gradientInitializer_86/zeros*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_5/attention/self/query/bias/gradient*
validate_shape(
Û
0bert/encoder/layer_5/attention/self/query/MatMulMatMul(bert/encoder/layer_4/output/StopGradientFmio_variable/bert/encoder/layer_5/attention/self/query/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Ô
1bert/encoder/layer_5/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_5/attention/self/query/MatMulDmio_variable/bert/encoder/layer_5/attention/self/query/bias/variable*
data_formatNHWC*
T0
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
Z
%Initializer_87/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_87/truncated_normal/meanConst*
dtype0*
valueB
 *    
S
&Initializer_87/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_87/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_87/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

#Initializer_87/truncated_normal/mulMul/Initializer_87/truncated_normal/TruncatedNormal&Initializer_87/truncated_normal/stddev*
T0
z
Initializer_87/truncated_normalAdd#Initializer_87/truncated_normal/mul$Initializer_87/truncated_normal/mean*
T0

	Assign_87AssignDmio_variable/bert/encoder/layer_5/attention/self/key/kernel/gradientInitializer_87/truncated_normal*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_5/attention/self/key/kernel/gradient*
validate_shape(
Å
Bmio_variable/bert/encoder/layer_5/attention/self/key/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_5/attention/self/key/bias*
shape:
Å
Bmio_variable/bert/encoder/layer_5/attention/self/key/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_5/attention/self/key/bias*
shape:
F
Initializer_88/zerosConst*
valueB*    *
dtype0
ö
	Assign_88AssignBmio_variable/bert/encoder/layer_5/attention/self/key/bias/gradientInitializer_88/zeros*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_5/attention/self/key/bias/gradient*
validate_shape(*
use_locking(
×
.bert/encoder/layer_5/attention/self/key/MatMulMatMul(bert/encoder/layer_4/output/StopGradientDmio_variable/bert/encoder/layer_5/attention/self/key/kernel/variable*
T0*
transpose_a( *
transpose_b( 
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
Z
%Initializer_89/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_89/truncated_normal/meanConst*
dtype0*
valueB
 *    
S
&Initializer_89/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_89/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_89/truncated_normal/shape*
seed2 *

seed *
T0*
dtype0

#Initializer_89/truncated_normal/mulMul/Initializer_89/truncated_normal/TruncatedNormal&Initializer_89/truncated_normal/stddev*
T0
z
Initializer_89/truncated_normalAdd#Initializer_89/truncated_normal/mul$Initializer_89/truncated_normal/mean*
T0

	Assign_89AssignFmio_variable/bert/encoder/layer_5/attention/self/value/kernel/gradientInitializer_89/truncated_normal*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_5/attention/self/value/kernel/gradient*
validate_shape(*
use_locking(
É
Dmio_variable/bert/encoder/layer_5/attention/self/value/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*=
	container0.bert/encoder/layer_5/attention/self/value/bias
É
Dmio_variable/bert/encoder/layer_5/attention/self/value/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_5/attention/self/value/bias*
shape:
F
Initializer_90/zerosConst*
valueB*    *
dtype0
ú
	Assign_90AssignDmio_variable/bert/encoder/layer_5/attention/self/value/bias/gradientInitializer_90/zeros*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_5/attention/self/value/bias/gradient*
validate_shape(*
use_locking(
Û
0bert/encoder/layer_5/attention/self/value/MatMulMatMul(bert/encoder/layer_4/output/StopGradientFmio_variable/bert/encoder/layer_5/attention/self/value/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Ô
1bert/encoder/layer_5/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_5/attention/self/value/MatMulDmio_variable/bert/encoder/layer_5/attention/self/value/bias/variable*
T0*
data_formatNHWC
]
3bert/encoder/layer_5/attention/self/Reshape/shape/1Const*
value	B :*
dtype0
]
3bert/encoder/layer_5/attention/self/Reshape/shape/2Const*
value	B :*
dtype0
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
+bert/encoder/layer_5/attention/self/ReshapeReshape1bert/encoder/layer_5/attention/self/query/BiasAdd1bert/encoder/layer_5/attention/self/Reshape/shape*
Tshape0*
T0
o
2bert/encoder/layer_5/attention/self/transpose/permConst*%
valueB"             *
dtype0
ą
-bert/encoder/layer_5/attention/self/transpose	Transpose+bert/encoder/layer_5/attention/self/Reshape2bert/encoder/layer_5/attention/self/transpose/perm*
T0*
Tperm0
_
5bert/encoder/layer_5/attention/self/Reshape_1/shape/1Const*
value	B :*
dtype0
_
5bert/encoder/layer_5/attention/self/Reshape_1/shape/2Const*
value	B :*
dtype0
_
5bert/encoder/layer_5/attention/self/Reshape_1/shape/3Const*
dtype0*
value	B : 
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
*bert/encoder/layer_5/attention/self/MatMulBatchMatMul-bert/encoder/layer_5/attention/self/transpose/bert/encoder/layer_5/attention/self/transpose_1*
T0*
adj_x( *
adj_y(
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
1bert/encoder/layer_5/attention/self/dropout/ShapeShape+bert/encoder/layer_5/attention/self/Softmax*
out_type0*
T0
k
>bert/encoder/layer_5/attention/self/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
k
>bert/encoder/layer_5/attention/self/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0
ģ
Hbert/encoder/layer_5/attention/self/dropout/random_uniform/RandomUniformRandomUniform1bert/encoder/layer_5/attention/self/dropout/Shape*
T0*
dtype0*
seed2 *

seed 
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
,bert/encoder/layer_5/attention/self/MatMul_1BatchMatMul/bert/encoder/layer_5/attention/self/dropout/mul/bert/encoder/layer_5/attention/self/transpose_2*
T0*
adj_x( *
adj_y( 
q
4bert/encoder/layer_5/attention/self/transpose_3/permConst*%
valueB"             *
dtype0
ļ
/bert/encoder/layer_5/attention/self/transpose_3	Transpose,bert/encoder/layer_5/attention/self/MatMul_14bert/encoder/layer_5/attention/self/transpose_3/perm*
T0*
Tperm0
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
Hmio_variable/bert/encoder/layer_5/attention/output/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*A
	container42bert/encoder/layer_5/attention/output/dense/kernel
Ö
Hmio_variable/bert/encoder/layer_5/attention/output/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*A
	container42bert/encoder/layer_5/attention/output/dense/kernel
Z
%Initializer_91/truncated_normal/shapeConst*
dtype0*
valueB"      
Q
$Initializer_91/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_91/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_91/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_91/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 

#Initializer_91/truncated_normal/mulMul/Initializer_91/truncated_normal/TruncatedNormal&Initializer_91/truncated_normal/stddev*
T0
z
Initializer_91/truncated_normalAdd#Initializer_91/truncated_normal/mul$Initializer_91/truncated_normal/mean*
T0

	Assign_91AssignHmio_variable/bert/encoder/layer_5/attention/output/dense/kernel/gradientInitializer_91/truncated_normal*
T0*[
_classQ
OMloc:@mio_variable/bert/encoder/layer_5/attention/output/dense/kernel/gradient*
validate_shape(*
use_locking(
Í
Fmio_variable/bert/encoder/layer_5/attention/output/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_5/attention/output/dense/bias*
shape:
Í
Fmio_variable/bert/encoder/layer_5/attention/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_5/attention/output/dense/bias*
shape:
F
Initializer_92/zerosConst*
dtype0*
valueB*    
ū
	Assign_92AssignFmio_variable/bert/encoder/layer_5/attention/output/dense/bias/gradientInitializer_92/zeros*
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
3bert/encoder/layer_5/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_5/attention/output/dense/MatMulFmio_variable/bert/encoder/layer_5/attention/output/dense/bias/variable*
T0*
data_formatNHWC
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
dtype0*
seed2 *

seed *
T0
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

)bert/encoder/layer_5/attention/output/addAdd1bert/encoder/layer_5/attention/output/dropout/mul(bert/encoder/layer_4/output/StopGradient*
T0
Õ
Jmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64bert/encoder/layer_5/attention/output/LayerNorm/beta*
shape:
Õ
Jmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*C
	container64bert/encoder/layer_5/attention/output/LayerNorm/beta
F
Initializer_93/zerosConst*
valueB*    *
dtype0

	Assign_93AssignJmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/beta/gradientInitializer_93/zeros*
validate_shape(*
use_locking(*
T0*]
_classS
QOloc:@mio_variable/bert/encoder/layer_5/attention/output/LayerNorm/beta/gradient
×
Kmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*D
	container75bert/encoder/layer_5/attention/output/LayerNorm/gamma*
shape:
×
Kmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*D
	container75bert/encoder/layer_5/attention/output/LayerNorm/gamma
E
Initializer_94/onesConst*
valueB*  ?*
dtype0

	Assign_94AssignKmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/gamma/gradientInitializer_94/ones*
T0*^
_classT
RPloc:@mio_variable/bert/encoder/layer_5/attention/output/LayerNorm/gamma/gradient*
validate_shape(*
use_locking(
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
%Initializer_95/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_95/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_95/truncated_normal/stddevConst*
dtype0*
valueB
 *
×Ŗ<

/Initializer_95/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_95/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

#Initializer_95/truncated_normal/mulMul/Initializer_95/truncated_normal/TruncatedNormal&Initializer_95/truncated_normal/stddev*
T0
z
Initializer_95/truncated_normalAdd#Initializer_95/truncated_normal/mul$Initializer_95/truncated_normal/mean*
T0

	Assign_95AssignDmio_variable/bert/encoder/layer_5/intermediate/dense/kernel/gradientInitializer_95/truncated_normal*
validate_shape(*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_5/intermediate/dense/kernel/gradient
Å
Bmio_variable/bert/encoder/layer_5/intermediate/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_5/intermediate/dense/bias*
shape:
Å
Bmio_variable/bert/encoder/layer_5/intermediate/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_5/intermediate/dense/bias*
shape:
S
$Initializer_96/zeros/shape_as_tensorConst*
valueB:*
dtype0
G
Initializer_96/zeros/ConstConst*
valueB
 *    *
dtype0
y
Initializer_96/zerosFill$Initializer_96/zeros/shape_as_tensorInitializer_96/zeros/Const*
T0*

index_type0
ö
	Assign_96AssignBmio_variable/bert/encoder/layer_5/intermediate/dense/bias/gradientInitializer_96/zeros*
use_locking(*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_5/intermediate/dense/bias/gradient*
validate_shape(
î
.bert/encoder/layer_5/intermediate/dense/MatMulMatMul?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add_1Dmio_variable/bert/encoder/layer_5/intermediate/dense/kernel/variable*
transpose_b( *
T0*
transpose_a( 
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
-bert/encoder/layer_5/intermediate/dense/mul/xConst*
dtype0*
valueB
 *'7=
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
>mio_variable/bert/encoder/layer_5/output/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*7
	container*(bert/encoder/layer_5/output/dense/kernel
Â
>mio_variable/bert/encoder/layer_5/output/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*7
	container*(bert/encoder/layer_5/output/dense/kernel
Z
%Initializer_97/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_97/truncated_normal/meanConst*
dtype0*
valueB
 *    
S
&Initializer_97/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_97/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_97/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0

#Initializer_97/truncated_normal/mulMul/Initializer_97/truncated_normal/TruncatedNormal&Initializer_97/truncated_normal/stddev*
T0
z
Initializer_97/truncated_normalAdd#Initializer_97/truncated_normal/mul$Initializer_97/truncated_normal/mean*
T0
ų
	Assign_97Assign>mio_variable/bert/encoder/layer_5/output/dense/kernel/gradientInitializer_97/truncated_normal*
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
Initializer_98/zerosConst*
valueB*    *
dtype0
ę
	Assign_98Assign<mio_variable/bert/encoder/layer_5/output/dense/bias/gradientInitializer_98/zeros*
validate_shape(*
use_locking(*
T0*O
_classE
CAloc:@mio_variable/bert/encoder/layer_5/output/dense/bias/gradient
Đ
(bert/encoder/layer_5/output/dense/MatMulMatMul-bert/encoder/layer_5/intermediate/dense/mul_3>mio_variable/bert/encoder/layer_5/output/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
ŧ
)bert/encoder/layer_5/output/dense/BiasAddBiasAdd(bert/encoder/layer_5/output/dense/MatMul<mio_variable/bert/encoder/layer_5/output/dense/bias/variable*
T0*
data_formatNHWC
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
6bert/encoder/layer_5/output/dropout/random_uniform/maxConst*
dtype0*
valueB
 *  ?
Ģ
@bert/encoder/layer_5/output/dropout/random_uniform/RandomUniformRandomUniform)bert/encoder/layer_5/output/dropout/Shape*
T0*
dtype0*
seed2 *

seed 
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
@mio_variable/bert/encoder/layer_5/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*bert/encoder/layer_5/output/LayerNorm/beta*
shape:
Á
@mio_variable/bert/encoder/layer_5/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*bert/encoder/layer_5/output/LayerNorm/beta*
shape:
F
Initializer_99/zerosConst*
valueB*    *
dtype0
ō
	Assign_99Assign@mio_variable/bert/encoder/layer_5/output/LayerNorm/beta/gradientInitializer_99/zeros*
T0*S
_classI
GEloc:@mio_variable/bert/encoder/layer_5/output/LayerNorm/beta/gradient*
validate_shape(*
use_locking(
Ã
Amio_variable/bert/encoder/layer_5/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+bert/encoder/layer_5/output/LayerNorm/gamma*
shape:
Ã
Amio_variable/bert/encoder/layer_5/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*:
	container-+bert/encoder/layer_5/output/LayerNorm/gamma
F
Initializer_100/onesConst*
valueB*  ?*
dtype0
õ

Assign_100AssignAmio_variable/bert/encoder/layer_5/output/LayerNorm/gamma/gradientInitializer_100/ones*
use_locking(*
T0*T
_classJ
HFloc:@mio_variable/bert/encoder/layer_5/output/LayerNorm/gamma/gradient*
validate_shape(
r
Dbert/encoder/layer_5/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0
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
6bert/encoder/layer_5/output/LayerNorm/moments/varianceMean?bert/encoder/layer_5/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_5/output/LayerNorm/moments/variance/reduction_indices*
T0*

Tidx0*
	keep_dims(
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
bert/encoder/Shape_3Shape5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
P
"bert/encoder/strided_slice_3/stackConst*
valueB: *
dtype0
R
$bert/encoder/strided_slice_3/stack_1Const*
dtype0*
valueB:
R
$bert/encoder/strided_slice_3/stack_2Const*
valueB:*
dtype0
Ŧ
bert/encoder/strided_slice_3StridedSlicebert/encoder/Shape_3"bert/encoder/strided_slice_3/stack$bert/encoder/strided_slice_3/stack_1$bert/encoder/strided_slice_3/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
H
bert/encoder/Reshape_2/shape/1Const*
dtype0*
value	B :
I
bert/encoder/Reshape_2/shape/2Const*
dtype0*
value
B :
 
bert/encoder/Reshape_2/shapePackbert/encoder/strided_slice_2bert/encoder/Reshape_2/shape/1bert/encoder/Reshape_2/shape/2*
T0*

axis *
N

bert/encoder/Reshape_2Reshape5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_2/shape*
T0*
Tshape0
m
bert/encoder/Shape_4Shape5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
P
"bert/encoder/strided_slice_4/stackConst*
valueB: *
dtype0
R
$bert/encoder/strided_slice_4/stack_1Const*
dtype0*
valueB:
R
$bert/encoder/strided_slice_4/stack_2Const*
valueB:*
dtype0
Ŧ
bert/encoder/strided_slice_4StridedSlicebert/encoder/Shape_4"bert/encoder/strided_slice_4/stack$bert/encoder/strided_slice_4/stack_1$bert/encoder/strided_slice_4/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0
H
bert/encoder/Reshape_3/shape/1Const*
value	B :*
dtype0
I
bert/encoder/Reshape_3/shape/2Const*
value
B :*
dtype0
 
bert/encoder/Reshape_3/shapePackbert/encoder/strided_slice_2bert/encoder/Reshape_3/shape/1bert/encoder/Reshape_3/shape/2*
T0*

axis *
N

bert/encoder/Reshape_3Reshape5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_3/shape*
T0*
Tshape0
m
bert/encoder/Shape_5Shape5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1*
out_type0*
T0
P
"bert/encoder/strided_slice_5/stackConst*
valueB: *
dtype0
R
$bert/encoder/strided_slice_5/stack_1Const*
valueB:*
dtype0
R
$bert/encoder/strided_slice_5/stack_2Const*
valueB:*
dtype0
Ŧ
bert/encoder/strided_slice_5StridedSlicebert/encoder/Shape_5"bert/encoder/strided_slice_5/stack$bert/encoder/strided_slice_5/stack_1$bert/encoder/strided_slice_5/stack_2*
end_mask *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
H
bert/encoder/Reshape_4/shape/1Const*
value	B :*
dtype0
I
bert/encoder/Reshape_4/shape/2Const*
dtype0*
value
B :
 
bert/encoder/Reshape_4/shapePackbert/encoder/strided_slice_2bert/encoder/Reshape_4/shape/1bert/encoder/Reshape_4/shape/2*
T0*

axis *
N

bert/encoder/Reshape_4Reshape5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_4/shape*
T0*
Tshape0
m
bert/encoder/Shape_6Shape5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
P
"bert/encoder/strided_slice_6/stackConst*
valueB: *
dtype0
R
$bert/encoder/strided_slice_6/stack_1Const*
valueB:*
dtype0
R
$bert/encoder/strided_slice_6/stack_2Const*
valueB:*
dtype0
Ŧ
bert/encoder/strided_slice_6StridedSlicebert/encoder/Shape_6"bert/encoder/strided_slice_6/stack$bert/encoder/strided_slice_6/stack_1$bert/encoder/strided_slice_6/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
H
bert/encoder/Reshape_5/shape/1Const*
value	B :*
dtype0
I
bert/encoder/Reshape_5/shape/2Const*
value
B :*
dtype0
 
bert/encoder/Reshape_5/shapePackbert/encoder/strided_slice_2bert/encoder/Reshape_5/shape/1bert/encoder/Reshape_5/shape/2*
N*
T0*

axis 

bert/encoder/Reshape_5Reshape5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_5/shape*
T0*
Tshape0
`
bert/encoder/Shape_7Shape(bert/encoder/layer_4/output/StopGradient*
T0*
out_type0
P
"bert/encoder/strided_slice_7/stackConst*
valueB: *
dtype0
R
$bert/encoder/strided_slice_7/stack_1Const*
valueB:*
dtype0
R
$bert/encoder/strided_slice_7/stack_2Const*
dtype0*
valueB:
Ŧ
bert/encoder/strided_slice_7StridedSlicebert/encoder/Shape_7"bert/encoder/strided_slice_7/stack$bert/encoder/strided_slice_7/stack_1$bert/encoder/strided_slice_7/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
H
bert/encoder/Reshape_6/shape/1Const*
value	B :*
dtype0
I
bert/encoder/Reshape_6/shape/2Const*
value
B :*
dtype0
 
bert/encoder/Reshape_6/shapePackbert/encoder/strided_slice_2bert/encoder/Reshape_6/shape/1bert/encoder/Reshape_6/shape/2*

axis *
N*
T0

bert/encoder/Reshape_6Reshape(bert/encoder/layer_4/output/StopGradientbert/encoder/Reshape_6/shape*
T0*
Tshape0
m
bert/encoder/Shape_8Shape5bert/encoder/layer_5/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
P
"bert/encoder/strided_slice_8/stackConst*
valueB: *
dtype0
R
$bert/encoder/strided_slice_8/stack_1Const*
dtype0*
valueB:
R
$bert/encoder/strided_slice_8/stack_2Const*
valueB:*
dtype0
Ŧ
bert/encoder/strided_slice_8StridedSlicebert/encoder/Shape_8"bert/encoder/strided_slice_8/stack$bert/encoder/strided_slice_8/stack_1$bert/encoder/strided_slice_8/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
H
bert/encoder/Reshape_7/shape/1Const*
value	B :*
dtype0
I
bert/encoder/Reshape_7/shape/2Const*
value
B :*
dtype0
 
bert/encoder/Reshape_7/shapePackbert/encoder/strided_slice_2bert/encoder/Reshape_7/shape/1bert/encoder/Reshape_7/shape/2*
T0*

axis *
N

bert/encoder/Reshape_7Reshape5bert/encoder/layer_5/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_7/shape*
T0*
Tshape0
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
bert/pooler/strided_sliceStridedSlicebert/encoder/Reshape_7bert/pooler/strided_slice/stack!bert/pooler/strided_slice/stack_1!bert/pooler/strided_slice/stack_2*
Index0*
T0*
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
.mio_variable/bert/pooler/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerbert/pooler/dense/kernel*
shape:

[
&Initializer_101/truncated_normal/shapeConst*
valueB"      *
dtype0
R
%Initializer_101/truncated_normal/meanConst*
valueB
 *    *
dtype0
T
'Initializer_101/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

0Initializer_101/truncated_normal/TruncatedNormalTruncatedNormal&Initializer_101/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 

$Initializer_101/truncated_normal/mulMul0Initializer_101/truncated_normal/TruncatedNormal'Initializer_101/truncated_normal/stddev*
T0
}
 Initializer_101/truncated_normalAdd$Initializer_101/truncated_normal/mul%Initializer_101/truncated_normal/mean*
T0
Û

Assign_101Assign.mio_variable/bert/pooler/dense/kernel/gradient Initializer_101/truncated_normal*A
_class7
53loc:@mio_variable/bert/pooler/dense/kernel/gradient*
validate_shape(*
use_locking(*
T0

,mio_variable/bert/pooler/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerbert/pooler/dense/bias*
shape:

,mio_variable/bert/pooler/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerbert/pooler/dense/bias*
shape:
G
Initializer_102/zerosConst*
valueB*    *
dtype0
Ė

Assign_102Assign,mio_variable/bert/pooler/dense/bias/gradientInitializer_102/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/bert/pooler/dense/bias/gradient*
validate_shape(
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
strided_slice_1StridedSlicebert/encoder/Reshape_7strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0
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
concat/values_7GatherV2)mio_embeddings/context_embedding/variableCastconcat/values_7/axis*
Taxis0*
Tindices0*
Tparams0
>
concat/axisConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Ø
concatConcatV2concat/values_0&mio_embeddings/c_id_embedding/variable(mio_embeddings/c_info_embedding/variableconcat/values_3concat/values_4concat/values_5concat/values_6concat/values_7/mio_embeddings/comment_genre_embedding/variable0mio_embeddings/comment_length_embedding/variableconcat/axis*
N
*

Tidx0*
T0
@
concat_1/axisConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Z
concat_1ConcatV2concatstrided_slice_1concat_1/axis*
T0*
N*

Tidx0
@
concat_2/values_0/axisConst*
dtype0*
value	B : 

concat_2/values_0GatherV2%mio_embeddings/did_embedding/variableCastconcat_2/values_0/axis*
Taxis0*
Tindices0*
Tparams0
@
concat_2/values_2/axisConst*
value	B : *
dtype0
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
concat_2ConcatV2concat_2/values_0*mio_embeddings/position_embedding/variableconcat_2/values_2concat_2/axis*
N*

Tidx0*
T0
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
Y
$Initializer_103/random_uniform/shapeConst*
valueB"°     *
dtype0
O
"Initializer_103/random_uniform/minConst*
valueB
 *ÃĐŊ*
dtype0
O
"Initializer_103/random_uniform/maxConst*
valueB
 *ÃĐ=*
dtype0

,Initializer_103/random_uniform/RandomUniformRandomUniform$Initializer_103/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_103/random_uniform/subSub"Initializer_103/random_uniform/max"Initializer_103/random_uniform/min*
T0

"Initializer_103/random_uniform/mulMul,Initializer_103/random_uniform/RandomUniform"Initializer_103/random_uniform/sub*
T0
v
Initializer_103/random_uniformAdd"Initializer_103/random_uniform/mul"Initializer_103/random_uniform/min*
T0
×

Assign_103Assign-mio_variable/expand_xtr/dense/kernel/gradientInitializer_103/random_uniform*
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
G
Initializer_104/zerosConst*
valueB*    *
dtype0
Ę

Assign_104Assign+mio_variable/expand_xtr/dense/bias/gradientInitializer_104/zeros*>
_class4
20loc:@mio_variable/expand_xtr/dense/bias/gradient*
validate_shape(*
use_locking(*
T0

expand_xtr/dense/MatMulMatMulconcat_1-mio_variable/expand_xtr/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0

expand_xtr/dense/BiasAddBiasAddexpand_xtr/dense/MatMul+mio_variable/expand_xtr/dense/bias/variable*
T0*
data_formatNHWC
M
 expand_xtr/dense/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
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
/mio_variable/expand_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*(
	containerexpand_xtr/dense_1/kernel
Y
$Initializer_105/random_uniform/shapeConst*
dtype0*
valueB"      
O
"Initializer_105/random_uniform/minConst*
valueB
 *   ž*
dtype0
O
"Initializer_105/random_uniform/maxConst*
dtype0*
valueB
 *   >

,Initializer_105/random_uniform/RandomUniformRandomUniform$Initializer_105/random_uniform/shape*
seed2 *

seed *
T0*
dtype0
z
"Initializer_105/random_uniform/subSub"Initializer_105/random_uniform/max"Initializer_105/random_uniform/min*
T0

"Initializer_105/random_uniform/mulMul,Initializer_105/random_uniform/RandomUniform"Initializer_105/random_uniform/sub*
T0
v
Initializer_105/random_uniformAdd"Initializer_105/random_uniform/mul"Initializer_105/random_uniform/min*
T0
Û

Assign_105Assign/mio_variable/expand_xtr/dense_1/kernel/gradientInitializer_105/random_uniform*
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
G
Initializer_106/zerosConst*
valueB*    *
dtype0
Î

Assign_106Assign-mio_variable/expand_xtr/dense_1/bias/gradientInitializer_106/zeros*
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
/mio_variable/expand_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*(
	containerexpand_xtr/dense_2/kernel
Ŗ
/mio_variable/expand_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerexpand_xtr/dense_2/kernel*
shape:	@
Y
$Initializer_107/random_uniform/shapeConst*
valueB"   @   *
dtype0
O
"Initializer_107/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
O
"Initializer_107/random_uniform/maxConst*
valueB
 *ķ5>*
dtype0

,Initializer_107/random_uniform/RandomUniformRandomUniform$Initializer_107/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_107/random_uniform/subSub"Initializer_107/random_uniform/max"Initializer_107/random_uniform/min*
T0

"Initializer_107/random_uniform/mulMul,Initializer_107/random_uniform/RandomUniform"Initializer_107/random_uniform/sub*
T0
v
Initializer_107/random_uniformAdd"Initializer_107/random_uniform/mul"Initializer_107/random_uniform/min*
T0
Û

Assign_107Assign/mio_variable/expand_xtr/dense_2/kernel/gradientInitializer_107/random_uniform*B
_class8
64loc:@mio_variable/expand_xtr/dense_2/kernel/gradient*
validate_shape(*
use_locking(*
T0

-mio_variable/expand_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerexpand_xtr/dense_2/bias*
shape:@

-mio_variable/expand_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*&
	containerexpand_xtr/dense_2/bias
F
Initializer_108/zerosConst*
valueB@*    *
dtype0
Î

Assign_108Assign-mio_variable/expand_xtr/dense_2/bias/gradientInitializer_108/zeros*
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
expand_xtr/dense_2/BiasAddBiasAddexpand_xtr/dense_2/MatMul-mio_variable/expand_xtr/dense_2/bias/variable*
data_formatNHWC*
T0
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
Y
$Initializer_109/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_109/random_uniform/minConst*
valueB
 *ž*
dtype0
O
"Initializer_109/random_uniform/maxConst*
valueB
 *>*
dtype0

,Initializer_109/random_uniform/RandomUniformRandomUniform$Initializer_109/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_109/random_uniform/subSub"Initializer_109/random_uniform/max"Initializer_109/random_uniform/min*
T0

"Initializer_109/random_uniform/mulMul,Initializer_109/random_uniform/RandomUniform"Initializer_109/random_uniform/sub*
T0
v
Initializer_109/random_uniformAdd"Initializer_109/random_uniform/mul"Initializer_109/random_uniform/min*
T0
Û

Assign_109Assign/mio_variable/expand_xtr/dense_3/kernel/gradientInitializer_109/random_uniform*
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
F
Initializer_110/zerosConst*
valueB*    *
dtype0
Î

Assign_110Assign-mio_variable/expand_xtr/dense_3/bias/gradientInitializer_110/zeros*
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
+mio_variable/like_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerlike_xtr/dense/kernel*
shape:
°
Y
$Initializer_111/random_uniform/shapeConst*
valueB"°     *
dtype0
O
"Initializer_111/random_uniform/minConst*
valueB
 *ÃĐŊ*
dtype0
O
"Initializer_111/random_uniform/maxConst*
valueB
 *ÃĐ=*
dtype0

,Initializer_111/random_uniform/RandomUniformRandomUniform$Initializer_111/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_111/random_uniform/subSub"Initializer_111/random_uniform/max"Initializer_111/random_uniform/min*
T0

"Initializer_111/random_uniform/mulMul,Initializer_111/random_uniform/RandomUniform"Initializer_111/random_uniform/sub*
T0
v
Initializer_111/random_uniformAdd"Initializer_111/random_uniform/mul"Initializer_111/random_uniform/min*
T0
Ķ

Assign_111Assign+mio_variable/like_xtr/dense/kernel/gradientInitializer_111/random_uniform*
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
)mio_variable/like_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*"
	containerlike_xtr/dense/bias*
shape:
G
Initializer_112/zerosConst*
valueB*    *
dtype0
Æ

Assign_112Assign)mio_variable/like_xtr/dense/bias/gradientInitializer_112/zeros*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@mio_variable/like_xtr/dense/bias/gradient

like_xtr/dense/MatMulMatMulconcat_1+mio_variable/like_xtr/dense/kernel/variable*
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
-mio_variable/like_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerlike_xtr/dense_1/kernel*
shape:

 
-mio_variable/like_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerlike_xtr/dense_1/kernel*
shape:

Y
$Initializer_113/random_uniform/shapeConst*
valueB"      *
dtype0
O
"Initializer_113/random_uniform/minConst*
dtype0*
valueB
 *   ž
O
"Initializer_113/random_uniform/maxConst*
valueB
 *   >*
dtype0

,Initializer_113/random_uniform/RandomUniformRandomUniform$Initializer_113/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_113/random_uniform/subSub"Initializer_113/random_uniform/max"Initializer_113/random_uniform/min*
T0

"Initializer_113/random_uniform/mulMul,Initializer_113/random_uniform/RandomUniform"Initializer_113/random_uniform/sub*
T0
v
Initializer_113/random_uniformAdd"Initializer_113/random_uniform/mul"Initializer_113/random_uniform/min*
T0
×

Assign_113Assign-mio_variable/like_xtr/dense_1/kernel/gradientInitializer_113/random_uniform*@
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
+mio_variable/like_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerlike_xtr/dense_1/bias*
shape:
G
Initializer_114/zerosConst*
valueB*    *
dtype0
Ę

Assign_114Assign+mio_variable/like_xtr/dense_1/bias/gradientInitializer_114/zeros*
T0*>
_class4
20loc:@mio_variable/like_xtr/dense_1/bias/gradient*
validate_shape(*
use_locking(

like_xtr/dense_1/MatMulMatMullike_xtr/dropout/Identity-mio_variable/like_xtr/dense_1/kernel/variable*
T0*
transpose_a( *
transpose_b( 

like_xtr/dense_1/BiasAddBiasAddlike_xtr/dense_1/MatMul+mio_variable/like_xtr/dense_1/bias/variable*
data_formatNHWC*
T0
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
-mio_variable/like_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerlike_xtr/dense_2/kernel*
shape:	@
Y
$Initializer_115/random_uniform/shapeConst*
valueB"   @   *
dtype0
O
"Initializer_115/random_uniform/minConst*
dtype0*
valueB
 *ķ5ž
O
"Initializer_115/random_uniform/maxConst*
dtype0*
valueB
 *ķ5>

,Initializer_115/random_uniform/RandomUniformRandomUniform$Initializer_115/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_115/random_uniform/subSub"Initializer_115/random_uniform/max"Initializer_115/random_uniform/min*
T0

"Initializer_115/random_uniform/mulMul,Initializer_115/random_uniform/RandomUniform"Initializer_115/random_uniform/sub*
T0
v
Initializer_115/random_uniformAdd"Initializer_115/random_uniform/mul"Initializer_115/random_uniform/min*
T0
×

Assign_115Assign-mio_variable/like_xtr/dense_2/kernel/gradientInitializer_115/random_uniform*
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
F
Initializer_116/zerosConst*
valueB@*    *
dtype0
Ę

Assign_116Assign+mio_variable/like_xtr/dense_2/bias/gradientInitializer_116/zeros*
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
Y
$Initializer_117/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_117/random_uniform/minConst*
valueB
 *ž*
dtype0
O
"Initializer_117/random_uniform/maxConst*
valueB
 *>*
dtype0

,Initializer_117/random_uniform/RandomUniformRandomUniform$Initializer_117/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_117/random_uniform/subSub"Initializer_117/random_uniform/max"Initializer_117/random_uniform/min*
T0

"Initializer_117/random_uniform/mulMul,Initializer_117/random_uniform/RandomUniform"Initializer_117/random_uniform/sub*
T0
v
Initializer_117/random_uniformAdd"Initializer_117/random_uniform/mul"Initializer_117/random_uniform/min*
T0
×

Assign_117Assign-mio_variable/like_xtr/dense_3/kernel/gradientInitializer_117/random_uniform*@
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
F
Initializer_118/zerosConst*
valueB*    *
dtype0
Ę

Assign_118Assign+mio_variable/like_xtr/dense_3/bias/gradientInitializer_118/zeros*
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
Y
$Initializer_119/random_uniform/shapeConst*
valueB"°     *
dtype0
O
"Initializer_119/random_uniform/minConst*
valueB
 *ÃĐŊ*
dtype0
O
"Initializer_119/random_uniform/maxConst*
valueB
 *ÃĐ=*
dtype0

,Initializer_119/random_uniform/RandomUniformRandomUniform$Initializer_119/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_119/random_uniform/subSub"Initializer_119/random_uniform/max"Initializer_119/random_uniform/min*
T0

"Initializer_119/random_uniform/mulMul,Initializer_119/random_uniform/RandomUniform"Initializer_119/random_uniform/sub*
T0
v
Initializer_119/random_uniformAdd"Initializer_119/random_uniform/mul"Initializer_119/random_uniform/min*
T0
Õ

Assign_119Assign,mio_variable/reply_xtr/dense/kernel/gradientInitializer_119/random_uniform*?
_class5
31loc:@mio_variable/reply_xtr/dense/kernel/gradient*
validate_shape(*
use_locking(*
T0

*mio_variable/reply_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerreply_xtr/dense/bias*
shape:

*mio_variable/reply_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*#
	containerreply_xtr/dense/bias
G
Initializer_120/zerosConst*
dtype0*
valueB*    
Č

Assign_120Assign*mio_variable/reply_xtr/dense/bias/gradientInitializer_120/zeros*
use_locking(*
T0*=
_class3
1/loc:@mio_variable/reply_xtr/dense/bias/gradient*
validate_shape(

reply_xtr/dense/MatMulMatMulconcat_1,mio_variable/reply_xtr/dense/kernel/variable*
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
J
reply_xtr/dropout/IdentityIdentityreply_xtr/dense/LeakyRelu*
T0
ĸ
.mio_variable/reply_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerreply_xtr/dense_1/kernel*
shape:

ĸ
.mio_variable/reply_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerreply_xtr/dense_1/kernel*
shape:

Y
$Initializer_121/random_uniform/shapeConst*
dtype0*
valueB"      
O
"Initializer_121/random_uniform/minConst*
valueB
 *   ž*
dtype0
O
"Initializer_121/random_uniform/maxConst*
valueB
 *   >*
dtype0

,Initializer_121/random_uniform/RandomUniformRandomUniform$Initializer_121/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_121/random_uniform/subSub"Initializer_121/random_uniform/max"Initializer_121/random_uniform/min*
T0

"Initializer_121/random_uniform/mulMul,Initializer_121/random_uniform/RandomUniform"Initializer_121/random_uniform/sub*
T0
v
Initializer_121/random_uniformAdd"Initializer_121/random_uniform/mul"Initializer_121/random_uniform/min*
T0
Ų

Assign_121Assign.mio_variable/reply_xtr/dense_1/kernel/gradientInitializer_121/random_uniform*A
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
,mio_variable/reply_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*%
	containerreply_xtr/dense_1/bias
G
Initializer_122/zerosConst*
valueB*    *
dtype0
Ė

Assign_122Assign,mio_variable/reply_xtr/dense_1/bias/gradientInitializer_122/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/reply_xtr/dense_1/bias/gradient*
validate_shape(
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
Y
$Initializer_123/random_uniform/shapeConst*
valueB"   @   *
dtype0
O
"Initializer_123/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
O
"Initializer_123/random_uniform/maxConst*
valueB
 *ķ5>*
dtype0

,Initializer_123/random_uniform/RandomUniformRandomUniform$Initializer_123/random_uniform/shape*
seed2 *

seed *
T0*
dtype0
z
"Initializer_123/random_uniform/subSub"Initializer_123/random_uniform/max"Initializer_123/random_uniform/min*
T0

"Initializer_123/random_uniform/mulMul,Initializer_123/random_uniform/RandomUniform"Initializer_123/random_uniform/sub*
T0
v
Initializer_123/random_uniformAdd"Initializer_123/random_uniform/mul"Initializer_123/random_uniform/min*
T0
Ų

Assign_123Assign.mio_variable/reply_xtr/dense_2/kernel/gradientInitializer_123/random_uniform*
validate_shape(*
use_locking(*
T0*A
_class7
53loc:@mio_variable/reply_xtr/dense_2/kernel/gradient

,mio_variable/reply_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense_2/bias*
shape:@

,mio_variable/reply_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense_2/bias*
shape:@
F
Initializer_124/zerosConst*
valueB@*    *
dtype0
Ė

Assign_124Assign,mio_variable/reply_xtr/dense_2/bias/gradientInitializer_124/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/reply_xtr/dense_2/bias/gradient*
validate_shape(
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
Y
$Initializer_125/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_125/random_uniform/minConst*
valueB
 *ž*
dtype0
O
"Initializer_125/random_uniform/maxConst*
valueB
 *>*
dtype0

,Initializer_125/random_uniform/RandomUniformRandomUniform$Initializer_125/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_125/random_uniform/subSub"Initializer_125/random_uniform/max"Initializer_125/random_uniform/min*
T0

"Initializer_125/random_uniform/mulMul,Initializer_125/random_uniform/RandomUniform"Initializer_125/random_uniform/sub*
T0
v
Initializer_125/random_uniformAdd"Initializer_125/random_uniform/mul"Initializer_125/random_uniform/min*
T0
Ų

Assign_125Assign.mio_variable/reply_xtr/dense_3/kernel/gradientInitializer_125/random_uniform*A
_class7
53loc:@mio_variable/reply_xtr/dense_3/kernel/gradient*
validate_shape(*
use_locking(*
T0

,mio_variable/reply_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*%
	containerreply_xtr/dense_3/bias

,mio_variable/reply_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense_3/bias*
shape:
F
Initializer_126/zerosConst*
valueB*    *
dtype0
Ė

Assign_126Assign,mio_variable/reply_xtr/dense_3/bias/gradientInitializer_126/zeros*
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
°

+mio_variable/copy_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense/kernel*
shape:
°
Y
$Initializer_127/random_uniform/shapeConst*
valueB"°     *
dtype0
O
"Initializer_127/random_uniform/minConst*
valueB
 *ÃĐŊ*
dtype0
O
"Initializer_127/random_uniform/maxConst*
valueB
 *ÃĐ=*
dtype0

,Initializer_127/random_uniform/RandomUniformRandomUniform$Initializer_127/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_127/random_uniform/subSub"Initializer_127/random_uniform/max"Initializer_127/random_uniform/min*
T0

"Initializer_127/random_uniform/mulMul,Initializer_127/random_uniform/RandomUniform"Initializer_127/random_uniform/sub*
T0
v
Initializer_127/random_uniformAdd"Initializer_127/random_uniform/mul"Initializer_127/random_uniform/min*
T0
Ķ

Assign_127Assign+mio_variable/copy_xtr/dense/kernel/gradientInitializer_127/random_uniform*>
_class4
20loc:@mio_variable/copy_xtr/dense/kernel/gradient*
validate_shape(*
use_locking(*
T0

)mio_variable/copy_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*"
	containercopy_xtr/dense/bias*
shape:

)mio_variable/copy_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*"
	containercopy_xtr/dense/bias*
shape:
G
Initializer_128/zerosConst*
dtype0*
valueB*    
Æ

Assign_128Assign)mio_variable/copy_xtr/dense/bias/gradientInitializer_128/zeros*
use_locking(*
T0*<
_class2
0.loc:@mio_variable/copy_xtr/dense/bias/gradient*
validate_shape(
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
Y
$Initializer_129/random_uniform/shapeConst*
valueB"      *
dtype0
O
"Initializer_129/random_uniform/minConst*
valueB
 *   ž*
dtype0
O
"Initializer_129/random_uniform/maxConst*
valueB
 *   >*
dtype0

,Initializer_129/random_uniform/RandomUniformRandomUniform$Initializer_129/random_uniform/shape*
seed2 *

seed *
T0*
dtype0
z
"Initializer_129/random_uniform/subSub"Initializer_129/random_uniform/max"Initializer_129/random_uniform/min*
T0

"Initializer_129/random_uniform/mulMul,Initializer_129/random_uniform/RandomUniform"Initializer_129/random_uniform/sub*
T0
v
Initializer_129/random_uniformAdd"Initializer_129/random_uniform/mul"Initializer_129/random_uniform/min*
T0
×

Assign_129Assign-mio_variable/copy_xtr/dense_1/kernel/gradientInitializer_129/random_uniform*
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
G
Initializer_130/zerosConst*
valueB*    *
dtype0
Ę

Assign_130Assign+mio_variable/copy_xtr/dense_1/bias/gradientInitializer_130/zeros*
use_locking(*
T0*>
_class4
20loc:@mio_variable/copy_xtr/dense_1/bias/gradient*
validate_shape(

copy_xtr/dense_1/MatMulMatMulcopy_xtr/dropout/Identity-mio_variable/copy_xtr/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
-mio_variable/copy_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*&
	containercopy_xtr/dense_2/kernel

-mio_variable/copy_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containercopy_xtr/dense_2/kernel*
shape:	@
Y
$Initializer_131/random_uniform/shapeConst*
valueB"   @   *
dtype0
O
"Initializer_131/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
O
"Initializer_131/random_uniform/maxConst*
valueB
 *ķ5>*
dtype0

,Initializer_131/random_uniform/RandomUniformRandomUniform$Initializer_131/random_uniform/shape*
seed2 *

seed *
T0*
dtype0
z
"Initializer_131/random_uniform/subSub"Initializer_131/random_uniform/max"Initializer_131/random_uniform/min*
T0

"Initializer_131/random_uniform/mulMul,Initializer_131/random_uniform/RandomUniform"Initializer_131/random_uniform/sub*
T0
v
Initializer_131/random_uniformAdd"Initializer_131/random_uniform/mul"Initializer_131/random_uniform/min*
T0
×

Assign_131Assign-mio_variable/copy_xtr/dense_2/kernel/gradientInitializer_131/random_uniform*
T0*@
_class6
42loc:@mio_variable/copy_xtr/dense_2/kernel/gradient*
validate_shape(*
use_locking(

+mio_variable/copy_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense_2/bias*
shape:@

+mio_variable/copy_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense_2/bias*
shape:@
F
Initializer_132/zerosConst*
valueB@*    *
dtype0
Ę

Assign_132Assign+mio_variable/copy_xtr/dense_2/bias/gradientInitializer_132/zeros*
use_locking(*
T0*>
_class4
20loc:@mio_variable/copy_xtr/dense_2/bias/gradient*
validate_shape(

copy_xtr/dense_2/MatMulMatMulcopy_xtr/dropout_1/Identity-mio_variable/copy_xtr/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
Y
$Initializer_133/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_133/random_uniform/minConst*
valueB
 *ž*
dtype0
O
"Initializer_133/random_uniform/maxConst*
valueB
 *>*
dtype0

,Initializer_133/random_uniform/RandomUniformRandomUniform$Initializer_133/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_133/random_uniform/subSub"Initializer_133/random_uniform/max"Initializer_133/random_uniform/min*
T0

"Initializer_133/random_uniform/mulMul,Initializer_133/random_uniform/RandomUniform"Initializer_133/random_uniform/sub*
T0
v
Initializer_133/random_uniformAdd"Initializer_133/random_uniform/mul"Initializer_133/random_uniform/min*
T0
×

Assign_133Assign-mio_variable/copy_xtr/dense_3/kernel/gradientInitializer_133/random_uniform*
use_locking(*
T0*@
_class6
42loc:@mio_variable/copy_xtr/dense_3/kernel/gradient*
validate_shape(

+mio_variable/copy_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containercopy_xtr/dense_3/bias

+mio_variable/copy_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense_3/bias*
shape:
F
Initializer_134/zerosConst*
dtype0*
valueB*    
Ę

Assign_134Assign+mio_variable/copy_xtr/dense_3/bias/gradientInitializer_134/zeros*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@mio_variable/copy_xtr/dense_3/bias/gradient
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
,mio_variable/share_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*%
	containershare_xtr/dense/kernel

,mio_variable/share_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containershare_xtr/dense/kernel*
shape:
°
Y
$Initializer_135/random_uniform/shapeConst*
valueB"°     *
dtype0
O
"Initializer_135/random_uniform/minConst*
valueB
 *ÃĐŊ*
dtype0
O
"Initializer_135/random_uniform/maxConst*
valueB
 *ÃĐ=*
dtype0

,Initializer_135/random_uniform/RandomUniformRandomUniform$Initializer_135/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_135/random_uniform/subSub"Initializer_135/random_uniform/max"Initializer_135/random_uniform/min*
T0

"Initializer_135/random_uniform/mulMul,Initializer_135/random_uniform/RandomUniform"Initializer_135/random_uniform/sub*
T0
v
Initializer_135/random_uniformAdd"Initializer_135/random_uniform/mul"Initializer_135/random_uniform/min*
T0
Õ

Assign_135Assign,mio_variable/share_xtr/dense/kernel/gradientInitializer_135/random_uniform*
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
G
Initializer_136/zerosConst*
valueB*    *
dtype0
Č

Assign_136Assign*mio_variable/share_xtr/dense/bias/gradientInitializer_136/zeros*=
_class3
1/loc:@mio_variable/share_xtr/dense/bias/gradient*
validate_shape(*
use_locking(*
T0

share_xtr/dense/MatMulMatMulconcat_1,mio_variable/share_xtr/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
Y
$Initializer_137/random_uniform/shapeConst*
valueB"      *
dtype0
O
"Initializer_137/random_uniform/minConst*
valueB
 *   ž*
dtype0
O
"Initializer_137/random_uniform/maxConst*
dtype0*
valueB
 *   >

,Initializer_137/random_uniform/RandomUniformRandomUniform$Initializer_137/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_137/random_uniform/subSub"Initializer_137/random_uniform/max"Initializer_137/random_uniform/min*
T0

"Initializer_137/random_uniform/mulMul,Initializer_137/random_uniform/RandomUniform"Initializer_137/random_uniform/sub*
T0
v
Initializer_137/random_uniformAdd"Initializer_137/random_uniform/mul"Initializer_137/random_uniform/min*
T0
Ų

Assign_137Assign.mio_variable/share_xtr/dense_1/kernel/gradientInitializer_137/random_uniform*
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
G
Initializer_138/zerosConst*
valueB*    *
dtype0
Ė

Assign_138Assign,mio_variable/share_xtr/dense_1/bias/gradientInitializer_138/zeros*
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
Y
$Initializer_139/random_uniform/shapeConst*
valueB"   @   *
dtype0
O
"Initializer_139/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
O
"Initializer_139/random_uniform/maxConst*
valueB
 *ķ5>*
dtype0

,Initializer_139/random_uniform/RandomUniformRandomUniform$Initializer_139/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_139/random_uniform/subSub"Initializer_139/random_uniform/max"Initializer_139/random_uniform/min*
T0

"Initializer_139/random_uniform/mulMul,Initializer_139/random_uniform/RandomUniform"Initializer_139/random_uniform/sub*
T0
v
Initializer_139/random_uniformAdd"Initializer_139/random_uniform/mul"Initializer_139/random_uniform/min*
T0
Ų

Assign_139Assign.mio_variable/share_xtr/dense_2/kernel/gradientInitializer_139/random_uniform*
use_locking(*
T0*A
_class7
53loc:@mio_variable/share_xtr/dense_2/kernel/gradient*
validate_shape(

,mio_variable/share_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*%
	containershare_xtr/dense_2/bias

,mio_variable/share_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containershare_xtr/dense_2/bias*
shape:@
F
Initializer_140/zerosConst*
valueB@*    *
dtype0
Ė

Assign_140Assign,mio_variable/share_xtr/dense_2/bias/gradientInitializer_140/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/share_xtr/dense_2/bias/gradient*
validate_shape(

share_xtr/dense_2/MatMulMatMulshare_xtr/dropout_1/Identity.mio_variable/share_xtr/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0

share_xtr/dense_2/BiasAddBiasAddshare_xtr/dense_2/MatMul,mio_variable/share_xtr/dense_2/bias/variable*
T0*
data_formatNHWC
N
!share_xtr/dense_2/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍĖL>
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
Y
$Initializer_141/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_141/random_uniform/minConst*
valueB
 *ž*
dtype0
O
"Initializer_141/random_uniform/maxConst*
valueB
 *>*
dtype0

,Initializer_141/random_uniform/RandomUniformRandomUniform$Initializer_141/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_141/random_uniform/subSub"Initializer_141/random_uniform/max"Initializer_141/random_uniform/min*
T0

"Initializer_141/random_uniform/mulMul,Initializer_141/random_uniform/RandomUniform"Initializer_141/random_uniform/sub*
T0
v
Initializer_141/random_uniformAdd"Initializer_141/random_uniform/mul"Initializer_141/random_uniform/min*
T0
Ų

Assign_141Assign.mio_variable/share_xtr/dense_3/kernel/gradientInitializer_141/random_uniform*
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
,mio_variable/share_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*%
	containershare_xtr/dense_3/bias
F
Initializer_142/zerosConst*
valueB*    *
dtype0
Ė

Assign_142Assign,mio_variable/share_xtr/dense_3/bias/gradientInitializer_142/zeros*?
_class5
31loc:@mio_variable/share_xtr/dense_3/bias/gradient*
validate_shape(*
use_locking(*
T0
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
Y
$Initializer_143/random_uniform/shapeConst*
dtype0*
valueB"°     
O
"Initializer_143/random_uniform/minConst*
valueB
 *ÃĐŊ*
dtype0
O
"Initializer_143/random_uniform/maxConst*
valueB
 *ÃĐ=*
dtype0

,Initializer_143/random_uniform/RandomUniformRandomUniform$Initializer_143/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_143/random_uniform/subSub"Initializer_143/random_uniform/max"Initializer_143/random_uniform/min*
T0

"Initializer_143/random_uniform/mulMul,Initializer_143/random_uniform/RandomUniform"Initializer_143/random_uniform/sub*
T0
v
Initializer_143/random_uniformAdd"Initializer_143/random_uniform/mul"Initializer_143/random_uniform/min*
T0
Û

Assign_143Assign/mio_variable/audience_xtr/dense/kernel/gradientInitializer_143/random_uniform*
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
G
Initializer_144/zerosConst*
dtype0*
valueB*    
Î

Assign_144Assign-mio_variable/audience_xtr/dense/bias/gradientInitializer_144/zeros*
use_locking(*
T0*@
_class6
42loc:@mio_variable/audience_xtr/dense/bias/gradient*
validate_shape(

audience_xtr/dense/MatMulMatMulconcat_1/mio_variable/audience_xtr/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
Y
$Initializer_145/random_uniform/shapeConst*
valueB"      *
dtype0
O
"Initializer_145/random_uniform/minConst*
valueB
 *   ž*
dtype0
O
"Initializer_145/random_uniform/maxConst*
dtype0*
valueB
 *   >

,Initializer_145/random_uniform/RandomUniformRandomUniform$Initializer_145/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_145/random_uniform/subSub"Initializer_145/random_uniform/max"Initializer_145/random_uniform/min*
T0

"Initializer_145/random_uniform/mulMul,Initializer_145/random_uniform/RandomUniform"Initializer_145/random_uniform/sub*
T0
v
Initializer_145/random_uniformAdd"Initializer_145/random_uniform/mul"Initializer_145/random_uniform/min*
T0
ß

Assign_145Assign1mio_variable/audience_xtr/dense_1/kernel/gradientInitializer_145/random_uniform*
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
G
Initializer_146/zerosConst*
valueB*    *
dtype0
Ō

Assign_146Assign/mio_variable/audience_xtr/dense_1/bias/gradientInitializer_146/zeros*
validate_shape(*
use_locking(*
T0*B
_class8
64loc:@mio_variable/audience_xtr/dense_1/bias/gradient
Ļ
audience_xtr/dense_1/MatMulMatMulaudience_xtr/dropout/Identity1mio_variable/audience_xtr/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0

audience_xtr/dense_1/BiasAddBiasAddaudience_xtr/dense_1/MatMul/mio_variable/audience_xtr/dense_1/bias/variable*
T0*
data_formatNHWC
Q
$audience_xtr/dense_1/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍĖL>
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
Y
$Initializer_147/random_uniform/shapeConst*
valueB"   @   *
dtype0
O
"Initializer_147/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
O
"Initializer_147/random_uniform/maxConst*
valueB
 *ķ5>*
dtype0

,Initializer_147/random_uniform/RandomUniformRandomUniform$Initializer_147/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_147/random_uniform/subSub"Initializer_147/random_uniform/max"Initializer_147/random_uniform/min*
T0

"Initializer_147/random_uniform/mulMul,Initializer_147/random_uniform/RandomUniform"Initializer_147/random_uniform/sub*
T0
v
Initializer_147/random_uniformAdd"Initializer_147/random_uniform/mul"Initializer_147/random_uniform/min*
T0
ß

Assign_147Assign1mio_variable/audience_xtr/dense_2/kernel/gradientInitializer_147/random_uniform*
validate_shape(*
use_locking(*
T0*D
_class:
86loc:@mio_variable/audience_xtr/dense_2/kernel/gradient

/mio_variable/audience_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeraudience_xtr/dense_2/bias*
shape:@

/mio_variable/audience_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeraudience_xtr/dense_2/bias*
shape:@
F
Initializer_148/zerosConst*
valueB@*    *
dtype0
Ō

Assign_148Assign/mio_variable/audience_xtr/dense_2/bias/gradientInitializer_148/zeros*
use_locking(*
T0*B
_class8
64loc:@mio_variable/audience_xtr/dense_2/bias/gradient*
validate_shape(
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
 *ÍĖL>*
dtype0
v
"audience_xtr/dense_2/LeakyRelu/mulMul$audience_xtr/dense_2/LeakyRelu/alphaaudience_xtr/dense_2/BiasAdd*
T0
t
audience_xtr/dense_2/LeakyReluMaximum"audience_xtr/dense_2/LeakyRelu/mulaudience_xtr/dense_2/BiasAdd*
T0
Ļ
1mio_variable/audience_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS**
	containeraudience_xtr/dense_3/kernel*
shape
:@
Ļ
1mio_variable/audience_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS**
	containeraudience_xtr/dense_3/kernel*
shape
:@
Y
$Initializer_149/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_149/random_uniform/minConst*
valueB
 *ž*
dtype0
O
"Initializer_149/random_uniform/maxConst*
valueB
 *>*
dtype0

,Initializer_149/random_uniform/RandomUniformRandomUniform$Initializer_149/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_149/random_uniform/subSub"Initializer_149/random_uniform/max"Initializer_149/random_uniform/min*
T0

"Initializer_149/random_uniform/mulMul,Initializer_149/random_uniform/RandomUniform"Initializer_149/random_uniform/sub*
T0
v
Initializer_149/random_uniformAdd"Initializer_149/random_uniform/mul"Initializer_149/random_uniform/min*
T0
ß

Assign_149Assign1mio_variable/audience_xtr/dense_3/kernel/gradientInitializer_149/random_uniform*
use_locking(*
T0*D
_class:
86loc:@mio_variable/audience_xtr/dense_3/kernel/gradient*
validate_shape(

/mio_variable/audience_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeraudience_xtr/dense_3/bias*
shape:

/mio_variable/audience_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containeraudience_xtr/dense_3/bias*
shape:
F
Initializer_150/zerosConst*
dtype0*
valueB*    
Ō

Assign_150Assign/mio_variable/audience_xtr/dense_3/bias/gradientInitializer_150/zeros*
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
ļ
8mio_variable/continuous_expand_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*1
	container$"continuous_expand_xtr/dense/kernel
ļ
8mio_variable/continuous_expand_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*1
	container$"continuous_expand_xtr/dense/kernel
Y
$Initializer_151/random_uniform/shapeConst*
dtype0*
valueB"°     
O
"Initializer_151/random_uniform/minConst*
valueB
 *ÃĐŊ*
dtype0
O
"Initializer_151/random_uniform/maxConst*
valueB
 *ÃĐ=*
dtype0

,Initializer_151/random_uniform/RandomUniformRandomUniform$Initializer_151/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_151/random_uniform/subSub"Initializer_151/random_uniform/max"Initializer_151/random_uniform/min*
T0

"Initializer_151/random_uniform/mulMul,Initializer_151/random_uniform/RandomUniform"Initializer_151/random_uniform/sub*
T0
v
Initializer_151/random_uniformAdd"Initializer_151/random_uniform/mul"Initializer_151/random_uniform/min*
T0
í

Assign_151Assign8mio_variable/continuous_expand_xtr/dense/kernel/gradientInitializer_151/random_uniform*
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
G
Initializer_152/zerosConst*
valueB*    *
dtype0
ā

Assign_152Assign6mio_variable/continuous_expand_xtr/dense/bias/gradientInitializer_152/zeros*
use_locking(*
T0*I
_class?
=;loc:@mio_variable/continuous_expand_xtr/dense/bias/gradient*
validate_shape(
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
:mio_variable/continuous_expand_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*3
	container&$continuous_expand_xtr/dense_1/kernel
Y
$Initializer_153/random_uniform/shapeConst*
valueB"      *
dtype0
O
"Initializer_153/random_uniform/minConst*
valueB
 *   ž*
dtype0
O
"Initializer_153/random_uniform/maxConst*
dtype0*
valueB
 *   >

,Initializer_153/random_uniform/RandomUniformRandomUniform$Initializer_153/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_153/random_uniform/subSub"Initializer_153/random_uniform/max"Initializer_153/random_uniform/min*
T0

"Initializer_153/random_uniform/mulMul,Initializer_153/random_uniform/RandomUniform"Initializer_153/random_uniform/sub*
T0
v
Initializer_153/random_uniformAdd"Initializer_153/random_uniform/mul"Initializer_153/random_uniform/min*
T0
ņ

Assign_153Assign:mio_variable/continuous_expand_xtr/dense_1/kernel/gradientInitializer_153/random_uniform*
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
G
Initializer_154/zerosConst*
valueB*    *
dtype0
ä

Assign_154Assign8mio_variable/continuous_expand_xtr/dense_1/bias/gradientInitializer_154/zeros*
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
%continuous_expand_xtr/dense_1/BiasAddBiasAdd$continuous_expand_xtr/dense_1/MatMul8mio_variable/continuous_expand_xtr/dense_1/bias/variable*
data_formatNHWC*
T0
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
Y
$Initializer_155/random_uniform/shapeConst*
valueB"   @   *
dtype0
O
"Initializer_155/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
O
"Initializer_155/random_uniform/maxConst*
valueB
 *ķ5>*
dtype0

,Initializer_155/random_uniform/RandomUniformRandomUniform$Initializer_155/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_155/random_uniform/subSub"Initializer_155/random_uniform/max"Initializer_155/random_uniform/min*
T0

"Initializer_155/random_uniform/mulMul,Initializer_155/random_uniform/RandomUniform"Initializer_155/random_uniform/sub*
T0
v
Initializer_155/random_uniformAdd"Initializer_155/random_uniform/mul"Initializer_155/random_uniform/min*
T0
ņ

Assign_155Assign:mio_variable/continuous_expand_xtr/dense_2/kernel/gradientInitializer_155/random_uniform*
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
F
Initializer_156/zerosConst*
valueB@*    *
dtype0
ä

Assign_156Assign8mio_variable/continuous_expand_xtr/dense_2/bias/gradientInitializer_156/zeros*
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
Y
$Initializer_157/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_157/random_uniform/minConst*
dtype0*
valueB
 *ž
O
"Initializer_157/random_uniform/maxConst*
valueB
 *>*
dtype0

,Initializer_157/random_uniform/RandomUniformRandomUniform$Initializer_157/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_157/random_uniform/subSub"Initializer_157/random_uniform/max"Initializer_157/random_uniform/min*
T0

"Initializer_157/random_uniform/mulMul,Initializer_157/random_uniform/RandomUniform"Initializer_157/random_uniform/sub*
T0
v
Initializer_157/random_uniformAdd"Initializer_157/random_uniform/mul"Initializer_157/random_uniform/min*
T0
ņ

Assign_157Assign:mio_variable/continuous_expand_xtr/dense_3/kernel/gradientInitializer_157/random_uniform*
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
F
Initializer_158/zerosConst*
valueB*    *
dtype0
ä

Assign_158Assign8mio_variable/continuous_expand_xtr/dense_3/bias/gradientInitializer_158/zeros*
use_locking(*
T0*K
_classA
?=loc:@mio_variable/continuous_expand_xtr/dense_3/bias/gradient*
validate_shape(
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
Ŧ
3mio_variable/duration_predict/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerduration_predict/dense/kernel*
shape:
°
Ŧ
3mio_variable/duration_predict/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*,
	containerduration_predict/dense/kernel*
shape:
°
Y
$Initializer_159/random_uniform/shapeConst*
valueB"°     *
dtype0
O
"Initializer_159/random_uniform/minConst*
valueB
 *ÃĐŊ*
dtype0
O
"Initializer_159/random_uniform/maxConst*
valueB
 *ÃĐ=*
dtype0

,Initializer_159/random_uniform/RandomUniformRandomUniform$Initializer_159/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_159/random_uniform/subSub"Initializer_159/random_uniform/max"Initializer_159/random_uniform/min*
T0

"Initializer_159/random_uniform/mulMul,Initializer_159/random_uniform/RandomUniform"Initializer_159/random_uniform/sub*
T0
v
Initializer_159/random_uniformAdd"Initializer_159/random_uniform/mul"Initializer_159/random_uniform/min*
T0
ã

Assign_159Assign3mio_variable/duration_predict/dense/kernel/gradientInitializer_159/random_uniform*
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
G
Initializer_160/zerosConst*
valueB*    *
dtype0
Ö

Assign_160Assign1mio_variable/duration_predict/dense/bias/gradientInitializer_160/zeros*
use_locking(*
T0*D
_class:
86loc:@mio_variable/duration_predict/dense/bias/gradient*
validate_shape(

duration_predict/dense/MatMulMatMulconcat_13mio_variable/duration_predict/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
5mio_variable/duration_predict/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*.
	container!duration_predict/dense_1/kernel
°
5mio_variable/duration_predict/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!duration_predict/dense_1/kernel*
shape:

Y
$Initializer_161/random_uniform/shapeConst*
valueB"      *
dtype0
O
"Initializer_161/random_uniform/minConst*
valueB
 *   ž*
dtype0
O
"Initializer_161/random_uniform/maxConst*
dtype0*
valueB
 *   >

,Initializer_161/random_uniform/RandomUniformRandomUniform$Initializer_161/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_161/random_uniform/subSub"Initializer_161/random_uniform/max"Initializer_161/random_uniform/min*
T0

"Initializer_161/random_uniform/mulMul,Initializer_161/random_uniform/RandomUniform"Initializer_161/random_uniform/sub*
T0
v
Initializer_161/random_uniformAdd"Initializer_161/random_uniform/mul"Initializer_161/random_uniform/min*
T0
į

Assign_161Assign5mio_variable/duration_predict/dense_1/kernel/gradientInitializer_161/random_uniform*
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
3mio_variable/duration_predict/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*,
	containerduration_predict/dense_1/bias
G
Initializer_162/zerosConst*
valueB*    *
dtype0
Ú

Assign_162Assign3mio_variable/duration_predict/dense_1/bias/gradientInitializer_162/zeros*
validate_shape(*
use_locking(*
T0*F
_class<
:8loc:@mio_variable/duration_predict/dense_1/bias/gradient
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
Y
$Initializer_163/random_uniform/shapeConst*
valueB"   @   *
dtype0
O
"Initializer_163/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
O
"Initializer_163/random_uniform/maxConst*
valueB
 *ķ5>*
dtype0

,Initializer_163/random_uniform/RandomUniformRandomUniform$Initializer_163/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_163/random_uniform/subSub"Initializer_163/random_uniform/max"Initializer_163/random_uniform/min*
T0

"Initializer_163/random_uniform/mulMul,Initializer_163/random_uniform/RandomUniform"Initializer_163/random_uniform/sub*
T0
v
Initializer_163/random_uniformAdd"Initializer_163/random_uniform/mul"Initializer_163/random_uniform/min*
T0
į

Assign_163Assign5mio_variable/duration_predict/dense_2/kernel/gradientInitializer_163/random_uniform*
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
F
Initializer_164/zerosConst*
valueB@*    *
dtype0
Ú

Assign_164Assign3mio_variable/duration_predict/dense_2/bias/gradientInitializer_164/zeros*
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
Ą
 duration_predict/dense_2/BiasAddBiasAddduration_predict/dense_2/MatMul3mio_variable/duration_predict/dense_2/bias/variable*
T0*
data_formatNHWC
U
(duration_predict/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0
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
5mio_variable/duration_predict/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!duration_predict/dense_3/kernel*
shape
:@
Y
$Initializer_165/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_165/random_uniform/minConst*
valueB
 *ž*
dtype0
O
"Initializer_165/random_uniform/maxConst*
valueB
 *>*
dtype0

,Initializer_165/random_uniform/RandomUniformRandomUniform$Initializer_165/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_165/random_uniform/subSub"Initializer_165/random_uniform/max"Initializer_165/random_uniform/min*
T0

"Initializer_165/random_uniform/mulMul,Initializer_165/random_uniform/RandomUniform"Initializer_165/random_uniform/sub*
T0
v
Initializer_165/random_uniformAdd"Initializer_165/random_uniform/mul"Initializer_165/random_uniform/min*
T0
į

Assign_165Assign5mio_variable/duration_predict/dense_3/kernel/gradientInitializer_165/random_uniform*
validate_shape(*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/duration_predict/dense_3/kernel/gradient
Ļ
3mio_variable/duration_predict/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*,
	containerduration_predict/dense_3/bias
Ļ
3mio_variable/duration_predict/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*,
	containerduration_predict/dense_3/bias
F
Initializer_166/zerosConst*
valueB*    *
dtype0
Ú

Assign_166Assign3mio_variable/duration_predict/dense_3/bias/gradientInitializer_166/zeros*
use_locking(*
T0*F
_class<
:8loc:@mio_variable/duration_predict/dense_3/bias/gradient*
validate_shape(
ŗ
duration_predict/dense_3/MatMulMatMul"duration_predict/dense_2/LeakyRelu5mio_variable/duration_predict/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 
Ą
 duration_predict/dense_3/BiasAddBiasAddduration_predict/dense_3/MatMul3mio_variable/duration_predict/dense_3/bias/variable*
T0*
data_formatNHWC
P
duration_predict/dense_3/ReluRelu duration_predict/dense_3/BiasAdd*
T0
ž
<mio_variable/duration_pos_bias_predict/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&duration_pos_bias_predict/dense/kernel*
shape:

ž
<mio_variable/duration_pos_bias_predict/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&duration_pos_bias_predict/dense/kernel*
shape:

Y
$Initializer_167/random_uniform/shapeConst*
dtype0*
valueB"      
O
"Initializer_167/random_uniform/minConst*
valueB
 *˛_ž*
dtype0
O
"Initializer_167/random_uniform/maxConst*
valueB
 *˛_>*
dtype0

,Initializer_167/random_uniform/RandomUniformRandomUniform$Initializer_167/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_167/random_uniform/subSub"Initializer_167/random_uniform/max"Initializer_167/random_uniform/min*
T0

"Initializer_167/random_uniform/mulMul,Initializer_167/random_uniform/RandomUniform"Initializer_167/random_uniform/sub*
T0
v
Initializer_167/random_uniformAdd"Initializer_167/random_uniform/mul"Initializer_167/random_uniform/min*
T0
õ

Assign_167Assign<mio_variable/duration_pos_bias_predict/dense/kernel/gradientInitializer_167/random_uniform*
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
G
Initializer_168/zerosConst*
valueB*    *
dtype0
č

Assign_168Assign:mio_variable/duration_pos_bias_predict/dense/bias/gradientInitializer_168/zeros*
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
/duration_pos_bias_predict/dense/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍĖL>
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
Y
$Initializer_169/random_uniform/shapeConst*
valueB"   @   *
dtype0
O
"Initializer_169/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
O
"Initializer_169/random_uniform/maxConst*
valueB
 *ķ5>*
dtype0

,Initializer_169/random_uniform/RandomUniformRandomUniform$Initializer_169/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_169/random_uniform/subSub"Initializer_169/random_uniform/max"Initializer_169/random_uniform/min*
T0

"Initializer_169/random_uniform/mulMul,Initializer_169/random_uniform/RandomUniform"Initializer_169/random_uniform/sub*
T0
v
Initializer_169/random_uniformAdd"Initializer_169/random_uniform/mul"Initializer_169/random_uniform/min*
T0
ų

Assign_169Assign>mio_variable/duration_pos_bias_predict/dense_1/kernel/gradientInitializer_169/random_uniform*
validate_shape(*
use_locking(*
T0*Q
_classG
ECloc:@mio_variable/duration_pos_bias_predict/dense_1/kernel/gradient
¸
<mio_variable/duration_pos_bias_predict/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&duration_pos_bias_predict/dense_1/bias*
shape:@
¸
<mio_variable/duration_pos_bias_predict/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&duration_pos_bias_predict/dense_1/bias*
shape:@
F
Initializer_170/zerosConst*
valueB@*    *
dtype0
ė

Assign_170Assign<mio_variable/duration_pos_bias_predict/dense_1/bias/gradientInitializer_170/zeros*
validate_shape(*
use_locking(*
T0*O
_classE
CAloc:@mio_variable/duration_pos_bias_predict/dense_1/bias/gradient
Í
(duration_pos_bias_predict/dense_1/MatMulMatMul*duration_pos_bias_predict/dropout/Identity>mio_variable/duration_pos_bias_predict/dense_1/kernel/variable*
transpose_b( *
T0*
transpose_a( 
ŧ
)duration_pos_bias_predict/dense_1/BiasAddBiasAdd(duration_pos_bias_predict/dense_1/MatMul<mio_variable/duration_pos_bias_predict/dense_1/bias/variable*
data_formatNHWC*
T0
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
>mio_variable/duration_pos_bias_predict/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(duration_pos_bias_predict/dense_2/kernel*
shape
:@
Y
$Initializer_171/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_171/random_uniform/minConst*
valueB
 *ž*
dtype0
O
"Initializer_171/random_uniform/maxConst*
valueB
 *>*
dtype0

,Initializer_171/random_uniform/RandomUniformRandomUniform$Initializer_171/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_171/random_uniform/subSub"Initializer_171/random_uniform/max"Initializer_171/random_uniform/min*
T0

"Initializer_171/random_uniform/mulMul,Initializer_171/random_uniform/RandomUniform"Initializer_171/random_uniform/sub*
T0
v
Initializer_171/random_uniformAdd"Initializer_171/random_uniform/mul"Initializer_171/random_uniform/min*
T0
ų

Assign_171Assign>mio_variable/duration_pos_bias_predict/dense_2/kernel/gradientInitializer_171/random_uniform*
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
F
Initializer_172/zerosConst*
dtype0*
valueB*    
ė

Assign_172Assign<mio_variable/duration_pos_bias_predict/dense_2/bias/gradientInitializer_172/zeros*
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
T0

+mio_variable/hate_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerhate_xtr/dense/kernel*
shape:
°

+mio_variable/hate_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerhate_xtr/dense/kernel*
shape:
°
Y
$Initializer_173/random_uniform/shapeConst*
valueB"°     *
dtype0
O
"Initializer_173/random_uniform/minConst*
valueB
 *ÃĐŊ*
dtype0
O
"Initializer_173/random_uniform/maxConst*
valueB
 *ÃĐ=*
dtype0

,Initializer_173/random_uniform/RandomUniformRandomUniform$Initializer_173/random_uniform/shape*
seed2 *

seed *
T0*
dtype0
z
"Initializer_173/random_uniform/subSub"Initializer_173/random_uniform/max"Initializer_173/random_uniform/min*
T0

"Initializer_173/random_uniform/mulMul,Initializer_173/random_uniform/RandomUniform"Initializer_173/random_uniform/sub*
T0
v
Initializer_173/random_uniformAdd"Initializer_173/random_uniform/mul"Initializer_173/random_uniform/min*
T0
Ķ

Assign_173Assign+mio_variable/hate_xtr/dense/kernel/gradientInitializer_173/random_uniform*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@mio_variable/hate_xtr/dense/kernel/gradient

)mio_variable/hate_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*"
	containerhate_xtr/dense/bias*
shape:

)mio_variable/hate_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*"
	containerhate_xtr/dense/bias
G
Initializer_174/zerosConst*
valueB*    *
dtype0
Æ

Assign_174Assign)mio_variable/hate_xtr/dense/bias/gradientInitializer_174/zeros*
T0*<
_class2
0.loc:@mio_variable/hate_xtr/dense/bias/gradient*
validate_shape(*
use_locking(

hate_xtr/dense/MatMulMatMulconcat_1+mio_variable/hate_xtr/dense/kernel/variable*
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
 *ÍĖL>*
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
-mio_variable/hate_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*&
	containerhate_xtr/dense_1/kernel
Y
$Initializer_175/random_uniform/shapeConst*
valueB"      *
dtype0
O
"Initializer_175/random_uniform/minConst*
dtype0*
valueB
 *   ž
O
"Initializer_175/random_uniform/maxConst*
valueB
 *   >*
dtype0

,Initializer_175/random_uniform/RandomUniformRandomUniform$Initializer_175/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_175/random_uniform/subSub"Initializer_175/random_uniform/max"Initializer_175/random_uniform/min*
T0

"Initializer_175/random_uniform/mulMul,Initializer_175/random_uniform/RandomUniform"Initializer_175/random_uniform/sub*
T0
v
Initializer_175/random_uniformAdd"Initializer_175/random_uniform/mul"Initializer_175/random_uniform/min*
T0
×

Assign_175Assign-mio_variable/hate_xtr/dense_1/kernel/gradientInitializer_175/random_uniform*@
_class6
42loc:@mio_variable/hate_xtr/dense_1/kernel/gradient*
validate_shape(*
use_locking(*
T0

+mio_variable/hate_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerhate_xtr/dense_1/bias*
shape:

+mio_variable/hate_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerhate_xtr/dense_1/bias*
shape:
G
Initializer_176/zerosConst*
valueB*    *
dtype0
Ę

Assign_176Assign+mio_variable/hate_xtr/dense_1/bias/gradientInitializer_176/zeros*
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
hate_xtr/dense_1/BiasAddBiasAddhate_xtr/dense_1/MatMul+mio_variable/hate_xtr/dense_1/bias/variable*
T0*
data_formatNHWC
M
 hate_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
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
Y
$Initializer_177/random_uniform/shapeConst*
valueB"   @   *
dtype0
O
"Initializer_177/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
O
"Initializer_177/random_uniform/maxConst*
valueB
 *ķ5>*
dtype0

,Initializer_177/random_uniform/RandomUniformRandomUniform$Initializer_177/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_177/random_uniform/subSub"Initializer_177/random_uniform/max"Initializer_177/random_uniform/min*
T0

"Initializer_177/random_uniform/mulMul,Initializer_177/random_uniform/RandomUniform"Initializer_177/random_uniform/sub*
T0
v
Initializer_177/random_uniformAdd"Initializer_177/random_uniform/mul"Initializer_177/random_uniform/min*
T0
×

Assign_177Assign-mio_variable/hate_xtr/dense_2/kernel/gradientInitializer_177/random_uniform*
T0*@
_class6
42loc:@mio_variable/hate_xtr/dense_2/kernel/gradient*
validate_shape(*
use_locking(

+mio_variable/hate_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerhate_xtr/dense_2/bias*
shape:@

+mio_variable/hate_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*$
	containerhate_xtr/dense_2/bias
F
Initializer_178/zerosConst*
valueB@*    *
dtype0
Ę

Assign_178Assign+mio_variable/hate_xtr/dense_2/bias/gradientInitializer_178/zeros*
use_locking(*
T0*>
_class4
20loc:@mio_variable/hate_xtr/dense_2/bias/gradient*
validate_shape(

hate_xtr/dense_2/MatMulMatMulhate_xtr/dense_1/LeakyRelu-mio_variable/hate_xtr/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 

hate_xtr/dense_2/BiasAddBiasAddhate_xtr/dense_2/MatMul+mio_variable/hate_xtr/dense_2/bias/variable*
data_formatNHWC*
T0
M
 hate_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
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
Y
$Initializer_179/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_179/random_uniform/minConst*
valueB
 *ž*
dtype0
O
"Initializer_179/random_uniform/maxConst*
valueB
 *>*
dtype0

,Initializer_179/random_uniform/RandomUniformRandomUniform$Initializer_179/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_179/random_uniform/subSub"Initializer_179/random_uniform/max"Initializer_179/random_uniform/min*
T0

"Initializer_179/random_uniform/mulMul,Initializer_179/random_uniform/RandomUniform"Initializer_179/random_uniform/sub*
T0
v
Initializer_179/random_uniformAdd"Initializer_179/random_uniform/mul"Initializer_179/random_uniform/min*
T0
×

Assign_179Assign-mio_variable/hate_xtr/dense_3/kernel/gradientInitializer_179/random_uniform*@
_class6
42loc:@mio_variable/hate_xtr/dense_3/kernel/gradient*
validate_shape(*
use_locking(*
T0

+mio_variable/hate_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerhate_xtr/dense_3/bias*
shape:

+mio_variable/hate_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerhate_xtr/dense_3/bias*
shape:
F
Initializer_180/zerosConst*
valueB*    *
dtype0
Ę

Assign_180Assign+mio_variable/hate_xtr/dense_3/bias/gradientInitializer_180/zeros*
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
-mio_variable/report_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreport_xtr/dense/kernel*
shape:
°
 
-mio_variable/report_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreport_xtr/dense/kernel*
shape:
°
Y
$Initializer_181/random_uniform/shapeConst*
valueB"°     *
dtype0
O
"Initializer_181/random_uniform/minConst*
valueB
 *ÃĐŊ*
dtype0
O
"Initializer_181/random_uniform/maxConst*
valueB
 *ÃĐ=*
dtype0

,Initializer_181/random_uniform/RandomUniformRandomUniform$Initializer_181/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_181/random_uniform/subSub"Initializer_181/random_uniform/max"Initializer_181/random_uniform/min*
T0

"Initializer_181/random_uniform/mulMul,Initializer_181/random_uniform/RandomUniform"Initializer_181/random_uniform/sub*
T0
v
Initializer_181/random_uniformAdd"Initializer_181/random_uniform/mul"Initializer_181/random_uniform/min*
T0
×

Assign_181Assign-mio_variable/report_xtr/dense/kernel/gradientInitializer_181/random_uniform*
use_locking(*
T0*@
_class6
42loc:@mio_variable/report_xtr/dense/kernel/gradient*
validate_shape(

+mio_variable/report_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containerreport_xtr/dense/bias

+mio_variable/report_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerreport_xtr/dense/bias*
shape:
G
Initializer_182/zerosConst*
valueB*    *
dtype0
Ę

Assign_182Assign+mio_variable/report_xtr/dense/bias/gradientInitializer_182/zeros*
use_locking(*
T0*>
_class4
20loc:@mio_variable/report_xtr/dense/bias/gradient*
validate_shape(

report_xtr/dense/MatMulMatMulconcat_1-mio_variable/report_xtr/dense/kernel/variable*
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
 *ÍĖL>*
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
Y
$Initializer_183/random_uniform/shapeConst*
valueB"      *
dtype0
O
"Initializer_183/random_uniform/minConst*
valueB
 *   ž*
dtype0
O
"Initializer_183/random_uniform/maxConst*
valueB
 *   >*
dtype0

,Initializer_183/random_uniform/RandomUniformRandomUniform$Initializer_183/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
z
"Initializer_183/random_uniform/subSub"Initializer_183/random_uniform/max"Initializer_183/random_uniform/min*
T0

"Initializer_183/random_uniform/mulMul,Initializer_183/random_uniform/RandomUniform"Initializer_183/random_uniform/sub*
T0
v
Initializer_183/random_uniformAdd"Initializer_183/random_uniform/mul"Initializer_183/random_uniform/min*
T0
Û

Assign_183Assign/mio_variable/report_xtr/dense_1/kernel/gradientInitializer_183/random_uniform*
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
G
Initializer_184/zerosConst*
valueB*    *
dtype0
Î

Assign_184Assign-mio_variable/report_xtr/dense_1/bias/gradientInitializer_184/zeros*
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
report_xtr/dense_1/BiasAddBiasAddreport_xtr/dense_1/MatMul-mio_variable/report_xtr/dense_1/bias/variable*
data_formatNHWC*
T0
O
"report_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0
p
 report_xtr/dense_1/LeakyRelu/mulMul"report_xtr/dense_1/LeakyRelu/alphareport_xtr/dense_1/BiasAdd*
T0
n
report_xtr/dense_1/LeakyReluMaximum report_xtr/dense_1/LeakyRelu/mulreport_xtr/dense_1/BiasAdd*
T0
Ŗ
/mio_variable/report_xtr/dense_2/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerreport_xtr/dense_2/kernel*
shape:	@
Ŗ
/mio_variable/report_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerreport_xtr/dense_2/kernel*
shape:	@
Y
$Initializer_185/random_uniform/shapeConst*
valueB"   @   *
dtype0
O
"Initializer_185/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
O
"Initializer_185/random_uniform/maxConst*
dtype0*
valueB
 *ķ5>

,Initializer_185/random_uniform/RandomUniformRandomUniform$Initializer_185/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
z
"Initializer_185/random_uniform/subSub"Initializer_185/random_uniform/max"Initializer_185/random_uniform/min*
T0

"Initializer_185/random_uniform/mulMul,Initializer_185/random_uniform/RandomUniform"Initializer_185/random_uniform/sub*
T0
v
Initializer_185/random_uniformAdd"Initializer_185/random_uniform/mul"Initializer_185/random_uniform/min*
T0
Û

Assign_185Assign/mio_variable/report_xtr/dense_2/kernel/gradientInitializer_185/random_uniform*
T0*B
_class8
64loc:@mio_variable/report_xtr/dense_2/kernel/gradient*
validate_shape(*
use_locking(

-mio_variable/report_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreport_xtr/dense_2/bias*
shape:@

-mio_variable/report_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreport_xtr/dense_2/bias*
shape:@
F
Initializer_186/zerosConst*
dtype0*
valueB@*    
Î

Assign_186Assign-mio_variable/report_xtr/dense_2/bias/gradientInitializer_186/zeros*
use_locking(*
T0*@
_class6
42loc:@mio_variable/report_xtr/dense_2/bias/gradient*
validate_shape(
Ą
report_xtr/dense_2/MatMulMatMulreport_xtr/dense_1/LeakyRelu/mio_variable/report_xtr/dense_2/kernel/variable*
T0*
transpose_a( *
transpose_b( 

report_xtr/dense_2/BiasAddBiasAddreport_xtr/dense_2/MatMul-mio_variable/report_xtr/dense_2/bias/variable*
T0*
data_formatNHWC
O
"report_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *ÍĖL>*
dtype0
p
 report_xtr/dense_2/LeakyRelu/mulMul"report_xtr/dense_2/LeakyRelu/alphareport_xtr/dense_2/BiasAdd*
T0
n
report_xtr/dense_2/LeakyReluMaximum report_xtr/dense_2/LeakyRelu/mulreport_xtr/dense_2/BiasAdd*
T0
ĸ
/mio_variable/report_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerreport_xtr/dense_3/kernel*
shape
:@
ĸ
/mio_variable/report_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*(
	containerreport_xtr/dense_3/kernel
Y
$Initializer_187/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_187/random_uniform/minConst*
valueB
 *ž*
dtype0
O
"Initializer_187/random_uniform/maxConst*
valueB
 *>*
dtype0

,Initializer_187/random_uniform/RandomUniformRandomUniform$Initializer_187/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
z
"Initializer_187/random_uniform/subSub"Initializer_187/random_uniform/max"Initializer_187/random_uniform/min*
T0

"Initializer_187/random_uniform/mulMul,Initializer_187/random_uniform/RandomUniform"Initializer_187/random_uniform/sub*
T0
v
Initializer_187/random_uniformAdd"Initializer_187/random_uniform/mul"Initializer_187/random_uniform/min*
T0
Û

Assign_187Assign/mio_variable/report_xtr/dense_3/kernel/gradientInitializer_187/random_uniform*
use_locking(*
T0*B
_class8
64loc:@mio_variable/report_xtr/dense_3/kernel/gradient*
validate_shape(

-mio_variable/report_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreport_xtr/dense_3/bias*
shape:

-mio_variable/report_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreport_xtr/dense_3/bias*
shape:
F
Initializer_188/zerosConst*
valueB*    *
dtype0
Î

Assign_188Assign-mio_variable/report_xtr/dense_3/bias/gradientInitializer_188/zeros*
validate_shape(*
use_locking(*
T0*@
_class6
42loc:@mio_variable/report_xtr/dense_3/bias/gradient
Ą
report_xtr/dense_3/MatMulMatMulreport_xtr/dense_2/LeakyRelu/mio_variable/report_xtr/dense_3/kernel/variable*
transpose_a( *
transpose_b( *
T0

report_xtr/dense_3/BiasAddBiasAddreport_xtr/dense_3/MatMul-mio_variable/report_xtr/dense_3/bias/variable*
T0*
data_formatNHWC
J
report_xtr/dense_3/SigmoidSigmoidreport_xtr/dense_3/BiasAdd*
T0"