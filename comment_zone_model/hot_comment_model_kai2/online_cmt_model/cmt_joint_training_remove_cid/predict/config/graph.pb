
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
2mio_compress_indices/COMPRESS_INDEX__USER/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:˙˙˙˙˙˙˙˙˙*#
	containerCOMPRESS_INDEX__USER
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
	containerpid_embedding*
shape:˙˙˙˙˙˙˙˙˙@
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
shape:˙˙˙˙˙˙˙˙˙@*
	containeruid_embedding

%mio_embeddings/uid_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:˙˙˙˙˙˙˙˙˙@*
	containeruid_embedding

%mio_embeddings/did_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
	containerdid_embedding*
shape:˙˙˙˙˙˙˙˙˙@

%mio_embeddings/did_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:˙˙˙˙˙˙˙˙˙@*
	containerdid_embedding

)mio_embeddings/context_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:˙˙˙˙˙˙˙˙˙@* 
	containercontext_embedding
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
shape:˙˙˙˙˙˙˙˙˙*
	containertoken_input_mask
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

)mio_embeddings/bert_id_embedding/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS* 
	containerbert_id_embedding*
shape:˙˙˙˙˙˙˙˙˙

)mio_embeddings/bert_id_embedding/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS* 
	containerbert_id_embedding*
shape:˙˙˙˙˙˙˙˙˙
1
Const_1Const*
value	B
 Z *
dtype0

0
cond/SwitchSwitchConst_1Const_1*
T0

1
cond/switch_tIdentitycond/Switch:1*
T0

/
cond/switch_fIdentitycond/Switch*
T0

*
cond/pred_idIdentityConst_1*
T0

M
	cond/CastCastcond/Cast/Switch:1*

DstT0*

SrcT0*
Truncate( 

cond/Cast/SwitchSwitch(mio_extra_param/token_input_ids/variablecond/pred_id*;
_class1
/-loc:@mio_extra_param/token_input_ids/variable*
T0
Q
cond/Cast_1Castcond/Cast_1/Switch:1*

SrcT0*
Truncate( *

DstT0

cond/Cast_1/SwitchSwitch)mio_extra_param/token_input_mask/variablecond/pred_id*<
_class2
0.loc:@mio_extra_param/token_input_mask/variable*
T0
Q
cond/Cast_2Castcond/Cast_2/Switch:1*
Truncate( *

DstT0*

SrcT0

cond/Cast_2/SwitchSwitch&mio_extra_param/token_sep_ids/variablecond/pred_id*
T0*9
_class/
-+loc:@mio_extra_param/token_sep_ids/variable
7

cond/ShapeShape	cond/Cast*
T0*
out_type0
V
cond/strided_slice/stackConst^cond/switch_t*
valueB: *
dtype0
X
cond/strided_slice/stack_1Const^cond/switch_t*
valueB:*
dtype0
X
cond/strided_slice/stack_2Const^cond/switch_t*
dtype0*
valueB:
ú
cond/strided_sliceStridedSlice
cond/Shapecond/strided_slice/stackcond/strided_slice/stack_1cond/strided_slice/stack_2*
end_mask *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
j
#cond/bert/embeddings/ExpandDims/dimConst^cond/switch_t*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
r
cond/bert/embeddings/ExpandDims
ExpandDims	cond/Cast#cond/bert/embeddings/ExpandDims/dim*

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
#Initializer/truncated_normal/stddevConst*
dtype0*
valueB
 *
×Ŗ<
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
i
"cond/bert/embeddings/Reshape/shapeConst^cond/switch_t*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0

cond/bert/embeddings/ReshapeReshapecond/bert/embeddings/ExpandDims"cond/bert/embeddings/Reshape/shape*
T0*
Tshape0
\
"cond/bert/embeddings/GatherV2/axisConst^cond/switch_t*
dtype0*
value	B : 
ŋ
cond/bert/embeddings/GatherV2GatherV2&cond/bert/embeddings/GatherV2/Switch:1cond/bert/embeddings/Reshape"cond/bert/embeddings/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0
Æ
$cond/bert/embeddings/GatherV2/SwitchSwitch5mio_variable/bert/embeddings/word_embeddings/variablecond/pred_id*
T0*H
_class>
<:loc:@mio_variable/bert/embeddings/word_embeddings/variable
]
cond/bert/embeddings/ShapeShapecond/bert/embeddings/ExpandDims*
T0*
out_type0
f
(cond/bert/embeddings/strided_slice/stackConst^cond/switch_t*
valueB: *
dtype0
h
*cond/bert/embeddings/strided_slice/stack_1Const^cond/switch_t*
valueB:*
dtype0
h
*cond/bert/embeddings/strided_slice/stack_2Const^cond/switch_t*
valueB:*
dtype0
Ę
"cond/bert/embeddings/strided_sliceStridedSlicecond/bert/embeddings/Shape(cond/bert/embeddings/strided_slice/stack*cond/bert/embeddings/strided_slice/stack_1*cond/bert/embeddings/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0
`
&cond/bert/embeddings/Reshape_1/shape/1Const^cond/switch_t*
value	B :*
dtype0
a
&cond/bert/embeddings/Reshape_1/shape/2Const^cond/switch_t*
value
B :*
dtype0
ž
$cond/bert/embeddings/Reshape_1/shapePack"cond/bert/embeddings/strided_slice&cond/bert/embeddings/Reshape_1/shape/1&cond/bert/embeddings/Reshape_1/shape/2*
T0*

axis *
N

cond/bert/embeddings/Reshape_1Reshapecond/bert/embeddings/GatherV2$cond/bert/embeddings/Reshape_1/shape*
T0*
Tshape0
^
cond/bert/embeddings/Shape_1Shapecond/bert/embeddings/Reshape_1*
T0*
out_type0
h
*cond/bert/embeddings/strided_slice_1/stackConst^cond/switch_t*
valueB: *
dtype0
j
,cond/bert/embeddings/strided_slice_1/stack_1Const^cond/switch_t*
valueB:*
dtype0
j
,cond/bert/embeddings/strided_slice_1/stack_2Const^cond/switch_t*
valueB:*
dtype0
Ô
$cond/bert/embeddings/strided_slice_1StridedSlicecond/bert/embeddings/Shape_1*cond/bert/embeddings/strided_slice_1/stack,cond/bert/embeddings/strided_slice_1/stack_1,cond/bert/embeddings/strided_slice_1/stack_2*
end_mask *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
ģ
;mio_variable/bert/embeddings/token_type_embeddings/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*4
	container'%bert/embeddings/token_type_embeddings*
shape:	
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
.Initializer_1/truncated_normal/TruncatedNormalTruncatedNormal$Initializer_1/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

"Initializer_1/truncated_normal/mulMul.Initializer_1/truncated_normal/TruncatedNormal%Initializer_1/truncated_normal/stddev*
T0
w
Initializer_1/truncated_normalAdd"Initializer_1/truncated_normal/mul#Initializer_1/truncated_normal/mean*
T0
ņ
Assign_1Assign;mio_variable/bert/embeddings/token_type_embeddings/gradientInitializer_1/truncated_normal*
validate_shape(*
use_locking(*
T0*N
_classD
B@loc:@mio_variable/bert/embeddings/token_type_embeddings/gradient
k
$cond/bert/embeddings/Reshape_2/shapeConst^cond/switch_t*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
s
cond/bert/embeddings/Reshape_2Reshapecond/Cast_2$cond/bert/embeddings/Reshape_2/shape*
T0*
Tshape0
b
%cond/bert/embeddings/one_hot/on_valueConst^cond/switch_t*
valueB
 *  ?*
dtype0
c
&cond/bert/embeddings/one_hot/off_valueConst^cond/switch_t*
valueB
 *    *
dtype0
\
"cond/bert/embeddings/one_hot/depthConst^cond/switch_t*
value	B :*
dtype0
á
cond/bert/embeddings/one_hotOneHotcond/bert/embeddings/Reshape_2"cond/bert/embeddings/one_hot/depth%cond/bert/embeddings/one_hot/on_value&cond/bert/embeddings/one_hot/off_value*
T0*
TI0*
axis˙˙˙˙˙˙˙˙˙

cond/bert/embeddings/MatMulMatMulcond/bert/embeddings/one_hot$cond/bert/embeddings/MatMul/Switch:1*
transpose_a( *
transpose_b( *
T0
Đ
"cond/bert/embeddings/MatMul/SwitchSwitch;mio_variable/bert/embeddings/token_type_embeddings/variablecond/pred_id*
T0*N
_classD
B@loc:@mio_variable/bert/embeddings/token_type_embeddings/variable
`
&cond/bert/embeddings/Reshape_3/shape/1Const^cond/switch_t*
value	B :*
dtype0
a
&cond/bert/embeddings/Reshape_3/shape/2Const^cond/switch_t*
value
B :*
dtype0
Ā
$cond/bert/embeddings/Reshape_3/shapePack$cond/bert/embeddings/strided_slice_1&cond/bert/embeddings/Reshape_3/shape/1&cond/bert/embeddings/Reshape_3/shape/2*
T0*

axis *
N

cond/bert/embeddings/Reshape_3Reshapecond/bert/embeddings/MatMul$cond/bert/embeddings/Reshape_3/shape*
Tshape0*
T0
h
cond/bert/embeddings/addAddcond/bert/embeddings/Reshape_1cond/bert/embeddings/Reshape_3*
T0
b
(cond/bert/embeddings/assert_less_equal/xConst^cond/switch_t*
value	B :*
dtype0
c
(cond/bert/embeddings/assert_less_equal/yConst^cond/switch_t*
value
B :*
dtype0

0cond/bert/embeddings/assert_less_equal/LessEqual	LessEqual(cond/bert/embeddings/assert_less_equal/x(cond/bert/embeddings/assert_less_equal/y*
T0
e
,cond/bert/embeddings/assert_less_equal/ConstConst^cond/switch_t*
valueB *
dtype0
Ž
*cond/bert/embeddings/assert_less_equal/AllAll0cond/bert/embeddings/assert_less_equal/LessEqual,cond/bert/embeddings/assert_less_equal/Const*

Tidx0*
	keep_dims( 
l
3cond/bert/embeddings/assert_less_equal/Assert/ConstConst^cond/switch_t*
valueB B *
dtype0
Ę
5cond/bert/embeddings/assert_less_equal/Assert/Const_1Const^cond/switch_t*m
valuedBb B\Condition x <= y did not hold element-wise:x (cond/bert/embeddings/assert_less_equal/x:0) = *
dtype0

5cond/bert/embeddings/assert_less_equal/Assert/Const_2Const^cond/switch_t*B
value9B7 B1y (cond/bert/embeddings/assert_less_equal/y:0) = *
dtype0
t
;cond/bert/embeddings/assert_less_equal/Assert/Assert/data_0Const^cond/switch_t*
valueB B *
dtype0
Đ
;cond/bert/embeddings/assert_less_equal/Assert/Assert/data_1Const^cond/switch_t*m
valuedBb B\Condition x <= y did not hold element-wise:x (cond/bert/embeddings/assert_less_equal/x:0) = *
dtype0
Ĩ
;cond/bert/embeddings/assert_less_equal/Assert/Assert/data_3Const^cond/switch_t*B
value9B7 B1y (cond/bert/embeddings/assert_less_equal/y:0) = *
dtype0

4cond/bert/embeddings/assert_less_equal/Assert/AssertAssert*cond/bert/embeddings/assert_less_equal/All;cond/bert/embeddings/assert_less_equal/Assert/Assert/data_0;cond/bert/embeddings/assert_less_equal/Assert/Assert/data_1(cond/bert/embeddings/assert_less_equal/x;cond/bert/embeddings/assert_less_equal/Assert/Assert/data_3(cond/bert/embeddings/assert_less_equal/y*
	summarize*
T	
2
¸
9mio_variable/bert/embeddings/position_embeddings/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*2
	container%#bert/embeddings/position_embeddings
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
í
Assign_2Assign9mio_variable/bert/embeddings/position_embeddings/gradientInitializer_2/truncated_normal*
validate_shape(*
use_locking(*
T0*L
_classB
@>loc:@mio_variable/bert/embeddings/position_embeddings/gradient

 cond/bert/embeddings/Slice/beginConst5^cond/bert/embeddings/assert_less_equal/Assert/Assert*
valueB"        *
dtype0

cond/bert/embeddings/Slice/sizeConst5^cond/bert/embeddings/assert_less_equal/Assert/Assert*
valueB"   ˙˙˙˙*
dtype0
Ą
cond/bert/embeddings/SliceSlice#cond/bert/embeddings/Slice/Switch:1 cond/bert/embeddings/Slice/begincond/bert/embeddings/Slice/size*
T0*
Index0
Ë
!cond/bert/embeddings/Slice/SwitchSwitch9mio_variable/bert/embeddings/position_embeddings/variablecond/pred_id*
T0*L
_classB
@>loc:@mio_variable/bert/embeddings/position_embeddings/variable

$cond/bert/embeddings/Reshape_4/shapeConst5^cond/bert/embeddings/assert_less_equal/Assert/Assert*!
valueB"         *
dtype0

cond/bert/embeddings/Reshape_4Reshapecond/bert/embeddings/Slice$cond/bert/embeddings/Reshape_4/shape*
Tshape0*
T0
d
cond/bert/embeddings/add_1Addcond/bert/embeddings/addcond/bert/embeddings/Reshape_4*
T0
Š
4mio_variable/bert/embeddings/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*-
	container bert/embeddings/LayerNorm/beta*
shape:
Š
4mio_variable/bert/embeddings/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*-
	container bert/embeddings/LayerNorm/beta*
shape:
E
Initializer_3/zerosConst*
valueB*    *
dtype0
Ø
Assign_3Assign4mio_variable/bert/embeddings/LayerNorm/beta/gradientInitializer_3/zeros*
use_locking(*
T0*G
_class=
;9loc:@mio_variable/bert/embeddings/LayerNorm/beta/gradient*
validate_shape(
Ģ
5mio_variable/bert/embeddings/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*.
	container!bert/embeddings/LayerNorm/gamma*
shape:
Ģ
5mio_variable/bert/embeddings/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*.
	container!bert/embeddings/LayerNorm/gamma
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
{
=cond/bert/embeddings/LayerNorm/moments/mean/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0
´
+cond/bert/embeddings/LayerNorm/moments/meanMeancond/bert/embeddings/add_1=cond/bert/embeddings/LayerNorm/moments/mean/reduction_indices*
T0*

Tidx0*
	keep_dims(
y
3cond/bert/embeddings/LayerNorm/moments/StopGradientStopGradient+cond/bert/embeddings/LayerNorm/moments/mean*
T0
§
8cond/bert/embeddings/LayerNorm/moments/SquaredDifferenceSquaredDifferencecond/bert/embeddings/add_13cond/bert/embeddings/LayerNorm/moments/StopGradient*
T0

Acond/bert/embeddings/LayerNorm/moments/variance/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0
Ú
/cond/bert/embeddings/LayerNorm/moments/varianceMean8cond/bert/embeddings/LayerNorm/moments/SquaredDifferenceAcond/bert/embeddings/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0
k
.cond/bert/embeddings/LayerNorm/batchnorm/add/yConst^cond/switch_t*
valueB
 *Ėŧ+*
dtype0

,cond/bert/embeddings/LayerNorm/batchnorm/addAdd/cond/bert/embeddings/LayerNorm/moments/variance.cond/bert/embeddings/LayerNorm/batchnorm/add/y*
T0
n
.cond/bert/embeddings/LayerNorm/batchnorm/RsqrtRsqrt,cond/bert/embeddings/LayerNorm/batchnorm/add*
T0
Ŗ
,cond/bert/embeddings/LayerNorm/batchnorm/mulMul.cond/bert/embeddings/LayerNorm/batchnorm/Rsqrt5cond/bert/embeddings/LayerNorm/batchnorm/mul/Switch:1*
T0
Õ
3cond/bert/embeddings/LayerNorm/batchnorm/mul/SwitchSwitch5mio_variable/bert/embeddings/LayerNorm/gamma/variablecond/pred_id*
T0*H
_class>
<:loc:@mio_variable/bert/embeddings/LayerNorm/gamma/variable

.cond/bert/embeddings/LayerNorm/batchnorm/mul_1Mulcond/bert/embeddings/add_1,cond/bert/embeddings/LayerNorm/batchnorm/mul*
T0

.cond/bert/embeddings/LayerNorm/batchnorm/mul_2Mul+cond/bert/embeddings/LayerNorm/moments/mean,cond/bert/embeddings/LayerNorm/batchnorm/mul*
T0
Ŗ
,cond/bert/embeddings/LayerNorm/batchnorm/subSub5cond/bert/embeddings/LayerNorm/batchnorm/sub/Switch:1.cond/bert/embeddings/LayerNorm/batchnorm/mul_2*
T0
Ķ
3cond/bert/embeddings/LayerNorm/batchnorm/sub/SwitchSwitch4mio_variable/bert/embeddings/LayerNorm/beta/variablecond/pred_id*
T0*G
_class=
;9loc:@mio_variable/bert/embeddings/LayerNorm/beta/variable

.cond/bert/embeddings/LayerNorm/batchnorm/add_1Add.cond/bert/embeddings/LayerNorm/batchnorm/mul_1,cond/bert/embeddings/LayerNorm/batchnorm/sub*
T0
c
&cond/bert/embeddings/dropout/keep_probConst^cond/switch_t*
valueB
 *fff?*
dtype0
t
"cond/bert/embeddings/dropout/ShapeShape.cond/bert/embeddings/LayerNorm/batchnorm/add_1*
T0*
out_type0
l
/cond/bert/embeddings/dropout/random_uniform/minConst^cond/switch_t*
dtype0*
valueB
 *    
l
/cond/bert/embeddings/dropout/random_uniform/maxConst^cond/switch_t*
dtype0*
valueB
 *  ?

9cond/bert/embeddings/dropout/random_uniform/RandomUniformRandomUniform"cond/bert/embeddings/dropout/Shape*
seed2 *

seed *
T0*
dtype0
Ą
/cond/bert/embeddings/dropout/random_uniform/subSub/cond/bert/embeddings/dropout/random_uniform/max/cond/bert/embeddings/dropout/random_uniform/min*
T0
Ģ
/cond/bert/embeddings/dropout/random_uniform/mulMul9cond/bert/embeddings/dropout/random_uniform/RandomUniform/cond/bert/embeddings/dropout/random_uniform/sub*
T0

+cond/bert/embeddings/dropout/random_uniformAdd/cond/bert/embeddings/dropout/random_uniform/mul/cond/bert/embeddings/dropout/random_uniform/min*
T0

 cond/bert/embeddings/dropout/addAdd&cond/bert/embeddings/dropout/keep_prob+cond/bert/embeddings/dropout/random_uniform*
T0
V
"cond/bert/embeddings/dropout/FloorFloor cond/bert/embeddings/dropout/add*
T0

 cond/bert/embeddings/dropout/divRealDiv.cond/bert/embeddings/LayerNorm/batchnorm/add_1&cond/bert/embeddings/dropout/keep_prob*
T0
v
 cond/bert/embeddings/dropout/mulMul cond/bert/embeddings/dropout/div"cond/bert/embeddings/dropout/Floor*
T0
D
cond/bert/encoder/ShapeShape	cond/Cast*
T0*
out_type0
c
%cond/bert/encoder/strided_slice/stackConst^cond/switch_t*
valueB: *
dtype0
e
'cond/bert/encoder/strided_slice/stack_1Const^cond/switch_t*
valueB:*
dtype0
e
'cond/bert/encoder/strided_slice/stack_2Const^cond/switch_t*
valueB:*
dtype0
ģ
cond/bert/encoder/strided_sliceStridedSlicecond/bert/encoder/Shape%cond/bert/encoder/strided_slice/stack'cond/bert/encoder/strided_slice/stack_1'cond/bert/encoder/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
H
cond/bert/encoder/Shape_1Shapecond/Cast_1*
out_type0*
T0
e
'cond/bert/encoder/strided_slice_1/stackConst^cond/switch_t*
valueB: *
dtype0
g
)cond/bert/encoder/strided_slice_1/stack_1Const^cond/switch_t*
dtype0*
valueB:
g
)cond/bert/encoder/strided_slice_1/stack_2Const^cond/switch_t*
valueB:*
dtype0
Å
!cond/bert/encoder/strided_slice_1StridedSlicecond/bert/encoder/Shape_1'cond/bert/encoder/strided_slice_1/stack)cond/bert/encoder/strided_slice_1/stack_1)cond/bert/encoder/strided_slice_1/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0
[
!cond/bert/encoder/Reshape/shape/1Const^cond/switch_t*
dtype0*
value	B :
[
!cond/bert/encoder/Reshape/shape/2Const^cond/switch_t*
value	B :*
dtype0
Ŧ
cond/bert/encoder/Reshape/shapePackcond/bert/encoder/strided_slice!cond/bert/encoder/Reshape/shape/1!cond/bert/encoder/Reshape/shape/2*
N*
T0*

axis 
i
cond/bert/encoder/ReshapeReshapecond/Cast_1cond/bert/encoder/Reshape/shape*
T0*
Tshape0
a
cond/bert/encoder/CastCastcond/bert/encoder/Reshape*

SrcT0*
Truncate( *

DstT0
V
cond/bert/encoder/ones/mul/yConst^cond/switch_t*
value	B :*
dtype0
i
cond/bert/encoder/ones/mulMulcond/bert/encoder/strided_slicecond/bert/encoder/ones/mul/y*
T0
X
cond/bert/encoder/ones/mul_1/yConst^cond/switch_t*
value	B :*
dtype0
h
cond/bert/encoder/ones/mul_1Mulcond/bert/encoder/ones/mulcond/bert/encoder/ones/mul_1/y*
T0
X
cond/bert/encoder/ones/Less/yConst^cond/switch_t*
value
B :č*
dtype0
i
cond/bert/encoder/ones/LessLesscond/bert/encoder/ones/mul_1cond/bert/encoder/ones/Less/y*
T0
Y
cond/bert/encoder/ones/packed/1Const^cond/switch_t*
value	B :*
dtype0
Y
cond/bert/encoder/ones/packed/2Const^cond/switch_t*
value	B :*
dtype0
Ļ
cond/bert/encoder/ones/packedPackcond/bert/encoder/strided_slicecond/bert/encoder/ones/packed/1cond/bert/encoder/ones/packed/2*
N*
T0*

axis 
Y
cond/bert/encoder/ones/ConstConst^cond/switch_t*
valueB
 *  ?*
dtype0
v
cond/bert/encoder/onesFillcond/bert/encoder/ones/packedcond/bert/encoder/ones/Const*
T0*

index_type0
U
cond/bert/encoder/mulMulcond/bert/encoder/onescond/bert/encoder/Cast*
T0
]
cond/bert/encoder/Shape_2Shape cond/bert/embeddings/dropout/mul*
T0*
out_type0
e
'cond/bert/encoder/strided_slice_2/stackConst^cond/switch_t*
dtype0*
valueB: 
g
)cond/bert/encoder/strided_slice_2/stack_1Const^cond/switch_t*
valueB:*
dtype0
g
)cond/bert/encoder/strided_slice_2/stack_2Const^cond/switch_t*
valueB:*
dtype0
Å
!cond/bert/encoder/strided_slice_2StridedSlicecond/bert/encoder/Shape_2'cond/bert/encoder/strided_slice_2/stack)cond/bert/encoder/strided_slice_2/stack_1)cond/bert/encoder/strided_slice_2/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask
f
!cond/bert/encoder/Reshape_1/shapeConst^cond/switch_t*
valueB"˙˙˙˙   *
dtype0

cond/bert/encoder/Reshape_1Reshape cond/bert/embeddings/dropout/mul!cond/bert/encoder/Reshape_1/shape*
Tshape0*
T0
m
.cond/bert/encoder/layer_0/attention/self/ShapeShapecond/bert/encoder/Reshape_1*
T0*
out_type0
z
<cond/bert/encoder/layer_0/attention/self/strided_slice/stackConst^cond/switch_t*
valueB: *
dtype0
|
>cond/bert/encoder/layer_0/attention/self/strided_slice/stack_1Const^cond/switch_t*
valueB:*
dtype0
|
>cond/bert/encoder/layer_0/attention/self/strided_slice/stack_2Const^cond/switch_t*
valueB:*
dtype0
Ž
6cond/bert/encoder/layer_0/attention/self/strided_sliceStridedSlice.cond/bert/encoder/layer_0/attention/self/Shape<cond/bert/encoder/layer_0/attention/self/strided_slice/stack>cond/bert/encoder/layer_0/attention/self/strided_slice/stack_1>cond/bert/encoder/layer_0/attention/self/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
o
0cond/bert/encoder/layer_0/attention/self/Shape_1Shapecond/bert/encoder/Reshape_1*
T0*
out_type0
|
>cond/bert/encoder/layer_0/attention/self/strided_slice_1/stackConst^cond/switch_t*
valueB: *
dtype0
~
@cond/bert/encoder/layer_0/attention/self/strided_slice_1/stack_1Const^cond/switch_t*
valueB:*
dtype0
~
@cond/bert/encoder/layer_0/attention/self/strided_slice_1/stack_2Const^cond/switch_t*
valueB:*
dtype0
¸
8cond/bert/encoder/layer_0/attention/self/strided_slice_1StridedSlice0cond/bert/encoder/layer_0/attention/self/Shape_1>cond/bert/encoder/layer_0/attention/self/strided_slice_1/stack@cond/bert/encoder/layer_0/attention/self/strided_slice_1/stack_1@cond/bert/encoder/layer_0/attention/self/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
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
Assign_5AssignFmio_variable/bert/encoder/layer_0/attention/self/query/kernel/gradientInitializer_5/truncated_normal*
validate_shape(*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_0/attention/self/query/kernel/gradient
É
Dmio_variable/bert/encoder/layer_0/attention/self/query/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_0/attention/self/query/bias*
shape:
É
Dmio_variable/bert/encoder/layer_0/attention/self/query/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_0/attention/self/query/bias*
shape:
E
Initializer_6/zerosConst*
dtype0*
valueB*    
ø
Assign_6AssignDmio_variable/bert/encoder/layer_0/attention/self/query/bias/gradientInitializer_6/zeros*
validate_shape(*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_0/attention/self/query/bias/gradient
Ë
5cond/bert/encoder/layer_0/attention/self/query/MatMulMatMulcond/bert/encoder/Reshape_1>cond/bert/encoder/layer_0/attention/self/query/MatMul/Switch:1*
transpose_a( *
transpose_b( *
T0

<cond/bert/encoder/layer_0/attention/self/query/MatMul/SwitchSwitchFmio_variable/bert/encoder/layer_0/attention/self/query/kernel/variablecond/pred_id*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_0/attention/self/query/kernel/variable
Ų
6cond/bert/encoder/layer_0/attention/self/query/BiasAddBiasAdd5cond/bert/encoder/layer_0/attention/self/query/MatMul?cond/bert/encoder/layer_0/attention/self/query/BiasAdd/Switch:1*
T0*
data_formatNHWC
ũ
=cond/bert/encoder/layer_0/attention/self/query/BiasAdd/SwitchSwitchDmio_variable/bert/encoder/layer_0/attention/self/query/bias/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_0/attention/self/query/bias/variable
Î
Dmio_variable/bert/encoder/layer_0/attention/self/key/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_0/attention/self/key/kernel*
shape:

Î
Dmio_variable/bert/encoder/layer_0/attention/self/key/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_0/attention/self/key/kernel*
shape:

Y
$Initializer_7/truncated_normal/shapeConst*
dtype0*
valueB"      
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
seed2 *

seed *
T0*
dtype0
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
Assign_8AssignBmio_variable/bert/encoder/layer_0/attention/self/key/bias/gradientInitializer_8/zeros*
use_locking(*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_0/attention/self/key/bias/gradient*
validate_shape(
Į
3cond/bert/encoder/layer_0/attention/self/key/MatMulMatMulcond/bert/encoder/Reshape_1<cond/bert/encoder/layer_0/attention/self/key/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 
ú
:cond/bert/encoder/layer_0/attention/self/key/MatMul/SwitchSwitchDmio_variable/bert/encoder/layer_0/attention/self/key/kernel/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_0/attention/self/key/kernel/variable
Ķ
4cond/bert/encoder/layer_0/attention/self/key/BiasAddBiasAdd3cond/bert/encoder/layer_0/attention/self/key/MatMul=cond/bert/encoder/layer_0/attention/self/key/BiasAdd/Switch:1*
data_formatNHWC*
T0
÷
;cond/bert/encoder/layer_0/attention/self/key/BiasAdd/SwitchSwitchBmio_variable/bert/encoder/layer_0/attention/self/key/bias/variablecond/pred_id*U
_classK
IGloc:@mio_variable/bert/encoder/layer_0/attention/self/key/bias/variable*
T0
Ō
Fmio_variable/bert/encoder/layer_0/attention/self/value/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*?
	container20bert/encoder/layer_0/attention/self/value/kernel
Ō
Fmio_variable/bert/encoder/layer_0/attention/self/value/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_0/attention/self/value/kernel*
shape:

Y
$Initializer_9/truncated_normal/shapeConst*
valueB"      *
dtype0
P
#Initializer_9/truncated_normal/meanConst*
dtype0*
valueB
 *    
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
Assign_9AssignFmio_variable/bert/encoder/layer_0/attention/self/value/kernel/gradientInitializer_9/truncated_normal*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_0/attention/self/value/kernel/gradient*
validate_shape(
É
Dmio_variable/bert/encoder/layer_0/attention/self/value/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_0/attention/self/value/bias*
shape:
É
Dmio_variable/bert/encoder/layer_0/attention/self/value/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*=
	container0.bert/encoder/layer_0/attention/self/value/bias
F
Initializer_10/zerosConst*
valueB*    *
dtype0
ú
	Assign_10AssignDmio_variable/bert/encoder/layer_0/attention/self/value/bias/gradientInitializer_10/zeros*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_0/attention/self/value/bias/gradient*
validate_shape(
Ë
5cond/bert/encoder/layer_0/attention/self/value/MatMulMatMulcond/bert/encoder/Reshape_1>cond/bert/encoder/layer_0/attention/self/value/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 

<cond/bert/encoder/layer_0/attention/self/value/MatMul/SwitchSwitchFmio_variable/bert/encoder/layer_0/attention/self/value/kernel/variablecond/pred_id*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_0/attention/self/value/kernel/variable
Ų
6cond/bert/encoder/layer_0/attention/self/value/BiasAddBiasAdd5cond/bert/encoder/layer_0/attention/self/value/MatMul?cond/bert/encoder/layer_0/attention/self/value/BiasAdd/Switch:1*
T0*
data_formatNHWC
ũ
=cond/bert/encoder/layer_0/attention/self/value/BiasAdd/SwitchSwitchDmio_variable/bert/encoder/layer_0/attention/self/value/bias/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_0/attention/self/value/bias/variable
r
8cond/bert/encoder/layer_0/attention/self/Reshape/shape/1Const^cond/switch_t*
value	B :*
dtype0
r
8cond/bert/encoder/layer_0/attention/self/Reshape/shape/2Const^cond/switch_t*
dtype0*
value	B :
r
8cond/bert/encoder/layer_0/attention/self/Reshape/shape/3Const^cond/switch_t*
value	B : *
dtype0
­
6cond/bert/encoder/layer_0/attention/self/Reshape/shapePack!cond/bert/encoder/strided_slice_28cond/bert/encoder/layer_0/attention/self/Reshape/shape/18cond/bert/encoder/layer_0/attention/self/Reshape/shape/28cond/bert/encoder/layer_0/attention/self/Reshape/shape/3*
T0*

axis *
N
Â
0cond/bert/encoder/layer_0/attention/self/ReshapeReshape6cond/bert/encoder/layer_0/attention/self/query/BiasAdd6cond/bert/encoder/layer_0/attention/self/Reshape/shape*
T0*
Tshape0

7cond/bert/encoder/layer_0/attention/self/transpose/permConst^cond/switch_t*%
valueB"             *
dtype0
Ā
2cond/bert/encoder/layer_0/attention/self/transpose	Transpose0cond/bert/encoder/layer_0/attention/self/Reshape7cond/bert/encoder/layer_0/attention/self/transpose/perm*
Tperm0*
T0
t
:cond/bert/encoder/layer_0/attention/self/Reshape_1/shape/1Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_0/attention/self/Reshape_1/shape/2Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_0/attention/self/Reshape_1/shape/3Const^cond/switch_t*
value	B : *
dtype0
ĩ
8cond/bert/encoder/layer_0/attention/self/Reshape_1/shapePack!cond/bert/encoder/strided_slice_2:cond/bert/encoder/layer_0/attention/self/Reshape_1/shape/1:cond/bert/encoder/layer_0/attention/self/Reshape_1/shape/2:cond/bert/encoder/layer_0/attention/self/Reshape_1/shape/3*
T0*

axis *
N
Ä
2cond/bert/encoder/layer_0/attention/self/Reshape_1Reshape4cond/bert/encoder/layer_0/attention/self/key/BiasAdd8cond/bert/encoder/layer_0/attention/self/Reshape_1/shape*
T0*
Tshape0

9cond/bert/encoder/layer_0/attention/self/transpose_1/permConst^cond/switch_t*%
valueB"             *
dtype0
Æ
4cond/bert/encoder/layer_0/attention/self/transpose_1	Transpose2cond/bert/encoder/layer_0/attention/self/Reshape_19cond/bert/encoder/layer_0/attention/self/transpose_1/perm*
T0*
Tperm0
Ë
/cond/bert/encoder/layer_0/attention/self/MatMulBatchMatMul2cond/bert/encoder/layer_0/attention/self/transpose4cond/bert/encoder/layer_0/attention/self/transpose_1*
adj_x( *
adj_y(*
T0
k
.cond/bert/encoder/layer_0/attention/self/Mul/yConst^cond/switch_t*
valueB
 *ķ5>*
dtype0

,cond/bert/encoder/layer_0/attention/self/MulMul/cond/bert/encoder/layer_0/attention/self/MatMul.cond/bert/encoder/layer_0/attention/self/Mul/y*
T0
u
7cond/bert/encoder/layer_0/attention/self/ExpandDims/dimConst^cond/switch_t*
valueB:*
dtype0
Ļ
3cond/bert/encoder/layer_0/attention/self/ExpandDims
ExpandDimscond/bert/encoder/mul7cond/bert/encoder/layer_0/attention/self/ExpandDims/dim*

Tdim0*
T0
k
.cond/bert/encoder/layer_0/attention/self/sub/xConst^cond/switch_t*
valueB
 *  ?*
dtype0
Ą
,cond/bert/encoder/layer_0/attention/self/subSub.cond/bert/encoder/layer_0/attention/self/sub/x3cond/bert/encoder/layer_0/attention/self/ExpandDims*
T0
m
0cond/bert/encoder/layer_0/attention/self/mul_1/yConst^cond/switch_t*
dtype0*
valueB
 * @Æ

.cond/bert/encoder/layer_0/attention/self/mul_1Mul,cond/bert/encoder/layer_0/attention/self/sub0cond/bert/encoder/layer_0/attention/self/mul_1/y*
T0

,cond/bert/encoder/layer_0/attention/self/addAdd,cond/bert/encoder/layer_0/attention/self/Mul.cond/bert/encoder/layer_0/attention/self/mul_1*
T0
r
0cond/bert/encoder/layer_0/attention/self/SoftmaxSoftmax,cond/bert/encoder/layer_0/attention/self/add*
T0
w
:cond/bert/encoder/layer_0/attention/self/dropout/keep_probConst^cond/switch_t*
valueB
 *fff?*
dtype0

6cond/bert/encoder/layer_0/attention/self/dropout/ShapeShape0cond/bert/encoder/layer_0/attention/self/Softmax*
T0*
out_type0

Ccond/bert/encoder/layer_0/attention/self/dropout/random_uniform/minConst^cond/switch_t*
valueB
 *    *
dtype0

Ccond/bert/encoder/layer_0/attention/self/dropout/random_uniform/maxConst^cond/switch_t*
valueB
 *  ?*
dtype0
Å
Mcond/bert/encoder/layer_0/attention/self/dropout/random_uniform/RandomUniformRandomUniform6cond/bert/encoder/layer_0/attention/self/dropout/Shape*
seed2 *

seed *
T0*
dtype0
Ũ
Ccond/bert/encoder/layer_0/attention/self/dropout/random_uniform/subSubCcond/bert/encoder/layer_0/attention/self/dropout/random_uniform/maxCcond/bert/encoder/layer_0/attention/self/dropout/random_uniform/min*
T0
į
Ccond/bert/encoder/layer_0/attention/self/dropout/random_uniform/mulMulMcond/bert/encoder/layer_0/attention/self/dropout/random_uniform/RandomUniformCcond/bert/encoder/layer_0/attention/self/dropout/random_uniform/sub*
T0
Ų
?cond/bert/encoder/layer_0/attention/self/dropout/random_uniformAddCcond/bert/encoder/layer_0/attention/self/dropout/random_uniform/mulCcond/bert/encoder/layer_0/attention/self/dropout/random_uniform/min*
T0
Á
4cond/bert/encoder/layer_0/attention/self/dropout/addAdd:cond/bert/encoder/layer_0/attention/self/dropout/keep_prob?cond/bert/encoder/layer_0/attention/self/dropout/random_uniform*
T0
~
6cond/bert/encoder/layer_0/attention/self/dropout/FloorFloor4cond/bert/encoder/layer_0/attention/self/dropout/add*
T0
ļ
4cond/bert/encoder/layer_0/attention/self/dropout/divRealDiv0cond/bert/encoder/layer_0/attention/self/Softmax:cond/bert/encoder/layer_0/attention/self/dropout/keep_prob*
T0
˛
4cond/bert/encoder/layer_0/attention/self/dropout/mulMul4cond/bert/encoder/layer_0/attention/self/dropout/div6cond/bert/encoder/layer_0/attention/self/dropout/Floor*
T0
t
:cond/bert/encoder/layer_0/attention/self/Reshape_2/shape/1Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_0/attention/self/Reshape_2/shape/2Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_0/attention/self/Reshape_2/shape/3Const^cond/switch_t*
value	B : *
dtype0
ĩ
8cond/bert/encoder/layer_0/attention/self/Reshape_2/shapePack!cond/bert/encoder/strided_slice_2:cond/bert/encoder/layer_0/attention/self/Reshape_2/shape/1:cond/bert/encoder/layer_0/attention/self/Reshape_2/shape/2:cond/bert/encoder/layer_0/attention/self/Reshape_2/shape/3*
N*
T0*

axis 
Æ
2cond/bert/encoder/layer_0/attention/self/Reshape_2Reshape6cond/bert/encoder/layer_0/attention/self/value/BiasAdd8cond/bert/encoder/layer_0/attention/self/Reshape_2/shape*
Tshape0*
T0

9cond/bert/encoder/layer_0/attention/self/transpose_2/permConst^cond/switch_t*%
valueB"             *
dtype0
Æ
4cond/bert/encoder/layer_0/attention/self/transpose_2	Transpose2cond/bert/encoder/layer_0/attention/self/Reshape_29cond/bert/encoder/layer_0/attention/self/transpose_2/perm*
Tperm0*
T0
Ī
1cond/bert/encoder/layer_0/attention/self/MatMul_1BatchMatMul4cond/bert/encoder/layer_0/attention/self/dropout/mul4cond/bert/encoder/layer_0/attention/self/transpose_2*
adj_x( *
adj_y( *
T0

9cond/bert/encoder/layer_0/attention/self/transpose_3/permConst^cond/switch_t*%
valueB"             *
dtype0
Å
4cond/bert/encoder/layer_0/attention/self/transpose_3	Transpose1cond/bert/encoder/layer_0/attention/self/MatMul_19cond/bert/encoder/layer_0/attention/self/transpose_3/perm*
Tperm0*
T0
j
0cond/bert/encoder/layer_0/attention/self/mul_2/yConst^cond/switch_t*
value	B :*
dtype0

.cond/bert/encoder/layer_0/attention/self/mul_2Mul!cond/bert/encoder/strided_slice_20cond/bert/encoder/layer_0/attention/self/mul_2/y*
T0
u
:cond/bert/encoder/layer_0/attention/self/Reshape_3/shape/1Const^cond/switch_t*
value
B :*
dtype0
Ę
8cond/bert/encoder/layer_0/attention/self/Reshape_3/shapePack.cond/bert/encoder/layer_0/attention/self/mul_2:cond/bert/encoder/layer_0/attention/self/Reshape_3/shape/1*
T0*

axis *
N
Ä
2cond/bert/encoder/layer_0/attention/self/Reshape_3Reshape4cond/bert/encoder/layer_0/attention/self/transpose_38cond/bert/encoder/layer_0/attention/self/Reshape_3/shape*
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
$Initializer_11/truncated_normal/meanConst*
valueB
 *    *
dtype0
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
Fmio_variable/bert/encoder/layer_0/attention/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_0/attention/output/dense/bias*
shape:
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
æ
7cond/bert/encoder/layer_0/attention/output/dense/MatMulMatMul2cond/bert/encoder/layer_0/attention/self/Reshape_3@cond/bert/encoder/layer_0/attention/output/dense/MatMul/Switch:1*
transpose_a( *
transpose_b( *
T0

>cond/bert/encoder/layer_0/attention/output/dense/MatMul/SwitchSwitchHmio_variable/bert/encoder/layer_0/attention/output/dense/kernel/variablecond/pred_id*
T0*[
_classQ
OMloc:@mio_variable/bert/encoder/layer_0/attention/output/dense/kernel/variable
ß
8cond/bert/encoder/layer_0/attention/output/dense/BiasAddBiasAdd7cond/bert/encoder/layer_0/attention/output/dense/MatMulAcond/bert/encoder/layer_0/attention/output/dense/BiasAdd/Switch:1*
T0*
data_formatNHWC

?cond/bert/encoder/layer_0/attention/output/dense/BiasAdd/SwitchSwitchFmio_variable/bert/encoder/layer_0/attention/output/dense/bias/variablecond/pred_id*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_0/attention/output/dense/bias/variable*
T0
y
<cond/bert/encoder/layer_0/attention/output/dropout/keep_probConst^cond/switch_t*
dtype0*
valueB
 *fff?

8cond/bert/encoder/layer_0/attention/output/dropout/ShapeShape8cond/bert/encoder/layer_0/attention/output/dense/BiasAdd*
T0*
out_type0

Econd/bert/encoder/layer_0/attention/output/dropout/random_uniform/minConst^cond/switch_t*
valueB
 *    *
dtype0

Econd/bert/encoder/layer_0/attention/output/dropout/random_uniform/maxConst^cond/switch_t*
dtype0*
valueB
 *  ?
É
Ocond/bert/encoder/layer_0/attention/output/dropout/random_uniform/RandomUniformRandomUniform8cond/bert/encoder/layer_0/attention/output/dropout/Shape*
T0*
dtype0*
seed2 *

seed 
ã
Econd/bert/encoder/layer_0/attention/output/dropout/random_uniform/subSubEcond/bert/encoder/layer_0/attention/output/dropout/random_uniform/maxEcond/bert/encoder/layer_0/attention/output/dropout/random_uniform/min*
T0
í
Econd/bert/encoder/layer_0/attention/output/dropout/random_uniform/mulMulOcond/bert/encoder/layer_0/attention/output/dropout/random_uniform/RandomUniformEcond/bert/encoder/layer_0/attention/output/dropout/random_uniform/sub*
T0
ß
Acond/bert/encoder/layer_0/attention/output/dropout/random_uniformAddEcond/bert/encoder/layer_0/attention/output/dropout/random_uniform/mulEcond/bert/encoder/layer_0/attention/output/dropout/random_uniform/min*
T0
Į
6cond/bert/encoder/layer_0/attention/output/dropout/addAdd<cond/bert/encoder/layer_0/attention/output/dropout/keep_probAcond/bert/encoder/layer_0/attention/output/dropout/random_uniform*
T0

8cond/bert/encoder/layer_0/attention/output/dropout/FloorFloor6cond/bert/encoder/layer_0/attention/output/dropout/add*
T0
Â
6cond/bert/encoder/layer_0/attention/output/dropout/divRealDiv8cond/bert/encoder/layer_0/attention/output/dense/BiasAdd<cond/bert/encoder/layer_0/attention/output/dropout/keep_prob*
T0
¸
6cond/bert/encoder/layer_0/attention/output/dropout/mulMul6cond/bert/encoder/layer_0/attention/output/dropout/div8cond/bert/encoder/layer_0/attention/output/dropout/Floor*
T0

.cond/bert/encoder/layer_0/attention/output/addAdd6cond/bert/encoder/layer_0/attention/output/dropout/mulcond/bert/encoder/Reshape_1*
T0
Õ
Jmio_variable/bert/encoder/layer_0/attention/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64bert/encoder/layer_0/attention/output/LayerNorm/beta*
shape:
Õ
Jmio_variable/bert/encoder/layer_0/attention/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64bert/encoder/layer_0/attention/output/LayerNorm/beta*
shape:
F
Initializer_13/zerosConst*
valueB*    *
dtype0

	Assign_13AssignJmio_variable/bert/encoder/layer_0/attention/output/LayerNorm/beta/gradientInitializer_13/zeros*]
_classS
QOloc:@mio_variable/bert/encoder/layer_0/attention/output/LayerNorm/beta/gradient*
validate_shape(*
use_locking(*
T0
×
Kmio_variable/bert/encoder/layer_0/attention/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*D
	container75bert/encoder/layer_0/attention/output/LayerNorm/gamma
×
Kmio_variable/bert/encoder/layer_0/attention/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*D
	container75bert/encoder/layer_0/attention/output/LayerNorm/gamma*
shape:
E
Initializer_14/onesConst*
dtype0*
valueB*  ?

	Assign_14AssignKmio_variable/bert/encoder/layer_0/attention/output/LayerNorm/gamma/gradientInitializer_14/ones*
T0*^
_classT
RPloc:@mio_variable/bert/encoder/layer_0/attention/output/LayerNorm/gamma/gradient*
validate_shape(*
use_locking(

Scond/bert/encoder/layer_0/attention/output/LayerNorm/moments/mean/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0
ô
Acond/bert/encoder/layer_0/attention/output/LayerNorm/moments/meanMean.cond/bert/encoder/layer_0/attention/output/addScond/bert/encoder/layer_0/attention/output/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
Ĩ
Icond/bert/encoder/layer_0/attention/output/LayerNorm/moments/StopGradientStopGradientAcond/bert/encoder/layer_0/attention/output/LayerNorm/moments/mean*
T0
į
Ncond/bert/encoder/layer_0/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference.cond/bert/encoder/layer_0/attention/output/addIcond/bert/encoder/layer_0/attention/output/LayerNorm/moments/StopGradient*
T0

Wcond/bert/encoder/layer_0/attention/output/LayerNorm/moments/variance/reduction_indicesConst^cond/switch_t*
dtype0*
valueB:

Econd/bert/encoder/layer_0/attention/output/LayerNorm/moments/varianceMeanNcond/bert/encoder/layer_0/attention/output/LayerNorm/moments/SquaredDifferenceWcond/bert/encoder/layer_0/attention/output/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0

Dcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add/yConst^cond/switch_t*
valueB
 *Ėŧ+*
dtype0
ß
Bcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/addAddEcond/bert/encoder/layer_0/attention/output/LayerNorm/moments/varianceDcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add/y*
T0

Dcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/RsqrtRsqrtBcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add*
T0
å
Bcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mulMulDcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/RsqrtKcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul/Switch:1*
T0

Icond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul/SwitchSwitchKmio_variable/bert/encoder/layer_0/attention/output/LayerNorm/gamma/variablecond/pred_id*
T0*^
_classT
RPloc:@mio_variable/bert/encoder/layer_0/attention/output/LayerNorm/gamma/variable
Č
Dcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_1Mul.cond/bert/encoder/layer_0/attention/output/addBcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul*
T0
Û
Dcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_2MulAcond/bert/encoder/layer_0/attention/output/LayerNorm/moments/meanBcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul*
T0
å
Bcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/subSubKcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/sub/Switch:1Dcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_2*
T0

Icond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/sub/SwitchSwitchJmio_variable/bert/encoder/layer_0/attention/output/LayerNorm/beta/variablecond/pred_id*
T0*]
_classS
QOloc:@mio_variable/bert/encoder/layer_0/attention/output/LayerNorm/beta/variable
Ū
Dcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1AddDcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_1Bcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/sub*
T0
Î
Dmio_variable/bert/encoder/layer_0/intermediate/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_0/intermediate/dense/kernel*
shape:

Î
Dmio_variable/bert/encoder/layer_0/intermediate/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*=
	container0.bert/encoder/layer_0/intermediate/dense/kernel
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
	Assign_16AssignBmio_variable/bert/encoder/layer_0/intermediate/dense/bias/gradientInitializer_16/zeros*U
_classK
IGloc:@mio_variable/bert/encoder/layer_0/intermediate/dense/bias/gradient*
validate_shape(*
use_locking(*
T0
đ
3cond/bert/encoder/layer_0/intermediate/dense/MatMulMatMulDcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1<cond/bert/encoder/layer_0/intermediate/dense/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 
ú
:cond/bert/encoder/layer_0/intermediate/dense/MatMul/SwitchSwitchDmio_variable/bert/encoder/layer_0/intermediate/dense/kernel/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_0/intermediate/dense/kernel/variable
Ķ
4cond/bert/encoder/layer_0/intermediate/dense/BiasAddBiasAdd3cond/bert/encoder/layer_0/intermediate/dense/MatMul=cond/bert/encoder/layer_0/intermediate/dense/BiasAdd/Switch:1*
T0*
data_formatNHWC
÷
;cond/bert/encoder/layer_0/intermediate/dense/BiasAdd/SwitchSwitchBmio_variable/bert/encoder/layer_0/intermediate/dense/bias/variablecond/pred_id*U
_classK
IGloc:@mio_variable/bert/encoder/layer_0/intermediate/dense/bias/variable*
T0
o
2cond/bert/encoder/layer_0/intermediate/dense/Pow/yConst^cond/switch_t*
valueB
 *  @@*
dtype0
Ē
0cond/bert/encoder/layer_0/intermediate/dense/PowPow4cond/bert/encoder/layer_0/intermediate/dense/BiasAdd2cond/bert/encoder/layer_0/intermediate/dense/Pow/y*
T0
o
2cond/bert/encoder/layer_0/intermediate/dense/mul/xConst^cond/switch_t*
valueB
 *'7=*
dtype0
Ļ
0cond/bert/encoder/layer_0/intermediate/dense/mulMul2cond/bert/encoder/layer_0/intermediate/dense/mul/x0cond/bert/encoder/layer_0/intermediate/dense/Pow*
T0
¨
0cond/bert/encoder/layer_0/intermediate/dense/addAdd4cond/bert/encoder/layer_0/intermediate/dense/BiasAdd0cond/bert/encoder/layer_0/intermediate/dense/mul*
T0
q
4cond/bert/encoder/layer_0/intermediate/dense/mul_1/xConst^cond/switch_t*
valueB
 **BL?*
dtype0
Ē
2cond/bert/encoder/layer_0/intermediate/dense/mul_1Mul4cond/bert/encoder/layer_0/intermediate/dense/mul_1/x0cond/bert/encoder/layer_0/intermediate/dense/add*
T0
v
1cond/bert/encoder/layer_0/intermediate/dense/TanhTanh2cond/bert/encoder/layer_0/intermediate/dense/mul_1*
T0
q
4cond/bert/encoder/layer_0/intermediate/dense/add_1/xConst^cond/switch_t*
valueB
 *  ?*
dtype0
Ģ
2cond/bert/encoder/layer_0/intermediate/dense/add_1Add4cond/bert/encoder/layer_0/intermediate/dense/add_1/x1cond/bert/encoder/layer_0/intermediate/dense/Tanh*
T0
q
4cond/bert/encoder/layer_0/intermediate/dense/mul_2/xConst^cond/switch_t*
valueB
 *   ?*
dtype0
Ŧ
2cond/bert/encoder/layer_0/intermediate/dense/mul_2Mul4cond/bert/encoder/layer_0/intermediate/dense/mul_2/x2cond/bert/encoder/layer_0/intermediate/dense/add_1*
T0
Ŧ
2cond/bert/encoder/layer_0/intermediate/dense/mul_3Mul4cond/bert/encoder/layer_0/intermediate/dense/BiasAdd2cond/bert/encoder/layer_0/intermediate/dense/mul_2*
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
$Initializer_17/truncated_normal/meanConst*
dtype0*
valueB
 *    
S
&Initializer_17/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_17/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_17/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 
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
<mio_variable/bert/encoder/layer_0/output/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*5
	container(&bert/encoder/layer_0/output/dense/bias
š
<mio_variable/bert/encoder/layer_0/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&bert/encoder/layer_0/output/dense/bias*
shape:
F
Initializer_18/zerosConst*
dtype0*
valueB*    
ę
	Assign_18Assign<mio_variable/bert/encoder/layer_0/output/dense/bias/gradientInitializer_18/zeros*
use_locking(*
T0*O
_classE
CAloc:@mio_variable/bert/encoder/layer_0/output/dense/bias/gradient*
validate_shape(
Ō
-cond/bert/encoder/layer_0/output/dense/MatMulMatMul2cond/bert/encoder/layer_0/intermediate/dense/mul_36cond/bert/encoder/layer_0/output/dense/MatMul/Switch:1*
transpose_b( *
T0*
transpose_a( 
č
4cond/bert/encoder/layer_0/output/dense/MatMul/SwitchSwitch>mio_variable/bert/encoder/layer_0/output/dense/kernel/variablecond/pred_id*
T0*Q
_classG
ECloc:@mio_variable/bert/encoder/layer_0/output/dense/kernel/variable
Á
.cond/bert/encoder/layer_0/output/dense/BiasAddBiasAdd-cond/bert/encoder/layer_0/output/dense/MatMul7cond/bert/encoder/layer_0/output/dense/BiasAdd/Switch:1*
T0*
data_formatNHWC
å
5cond/bert/encoder/layer_0/output/dense/BiasAdd/SwitchSwitch<mio_variable/bert/encoder/layer_0/output/dense/bias/variablecond/pred_id*
T0*O
_classE
CAloc:@mio_variable/bert/encoder/layer_0/output/dense/bias/variable
o
2cond/bert/encoder/layer_0/output/dropout/keep_probConst^cond/switch_t*
valueB
 *fff?*
dtype0

.cond/bert/encoder/layer_0/output/dropout/ShapeShape.cond/bert/encoder/layer_0/output/dense/BiasAdd*
out_type0*
T0
x
;cond/bert/encoder/layer_0/output/dropout/random_uniform/minConst^cond/switch_t*
valueB
 *    *
dtype0
x
;cond/bert/encoder/layer_0/output/dropout/random_uniform/maxConst^cond/switch_t*
valueB
 *  ?*
dtype0
ĩ
Econd/bert/encoder/layer_0/output/dropout/random_uniform/RandomUniformRandomUniform.cond/bert/encoder/layer_0/output/dropout/Shape*
seed2 *

seed *
T0*
dtype0
Å
;cond/bert/encoder/layer_0/output/dropout/random_uniform/subSub;cond/bert/encoder/layer_0/output/dropout/random_uniform/max;cond/bert/encoder/layer_0/output/dropout/random_uniform/min*
T0
Ī
;cond/bert/encoder/layer_0/output/dropout/random_uniform/mulMulEcond/bert/encoder/layer_0/output/dropout/random_uniform/RandomUniform;cond/bert/encoder/layer_0/output/dropout/random_uniform/sub*
T0
Á
7cond/bert/encoder/layer_0/output/dropout/random_uniformAdd;cond/bert/encoder/layer_0/output/dropout/random_uniform/mul;cond/bert/encoder/layer_0/output/dropout/random_uniform/min*
T0
Š
,cond/bert/encoder/layer_0/output/dropout/addAdd2cond/bert/encoder/layer_0/output/dropout/keep_prob7cond/bert/encoder/layer_0/output/dropout/random_uniform*
T0
n
.cond/bert/encoder/layer_0/output/dropout/FloorFloor,cond/bert/encoder/layer_0/output/dropout/add*
T0
¤
,cond/bert/encoder/layer_0/output/dropout/divRealDiv.cond/bert/encoder/layer_0/output/dense/BiasAdd2cond/bert/encoder/layer_0/output/dropout/keep_prob*
T0

,cond/bert/encoder/layer_0/output/dropout/mulMul,cond/bert/encoder/layer_0/output/dropout/div.cond/bert/encoder/layer_0/output/dropout/Floor*
T0
¨
$cond/bert/encoder/layer_0/output/addAdd,cond/bert/encoder/layer_0/output/dropout/mulDcond/bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1*
T0
Á
@mio_variable/bert/encoder/layer_0/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*bert/encoder/layer_0/output/LayerNorm/beta*
shape:
Á
@mio_variable/bert/encoder/layer_0/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*bert/encoder/layer_0/output/LayerNorm/beta*
shape:
F
Initializer_19/zerosConst*
valueB*    *
dtype0
ō
	Assign_19Assign@mio_variable/bert/encoder/layer_0/output/LayerNorm/beta/gradientInitializer_19/zeros*
use_locking(*
T0*S
_classI
GEloc:@mio_variable/bert/encoder/layer_0/output/LayerNorm/beta/gradient*
validate_shape(
Ã
Amio_variable/bert/encoder/layer_0/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+bert/encoder/layer_0/output/LayerNorm/gamma*
shape:
Ã
Amio_variable/bert/encoder/layer_0/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*:
	container-+bert/encoder/layer_0/output/LayerNorm/gamma
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

Icond/bert/encoder/layer_0/output/LayerNorm/moments/mean/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0
Ö
7cond/bert/encoder/layer_0/output/LayerNorm/moments/meanMean$cond/bert/encoder/layer_0/output/addIcond/bert/encoder/layer_0/output/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0

?cond/bert/encoder/layer_0/output/LayerNorm/moments/StopGradientStopGradient7cond/bert/encoder/layer_0/output/LayerNorm/moments/mean*
T0
É
Dcond/bert/encoder/layer_0/output/LayerNorm/moments/SquaredDifferenceSquaredDifference$cond/bert/encoder/layer_0/output/add?cond/bert/encoder/layer_0/output/LayerNorm/moments/StopGradient*
T0

Mcond/bert/encoder/layer_0/output/LayerNorm/moments/variance/reduction_indicesConst^cond/switch_t*
dtype0*
valueB:
ū
;cond/bert/encoder/layer_0/output/LayerNorm/moments/varianceMeanDcond/bert/encoder/layer_0/output/LayerNorm/moments/SquaredDifferenceMcond/bert/encoder/layer_0/output/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0
w
:cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/add/yConst^cond/switch_t*
valueB
 *Ėŧ+*
dtype0
Á
8cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/addAdd;cond/bert/encoder/layer_0/output/LayerNorm/moments/variance:cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/add/y*
T0

:cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/RsqrtRsqrt8cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/add*
T0
Į
8cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/mulMul:cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/RsqrtAcond/bert/encoder/layer_0/output/LayerNorm/batchnorm/mul/Switch:1*
T0
ų
?cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/mul/SwitchSwitchAmio_variable/bert/encoder/layer_0/output/LayerNorm/gamma/variablecond/pred_id*T
_classJ
HFloc:@mio_variable/bert/encoder/layer_0/output/LayerNorm/gamma/variable*
T0
Ē
:cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_1Mul$cond/bert/encoder/layer_0/output/add8cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/mul*
T0
Ŋ
:cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_2Mul7cond/bert/encoder/layer_0/output/LayerNorm/moments/mean8cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/mul*
T0
Į
8cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/subSubAcond/bert/encoder/layer_0/output/LayerNorm/batchnorm/sub/Switch:1:cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_2*
T0
÷
?cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/sub/SwitchSwitch@mio_variable/bert/encoder/layer_0/output/LayerNorm/beta/variablecond/pred_id*
T0*S
_classI
GEloc:@mio_variable/bert/encoder/layer_0/output/LayerNorm/beta/variable
Ā
:cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1Add:cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_18cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/sub*
T0

.cond/bert/encoder/layer_1/attention/self/ShapeShape:cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
z
<cond/bert/encoder/layer_1/attention/self/strided_slice/stackConst^cond/switch_t*
valueB: *
dtype0
|
>cond/bert/encoder/layer_1/attention/self/strided_slice/stack_1Const^cond/switch_t*
valueB:*
dtype0
|
>cond/bert/encoder/layer_1/attention/self/strided_slice/stack_2Const^cond/switch_t*
valueB:*
dtype0
Ž
6cond/bert/encoder/layer_1/attention/self/strided_sliceStridedSlice.cond/bert/encoder/layer_1/attention/self/Shape<cond/bert/encoder/layer_1/attention/self/strided_slice/stack>cond/bert/encoder/layer_1/attention/self/strided_slice/stack_1>cond/bert/encoder/layer_1/attention/self/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0

0cond/bert/encoder/layer_1/attention/self/Shape_1Shape:cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
|
>cond/bert/encoder/layer_1/attention/self/strided_slice_1/stackConst^cond/switch_t*
valueB: *
dtype0
~
@cond/bert/encoder/layer_1/attention/self/strided_slice_1/stack_1Const^cond/switch_t*
valueB:*
dtype0
~
@cond/bert/encoder/layer_1/attention/self/strided_slice_1/stack_2Const^cond/switch_t*
valueB:*
dtype0
¸
8cond/bert/encoder/layer_1/attention/self/strided_slice_1StridedSlice0cond/bert/encoder/layer_1/attention/self/Shape_1>cond/bert/encoder/layer_1/attention/self/strided_slice_1/stack@cond/bert/encoder/layer_1/attention/self/strided_slice_1/stack_1@cond/bert/encoder/layer_1/attention/self/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
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
&Initializer_21/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_21/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_21/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 

#Initializer_21/truncated_normal/mulMul/Initializer_21/truncated_normal/TruncatedNormal&Initializer_21/truncated_normal/stddev*
T0
z
Initializer_21/truncated_normalAdd#Initializer_21/truncated_normal/mul$Initializer_21/truncated_normal/mean*
T0

	Assign_21AssignFmio_variable/bert/encoder/layer_1/attention/self/query/kernel/gradientInitializer_21/truncated_normal*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_1/attention/self/query/kernel/gradient*
validate_shape(
É
Dmio_variable/bert/encoder/layer_1/attention/self/query/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*=
	container0.bert/encoder/layer_1/attention/self/query/bias
É
Dmio_variable/bert/encoder/layer_1/attention/self/query/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_1/attention/self/query/bias*
shape:
F
Initializer_22/zerosConst*
valueB*    *
dtype0
ú
	Assign_22AssignDmio_variable/bert/encoder/layer_1/attention/self/query/bias/gradientInitializer_22/zeros*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_1/attention/self/query/bias/gradient*
validate_shape(
ę
5cond/bert/encoder/layer_1/attention/self/query/MatMulMatMul:cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1>cond/bert/encoder/layer_1/attention/self/query/MatMul/Switch:1*
transpose_a( *
transpose_b( *
T0

<cond/bert/encoder/layer_1/attention/self/query/MatMul/SwitchSwitchFmio_variable/bert/encoder/layer_1/attention/self/query/kernel/variablecond/pred_id*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_1/attention/self/query/kernel/variable
Ų
6cond/bert/encoder/layer_1/attention/self/query/BiasAddBiasAdd5cond/bert/encoder/layer_1/attention/self/query/MatMul?cond/bert/encoder/layer_1/attention/self/query/BiasAdd/Switch:1*
data_formatNHWC*
T0
ũ
=cond/bert/encoder/layer_1/attention/self/query/BiasAdd/SwitchSwitchDmio_variable/bert/encoder/layer_1/attention/self/query/bias/variablecond/pred_id*W
_classM
KIloc:@mio_variable/bert/encoder/layer_1/attention/self/query/bias/variable*
T0
Î
Dmio_variable/bert/encoder/layer_1/attention/self/key/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_1/attention/self/key/kernel*
shape:

Î
Dmio_variable/bert/encoder/layer_1/attention/self/key/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*=
	container0.bert/encoder/layer_1/attention/self/key/kernel
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
dtype0*
seed2 *

seed *
T0

#Initializer_23/truncated_normal/mulMul/Initializer_23/truncated_normal/TruncatedNormal&Initializer_23/truncated_normal/stddev*
T0
z
Initializer_23/truncated_normalAdd#Initializer_23/truncated_normal/mul$Initializer_23/truncated_normal/mean*
T0

	Assign_23AssignDmio_variable/bert/encoder/layer_1/attention/self/key/kernel/gradientInitializer_23/truncated_normal*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_1/attention/self/key/kernel/gradient*
validate_shape(*
use_locking(
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
	Assign_24AssignBmio_variable/bert/encoder/layer_1/attention/self/key/bias/gradientInitializer_24/zeros*U
_classK
IGloc:@mio_variable/bert/encoder/layer_1/attention/self/key/bias/gradient*
validate_shape(*
use_locking(*
T0
æ
3cond/bert/encoder/layer_1/attention/self/key/MatMulMatMul:cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1<cond/bert/encoder/layer_1/attention/self/key/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 
ú
:cond/bert/encoder/layer_1/attention/self/key/MatMul/SwitchSwitchDmio_variable/bert/encoder/layer_1/attention/self/key/kernel/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_1/attention/self/key/kernel/variable
Ķ
4cond/bert/encoder/layer_1/attention/self/key/BiasAddBiasAdd3cond/bert/encoder/layer_1/attention/self/key/MatMul=cond/bert/encoder/layer_1/attention/self/key/BiasAdd/Switch:1*
data_formatNHWC*
T0
÷
;cond/bert/encoder/layer_1/attention/self/key/BiasAdd/SwitchSwitchBmio_variable/bert/encoder/layer_1/attention/self/key/bias/variablecond/pred_id*U
_classK
IGloc:@mio_variable/bert/encoder/layer_1/attention/self/key/bias/variable*
T0
Ō
Fmio_variable/bert/encoder/layer_1/attention/self/value/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_1/attention/self/value/kernel*
shape:

Ō
Fmio_variable/bert/encoder/layer_1/attention/self/value/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*?
	container20bert/encoder/layer_1/attention/self/value/kernel
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
	Assign_25AssignFmio_variable/bert/encoder/layer_1/attention/self/value/kernel/gradientInitializer_25/truncated_normal*
validate_shape(*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_1/attention/self/value/kernel/gradient
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
	Assign_26AssignDmio_variable/bert/encoder/layer_1/attention/self/value/bias/gradientInitializer_26/zeros*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_1/attention/self/value/bias/gradient*
validate_shape(*
use_locking(
ę
5cond/bert/encoder/layer_1/attention/self/value/MatMulMatMul:cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1>cond/bert/encoder/layer_1/attention/self/value/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 

<cond/bert/encoder/layer_1/attention/self/value/MatMul/SwitchSwitchFmio_variable/bert/encoder/layer_1/attention/self/value/kernel/variablecond/pred_id*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_1/attention/self/value/kernel/variable
Ų
6cond/bert/encoder/layer_1/attention/self/value/BiasAddBiasAdd5cond/bert/encoder/layer_1/attention/self/value/MatMul?cond/bert/encoder/layer_1/attention/self/value/BiasAdd/Switch:1*
T0*
data_formatNHWC
ũ
=cond/bert/encoder/layer_1/attention/self/value/BiasAdd/SwitchSwitchDmio_variable/bert/encoder/layer_1/attention/self/value/bias/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_1/attention/self/value/bias/variable
r
8cond/bert/encoder/layer_1/attention/self/Reshape/shape/1Const^cond/switch_t*
value	B :*
dtype0
r
8cond/bert/encoder/layer_1/attention/self/Reshape/shape/2Const^cond/switch_t*
value	B :*
dtype0
r
8cond/bert/encoder/layer_1/attention/self/Reshape/shape/3Const^cond/switch_t*
value	B : *
dtype0
­
6cond/bert/encoder/layer_1/attention/self/Reshape/shapePack!cond/bert/encoder/strided_slice_28cond/bert/encoder/layer_1/attention/self/Reshape/shape/18cond/bert/encoder/layer_1/attention/self/Reshape/shape/28cond/bert/encoder/layer_1/attention/self/Reshape/shape/3*
T0*

axis *
N
Â
0cond/bert/encoder/layer_1/attention/self/ReshapeReshape6cond/bert/encoder/layer_1/attention/self/query/BiasAdd6cond/bert/encoder/layer_1/attention/self/Reshape/shape*
T0*
Tshape0

7cond/bert/encoder/layer_1/attention/self/transpose/permConst^cond/switch_t*%
valueB"             *
dtype0
Ā
2cond/bert/encoder/layer_1/attention/self/transpose	Transpose0cond/bert/encoder/layer_1/attention/self/Reshape7cond/bert/encoder/layer_1/attention/self/transpose/perm*
T0*
Tperm0
t
:cond/bert/encoder/layer_1/attention/self/Reshape_1/shape/1Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_1/attention/self/Reshape_1/shape/2Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_1/attention/self/Reshape_1/shape/3Const^cond/switch_t*
value	B : *
dtype0
ĩ
8cond/bert/encoder/layer_1/attention/self/Reshape_1/shapePack!cond/bert/encoder/strided_slice_2:cond/bert/encoder/layer_1/attention/self/Reshape_1/shape/1:cond/bert/encoder/layer_1/attention/self/Reshape_1/shape/2:cond/bert/encoder/layer_1/attention/self/Reshape_1/shape/3*
T0*

axis *
N
Ä
2cond/bert/encoder/layer_1/attention/self/Reshape_1Reshape4cond/bert/encoder/layer_1/attention/self/key/BiasAdd8cond/bert/encoder/layer_1/attention/self/Reshape_1/shape*
T0*
Tshape0

9cond/bert/encoder/layer_1/attention/self/transpose_1/permConst^cond/switch_t*%
valueB"             *
dtype0
Æ
4cond/bert/encoder/layer_1/attention/self/transpose_1	Transpose2cond/bert/encoder/layer_1/attention/self/Reshape_19cond/bert/encoder/layer_1/attention/self/transpose_1/perm*
Tperm0*
T0
Ë
/cond/bert/encoder/layer_1/attention/self/MatMulBatchMatMul2cond/bert/encoder/layer_1/attention/self/transpose4cond/bert/encoder/layer_1/attention/self/transpose_1*
adj_x( *
adj_y(*
T0
k
.cond/bert/encoder/layer_1/attention/self/Mul/yConst^cond/switch_t*
valueB
 *ķ5>*
dtype0

,cond/bert/encoder/layer_1/attention/self/MulMul/cond/bert/encoder/layer_1/attention/self/MatMul.cond/bert/encoder/layer_1/attention/self/Mul/y*
T0
u
7cond/bert/encoder/layer_1/attention/self/ExpandDims/dimConst^cond/switch_t*
valueB:*
dtype0
Ļ
3cond/bert/encoder/layer_1/attention/self/ExpandDims
ExpandDimscond/bert/encoder/mul7cond/bert/encoder/layer_1/attention/self/ExpandDims/dim*

Tdim0*
T0
k
.cond/bert/encoder/layer_1/attention/self/sub/xConst^cond/switch_t*
valueB
 *  ?*
dtype0
Ą
,cond/bert/encoder/layer_1/attention/self/subSub.cond/bert/encoder/layer_1/attention/self/sub/x3cond/bert/encoder/layer_1/attention/self/ExpandDims*
T0
m
0cond/bert/encoder/layer_1/attention/self/mul_1/yConst^cond/switch_t*
valueB
 * @Æ*
dtype0

.cond/bert/encoder/layer_1/attention/self/mul_1Mul,cond/bert/encoder/layer_1/attention/self/sub0cond/bert/encoder/layer_1/attention/self/mul_1/y*
T0

,cond/bert/encoder/layer_1/attention/self/addAdd,cond/bert/encoder/layer_1/attention/self/Mul.cond/bert/encoder/layer_1/attention/self/mul_1*
T0
r
0cond/bert/encoder/layer_1/attention/self/SoftmaxSoftmax,cond/bert/encoder/layer_1/attention/self/add*
T0
w
:cond/bert/encoder/layer_1/attention/self/dropout/keep_probConst^cond/switch_t*
valueB
 *fff?*
dtype0

6cond/bert/encoder/layer_1/attention/self/dropout/ShapeShape0cond/bert/encoder/layer_1/attention/self/Softmax*
T0*
out_type0

Ccond/bert/encoder/layer_1/attention/self/dropout/random_uniform/minConst^cond/switch_t*
dtype0*
valueB
 *    

Ccond/bert/encoder/layer_1/attention/self/dropout/random_uniform/maxConst^cond/switch_t*
valueB
 *  ?*
dtype0
Å
Mcond/bert/encoder/layer_1/attention/self/dropout/random_uniform/RandomUniformRandomUniform6cond/bert/encoder/layer_1/attention/self/dropout/Shape*
dtype0*
seed2 *

seed *
T0
Ũ
Ccond/bert/encoder/layer_1/attention/self/dropout/random_uniform/subSubCcond/bert/encoder/layer_1/attention/self/dropout/random_uniform/maxCcond/bert/encoder/layer_1/attention/self/dropout/random_uniform/min*
T0
į
Ccond/bert/encoder/layer_1/attention/self/dropout/random_uniform/mulMulMcond/bert/encoder/layer_1/attention/self/dropout/random_uniform/RandomUniformCcond/bert/encoder/layer_1/attention/self/dropout/random_uniform/sub*
T0
Ų
?cond/bert/encoder/layer_1/attention/self/dropout/random_uniformAddCcond/bert/encoder/layer_1/attention/self/dropout/random_uniform/mulCcond/bert/encoder/layer_1/attention/self/dropout/random_uniform/min*
T0
Á
4cond/bert/encoder/layer_1/attention/self/dropout/addAdd:cond/bert/encoder/layer_1/attention/self/dropout/keep_prob?cond/bert/encoder/layer_1/attention/self/dropout/random_uniform*
T0
~
6cond/bert/encoder/layer_1/attention/self/dropout/FloorFloor4cond/bert/encoder/layer_1/attention/self/dropout/add*
T0
ļ
4cond/bert/encoder/layer_1/attention/self/dropout/divRealDiv0cond/bert/encoder/layer_1/attention/self/Softmax:cond/bert/encoder/layer_1/attention/self/dropout/keep_prob*
T0
˛
4cond/bert/encoder/layer_1/attention/self/dropout/mulMul4cond/bert/encoder/layer_1/attention/self/dropout/div6cond/bert/encoder/layer_1/attention/self/dropout/Floor*
T0
t
:cond/bert/encoder/layer_1/attention/self/Reshape_2/shape/1Const^cond/switch_t*
dtype0*
value	B :
t
:cond/bert/encoder/layer_1/attention/self/Reshape_2/shape/2Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_1/attention/self/Reshape_2/shape/3Const^cond/switch_t*
value	B : *
dtype0
ĩ
8cond/bert/encoder/layer_1/attention/self/Reshape_2/shapePack!cond/bert/encoder/strided_slice_2:cond/bert/encoder/layer_1/attention/self/Reshape_2/shape/1:cond/bert/encoder/layer_1/attention/self/Reshape_2/shape/2:cond/bert/encoder/layer_1/attention/self/Reshape_2/shape/3*
T0*

axis *
N
Æ
2cond/bert/encoder/layer_1/attention/self/Reshape_2Reshape6cond/bert/encoder/layer_1/attention/self/value/BiasAdd8cond/bert/encoder/layer_1/attention/self/Reshape_2/shape*
T0*
Tshape0

9cond/bert/encoder/layer_1/attention/self/transpose_2/permConst^cond/switch_t*%
valueB"             *
dtype0
Æ
4cond/bert/encoder/layer_1/attention/self/transpose_2	Transpose2cond/bert/encoder/layer_1/attention/self/Reshape_29cond/bert/encoder/layer_1/attention/self/transpose_2/perm*
T0*
Tperm0
Ī
1cond/bert/encoder/layer_1/attention/self/MatMul_1BatchMatMul4cond/bert/encoder/layer_1/attention/self/dropout/mul4cond/bert/encoder/layer_1/attention/self/transpose_2*
adj_x( *
adj_y( *
T0

9cond/bert/encoder/layer_1/attention/self/transpose_3/permConst^cond/switch_t*%
valueB"             *
dtype0
Å
4cond/bert/encoder/layer_1/attention/self/transpose_3	Transpose1cond/bert/encoder/layer_1/attention/self/MatMul_19cond/bert/encoder/layer_1/attention/self/transpose_3/perm*
Tperm0*
T0
j
0cond/bert/encoder/layer_1/attention/self/mul_2/yConst^cond/switch_t*
value	B :*
dtype0

.cond/bert/encoder/layer_1/attention/self/mul_2Mul!cond/bert/encoder/strided_slice_20cond/bert/encoder/layer_1/attention/self/mul_2/y*
T0
u
:cond/bert/encoder/layer_1/attention/self/Reshape_3/shape/1Const^cond/switch_t*
value
B :*
dtype0
Ę
8cond/bert/encoder/layer_1/attention/self/Reshape_3/shapePack.cond/bert/encoder/layer_1/attention/self/mul_2:cond/bert/encoder/layer_1/attention/self/Reshape_3/shape/1*
N*
T0*

axis 
Ä
2cond/bert/encoder/layer_1/attention/self/Reshape_3Reshape4cond/bert/encoder/layer_1/attention/self/transpose_38cond/bert/encoder/layer_1/attention/self/Reshape_3/shape*
T0*
Tshape0
Ö
Hmio_variable/bert/encoder/layer_1/attention/output/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42bert/encoder/layer_1/attention/output/dense/kernel*
shape:

Ö
Hmio_variable/bert/encoder/layer_1/attention/output/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*A
	container42bert/encoder/layer_1/attention/output/dense/kernel
Z
%Initializer_27/truncated_normal/shapeConst*
dtype0*
valueB"      
Q
$Initializer_27/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_27/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_27/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_27/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 
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
æ
7cond/bert/encoder/layer_1/attention/output/dense/MatMulMatMul2cond/bert/encoder/layer_1/attention/self/Reshape_3@cond/bert/encoder/layer_1/attention/output/dense/MatMul/Switch:1*
transpose_a( *
transpose_b( *
T0

>cond/bert/encoder/layer_1/attention/output/dense/MatMul/SwitchSwitchHmio_variable/bert/encoder/layer_1/attention/output/dense/kernel/variablecond/pred_id*
T0*[
_classQ
OMloc:@mio_variable/bert/encoder/layer_1/attention/output/dense/kernel/variable
ß
8cond/bert/encoder/layer_1/attention/output/dense/BiasAddBiasAdd7cond/bert/encoder/layer_1/attention/output/dense/MatMulAcond/bert/encoder/layer_1/attention/output/dense/BiasAdd/Switch:1*
T0*
data_formatNHWC

?cond/bert/encoder/layer_1/attention/output/dense/BiasAdd/SwitchSwitchFmio_variable/bert/encoder/layer_1/attention/output/dense/bias/variablecond/pred_id*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_1/attention/output/dense/bias/variable
y
<cond/bert/encoder/layer_1/attention/output/dropout/keep_probConst^cond/switch_t*
valueB
 *fff?*
dtype0

8cond/bert/encoder/layer_1/attention/output/dropout/ShapeShape8cond/bert/encoder/layer_1/attention/output/dense/BiasAdd*
T0*
out_type0

Econd/bert/encoder/layer_1/attention/output/dropout/random_uniform/minConst^cond/switch_t*
dtype0*
valueB
 *    

Econd/bert/encoder/layer_1/attention/output/dropout/random_uniform/maxConst^cond/switch_t*
valueB
 *  ?*
dtype0
É
Ocond/bert/encoder/layer_1/attention/output/dropout/random_uniform/RandomUniformRandomUniform8cond/bert/encoder/layer_1/attention/output/dropout/Shape*

seed *
T0*
dtype0*
seed2 
ã
Econd/bert/encoder/layer_1/attention/output/dropout/random_uniform/subSubEcond/bert/encoder/layer_1/attention/output/dropout/random_uniform/maxEcond/bert/encoder/layer_1/attention/output/dropout/random_uniform/min*
T0
í
Econd/bert/encoder/layer_1/attention/output/dropout/random_uniform/mulMulOcond/bert/encoder/layer_1/attention/output/dropout/random_uniform/RandomUniformEcond/bert/encoder/layer_1/attention/output/dropout/random_uniform/sub*
T0
ß
Acond/bert/encoder/layer_1/attention/output/dropout/random_uniformAddEcond/bert/encoder/layer_1/attention/output/dropout/random_uniform/mulEcond/bert/encoder/layer_1/attention/output/dropout/random_uniform/min*
T0
Į
6cond/bert/encoder/layer_1/attention/output/dropout/addAdd<cond/bert/encoder/layer_1/attention/output/dropout/keep_probAcond/bert/encoder/layer_1/attention/output/dropout/random_uniform*
T0

8cond/bert/encoder/layer_1/attention/output/dropout/FloorFloor6cond/bert/encoder/layer_1/attention/output/dropout/add*
T0
Â
6cond/bert/encoder/layer_1/attention/output/dropout/divRealDiv8cond/bert/encoder/layer_1/attention/output/dense/BiasAdd<cond/bert/encoder/layer_1/attention/output/dropout/keep_prob*
T0
¸
6cond/bert/encoder/layer_1/attention/output/dropout/mulMul6cond/bert/encoder/layer_1/attention/output/dropout/div8cond/bert/encoder/layer_1/attention/output/dropout/Floor*
T0
˛
.cond/bert/encoder/layer_1/attention/output/addAdd6cond/bert/encoder/layer_1/attention/output/dropout/mul:cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1*
T0
Õ
Jmio_variable/bert/encoder/layer_1/attention/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64bert/encoder/layer_1/attention/output/LayerNorm/beta*
shape:
Õ
Jmio_variable/bert/encoder/layer_1/attention/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64bert/encoder/layer_1/attention/output/LayerNorm/beta*
shape:
F
Initializer_29/zerosConst*
dtype0*
valueB*    

	Assign_29AssignJmio_variable/bert/encoder/layer_1/attention/output/LayerNorm/beta/gradientInitializer_29/zeros*
validate_shape(*
use_locking(*
T0*]
_classS
QOloc:@mio_variable/bert/encoder/layer_1/attention/output/LayerNorm/beta/gradient
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

Scond/bert/encoder/layer_1/attention/output/LayerNorm/moments/mean/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0
ô
Acond/bert/encoder/layer_1/attention/output/LayerNorm/moments/meanMean.cond/bert/encoder/layer_1/attention/output/addScond/bert/encoder/layer_1/attention/output/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
Ĩ
Icond/bert/encoder/layer_1/attention/output/LayerNorm/moments/StopGradientStopGradientAcond/bert/encoder/layer_1/attention/output/LayerNorm/moments/mean*
T0
į
Ncond/bert/encoder/layer_1/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference.cond/bert/encoder/layer_1/attention/output/addIcond/bert/encoder/layer_1/attention/output/LayerNorm/moments/StopGradient*
T0

Wcond/bert/encoder/layer_1/attention/output/LayerNorm/moments/variance/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0

Econd/bert/encoder/layer_1/attention/output/LayerNorm/moments/varianceMeanNcond/bert/encoder/layer_1/attention/output/LayerNorm/moments/SquaredDifferenceWcond/bert/encoder/layer_1/attention/output/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0

Dcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add/yConst^cond/switch_t*
valueB
 *Ėŧ+*
dtype0
ß
Bcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/addAddEcond/bert/encoder/layer_1/attention/output/LayerNorm/moments/varianceDcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add/y*
T0

Dcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/RsqrtRsqrtBcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add*
T0
å
Bcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mulMulDcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/RsqrtKcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul/Switch:1*
T0

Icond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul/SwitchSwitchKmio_variable/bert/encoder/layer_1/attention/output/LayerNorm/gamma/variablecond/pred_id*
T0*^
_classT
RPloc:@mio_variable/bert/encoder/layer_1/attention/output/LayerNorm/gamma/variable
Č
Dcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_1Mul.cond/bert/encoder/layer_1/attention/output/addBcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul*
T0
Û
Dcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_2MulAcond/bert/encoder/layer_1/attention/output/LayerNorm/moments/meanBcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul*
T0
å
Bcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/subSubKcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/sub/Switch:1Dcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_2*
T0

Icond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/sub/SwitchSwitchJmio_variable/bert/encoder/layer_1/attention/output/LayerNorm/beta/variablecond/pred_id*
T0*]
_classS
QOloc:@mio_variable/bert/encoder/layer_1/attention/output/LayerNorm/beta/variable
Ū
Dcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add_1AddDcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_1Bcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/sub*
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
dtype0*
seed2 *

seed *
T0

#Initializer_31/truncated_normal/mulMul/Initializer_31/truncated_normal/TruncatedNormal&Initializer_31/truncated_normal/stddev*
T0
z
Initializer_31/truncated_normalAdd#Initializer_31/truncated_normal/mul$Initializer_31/truncated_normal/mean*
T0

	Assign_31AssignDmio_variable/bert/encoder/layer_1/intermediate/dense/kernel/gradientInitializer_31/truncated_normal*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_1/intermediate/dense/kernel/gradient*
validate_shape(*
use_locking(
Å
Bmio_variable/bert/encoder/layer_1/intermediate/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_1/intermediate/dense/bias*
shape:
Å
Bmio_variable/bert/encoder/layer_1/intermediate/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_1/intermediate/dense/bias*
shape:
S
$Initializer_32/zeros/shape_as_tensorConst*
dtype0*
valueB:
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
	Assign_32AssignBmio_variable/bert/encoder/layer_1/intermediate/dense/bias/gradientInitializer_32/zeros*
validate_shape(*
use_locking(*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_1/intermediate/dense/bias/gradient
đ
3cond/bert/encoder/layer_1/intermediate/dense/MatMulMatMulDcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add_1<cond/bert/encoder/layer_1/intermediate/dense/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 
ú
:cond/bert/encoder/layer_1/intermediate/dense/MatMul/SwitchSwitchDmio_variable/bert/encoder/layer_1/intermediate/dense/kernel/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_1/intermediate/dense/kernel/variable
Ķ
4cond/bert/encoder/layer_1/intermediate/dense/BiasAddBiasAdd3cond/bert/encoder/layer_1/intermediate/dense/MatMul=cond/bert/encoder/layer_1/intermediate/dense/BiasAdd/Switch:1*
T0*
data_formatNHWC
÷
;cond/bert/encoder/layer_1/intermediate/dense/BiasAdd/SwitchSwitchBmio_variable/bert/encoder/layer_1/intermediate/dense/bias/variablecond/pred_id*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_1/intermediate/dense/bias/variable
o
2cond/bert/encoder/layer_1/intermediate/dense/Pow/yConst^cond/switch_t*
valueB
 *  @@*
dtype0
Ē
0cond/bert/encoder/layer_1/intermediate/dense/PowPow4cond/bert/encoder/layer_1/intermediate/dense/BiasAdd2cond/bert/encoder/layer_1/intermediate/dense/Pow/y*
T0
o
2cond/bert/encoder/layer_1/intermediate/dense/mul/xConst^cond/switch_t*
valueB
 *'7=*
dtype0
Ļ
0cond/bert/encoder/layer_1/intermediate/dense/mulMul2cond/bert/encoder/layer_1/intermediate/dense/mul/x0cond/bert/encoder/layer_1/intermediate/dense/Pow*
T0
¨
0cond/bert/encoder/layer_1/intermediate/dense/addAdd4cond/bert/encoder/layer_1/intermediate/dense/BiasAdd0cond/bert/encoder/layer_1/intermediate/dense/mul*
T0
q
4cond/bert/encoder/layer_1/intermediate/dense/mul_1/xConst^cond/switch_t*
dtype0*
valueB
 **BL?
Ē
2cond/bert/encoder/layer_1/intermediate/dense/mul_1Mul4cond/bert/encoder/layer_1/intermediate/dense/mul_1/x0cond/bert/encoder/layer_1/intermediate/dense/add*
T0
v
1cond/bert/encoder/layer_1/intermediate/dense/TanhTanh2cond/bert/encoder/layer_1/intermediate/dense/mul_1*
T0
q
4cond/bert/encoder/layer_1/intermediate/dense/add_1/xConst^cond/switch_t*
dtype0*
valueB
 *  ?
Ģ
2cond/bert/encoder/layer_1/intermediate/dense/add_1Add4cond/bert/encoder/layer_1/intermediate/dense/add_1/x1cond/bert/encoder/layer_1/intermediate/dense/Tanh*
T0
q
4cond/bert/encoder/layer_1/intermediate/dense/mul_2/xConst^cond/switch_t*
valueB
 *   ?*
dtype0
Ŧ
2cond/bert/encoder/layer_1/intermediate/dense/mul_2Mul4cond/bert/encoder/layer_1/intermediate/dense/mul_2/x2cond/bert/encoder/layer_1/intermediate/dense/add_1*
T0
Ŧ
2cond/bert/encoder/layer_1/intermediate/dense/mul_3Mul4cond/bert/encoder/layer_1/intermediate/dense/BiasAdd2cond/bert/encoder/layer_1/intermediate/dense/mul_2*
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
valueB"      *
dtype0
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
/Initializer_33/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_33/truncated_normal/shape*
seed2 *

seed *
T0*
dtype0

#Initializer_33/truncated_normal/mulMul/Initializer_33/truncated_normal/TruncatedNormal&Initializer_33/truncated_normal/stddev*
T0
z
Initializer_33/truncated_normalAdd#Initializer_33/truncated_normal/mul$Initializer_33/truncated_normal/mean*
T0
ų
	Assign_33Assign>mio_variable/bert/encoder/layer_1/output/dense/kernel/gradientInitializer_33/truncated_normal*
use_locking(*
T0*Q
_classG
ECloc:@mio_variable/bert/encoder/layer_1/output/dense/kernel/gradient*
validate_shape(
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
Ō
-cond/bert/encoder/layer_1/output/dense/MatMulMatMul2cond/bert/encoder/layer_1/intermediate/dense/mul_36cond/bert/encoder/layer_1/output/dense/MatMul/Switch:1*
transpose_a( *
transpose_b( *
T0
č
4cond/bert/encoder/layer_1/output/dense/MatMul/SwitchSwitch>mio_variable/bert/encoder/layer_1/output/dense/kernel/variablecond/pred_id*Q
_classG
ECloc:@mio_variable/bert/encoder/layer_1/output/dense/kernel/variable*
T0
Á
.cond/bert/encoder/layer_1/output/dense/BiasAddBiasAdd-cond/bert/encoder/layer_1/output/dense/MatMul7cond/bert/encoder/layer_1/output/dense/BiasAdd/Switch:1*
T0*
data_formatNHWC
å
5cond/bert/encoder/layer_1/output/dense/BiasAdd/SwitchSwitch<mio_variable/bert/encoder/layer_1/output/dense/bias/variablecond/pred_id*
T0*O
_classE
CAloc:@mio_variable/bert/encoder/layer_1/output/dense/bias/variable
o
2cond/bert/encoder/layer_1/output/dropout/keep_probConst^cond/switch_t*
valueB
 *fff?*
dtype0

.cond/bert/encoder/layer_1/output/dropout/ShapeShape.cond/bert/encoder/layer_1/output/dense/BiasAdd*
T0*
out_type0
x
;cond/bert/encoder/layer_1/output/dropout/random_uniform/minConst^cond/switch_t*
valueB
 *    *
dtype0
x
;cond/bert/encoder/layer_1/output/dropout/random_uniform/maxConst^cond/switch_t*
valueB
 *  ?*
dtype0
ĩ
Econd/bert/encoder/layer_1/output/dropout/random_uniform/RandomUniformRandomUniform.cond/bert/encoder/layer_1/output/dropout/Shape*
seed2 *

seed *
T0*
dtype0
Å
;cond/bert/encoder/layer_1/output/dropout/random_uniform/subSub;cond/bert/encoder/layer_1/output/dropout/random_uniform/max;cond/bert/encoder/layer_1/output/dropout/random_uniform/min*
T0
Ī
;cond/bert/encoder/layer_1/output/dropout/random_uniform/mulMulEcond/bert/encoder/layer_1/output/dropout/random_uniform/RandomUniform;cond/bert/encoder/layer_1/output/dropout/random_uniform/sub*
T0
Á
7cond/bert/encoder/layer_1/output/dropout/random_uniformAdd;cond/bert/encoder/layer_1/output/dropout/random_uniform/mul;cond/bert/encoder/layer_1/output/dropout/random_uniform/min*
T0
Š
,cond/bert/encoder/layer_1/output/dropout/addAdd2cond/bert/encoder/layer_1/output/dropout/keep_prob7cond/bert/encoder/layer_1/output/dropout/random_uniform*
T0
n
.cond/bert/encoder/layer_1/output/dropout/FloorFloor,cond/bert/encoder/layer_1/output/dropout/add*
T0
¤
,cond/bert/encoder/layer_1/output/dropout/divRealDiv.cond/bert/encoder/layer_1/output/dense/BiasAdd2cond/bert/encoder/layer_1/output/dropout/keep_prob*
T0

,cond/bert/encoder/layer_1/output/dropout/mulMul,cond/bert/encoder/layer_1/output/dropout/div.cond/bert/encoder/layer_1/output/dropout/Floor*
T0
¨
$cond/bert/encoder/layer_1/output/addAdd,cond/bert/encoder/layer_1/output/dropout/mulDcond/bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add_1*
T0
Á
@mio_variable/bert/encoder/layer_1/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*bert/encoder/layer_1/output/LayerNorm/beta*
shape:
Á
@mio_variable/bert/encoder/layer_1/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*bert/encoder/layer_1/output/LayerNorm/beta*
shape:
F
Initializer_35/zerosConst*
valueB*    *
dtype0
ō
	Assign_35Assign@mio_variable/bert/encoder/layer_1/output/LayerNorm/beta/gradientInitializer_35/zeros*S
_classI
GEloc:@mio_variable/bert/encoder/layer_1/output/LayerNorm/beta/gradient*
validate_shape(*
use_locking(*
T0
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

Icond/bert/encoder/layer_1/output/LayerNorm/moments/mean/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0
Ö
7cond/bert/encoder/layer_1/output/LayerNorm/moments/meanMean$cond/bert/encoder/layer_1/output/addIcond/bert/encoder/layer_1/output/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0

?cond/bert/encoder/layer_1/output/LayerNorm/moments/StopGradientStopGradient7cond/bert/encoder/layer_1/output/LayerNorm/moments/mean*
T0
É
Dcond/bert/encoder/layer_1/output/LayerNorm/moments/SquaredDifferenceSquaredDifference$cond/bert/encoder/layer_1/output/add?cond/bert/encoder/layer_1/output/LayerNorm/moments/StopGradient*
T0

Mcond/bert/encoder/layer_1/output/LayerNorm/moments/variance/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0
ū
;cond/bert/encoder/layer_1/output/LayerNorm/moments/varianceMeanDcond/bert/encoder/layer_1/output/LayerNorm/moments/SquaredDifferenceMcond/bert/encoder/layer_1/output/LayerNorm/moments/variance/reduction_indices*
T0*

Tidx0*
	keep_dims(
w
:cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/add/yConst^cond/switch_t*
valueB
 *Ėŧ+*
dtype0
Á
8cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/addAdd;cond/bert/encoder/layer_1/output/LayerNorm/moments/variance:cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/add/y*
T0

:cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/RsqrtRsqrt8cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/add*
T0
Į
8cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/mulMul:cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/RsqrtAcond/bert/encoder/layer_1/output/LayerNorm/batchnorm/mul/Switch:1*
T0
ų
?cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/mul/SwitchSwitchAmio_variable/bert/encoder/layer_1/output/LayerNorm/gamma/variablecond/pred_id*
T0*T
_classJ
HFloc:@mio_variable/bert/encoder/layer_1/output/LayerNorm/gamma/variable
Ē
:cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_1Mul$cond/bert/encoder/layer_1/output/add8cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/mul*
T0
Ŋ
:cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_2Mul7cond/bert/encoder/layer_1/output/LayerNorm/moments/mean8cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/mul*
T0
Į
8cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/subSubAcond/bert/encoder/layer_1/output/LayerNorm/batchnorm/sub/Switch:1:cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_2*
T0
÷
?cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/sub/SwitchSwitch@mio_variable/bert/encoder/layer_1/output/LayerNorm/beta/variablecond/pred_id*
T0*S
_classI
GEloc:@mio_variable/bert/encoder/layer_1/output/LayerNorm/beta/variable
Ā
:cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1Add:cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_18cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/sub*
T0

.cond/bert/encoder/layer_2/attention/self/ShapeShape:cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
z
<cond/bert/encoder/layer_2/attention/self/strided_slice/stackConst^cond/switch_t*
valueB: *
dtype0
|
>cond/bert/encoder/layer_2/attention/self/strided_slice/stack_1Const^cond/switch_t*
valueB:*
dtype0
|
>cond/bert/encoder/layer_2/attention/self/strided_slice/stack_2Const^cond/switch_t*
valueB:*
dtype0
Ž
6cond/bert/encoder/layer_2/attention/self/strided_sliceStridedSlice.cond/bert/encoder/layer_2/attention/self/Shape<cond/bert/encoder/layer_2/attention/self/strided_slice/stack>cond/bert/encoder/layer_2/attention/self/strided_slice/stack_1>cond/bert/encoder/layer_2/attention/self/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 

0cond/bert/encoder/layer_2/attention/self/Shape_1Shape:cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
|
>cond/bert/encoder/layer_2/attention/self/strided_slice_1/stackConst^cond/switch_t*
valueB: *
dtype0
~
@cond/bert/encoder/layer_2/attention/self/strided_slice_1/stack_1Const^cond/switch_t*
valueB:*
dtype0
~
@cond/bert/encoder/layer_2/attention/self/strided_slice_1/stack_2Const^cond/switch_t*
valueB:*
dtype0
¸
8cond/bert/encoder/layer_2/attention/self/strided_slice_1StridedSlice0cond/bert/encoder/layer_2/attention/self/Shape_1>cond/bert/encoder/layer_2/attention/self/strided_slice_1/stack@cond/bert/encoder/layer_2/attention/self/strided_slice_1/stack_1@cond/bert/encoder/layer_2/attention/self/strided_slice_1/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
Ō
Fmio_variable/bert/encoder/layer_2/attention/self/query/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_2/attention/self/query/kernel*
shape:

Ō
Fmio_variable/bert/encoder/layer_2/attention/self/query/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_2/attention/self/query/kernel*
shape:

Z
%Initializer_37/truncated_normal/shapeConst*
dtype0*
valueB"      
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
	Assign_37AssignFmio_variable/bert/encoder/layer_2/attention/self/query/kernel/gradientInitializer_37/truncated_normal*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_2/attention/self/query/kernel/gradient*
validate_shape(
É
Dmio_variable/bert/encoder/layer_2/attention/self/query/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*=
	container0.bert/encoder/layer_2/attention/self/query/bias
É
Dmio_variable/bert/encoder/layer_2/attention/self/query/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_2/attention/self/query/bias*
shape:
F
Initializer_38/zerosConst*
valueB*    *
dtype0
ú
	Assign_38AssignDmio_variable/bert/encoder/layer_2/attention/self/query/bias/gradientInitializer_38/zeros*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_2/attention/self/query/bias/gradient*
validate_shape(
ę
5cond/bert/encoder/layer_2/attention/self/query/MatMulMatMul:cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1>cond/bert/encoder/layer_2/attention/self/query/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 

<cond/bert/encoder/layer_2/attention/self/query/MatMul/SwitchSwitchFmio_variable/bert/encoder/layer_2/attention/self/query/kernel/variablecond/pred_id*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_2/attention/self/query/kernel/variable
Ų
6cond/bert/encoder/layer_2/attention/self/query/BiasAddBiasAdd5cond/bert/encoder/layer_2/attention/self/query/MatMul?cond/bert/encoder/layer_2/attention/self/query/BiasAdd/Switch:1*
T0*
data_formatNHWC
ũ
=cond/bert/encoder/layer_2/attention/self/query/BiasAdd/SwitchSwitchDmio_variable/bert/encoder/layer_2/attention/self/query/bias/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_2/attention/self/query/bias/variable
Î
Dmio_variable/bert/encoder/layer_2/attention/self/key/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_2/attention/self/key/kernel*
shape:

Î
Dmio_variable/bert/encoder/layer_2/attention/self/key/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_2/attention/self/key/kernel*
shape:

Z
%Initializer_39/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_39/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_39/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_39/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_39/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 
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
	Assign_40AssignBmio_variable/bert/encoder/layer_2/attention/self/key/bias/gradientInitializer_40/zeros*
validate_shape(*
use_locking(*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_2/attention/self/key/bias/gradient
æ
3cond/bert/encoder/layer_2/attention/self/key/MatMulMatMul:cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1<cond/bert/encoder/layer_2/attention/self/key/MatMul/Switch:1*
transpose_a( *
transpose_b( *
T0
ú
:cond/bert/encoder/layer_2/attention/self/key/MatMul/SwitchSwitchDmio_variable/bert/encoder/layer_2/attention/self/key/kernel/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_2/attention/self/key/kernel/variable
Ķ
4cond/bert/encoder/layer_2/attention/self/key/BiasAddBiasAdd3cond/bert/encoder/layer_2/attention/self/key/MatMul=cond/bert/encoder/layer_2/attention/self/key/BiasAdd/Switch:1*
T0*
data_formatNHWC
÷
;cond/bert/encoder/layer_2/attention/self/key/BiasAdd/SwitchSwitchBmio_variable/bert/encoder/layer_2/attention/self/key/bias/variablecond/pred_id*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_2/attention/self/key/bias/variable
Ō
Fmio_variable/bert/encoder/layer_2/attention/self/value/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_2/attention/self/value/kernel*
shape:

Ō
Fmio_variable/bert/encoder/layer_2/attention/self/value/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_2/attention/self/value/kernel*
shape:

Z
%Initializer_41/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_41/truncated_normal/meanConst*
dtype0*
valueB
 *    
S
&Initializer_41/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_41/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_41/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0
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
Initializer_42/zerosConst*
dtype0*
valueB*    
ú
	Assign_42AssignDmio_variable/bert/encoder/layer_2/attention/self/value/bias/gradientInitializer_42/zeros*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_2/attention/self/value/bias/gradient*
validate_shape(*
use_locking(
ę
5cond/bert/encoder/layer_2/attention/self/value/MatMulMatMul:cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1>cond/bert/encoder/layer_2/attention/self/value/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 

<cond/bert/encoder/layer_2/attention/self/value/MatMul/SwitchSwitchFmio_variable/bert/encoder/layer_2/attention/self/value/kernel/variablecond/pred_id*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_2/attention/self/value/kernel/variable
Ų
6cond/bert/encoder/layer_2/attention/self/value/BiasAddBiasAdd5cond/bert/encoder/layer_2/attention/self/value/MatMul?cond/bert/encoder/layer_2/attention/self/value/BiasAdd/Switch:1*
T0*
data_formatNHWC
ũ
=cond/bert/encoder/layer_2/attention/self/value/BiasAdd/SwitchSwitchDmio_variable/bert/encoder/layer_2/attention/self/value/bias/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_2/attention/self/value/bias/variable
r
8cond/bert/encoder/layer_2/attention/self/Reshape/shape/1Const^cond/switch_t*
value	B :*
dtype0
r
8cond/bert/encoder/layer_2/attention/self/Reshape/shape/2Const^cond/switch_t*
value	B :*
dtype0
r
8cond/bert/encoder/layer_2/attention/self/Reshape/shape/3Const^cond/switch_t*
dtype0*
value	B : 
­
6cond/bert/encoder/layer_2/attention/self/Reshape/shapePack!cond/bert/encoder/strided_slice_28cond/bert/encoder/layer_2/attention/self/Reshape/shape/18cond/bert/encoder/layer_2/attention/self/Reshape/shape/28cond/bert/encoder/layer_2/attention/self/Reshape/shape/3*
T0*

axis *
N
Â
0cond/bert/encoder/layer_2/attention/self/ReshapeReshape6cond/bert/encoder/layer_2/attention/self/query/BiasAdd6cond/bert/encoder/layer_2/attention/self/Reshape/shape*
T0*
Tshape0

7cond/bert/encoder/layer_2/attention/self/transpose/permConst^cond/switch_t*%
valueB"             *
dtype0
Ā
2cond/bert/encoder/layer_2/attention/self/transpose	Transpose0cond/bert/encoder/layer_2/attention/self/Reshape7cond/bert/encoder/layer_2/attention/self/transpose/perm*
T0*
Tperm0
t
:cond/bert/encoder/layer_2/attention/self/Reshape_1/shape/1Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_2/attention/self/Reshape_1/shape/2Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_2/attention/self/Reshape_1/shape/3Const^cond/switch_t*
value	B : *
dtype0
ĩ
8cond/bert/encoder/layer_2/attention/self/Reshape_1/shapePack!cond/bert/encoder/strided_slice_2:cond/bert/encoder/layer_2/attention/self/Reshape_1/shape/1:cond/bert/encoder/layer_2/attention/self/Reshape_1/shape/2:cond/bert/encoder/layer_2/attention/self/Reshape_1/shape/3*
T0*

axis *
N
Ä
2cond/bert/encoder/layer_2/attention/self/Reshape_1Reshape4cond/bert/encoder/layer_2/attention/self/key/BiasAdd8cond/bert/encoder/layer_2/attention/self/Reshape_1/shape*
T0*
Tshape0

9cond/bert/encoder/layer_2/attention/self/transpose_1/permConst^cond/switch_t*%
valueB"             *
dtype0
Æ
4cond/bert/encoder/layer_2/attention/self/transpose_1	Transpose2cond/bert/encoder/layer_2/attention/self/Reshape_19cond/bert/encoder/layer_2/attention/self/transpose_1/perm*
Tperm0*
T0
Ë
/cond/bert/encoder/layer_2/attention/self/MatMulBatchMatMul2cond/bert/encoder/layer_2/attention/self/transpose4cond/bert/encoder/layer_2/attention/self/transpose_1*
adj_y(*
T0*
adj_x( 
k
.cond/bert/encoder/layer_2/attention/self/Mul/yConst^cond/switch_t*
valueB
 *ķ5>*
dtype0

,cond/bert/encoder/layer_2/attention/self/MulMul/cond/bert/encoder/layer_2/attention/self/MatMul.cond/bert/encoder/layer_2/attention/self/Mul/y*
T0
u
7cond/bert/encoder/layer_2/attention/self/ExpandDims/dimConst^cond/switch_t*
valueB:*
dtype0
Ļ
3cond/bert/encoder/layer_2/attention/self/ExpandDims
ExpandDimscond/bert/encoder/mul7cond/bert/encoder/layer_2/attention/self/ExpandDims/dim*

Tdim0*
T0
k
.cond/bert/encoder/layer_2/attention/self/sub/xConst^cond/switch_t*
valueB
 *  ?*
dtype0
Ą
,cond/bert/encoder/layer_2/attention/self/subSub.cond/bert/encoder/layer_2/attention/self/sub/x3cond/bert/encoder/layer_2/attention/self/ExpandDims*
T0
m
0cond/bert/encoder/layer_2/attention/self/mul_1/yConst^cond/switch_t*
valueB
 * @Æ*
dtype0

.cond/bert/encoder/layer_2/attention/self/mul_1Mul,cond/bert/encoder/layer_2/attention/self/sub0cond/bert/encoder/layer_2/attention/self/mul_1/y*
T0

,cond/bert/encoder/layer_2/attention/self/addAdd,cond/bert/encoder/layer_2/attention/self/Mul.cond/bert/encoder/layer_2/attention/self/mul_1*
T0
r
0cond/bert/encoder/layer_2/attention/self/SoftmaxSoftmax,cond/bert/encoder/layer_2/attention/self/add*
T0
w
:cond/bert/encoder/layer_2/attention/self/dropout/keep_probConst^cond/switch_t*
valueB
 *fff?*
dtype0

6cond/bert/encoder/layer_2/attention/self/dropout/ShapeShape0cond/bert/encoder/layer_2/attention/self/Softmax*
T0*
out_type0

Ccond/bert/encoder/layer_2/attention/self/dropout/random_uniform/minConst^cond/switch_t*
valueB
 *    *
dtype0

Ccond/bert/encoder/layer_2/attention/self/dropout/random_uniform/maxConst^cond/switch_t*
dtype0*
valueB
 *  ?
Å
Mcond/bert/encoder/layer_2/attention/self/dropout/random_uniform/RandomUniformRandomUniform6cond/bert/encoder/layer_2/attention/self/dropout/Shape*
T0*
dtype0*
seed2 *

seed 
Ũ
Ccond/bert/encoder/layer_2/attention/self/dropout/random_uniform/subSubCcond/bert/encoder/layer_2/attention/self/dropout/random_uniform/maxCcond/bert/encoder/layer_2/attention/self/dropout/random_uniform/min*
T0
į
Ccond/bert/encoder/layer_2/attention/self/dropout/random_uniform/mulMulMcond/bert/encoder/layer_2/attention/self/dropout/random_uniform/RandomUniformCcond/bert/encoder/layer_2/attention/self/dropout/random_uniform/sub*
T0
Ų
?cond/bert/encoder/layer_2/attention/self/dropout/random_uniformAddCcond/bert/encoder/layer_2/attention/self/dropout/random_uniform/mulCcond/bert/encoder/layer_2/attention/self/dropout/random_uniform/min*
T0
Á
4cond/bert/encoder/layer_2/attention/self/dropout/addAdd:cond/bert/encoder/layer_2/attention/self/dropout/keep_prob?cond/bert/encoder/layer_2/attention/self/dropout/random_uniform*
T0
~
6cond/bert/encoder/layer_2/attention/self/dropout/FloorFloor4cond/bert/encoder/layer_2/attention/self/dropout/add*
T0
ļ
4cond/bert/encoder/layer_2/attention/self/dropout/divRealDiv0cond/bert/encoder/layer_2/attention/self/Softmax:cond/bert/encoder/layer_2/attention/self/dropout/keep_prob*
T0
˛
4cond/bert/encoder/layer_2/attention/self/dropout/mulMul4cond/bert/encoder/layer_2/attention/self/dropout/div6cond/bert/encoder/layer_2/attention/self/dropout/Floor*
T0
t
:cond/bert/encoder/layer_2/attention/self/Reshape_2/shape/1Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_2/attention/self/Reshape_2/shape/2Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_2/attention/self/Reshape_2/shape/3Const^cond/switch_t*
value	B : *
dtype0
ĩ
8cond/bert/encoder/layer_2/attention/self/Reshape_2/shapePack!cond/bert/encoder/strided_slice_2:cond/bert/encoder/layer_2/attention/self/Reshape_2/shape/1:cond/bert/encoder/layer_2/attention/self/Reshape_2/shape/2:cond/bert/encoder/layer_2/attention/self/Reshape_2/shape/3*
T0*

axis *
N
Æ
2cond/bert/encoder/layer_2/attention/self/Reshape_2Reshape6cond/bert/encoder/layer_2/attention/self/value/BiasAdd8cond/bert/encoder/layer_2/attention/self/Reshape_2/shape*
T0*
Tshape0

9cond/bert/encoder/layer_2/attention/self/transpose_2/permConst^cond/switch_t*%
valueB"             *
dtype0
Æ
4cond/bert/encoder/layer_2/attention/self/transpose_2	Transpose2cond/bert/encoder/layer_2/attention/self/Reshape_29cond/bert/encoder/layer_2/attention/self/transpose_2/perm*
Tperm0*
T0
Ī
1cond/bert/encoder/layer_2/attention/self/MatMul_1BatchMatMul4cond/bert/encoder/layer_2/attention/self/dropout/mul4cond/bert/encoder/layer_2/attention/self/transpose_2*
adj_x( *
adj_y( *
T0

9cond/bert/encoder/layer_2/attention/self/transpose_3/permConst^cond/switch_t*%
valueB"             *
dtype0
Å
4cond/bert/encoder/layer_2/attention/self/transpose_3	Transpose1cond/bert/encoder/layer_2/attention/self/MatMul_19cond/bert/encoder/layer_2/attention/self/transpose_3/perm*
Tperm0*
T0
j
0cond/bert/encoder/layer_2/attention/self/mul_2/yConst^cond/switch_t*
value	B :*
dtype0

.cond/bert/encoder/layer_2/attention/self/mul_2Mul!cond/bert/encoder/strided_slice_20cond/bert/encoder/layer_2/attention/self/mul_2/y*
T0
u
:cond/bert/encoder/layer_2/attention/self/Reshape_3/shape/1Const^cond/switch_t*
value
B :*
dtype0
Ę
8cond/bert/encoder/layer_2/attention/self/Reshape_3/shapePack.cond/bert/encoder/layer_2/attention/self/mul_2:cond/bert/encoder/layer_2/attention/self/Reshape_3/shape/1*
T0*

axis *
N
Ä
2cond/bert/encoder/layer_2/attention/self/Reshape_3Reshape4cond/bert/encoder/layer_2/attention/self/transpose_38cond/bert/encoder/layer_2/attention/self/Reshape_3/shape*
Tshape0*
T0
Ö
Hmio_variable/bert/encoder/layer_2/attention/output/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*A
	container42bert/encoder/layer_2/attention/output/dense/kernel
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
/Initializer_43/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_43/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

#Initializer_43/truncated_normal/mulMul/Initializer_43/truncated_normal/TruncatedNormal&Initializer_43/truncated_normal/stddev*
T0
z
Initializer_43/truncated_normalAdd#Initializer_43/truncated_normal/mul$Initializer_43/truncated_normal/mean*
T0

	Assign_43AssignHmio_variable/bert/encoder/layer_2/attention/output/dense/kernel/gradientInitializer_43/truncated_normal*
use_locking(*
T0*[
_classQ
OMloc:@mio_variable/bert/encoder/layer_2/attention/output/dense/kernel/gradient*
validate_shape(
Í
Fmio_variable/bert/encoder/layer_2/attention/output/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_2/attention/output/dense/bias*
shape:
Í
Fmio_variable/bert/encoder/layer_2/attention/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_2/attention/output/dense/bias*
shape:
F
Initializer_44/zerosConst*
valueB*    *
dtype0
ū
	Assign_44AssignFmio_variable/bert/encoder/layer_2/attention/output/dense/bias/gradientInitializer_44/zeros*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_2/attention/output/dense/bias/gradient*
validate_shape(*
use_locking(*
T0
æ
7cond/bert/encoder/layer_2/attention/output/dense/MatMulMatMul2cond/bert/encoder/layer_2/attention/self/Reshape_3@cond/bert/encoder/layer_2/attention/output/dense/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 

>cond/bert/encoder/layer_2/attention/output/dense/MatMul/SwitchSwitchHmio_variable/bert/encoder/layer_2/attention/output/dense/kernel/variablecond/pred_id*
T0*[
_classQ
OMloc:@mio_variable/bert/encoder/layer_2/attention/output/dense/kernel/variable
ß
8cond/bert/encoder/layer_2/attention/output/dense/BiasAddBiasAdd7cond/bert/encoder/layer_2/attention/output/dense/MatMulAcond/bert/encoder/layer_2/attention/output/dense/BiasAdd/Switch:1*
T0*
data_formatNHWC

?cond/bert/encoder/layer_2/attention/output/dense/BiasAdd/SwitchSwitchFmio_variable/bert/encoder/layer_2/attention/output/dense/bias/variablecond/pred_id*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_2/attention/output/dense/bias/variable
y
<cond/bert/encoder/layer_2/attention/output/dropout/keep_probConst^cond/switch_t*
valueB
 *fff?*
dtype0

8cond/bert/encoder/layer_2/attention/output/dropout/ShapeShape8cond/bert/encoder/layer_2/attention/output/dense/BiasAdd*
T0*
out_type0

Econd/bert/encoder/layer_2/attention/output/dropout/random_uniform/minConst^cond/switch_t*
dtype0*
valueB
 *    

Econd/bert/encoder/layer_2/attention/output/dropout/random_uniform/maxConst^cond/switch_t*
valueB
 *  ?*
dtype0
É
Ocond/bert/encoder/layer_2/attention/output/dropout/random_uniform/RandomUniformRandomUniform8cond/bert/encoder/layer_2/attention/output/dropout/Shape*
T0*
dtype0*
seed2 *

seed 
ã
Econd/bert/encoder/layer_2/attention/output/dropout/random_uniform/subSubEcond/bert/encoder/layer_2/attention/output/dropout/random_uniform/maxEcond/bert/encoder/layer_2/attention/output/dropout/random_uniform/min*
T0
í
Econd/bert/encoder/layer_2/attention/output/dropout/random_uniform/mulMulOcond/bert/encoder/layer_2/attention/output/dropout/random_uniform/RandomUniformEcond/bert/encoder/layer_2/attention/output/dropout/random_uniform/sub*
T0
ß
Acond/bert/encoder/layer_2/attention/output/dropout/random_uniformAddEcond/bert/encoder/layer_2/attention/output/dropout/random_uniform/mulEcond/bert/encoder/layer_2/attention/output/dropout/random_uniform/min*
T0
Į
6cond/bert/encoder/layer_2/attention/output/dropout/addAdd<cond/bert/encoder/layer_2/attention/output/dropout/keep_probAcond/bert/encoder/layer_2/attention/output/dropout/random_uniform*
T0

8cond/bert/encoder/layer_2/attention/output/dropout/FloorFloor6cond/bert/encoder/layer_2/attention/output/dropout/add*
T0
Â
6cond/bert/encoder/layer_2/attention/output/dropout/divRealDiv8cond/bert/encoder/layer_2/attention/output/dense/BiasAdd<cond/bert/encoder/layer_2/attention/output/dropout/keep_prob*
T0
¸
6cond/bert/encoder/layer_2/attention/output/dropout/mulMul6cond/bert/encoder/layer_2/attention/output/dropout/div8cond/bert/encoder/layer_2/attention/output/dropout/Floor*
T0
˛
.cond/bert/encoder/layer_2/attention/output/addAdd6cond/bert/encoder/layer_2/attention/output/dropout/mul:cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1*
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
	Assign_45AssignJmio_variable/bert/encoder/layer_2/attention/output/LayerNorm/beta/gradientInitializer_45/zeros*
T0*]
_classS
QOloc:@mio_variable/bert/encoder/layer_2/attention/output/LayerNorm/beta/gradient*
validate_shape(*
use_locking(
×
Kmio_variable/bert/encoder/layer_2/attention/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*D
	container75bert/encoder/layer_2/attention/output/LayerNorm/gamma
×
Kmio_variable/bert/encoder/layer_2/attention/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*D
	container75bert/encoder/layer_2/attention/output/LayerNorm/gamma*
shape:
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

Scond/bert/encoder/layer_2/attention/output/LayerNorm/moments/mean/reduction_indicesConst^cond/switch_t*
dtype0*
valueB:
ô
Acond/bert/encoder/layer_2/attention/output/LayerNorm/moments/meanMean.cond/bert/encoder/layer_2/attention/output/addScond/bert/encoder/layer_2/attention/output/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
Ĩ
Icond/bert/encoder/layer_2/attention/output/LayerNorm/moments/StopGradientStopGradientAcond/bert/encoder/layer_2/attention/output/LayerNorm/moments/mean*
T0
į
Ncond/bert/encoder/layer_2/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference.cond/bert/encoder/layer_2/attention/output/addIcond/bert/encoder/layer_2/attention/output/LayerNorm/moments/StopGradient*
T0

Wcond/bert/encoder/layer_2/attention/output/LayerNorm/moments/variance/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0

Econd/bert/encoder/layer_2/attention/output/LayerNorm/moments/varianceMeanNcond/bert/encoder/layer_2/attention/output/LayerNorm/moments/SquaredDifferenceWcond/bert/encoder/layer_2/attention/output/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0

Dcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add/yConst^cond/switch_t*
valueB
 *Ėŧ+*
dtype0
ß
Bcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/addAddEcond/bert/encoder/layer_2/attention/output/LayerNorm/moments/varianceDcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add/y*
T0

Dcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/RsqrtRsqrtBcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add*
T0
å
Bcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mulMulDcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/RsqrtKcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul/Switch:1*
T0

Icond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul/SwitchSwitchKmio_variable/bert/encoder/layer_2/attention/output/LayerNorm/gamma/variablecond/pred_id*
T0*^
_classT
RPloc:@mio_variable/bert/encoder/layer_2/attention/output/LayerNorm/gamma/variable
Č
Dcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul_1Mul.cond/bert/encoder/layer_2/attention/output/addBcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul*
T0
Û
Dcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul_2MulAcond/bert/encoder/layer_2/attention/output/LayerNorm/moments/meanBcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul*
T0
å
Bcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/subSubKcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/sub/Switch:1Dcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul_2*
T0

Icond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/sub/SwitchSwitchJmio_variable/bert/encoder/layer_2/attention/output/LayerNorm/beta/variablecond/pred_id*
T0*]
_classS
QOloc:@mio_variable/bert/encoder/layer_2/attention/output/LayerNorm/beta/variable
Ū
Dcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add_1AddDcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul_1Bcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/sub*
T0
Î
Dmio_variable/bert/encoder/layer_2/intermediate/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_2/intermediate/dense/kernel*
shape:

Î
Dmio_variable/bert/encoder/layer_2/intermediate/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*=
	container0.bert/encoder/layer_2/intermediate/dense/kernel
Z
%Initializer_47/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_47/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_47/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_47/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_47/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0
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
Bmio_variable/bert/encoder/layer_2/intermediate/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_2/intermediate/dense/bias*
shape:
S
$Initializer_48/zeros/shape_as_tensorConst*
valueB:*
dtype0
G
Initializer_48/zeros/ConstConst*
valueB
 *    *
dtype0
y
Initializer_48/zerosFill$Initializer_48/zeros/shape_as_tensorInitializer_48/zeros/Const*
T0*

index_type0
ö
	Assign_48AssignBmio_variable/bert/encoder/layer_2/intermediate/dense/bias/gradientInitializer_48/zeros*
use_locking(*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_2/intermediate/dense/bias/gradient*
validate_shape(
đ
3cond/bert/encoder/layer_2/intermediate/dense/MatMulMatMulDcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add_1<cond/bert/encoder/layer_2/intermediate/dense/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 
ú
:cond/bert/encoder/layer_2/intermediate/dense/MatMul/SwitchSwitchDmio_variable/bert/encoder/layer_2/intermediate/dense/kernel/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_2/intermediate/dense/kernel/variable
Ķ
4cond/bert/encoder/layer_2/intermediate/dense/BiasAddBiasAdd3cond/bert/encoder/layer_2/intermediate/dense/MatMul=cond/bert/encoder/layer_2/intermediate/dense/BiasAdd/Switch:1*
data_formatNHWC*
T0
÷
;cond/bert/encoder/layer_2/intermediate/dense/BiasAdd/SwitchSwitchBmio_variable/bert/encoder/layer_2/intermediate/dense/bias/variablecond/pred_id*U
_classK
IGloc:@mio_variable/bert/encoder/layer_2/intermediate/dense/bias/variable*
T0
o
2cond/bert/encoder/layer_2/intermediate/dense/Pow/yConst^cond/switch_t*
valueB
 *  @@*
dtype0
Ē
0cond/bert/encoder/layer_2/intermediate/dense/PowPow4cond/bert/encoder/layer_2/intermediate/dense/BiasAdd2cond/bert/encoder/layer_2/intermediate/dense/Pow/y*
T0
o
2cond/bert/encoder/layer_2/intermediate/dense/mul/xConst^cond/switch_t*
valueB
 *'7=*
dtype0
Ļ
0cond/bert/encoder/layer_2/intermediate/dense/mulMul2cond/bert/encoder/layer_2/intermediate/dense/mul/x0cond/bert/encoder/layer_2/intermediate/dense/Pow*
T0
¨
0cond/bert/encoder/layer_2/intermediate/dense/addAdd4cond/bert/encoder/layer_2/intermediate/dense/BiasAdd0cond/bert/encoder/layer_2/intermediate/dense/mul*
T0
q
4cond/bert/encoder/layer_2/intermediate/dense/mul_1/xConst^cond/switch_t*
valueB
 **BL?*
dtype0
Ē
2cond/bert/encoder/layer_2/intermediate/dense/mul_1Mul4cond/bert/encoder/layer_2/intermediate/dense/mul_1/x0cond/bert/encoder/layer_2/intermediate/dense/add*
T0
v
1cond/bert/encoder/layer_2/intermediate/dense/TanhTanh2cond/bert/encoder/layer_2/intermediate/dense/mul_1*
T0
q
4cond/bert/encoder/layer_2/intermediate/dense/add_1/xConst^cond/switch_t*
valueB
 *  ?*
dtype0
Ģ
2cond/bert/encoder/layer_2/intermediate/dense/add_1Add4cond/bert/encoder/layer_2/intermediate/dense/add_1/x1cond/bert/encoder/layer_2/intermediate/dense/Tanh*
T0
q
4cond/bert/encoder/layer_2/intermediate/dense/mul_2/xConst^cond/switch_t*
valueB
 *   ?*
dtype0
Ŧ
2cond/bert/encoder/layer_2/intermediate/dense/mul_2Mul4cond/bert/encoder/layer_2/intermediate/dense/mul_2/x2cond/bert/encoder/layer_2/intermediate/dense/add_1*
T0
Ŧ
2cond/bert/encoder/layer_2/intermediate/dense/mul_3Mul4cond/bert/encoder/layer_2/intermediate/dense/BiasAdd2cond/bert/encoder/layer_2/intermediate/dense/mul_2*
T0
Â
>mio_variable/bert/encoder/layer_2/output/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*7
	container*(bert/encoder/layer_2/output/dense/kernel
Â
>mio_variable/bert/encoder/layer_2/output/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(bert/encoder/layer_2/output/dense/kernel*
shape:

Z
%Initializer_49/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_49/truncated_normal/meanConst*
valueB
 *    *
dtype0
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
	Assign_49Assign>mio_variable/bert/encoder/layer_2/output/dense/kernel/gradientInitializer_49/truncated_normal*
T0*Q
_classG
ECloc:@mio_variable/bert/encoder/layer_2/output/dense/kernel/gradient*
validate_shape(*
use_locking(
š
<mio_variable/bert/encoder/layer_2/output/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&bert/encoder/layer_2/output/dense/bias*
shape:
š
<mio_variable/bert/encoder/layer_2/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&bert/encoder/layer_2/output/dense/bias*
shape:
F
Initializer_50/zerosConst*
valueB*    *
dtype0
ę
	Assign_50Assign<mio_variable/bert/encoder/layer_2/output/dense/bias/gradientInitializer_50/zeros*
use_locking(*
T0*O
_classE
CAloc:@mio_variable/bert/encoder/layer_2/output/dense/bias/gradient*
validate_shape(
Ō
-cond/bert/encoder/layer_2/output/dense/MatMulMatMul2cond/bert/encoder/layer_2/intermediate/dense/mul_36cond/bert/encoder/layer_2/output/dense/MatMul/Switch:1*
transpose_a( *
transpose_b( *
T0
č
4cond/bert/encoder/layer_2/output/dense/MatMul/SwitchSwitch>mio_variable/bert/encoder/layer_2/output/dense/kernel/variablecond/pred_id*
T0*Q
_classG
ECloc:@mio_variable/bert/encoder/layer_2/output/dense/kernel/variable
Á
.cond/bert/encoder/layer_2/output/dense/BiasAddBiasAdd-cond/bert/encoder/layer_2/output/dense/MatMul7cond/bert/encoder/layer_2/output/dense/BiasAdd/Switch:1*
T0*
data_formatNHWC
å
5cond/bert/encoder/layer_2/output/dense/BiasAdd/SwitchSwitch<mio_variable/bert/encoder/layer_2/output/dense/bias/variablecond/pred_id*
T0*O
_classE
CAloc:@mio_variable/bert/encoder/layer_2/output/dense/bias/variable
o
2cond/bert/encoder/layer_2/output/dropout/keep_probConst^cond/switch_t*
valueB
 *fff?*
dtype0

.cond/bert/encoder/layer_2/output/dropout/ShapeShape.cond/bert/encoder/layer_2/output/dense/BiasAdd*
T0*
out_type0
x
;cond/bert/encoder/layer_2/output/dropout/random_uniform/minConst^cond/switch_t*
valueB
 *    *
dtype0
x
;cond/bert/encoder/layer_2/output/dropout/random_uniform/maxConst^cond/switch_t*
valueB
 *  ?*
dtype0
ĩ
Econd/bert/encoder/layer_2/output/dropout/random_uniform/RandomUniformRandomUniform.cond/bert/encoder/layer_2/output/dropout/Shape*
T0*
dtype0*
seed2 *

seed 
Å
;cond/bert/encoder/layer_2/output/dropout/random_uniform/subSub;cond/bert/encoder/layer_2/output/dropout/random_uniform/max;cond/bert/encoder/layer_2/output/dropout/random_uniform/min*
T0
Ī
;cond/bert/encoder/layer_2/output/dropout/random_uniform/mulMulEcond/bert/encoder/layer_2/output/dropout/random_uniform/RandomUniform;cond/bert/encoder/layer_2/output/dropout/random_uniform/sub*
T0
Á
7cond/bert/encoder/layer_2/output/dropout/random_uniformAdd;cond/bert/encoder/layer_2/output/dropout/random_uniform/mul;cond/bert/encoder/layer_2/output/dropout/random_uniform/min*
T0
Š
,cond/bert/encoder/layer_2/output/dropout/addAdd2cond/bert/encoder/layer_2/output/dropout/keep_prob7cond/bert/encoder/layer_2/output/dropout/random_uniform*
T0
n
.cond/bert/encoder/layer_2/output/dropout/FloorFloor,cond/bert/encoder/layer_2/output/dropout/add*
T0
¤
,cond/bert/encoder/layer_2/output/dropout/divRealDiv.cond/bert/encoder/layer_2/output/dense/BiasAdd2cond/bert/encoder/layer_2/output/dropout/keep_prob*
T0

,cond/bert/encoder/layer_2/output/dropout/mulMul,cond/bert/encoder/layer_2/output/dropout/div.cond/bert/encoder/layer_2/output/dropout/Floor*
T0
¨
$cond/bert/encoder/layer_2/output/addAdd,cond/bert/encoder/layer_2/output/dropout/mulDcond/bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add_1*
T0
Á
@mio_variable/bert/encoder/layer_2/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*bert/encoder/layer_2/output/LayerNorm/beta*
shape:
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
Amio_variable/bert/encoder/layer_2/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+bert/encoder/layer_2/output/LayerNorm/gamma*
shape:
E
Initializer_52/onesConst*
valueB*  ?*
dtype0
ķ
	Assign_52AssignAmio_variable/bert/encoder/layer_2/output/LayerNorm/gamma/gradientInitializer_52/ones*
use_locking(*
T0*T
_classJ
HFloc:@mio_variable/bert/encoder/layer_2/output/LayerNorm/gamma/gradient*
validate_shape(

Icond/bert/encoder/layer_2/output/LayerNorm/moments/mean/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0
Ö
7cond/bert/encoder/layer_2/output/LayerNorm/moments/meanMean$cond/bert/encoder/layer_2/output/addIcond/bert/encoder/layer_2/output/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0

?cond/bert/encoder/layer_2/output/LayerNorm/moments/StopGradientStopGradient7cond/bert/encoder/layer_2/output/LayerNorm/moments/mean*
T0
É
Dcond/bert/encoder/layer_2/output/LayerNorm/moments/SquaredDifferenceSquaredDifference$cond/bert/encoder/layer_2/output/add?cond/bert/encoder/layer_2/output/LayerNorm/moments/StopGradient*
T0

Mcond/bert/encoder/layer_2/output/LayerNorm/moments/variance/reduction_indicesConst^cond/switch_t*
dtype0*
valueB:
ū
;cond/bert/encoder/layer_2/output/LayerNorm/moments/varianceMeanDcond/bert/encoder/layer_2/output/LayerNorm/moments/SquaredDifferenceMcond/bert/encoder/layer_2/output/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0
w
:cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/add/yConst^cond/switch_t*
valueB
 *Ėŧ+*
dtype0
Á
8cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/addAdd;cond/bert/encoder/layer_2/output/LayerNorm/moments/variance:cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/add/y*
T0

:cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/RsqrtRsqrt8cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/add*
T0
Į
8cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/mulMul:cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/RsqrtAcond/bert/encoder/layer_2/output/LayerNorm/batchnorm/mul/Switch:1*
T0
ų
?cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/mul/SwitchSwitchAmio_variable/bert/encoder/layer_2/output/LayerNorm/gamma/variablecond/pred_id*
T0*T
_classJ
HFloc:@mio_variable/bert/encoder/layer_2/output/LayerNorm/gamma/variable
Ē
:cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/mul_1Mul$cond/bert/encoder/layer_2/output/add8cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/mul*
T0
Ŋ
:cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/mul_2Mul7cond/bert/encoder/layer_2/output/LayerNorm/moments/mean8cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/mul*
T0
Į
8cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/subSubAcond/bert/encoder/layer_2/output/LayerNorm/batchnorm/sub/Switch:1:cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/mul_2*
T0
÷
?cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/sub/SwitchSwitch@mio_variable/bert/encoder/layer_2/output/LayerNorm/beta/variablecond/pred_id*
T0*S
_classI
GEloc:@mio_variable/bert/encoder/layer_2/output/LayerNorm/beta/variable
Ā
:cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1Add:cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/mul_18cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/sub*
T0

.cond/bert/encoder/layer_3/attention/self/ShapeShape:cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
z
<cond/bert/encoder/layer_3/attention/self/strided_slice/stackConst^cond/switch_t*
valueB: *
dtype0
|
>cond/bert/encoder/layer_3/attention/self/strided_slice/stack_1Const^cond/switch_t*
dtype0*
valueB:
|
>cond/bert/encoder/layer_3/attention/self/strided_slice/stack_2Const^cond/switch_t*
valueB:*
dtype0
Ž
6cond/bert/encoder/layer_3/attention/self/strided_sliceStridedSlice.cond/bert/encoder/layer_3/attention/self/Shape<cond/bert/encoder/layer_3/attention/self/strided_slice/stack>cond/bert/encoder/layer_3/attention/self/strided_slice/stack_1>cond/bert/encoder/layer_3/attention/self/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0

0cond/bert/encoder/layer_3/attention/self/Shape_1Shape:cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
|
>cond/bert/encoder/layer_3/attention/self/strided_slice_1/stackConst^cond/switch_t*
valueB: *
dtype0
~
@cond/bert/encoder/layer_3/attention/self/strided_slice_1/stack_1Const^cond/switch_t*
valueB:*
dtype0
~
@cond/bert/encoder/layer_3/attention/self/strided_slice_1/stack_2Const^cond/switch_t*
valueB:*
dtype0
¸
8cond/bert/encoder/layer_3/attention/self/strided_slice_1StridedSlice0cond/bert/encoder/layer_3/attention/self/Shape_1>cond/bert/encoder/layer_3/attention/self/strided_slice_1/stack@cond/bert/encoder/layer_3/attention/self/strided_slice_1/stack_1@cond/bert/encoder/layer_3/attention/self/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
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
valueB"      *
dtype0
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
/Initializer_53/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_53/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

#Initializer_53/truncated_normal/mulMul/Initializer_53/truncated_normal/TruncatedNormal&Initializer_53/truncated_normal/stddev*
T0
z
Initializer_53/truncated_normalAdd#Initializer_53/truncated_normal/mul$Initializer_53/truncated_normal/mean*
T0

	Assign_53AssignFmio_variable/bert/encoder/layer_3/attention/self/query/kernel/gradientInitializer_53/truncated_normal*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_3/attention/self/query/kernel/gradient*
validate_shape(
É
Dmio_variable/bert/encoder/layer_3/attention/self/query/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*=
	container0.bert/encoder/layer_3/attention/self/query/bias
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
ę
5cond/bert/encoder/layer_3/attention/self/query/MatMulMatMul:cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1>cond/bert/encoder/layer_3/attention/self/query/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 

<cond/bert/encoder/layer_3/attention/self/query/MatMul/SwitchSwitchFmio_variable/bert/encoder/layer_3/attention/self/query/kernel/variablecond/pred_id*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_3/attention/self/query/kernel/variable
Ų
6cond/bert/encoder/layer_3/attention/self/query/BiasAddBiasAdd5cond/bert/encoder/layer_3/attention/self/query/MatMul?cond/bert/encoder/layer_3/attention/self/query/BiasAdd/Switch:1*
T0*
data_formatNHWC
ũ
=cond/bert/encoder/layer_3/attention/self/query/BiasAdd/SwitchSwitchDmio_variable/bert/encoder/layer_3/attention/self/query/bias/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_3/attention/self/query/bias/variable
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
&Initializer_55/truncated_normal/stddevConst*
dtype0*
valueB
 *
×Ŗ<

/Initializer_55/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_55/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

#Initializer_55/truncated_normal/mulMul/Initializer_55/truncated_normal/TruncatedNormal&Initializer_55/truncated_normal/stddev*
T0
z
Initializer_55/truncated_normalAdd#Initializer_55/truncated_normal/mul$Initializer_55/truncated_normal/mean*
T0

	Assign_55AssignDmio_variable/bert/encoder/layer_3/attention/self/key/kernel/gradientInitializer_55/truncated_normal*
validate_shape(*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_3/attention/self/key/kernel/gradient
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
	Assign_56AssignBmio_variable/bert/encoder/layer_3/attention/self/key/bias/gradientInitializer_56/zeros*
use_locking(*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_3/attention/self/key/bias/gradient*
validate_shape(
æ
3cond/bert/encoder/layer_3/attention/self/key/MatMulMatMul:cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1<cond/bert/encoder/layer_3/attention/self/key/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 
ú
:cond/bert/encoder/layer_3/attention/self/key/MatMul/SwitchSwitchDmio_variable/bert/encoder/layer_3/attention/self/key/kernel/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_3/attention/self/key/kernel/variable
Ķ
4cond/bert/encoder/layer_3/attention/self/key/BiasAddBiasAdd3cond/bert/encoder/layer_3/attention/self/key/MatMul=cond/bert/encoder/layer_3/attention/self/key/BiasAdd/Switch:1*
data_formatNHWC*
T0
÷
;cond/bert/encoder/layer_3/attention/self/key/BiasAdd/SwitchSwitchBmio_variable/bert/encoder/layer_3/attention/self/key/bias/variablecond/pred_id*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_3/attention/self/key/bias/variable
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
seed2 *

seed *
T0*
dtype0

#Initializer_57/truncated_normal/mulMul/Initializer_57/truncated_normal/TruncatedNormal&Initializer_57/truncated_normal/stddev*
T0
z
Initializer_57/truncated_normalAdd#Initializer_57/truncated_normal/mul$Initializer_57/truncated_normal/mean*
T0

	Assign_57AssignFmio_variable/bert/encoder/layer_3/attention/self/value/kernel/gradientInitializer_57/truncated_normal*
validate_shape(*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_3/attention/self/value/kernel/gradient
É
Dmio_variable/bert/encoder/layer_3/attention/self/value/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*=
	container0.bert/encoder/layer_3/attention/self/value/bias
É
Dmio_variable/bert/encoder/layer_3/attention/self/value/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*=
	container0.bert/encoder/layer_3/attention/self/value/bias
F
Initializer_58/zerosConst*
valueB*    *
dtype0
ú
	Assign_58AssignDmio_variable/bert/encoder/layer_3/attention/self/value/bias/gradientInitializer_58/zeros*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_3/attention/self/value/bias/gradient*
validate_shape(
ę
5cond/bert/encoder/layer_3/attention/self/value/MatMulMatMul:cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1>cond/bert/encoder/layer_3/attention/self/value/MatMul/Switch:1*
transpose_b( *
T0*
transpose_a( 

<cond/bert/encoder/layer_3/attention/self/value/MatMul/SwitchSwitchFmio_variable/bert/encoder/layer_3/attention/self/value/kernel/variablecond/pred_id*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_3/attention/self/value/kernel/variable
Ų
6cond/bert/encoder/layer_3/attention/self/value/BiasAddBiasAdd5cond/bert/encoder/layer_3/attention/self/value/MatMul?cond/bert/encoder/layer_3/attention/self/value/BiasAdd/Switch:1*
T0*
data_formatNHWC
ũ
=cond/bert/encoder/layer_3/attention/self/value/BiasAdd/SwitchSwitchDmio_variable/bert/encoder/layer_3/attention/self/value/bias/variablecond/pred_id*W
_classM
KIloc:@mio_variable/bert/encoder/layer_3/attention/self/value/bias/variable*
T0
r
8cond/bert/encoder/layer_3/attention/self/Reshape/shape/1Const^cond/switch_t*
value	B :*
dtype0
r
8cond/bert/encoder/layer_3/attention/self/Reshape/shape/2Const^cond/switch_t*
value	B :*
dtype0
r
8cond/bert/encoder/layer_3/attention/self/Reshape/shape/3Const^cond/switch_t*
value	B : *
dtype0
­
6cond/bert/encoder/layer_3/attention/self/Reshape/shapePack!cond/bert/encoder/strided_slice_28cond/bert/encoder/layer_3/attention/self/Reshape/shape/18cond/bert/encoder/layer_3/attention/self/Reshape/shape/28cond/bert/encoder/layer_3/attention/self/Reshape/shape/3*
T0*

axis *
N
Â
0cond/bert/encoder/layer_3/attention/self/ReshapeReshape6cond/bert/encoder/layer_3/attention/self/query/BiasAdd6cond/bert/encoder/layer_3/attention/self/Reshape/shape*
Tshape0*
T0

7cond/bert/encoder/layer_3/attention/self/transpose/permConst^cond/switch_t*%
valueB"             *
dtype0
Ā
2cond/bert/encoder/layer_3/attention/self/transpose	Transpose0cond/bert/encoder/layer_3/attention/self/Reshape7cond/bert/encoder/layer_3/attention/self/transpose/perm*
T0*
Tperm0
t
:cond/bert/encoder/layer_3/attention/self/Reshape_1/shape/1Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_3/attention/self/Reshape_1/shape/2Const^cond/switch_t*
dtype0*
value	B :
t
:cond/bert/encoder/layer_3/attention/self/Reshape_1/shape/3Const^cond/switch_t*
value	B : *
dtype0
ĩ
8cond/bert/encoder/layer_3/attention/self/Reshape_1/shapePack!cond/bert/encoder/strided_slice_2:cond/bert/encoder/layer_3/attention/self/Reshape_1/shape/1:cond/bert/encoder/layer_3/attention/self/Reshape_1/shape/2:cond/bert/encoder/layer_3/attention/self/Reshape_1/shape/3*
T0*

axis *
N
Ä
2cond/bert/encoder/layer_3/attention/self/Reshape_1Reshape4cond/bert/encoder/layer_3/attention/self/key/BiasAdd8cond/bert/encoder/layer_3/attention/self/Reshape_1/shape*
T0*
Tshape0

9cond/bert/encoder/layer_3/attention/self/transpose_1/permConst^cond/switch_t*%
valueB"             *
dtype0
Æ
4cond/bert/encoder/layer_3/attention/self/transpose_1	Transpose2cond/bert/encoder/layer_3/attention/self/Reshape_19cond/bert/encoder/layer_3/attention/self/transpose_1/perm*
T0*
Tperm0
Ë
/cond/bert/encoder/layer_3/attention/self/MatMulBatchMatMul2cond/bert/encoder/layer_3/attention/self/transpose4cond/bert/encoder/layer_3/attention/self/transpose_1*
adj_x( *
adj_y(*
T0
k
.cond/bert/encoder/layer_3/attention/self/Mul/yConst^cond/switch_t*
valueB
 *ķ5>*
dtype0

,cond/bert/encoder/layer_3/attention/self/MulMul/cond/bert/encoder/layer_3/attention/self/MatMul.cond/bert/encoder/layer_3/attention/self/Mul/y*
T0
u
7cond/bert/encoder/layer_3/attention/self/ExpandDims/dimConst^cond/switch_t*
valueB:*
dtype0
Ļ
3cond/bert/encoder/layer_3/attention/self/ExpandDims
ExpandDimscond/bert/encoder/mul7cond/bert/encoder/layer_3/attention/self/ExpandDims/dim*
T0*

Tdim0
k
.cond/bert/encoder/layer_3/attention/self/sub/xConst^cond/switch_t*
valueB
 *  ?*
dtype0
Ą
,cond/bert/encoder/layer_3/attention/self/subSub.cond/bert/encoder/layer_3/attention/self/sub/x3cond/bert/encoder/layer_3/attention/self/ExpandDims*
T0
m
0cond/bert/encoder/layer_3/attention/self/mul_1/yConst^cond/switch_t*
valueB
 * @Æ*
dtype0

.cond/bert/encoder/layer_3/attention/self/mul_1Mul,cond/bert/encoder/layer_3/attention/self/sub0cond/bert/encoder/layer_3/attention/self/mul_1/y*
T0

,cond/bert/encoder/layer_3/attention/self/addAdd,cond/bert/encoder/layer_3/attention/self/Mul.cond/bert/encoder/layer_3/attention/self/mul_1*
T0
r
0cond/bert/encoder/layer_3/attention/self/SoftmaxSoftmax,cond/bert/encoder/layer_3/attention/self/add*
T0
w
:cond/bert/encoder/layer_3/attention/self/dropout/keep_probConst^cond/switch_t*
dtype0*
valueB
 *fff?

6cond/bert/encoder/layer_3/attention/self/dropout/ShapeShape0cond/bert/encoder/layer_3/attention/self/Softmax*
T0*
out_type0

Ccond/bert/encoder/layer_3/attention/self/dropout/random_uniform/minConst^cond/switch_t*
valueB
 *    *
dtype0

Ccond/bert/encoder/layer_3/attention/self/dropout/random_uniform/maxConst^cond/switch_t*
valueB
 *  ?*
dtype0
Å
Mcond/bert/encoder/layer_3/attention/self/dropout/random_uniform/RandomUniformRandomUniform6cond/bert/encoder/layer_3/attention/self/dropout/Shape*
dtype0*
seed2 *

seed *
T0
Ũ
Ccond/bert/encoder/layer_3/attention/self/dropout/random_uniform/subSubCcond/bert/encoder/layer_3/attention/self/dropout/random_uniform/maxCcond/bert/encoder/layer_3/attention/self/dropout/random_uniform/min*
T0
į
Ccond/bert/encoder/layer_3/attention/self/dropout/random_uniform/mulMulMcond/bert/encoder/layer_3/attention/self/dropout/random_uniform/RandomUniformCcond/bert/encoder/layer_3/attention/self/dropout/random_uniform/sub*
T0
Ų
?cond/bert/encoder/layer_3/attention/self/dropout/random_uniformAddCcond/bert/encoder/layer_3/attention/self/dropout/random_uniform/mulCcond/bert/encoder/layer_3/attention/self/dropout/random_uniform/min*
T0
Á
4cond/bert/encoder/layer_3/attention/self/dropout/addAdd:cond/bert/encoder/layer_3/attention/self/dropout/keep_prob?cond/bert/encoder/layer_3/attention/self/dropout/random_uniform*
T0
~
6cond/bert/encoder/layer_3/attention/self/dropout/FloorFloor4cond/bert/encoder/layer_3/attention/self/dropout/add*
T0
ļ
4cond/bert/encoder/layer_3/attention/self/dropout/divRealDiv0cond/bert/encoder/layer_3/attention/self/Softmax:cond/bert/encoder/layer_3/attention/self/dropout/keep_prob*
T0
˛
4cond/bert/encoder/layer_3/attention/self/dropout/mulMul4cond/bert/encoder/layer_3/attention/self/dropout/div6cond/bert/encoder/layer_3/attention/self/dropout/Floor*
T0
t
:cond/bert/encoder/layer_3/attention/self/Reshape_2/shape/1Const^cond/switch_t*
dtype0*
value	B :
t
:cond/bert/encoder/layer_3/attention/self/Reshape_2/shape/2Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_3/attention/self/Reshape_2/shape/3Const^cond/switch_t*
value	B : *
dtype0
ĩ
8cond/bert/encoder/layer_3/attention/self/Reshape_2/shapePack!cond/bert/encoder/strided_slice_2:cond/bert/encoder/layer_3/attention/self/Reshape_2/shape/1:cond/bert/encoder/layer_3/attention/self/Reshape_2/shape/2:cond/bert/encoder/layer_3/attention/self/Reshape_2/shape/3*
T0*

axis *
N
Æ
2cond/bert/encoder/layer_3/attention/self/Reshape_2Reshape6cond/bert/encoder/layer_3/attention/self/value/BiasAdd8cond/bert/encoder/layer_3/attention/self/Reshape_2/shape*
T0*
Tshape0

9cond/bert/encoder/layer_3/attention/self/transpose_2/permConst^cond/switch_t*%
valueB"             *
dtype0
Æ
4cond/bert/encoder/layer_3/attention/self/transpose_2	Transpose2cond/bert/encoder/layer_3/attention/self/Reshape_29cond/bert/encoder/layer_3/attention/self/transpose_2/perm*
T0*
Tperm0
Ī
1cond/bert/encoder/layer_3/attention/self/MatMul_1BatchMatMul4cond/bert/encoder/layer_3/attention/self/dropout/mul4cond/bert/encoder/layer_3/attention/self/transpose_2*
adj_x( *
adj_y( *
T0

9cond/bert/encoder/layer_3/attention/self/transpose_3/permConst^cond/switch_t*
dtype0*%
valueB"             
Å
4cond/bert/encoder/layer_3/attention/self/transpose_3	Transpose1cond/bert/encoder/layer_3/attention/self/MatMul_19cond/bert/encoder/layer_3/attention/self/transpose_3/perm*
Tperm0*
T0
j
0cond/bert/encoder/layer_3/attention/self/mul_2/yConst^cond/switch_t*
value	B :*
dtype0

.cond/bert/encoder/layer_3/attention/self/mul_2Mul!cond/bert/encoder/strided_slice_20cond/bert/encoder/layer_3/attention/self/mul_2/y*
T0
u
:cond/bert/encoder/layer_3/attention/self/Reshape_3/shape/1Const^cond/switch_t*
dtype0*
value
B :
Ę
8cond/bert/encoder/layer_3/attention/self/Reshape_3/shapePack.cond/bert/encoder/layer_3/attention/self/mul_2:cond/bert/encoder/layer_3/attention/self/Reshape_3/shape/1*
T0*

axis *
N
Ä
2cond/bert/encoder/layer_3/attention/self/Reshape_3Reshape4cond/bert/encoder/layer_3/attention/self/transpose_38cond/bert/encoder/layer_3/attention/self/Reshape_3/shape*
T0*
Tshape0
Ö
Hmio_variable/bert/encoder/layer_3/attention/output/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42bert/encoder/layer_3/attention/output/dense/kernel*
shape:

Ö
Hmio_variable/bert/encoder/layer_3/attention/output/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*A
	container42bert/encoder/layer_3/attention/output/dense/kernel*
shape:

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
Fmio_variable/bert/encoder/layer_3/attention/output/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_3/attention/output/dense/bias*
shape:
Í
Fmio_variable/bert/encoder/layer_3/attention/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*?
	container20bert/encoder/layer_3/attention/output/dense/bias
F
Initializer_60/zerosConst*
valueB*    *
dtype0
ū
	Assign_60AssignFmio_variable/bert/encoder/layer_3/attention/output/dense/bias/gradientInitializer_60/zeros*
validate_shape(*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_3/attention/output/dense/bias/gradient
æ
7cond/bert/encoder/layer_3/attention/output/dense/MatMulMatMul2cond/bert/encoder/layer_3/attention/self/Reshape_3@cond/bert/encoder/layer_3/attention/output/dense/MatMul/Switch:1*
transpose_a( *
transpose_b( *
T0

>cond/bert/encoder/layer_3/attention/output/dense/MatMul/SwitchSwitchHmio_variable/bert/encoder/layer_3/attention/output/dense/kernel/variablecond/pred_id*
T0*[
_classQ
OMloc:@mio_variable/bert/encoder/layer_3/attention/output/dense/kernel/variable
ß
8cond/bert/encoder/layer_3/attention/output/dense/BiasAddBiasAdd7cond/bert/encoder/layer_3/attention/output/dense/MatMulAcond/bert/encoder/layer_3/attention/output/dense/BiasAdd/Switch:1*
T0*
data_formatNHWC

?cond/bert/encoder/layer_3/attention/output/dense/BiasAdd/SwitchSwitchFmio_variable/bert/encoder/layer_3/attention/output/dense/bias/variablecond/pred_id*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_3/attention/output/dense/bias/variable
y
<cond/bert/encoder/layer_3/attention/output/dropout/keep_probConst^cond/switch_t*
valueB
 *fff?*
dtype0

8cond/bert/encoder/layer_3/attention/output/dropout/ShapeShape8cond/bert/encoder/layer_3/attention/output/dense/BiasAdd*
T0*
out_type0

Econd/bert/encoder/layer_3/attention/output/dropout/random_uniform/minConst^cond/switch_t*
valueB
 *    *
dtype0

Econd/bert/encoder/layer_3/attention/output/dropout/random_uniform/maxConst^cond/switch_t*
valueB
 *  ?*
dtype0
É
Ocond/bert/encoder/layer_3/attention/output/dropout/random_uniform/RandomUniformRandomUniform8cond/bert/encoder/layer_3/attention/output/dropout/Shape*

seed *
T0*
dtype0*
seed2 
ã
Econd/bert/encoder/layer_3/attention/output/dropout/random_uniform/subSubEcond/bert/encoder/layer_3/attention/output/dropout/random_uniform/maxEcond/bert/encoder/layer_3/attention/output/dropout/random_uniform/min*
T0
í
Econd/bert/encoder/layer_3/attention/output/dropout/random_uniform/mulMulOcond/bert/encoder/layer_3/attention/output/dropout/random_uniform/RandomUniformEcond/bert/encoder/layer_3/attention/output/dropout/random_uniform/sub*
T0
ß
Acond/bert/encoder/layer_3/attention/output/dropout/random_uniformAddEcond/bert/encoder/layer_3/attention/output/dropout/random_uniform/mulEcond/bert/encoder/layer_3/attention/output/dropout/random_uniform/min*
T0
Į
6cond/bert/encoder/layer_3/attention/output/dropout/addAdd<cond/bert/encoder/layer_3/attention/output/dropout/keep_probAcond/bert/encoder/layer_3/attention/output/dropout/random_uniform*
T0

8cond/bert/encoder/layer_3/attention/output/dropout/FloorFloor6cond/bert/encoder/layer_3/attention/output/dropout/add*
T0
Â
6cond/bert/encoder/layer_3/attention/output/dropout/divRealDiv8cond/bert/encoder/layer_3/attention/output/dense/BiasAdd<cond/bert/encoder/layer_3/attention/output/dropout/keep_prob*
T0
¸
6cond/bert/encoder/layer_3/attention/output/dropout/mulMul6cond/bert/encoder/layer_3/attention/output/dropout/div8cond/bert/encoder/layer_3/attention/output/dropout/Floor*
T0
˛
.cond/bert/encoder/layer_3/attention/output/addAdd6cond/bert/encoder/layer_3/attention/output/dropout/mul:cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1*
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
	Assign_62AssignKmio_variable/bert/encoder/layer_3/attention/output/LayerNorm/gamma/gradientInitializer_62/ones*
validate_shape(*
use_locking(*
T0*^
_classT
RPloc:@mio_variable/bert/encoder/layer_3/attention/output/LayerNorm/gamma/gradient

Scond/bert/encoder/layer_3/attention/output/LayerNorm/moments/mean/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0
ô
Acond/bert/encoder/layer_3/attention/output/LayerNorm/moments/meanMean.cond/bert/encoder/layer_3/attention/output/addScond/bert/encoder/layer_3/attention/output/LayerNorm/moments/mean/reduction_indices*
T0*

Tidx0*
	keep_dims(
Ĩ
Icond/bert/encoder/layer_3/attention/output/LayerNorm/moments/StopGradientStopGradientAcond/bert/encoder/layer_3/attention/output/LayerNorm/moments/mean*
T0
į
Ncond/bert/encoder/layer_3/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference.cond/bert/encoder/layer_3/attention/output/addIcond/bert/encoder/layer_3/attention/output/LayerNorm/moments/StopGradient*
T0

Wcond/bert/encoder/layer_3/attention/output/LayerNorm/moments/variance/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0

Econd/bert/encoder/layer_3/attention/output/LayerNorm/moments/varianceMeanNcond/bert/encoder/layer_3/attention/output/LayerNorm/moments/SquaredDifferenceWcond/bert/encoder/layer_3/attention/output/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0

Dcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add/yConst^cond/switch_t*
valueB
 *Ėŧ+*
dtype0
ß
Bcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/addAddEcond/bert/encoder/layer_3/attention/output/LayerNorm/moments/varianceDcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add/y*
T0

Dcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/RsqrtRsqrtBcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add*
T0
å
Bcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mulMulDcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/RsqrtKcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul/Switch:1*
T0

Icond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul/SwitchSwitchKmio_variable/bert/encoder/layer_3/attention/output/LayerNorm/gamma/variablecond/pred_id*
T0*^
_classT
RPloc:@mio_variable/bert/encoder/layer_3/attention/output/LayerNorm/gamma/variable
Č
Dcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul_1Mul.cond/bert/encoder/layer_3/attention/output/addBcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul*
T0
Û
Dcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul_2MulAcond/bert/encoder/layer_3/attention/output/LayerNorm/moments/meanBcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul*
T0
å
Bcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/subSubKcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/sub/Switch:1Dcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul_2*
T0

Icond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/sub/SwitchSwitchJmio_variable/bert/encoder/layer_3/attention/output/LayerNorm/beta/variablecond/pred_id*
T0*]
_classS
QOloc:@mio_variable/bert/encoder/layer_3/attention/output/LayerNorm/beta/variable
Ū
Dcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add_1AddDcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul_1Bcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/sub*
T0
Î
Dmio_variable/bert/encoder/layer_3/intermediate/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_3/intermediate/dense/kernel*
shape:

Î
Dmio_variable/bert/encoder/layer_3/intermediate/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_3/intermediate/dense/kernel*
shape:

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
&Initializer_63/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_63/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_63/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 
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
Bmio_variable/bert/encoder/layer_3/intermediate/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_3/intermediate/dense/bias*
shape:
Å
Bmio_variable/bert/encoder/layer_3/intermediate/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*;
	container.,bert/encoder/layer_3/intermediate/dense/bias
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
	Assign_64AssignBmio_variable/bert/encoder/layer_3/intermediate/dense/bias/gradientInitializer_64/zeros*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_3/intermediate/dense/bias/gradient*
validate_shape(*
use_locking(
đ
3cond/bert/encoder/layer_3/intermediate/dense/MatMulMatMulDcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add_1<cond/bert/encoder/layer_3/intermediate/dense/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 
ú
:cond/bert/encoder/layer_3/intermediate/dense/MatMul/SwitchSwitchDmio_variable/bert/encoder/layer_3/intermediate/dense/kernel/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_3/intermediate/dense/kernel/variable
Ķ
4cond/bert/encoder/layer_3/intermediate/dense/BiasAddBiasAdd3cond/bert/encoder/layer_3/intermediate/dense/MatMul=cond/bert/encoder/layer_3/intermediate/dense/BiasAdd/Switch:1*
T0*
data_formatNHWC
÷
;cond/bert/encoder/layer_3/intermediate/dense/BiasAdd/SwitchSwitchBmio_variable/bert/encoder/layer_3/intermediate/dense/bias/variablecond/pred_id*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_3/intermediate/dense/bias/variable
o
2cond/bert/encoder/layer_3/intermediate/dense/Pow/yConst^cond/switch_t*
valueB
 *  @@*
dtype0
Ē
0cond/bert/encoder/layer_3/intermediate/dense/PowPow4cond/bert/encoder/layer_3/intermediate/dense/BiasAdd2cond/bert/encoder/layer_3/intermediate/dense/Pow/y*
T0
o
2cond/bert/encoder/layer_3/intermediate/dense/mul/xConst^cond/switch_t*
valueB
 *'7=*
dtype0
Ļ
0cond/bert/encoder/layer_3/intermediate/dense/mulMul2cond/bert/encoder/layer_3/intermediate/dense/mul/x0cond/bert/encoder/layer_3/intermediate/dense/Pow*
T0
¨
0cond/bert/encoder/layer_3/intermediate/dense/addAdd4cond/bert/encoder/layer_3/intermediate/dense/BiasAdd0cond/bert/encoder/layer_3/intermediate/dense/mul*
T0
q
4cond/bert/encoder/layer_3/intermediate/dense/mul_1/xConst^cond/switch_t*
valueB
 **BL?*
dtype0
Ē
2cond/bert/encoder/layer_3/intermediate/dense/mul_1Mul4cond/bert/encoder/layer_3/intermediate/dense/mul_1/x0cond/bert/encoder/layer_3/intermediate/dense/add*
T0
v
1cond/bert/encoder/layer_3/intermediate/dense/TanhTanh2cond/bert/encoder/layer_3/intermediate/dense/mul_1*
T0
q
4cond/bert/encoder/layer_3/intermediate/dense/add_1/xConst^cond/switch_t*
valueB
 *  ?*
dtype0
Ģ
2cond/bert/encoder/layer_3/intermediate/dense/add_1Add4cond/bert/encoder/layer_3/intermediate/dense/add_1/x1cond/bert/encoder/layer_3/intermediate/dense/Tanh*
T0
q
4cond/bert/encoder/layer_3/intermediate/dense/mul_2/xConst^cond/switch_t*
valueB
 *   ?*
dtype0
Ŧ
2cond/bert/encoder/layer_3/intermediate/dense/mul_2Mul4cond/bert/encoder/layer_3/intermediate/dense/mul_2/x2cond/bert/encoder/layer_3/intermediate/dense/add_1*
T0
Ŧ
2cond/bert/encoder/layer_3/intermediate/dense/mul_3Mul4cond/bert/encoder/layer_3/intermediate/dense/BiasAdd2cond/bert/encoder/layer_3/intermediate/dense/mul_2*
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
&Initializer_65/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_65/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_65/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 

#Initializer_65/truncated_normal/mulMul/Initializer_65/truncated_normal/TruncatedNormal&Initializer_65/truncated_normal/stddev*
T0
z
Initializer_65/truncated_normalAdd#Initializer_65/truncated_normal/mul$Initializer_65/truncated_normal/mean*
T0
ų
	Assign_65Assign>mio_variable/bert/encoder/layer_3/output/dense/kernel/gradientInitializer_65/truncated_normal*
T0*Q
_classG
ECloc:@mio_variable/bert/encoder/layer_3/output/dense/kernel/gradient*
validate_shape(*
use_locking(
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
Ō
-cond/bert/encoder/layer_3/output/dense/MatMulMatMul2cond/bert/encoder/layer_3/intermediate/dense/mul_36cond/bert/encoder/layer_3/output/dense/MatMul/Switch:1*
transpose_a( *
transpose_b( *
T0
č
4cond/bert/encoder/layer_3/output/dense/MatMul/SwitchSwitch>mio_variable/bert/encoder/layer_3/output/dense/kernel/variablecond/pred_id*
T0*Q
_classG
ECloc:@mio_variable/bert/encoder/layer_3/output/dense/kernel/variable
Á
.cond/bert/encoder/layer_3/output/dense/BiasAddBiasAdd-cond/bert/encoder/layer_3/output/dense/MatMul7cond/bert/encoder/layer_3/output/dense/BiasAdd/Switch:1*
T0*
data_formatNHWC
å
5cond/bert/encoder/layer_3/output/dense/BiasAdd/SwitchSwitch<mio_variable/bert/encoder/layer_3/output/dense/bias/variablecond/pred_id*
T0*O
_classE
CAloc:@mio_variable/bert/encoder/layer_3/output/dense/bias/variable
o
2cond/bert/encoder/layer_3/output/dropout/keep_probConst^cond/switch_t*
valueB
 *fff?*
dtype0

.cond/bert/encoder/layer_3/output/dropout/ShapeShape.cond/bert/encoder/layer_3/output/dense/BiasAdd*
T0*
out_type0
x
;cond/bert/encoder/layer_3/output/dropout/random_uniform/minConst^cond/switch_t*
dtype0*
valueB
 *    
x
;cond/bert/encoder/layer_3/output/dropout/random_uniform/maxConst^cond/switch_t*
valueB
 *  ?*
dtype0
ĩ
Econd/bert/encoder/layer_3/output/dropout/random_uniform/RandomUniformRandomUniform.cond/bert/encoder/layer_3/output/dropout/Shape*
dtype0*
seed2 *

seed *
T0
Å
;cond/bert/encoder/layer_3/output/dropout/random_uniform/subSub;cond/bert/encoder/layer_3/output/dropout/random_uniform/max;cond/bert/encoder/layer_3/output/dropout/random_uniform/min*
T0
Ī
;cond/bert/encoder/layer_3/output/dropout/random_uniform/mulMulEcond/bert/encoder/layer_3/output/dropout/random_uniform/RandomUniform;cond/bert/encoder/layer_3/output/dropout/random_uniform/sub*
T0
Á
7cond/bert/encoder/layer_3/output/dropout/random_uniformAdd;cond/bert/encoder/layer_3/output/dropout/random_uniform/mul;cond/bert/encoder/layer_3/output/dropout/random_uniform/min*
T0
Š
,cond/bert/encoder/layer_3/output/dropout/addAdd2cond/bert/encoder/layer_3/output/dropout/keep_prob7cond/bert/encoder/layer_3/output/dropout/random_uniform*
T0
n
.cond/bert/encoder/layer_3/output/dropout/FloorFloor,cond/bert/encoder/layer_3/output/dropout/add*
T0
¤
,cond/bert/encoder/layer_3/output/dropout/divRealDiv.cond/bert/encoder/layer_3/output/dense/BiasAdd2cond/bert/encoder/layer_3/output/dropout/keep_prob*
T0

,cond/bert/encoder/layer_3/output/dropout/mulMul,cond/bert/encoder/layer_3/output/dropout/div.cond/bert/encoder/layer_3/output/dropout/Floor*
T0
¨
$cond/bert/encoder/layer_3/output/addAdd,cond/bert/encoder/layer_3/output/dropout/mulDcond/bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add_1*
T0
Á
@mio_variable/bert/encoder/layer_3/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*9
	container,*bert/encoder/layer_3/output/LayerNorm/beta
Á
@mio_variable/bert/encoder/layer_3/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*9
	container,*bert/encoder/layer_3/output/LayerNorm/beta*
shape:
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
Amio_variable/bert/encoder/layer_3/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+bert/encoder/layer_3/output/LayerNorm/gamma*
shape:
E
Initializer_68/onesConst*
valueB*  ?*
dtype0
ķ
	Assign_68AssignAmio_variable/bert/encoder/layer_3/output/LayerNorm/gamma/gradientInitializer_68/ones*
use_locking(*
T0*T
_classJ
HFloc:@mio_variable/bert/encoder/layer_3/output/LayerNorm/gamma/gradient*
validate_shape(

Icond/bert/encoder/layer_3/output/LayerNorm/moments/mean/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0
Ö
7cond/bert/encoder/layer_3/output/LayerNorm/moments/meanMean$cond/bert/encoder/layer_3/output/addIcond/bert/encoder/layer_3/output/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0

?cond/bert/encoder/layer_3/output/LayerNorm/moments/StopGradientStopGradient7cond/bert/encoder/layer_3/output/LayerNorm/moments/mean*
T0
É
Dcond/bert/encoder/layer_3/output/LayerNorm/moments/SquaredDifferenceSquaredDifference$cond/bert/encoder/layer_3/output/add?cond/bert/encoder/layer_3/output/LayerNorm/moments/StopGradient*
T0

Mcond/bert/encoder/layer_3/output/LayerNorm/moments/variance/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0
ū
;cond/bert/encoder/layer_3/output/LayerNorm/moments/varianceMeanDcond/bert/encoder/layer_3/output/LayerNorm/moments/SquaredDifferenceMcond/bert/encoder/layer_3/output/LayerNorm/moments/variance/reduction_indices*
T0*

Tidx0*
	keep_dims(
w
:cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/add/yConst^cond/switch_t*
dtype0*
valueB
 *Ėŧ+
Á
8cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/addAdd;cond/bert/encoder/layer_3/output/LayerNorm/moments/variance:cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/add/y*
T0

:cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/RsqrtRsqrt8cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/add*
T0
Į
8cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/mulMul:cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/RsqrtAcond/bert/encoder/layer_3/output/LayerNorm/batchnorm/mul/Switch:1*
T0
ų
?cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/mul/SwitchSwitchAmio_variable/bert/encoder/layer_3/output/LayerNorm/gamma/variablecond/pred_id*
T0*T
_classJ
HFloc:@mio_variable/bert/encoder/layer_3/output/LayerNorm/gamma/variable
Ē
:cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/mul_1Mul$cond/bert/encoder/layer_3/output/add8cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/mul*
T0
Ŋ
:cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/mul_2Mul7cond/bert/encoder/layer_3/output/LayerNorm/moments/mean8cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/mul*
T0
Į
8cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/subSubAcond/bert/encoder/layer_3/output/LayerNorm/batchnorm/sub/Switch:1:cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/mul_2*
T0
÷
?cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/sub/SwitchSwitch@mio_variable/bert/encoder/layer_3/output/LayerNorm/beta/variablecond/pred_id*
T0*S
_classI
GEloc:@mio_variable/bert/encoder/layer_3/output/LayerNorm/beta/variable
Ā
:cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1Add:cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/mul_18cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/sub*
T0

.cond/bert/encoder/layer_4/attention/self/ShapeShape:cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
z
<cond/bert/encoder/layer_4/attention/self/strided_slice/stackConst^cond/switch_t*
valueB: *
dtype0
|
>cond/bert/encoder/layer_4/attention/self/strided_slice/stack_1Const^cond/switch_t*
valueB:*
dtype0
|
>cond/bert/encoder/layer_4/attention/self/strided_slice/stack_2Const^cond/switch_t*
valueB:*
dtype0
Ž
6cond/bert/encoder/layer_4/attention/self/strided_sliceStridedSlice.cond/bert/encoder/layer_4/attention/self/Shape<cond/bert/encoder/layer_4/attention/self/strided_slice/stack>cond/bert/encoder/layer_4/attention/self/strided_slice/stack_1>cond/bert/encoder/layer_4/attention/self/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0

0cond/bert/encoder/layer_4/attention/self/Shape_1Shape:cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
|
>cond/bert/encoder/layer_4/attention/self/strided_slice_1/stackConst^cond/switch_t*
valueB: *
dtype0
~
@cond/bert/encoder/layer_4/attention/self/strided_slice_1/stack_1Const^cond/switch_t*
dtype0*
valueB:
~
@cond/bert/encoder/layer_4/attention/self/strided_slice_1/stack_2Const^cond/switch_t*
valueB:*
dtype0
¸
8cond/bert/encoder/layer_4/attention/self/strided_slice_1StridedSlice0cond/bert/encoder/layer_4/attention/self/Shape_1>cond/bert/encoder/layer_4/attention/self/strided_slice_1/stack@cond/bert/encoder/layer_4/attention/self/strided_slice_1/stack_1@cond/bert/encoder/layer_4/attention/self/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
Ō
Fmio_variable/bert/encoder/layer_4/attention/self/query/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_4/attention/self/query/kernel*
shape:

Ō
Fmio_variable/bert/encoder/layer_4/attention/self/query/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*?
	container20bert/encoder/layer_4/attention/self/query/kernel
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
/Initializer_69/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_69/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 
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
Dmio_variable/bert/encoder/layer_4/attention/self/query/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*=
	container0.bert/encoder/layer_4/attention/self/query/bias
F
Initializer_70/zerosConst*
dtype0*
valueB*    
ú
	Assign_70AssignDmio_variable/bert/encoder/layer_4/attention/self/query/bias/gradientInitializer_70/zeros*
validate_shape(*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_4/attention/self/query/bias/gradient
ę
5cond/bert/encoder/layer_4/attention/self/query/MatMulMatMul:cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1>cond/bert/encoder/layer_4/attention/self/query/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 

<cond/bert/encoder/layer_4/attention/self/query/MatMul/SwitchSwitchFmio_variable/bert/encoder/layer_4/attention/self/query/kernel/variablecond/pred_id*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_4/attention/self/query/kernel/variable
Ų
6cond/bert/encoder/layer_4/attention/self/query/BiasAddBiasAdd5cond/bert/encoder/layer_4/attention/self/query/MatMul?cond/bert/encoder/layer_4/attention/self/query/BiasAdd/Switch:1*
T0*
data_formatNHWC
ũ
=cond/bert/encoder/layer_4/attention/self/query/BiasAdd/SwitchSwitchDmio_variable/bert/encoder/layer_4/attention/self/query/bias/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_4/attention/self/query/bias/variable
Î
Dmio_variable/bert/encoder/layer_4/attention/self/key/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*=
	container0.bert/encoder/layer_4/attention/self/key/kernel
Î
Dmio_variable/bert/encoder/layer_4/attention/self/key/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*=
	container0.bert/encoder/layer_4/attention/self/key/kernel
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
&Initializer_71/truncated_normal/stddevConst*
dtype0*
valueB
 *
×Ŗ<

/Initializer_71/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_71/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 
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
æ
3cond/bert/encoder/layer_4/attention/self/key/MatMulMatMul:cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1<cond/bert/encoder/layer_4/attention/self/key/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 
ú
:cond/bert/encoder/layer_4/attention/self/key/MatMul/SwitchSwitchDmio_variable/bert/encoder/layer_4/attention/self/key/kernel/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_4/attention/self/key/kernel/variable
Ķ
4cond/bert/encoder/layer_4/attention/self/key/BiasAddBiasAdd3cond/bert/encoder/layer_4/attention/self/key/MatMul=cond/bert/encoder/layer_4/attention/self/key/BiasAdd/Switch:1*
T0*
data_formatNHWC
÷
;cond/bert/encoder/layer_4/attention/self/key/BiasAdd/SwitchSwitchBmio_variable/bert/encoder/layer_4/attention/self/key/bias/variablecond/pred_id*U
_classK
IGloc:@mio_variable/bert/encoder/layer_4/attention/self/key/bias/variable*
T0
Ō
Fmio_variable/bert/encoder/layer_4/attention/self/value/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*?
	container20bert/encoder/layer_4/attention/self/value/kernel
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
$Initializer_73/truncated_normal/meanConst*
dtype0*
valueB
 *    
S
&Initializer_73/truncated_normal/stddevConst*
dtype0*
valueB
 *
×Ŗ<
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
Dmio_variable/bert/encoder/layer_4/attention/self/value/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_4/attention/self/value/bias*
shape:
É
Dmio_variable/bert/encoder/layer_4/attention/self/value/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_4/attention/self/value/bias*
shape:
F
Initializer_74/zerosConst*
valueB*    *
dtype0
ú
	Assign_74AssignDmio_variable/bert/encoder/layer_4/attention/self/value/bias/gradientInitializer_74/zeros*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_4/attention/self/value/bias/gradient*
validate_shape(*
use_locking(
ę
5cond/bert/encoder/layer_4/attention/self/value/MatMulMatMul:cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1>cond/bert/encoder/layer_4/attention/self/value/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 

<cond/bert/encoder/layer_4/attention/self/value/MatMul/SwitchSwitchFmio_variable/bert/encoder/layer_4/attention/self/value/kernel/variablecond/pred_id*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_4/attention/self/value/kernel/variable*
T0
Ų
6cond/bert/encoder/layer_4/attention/self/value/BiasAddBiasAdd5cond/bert/encoder/layer_4/attention/self/value/MatMul?cond/bert/encoder/layer_4/attention/self/value/BiasAdd/Switch:1*
T0*
data_formatNHWC
ũ
=cond/bert/encoder/layer_4/attention/self/value/BiasAdd/SwitchSwitchDmio_variable/bert/encoder/layer_4/attention/self/value/bias/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_4/attention/self/value/bias/variable
r
8cond/bert/encoder/layer_4/attention/self/Reshape/shape/1Const^cond/switch_t*
value	B :*
dtype0
r
8cond/bert/encoder/layer_4/attention/self/Reshape/shape/2Const^cond/switch_t*
value	B :*
dtype0
r
8cond/bert/encoder/layer_4/attention/self/Reshape/shape/3Const^cond/switch_t*
dtype0*
value	B : 
­
6cond/bert/encoder/layer_4/attention/self/Reshape/shapePack!cond/bert/encoder/strided_slice_28cond/bert/encoder/layer_4/attention/self/Reshape/shape/18cond/bert/encoder/layer_4/attention/self/Reshape/shape/28cond/bert/encoder/layer_4/attention/self/Reshape/shape/3*
T0*

axis *
N
Â
0cond/bert/encoder/layer_4/attention/self/ReshapeReshape6cond/bert/encoder/layer_4/attention/self/query/BiasAdd6cond/bert/encoder/layer_4/attention/self/Reshape/shape*
T0*
Tshape0

7cond/bert/encoder/layer_4/attention/self/transpose/permConst^cond/switch_t*
dtype0*%
valueB"             
Ā
2cond/bert/encoder/layer_4/attention/self/transpose	Transpose0cond/bert/encoder/layer_4/attention/self/Reshape7cond/bert/encoder/layer_4/attention/self/transpose/perm*
T0*
Tperm0
t
:cond/bert/encoder/layer_4/attention/self/Reshape_1/shape/1Const^cond/switch_t*
dtype0*
value	B :
t
:cond/bert/encoder/layer_4/attention/self/Reshape_1/shape/2Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_4/attention/self/Reshape_1/shape/3Const^cond/switch_t*
value	B : *
dtype0
ĩ
8cond/bert/encoder/layer_4/attention/self/Reshape_1/shapePack!cond/bert/encoder/strided_slice_2:cond/bert/encoder/layer_4/attention/self/Reshape_1/shape/1:cond/bert/encoder/layer_4/attention/self/Reshape_1/shape/2:cond/bert/encoder/layer_4/attention/self/Reshape_1/shape/3*
T0*

axis *
N
Ä
2cond/bert/encoder/layer_4/attention/self/Reshape_1Reshape4cond/bert/encoder/layer_4/attention/self/key/BiasAdd8cond/bert/encoder/layer_4/attention/self/Reshape_1/shape*
T0*
Tshape0

9cond/bert/encoder/layer_4/attention/self/transpose_1/permConst^cond/switch_t*
dtype0*%
valueB"             
Æ
4cond/bert/encoder/layer_4/attention/self/transpose_1	Transpose2cond/bert/encoder/layer_4/attention/self/Reshape_19cond/bert/encoder/layer_4/attention/self/transpose_1/perm*
T0*
Tperm0
Ë
/cond/bert/encoder/layer_4/attention/self/MatMulBatchMatMul2cond/bert/encoder/layer_4/attention/self/transpose4cond/bert/encoder/layer_4/attention/self/transpose_1*
adj_x( *
adj_y(*
T0
k
.cond/bert/encoder/layer_4/attention/self/Mul/yConst^cond/switch_t*
valueB
 *ķ5>*
dtype0

,cond/bert/encoder/layer_4/attention/self/MulMul/cond/bert/encoder/layer_4/attention/self/MatMul.cond/bert/encoder/layer_4/attention/self/Mul/y*
T0
u
7cond/bert/encoder/layer_4/attention/self/ExpandDims/dimConst^cond/switch_t*
valueB:*
dtype0
Ļ
3cond/bert/encoder/layer_4/attention/self/ExpandDims
ExpandDimscond/bert/encoder/mul7cond/bert/encoder/layer_4/attention/self/ExpandDims/dim*
T0*

Tdim0
k
.cond/bert/encoder/layer_4/attention/self/sub/xConst^cond/switch_t*
valueB
 *  ?*
dtype0
Ą
,cond/bert/encoder/layer_4/attention/self/subSub.cond/bert/encoder/layer_4/attention/self/sub/x3cond/bert/encoder/layer_4/attention/self/ExpandDims*
T0
m
0cond/bert/encoder/layer_4/attention/self/mul_1/yConst^cond/switch_t*
valueB
 * @Æ*
dtype0

.cond/bert/encoder/layer_4/attention/self/mul_1Mul,cond/bert/encoder/layer_4/attention/self/sub0cond/bert/encoder/layer_4/attention/self/mul_1/y*
T0

,cond/bert/encoder/layer_4/attention/self/addAdd,cond/bert/encoder/layer_4/attention/self/Mul.cond/bert/encoder/layer_4/attention/self/mul_1*
T0
r
0cond/bert/encoder/layer_4/attention/self/SoftmaxSoftmax,cond/bert/encoder/layer_4/attention/self/add*
T0
w
:cond/bert/encoder/layer_4/attention/self/dropout/keep_probConst^cond/switch_t*
dtype0*
valueB
 *fff?

6cond/bert/encoder/layer_4/attention/self/dropout/ShapeShape0cond/bert/encoder/layer_4/attention/self/Softmax*
T0*
out_type0

Ccond/bert/encoder/layer_4/attention/self/dropout/random_uniform/minConst^cond/switch_t*
valueB
 *    *
dtype0

Ccond/bert/encoder/layer_4/attention/self/dropout/random_uniform/maxConst^cond/switch_t*
dtype0*
valueB
 *  ?
Å
Mcond/bert/encoder/layer_4/attention/self/dropout/random_uniform/RandomUniformRandomUniform6cond/bert/encoder/layer_4/attention/self/dropout/Shape*

seed *
T0*
dtype0*
seed2 
Ũ
Ccond/bert/encoder/layer_4/attention/self/dropout/random_uniform/subSubCcond/bert/encoder/layer_4/attention/self/dropout/random_uniform/maxCcond/bert/encoder/layer_4/attention/self/dropout/random_uniform/min*
T0
į
Ccond/bert/encoder/layer_4/attention/self/dropout/random_uniform/mulMulMcond/bert/encoder/layer_4/attention/self/dropout/random_uniform/RandomUniformCcond/bert/encoder/layer_4/attention/self/dropout/random_uniform/sub*
T0
Ų
?cond/bert/encoder/layer_4/attention/self/dropout/random_uniformAddCcond/bert/encoder/layer_4/attention/self/dropout/random_uniform/mulCcond/bert/encoder/layer_4/attention/self/dropout/random_uniform/min*
T0
Á
4cond/bert/encoder/layer_4/attention/self/dropout/addAdd:cond/bert/encoder/layer_4/attention/self/dropout/keep_prob?cond/bert/encoder/layer_4/attention/self/dropout/random_uniform*
T0
~
6cond/bert/encoder/layer_4/attention/self/dropout/FloorFloor4cond/bert/encoder/layer_4/attention/self/dropout/add*
T0
ļ
4cond/bert/encoder/layer_4/attention/self/dropout/divRealDiv0cond/bert/encoder/layer_4/attention/self/Softmax:cond/bert/encoder/layer_4/attention/self/dropout/keep_prob*
T0
˛
4cond/bert/encoder/layer_4/attention/self/dropout/mulMul4cond/bert/encoder/layer_4/attention/self/dropout/div6cond/bert/encoder/layer_4/attention/self/dropout/Floor*
T0
t
:cond/bert/encoder/layer_4/attention/self/Reshape_2/shape/1Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_4/attention/self/Reshape_2/shape/2Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_4/attention/self/Reshape_2/shape/3Const^cond/switch_t*
value	B : *
dtype0
ĩ
8cond/bert/encoder/layer_4/attention/self/Reshape_2/shapePack!cond/bert/encoder/strided_slice_2:cond/bert/encoder/layer_4/attention/self/Reshape_2/shape/1:cond/bert/encoder/layer_4/attention/self/Reshape_2/shape/2:cond/bert/encoder/layer_4/attention/self/Reshape_2/shape/3*
T0*

axis *
N
Æ
2cond/bert/encoder/layer_4/attention/self/Reshape_2Reshape6cond/bert/encoder/layer_4/attention/self/value/BiasAdd8cond/bert/encoder/layer_4/attention/self/Reshape_2/shape*
T0*
Tshape0

9cond/bert/encoder/layer_4/attention/self/transpose_2/permConst^cond/switch_t*%
valueB"             *
dtype0
Æ
4cond/bert/encoder/layer_4/attention/self/transpose_2	Transpose2cond/bert/encoder/layer_4/attention/self/Reshape_29cond/bert/encoder/layer_4/attention/self/transpose_2/perm*
T0*
Tperm0
Ī
1cond/bert/encoder/layer_4/attention/self/MatMul_1BatchMatMul4cond/bert/encoder/layer_4/attention/self/dropout/mul4cond/bert/encoder/layer_4/attention/self/transpose_2*
adj_x( *
adj_y( *
T0

9cond/bert/encoder/layer_4/attention/self/transpose_3/permConst^cond/switch_t*
dtype0*%
valueB"             
Å
4cond/bert/encoder/layer_4/attention/self/transpose_3	Transpose1cond/bert/encoder/layer_4/attention/self/MatMul_19cond/bert/encoder/layer_4/attention/self/transpose_3/perm*
Tperm0*
T0
j
0cond/bert/encoder/layer_4/attention/self/mul_2/yConst^cond/switch_t*
value	B :*
dtype0

.cond/bert/encoder/layer_4/attention/self/mul_2Mul!cond/bert/encoder/strided_slice_20cond/bert/encoder/layer_4/attention/self/mul_2/y*
T0
u
:cond/bert/encoder/layer_4/attention/self/Reshape_3/shape/1Const^cond/switch_t*
value
B :*
dtype0
Ę
8cond/bert/encoder/layer_4/attention/self/Reshape_3/shapePack.cond/bert/encoder/layer_4/attention/self/mul_2:cond/bert/encoder/layer_4/attention/self/Reshape_3/shape/1*
T0*

axis *
N
Ä
2cond/bert/encoder/layer_4/attention/self/Reshape_3Reshape4cond/bert/encoder/layer_4/attention/self/transpose_38cond/bert/encoder/layer_4/attention/self/Reshape_3/shape*
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
$Initializer_75/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_75/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0
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
Fmio_variable/bert/encoder/layer_4/attention/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_4/attention/output/dense/bias*
shape:
F
Initializer_76/zerosConst*
dtype0*
valueB*    
ū
	Assign_76AssignFmio_variable/bert/encoder/layer_4/attention/output/dense/bias/gradientInitializer_76/zeros*
validate_shape(*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_4/attention/output/dense/bias/gradient
æ
7cond/bert/encoder/layer_4/attention/output/dense/MatMulMatMul2cond/bert/encoder/layer_4/attention/self/Reshape_3@cond/bert/encoder/layer_4/attention/output/dense/MatMul/Switch:1*
transpose_a( *
transpose_b( *
T0

>cond/bert/encoder/layer_4/attention/output/dense/MatMul/SwitchSwitchHmio_variable/bert/encoder/layer_4/attention/output/dense/kernel/variablecond/pred_id*[
_classQ
OMloc:@mio_variable/bert/encoder/layer_4/attention/output/dense/kernel/variable*
T0
ß
8cond/bert/encoder/layer_4/attention/output/dense/BiasAddBiasAdd7cond/bert/encoder/layer_4/attention/output/dense/MatMulAcond/bert/encoder/layer_4/attention/output/dense/BiasAdd/Switch:1*
data_formatNHWC*
T0

?cond/bert/encoder/layer_4/attention/output/dense/BiasAdd/SwitchSwitchFmio_variable/bert/encoder/layer_4/attention/output/dense/bias/variablecond/pred_id*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_4/attention/output/dense/bias/variable
y
<cond/bert/encoder/layer_4/attention/output/dropout/keep_probConst^cond/switch_t*
dtype0*
valueB
 *fff?

8cond/bert/encoder/layer_4/attention/output/dropout/ShapeShape8cond/bert/encoder/layer_4/attention/output/dense/BiasAdd*
out_type0*
T0

Econd/bert/encoder/layer_4/attention/output/dropout/random_uniform/minConst^cond/switch_t*
valueB
 *    *
dtype0

Econd/bert/encoder/layer_4/attention/output/dropout/random_uniform/maxConst^cond/switch_t*
dtype0*
valueB
 *  ?
É
Ocond/bert/encoder/layer_4/attention/output/dropout/random_uniform/RandomUniformRandomUniform8cond/bert/encoder/layer_4/attention/output/dropout/Shape*

seed *
T0*
dtype0*
seed2 
ã
Econd/bert/encoder/layer_4/attention/output/dropout/random_uniform/subSubEcond/bert/encoder/layer_4/attention/output/dropout/random_uniform/maxEcond/bert/encoder/layer_4/attention/output/dropout/random_uniform/min*
T0
í
Econd/bert/encoder/layer_4/attention/output/dropout/random_uniform/mulMulOcond/bert/encoder/layer_4/attention/output/dropout/random_uniform/RandomUniformEcond/bert/encoder/layer_4/attention/output/dropout/random_uniform/sub*
T0
ß
Acond/bert/encoder/layer_4/attention/output/dropout/random_uniformAddEcond/bert/encoder/layer_4/attention/output/dropout/random_uniform/mulEcond/bert/encoder/layer_4/attention/output/dropout/random_uniform/min*
T0
Į
6cond/bert/encoder/layer_4/attention/output/dropout/addAdd<cond/bert/encoder/layer_4/attention/output/dropout/keep_probAcond/bert/encoder/layer_4/attention/output/dropout/random_uniform*
T0

8cond/bert/encoder/layer_4/attention/output/dropout/FloorFloor6cond/bert/encoder/layer_4/attention/output/dropout/add*
T0
Â
6cond/bert/encoder/layer_4/attention/output/dropout/divRealDiv8cond/bert/encoder/layer_4/attention/output/dense/BiasAdd<cond/bert/encoder/layer_4/attention/output/dropout/keep_prob*
T0
¸
6cond/bert/encoder/layer_4/attention/output/dropout/mulMul6cond/bert/encoder/layer_4/attention/output/dropout/div8cond/bert/encoder/layer_4/attention/output/dropout/Floor*
T0
˛
.cond/bert/encoder/layer_4/attention/output/addAdd6cond/bert/encoder/layer_4/attention/output/dropout/mul:cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1*
T0
Õ
Jmio_variable/bert/encoder/layer_4/attention/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*C
	container64bert/encoder/layer_4/attention/output/LayerNorm/beta
Õ
Jmio_variable/bert/encoder/layer_4/attention/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*C
	container64bert/encoder/layer_4/attention/output/LayerNorm/beta
F
Initializer_77/zerosConst*
valueB*    *
dtype0

	Assign_77AssignJmio_variable/bert/encoder/layer_4/attention/output/LayerNorm/beta/gradientInitializer_77/zeros*
use_locking(*
T0*]
_classS
QOloc:@mio_variable/bert/encoder/layer_4/attention/output/LayerNorm/beta/gradient*
validate_shape(
×
Kmio_variable/bert/encoder/layer_4/attention/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*D
	container75bert/encoder/layer_4/attention/output/LayerNorm/gamma
×
Kmio_variable/bert/encoder/layer_4/attention/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*D
	container75bert/encoder/layer_4/attention/output/LayerNorm/gamma
E
Initializer_78/onesConst*
dtype0*
valueB*  ?

	Assign_78AssignKmio_variable/bert/encoder/layer_4/attention/output/LayerNorm/gamma/gradientInitializer_78/ones*
T0*^
_classT
RPloc:@mio_variable/bert/encoder/layer_4/attention/output/LayerNorm/gamma/gradient*
validate_shape(*
use_locking(

Scond/bert/encoder/layer_4/attention/output/LayerNorm/moments/mean/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0
ô
Acond/bert/encoder/layer_4/attention/output/LayerNorm/moments/meanMean.cond/bert/encoder/layer_4/attention/output/addScond/bert/encoder/layer_4/attention/output/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
Ĩ
Icond/bert/encoder/layer_4/attention/output/LayerNorm/moments/StopGradientStopGradientAcond/bert/encoder/layer_4/attention/output/LayerNorm/moments/mean*
T0
į
Ncond/bert/encoder/layer_4/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference.cond/bert/encoder/layer_4/attention/output/addIcond/bert/encoder/layer_4/attention/output/LayerNorm/moments/StopGradient*
T0

Wcond/bert/encoder/layer_4/attention/output/LayerNorm/moments/variance/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0

Econd/bert/encoder/layer_4/attention/output/LayerNorm/moments/varianceMeanNcond/bert/encoder/layer_4/attention/output/LayerNorm/moments/SquaredDifferenceWcond/bert/encoder/layer_4/attention/output/LayerNorm/moments/variance/reduction_indices*
T0*

Tidx0*
	keep_dims(

Dcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add/yConst^cond/switch_t*
valueB
 *Ėŧ+*
dtype0
ß
Bcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/addAddEcond/bert/encoder/layer_4/attention/output/LayerNorm/moments/varianceDcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add/y*
T0

Dcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/RsqrtRsqrtBcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add*
T0
å
Bcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mulMulDcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/RsqrtKcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul/Switch:1*
T0

Icond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul/SwitchSwitchKmio_variable/bert/encoder/layer_4/attention/output/LayerNorm/gamma/variablecond/pred_id*
T0*^
_classT
RPloc:@mio_variable/bert/encoder/layer_4/attention/output/LayerNorm/gamma/variable
Č
Dcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul_1Mul.cond/bert/encoder/layer_4/attention/output/addBcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul*
T0
Û
Dcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul_2MulAcond/bert/encoder/layer_4/attention/output/LayerNorm/moments/meanBcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul*
T0
å
Bcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/subSubKcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/sub/Switch:1Dcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul_2*
T0

Icond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/sub/SwitchSwitchJmio_variable/bert/encoder/layer_4/attention/output/LayerNorm/beta/variablecond/pred_id*
T0*]
_classS
QOloc:@mio_variable/bert/encoder/layer_4/attention/output/LayerNorm/beta/variable
Ū
Dcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add_1AddDcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul_1Bcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/sub*
T0
Î
Dmio_variable/bert/encoder/layer_4/intermediate/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_4/intermediate/dense/kernel*
shape:

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
seed2 *

seed *
T0*
dtype0
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
$Initializer_80/zeros/shape_as_tensorConst*
dtype0*
valueB:
G
Initializer_80/zeros/ConstConst*
valueB
 *    *
dtype0
y
Initializer_80/zerosFill$Initializer_80/zeros/shape_as_tensorInitializer_80/zeros/Const*
T0*

index_type0
ö
	Assign_80AssignBmio_variable/bert/encoder/layer_4/intermediate/dense/bias/gradientInitializer_80/zeros*
validate_shape(*
use_locking(*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_4/intermediate/dense/bias/gradient
đ
3cond/bert/encoder/layer_4/intermediate/dense/MatMulMatMulDcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add_1<cond/bert/encoder/layer_4/intermediate/dense/MatMul/Switch:1*
transpose_b( *
T0*
transpose_a( 
ú
:cond/bert/encoder/layer_4/intermediate/dense/MatMul/SwitchSwitchDmio_variable/bert/encoder/layer_4/intermediate/dense/kernel/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_4/intermediate/dense/kernel/variable
Ķ
4cond/bert/encoder/layer_4/intermediate/dense/BiasAddBiasAdd3cond/bert/encoder/layer_4/intermediate/dense/MatMul=cond/bert/encoder/layer_4/intermediate/dense/BiasAdd/Switch:1*
T0*
data_formatNHWC
÷
;cond/bert/encoder/layer_4/intermediate/dense/BiasAdd/SwitchSwitchBmio_variable/bert/encoder/layer_4/intermediate/dense/bias/variablecond/pred_id*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_4/intermediate/dense/bias/variable
o
2cond/bert/encoder/layer_4/intermediate/dense/Pow/yConst^cond/switch_t*
valueB
 *  @@*
dtype0
Ē
0cond/bert/encoder/layer_4/intermediate/dense/PowPow4cond/bert/encoder/layer_4/intermediate/dense/BiasAdd2cond/bert/encoder/layer_4/intermediate/dense/Pow/y*
T0
o
2cond/bert/encoder/layer_4/intermediate/dense/mul/xConst^cond/switch_t*
valueB
 *'7=*
dtype0
Ļ
0cond/bert/encoder/layer_4/intermediate/dense/mulMul2cond/bert/encoder/layer_4/intermediate/dense/mul/x0cond/bert/encoder/layer_4/intermediate/dense/Pow*
T0
¨
0cond/bert/encoder/layer_4/intermediate/dense/addAdd4cond/bert/encoder/layer_4/intermediate/dense/BiasAdd0cond/bert/encoder/layer_4/intermediate/dense/mul*
T0
q
4cond/bert/encoder/layer_4/intermediate/dense/mul_1/xConst^cond/switch_t*
dtype0*
valueB
 **BL?
Ē
2cond/bert/encoder/layer_4/intermediate/dense/mul_1Mul4cond/bert/encoder/layer_4/intermediate/dense/mul_1/x0cond/bert/encoder/layer_4/intermediate/dense/add*
T0
v
1cond/bert/encoder/layer_4/intermediate/dense/TanhTanh2cond/bert/encoder/layer_4/intermediate/dense/mul_1*
T0
q
4cond/bert/encoder/layer_4/intermediate/dense/add_1/xConst^cond/switch_t*
valueB
 *  ?*
dtype0
Ģ
2cond/bert/encoder/layer_4/intermediate/dense/add_1Add4cond/bert/encoder/layer_4/intermediate/dense/add_1/x1cond/bert/encoder/layer_4/intermediate/dense/Tanh*
T0
q
4cond/bert/encoder/layer_4/intermediate/dense/mul_2/xConst^cond/switch_t*
valueB
 *   ?*
dtype0
Ŧ
2cond/bert/encoder/layer_4/intermediate/dense/mul_2Mul4cond/bert/encoder/layer_4/intermediate/dense/mul_2/x2cond/bert/encoder/layer_4/intermediate/dense/add_1*
T0
Ŧ
2cond/bert/encoder/layer_4/intermediate/dense/mul_3Mul4cond/bert/encoder/layer_4/intermediate/dense/BiasAdd2cond/bert/encoder/layer_4/intermediate/dense/mul_2*
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
&Initializer_81/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_81/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_81/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0

#Initializer_81/truncated_normal/mulMul/Initializer_81/truncated_normal/TruncatedNormal&Initializer_81/truncated_normal/stddev*
T0
z
Initializer_81/truncated_normalAdd#Initializer_81/truncated_normal/mul$Initializer_81/truncated_normal/mean*
T0
ų
	Assign_81Assign>mio_variable/bert/encoder/layer_4/output/dense/kernel/gradientInitializer_81/truncated_normal*
T0*Q
_classG
ECloc:@mio_variable/bert/encoder/layer_4/output/dense/kernel/gradient*
validate_shape(*
use_locking(
š
<mio_variable/bert/encoder/layer_4/output/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&bert/encoder/layer_4/output/dense/bias*
shape:
š
<mio_variable/bert/encoder/layer_4/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&bert/encoder/layer_4/output/dense/bias*
shape:
F
Initializer_82/zerosConst*
valueB*    *
dtype0
ę
	Assign_82Assign<mio_variable/bert/encoder/layer_4/output/dense/bias/gradientInitializer_82/zeros*O
_classE
CAloc:@mio_variable/bert/encoder/layer_4/output/dense/bias/gradient*
validate_shape(*
use_locking(*
T0
Ō
-cond/bert/encoder/layer_4/output/dense/MatMulMatMul2cond/bert/encoder/layer_4/intermediate/dense/mul_36cond/bert/encoder/layer_4/output/dense/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 
č
4cond/bert/encoder/layer_4/output/dense/MatMul/SwitchSwitch>mio_variable/bert/encoder/layer_4/output/dense/kernel/variablecond/pred_id*
T0*Q
_classG
ECloc:@mio_variable/bert/encoder/layer_4/output/dense/kernel/variable
Á
.cond/bert/encoder/layer_4/output/dense/BiasAddBiasAdd-cond/bert/encoder/layer_4/output/dense/MatMul7cond/bert/encoder/layer_4/output/dense/BiasAdd/Switch:1*
T0*
data_formatNHWC
å
5cond/bert/encoder/layer_4/output/dense/BiasAdd/SwitchSwitch<mio_variable/bert/encoder/layer_4/output/dense/bias/variablecond/pred_id*
T0*O
_classE
CAloc:@mio_variable/bert/encoder/layer_4/output/dense/bias/variable
o
2cond/bert/encoder/layer_4/output/dropout/keep_probConst^cond/switch_t*
valueB
 *fff?*
dtype0

.cond/bert/encoder/layer_4/output/dropout/ShapeShape.cond/bert/encoder/layer_4/output/dense/BiasAdd*
T0*
out_type0
x
;cond/bert/encoder/layer_4/output/dropout/random_uniform/minConst^cond/switch_t*
valueB
 *    *
dtype0
x
;cond/bert/encoder/layer_4/output/dropout/random_uniform/maxConst^cond/switch_t*
valueB
 *  ?*
dtype0
ĩ
Econd/bert/encoder/layer_4/output/dropout/random_uniform/RandomUniformRandomUniform.cond/bert/encoder/layer_4/output/dropout/Shape*
T0*
dtype0*
seed2 *

seed 
Å
;cond/bert/encoder/layer_4/output/dropout/random_uniform/subSub;cond/bert/encoder/layer_4/output/dropout/random_uniform/max;cond/bert/encoder/layer_4/output/dropout/random_uniform/min*
T0
Ī
;cond/bert/encoder/layer_4/output/dropout/random_uniform/mulMulEcond/bert/encoder/layer_4/output/dropout/random_uniform/RandomUniform;cond/bert/encoder/layer_4/output/dropout/random_uniform/sub*
T0
Á
7cond/bert/encoder/layer_4/output/dropout/random_uniformAdd;cond/bert/encoder/layer_4/output/dropout/random_uniform/mul;cond/bert/encoder/layer_4/output/dropout/random_uniform/min*
T0
Š
,cond/bert/encoder/layer_4/output/dropout/addAdd2cond/bert/encoder/layer_4/output/dropout/keep_prob7cond/bert/encoder/layer_4/output/dropout/random_uniform*
T0
n
.cond/bert/encoder/layer_4/output/dropout/FloorFloor,cond/bert/encoder/layer_4/output/dropout/add*
T0
¤
,cond/bert/encoder/layer_4/output/dropout/divRealDiv.cond/bert/encoder/layer_4/output/dense/BiasAdd2cond/bert/encoder/layer_4/output/dropout/keep_prob*
T0

,cond/bert/encoder/layer_4/output/dropout/mulMul,cond/bert/encoder/layer_4/output/dropout/div.cond/bert/encoder/layer_4/output/dropout/Floor*
T0
¨
$cond/bert/encoder/layer_4/output/addAdd,cond/bert/encoder/layer_4/output/dropout/mulDcond/bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add_1*
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
	Assign_83Assign@mio_variable/bert/encoder/layer_4/output/LayerNorm/beta/gradientInitializer_83/zeros*
use_locking(*
T0*S
_classI
GEloc:@mio_variable/bert/encoder/layer_4/output/LayerNorm/beta/gradient*
validate_shape(
Ã
Amio_variable/bert/encoder/layer_4/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+bert/encoder/layer_4/output/LayerNorm/gamma*
shape:
Ã
Amio_variable/bert/encoder/layer_4/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*:
	container-+bert/encoder/layer_4/output/LayerNorm/gamma
E
Initializer_84/onesConst*
valueB*  ?*
dtype0
ķ
	Assign_84AssignAmio_variable/bert/encoder/layer_4/output/LayerNorm/gamma/gradientInitializer_84/ones*T
_classJ
HFloc:@mio_variable/bert/encoder/layer_4/output/LayerNorm/gamma/gradient*
validate_shape(*
use_locking(*
T0

Icond/bert/encoder/layer_4/output/LayerNorm/moments/mean/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0
Ö
7cond/bert/encoder/layer_4/output/LayerNorm/moments/meanMean$cond/bert/encoder/layer_4/output/addIcond/bert/encoder/layer_4/output/LayerNorm/moments/mean/reduction_indices*
T0*

Tidx0*
	keep_dims(

?cond/bert/encoder/layer_4/output/LayerNorm/moments/StopGradientStopGradient7cond/bert/encoder/layer_4/output/LayerNorm/moments/mean*
T0
É
Dcond/bert/encoder/layer_4/output/LayerNorm/moments/SquaredDifferenceSquaredDifference$cond/bert/encoder/layer_4/output/add?cond/bert/encoder/layer_4/output/LayerNorm/moments/StopGradient*
T0

Mcond/bert/encoder/layer_4/output/LayerNorm/moments/variance/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0
ū
;cond/bert/encoder/layer_4/output/LayerNorm/moments/varianceMeanDcond/bert/encoder/layer_4/output/LayerNorm/moments/SquaredDifferenceMcond/bert/encoder/layer_4/output/LayerNorm/moments/variance/reduction_indices*
T0*

Tidx0*
	keep_dims(
w
:cond/bert/encoder/layer_4/output/LayerNorm/batchnorm/add/yConst^cond/switch_t*
valueB
 *Ėŧ+*
dtype0
Á
8cond/bert/encoder/layer_4/output/LayerNorm/batchnorm/addAdd;cond/bert/encoder/layer_4/output/LayerNorm/moments/variance:cond/bert/encoder/layer_4/output/LayerNorm/batchnorm/add/y*
T0

:cond/bert/encoder/layer_4/output/LayerNorm/batchnorm/RsqrtRsqrt8cond/bert/encoder/layer_4/output/LayerNorm/batchnorm/add*
T0
Į
8cond/bert/encoder/layer_4/output/LayerNorm/batchnorm/mulMul:cond/bert/encoder/layer_4/output/LayerNorm/batchnorm/RsqrtAcond/bert/encoder/layer_4/output/LayerNorm/batchnorm/mul/Switch:1*
T0
ų
?cond/bert/encoder/layer_4/output/LayerNorm/batchnorm/mul/SwitchSwitchAmio_variable/bert/encoder/layer_4/output/LayerNorm/gamma/variablecond/pred_id*
T0*T
_classJ
HFloc:@mio_variable/bert/encoder/layer_4/output/LayerNorm/gamma/variable
Ē
:cond/bert/encoder/layer_4/output/LayerNorm/batchnorm/mul_1Mul$cond/bert/encoder/layer_4/output/add8cond/bert/encoder/layer_4/output/LayerNorm/batchnorm/mul*
T0
Ŋ
:cond/bert/encoder/layer_4/output/LayerNorm/batchnorm/mul_2Mul7cond/bert/encoder/layer_4/output/LayerNorm/moments/mean8cond/bert/encoder/layer_4/output/LayerNorm/batchnorm/mul*
T0
Į
8cond/bert/encoder/layer_4/output/LayerNorm/batchnorm/subSubAcond/bert/encoder/layer_4/output/LayerNorm/batchnorm/sub/Switch:1:cond/bert/encoder/layer_4/output/LayerNorm/batchnorm/mul_2*
T0
÷
?cond/bert/encoder/layer_4/output/LayerNorm/batchnorm/sub/SwitchSwitch@mio_variable/bert/encoder/layer_4/output/LayerNorm/beta/variablecond/pred_id*
T0*S
_classI
GEloc:@mio_variable/bert/encoder/layer_4/output/LayerNorm/beta/variable
Ā
:cond/bert/encoder/layer_4/output/LayerNorm/batchnorm/add_1Add:cond/bert/encoder/layer_4/output/LayerNorm/batchnorm/mul_18cond/bert/encoder/layer_4/output/LayerNorm/batchnorm/sub*
T0

-cond/bert/encoder/layer_4/output/StopGradientStopGradient:cond/bert/encoder/layer_4/output/LayerNorm/batchnorm/add_1*
T0

.cond/bert/encoder/layer_5/attention/self/ShapeShape-cond/bert/encoder/layer_4/output/StopGradient*
out_type0*
T0
z
<cond/bert/encoder/layer_5/attention/self/strided_slice/stackConst^cond/switch_t*
valueB: *
dtype0
|
>cond/bert/encoder/layer_5/attention/self/strided_slice/stack_1Const^cond/switch_t*
valueB:*
dtype0
|
>cond/bert/encoder/layer_5/attention/self/strided_slice/stack_2Const^cond/switch_t*
valueB:*
dtype0
Ž
6cond/bert/encoder/layer_5/attention/self/strided_sliceStridedSlice.cond/bert/encoder/layer_5/attention/self/Shape<cond/bert/encoder/layer_5/attention/self/strided_slice/stack>cond/bert/encoder/layer_5/attention/self/strided_slice/stack_1>cond/bert/encoder/layer_5/attention/self/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 

0cond/bert/encoder/layer_5/attention/self/Shape_1Shape-cond/bert/encoder/layer_4/output/StopGradient*
T0*
out_type0
|
>cond/bert/encoder/layer_5/attention/self/strided_slice_1/stackConst^cond/switch_t*
valueB: *
dtype0
~
@cond/bert/encoder/layer_5/attention/self/strided_slice_1/stack_1Const^cond/switch_t*
valueB:*
dtype0
~
@cond/bert/encoder/layer_5/attention/self/strided_slice_1/stack_2Const^cond/switch_t*
valueB:*
dtype0
¸
8cond/bert/encoder/layer_5/attention/self/strided_slice_1StridedSlice0cond/bert/encoder/layer_5/attention/self/Shape_1>cond/bert/encoder/layer_5/attention/self/strided_slice_1/stack@cond/bert/encoder/layer_5/attention/self/strided_slice_1/stack_1@cond/bert/encoder/layer_5/attention/self/strided_slice_1/stack_2*
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
Z
%Initializer_85/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_85/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_85/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_85/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_85/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0
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
Dmio_variable/bert/encoder/layer_5/attention/self/query/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*=
	container0.bert/encoder/layer_5/attention/self/query/bias
F
Initializer_86/zerosConst*
valueB*    *
dtype0
ú
	Assign_86AssignDmio_variable/bert/encoder/layer_5/attention/self/query/bias/gradientInitializer_86/zeros*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_5/attention/self/query/bias/gradient*
validate_shape(
Ũ
5cond/bert/encoder/layer_5/attention/self/query/MatMulMatMul-cond/bert/encoder/layer_4/output/StopGradient>cond/bert/encoder/layer_5/attention/self/query/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 

<cond/bert/encoder/layer_5/attention/self/query/MatMul/SwitchSwitchFmio_variable/bert/encoder/layer_5/attention/self/query/kernel/variablecond/pred_id*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_5/attention/self/query/kernel/variable
Ų
6cond/bert/encoder/layer_5/attention/self/query/BiasAddBiasAdd5cond/bert/encoder/layer_5/attention/self/query/MatMul?cond/bert/encoder/layer_5/attention/self/query/BiasAdd/Switch:1*
T0*
data_formatNHWC
ũ
=cond/bert/encoder/layer_5/attention/self/query/BiasAdd/SwitchSwitchDmio_variable/bert/encoder/layer_5/attention/self/query/bias/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_5/attention/self/query/bias/variable
Î
Dmio_variable/bert/encoder/layer_5/attention/self/key/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*=
	container0.bert/encoder/layer_5/attention/self/key/kernel
Î
Dmio_variable/bert/encoder/layer_5/attention/self/key/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_5/attention/self/key/kernel*
shape:

Z
%Initializer_87/truncated_normal/shapeConst*
dtype0*
valueB"      
Q
$Initializer_87/truncated_normal/meanConst*
valueB
 *    *
dtype0
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
	Assign_87AssignDmio_variable/bert/encoder/layer_5/attention/self/key/kernel/gradientInitializer_87/truncated_normal*
validate_shape(*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_5/attention/self/key/kernel/gradient
Å
Bmio_variable/bert/encoder/layer_5/attention/self/key/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*;
	container.,bert/encoder/layer_5/attention/self/key/bias*
shape:
Å
Bmio_variable/bert/encoder/layer_5/attention/self/key/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*;
	container.,bert/encoder/layer_5/attention/self/key/bias
F
Initializer_88/zerosConst*
valueB*    *
dtype0
ö
	Assign_88AssignBmio_variable/bert/encoder/layer_5/attention/self/key/bias/gradientInitializer_88/zeros*
use_locking(*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_5/attention/self/key/bias/gradient*
validate_shape(
Ų
3cond/bert/encoder/layer_5/attention/self/key/MatMulMatMul-cond/bert/encoder/layer_4/output/StopGradient<cond/bert/encoder/layer_5/attention/self/key/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 
ú
:cond/bert/encoder/layer_5/attention/self/key/MatMul/SwitchSwitchDmio_variable/bert/encoder/layer_5/attention/self/key/kernel/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_5/attention/self/key/kernel/variable
Ķ
4cond/bert/encoder/layer_5/attention/self/key/BiasAddBiasAdd3cond/bert/encoder/layer_5/attention/self/key/MatMul=cond/bert/encoder/layer_5/attention/self/key/BiasAdd/Switch:1*
T0*
data_formatNHWC
÷
;cond/bert/encoder/layer_5/attention/self/key/BiasAdd/SwitchSwitchBmio_variable/bert/encoder/layer_5/attention/self/key/bias/variablecond/pred_id*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_5/attention/self/key/bias/variable
Ō
Fmio_variable/bert/encoder/layer_5/attention/self/value/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_5/attention/self/value/kernel*
shape:

Ō
Fmio_variable/bert/encoder/layer_5/attention/self/value/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_5/attention/self/value/kernel*
shape:

Z
%Initializer_89/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_89/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_89/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_89/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_89/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 

#Initializer_89/truncated_normal/mulMul/Initializer_89/truncated_normal/TruncatedNormal&Initializer_89/truncated_normal/stddev*
T0
z
Initializer_89/truncated_normalAdd#Initializer_89/truncated_normal/mul$Initializer_89/truncated_normal/mean*
T0

	Assign_89AssignFmio_variable/bert/encoder/layer_5/attention/self/value/kernel/gradientInitializer_89/truncated_normal*
use_locking(*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_5/attention/self/value/kernel/gradient*
validate_shape(
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
	Assign_90AssignDmio_variable/bert/encoder/layer_5/attention/self/value/bias/gradientInitializer_90/zeros*
use_locking(*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_5/attention/self/value/bias/gradient*
validate_shape(
Ũ
5cond/bert/encoder/layer_5/attention/self/value/MatMulMatMul-cond/bert/encoder/layer_4/output/StopGradient>cond/bert/encoder/layer_5/attention/self/value/MatMul/Switch:1*
transpose_a( *
transpose_b( *
T0

<cond/bert/encoder/layer_5/attention/self/value/MatMul/SwitchSwitchFmio_variable/bert/encoder/layer_5/attention/self/value/kernel/variablecond/pred_id*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_5/attention/self/value/kernel/variable
Ų
6cond/bert/encoder/layer_5/attention/self/value/BiasAddBiasAdd5cond/bert/encoder/layer_5/attention/self/value/MatMul?cond/bert/encoder/layer_5/attention/self/value/BiasAdd/Switch:1*
T0*
data_formatNHWC
ũ
=cond/bert/encoder/layer_5/attention/self/value/BiasAdd/SwitchSwitchDmio_variable/bert/encoder/layer_5/attention/self/value/bias/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_5/attention/self/value/bias/variable
r
8cond/bert/encoder/layer_5/attention/self/Reshape/shape/1Const^cond/switch_t*
dtype0*
value	B :
r
8cond/bert/encoder/layer_5/attention/self/Reshape/shape/2Const^cond/switch_t*
value	B :*
dtype0
r
8cond/bert/encoder/layer_5/attention/self/Reshape/shape/3Const^cond/switch_t*
value	B : *
dtype0
­
6cond/bert/encoder/layer_5/attention/self/Reshape/shapePack!cond/bert/encoder/strided_slice_28cond/bert/encoder/layer_5/attention/self/Reshape/shape/18cond/bert/encoder/layer_5/attention/self/Reshape/shape/28cond/bert/encoder/layer_5/attention/self/Reshape/shape/3*
T0*

axis *
N
Â
0cond/bert/encoder/layer_5/attention/self/ReshapeReshape6cond/bert/encoder/layer_5/attention/self/query/BiasAdd6cond/bert/encoder/layer_5/attention/self/Reshape/shape*
T0*
Tshape0

7cond/bert/encoder/layer_5/attention/self/transpose/permConst^cond/switch_t*
dtype0*%
valueB"             
Ā
2cond/bert/encoder/layer_5/attention/self/transpose	Transpose0cond/bert/encoder/layer_5/attention/self/Reshape7cond/bert/encoder/layer_5/attention/self/transpose/perm*
T0*
Tperm0
t
:cond/bert/encoder/layer_5/attention/self/Reshape_1/shape/1Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_5/attention/self/Reshape_1/shape/2Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_5/attention/self/Reshape_1/shape/3Const^cond/switch_t*
value	B : *
dtype0
ĩ
8cond/bert/encoder/layer_5/attention/self/Reshape_1/shapePack!cond/bert/encoder/strided_slice_2:cond/bert/encoder/layer_5/attention/self/Reshape_1/shape/1:cond/bert/encoder/layer_5/attention/self/Reshape_1/shape/2:cond/bert/encoder/layer_5/attention/self/Reshape_1/shape/3*

axis *
N*
T0
Ä
2cond/bert/encoder/layer_5/attention/self/Reshape_1Reshape4cond/bert/encoder/layer_5/attention/self/key/BiasAdd8cond/bert/encoder/layer_5/attention/self/Reshape_1/shape*
T0*
Tshape0

9cond/bert/encoder/layer_5/attention/self/transpose_1/permConst^cond/switch_t*%
valueB"             *
dtype0
Æ
4cond/bert/encoder/layer_5/attention/self/transpose_1	Transpose2cond/bert/encoder/layer_5/attention/self/Reshape_19cond/bert/encoder/layer_5/attention/self/transpose_1/perm*
T0*
Tperm0
Ë
/cond/bert/encoder/layer_5/attention/self/MatMulBatchMatMul2cond/bert/encoder/layer_5/attention/self/transpose4cond/bert/encoder/layer_5/attention/self/transpose_1*
adj_x( *
adj_y(*
T0
k
.cond/bert/encoder/layer_5/attention/self/Mul/yConst^cond/switch_t*
dtype0*
valueB
 *ķ5>

,cond/bert/encoder/layer_5/attention/self/MulMul/cond/bert/encoder/layer_5/attention/self/MatMul.cond/bert/encoder/layer_5/attention/self/Mul/y*
T0
u
7cond/bert/encoder/layer_5/attention/self/ExpandDims/dimConst^cond/switch_t*
dtype0*
valueB:
Ļ
3cond/bert/encoder/layer_5/attention/self/ExpandDims
ExpandDimscond/bert/encoder/mul7cond/bert/encoder/layer_5/attention/self/ExpandDims/dim*

Tdim0*
T0
k
.cond/bert/encoder/layer_5/attention/self/sub/xConst^cond/switch_t*
valueB
 *  ?*
dtype0
Ą
,cond/bert/encoder/layer_5/attention/self/subSub.cond/bert/encoder/layer_5/attention/self/sub/x3cond/bert/encoder/layer_5/attention/self/ExpandDims*
T0
m
0cond/bert/encoder/layer_5/attention/self/mul_1/yConst^cond/switch_t*
valueB
 * @Æ*
dtype0

.cond/bert/encoder/layer_5/attention/self/mul_1Mul,cond/bert/encoder/layer_5/attention/self/sub0cond/bert/encoder/layer_5/attention/self/mul_1/y*
T0

,cond/bert/encoder/layer_5/attention/self/addAdd,cond/bert/encoder/layer_5/attention/self/Mul.cond/bert/encoder/layer_5/attention/self/mul_1*
T0
r
0cond/bert/encoder/layer_5/attention/self/SoftmaxSoftmax,cond/bert/encoder/layer_5/attention/self/add*
T0
w
:cond/bert/encoder/layer_5/attention/self/dropout/keep_probConst^cond/switch_t*
valueB
 *fff?*
dtype0

6cond/bert/encoder/layer_5/attention/self/dropout/ShapeShape0cond/bert/encoder/layer_5/attention/self/Softmax*
T0*
out_type0

Ccond/bert/encoder/layer_5/attention/self/dropout/random_uniform/minConst^cond/switch_t*
valueB
 *    *
dtype0

Ccond/bert/encoder/layer_5/attention/self/dropout/random_uniform/maxConst^cond/switch_t*
valueB
 *  ?*
dtype0
Å
Mcond/bert/encoder/layer_5/attention/self/dropout/random_uniform/RandomUniformRandomUniform6cond/bert/encoder/layer_5/attention/self/dropout/Shape*
dtype0*
seed2 *

seed *
T0
Ũ
Ccond/bert/encoder/layer_5/attention/self/dropout/random_uniform/subSubCcond/bert/encoder/layer_5/attention/self/dropout/random_uniform/maxCcond/bert/encoder/layer_5/attention/self/dropout/random_uniform/min*
T0
į
Ccond/bert/encoder/layer_5/attention/self/dropout/random_uniform/mulMulMcond/bert/encoder/layer_5/attention/self/dropout/random_uniform/RandomUniformCcond/bert/encoder/layer_5/attention/self/dropout/random_uniform/sub*
T0
Ų
?cond/bert/encoder/layer_5/attention/self/dropout/random_uniformAddCcond/bert/encoder/layer_5/attention/self/dropout/random_uniform/mulCcond/bert/encoder/layer_5/attention/self/dropout/random_uniform/min*
T0
Á
4cond/bert/encoder/layer_5/attention/self/dropout/addAdd:cond/bert/encoder/layer_5/attention/self/dropout/keep_prob?cond/bert/encoder/layer_5/attention/self/dropout/random_uniform*
T0
~
6cond/bert/encoder/layer_5/attention/self/dropout/FloorFloor4cond/bert/encoder/layer_5/attention/self/dropout/add*
T0
ļ
4cond/bert/encoder/layer_5/attention/self/dropout/divRealDiv0cond/bert/encoder/layer_5/attention/self/Softmax:cond/bert/encoder/layer_5/attention/self/dropout/keep_prob*
T0
˛
4cond/bert/encoder/layer_5/attention/self/dropout/mulMul4cond/bert/encoder/layer_5/attention/self/dropout/div6cond/bert/encoder/layer_5/attention/self/dropout/Floor*
T0
t
:cond/bert/encoder/layer_5/attention/self/Reshape_2/shape/1Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_5/attention/self/Reshape_2/shape/2Const^cond/switch_t*
value	B :*
dtype0
t
:cond/bert/encoder/layer_5/attention/self/Reshape_2/shape/3Const^cond/switch_t*
value	B : *
dtype0
ĩ
8cond/bert/encoder/layer_5/attention/self/Reshape_2/shapePack!cond/bert/encoder/strided_slice_2:cond/bert/encoder/layer_5/attention/self/Reshape_2/shape/1:cond/bert/encoder/layer_5/attention/self/Reshape_2/shape/2:cond/bert/encoder/layer_5/attention/self/Reshape_2/shape/3*
T0*

axis *
N
Æ
2cond/bert/encoder/layer_5/attention/self/Reshape_2Reshape6cond/bert/encoder/layer_5/attention/self/value/BiasAdd8cond/bert/encoder/layer_5/attention/self/Reshape_2/shape*
T0*
Tshape0

9cond/bert/encoder/layer_5/attention/self/transpose_2/permConst^cond/switch_t*%
valueB"             *
dtype0
Æ
4cond/bert/encoder/layer_5/attention/self/transpose_2	Transpose2cond/bert/encoder/layer_5/attention/self/Reshape_29cond/bert/encoder/layer_5/attention/self/transpose_2/perm*
Tperm0*
T0
Ī
1cond/bert/encoder/layer_5/attention/self/MatMul_1BatchMatMul4cond/bert/encoder/layer_5/attention/self/dropout/mul4cond/bert/encoder/layer_5/attention/self/transpose_2*
adj_x( *
adj_y( *
T0

9cond/bert/encoder/layer_5/attention/self/transpose_3/permConst^cond/switch_t*%
valueB"             *
dtype0
Å
4cond/bert/encoder/layer_5/attention/self/transpose_3	Transpose1cond/bert/encoder/layer_5/attention/self/MatMul_19cond/bert/encoder/layer_5/attention/self/transpose_3/perm*
Tperm0*
T0
j
0cond/bert/encoder/layer_5/attention/self/mul_2/yConst^cond/switch_t*
value	B :*
dtype0

.cond/bert/encoder/layer_5/attention/self/mul_2Mul!cond/bert/encoder/strided_slice_20cond/bert/encoder/layer_5/attention/self/mul_2/y*
T0
u
:cond/bert/encoder/layer_5/attention/self/Reshape_3/shape/1Const^cond/switch_t*
value
B :*
dtype0
Ę
8cond/bert/encoder/layer_5/attention/self/Reshape_3/shapePack.cond/bert/encoder/layer_5/attention/self/mul_2:cond/bert/encoder/layer_5/attention/self/Reshape_3/shape/1*

axis *
N*
T0
Ä
2cond/bert/encoder/layer_5/attention/self/Reshape_3Reshape4cond/bert/encoder/layer_5/attention/self/transpose_38cond/bert/encoder/layer_5/attention/self/Reshape_3/shape*
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
Z
%Initializer_91/truncated_normal/shapeConst*
valueB"      *
dtype0
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
/Initializer_91/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_91/truncated_normal/shape*
seed2 *

seed *
T0*
dtype0

#Initializer_91/truncated_normal/mulMul/Initializer_91/truncated_normal/TruncatedNormal&Initializer_91/truncated_normal/stddev*
T0
z
Initializer_91/truncated_normalAdd#Initializer_91/truncated_normal/mul$Initializer_91/truncated_normal/mean*
T0

	Assign_91AssignHmio_variable/bert/encoder/layer_5/attention/output/dense/kernel/gradientInitializer_91/truncated_normal*[
_classQ
OMloc:@mio_variable/bert/encoder/layer_5/attention/output/dense/kernel/gradient*
validate_shape(*
use_locking(*
T0
Í
Fmio_variable/bert/encoder/layer_5/attention/output/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*?
	container20bert/encoder/layer_5/attention/output/dense/bias*
shape:
Í
Fmio_variable/bert/encoder/layer_5/attention/output/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*?
	container20bert/encoder/layer_5/attention/output/dense/bias
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
æ
7cond/bert/encoder/layer_5/attention/output/dense/MatMulMatMul2cond/bert/encoder/layer_5/attention/self/Reshape_3@cond/bert/encoder/layer_5/attention/output/dense/MatMul/Switch:1*
transpose_a( *
transpose_b( *
T0

>cond/bert/encoder/layer_5/attention/output/dense/MatMul/SwitchSwitchHmio_variable/bert/encoder/layer_5/attention/output/dense/kernel/variablecond/pred_id*
T0*[
_classQ
OMloc:@mio_variable/bert/encoder/layer_5/attention/output/dense/kernel/variable
ß
8cond/bert/encoder/layer_5/attention/output/dense/BiasAddBiasAdd7cond/bert/encoder/layer_5/attention/output/dense/MatMulAcond/bert/encoder/layer_5/attention/output/dense/BiasAdd/Switch:1*
T0*
data_formatNHWC

?cond/bert/encoder/layer_5/attention/output/dense/BiasAdd/SwitchSwitchFmio_variable/bert/encoder/layer_5/attention/output/dense/bias/variablecond/pred_id*
T0*Y
_classO
MKloc:@mio_variable/bert/encoder/layer_5/attention/output/dense/bias/variable
y
<cond/bert/encoder/layer_5/attention/output/dropout/keep_probConst^cond/switch_t*
valueB
 *fff?*
dtype0

8cond/bert/encoder/layer_5/attention/output/dropout/ShapeShape8cond/bert/encoder/layer_5/attention/output/dense/BiasAdd*
out_type0*
T0

Econd/bert/encoder/layer_5/attention/output/dropout/random_uniform/minConst^cond/switch_t*
valueB
 *    *
dtype0

Econd/bert/encoder/layer_5/attention/output/dropout/random_uniform/maxConst^cond/switch_t*
dtype0*
valueB
 *  ?
É
Ocond/bert/encoder/layer_5/attention/output/dropout/random_uniform/RandomUniformRandomUniform8cond/bert/encoder/layer_5/attention/output/dropout/Shape*
dtype0*
seed2 *

seed *
T0
ã
Econd/bert/encoder/layer_5/attention/output/dropout/random_uniform/subSubEcond/bert/encoder/layer_5/attention/output/dropout/random_uniform/maxEcond/bert/encoder/layer_5/attention/output/dropout/random_uniform/min*
T0
í
Econd/bert/encoder/layer_5/attention/output/dropout/random_uniform/mulMulOcond/bert/encoder/layer_5/attention/output/dropout/random_uniform/RandomUniformEcond/bert/encoder/layer_5/attention/output/dropout/random_uniform/sub*
T0
ß
Acond/bert/encoder/layer_5/attention/output/dropout/random_uniformAddEcond/bert/encoder/layer_5/attention/output/dropout/random_uniform/mulEcond/bert/encoder/layer_5/attention/output/dropout/random_uniform/min*
T0
Į
6cond/bert/encoder/layer_5/attention/output/dropout/addAdd<cond/bert/encoder/layer_5/attention/output/dropout/keep_probAcond/bert/encoder/layer_5/attention/output/dropout/random_uniform*
T0

8cond/bert/encoder/layer_5/attention/output/dropout/FloorFloor6cond/bert/encoder/layer_5/attention/output/dropout/add*
T0
Â
6cond/bert/encoder/layer_5/attention/output/dropout/divRealDiv8cond/bert/encoder/layer_5/attention/output/dense/BiasAdd<cond/bert/encoder/layer_5/attention/output/dropout/keep_prob*
T0
¸
6cond/bert/encoder/layer_5/attention/output/dropout/mulMul6cond/bert/encoder/layer_5/attention/output/dropout/div8cond/bert/encoder/layer_5/attention/output/dropout/Floor*
T0
Ĩ
.cond/bert/encoder/layer_5/attention/output/addAdd6cond/bert/encoder/layer_5/attention/output/dropout/mul-cond/bert/encoder/layer_4/output/StopGradient*
T0
Õ
Jmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/beta/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64bert/encoder/layer_5/attention/output/LayerNorm/beta*
shape:
Õ
Jmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/beta/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*C
	container64bert/encoder/layer_5/attention/output/LayerNorm/beta*
shape:
F
Initializer_93/zerosConst*
dtype0*
valueB*    
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
Kmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*D
	container75bert/encoder/layer_5/attention/output/LayerNorm/gamma*
shape:
E
Initializer_94/onesConst*
valueB*  ?*
dtype0

	Assign_94AssignKmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/gamma/gradientInitializer_94/ones*
use_locking(*
T0*^
_classT
RPloc:@mio_variable/bert/encoder/layer_5/attention/output/LayerNorm/gamma/gradient*
validate_shape(

Scond/bert/encoder/layer_5/attention/output/LayerNorm/moments/mean/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0
ô
Acond/bert/encoder/layer_5/attention/output/LayerNorm/moments/meanMean.cond/bert/encoder/layer_5/attention/output/addScond/bert/encoder/layer_5/attention/output/LayerNorm/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
Ĩ
Icond/bert/encoder/layer_5/attention/output/LayerNorm/moments/StopGradientStopGradientAcond/bert/encoder/layer_5/attention/output/LayerNorm/moments/mean*
T0
į
Ncond/bert/encoder/layer_5/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference.cond/bert/encoder/layer_5/attention/output/addIcond/bert/encoder/layer_5/attention/output/LayerNorm/moments/StopGradient*
T0

Wcond/bert/encoder/layer_5/attention/output/LayerNorm/moments/variance/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0

Econd/bert/encoder/layer_5/attention/output/LayerNorm/moments/varianceMeanNcond/bert/encoder/layer_5/attention/output/LayerNorm/moments/SquaredDifferenceWcond/bert/encoder/layer_5/attention/output/LayerNorm/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0

Dcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add/yConst^cond/switch_t*
valueB
 *Ėŧ+*
dtype0
ß
Bcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/addAddEcond/bert/encoder/layer_5/attention/output/LayerNorm/moments/varianceDcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add/y*
T0

Dcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/RsqrtRsqrtBcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add*
T0
å
Bcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mulMulDcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/RsqrtKcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul/Switch:1*
T0

Icond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul/SwitchSwitchKmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/gamma/variablecond/pred_id*
T0*^
_classT
RPloc:@mio_variable/bert/encoder/layer_5/attention/output/LayerNorm/gamma/variable
Č
Dcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul_1Mul.cond/bert/encoder/layer_5/attention/output/addBcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul*
T0
Û
Dcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul_2MulAcond/bert/encoder/layer_5/attention/output/LayerNorm/moments/meanBcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul*
T0
å
Bcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/subSubKcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/sub/Switch:1Dcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul_2*
T0

Icond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/sub/SwitchSwitchJmio_variable/bert/encoder/layer_5/attention/output/LayerNorm/beta/variablecond/pred_id*
T0*]
_classS
QOloc:@mio_variable/bert/encoder/layer_5/attention/output/LayerNorm/beta/variable
Ū
Dcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add_1AddDcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul_1Bcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/sub*
T0
Î
Dmio_variable/bert/encoder/layer_5/intermediate/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*=
	container0.bert/encoder/layer_5/intermediate/dense/kernel*
shape:

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
&Initializer_95/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_95/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_95/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0

#Initializer_95/truncated_normal/mulMul/Initializer_95/truncated_normal/TruncatedNormal&Initializer_95/truncated_normal/stddev*
T0
z
Initializer_95/truncated_normalAdd#Initializer_95/truncated_normal/mul$Initializer_95/truncated_normal/mean*
T0

	Assign_95AssignDmio_variable/bert/encoder/layer_5/intermediate/dense/kernel/gradientInitializer_95/truncated_normal*
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
$Initializer_96/zeros/shape_as_tensorConst*
valueB:*
dtype0
G
Initializer_96/zeros/ConstConst*
valueB
 *    *
dtype0
y
Initializer_96/zerosFill$Initializer_96/zeros/shape_as_tensorInitializer_96/zeros/Const*

index_type0*
T0
ö
	Assign_96AssignBmio_variable/bert/encoder/layer_5/intermediate/dense/bias/gradientInitializer_96/zeros*
use_locking(*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_5/intermediate/dense/bias/gradient*
validate_shape(
đ
3cond/bert/encoder/layer_5/intermediate/dense/MatMulMatMulDcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add_1<cond/bert/encoder/layer_5/intermediate/dense/MatMul/Switch:1*
T0*
transpose_a( *
transpose_b( 
ú
:cond/bert/encoder/layer_5/intermediate/dense/MatMul/SwitchSwitchDmio_variable/bert/encoder/layer_5/intermediate/dense/kernel/variablecond/pred_id*
T0*W
_classM
KIloc:@mio_variable/bert/encoder/layer_5/intermediate/dense/kernel/variable
Ķ
4cond/bert/encoder/layer_5/intermediate/dense/BiasAddBiasAdd3cond/bert/encoder/layer_5/intermediate/dense/MatMul=cond/bert/encoder/layer_5/intermediate/dense/BiasAdd/Switch:1*
T0*
data_formatNHWC
÷
;cond/bert/encoder/layer_5/intermediate/dense/BiasAdd/SwitchSwitchBmio_variable/bert/encoder/layer_5/intermediate/dense/bias/variablecond/pred_id*
T0*U
_classK
IGloc:@mio_variable/bert/encoder/layer_5/intermediate/dense/bias/variable
o
2cond/bert/encoder/layer_5/intermediate/dense/Pow/yConst^cond/switch_t*
valueB
 *  @@*
dtype0
Ē
0cond/bert/encoder/layer_5/intermediate/dense/PowPow4cond/bert/encoder/layer_5/intermediate/dense/BiasAdd2cond/bert/encoder/layer_5/intermediate/dense/Pow/y*
T0
o
2cond/bert/encoder/layer_5/intermediate/dense/mul/xConst^cond/switch_t*
valueB
 *'7=*
dtype0
Ļ
0cond/bert/encoder/layer_5/intermediate/dense/mulMul2cond/bert/encoder/layer_5/intermediate/dense/mul/x0cond/bert/encoder/layer_5/intermediate/dense/Pow*
T0
¨
0cond/bert/encoder/layer_5/intermediate/dense/addAdd4cond/bert/encoder/layer_5/intermediate/dense/BiasAdd0cond/bert/encoder/layer_5/intermediate/dense/mul*
T0
q
4cond/bert/encoder/layer_5/intermediate/dense/mul_1/xConst^cond/switch_t*
valueB
 **BL?*
dtype0
Ē
2cond/bert/encoder/layer_5/intermediate/dense/mul_1Mul4cond/bert/encoder/layer_5/intermediate/dense/mul_1/x0cond/bert/encoder/layer_5/intermediate/dense/add*
T0
v
1cond/bert/encoder/layer_5/intermediate/dense/TanhTanh2cond/bert/encoder/layer_5/intermediate/dense/mul_1*
T0
q
4cond/bert/encoder/layer_5/intermediate/dense/add_1/xConst^cond/switch_t*
valueB
 *  ?*
dtype0
Ģ
2cond/bert/encoder/layer_5/intermediate/dense/add_1Add4cond/bert/encoder/layer_5/intermediate/dense/add_1/x1cond/bert/encoder/layer_5/intermediate/dense/Tanh*
T0
q
4cond/bert/encoder/layer_5/intermediate/dense/mul_2/xConst^cond/switch_t*
valueB
 *   ?*
dtype0
Ŧ
2cond/bert/encoder/layer_5/intermediate/dense/mul_2Mul4cond/bert/encoder/layer_5/intermediate/dense/mul_2/x2cond/bert/encoder/layer_5/intermediate/dense/add_1*
T0
Ŧ
2cond/bert/encoder/layer_5/intermediate/dense/mul_3Mul4cond/bert/encoder/layer_5/intermediate/dense/BiasAdd2cond/bert/encoder/layer_5/intermediate/dense/mul_2*
T0
Â
>mio_variable/bert/encoder/layer_5/output/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*7
	container*(bert/encoder/layer_5/output/dense/kernel
Â
>mio_variable/bert/encoder/layer_5/output/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*7
	container*(bert/encoder/layer_5/output/dense/kernel*
shape:

Z
%Initializer_97/truncated_normal/shapeConst*
valueB"      *
dtype0
Q
$Initializer_97/truncated_normal/meanConst*
valueB
 *    *
dtype0
S
&Initializer_97/truncated_normal/stddevConst*
valueB
 *
×Ŗ<*
dtype0

/Initializer_97/truncated_normal/TruncatedNormalTruncatedNormal%Initializer_97/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 
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
	Assign_98Assign<mio_variable/bert/encoder/layer_5/output/dense/bias/gradientInitializer_98/zeros*
use_locking(*
T0*O
_classE
CAloc:@mio_variable/bert/encoder/layer_5/output/dense/bias/gradient*
validate_shape(
Ō
-cond/bert/encoder/layer_5/output/dense/MatMulMatMul2cond/bert/encoder/layer_5/intermediate/dense/mul_36cond/bert/encoder/layer_5/output/dense/MatMul/Switch:1*
transpose_b( *
T0*
transpose_a( 
č
4cond/bert/encoder/layer_5/output/dense/MatMul/SwitchSwitch>mio_variable/bert/encoder/layer_5/output/dense/kernel/variablecond/pred_id*
T0*Q
_classG
ECloc:@mio_variable/bert/encoder/layer_5/output/dense/kernel/variable
Á
.cond/bert/encoder/layer_5/output/dense/BiasAddBiasAdd-cond/bert/encoder/layer_5/output/dense/MatMul7cond/bert/encoder/layer_5/output/dense/BiasAdd/Switch:1*
T0*
data_formatNHWC
å
5cond/bert/encoder/layer_5/output/dense/BiasAdd/SwitchSwitch<mio_variable/bert/encoder/layer_5/output/dense/bias/variablecond/pred_id*
T0*O
_classE
CAloc:@mio_variable/bert/encoder/layer_5/output/dense/bias/variable
o
2cond/bert/encoder/layer_5/output/dropout/keep_probConst^cond/switch_t*
valueB
 *fff?*
dtype0

.cond/bert/encoder/layer_5/output/dropout/ShapeShape.cond/bert/encoder/layer_5/output/dense/BiasAdd*
out_type0*
T0
x
;cond/bert/encoder/layer_5/output/dropout/random_uniform/minConst^cond/switch_t*
valueB
 *    *
dtype0
x
;cond/bert/encoder/layer_5/output/dropout/random_uniform/maxConst^cond/switch_t*
valueB
 *  ?*
dtype0
ĩ
Econd/bert/encoder/layer_5/output/dropout/random_uniform/RandomUniformRandomUniform.cond/bert/encoder/layer_5/output/dropout/Shape*
T0*
dtype0*
seed2 *

seed 
Å
;cond/bert/encoder/layer_5/output/dropout/random_uniform/subSub;cond/bert/encoder/layer_5/output/dropout/random_uniform/max;cond/bert/encoder/layer_5/output/dropout/random_uniform/min*
T0
Ī
;cond/bert/encoder/layer_5/output/dropout/random_uniform/mulMulEcond/bert/encoder/layer_5/output/dropout/random_uniform/RandomUniform;cond/bert/encoder/layer_5/output/dropout/random_uniform/sub*
T0
Á
7cond/bert/encoder/layer_5/output/dropout/random_uniformAdd;cond/bert/encoder/layer_5/output/dropout/random_uniform/mul;cond/bert/encoder/layer_5/output/dropout/random_uniform/min*
T0
Š
,cond/bert/encoder/layer_5/output/dropout/addAdd2cond/bert/encoder/layer_5/output/dropout/keep_prob7cond/bert/encoder/layer_5/output/dropout/random_uniform*
T0
n
.cond/bert/encoder/layer_5/output/dropout/FloorFloor,cond/bert/encoder/layer_5/output/dropout/add*
T0
¤
,cond/bert/encoder/layer_5/output/dropout/divRealDiv.cond/bert/encoder/layer_5/output/dense/BiasAdd2cond/bert/encoder/layer_5/output/dropout/keep_prob*
T0

,cond/bert/encoder/layer_5/output/dropout/mulMul,cond/bert/encoder/layer_5/output/dropout/div.cond/bert/encoder/layer_5/output/dropout/Floor*
T0
¨
$cond/bert/encoder/layer_5/output/addAdd,cond/bert/encoder/layer_5/output/dropout/mulDcond/bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add_1*
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
	Assign_99Assign@mio_variable/bert/encoder/layer_5/output/LayerNorm/beta/gradientInitializer_99/zeros*
validate_shape(*
use_locking(*
T0*S
_classI
GEloc:@mio_variable/bert/encoder/layer_5/output/LayerNorm/beta/gradient
Ã
Amio_variable/bert/encoder/layer_5/output/LayerNorm/gamma/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+bert/encoder/layer_5/output/LayerNorm/gamma*
shape:
Ã
Amio_variable/bert/encoder/layer_5/output/LayerNorm/gamma/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*:
	container-+bert/encoder/layer_5/output/LayerNorm/gamma*
shape:
F
Initializer_100/onesConst*
valueB*  ?*
dtype0
õ

Assign_100AssignAmio_variable/bert/encoder/layer_5/output/LayerNorm/gamma/gradientInitializer_100/ones*T
_classJ
HFloc:@mio_variable/bert/encoder/layer_5/output/LayerNorm/gamma/gradient*
validate_shape(*
use_locking(*
T0

Icond/bert/encoder/layer_5/output/LayerNorm/moments/mean/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0
Ö
7cond/bert/encoder/layer_5/output/LayerNorm/moments/meanMean$cond/bert/encoder/layer_5/output/addIcond/bert/encoder/layer_5/output/LayerNorm/moments/mean/reduction_indices*
T0*

Tidx0*
	keep_dims(

?cond/bert/encoder/layer_5/output/LayerNorm/moments/StopGradientStopGradient7cond/bert/encoder/layer_5/output/LayerNorm/moments/mean*
T0
É
Dcond/bert/encoder/layer_5/output/LayerNorm/moments/SquaredDifferenceSquaredDifference$cond/bert/encoder/layer_5/output/add?cond/bert/encoder/layer_5/output/LayerNorm/moments/StopGradient*
T0

Mcond/bert/encoder/layer_5/output/LayerNorm/moments/variance/reduction_indicesConst^cond/switch_t*
valueB:*
dtype0
ū
;cond/bert/encoder/layer_5/output/LayerNorm/moments/varianceMeanDcond/bert/encoder/layer_5/output/LayerNorm/moments/SquaredDifferenceMcond/bert/encoder/layer_5/output/LayerNorm/moments/variance/reduction_indices*
T0*

Tidx0*
	keep_dims(
w
:cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/add/yConst^cond/switch_t*
valueB
 *Ėŧ+*
dtype0
Á
8cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/addAdd;cond/bert/encoder/layer_5/output/LayerNorm/moments/variance:cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/add/y*
T0

:cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/RsqrtRsqrt8cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/add*
T0
Į
8cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/mulMul:cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/RsqrtAcond/bert/encoder/layer_5/output/LayerNorm/batchnorm/mul/Switch:1*
T0
ų
?cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/mul/SwitchSwitchAmio_variable/bert/encoder/layer_5/output/LayerNorm/gamma/variablecond/pred_id*
T0*T
_classJ
HFloc:@mio_variable/bert/encoder/layer_5/output/LayerNorm/gamma/variable
Ē
:cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/mul_1Mul$cond/bert/encoder/layer_5/output/add8cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/mul*
T0
Ŋ
:cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/mul_2Mul7cond/bert/encoder/layer_5/output/LayerNorm/moments/mean8cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/mul*
T0
Į
8cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/subSubAcond/bert/encoder/layer_5/output/LayerNorm/batchnorm/sub/Switch:1:cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/mul_2*
T0
÷
?cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/sub/SwitchSwitch@mio_variable/bert/encoder/layer_5/output/LayerNorm/beta/variablecond/pred_id*
T0*S
_classI
GEloc:@mio_variable/bert/encoder/layer_5/output/LayerNorm/beta/variable
Ā
:cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/add_1Add:cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/mul_18cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/sub*
T0
w
cond/bert/encoder/Shape_3Shape:cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
e
'cond/bert/encoder/strided_slice_3/stackConst^cond/switch_t*
dtype0*
valueB: 
g
)cond/bert/encoder/strided_slice_3/stack_1Const^cond/switch_t*
valueB:*
dtype0
g
)cond/bert/encoder/strided_slice_3/stack_2Const^cond/switch_t*
valueB:*
dtype0
Å
!cond/bert/encoder/strided_slice_3StridedSlicecond/bert/encoder/Shape_3'cond/bert/encoder/strided_slice_3/stack)cond/bert/encoder/strided_slice_3/stack_1)cond/bert/encoder/strided_slice_3/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
]
#cond/bert/encoder/Reshape_2/shape/1Const^cond/switch_t*
value	B :*
dtype0
^
#cond/bert/encoder/Reshape_2/shape/2Const^cond/switch_t*
value
B :*
dtype0
´
!cond/bert/encoder/Reshape_2/shapePack!cond/bert/encoder/strided_slice_2#cond/bert/encoder/Reshape_2/shape/1#cond/bert/encoder/Reshape_2/shape/2*

axis *
N*
T0

cond/bert/encoder/Reshape_2Reshape:cond/bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1!cond/bert/encoder/Reshape_2/shape*
T0*
Tshape0
w
cond/bert/encoder/Shape_4Shape:cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
e
'cond/bert/encoder/strided_slice_4/stackConst^cond/switch_t*
valueB: *
dtype0
g
)cond/bert/encoder/strided_slice_4/stack_1Const^cond/switch_t*
valueB:*
dtype0
g
)cond/bert/encoder/strided_slice_4/stack_2Const^cond/switch_t*
dtype0*
valueB:
Å
!cond/bert/encoder/strided_slice_4StridedSlicecond/bert/encoder/Shape_4'cond/bert/encoder/strided_slice_4/stack)cond/bert/encoder/strided_slice_4/stack_1)cond/bert/encoder/strided_slice_4/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0
]
#cond/bert/encoder/Reshape_3/shape/1Const^cond/switch_t*
value	B :*
dtype0
^
#cond/bert/encoder/Reshape_3/shape/2Const^cond/switch_t*
dtype0*
value
B :
´
!cond/bert/encoder/Reshape_3/shapePack!cond/bert/encoder/strided_slice_2#cond/bert/encoder/Reshape_3/shape/1#cond/bert/encoder/Reshape_3/shape/2*
T0*

axis *
N

cond/bert/encoder/Reshape_3Reshape:cond/bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1!cond/bert/encoder/Reshape_3/shape*
T0*
Tshape0
w
cond/bert/encoder/Shape_5Shape:cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
e
'cond/bert/encoder/strided_slice_5/stackConst^cond/switch_t*
valueB: *
dtype0
g
)cond/bert/encoder/strided_slice_5/stack_1Const^cond/switch_t*
valueB:*
dtype0
g
)cond/bert/encoder/strided_slice_5/stack_2Const^cond/switch_t*
valueB:*
dtype0
Å
!cond/bert/encoder/strided_slice_5StridedSlicecond/bert/encoder/Shape_5'cond/bert/encoder/strided_slice_5/stack)cond/bert/encoder/strided_slice_5/stack_1)cond/bert/encoder/strided_slice_5/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
]
#cond/bert/encoder/Reshape_4/shape/1Const^cond/switch_t*
value	B :*
dtype0
^
#cond/bert/encoder/Reshape_4/shape/2Const^cond/switch_t*
value
B :*
dtype0
´
!cond/bert/encoder/Reshape_4/shapePack!cond/bert/encoder/strided_slice_2#cond/bert/encoder/Reshape_4/shape/1#cond/bert/encoder/Reshape_4/shape/2*
T0*

axis *
N

cond/bert/encoder/Reshape_4Reshape:cond/bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1!cond/bert/encoder/Reshape_4/shape*
T0*
Tshape0
w
cond/bert/encoder/Shape_6Shape:cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
e
'cond/bert/encoder/strided_slice_6/stackConst^cond/switch_t*
valueB: *
dtype0
g
)cond/bert/encoder/strided_slice_6/stack_1Const^cond/switch_t*
valueB:*
dtype0
g
)cond/bert/encoder/strided_slice_6/stack_2Const^cond/switch_t*
valueB:*
dtype0
Å
!cond/bert/encoder/strided_slice_6StridedSlicecond/bert/encoder/Shape_6'cond/bert/encoder/strided_slice_6/stack)cond/bert/encoder/strided_slice_6/stack_1)cond/bert/encoder/strided_slice_6/stack_2*
end_mask *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
]
#cond/bert/encoder/Reshape_5/shape/1Const^cond/switch_t*
value	B :*
dtype0
^
#cond/bert/encoder/Reshape_5/shape/2Const^cond/switch_t*
value
B :*
dtype0
´
!cond/bert/encoder/Reshape_5/shapePack!cond/bert/encoder/strided_slice_2#cond/bert/encoder/Reshape_5/shape/1#cond/bert/encoder/Reshape_5/shape/2*
N*
T0*

axis 

cond/bert/encoder/Reshape_5Reshape:cond/bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1!cond/bert/encoder/Reshape_5/shape*
T0*
Tshape0
j
cond/bert/encoder/Shape_7Shape-cond/bert/encoder/layer_4/output/StopGradient*
T0*
out_type0
e
'cond/bert/encoder/strided_slice_7/stackConst^cond/switch_t*
valueB: *
dtype0
g
)cond/bert/encoder/strided_slice_7/stack_1Const^cond/switch_t*
valueB:*
dtype0
g
)cond/bert/encoder/strided_slice_7/stack_2Const^cond/switch_t*
valueB:*
dtype0
Å
!cond/bert/encoder/strided_slice_7StridedSlicecond/bert/encoder/Shape_7'cond/bert/encoder/strided_slice_7/stack)cond/bert/encoder/strided_slice_7/stack_1)cond/bert/encoder/strided_slice_7/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
]
#cond/bert/encoder/Reshape_6/shape/1Const^cond/switch_t*
value	B :*
dtype0
^
#cond/bert/encoder/Reshape_6/shape/2Const^cond/switch_t*
value
B :*
dtype0
´
!cond/bert/encoder/Reshape_6/shapePack!cond/bert/encoder/strided_slice_2#cond/bert/encoder/Reshape_6/shape/1#cond/bert/encoder/Reshape_6/shape/2*
T0*

axis *
N

cond/bert/encoder/Reshape_6Reshape-cond/bert/encoder/layer_4/output/StopGradient!cond/bert/encoder/Reshape_6/shape*
T0*
Tshape0
w
cond/bert/encoder/Shape_8Shape:cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/add_1*
T0*
out_type0
e
'cond/bert/encoder/strided_slice_8/stackConst^cond/switch_t*
dtype0*
valueB: 
g
)cond/bert/encoder/strided_slice_8/stack_1Const^cond/switch_t*
valueB:*
dtype0
g
)cond/bert/encoder/strided_slice_8/stack_2Const^cond/switch_t*
valueB:*
dtype0
Å
!cond/bert/encoder/strided_slice_8StridedSlicecond/bert/encoder/Shape_8'cond/bert/encoder/strided_slice_8/stack)cond/bert/encoder/strided_slice_8/stack_1)cond/bert/encoder/strided_slice_8/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask
]
#cond/bert/encoder/Reshape_7/shape/1Const^cond/switch_t*
value	B :*
dtype0
^
#cond/bert/encoder/Reshape_7/shape/2Const^cond/switch_t*
value
B :*
dtype0
´
!cond/bert/encoder/Reshape_7/shapePack!cond/bert/encoder/strided_slice_2#cond/bert/encoder/Reshape_7/shape/1#cond/bert/encoder/Reshape_7/shape/2*

axis *
N*
T0

cond/bert/encoder/Reshape_7Reshape:cond/bert/encoder/layer_5/output/LayerNorm/batchnorm/add_1!cond/bert/encoder/Reshape_7/shape*
T0*
Tshape0
m
$cond/bert/pooler/strided_slice/stackConst^cond/switch_t*!
valueB"            *
dtype0
o
&cond/bert/pooler/strided_slice/stack_1Const^cond/switch_t*!
valueB"           *
dtype0
o
&cond/bert/pooler/strided_slice/stack_2Const^cond/switch_t*
dtype0*!
valueB"         
ģ
cond/bert/pooler/strided_sliceStridedSlicecond/bert/encoder/Reshape_7$cond/bert/pooler/strided_slice/stack&cond/bert/pooler/strided_slice/stack_1&cond/bert/pooler/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
c
cond/bert/pooler/SqueezeSqueezecond/bert/pooler/strided_slice*
squeeze_dims
*
T0
ĸ
.mio_variable/bert/pooler/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*'
	containerbert/pooler/dense/kernel
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

Assign_101Assign.mio_variable/bert/pooler/dense/kernel/gradient Initializer_101/truncated_normal*
T0*A
_class7
53loc:@mio_variable/bert/pooler/dense/kernel/gradient*
validate_shape(*
use_locking(

,mio_variable/bert/pooler/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*%
	containerbert/pooler/dense/bias

,mio_variable/bert/pooler/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerbert/pooler/dense/bias*
shape:
G
Initializer_102/zerosConst*
dtype0*
valueB*    
Ė

Assign_102Assign,mio_variable/bert/pooler/dense/bias/gradientInitializer_102/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/bert/pooler/dense/bias/gradient*
validate_shape(

cond/bert/pooler/dense/MatMulMatMulcond/bert/pooler/Squeeze&cond/bert/pooler/dense/MatMul/Switch:1*
transpose_a( *
transpose_b( *
T0
¸
$cond/bert/pooler/dense/MatMul/SwitchSwitch.mio_variable/bert/pooler/dense/kernel/variablecond/pred_id*
T0*A
_class7
53loc:@mio_variable/bert/pooler/dense/kernel/variable

cond/bert/pooler/dense/BiasAddBiasAddcond/bert/pooler/dense/MatMul'cond/bert/pooler/dense/BiasAdd/Switch:1*
T0*
data_formatNHWC
ĩ
%cond/bert/pooler/dense/BiasAdd/SwitchSwitch,mio_variable/bert/pooler/dense/bias/variablecond/pred_id*
T0*?
_class5
31loc:@mio_variable/bert/pooler/dense/bias/variable
L
cond/bert/pooler/dense/TanhTanhcond/bert/pooler/dense/BiasAdd*
T0
c
cond/strided_slice_1/stackConst^cond/switch_t*!
valueB"            *
dtype0
e
cond/strided_slice_1/stack_1Const^cond/switch_t*!
valueB"           *
dtype0
e
cond/strided_slice_1/stack_2Const^cond/switch_t*!
valueB"         *
dtype0

cond/strided_slice_1StridedSlicecond/bert/encoder/Reshape_7cond/strided_slice_1/stackcond/strided_slice_1/stack_1cond/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask

cond/Switch_1Switch)mio_embeddings/bert_id_embedding/variablecond/pred_id*
T0*<
_class2
0.loc:@mio_embeddings/bert_id_embedding/variable
J

cond/MergeMergecond/Switch_1cond/strided_slice_1*
T0*
N
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
concat/values_3GatherV2%mio_embeddings/pid_embedding/variableCastconcat/values_3/axis*
Tindices0*
Tparams0*
Taxis0
>
concat/values_4/axisConst*
value	B : *
dtype0

concat/values_4GatherV2%mio_embeddings/aid_embedding/variableCastconcat/values_4/axis*
Tindices0*
Tparams0*
Taxis0
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
concat/values_6GatherV2%mio_embeddings/did_embedding/variableCastconcat/values_6/axis*
Tindices0*
Tparams0*
Taxis0
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
concatConcatV2concat/values_0&mio_embeddings/c_id_embedding/variable(mio_embeddings/c_info_embedding/variableconcat/values_3concat/values_4concat/values_5concat/values_6concat/values_7/mio_embeddings/comment_genre_embedding/variable0mio_embeddings/comment_length_embedding/variableconcat/axis*

Tidx0*
T0*
N

@
concat_1/axisConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
U
concat_1ConcatV2concat
cond/Mergeconcat_1/axis*
N*

Tidx0*
T0
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
,Initializer_103/random_uniform/RandomUniformRandomUniform$Initializer_103/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
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

Assign_104Assign+mio_variable/expand_xtr/dense/bias/gradientInitializer_104/zeros*
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
/mio_variable/expand_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*(
	containerexpand_xtr/dense_1/kernel*
shape:

¤
/mio_variable/expand_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*(
	containerexpand_xtr/dense_1/kernel
Y
$Initializer_105/random_uniform/shapeConst*
valueB"      *
dtype0
O
"Initializer_105/random_uniform/minConst*
valueB
 *   ž*
dtype0
O
"Initializer_105/random_uniform/maxConst*
valueB
 *   >*
dtype0

,Initializer_105/random_uniform/RandomUniformRandomUniform$Initializer_105/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
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

Assign_105Assign/mio_variable/expand_xtr/dense_1/kernel/gradientInitializer_105/random_uniform*
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
"expand_xtr/dense_1/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍĖL>
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
,Initializer_107/random_uniform/RandomUniformRandomUniform$Initializer_107/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
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

Assign_107Assign/mio_variable/expand_xtr/dense_2/kernel/gradientInitializer_107/random_uniform*
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
 *ÍĖL>*
dtype0
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
,Initializer_109/random_uniform/RandomUniformRandomUniform$Initializer_109/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
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

Assign_110Assign-mio_variable/expand_xtr/dense_3/bias/gradientInitializer_110/zeros*
validate_shape(*
use_locking(*
T0*@
_class6
42loc:@mio_variable/expand_xtr/dense_3/bias/gradient
Ą
expand_xtr/dense_3/MatMulMatMulexpand_xtr/dense_2/LeakyRelu/mio_variable/expand_xtr/dense_3/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
dtype0*
seed2 *

seed *
T0
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

Assign_112Assign)mio_variable/like_xtr/dense/bias/gradientInitializer_112/zeros*
use_locking(*
T0*<
_class2
0.loc:@mio_variable/like_xtr/dense/bias/gradient*
validate_shape(

like_xtr/dense/MatMulMatMulconcat_1+mio_variable/like_xtr/dense/kernel/variable*
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
H
like_xtr/dropout/IdentityIdentitylike_xtr/dense/LeakyRelu*
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
Y
$Initializer_113/random_uniform/shapeConst*
valueB"      *
dtype0
O
"Initializer_113/random_uniform/minConst*
valueB
 *   ž*
dtype0
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

Assign_113Assign-mio_variable/like_xtr/dense_1/kernel/gradientInitializer_113/random_uniform*
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
+mio_variable/like_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerlike_xtr/dense_1/bias*
shape:
G
Initializer_114/zerosConst*
valueB*    *
dtype0
Ę

Assign_114Assign+mio_variable/like_xtr/dense_1/bias/gradientInitializer_114/zeros*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@mio_variable/like_xtr/dense_1/bias/gradient

like_xtr/dense_1/MatMulMatMullike_xtr/dropout/Identity-mio_variable/like_xtr/dense_1/kernel/variable*
transpose_b( *
T0*
transpose_a( 
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
-mio_variable/like_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerlike_xtr/dense_2/kernel*
shape:	@
Y
$Initializer_115/random_uniform/shapeConst*
valueB"   @   *
dtype0
O
"Initializer_115/random_uniform/minConst*
valueB
 *ķ5ž*
dtype0
O
"Initializer_115/random_uniform/maxConst*
valueB
 *ķ5>*
dtype0
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
Initializer_116/zerosConst*
dtype0*
valueB@*    
Ę

Assign_116Assign+mio_variable/like_xtr/dense_2/bias/gradientInitializer_116/zeros*
use_locking(*
T0*>
_class4
20loc:@mio_variable/like_xtr/dense_2/bias/gradient*
validate_shape(

like_xtr/dense_2/MatMulMatMullike_xtr/dropout_1/Identity-mio_variable/like_xtr/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
Y
$Initializer_117/random_uniform/shapeConst*
valueB"@      *
dtype0
O
"Initializer_117/random_uniform/minConst*
dtype0*
valueB
 *ž
O
"Initializer_117/random_uniform/maxConst*
dtype0*
valueB
 *>
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

Assign_117Assign-mio_variable/like_xtr/dense_3/kernel/gradientInitializer_117/random_uniform*
validate_shape(*
use_locking(*
T0*@
_class6
42loc:@mio_variable/like_xtr/dense_3/kernel/gradient

+mio_variable/like_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containerlike_xtr/dense_3/bias

+mio_variable/like_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containerlike_xtr/dense_3/bias
F
Initializer_118/zerosConst*
dtype0*
valueB*    
Ę

Assign_118Assign+mio_variable/like_xtr/dense_3/bias/gradientInitializer_118/zeros*
T0*>
_class4
20loc:@mio_variable/like_xtr/dense_3/bias/gradient*
validate_shape(*
use_locking(
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

Assign_119Assign,mio_variable/reply_xtr/dense/kernel/gradientInitializer_119/random_uniform*
T0*?
_class5
31loc:@mio_variable/reply_xtr/dense/kernel/gradient*
validate_shape(*
use_locking(

*mio_variable/reply_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerreply_xtr/dense/bias*
shape:

*mio_variable/reply_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*#
	containerreply_xtr/dense/bias*
shape:
G
Initializer_120/zerosConst*
valueB*    *
dtype0
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
reply_xtr/dense/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍĖL>
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
valueB"      *
dtype0
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
,Initializer_121/random_uniform/RandomUniformRandomUniform$Initializer_121/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
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
,mio_variable/reply_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*%
	containerreply_xtr/dense_1/bias

,mio_variable/reply_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*%
	containerreply_xtr/dense_1/bias*
shape:
G
Initializer_122/zerosConst*
dtype0*
valueB*    
Ė

Assign_122Assign,mio_variable/reply_xtr/dense_1/bias/gradientInitializer_122/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/reply_xtr/dense_1/bias/gradient*
validate_shape(
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
,Initializer_123/random_uniform/RandomUniformRandomUniform$Initializer_123/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
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
Initializer_124/zerosConst*
dtype0*
valueB@*    
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
.mio_variable/reply_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*'
	containerreply_xtr/dense_3/kernel*
shape
:@
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

Assign_125Assign.mio_variable/reply_xtr/dense_3/kernel/gradientInitializer_125/random_uniform*
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
F
Initializer_126/zerosConst*
valueB*    *
dtype0
Ė

Assign_126Assign,mio_variable/reply_xtr/dense_3/bias/gradientInitializer_126/zeros*
use_locking(*
T0*?
_class5
31loc:@mio_variable/reply_xtr/dense_3/bias/gradient*
validate_shape(

reply_xtr/dense_3/MatMulMatMulreply_xtr/dense_2/LeakyRelu.mio_variable/reply_xtr/dense_3/kernel/variable*
transpose_b( *
T0*
transpose_a( 
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

Assign_127Assign+mio_variable/copy_xtr/dense/kernel/gradientInitializer_127/random_uniform*
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
G
Initializer_128/zerosConst*
valueB*    *
dtype0
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
copy_xtr/dense/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍĖL>
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

seed *
T0*
dtype0*
seed2 
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
+mio_variable/copy_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense_1/bias*
shape:
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

seed *
T0*
dtype0*
seed2 
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

Assign_131Assign-mio_variable/copy_xtr/dense_2/kernel/gradientInitializer_131/random_uniform*
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
-mio_variable/copy_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*&
	containercopy_xtr/dense_3/kernel
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
,Initializer_133/random_uniform/RandomUniformRandomUniform$Initializer_133/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
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
+mio_variable/copy_xtr/dense_3/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense_3/bias*
shape:

+mio_variable/copy_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*$
	containercopy_xtr/dense_3/bias*
shape:
F
Initializer_134/zerosConst*
valueB*    *
dtype0
Ę

Assign_134Assign+mio_variable/copy_xtr/dense_3/bias/gradientInitializer_134/zeros*
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
,Initializer_135/random_uniform/RandomUniformRandomUniform$Initializer_135/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
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

Assign_135Assign,mio_variable/share_xtr/dense/kernel/gradientInitializer_135/random_uniform*
T0*?
_class5
31loc:@mio_variable/share_xtr/dense/kernel/gradient*
validate_shape(*
use_locking(

*mio_variable/share_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*#
	containershare_xtr/dense/bias*
shape:

*mio_variable/share_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*#
	containershare_xtr/dense/bias*
shape:
G
Initializer_136/zerosConst*
valueB*    *
dtype0
Č

Assign_136Assign*mio_variable/share_xtr/dense/bias/gradientInitializer_136/zeros*
validate_shape(*
use_locking(*
T0*=
_class3
1/loc:@mio_variable/share_xtr/dense/bias/gradient

share_xtr/dense/MatMulMatMulconcat_1,mio_variable/share_xtr/dense/kernel/variable*
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
J
share_xtr/dropout/IdentityIdentityshare_xtr/dense/LeakyRelu*
T0
ĸ
.mio_variable/share_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*'
	containershare_xtr/dense_1/kernel
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
"Initializer_137/random_uniform/minConst*
dtype0*
valueB
 *   ž
O
"Initializer_137/random_uniform/maxConst*
valueB
 *   >*
dtype0

,Initializer_137/random_uniform/RandomUniformRandomUniform$Initializer_137/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
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
,mio_variable/share_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*%
	containershare_xtr/dense_1/bias
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
transpose_b( *
T0*
transpose_a( 

share_xtr/dense_1/BiasAddBiasAddshare_xtr/dense_1/MatMul,mio_variable/share_xtr/dense_1/bias/variable*
data_formatNHWC*
T0
N
!share_xtr/dense_1/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍĖL>
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
,Initializer_139/random_uniform/RandomUniformRandomUniform$Initializer_139/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
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
,mio_variable/share_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*%
	containershare_xtr/dense_2/bias*
shape:@

,mio_variable/share_xtr/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*%
	containershare_xtr/dense_2/bias
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
Y
$Initializer_141/random_uniform/shapeConst*
dtype0*
valueB"@      
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
,Initializer_141/random_uniform/RandomUniformRandomUniform$Initializer_141/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
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

Assign_141Assign.mio_variable/share_xtr/dense_3/kernel/gradientInitializer_141/random_uniform*
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
,mio_variable/share_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*%
	containershare_xtr/dense_3/bias
F
Initializer_142/zerosConst*
valueB*    *
dtype0
Ė

Assign_142Assign,mio_variable/share_xtr/dense_3/bias/gradientInitializer_142/zeros*
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
°
¤
/mio_variable/audience_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*(
	containeraudience_xtr/dense/kernel
Y
$Initializer_143/random_uniform/shapeConst*
valueB"°     *
dtype0
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

Assign_143Assign/mio_variable/audience_xtr/dense/kernel/gradientInitializer_143/random_uniform*
T0*B
_class8
64loc:@mio_variable/audience_xtr/dense/kernel/gradient*
validate_shape(*
use_locking(

-mio_variable/audience_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*&
	containeraudience_xtr/dense/bias

-mio_variable/audience_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containeraudience_xtr/dense/bias*
shape:
G
Initializer_144/zerosConst*
valueB*    *
dtype0
Î

Assign_144Assign-mio_variable/audience_xtr/dense/bias/gradientInitializer_144/zeros*
use_locking(*
T0*@
_class6
42loc:@mio_variable/audience_xtr/dense/bias/gradient*
validate_shape(

audience_xtr/dense/MatMulMatMulconcat_1/mio_variable/audience_xtr/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
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
Y
$Initializer_145/random_uniform/shapeConst*
valueB"      *
dtype0
O
"Initializer_145/random_uniform/minConst*
dtype0*
valueB
 *   ž
O
"Initializer_145/random_uniform/maxConst*
valueB
 *   >*
dtype0
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

Assign_145Assign1mio_variable/audience_xtr/dense_1/kernel/gradientInitializer_145/random_uniform*
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
G
Initializer_146/zerosConst*
valueB*    *
dtype0
Ō

Assign_146Assign/mio_variable/audience_xtr/dense_1/bias/gradientInitializer_146/zeros*
T0*B
_class8
64loc:@mio_variable/audience_xtr/dense_1/bias/gradient*
validate_shape(*
use_locking(
Ļ
audience_xtr/dense_1/MatMulMatMulaudience_xtr/dropout/Identity1mio_variable/audience_xtr/dense_1/kernel/variable*
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
Y
$Initializer_147/random_uniform/shapeConst*
dtype0*
valueB"   @   
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
,Initializer_147/random_uniform/RandomUniformRandomUniform$Initializer_147/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
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

Assign_147Assign1mio_variable/audience_xtr/dense_2/kernel/gradientInitializer_147/random_uniform*
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
,Initializer_149/random_uniform/RandomUniformRandomUniform$Initializer_149/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
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
Initializer_150/zerosConst*
valueB*    *
dtype0
Ō

Assign_150Assign/mio_variable/audience_xtr/dense_3/bias/gradientInitializer_150/zeros*B
_class8
64loc:@mio_variable/audience_xtr/dense_3/bias/gradient*
validate_shape(*
use_locking(*
T0
§
audience_xtr/dense_3/MatMulMatMulaudience_xtr/dense_2/LeakyRelu1mio_variable/audience_xtr/dense_3/kernel/variable*
transpose_a( *
transpose_b( *
T0

audience_xtr/dense_3/BiasAddBiasAddaudience_xtr/dense_3/MatMul/mio_variable/audience_xtr/dense_3/bias/variable*
data_formatNHWC*
T0
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
valueB"°     *
dtype0
O
"Initializer_151/random_uniform/minConst*
valueB
 *ÃĐŊ*
dtype0
O
"Initializer_151/random_uniform/maxConst*
dtype0*
valueB
 *ÃĐ=
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
6mio_variable/continuous_expand_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*/
	container" continuous_expand_xtr/dense/bias
­
6mio_variable/continuous_expand_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*/
	container" continuous_expand_xtr/dense/bias
G
Initializer_152/zerosConst*
valueB*    *
dtype0
ā

Assign_152Assign6mio_variable/continuous_expand_xtr/dense/bias/gradientInitializer_152/zeros*
T0*I
_class?
=;loc:@mio_variable/continuous_expand_xtr/dense/bias/gradient*
validate_shape(*
use_locking(

"continuous_expand_xtr/dense/MatMulMatMulconcat_18mio_variable/continuous_expand_xtr/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
:mio_variable/continuous_expand_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*3
	container&$continuous_expand_xtr/dense_1/kernel
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
"Initializer_153/random_uniform/minConst*
dtype0*
valueB
 *   ž
O
"Initializer_153/random_uniform/maxConst*
dtype0*
valueB
 *   >

,Initializer_153/random_uniform/RandomUniformRandomUniform$Initializer_153/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
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
8mio_variable/continuous_expand_xtr/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*1
	container$"continuous_expand_xtr/dense_1/bias
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
:mio_variable/continuous_expand_xtr/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$continuous_expand_xtr/dense_2/kernel*
shape:	@
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
,Initializer_155/random_uniform/RandomUniformRandomUniform$Initializer_155/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
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
8mio_variable/continuous_expand_xtr/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*1
	container$"continuous_expand_xtr/dense_2/bias
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
"Initializer_157/random_uniform/minConst*
valueB
 *ž*
dtype0
O
"Initializer_157/random_uniform/maxConst*
dtype0*
valueB
 *>
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

Assign_157Assign:mio_variable/continuous_expand_xtr/dense_3/kernel/gradientInitializer_157/random_uniform*
T0*M
_classC
A?loc:@mio_variable/continuous_expand_xtr/dense_3/kernel/gradient*
validate_shape(*
use_locking(
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

Assign_158Assign8mio_variable/continuous_expand_xtr/dense_3/bias/gradientInitializer_158/zeros*
T0*K
_classA
?=loc:@mio_variable/continuous_expand_xtr/dense_3/bias/gradient*
validate_shape(*
use_locking(
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
3mio_variable/duration_predict/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*,
	containerduration_predict/dense/kernel
Ŧ
3mio_variable/duration_predict/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*,
	containerduration_predict/dense/kernel
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
dtype0*
seed2 *

seed *
T0
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
1mio_variable/duration_predict/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:**
	containerduration_predict/dense/bias
Ŗ
1mio_variable/duration_predict/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:**
	containerduration_predict/dense/bias
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
duration_predict/dense/MatMulMatMulconcat_13mio_variable/duration_predict/dense/kernel/variable*
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
"Initializer_161/random_uniform/maxConst*
valueB
 *   >*
dtype0
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

Assign_161Assign5mio_variable/duration_predict/dense_1/kernel/gradientInitializer_161/random_uniform*
validate_shape(*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/duration_predict/dense_1/kernel/gradient
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

Assign_162Assign3mio_variable/duration_predict/dense_1/bias/gradientInitializer_162/zeros*
use_locking(*
T0*F
_class<
:8loc:@mio_variable/duration_predict/dense_1/bias/gradient*
validate_shape(
˛
duration_predict/dense_1/MatMulMatMul!duration_predict/dropout/Identity5mio_variable/duration_predict/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0
Ą
 duration_predict/dense_1/BiasAddBiasAddduration_predict/dense_1/MatMul3mio_variable/duration_predict/dense_1/bias/variable*
T0*
data_formatNHWC
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
5mio_variable/duration_predict/dense_2/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:	@*.
	container!duration_predict/dense_2/kernel
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

seed *
T0*
dtype0*
seed2 
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
3mio_variable/duration_predict/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*,
	containerduration_predict/dense_2/bias
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

seed *
T0*
dtype0*
seed2 
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

Assign_165Assign5mio_variable/duration_predict/dense_3/kernel/gradientInitializer_165/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@mio_variable/duration_predict/dense_3/kernel/gradient*
validate_shape(
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

Assign_166Assign3mio_variable/duration_predict/dense_3/bias/gradientInitializer_166/zeros*
T0*F
_class<
:8loc:@mio_variable/duration_predict/dense_3/bias/gradient*
validate_shape(*
use_locking(
ŗ
duration_predict/dense_3/MatMulMatMul"duration_predict/dense_2/LeakyRelu5mio_variable/duration_predict/dense_3/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
,Initializer_167/random_uniform/RandomUniformRandomUniform$Initializer_167/random_uniform/shape*
dtype0*
seed2 *

seed *
T0
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

Assign_167Assign<mio_variable/duration_pos_bias_predict/dense/kernel/gradientInitializer_167/random_uniform*
T0*O
_classE
CAloc:@mio_variable/duration_pos_bias_predict/dense/kernel/gradient*
validate_shape(*
use_locking(
ĩ
:mio_variable/duration_pos_bias_predict/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$duration_pos_bias_predict/dense/bias*
shape:
ĩ
:mio_variable/duration_pos_bias_predict/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*3
	container&$duration_pos_bias_predict/dense/bias*
shape:
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
&duration_pos_bias_predict/dense/MatMulMatMulconcat_2<mio_variable/duration_pos_bias_predict/dense/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
Y
$Initializer_169/random_uniform/shapeConst*
dtype0*
valueB"   @   
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

Assign_169Assign>mio_variable/duration_pos_bias_predict/dense_1/kernel/gradientInitializer_169/random_uniform*
T0*Q
_classG
ECloc:@mio_variable/duration_pos_bias_predict/dense_1/kernel/gradient*
validate_shape(*
use_locking(
¸
<mio_variable/duration_pos_bias_predict/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&duration_pos_bias_predict/dense_1/bias*
shape:@
¸
<mio_variable/duration_pos_bias_predict/dense_1/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:@*5
	container(&duration_pos_bias_predict/dense_1/bias
F
Initializer_170/zerosConst*
valueB@*    *
dtype0
ė

Assign_170Assign<mio_variable/duration_pos_bias_predict/dense_1/bias/gradientInitializer_170/zeros*
use_locking(*
T0*O
_classE
CAloc:@mio_variable/duration_pos_bias_predict/dense_1/bias/gradient*
validate_shape(
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
dtype0*
valueB"@      
O
"Initializer_171/random_uniform/minConst*
dtype0*
valueB
 *ž
O
"Initializer_171/random_uniform/maxConst*
valueB
 *>*
dtype0

,Initializer_171/random_uniform/RandomUniformRandomUniform$Initializer_171/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
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

Assign_171Assign>mio_variable/duration_pos_bias_predict/dense_2/kernel/gradientInitializer_171/random_uniform*Q
_classG
ECloc:@mio_variable/duration_pos_bias_predict/dense_2/kernel/gradient*
validate_shape(*
use_locking(*
T0
¸
<mio_variable/duration_pos_bias_predict/dense_2/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&duration_pos_bias_predict/dense_2/bias*
shape:
¸
<mio_variable/duration_pos_bias_predict/dense_2/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*5
	container(&duration_pos_bias_predict/dense_2/bias*
shape:
F
Initializer_172/zerosConst*
valueB*    *
dtype0
ė

Assign_172Assign<mio_variable/duration_pos_bias_predict/dense_2/bias/gradientInitializer_172/zeros*
validate_shape(*
use_locking(*
T0*O
_classE
CAloc:@mio_variable/duration_pos_bias_predict/dense_2/bias/gradient
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
+mio_variable/hate_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*$
	containerhate_xtr/dense/kernel
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

seed *
T0*
dtype0*
seed2 
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
)mio_variable/hate_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*"
	containerhate_xtr/dense/bias

)mio_variable/hate_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*"
	containerhate_xtr/dense/bias*
shape:
G
Initializer_174/zerosConst*
valueB*    *
dtype0
Æ

Assign_174Assign)mio_variable/hate_xtr/dense/bias/gradientInitializer_174/zeros*
use_locking(*
T0*<
_class2
0.loc:@mio_variable/hate_xtr/dense/bias/gradient*
validate_shape(
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
-mio_variable/hate_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*&
	containerhate_xtr/dense_1/kernel
 
-mio_variable/hate_xtr/dense_1/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerhate_xtr/dense_1/kernel*
shape:

Y
$Initializer_175/random_uniform/shapeConst*
valueB"      *
dtype0
O
"Initializer_175/random_uniform/minConst*
valueB
 *   ž*
dtype0
O
"Initializer_175/random_uniform/maxConst*
valueB
 *   >*
dtype0

,Initializer_175/random_uniform/RandomUniformRandomUniform$Initializer_175/random_uniform/shape*
seed2 *

seed *
T0*
dtype0
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

Assign_175Assign-mio_variable/hate_xtr/dense_1/kernel/gradientInitializer_175/random_uniform*
use_locking(*
T0*@
_class6
42loc:@mio_variable/hate_xtr/dense_1/kernel/gradient*
validate_shape(

+mio_variable/hate_xtr/dense_1/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containerhate_xtr/dense_1/bias
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
hate_xtr/dense_1/MatMulMatMulhate_xtr/dense/LeakyRelu-mio_variable/hate_xtr/dense_1/kernel/variable*
transpose_b( *
T0*
transpose_a( 

hate_xtr/dense_1/BiasAddBiasAddhate_xtr/dense_1/MatMul+mio_variable/hate_xtr/dense_1/bias/variable*
data_formatNHWC*
T0
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
,Initializer_177/random_uniform/RandomUniformRandomUniform$Initializer_177/random_uniform/shape*
T0*
dtype0*
seed2 *

seed 
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

Assign_177Assign-mio_variable/hate_xtr/dense_2/kernel/gradientInitializer_177/random_uniform*
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
hate_xtr/dense_2/MatMulMatMulhate_xtr/dense_1/LeakyRelu-mio_variable/hate_xtr/dense_2/kernel/variable*
transpose_a( *
transpose_b( *
T0
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
-mio_variable/hate_xtr/dense_3/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*&
	containerhate_xtr/dense_3/kernel

-mio_variable/hate_xtr/dense_3/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape
:@*&
	containerhate_xtr/dense_3/kernel
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
,Initializer_179/random_uniform/RandomUniformRandomUniform$Initializer_179/random_uniform/shape*
seed2 *

seed *
T0*
dtype0
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

Assign_179Assign-mio_variable/hate_xtr/dense_3/kernel/gradientInitializer_179/random_uniform*
T0*@
_class6
42loc:@mio_variable/hate_xtr/dense_3/kernel/gradient*
validate_shape(*
use_locking(
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
hate_xtr/dense_3/BiasAddBiasAddhate_xtr/dense_3/MatMul+mio_variable/hate_xtr/dense_3/bias/variable*
data_formatNHWC*
T0
F
hate_xtr/dense_3/SigmoidSigmoidhate_xtr/dense_3/BiasAdd*
T0
 
-mio_variable/report_xtr/dense/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*&
	containerreport_xtr/dense/kernel*
shape:
°
 
-mio_variable/report_xtr/dense/kernel/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
°*&
	containerreport_xtr/dense/kernel
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

Assign_181Assign-mio_variable/report_xtr/dense/kernel/gradientInitializer_181/random_uniform*@
_class6
42loc:@mio_variable/report_xtr/dense/kernel/gradient*
validate_shape(*
use_locking(*
T0

+mio_variable/report_xtr/dense/bias/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*$
	containerreport_xtr/dense/bias*
shape:

+mio_variable/report_xtr/dense/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*$
	containerreport_xtr/dense/bias
G
Initializer_182/zerosConst*
valueB*    *
dtype0
Ę

Assign_182Assign+mio_variable/report_xtr/dense/bias/gradientInitializer_182/zeros*
T0*>
_class4
20loc:@mio_variable/report_xtr/dense/bias/gradient*
validate_shape(*
use_locking(

report_xtr/dense/MatMulMatMulconcat_1-mio_variable/report_xtr/dense/kernel/variable*
T0*
transpose_a( *
transpose_b( 
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
/mio_variable/report_xtr/dense_1/kernel/variableVariableFromMioComponentTableMIO_TABLE_ADDRESS*
shape:
*(
	containerreport_xtr/dense_1/kernel
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
report_xtr/dense_1/MatMulMatMulreport_xtr/dense/LeakyRelu/mio_variable/report_xtr/dense_1/kernel/variable*
transpose_a( *
transpose_b( *
T0

report_xtr/dense_1/BiasAddBiasAddreport_xtr/dense_1/MatMul-mio_variable/report_xtr/dense_1/bias/variable*
T0*
data_formatNHWC
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
"Initializer_185/random_uniform/maxConst*
valueB
 *ķ5>*
dtype0

,Initializer_185/random_uniform/RandomUniformRandomUniform$Initializer_185/random_uniform/shape*
seed2 *

seed *
T0*
dtype0
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

Assign_185Assign/mio_variable/report_xtr/dense_2/kernel/gradientInitializer_185/random_uniform*
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
,Initializer_187/random_uniform/RandomUniformRandomUniform$Initializer_187/random_uniform/shape*

seed *
T0*
dtype0*
seed2 
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
-mio_variable/report_xtr/dense_3/bias/gradientGradientFromMioComponentTableMIO_TABLE_ADDRESS*
shape:*&
	containerreport_xtr/dense_3/bias
F
Initializer_188/zerosConst*
valueB*    *
dtype0
Î

Assign_188Assign-mio_variable/report_xtr/dense_3/bias/gradientInitializer_188/zeros*@
_class6
42loc:@mio_variable/report_xtr/dense_3/bias/gradient*
validate_shape(*
use_locking(*
T0
Ą
report_xtr/dense_3/MatMulMatMulreport_xtr/dense_2/LeakyRelu/mio_variable/report_xtr/dense_3/kernel/variable*
T0*
transpose_a( *
transpose_b( 

report_xtr/dense_3/BiasAddBiasAddreport_xtr/dense_3/MatMul-mio_variable/report_xtr/dense_3/bias/variable*
T0*
data_formatNHWC
J
report_xtr/dense_3/SigmoidSigmoidreport_xtr/dense_3/BiasAdd*
T0"