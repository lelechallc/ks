
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
0
labelPlaceholder*
dtype0*
shape:
>
varlen_embed_offsetPlaceholder*
shape:*
dtype0
7
varlen_embedPlaceholder*
shape:*
dtype0
=
user_embedding_idsPlaceholder*
shape:*
dtype0
@
user_embedding_cumsumPlaceholder*
dtype0*
shape:
6
level1_id_4Placeholder*
dtype0*
shape:
�
varlen_gather_4/VarlenGatherVarlenGathervarlen_embedlevel1_id_4varlen_embed_offset"/device:GPU:0*
Tindices0*
Tparams0*	
dim
R
varlen_gather_4/Reshape/shapeConst*
valueB"����   *
dtype0
e
varlen_gather_4/ReshapeReshapelevel1_id_4varlen_gather_4/Reshape/shape*
T0*
Tshape0
D
varlen_gather_4/ShapeShapelevel1_id_4*
T0*
out_type0
Q
#varlen_gather_4/strided_slice/stackConst*
valueB: *
dtype0
S
%varlen_gather_4/strided_slice/stack_1Const*
valueB:*
dtype0
S
%varlen_gather_4/strided_slice/stack_2Const*
valueB:*
dtype0
�
varlen_gather_4/strided_sliceStridedSlicevarlen_gather_4/Shape#varlen_gather_4/strided_slice/stack%varlen_gather_4/strided_slice/stack_1%varlen_gather_4/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
?
varlen_gather_4/add/yConst*
value	B :*
dtype0
Y
varlen_gather_4/addAddvarlen_gather_4/strided_slicevarlen_gather_4/add/y*
T0
E
varlen_gather_4/range/startConst*
value	B :*
dtype0
E
varlen_gather_4/range/deltaConst*
dtype0*
value	B :
y
varlen_gather_4/rangeRangevarlen_gather_4/range/startvarlen_gather_4/addvarlen_gather_4/range/delta*

Tidx0
J
varlen_gather_4/SizeSizevarlen_embed_offset*
T0*
out_type0
[
varlen_gather_4/ScatterNd/shapePackvarlen_gather_4/Size*
T0*

axis *
N
�
varlen_gather_4/ScatterNd	ScatterNdvarlen_gather_4/Reshapevarlen_gather_4/rangevarlen_gather_4/ScatterNd/shape*
Tindices0*
T0
?
varlen_gather_4/sub/yConst*
value	B :*
dtype0
U
varlen_gather_4/subSubvarlen_gather_4/ScatterNdvarlen_gather_4/sub/y*
T0
I
varlen_gather_4/ps_embed_4/yConst*
dtype0*
valueB
 *  �?
f
varlen_gather_4/ps_embed_4Mulvarlen_gather_4/VarlenGathervarlen_gather_4/ps_embed_4/y*
T0
L
"input_user_embedding/GatherV2/axisConst*
value	B : *
dtype0
�
input_user_embedding/GatherV2GatherV2varlen_gather_4/subuser_embedding_ids"input_user_embedding/GatherV2/axis*
Tindices0*
Tparams0*
Taxis0
S
input_user_embedding/ShapeShapeuser_embedding_cumsum*
T0*
out_type0
V
(input_user_embedding/strided_slice/stackConst*
valueB: *
dtype0
X
*input_user_embedding/strided_slice/stack_1Const*
valueB:*
dtype0
X
*input_user_embedding/strided_slice/stack_2Const*
valueB:*
dtype0
�
"input_user_embedding/strided_sliceStridedSliceinput_user_embedding/Shape(input_user_embedding/strided_slice/stack*input_user_embedding/strided_slice/stack_1*input_user_embedding/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
D
input_user_embedding/sub/yConst*
dtype0*
value	B :
h
input_user_embedding/subSub"input_user_embedding/strided_sliceinput_user_embedding/sub/y*
T0
Y
input_user_embedding/SizeSizeinput_user_embedding/GatherV2*
T0*
out_type0
H
input_user_embedding/Greater/yConst*
value	B : *
dtype0
k
input_user_embedding/GreaterGreaterinput_user_embedding/Sizeinput_user_embedding/Greater/y*
T0
o
 input_user_embedding/cond/SwitchSwitchinput_user_embedding/Greaterinput_user_embedding/Greater*
T0

[
"input_user_embedding/cond/switch_tIdentity"input_user_embedding/cond/Switch:1*
T0

Y
"input_user_embedding/cond/switch_fIdentity input_user_embedding/cond/Switch*
T0

T
!input_user_embedding/cond/pred_idIdentityinput_user_embedding/Greater*
T0

�
@input_user_embedding/cond/make_sparse_indice/strided_slice/stackConst#^input_user_embedding/cond/switch_t*
valueB:
���������*
dtype0
�
Binput_user_embedding/cond/make_sparse_indice/strided_slice/stack_1Const#^input_user_embedding/cond/switch_t*
valueB: *
dtype0
�
Binput_user_embedding/cond/make_sparse_indice/strided_slice/stack_2Const#^input_user_embedding/cond/switch_t*
dtype0*
valueB:
�
:input_user_embedding/cond/make_sparse_indice/strided_sliceStridedSliceCinput_user_embedding/cond/make_sparse_indice/strided_slice/Switch:1@input_user_embedding/cond/make_sparse_indice/strided_slice/stackBinput_user_embedding/cond/make_sparse_indice/strided_slice/stack_1Binput_user_embedding/cond/make_sparse_indice/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
�
Ainput_user_embedding/cond/make_sparse_indice/strided_slice/SwitchSwitchuser_embedding_cumsum!input_user_embedding/cond/pred_id*
T0*(
_class
loc:@user_embedding_cumsum
�
8input_user_embedding/cond/make_sparse_indice/range/startConst#^input_user_embedding/cond/switch_t*
value	B : *
dtype0
�
8input_user_embedding/cond/make_sparse_indice/range/deltaConst#^input_user_embedding/cond/switch_t*
value	B :*
dtype0
�
2input_user_embedding/cond/make_sparse_indice/rangeRange8input_user_embedding/cond/make_sparse_indice/range/start:input_user_embedding/cond/make_sparse_indice/strided_slice8input_user_embedding/cond/make_sparse_indice/range/delta*

Tidx0
�
2input_user_embedding/cond/make_sparse_indice/ShapeShapeCinput_user_embedding/cond/make_sparse_indice/strided_slice/Switch:1*
T0*
out_type0
�
Binput_user_embedding/cond/make_sparse_indice/strided_slice_1/stackConst#^input_user_embedding/cond/switch_t*
valueB:
���������*
dtype0
�
Dinput_user_embedding/cond/make_sparse_indice/strided_slice_1/stack_1Const#^input_user_embedding/cond/switch_t*
dtype0*
valueB: 
�
Dinput_user_embedding/cond/make_sparse_indice/strided_slice_1/stack_2Const#^input_user_embedding/cond/switch_t*
valueB:*
dtype0
�
<input_user_embedding/cond/make_sparse_indice/strided_slice_1StridedSlice2input_user_embedding/cond/make_sparse_indice/ShapeBinput_user_embedding/cond/make_sparse_indice/strided_slice_1/stackDinput_user_embedding/cond/make_sparse_indice/strided_slice_1/stack_1Dinput_user_embedding/cond/make_sparse_indice/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0
�
4input_user_embedding/cond/make_sparse_indice/Shape_1Shape2input_user_embedding/cond/make_sparse_indice/range*
T0*
out_type0
�
Binput_user_embedding/cond/make_sparse_indice/strided_slice_2/stackConst#^input_user_embedding/cond/switch_t*
valueB:
���������*
dtype0
�
Dinput_user_embedding/cond/make_sparse_indice/strided_slice_2/stack_1Const#^input_user_embedding/cond/switch_t*
valueB: *
dtype0
�
Dinput_user_embedding/cond/make_sparse_indice/strided_slice_2/stack_2Const#^input_user_embedding/cond/switch_t*
valueB:*
dtype0
�
<input_user_embedding/cond/make_sparse_indice/strided_slice_2StridedSlice4input_user_embedding/cond/make_sparse_indice/Shape_1Binput_user_embedding/cond/make_sparse_indice/strided_slice_2/stackDinput_user_embedding/cond/make_sparse_indice/strided_slice_2/stack_1Dinput_user_embedding/cond/make_sparse_indice/strided_slice_2/stack_2*
end_mask *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
�
<input_user_embedding/cond/make_sparse_indice/Reshape/shape/0Const#^input_user_embedding/cond/switch_t*
valueB :
���������*
dtype0
�
:input_user_embedding/cond/make_sparse_indice/Reshape/shapePack<input_user_embedding/cond/make_sparse_indice/Reshape/shape/0<input_user_embedding/cond/make_sparse_indice/strided_slice_1*
T0*

axis *
N
�
4input_user_embedding/cond/make_sparse_indice/ReshapeReshapeCinput_user_embedding/cond/make_sparse_indice/strided_slice/Switch:1:input_user_embedding/cond/make_sparse_indice/Reshape/shape*
T0*
Tshape0
�
>input_user_embedding/cond/make_sparse_indice/Reshape_1/shape/0Const#^input_user_embedding/cond/switch_t*
valueB :
���������*
dtype0
�
<input_user_embedding/cond/make_sparse_indice/Reshape_1/shapePack>input_user_embedding/cond/make_sparse_indice/Reshape_1/shape/0<input_user_embedding/cond/make_sparse_indice/strided_slice_2*
N*
T0*

axis 
�
6input_user_embedding/cond/make_sparse_indice/Reshape_1Reshape2input_user_embedding/cond/make_sparse_indice/range<input_user_embedding/cond/make_sparse_indice/Reshape_1/shape*
T0*
Tshape0
�
7input_user_embedding/cond/make_sparse_indice/UpperBound
UpperBound4input_user_embedding/cond/make_sparse_indice/Reshape6input_user_embedding/cond/make_sparse_indice/Reshape_1*
T0*
out_type0
�
4input_user_embedding/cond/make_sparse_indice/Shape_2Shape2input_user_embedding/cond/make_sparse_indice/range*
T0*
out_type0
�
6input_user_embedding/cond/make_sparse_indice/Reshape_2Reshape7input_user_embedding/cond/make_sparse_indice/UpperBound4input_user_embedding/cond/make_sparse_indice/Shape_2*
T0*
Tshape0
�
2input_user_embedding/cond/make_sparse_indice/sub/yConst#^input_user_embedding/cond/switch_t*
value	B :*
dtype0
�
0input_user_embedding/cond/make_sparse_indice/subSub6input_user_embedding/cond/make_sparse_indice/Reshape_22input_user_embedding/cond/make_sparse_indice/sub/y*
T0
v
'input_user_embedding/cond/GatherV2/axisConst#^input_user_embedding/cond/switch_t*
value	B : *
dtype0
�
"input_user_embedding/cond/GatherV2GatherV2+input_user_embedding/cond/GatherV2/Switch:1-input_user_embedding/cond/GatherV2/Switch_1:1'input_user_embedding/cond/GatherV2/axis*
Tparams0*
Taxis0*
Tindices0
�
)input_user_embedding/cond/GatherV2/SwitchSwitchvarlen_gather_4/ps_embed_4!input_user_embedding/cond/pred_id*
T0*-
_class#
!loc:@varlen_gather_4/ps_embed_4
�
+input_user_embedding/cond/GatherV2/Switch_1Switchinput_user_embedding/GatherV2!input_user_embedding/cond/pred_id*
T0*0
_class&
$"loc:@input_user_embedding/GatherV2
�
$input_user_embedding/cond/SegmentSum
SegmentSum"input_user_embedding/cond/GatherV20input_user_embedding/cond/make_sparse_indice/sub*
Tindices0*
T0
�
input_user_embedding/cond/ShapeShapeCinput_user_embedding/cond/make_sparse_indice/strided_slice/Switch:1*
T0*
out_type0
�
-input_user_embedding/cond/strided_slice/stackConst#^input_user_embedding/cond/switch_t*
dtype0*
valueB: 
�
/input_user_embedding/cond/strided_slice/stack_1Const#^input_user_embedding/cond/switch_t*
valueB:*
dtype0
�
/input_user_embedding/cond/strided_slice/stack_2Const#^input_user_embedding/cond/switch_t*
dtype0*
valueB:
�
'input_user_embedding/cond/strided_sliceStridedSliceinput_user_embedding/cond/Shape-input_user_embedding/cond/strided_slice/stack/input_user_embedding/cond/strided_slice/stack_1/input_user_embedding/cond/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
n
input_user_embedding/cond/sub/yConst#^input_user_embedding/cond/switch_t*
value	B :*
dtype0
w
input_user_embedding/cond/subSub'input_user_embedding/cond/strided_sliceinput_user_embedding/cond/sub/y*
T0
i
!input_user_embedding/cond/Shape_1Shape$input_user_embedding/cond/SegmentSum*
T0*
out_type0
�
/input_user_embedding/cond/strided_slice_1/stackConst#^input_user_embedding/cond/switch_t*
valueB: *
dtype0
�
1input_user_embedding/cond/strided_slice_1/stack_1Const#^input_user_embedding/cond/switch_t*
dtype0*
valueB:
�
1input_user_embedding/cond/strided_slice_1/stack_2Const#^input_user_embedding/cond/switch_t*
dtype0*
valueB:
�
)input_user_embedding/cond/strided_slice_1StridedSlice!input_user_embedding/cond/Shape_1/input_user_embedding/cond/strided_slice_1/stack1input_user_embedding/cond/strided_slice_1/stack_11input_user_embedding/cond/strided_slice_1/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
y
input_user_embedding/cond/sub_1Subinput_user_embedding/cond/sub)input_user_embedding/cond/strided_slice_1*
T0
y
*input_user_embedding/cond/Pad/paddings/0/0Const#^input_user_embedding/cond/switch_t*
dtype0*
value	B : 
�
(input_user_embedding/cond/Pad/paddings/0Pack*input_user_embedding/cond/Pad/paddings/0/0input_user_embedding/cond/sub_1*
T0*

axis *
N
�
*input_user_embedding/cond/Pad/paddings/1_1Const#^input_user_embedding/cond/switch_t*
dtype0*
valueB"        
�
&input_user_embedding/cond/Pad/paddingsPack(input_user_embedding/cond/Pad/paddings/0*input_user_embedding/cond/Pad/paddings/1_1*
N*
T0*

axis 
�
input_user_embedding/cond/PadPad$input_user_embedding/cond/SegmentSum&input_user_embedding/cond/Pad/paddings*
T0*
	Tpaddings0
t
%input_user_embedding/cond/zeros/mul/yConst#^input_user_embedding/cond/switch_f*
value	B :*
dtype0
�
#input_user_embedding/cond/zeros/mulMul*input_user_embedding/cond/zeros/mul/Switch%input_user_embedding/cond/zeros/mul/y*
T0
�
*input_user_embedding/cond/zeros/mul/SwitchSwitchinput_user_embedding/sub!input_user_embedding/cond/pred_id*
T0*+
_class!
loc:@input_user_embedding/sub
v
&input_user_embedding/cond/zeros/Less/yConst#^input_user_embedding/cond/switch_f*
value
B :�*
dtype0
�
$input_user_embedding/cond/zeros/LessLess#input_user_embedding/cond/zeros/mul&input_user_embedding/cond/zeros/Less/y*
T0
w
(input_user_embedding/cond/zeros/packed/1Const#^input_user_embedding/cond/switch_f*
value	B :*
dtype0
�
&input_user_embedding/cond/zeros/packedPack*input_user_embedding/cond/zeros/mul/Switch(input_user_embedding/cond/zeros/packed/1*
T0*

axis *
N
w
%input_user_embedding/cond/zeros/ConstConst#^input_user_embedding/cond/switch_f*
valueB
 *    *
dtype0
�
input_user_embedding/cond/zerosFill&input_user_embedding/cond/zeros/packed%input_user_embedding/cond/zeros/Const*
T0*

index_type0
z
input_user_embedding/cond/MergeMergeinput_user_embedding/cond/zerosinput_user_embedding/cond/Pad*
T0*
N
N
kai_input_user_embeddingIdentityinput_user_embedding/cond/Merge*
T0
B
Reshape/shapeConst*
valueB"����   *
dtype0
R
ReshapeReshapekai_input_user_embeddingReshape/shape*
T0*
Tshape0
D
Reshape_1/shapeConst*
valueB"����   *
dtype0
E
	Reshape_1ReshapeReshapeReshape_1/shape*
T0*
Tshape0
=
c_id_embedding_idsPlaceholder*
dtype0*
shape:
@
c_id_embedding_cumsumPlaceholder*
dtype0*
shape:
7
level1_id_64Placeholder*
dtype0*
shape:
�
varlen_gather_64/VarlenGatherVarlenGathervarlen_embedlevel1_id_64varlen_embed_offset"/device:GPU:0*
Tindices0*
Tparams0*	
dim@
S
varlen_gather_64/Reshape/shapeConst*
valueB"����   *
dtype0
h
varlen_gather_64/ReshapeReshapelevel1_id_64varlen_gather_64/Reshape/shape*
T0*
Tshape0
F
varlen_gather_64/ShapeShapelevel1_id_64*
T0*
out_type0
R
$varlen_gather_64/strided_slice/stackConst*
valueB: *
dtype0
T
&varlen_gather_64/strided_slice/stack_1Const*
valueB:*
dtype0
T
&varlen_gather_64/strided_slice/stack_2Const*
valueB:*
dtype0
�
varlen_gather_64/strided_sliceStridedSlicevarlen_gather_64/Shape$varlen_gather_64/strided_slice/stack&varlen_gather_64/strided_slice/stack_1&varlen_gather_64/strided_slice/stack_2*
end_mask *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
@
varlen_gather_64/add/yConst*
dtype0*
value	B :
\
varlen_gather_64/addAddvarlen_gather_64/strided_slicevarlen_gather_64/add/y*
T0
F
varlen_gather_64/range/startConst*
value	B :*
dtype0
F
varlen_gather_64/range/deltaConst*
value	B :*
dtype0
}
varlen_gather_64/rangeRangevarlen_gather_64/range/startvarlen_gather_64/addvarlen_gather_64/range/delta*

Tidx0
K
varlen_gather_64/SizeSizevarlen_embed_offset*
T0*
out_type0
]
 varlen_gather_64/ScatterNd/shapePackvarlen_gather_64/Size*
T0*

axis *
N
�
varlen_gather_64/ScatterNd	ScatterNdvarlen_gather_64/Reshapevarlen_gather_64/range varlen_gather_64/ScatterNd/shape*
T0*
Tindices0
@
varlen_gather_64/sub/yConst*
value	B :*
dtype0
X
varlen_gather_64/subSubvarlen_gather_64/ScatterNdvarlen_gather_64/sub/y*
T0
K
varlen_gather_64/ps_embed_64/yConst*
dtype0*
valueB
 *  �?
k
varlen_gather_64/ps_embed_64Mulvarlen_gather_64/VarlenGathervarlen_gather_64/ps_embed_64/y*
T0
L
"input_c_id_embedding/GatherV2/axisConst*
value	B : *
dtype0
�
input_c_id_embedding/GatherV2GatherV2varlen_gather_64/subc_id_embedding_ids"input_c_id_embedding/GatherV2/axis*
Tindices0*
Tparams0*
Taxis0
S
input_c_id_embedding/ShapeShapec_id_embedding_cumsum*
T0*
out_type0
V
(input_c_id_embedding/strided_slice/stackConst*
dtype0*
valueB: 
X
*input_c_id_embedding/strided_slice/stack_1Const*
dtype0*
valueB:
X
*input_c_id_embedding/strided_slice/stack_2Const*
valueB:*
dtype0
�
"input_c_id_embedding/strided_sliceStridedSliceinput_c_id_embedding/Shape(input_c_id_embedding/strided_slice/stack*input_c_id_embedding/strided_slice/stack_1*input_c_id_embedding/strided_slice/stack_2*
end_mask *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
D
input_c_id_embedding/sub/yConst*
dtype0*
value	B :
h
input_c_id_embedding/subSub"input_c_id_embedding/strided_sliceinput_c_id_embedding/sub/y*
T0
Y
input_c_id_embedding/SizeSizeinput_c_id_embedding/GatherV2*
T0*
out_type0
H
input_c_id_embedding/Greater/yConst*
value	B : *
dtype0
k
input_c_id_embedding/GreaterGreaterinput_c_id_embedding/Sizeinput_c_id_embedding/Greater/y*
T0
o
 input_c_id_embedding/cond/SwitchSwitchinput_c_id_embedding/Greaterinput_c_id_embedding/Greater*
T0

[
"input_c_id_embedding/cond/switch_tIdentity"input_c_id_embedding/cond/Switch:1*
T0

Y
"input_c_id_embedding/cond/switch_fIdentity input_c_id_embedding/cond/Switch*
T0

T
!input_c_id_embedding/cond/pred_idIdentityinput_c_id_embedding/Greater*
T0

�
@input_c_id_embedding/cond/make_sparse_indice/strided_slice/stackConst#^input_c_id_embedding/cond/switch_t*
dtype0*
valueB:
���������
�
Binput_c_id_embedding/cond/make_sparse_indice/strided_slice/stack_1Const#^input_c_id_embedding/cond/switch_t*
valueB: *
dtype0
�
Binput_c_id_embedding/cond/make_sparse_indice/strided_slice/stack_2Const#^input_c_id_embedding/cond/switch_t*
valueB:*
dtype0
�
:input_c_id_embedding/cond/make_sparse_indice/strided_sliceStridedSliceCinput_c_id_embedding/cond/make_sparse_indice/strided_slice/Switch:1@input_c_id_embedding/cond/make_sparse_indice/strided_slice/stackBinput_c_id_embedding/cond/make_sparse_indice/strided_slice/stack_1Binput_c_id_embedding/cond/make_sparse_indice/strided_slice/stack_2*
end_mask *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
�
Ainput_c_id_embedding/cond/make_sparse_indice/strided_slice/SwitchSwitchc_id_embedding_cumsum!input_c_id_embedding/cond/pred_id*
T0*(
_class
loc:@c_id_embedding_cumsum
�
8input_c_id_embedding/cond/make_sparse_indice/range/startConst#^input_c_id_embedding/cond/switch_t*
value	B : *
dtype0
�
8input_c_id_embedding/cond/make_sparse_indice/range/deltaConst#^input_c_id_embedding/cond/switch_t*
value	B :*
dtype0
�
2input_c_id_embedding/cond/make_sparse_indice/rangeRange8input_c_id_embedding/cond/make_sparse_indice/range/start:input_c_id_embedding/cond/make_sparse_indice/strided_slice8input_c_id_embedding/cond/make_sparse_indice/range/delta*

Tidx0
�
2input_c_id_embedding/cond/make_sparse_indice/ShapeShapeCinput_c_id_embedding/cond/make_sparse_indice/strided_slice/Switch:1*
T0*
out_type0
�
Binput_c_id_embedding/cond/make_sparse_indice/strided_slice_1/stackConst#^input_c_id_embedding/cond/switch_t*
valueB:
���������*
dtype0
�
Dinput_c_id_embedding/cond/make_sparse_indice/strided_slice_1/stack_1Const#^input_c_id_embedding/cond/switch_t*
valueB: *
dtype0
�
Dinput_c_id_embedding/cond/make_sparse_indice/strided_slice_1/stack_2Const#^input_c_id_embedding/cond/switch_t*
valueB:*
dtype0
�
<input_c_id_embedding/cond/make_sparse_indice/strided_slice_1StridedSlice2input_c_id_embedding/cond/make_sparse_indice/ShapeBinput_c_id_embedding/cond/make_sparse_indice/strided_slice_1/stackDinput_c_id_embedding/cond/make_sparse_indice/strided_slice_1/stack_1Dinput_c_id_embedding/cond/make_sparse_indice/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0
�
4input_c_id_embedding/cond/make_sparse_indice/Shape_1Shape2input_c_id_embedding/cond/make_sparse_indice/range*
T0*
out_type0
�
Binput_c_id_embedding/cond/make_sparse_indice/strided_slice_2/stackConst#^input_c_id_embedding/cond/switch_t*
valueB:
���������*
dtype0
�
Dinput_c_id_embedding/cond/make_sparse_indice/strided_slice_2/stack_1Const#^input_c_id_embedding/cond/switch_t*
dtype0*
valueB: 
�
Dinput_c_id_embedding/cond/make_sparse_indice/strided_slice_2/stack_2Const#^input_c_id_embedding/cond/switch_t*
valueB:*
dtype0
�
<input_c_id_embedding/cond/make_sparse_indice/strided_slice_2StridedSlice4input_c_id_embedding/cond/make_sparse_indice/Shape_1Binput_c_id_embedding/cond/make_sparse_indice/strided_slice_2/stackDinput_c_id_embedding/cond/make_sparse_indice/strided_slice_2/stack_1Dinput_c_id_embedding/cond/make_sparse_indice/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
�
<input_c_id_embedding/cond/make_sparse_indice/Reshape/shape/0Const#^input_c_id_embedding/cond/switch_t*
valueB :
���������*
dtype0
�
:input_c_id_embedding/cond/make_sparse_indice/Reshape/shapePack<input_c_id_embedding/cond/make_sparse_indice/Reshape/shape/0<input_c_id_embedding/cond/make_sparse_indice/strided_slice_1*
T0*

axis *
N
�
4input_c_id_embedding/cond/make_sparse_indice/ReshapeReshapeCinput_c_id_embedding/cond/make_sparse_indice/strided_slice/Switch:1:input_c_id_embedding/cond/make_sparse_indice/Reshape/shape*
T0*
Tshape0
�
>input_c_id_embedding/cond/make_sparse_indice/Reshape_1/shape/0Const#^input_c_id_embedding/cond/switch_t*
valueB :
���������*
dtype0
�
<input_c_id_embedding/cond/make_sparse_indice/Reshape_1/shapePack>input_c_id_embedding/cond/make_sparse_indice/Reshape_1/shape/0<input_c_id_embedding/cond/make_sparse_indice/strided_slice_2*
T0*

axis *
N
�
6input_c_id_embedding/cond/make_sparse_indice/Reshape_1Reshape2input_c_id_embedding/cond/make_sparse_indice/range<input_c_id_embedding/cond/make_sparse_indice/Reshape_1/shape*
T0*
Tshape0
�
7input_c_id_embedding/cond/make_sparse_indice/UpperBound
UpperBound4input_c_id_embedding/cond/make_sparse_indice/Reshape6input_c_id_embedding/cond/make_sparse_indice/Reshape_1*
T0*
out_type0
�
4input_c_id_embedding/cond/make_sparse_indice/Shape_2Shape2input_c_id_embedding/cond/make_sparse_indice/range*
T0*
out_type0
�
6input_c_id_embedding/cond/make_sparse_indice/Reshape_2Reshape7input_c_id_embedding/cond/make_sparse_indice/UpperBound4input_c_id_embedding/cond/make_sparse_indice/Shape_2*
T0*
Tshape0
�
2input_c_id_embedding/cond/make_sparse_indice/sub/yConst#^input_c_id_embedding/cond/switch_t*
value	B :*
dtype0
�
0input_c_id_embedding/cond/make_sparse_indice/subSub6input_c_id_embedding/cond/make_sparse_indice/Reshape_22input_c_id_embedding/cond/make_sparse_indice/sub/y*
T0
v
'input_c_id_embedding/cond/GatherV2/axisConst#^input_c_id_embedding/cond/switch_t*
value	B : *
dtype0
�
"input_c_id_embedding/cond/GatherV2GatherV2+input_c_id_embedding/cond/GatherV2/Switch:1-input_c_id_embedding/cond/GatherV2/Switch_1:1'input_c_id_embedding/cond/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0
�
)input_c_id_embedding/cond/GatherV2/SwitchSwitchvarlen_gather_64/ps_embed_64!input_c_id_embedding/cond/pred_id*
T0*/
_class%
#!loc:@varlen_gather_64/ps_embed_64
�
+input_c_id_embedding/cond/GatherV2/Switch_1Switchinput_c_id_embedding/GatherV2!input_c_id_embedding/cond/pred_id*
T0*0
_class&
$"loc:@input_c_id_embedding/GatherV2
�
$input_c_id_embedding/cond/SegmentSum
SegmentSum"input_c_id_embedding/cond/GatherV20input_c_id_embedding/cond/make_sparse_indice/sub*
Tindices0*
T0
�
input_c_id_embedding/cond/ShapeShapeCinput_c_id_embedding/cond/make_sparse_indice/strided_slice/Switch:1*
T0*
out_type0
�
-input_c_id_embedding/cond/strided_slice/stackConst#^input_c_id_embedding/cond/switch_t*
valueB: *
dtype0
�
/input_c_id_embedding/cond/strided_slice/stack_1Const#^input_c_id_embedding/cond/switch_t*
dtype0*
valueB:
�
/input_c_id_embedding/cond/strided_slice/stack_2Const#^input_c_id_embedding/cond/switch_t*
valueB:*
dtype0
�
'input_c_id_embedding/cond/strided_sliceStridedSliceinput_c_id_embedding/cond/Shape-input_c_id_embedding/cond/strided_slice/stack/input_c_id_embedding/cond/strided_slice/stack_1/input_c_id_embedding/cond/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
n
input_c_id_embedding/cond/sub/yConst#^input_c_id_embedding/cond/switch_t*
value	B :*
dtype0
w
input_c_id_embedding/cond/subSub'input_c_id_embedding/cond/strided_sliceinput_c_id_embedding/cond/sub/y*
T0
i
!input_c_id_embedding/cond/Shape_1Shape$input_c_id_embedding/cond/SegmentSum*
T0*
out_type0
�
/input_c_id_embedding/cond/strided_slice_1/stackConst#^input_c_id_embedding/cond/switch_t*
valueB: *
dtype0
�
1input_c_id_embedding/cond/strided_slice_1/stack_1Const#^input_c_id_embedding/cond/switch_t*
valueB:*
dtype0
�
1input_c_id_embedding/cond/strided_slice_1/stack_2Const#^input_c_id_embedding/cond/switch_t*
valueB:*
dtype0
�
)input_c_id_embedding/cond/strided_slice_1StridedSlice!input_c_id_embedding/cond/Shape_1/input_c_id_embedding/cond/strided_slice_1/stack1input_c_id_embedding/cond/strided_slice_1/stack_11input_c_id_embedding/cond/strided_slice_1/stack_2*
end_mask *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
y
input_c_id_embedding/cond/sub_1Subinput_c_id_embedding/cond/sub)input_c_id_embedding/cond/strided_slice_1*
T0
y
*input_c_id_embedding/cond/Pad/paddings/0/0Const#^input_c_id_embedding/cond/switch_t*
value	B : *
dtype0
�
(input_c_id_embedding/cond/Pad/paddings/0Pack*input_c_id_embedding/cond/Pad/paddings/0/0input_c_id_embedding/cond/sub_1*
T0*

axis *
N
�
*input_c_id_embedding/cond/Pad/paddings/1_1Const#^input_c_id_embedding/cond/switch_t*
dtype0*
valueB"        
�
&input_c_id_embedding/cond/Pad/paddingsPack(input_c_id_embedding/cond/Pad/paddings/0*input_c_id_embedding/cond/Pad/paddings/1_1*
T0*

axis *
N
�
input_c_id_embedding/cond/PadPad$input_c_id_embedding/cond/SegmentSum&input_c_id_embedding/cond/Pad/paddings*
	Tpaddings0*
T0
t
%input_c_id_embedding/cond/zeros/mul/yConst#^input_c_id_embedding/cond/switch_f*
dtype0*
value	B :@
�
#input_c_id_embedding/cond/zeros/mulMul*input_c_id_embedding/cond/zeros/mul/Switch%input_c_id_embedding/cond/zeros/mul/y*
T0
�
*input_c_id_embedding/cond/zeros/mul/SwitchSwitchinput_c_id_embedding/sub!input_c_id_embedding/cond/pred_id*
T0*+
_class!
loc:@input_c_id_embedding/sub
v
&input_c_id_embedding/cond/zeros/Less/yConst#^input_c_id_embedding/cond/switch_f*
value
B :�*
dtype0
�
$input_c_id_embedding/cond/zeros/LessLess#input_c_id_embedding/cond/zeros/mul&input_c_id_embedding/cond/zeros/Less/y*
T0
w
(input_c_id_embedding/cond/zeros/packed/1Const#^input_c_id_embedding/cond/switch_f*
value	B :@*
dtype0
�
&input_c_id_embedding/cond/zeros/packedPack*input_c_id_embedding/cond/zeros/mul/Switch(input_c_id_embedding/cond/zeros/packed/1*
T0*

axis *
N
w
%input_c_id_embedding/cond/zeros/ConstConst#^input_c_id_embedding/cond/switch_f*
valueB
 *    *
dtype0
�
input_c_id_embedding/cond/zerosFill&input_c_id_embedding/cond/zeros/packed%input_c_id_embedding/cond/zeros/Const*
T0*

index_type0
z
input_c_id_embedding/cond/MergeMergeinput_c_id_embedding/cond/zerosinput_c_id_embedding/cond/Pad*
T0*
N
N
kai_input_c_id_embeddingIdentityinput_c_id_embedding/cond/Merge*
T0
D
Reshape_2/shapeConst*
valueB"�����   *
dtype0
V
	Reshape_2Reshapekai_input_c_id_embeddingReshape_2/shape*
T0*
Tshape0
D
Reshape_3/shapeConst*
dtype0*
valueB"�����   
G
	Reshape_3Reshape	Reshape_2Reshape_3/shape*
T0*
Tshape0
?
c_info_embedding_idsPlaceholder*
dtype0*
shape:
B
c_info_embedding_cumsumPlaceholder*
dtype0*
shape:
7
level1_id_32Placeholder*
dtype0*
shape:
�
varlen_gather_32/VarlenGatherVarlenGathervarlen_embedlevel1_id_32varlen_embed_offset"/device:GPU:0*
Tindices0*
Tparams0*	
dim 
S
varlen_gather_32/Reshape/shapeConst*
valueB"����   *
dtype0
h
varlen_gather_32/ReshapeReshapelevel1_id_32varlen_gather_32/Reshape/shape*
T0*
Tshape0
F
varlen_gather_32/ShapeShapelevel1_id_32*
T0*
out_type0
R
$varlen_gather_32/strided_slice/stackConst*
valueB: *
dtype0
T
&varlen_gather_32/strided_slice/stack_1Const*
valueB:*
dtype0
T
&varlen_gather_32/strided_slice/stack_2Const*
valueB:*
dtype0
�
varlen_gather_32/strided_sliceStridedSlicevarlen_gather_32/Shape$varlen_gather_32/strided_slice/stack&varlen_gather_32/strided_slice/stack_1&varlen_gather_32/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
@
varlen_gather_32/add/yConst*
value	B :*
dtype0
\
varlen_gather_32/addAddvarlen_gather_32/strided_slicevarlen_gather_32/add/y*
T0
F
varlen_gather_32/range/startConst*
dtype0*
value	B :
F
varlen_gather_32/range/deltaConst*
dtype0*
value	B :
}
varlen_gather_32/rangeRangevarlen_gather_32/range/startvarlen_gather_32/addvarlen_gather_32/range/delta*

Tidx0
K
varlen_gather_32/SizeSizevarlen_embed_offset*
T0*
out_type0
]
 varlen_gather_32/ScatterNd/shapePackvarlen_gather_32/Size*
T0*

axis *
N
�
varlen_gather_32/ScatterNd	ScatterNdvarlen_gather_32/Reshapevarlen_gather_32/range varlen_gather_32/ScatterNd/shape*
T0*
Tindices0
@
varlen_gather_32/sub/yConst*
value	B :*
dtype0
X
varlen_gather_32/subSubvarlen_gather_32/ScatterNdvarlen_gather_32/sub/y*
T0
K
varlen_gather_32/ps_embed_32/yConst*
valueB
 *  �?*
dtype0
k
varlen_gather_32/ps_embed_32Mulvarlen_gather_32/VarlenGathervarlen_gather_32/ps_embed_32/y*
T0
N
$input_c_info_embedding/GatherV2/axisConst*
value	B : *
dtype0
�
input_c_info_embedding/GatherV2GatherV2varlen_gather_32/subc_info_embedding_ids$input_c_info_embedding/GatherV2/axis*
Tparams0*
Taxis0*
Tindices0
W
input_c_info_embedding/ShapeShapec_info_embedding_cumsum*
T0*
out_type0
X
*input_c_info_embedding/strided_slice/stackConst*
valueB: *
dtype0
Z
,input_c_info_embedding/strided_slice/stack_1Const*
valueB:*
dtype0
Z
,input_c_info_embedding/strided_slice/stack_2Const*
valueB:*
dtype0
�
$input_c_info_embedding/strided_sliceStridedSliceinput_c_info_embedding/Shape*input_c_info_embedding/strided_slice/stack,input_c_info_embedding/strided_slice/stack_1,input_c_info_embedding/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
F
input_c_info_embedding/sub/yConst*
value	B :*
dtype0
n
input_c_info_embedding/subSub$input_c_info_embedding/strided_sliceinput_c_info_embedding/sub/y*
T0
]
input_c_info_embedding/SizeSizeinput_c_info_embedding/GatherV2*
T0*
out_type0
J
 input_c_info_embedding/Greater/yConst*
dtype0*
value	B : 
q
input_c_info_embedding/GreaterGreaterinput_c_info_embedding/Size input_c_info_embedding/Greater/y*
T0
u
"input_c_info_embedding/cond/SwitchSwitchinput_c_info_embedding/Greaterinput_c_info_embedding/Greater*
T0

_
$input_c_info_embedding/cond/switch_tIdentity$input_c_info_embedding/cond/Switch:1*
T0

]
$input_c_info_embedding/cond/switch_fIdentity"input_c_info_embedding/cond/Switch*
T0

X
#input_c_info_embedding/cond/pred_idIdentityinput_c_info_embedding/Greater*
T0

�
Binput_c_info_embedding/cond/make_sparse_indice/strided_slice/stackConst%^input_c_info_embedding/cond/switch_t*
valueB:
���������*
dtype0
�
Dinput_c_info_embedding/cond/make_sparse_indice/strided_slice/stack_1Const%^input_c_info_embedding/cond/switch_t*
valueB: *
dtype0
�
Dinput_c_info_embedding/cond/make_sparse_indice/strided_slice/stack_2Const%^input_c_info_embedding/cond/switch_t*
valueB:*
dtype0
�
<input_c_info_embedding/cond/make_sparse_indice/strided_sliceStridedSliceEinput_c_info_embedding/cond/make_sparse_indice/strided_slice/Switch:1Binput_c_info_embedding/cond/make_sparse_indice/strided_slice/stackDinput_c_info_embedding/cond/make_sparse_indice/strided_slice/stack_1Dinput_c_info_embedding/cond/make_sparse_indice/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
�
Cinput_c_info_embedding/cond/make_sparse_indice/strided_slice/SwitchSwitchc_info_embedding_cumsum#input_c_info_embedding/cond/pred_id*
T0**
_class 
loc:@c_info_embedding_cumsum
�
:input_c_info_embedding/cond/make_sparse_indice/range/startConst%^input_c_info_embedding/cond/switch_t*
dtype0*
value	B : 
�
:input_c_info_embedding/cond/make_sparse_indice/range/deltaConst%^input_c_info_embedding/cond/switch_t*
value	B :*
dtype0
�
4input_c_info_embedding/cond/make_sparse_indice/rangeRange:input_c_info_embedding/cond/make_sparse_indice/range/start<input_c_info_embedding/cond/make_sparse_indice/strided_slice:input_c_info_embedding/cond/make_sparse_indice/range/delta*

Tidx0
�
4input_c_info_embedding/cond/make_sparse_indice/ShapeShapeEinput_c_info_embedding/cond/make_sparse_indice/strided_slice/Switch:1*
T0*
out_type0
�
Dinput_c_info_embedding/cond/make_sparse_indice/strided_slice_1/stackConst%^input_c_info_embedding/cond/switch_t*
dtype0*
valueB:
���������
�
Finput_c_info_embedding/cond/make_sparse_indice/strided_slice_1/stack_1Const%^input_c_info_embedding/cond/switch_t*
valueB: *
dtype0
�
Finput_c_info_embedding/cond/make_sparse_indice/strided_slice_1/stack_2Const%^input_c_info_embedding/cond/switch_t*
valueB:*
dtype0
�
>input_c_info_embedding/cond/make_sparse_indice/strided_slice_1StridedSlice4input_c_info_embedding/cond/make_sparse_indice/ShapeDinput_c_info_embedding/cond/make_sparse_indice/strided_slice_1/stackFinput_c_info_embedding/cond/make_sparse_indice/strided_slice_1/stack_1Finput_c_info_embedding/cond/make_sparse_indice/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
�
6input_c_info_embedding/cond/make_sparse_indice/Shape_1Shape4input_c_info_embedding/cond/make_sparse_indice/range*
T0*
out_type0
�
Dinput_c_info_embedding/cond/make_sparse_indice/strided_slice_2/stackConst%^input_c_info_embedding/cond/switch_t*
valueB:
���������*
dtype0
�
Finput_c_info_embedding/cond/make_sparse_indice/strided_slice_2/stack_1Const%^input_c_info_embedding/cond/switch_t*
valueB: *
dtype0
�
Finput_c_info_embedding/cond/make_sparse_indice/strided_slice_2/stack_2Const%^input_c_info_embedding/cond/switch_t*
dtype0*
valueB:
�
>input_c_info_embedding/cond/make_sparse_indice/strided_slice_2StridedSlice6input_c_info_embedding/cond/make_sparse_indice/Shape_1Dinput_c_info_embedding/cond/make_sparse_indice/strided_slice_2/stackFinput_c_info_embedding/cond/make_sparse_indice/strided_slice_2/stack_1Finput_c_info_embedding/cond/make_sparse_indice/strided_slice_2/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
�
>input_c_info_embedding/cond/make_sparse_indice/Reshape/shape/0Const%^input_c_info_embedding/cond/switch_t*
valueB :
���������*
dtype0
�
<input_c_info_embedding/cond/make_sparse_indice/Reshape/shapePack>input_c_info_embedding/cond/make_sparse_indice/Reshape/shape/0>input_c_info_embedding/cond/make_sparse_indice/strided_slice_1*
T0*

axis *
N
�
6input_c_info_embedding/cond/make_sparse_indice/ReshapeReshapeEinput_c_info_embedding/cond/make_sparse_indice/strided_slice/Switch:1<input_c_info_embedding/cond/make_sparse_indice/Reshape/shape*
T0*
Tshape0
�
@input_c_info_embedding/cond/make_sparse_indice/Reshape_1/shape/0Const%^input_c_info_embedding/cond/switch_t*
dtype0*
valueB :
���������
�
>input_c_info_embedding/cond/make_sparse_indice/Reshape_1/shapePack@input_c_info_embedding/cond/make_sparse_indice/Reshape_1/shape/0>input_c_info_embedding/cond/make_sparse_indice/strided_slice_2*
T0*

axis *
N
�
8input_c_info_embedding/cond/make_sparse_indice/Reshape_1Reshape4input_c_info_embedding/cond/make_sparse_indice/range>input_c_info_embedding/cond/make_sparse_indice/Reshape_1/shape*
T0*
Tshape0
�
9input_c_info_embedding/cond/make_sparse_indice/UpperBound
UpperBound6input_c_info_embedding/cond/make_sparse_indice/Reshape8input_c_info_embedding/cond/make_sparse_indice/Reshape_1*
T0*
out_type0
�
6input_c_info_embedding/cond/make_sparse_indice/Shape_2Shape4input_c_info_embedding/cond/make_sparse_indice/range*
T0*
out_type0
�
8input_c_info_embedding/cond/make_sparse_indice/Reshape_2Reshape9input_c_info_embedding/cond/make_sparse_indice/UpperBound6input_c_info_embedding/cond/make_sparse_indice/Shape_2*
T0*
Tshape0
�
4input_c_info_embedding/cond/make_sparse_indice/sub/yConst%^input_c_info_embedding/cond/switch_t*
value	B :*
dtype0
�
2input_c_info_embedding/cond/make_sparse_indice/subSub8input_c_info_embedding/cond/make_sparse_indice/Reshape_24input_c_info_embedding/cond/make_sparse_indice/sub/y*
T0
z
)input_c_info_embedding/cond/GatherV2/axisConst%^input_c_info_embedding/cond/switch_t*
dtype0*
value	B : 
�
$input_c_info_embedding/cond/GatherV2GatherV2-input_c_info_embedding/cond/GatherV2/Switch:1/input_c_info_embedding/cond/GatherV2/Switch_1:1)input_c_info_embedding/cond/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0
�
+input_c_info_embedding/cond/GatherV2/SwitchSwitchvarlen_gather_32/ps_embed_32#input_c_info_embedding/cond/pred_id*
T0*/
_class%
#!loc:@varlen_gather_32/ps_embed_32
�
-input_c_info_embedding/cond/GatherV2/Switch_1Switchinput_c_info_embedding/GatherV2#input_c_info_embedding/cond/pred_id*
T0*2
_class(
&$loc:@input_c_info_embedding/GatherV2
�
&input_c_info_embedding/cond/SegmentSum
SegmentSum$input_c_info_embedding/cond/GatherV22input_c_info_embedding/cond/make_sparse_indice/sub*
T0*
Tindices0
�
!input_c_info_embedding/cond/ShapeShapeEinput_c_info_embedding/cond/make_sparse_indice/strided_slice/Switch:1*
T0*
out_type0
�
/input_c_info_embedding/cond/strided_slice/stackConst%^input_c_info_embedding/cond/switch_t*
valueB: *
dtype0
�
1input_c_info_embedding/cond/strided_slice/stack_1Const%^input_c_info_embedding/cond/switch_t*
dtype0*
valueB:
�
1input_c_info_embedding/cond/strided_slice/stack_2Const%^input_c_info_embedding/cond/switch_t*
dtype0*
valueB:
�
)input_c_info_embedding/cond/strided_sliceStridedSlice!input_c_info_embedding/cond/Shape/input_c_info_embedding/cond/strided_slice/stack1input_c_info_embedding/cond/strided_slice/stack_11input_c_info_embedding/cond/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
r
!input_c_info_embedding/cond/sub/yConst%^input_c_info_embedding/cond/switch_t*
value	B :*
dtype0
}
input_c_info_embedding/cond/subSub)input_c_info_embedding/cond/strided_slice!input_c_info_embedding/cond/sub/y*
T0
m
#input_c_info_embedding/cond/Shape_1Shape&input_c_info_embedding/cond/SegmentSum*
T0*
out_type0
�
1input_c_info_embedding/cond/strided_slice_1/stackConst%^input_c_info_embedding/cond/switch_t*
dtype0*
valueB: 
�
3input_c_info_embedding/cond/strided_slice_1/stack_1Const%^input_c_info_embedding/cond/switch_t*
valueB:*
dtype0
�
3input_c_info_embedding/cond/strided_slice_1/stack_2Const%^input_c_info_embedding/cond/switch_t*
dtype0*
valueB:
�
+input_c_info_embedding/cond/strided_slice_1StridedSlice#input_c_info_embedding/cond/Shape_11input_c_info_embedding/cond/strided_slice_1/stack3input_c_info_embedding/cond/strided_slice_1/stack_13input_c_info_embedding/cond/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 

!input_c_info_embedding/cond/sub_1Subinput_c_info_embedding/cond/sub+input_c_info_embedding/cond/strided_slice_1*
T0
}
,input_c_info_embedding/cond/Pad/paddings/0/0Const%^input_c_info_embedding/cond/switch_t*
dtype0*
value	B : 
�
*input_c_info_embedding/cond/Pad/paddings/0Pack,input_c_info_embedding/cond/Pad/paddings/0/0!input_c_info_embedding/cond/sub_1*
N*
T0*

axis 
�
,input_c_info_embedding/cond/Pad/paddings/1_1Const%^input_c_info_embedding/cond/switch_t*
valueB"        *
dtype0
�
(input_c_info_embedding/cond/Pad/paddingsPack*input_c_info_embedding/cond/Pad/paddings/0,input_c_info_embedding/cond/Pad/paddings/1_1*
T0*

axis *
N
�
input_c_info_embedding/cond/PadPad&input_c_info_embedding/cond/SegmentSum(input_c_info_embedding/cond/Pad/paddings*
T0*
	Tpaddings0
x
'input_c_info_embedding/cond/zeros/mul/yConst%^input_c_info_embedding/cond/switch_f*
value	B : *
dtype0
�
%input_c_info_embedding/cond/zeros/mulMul,input_c_info_embedding/cond/zeros/mul/Switch'input_c_info_embedding/cond/zeros/mul/y*
T0
�
,input_c_info_embedding/cond/zeros/mul/SwitchSwitchinput_c_info_embedding/sub#input_c_info_embedding/cond/pred_id*
T0*-
_class#
!loc:@input_c_info_embedding/sub
z
(input_c_info_embedding/cond/zeros/Less/yConst%^input_c_info_embedding/cond/switch_f*
dtype0*
value
B :�
�
&input_c_info_embedding/cond/zeros/LessLess%input_c_info_embedding/cond/zeros/mul(input_c_info_embedding/cond/zeros/Less/y*
T0
{
*input_c_info_embedding/cond/zeros/packed/1Const%^input_c_info_embedding/cond/switch_f*
dtype0*
value	B : 
�
(input_c_info_embedding/cond/zeros/packedPack,input_c_info_embedding/cond/zeros/mul/Switch*input_c_info_embedding/cond/zeros/packed/1*
T0*

axis *
N
{
'input_c_info_embedding/cond/zeros/ConstConst%^input_c_info_embedding/cond/switch_f*
valueB
 *    *
dtype0
�
!input_c_info_embedding/cond/zerosFill(input_c_info_embedding/cond/zeros/packed'input_c_info_embedding/cond/zeros/Const*
T0*

index_type0
�
!input_c_info_embedding/cond/MergeMerge!input_c_info_embedding/cond/zerosinput_c_info_embedding/cond/Pad*
T0*
N
R
kai_input_c_info_embeddingIdentity!input_c_info_embedding/cond/Merge*
T0
D
Reshape_4/shapeConst*
valueB"�����   *
dtype0
X
	Reshape_4Reshapekai_input_c_info_embeddingReshape_4/shape*
T0*
Tshape0
D
Reshape_5/shapeConst*
valueB"�����   *
dtype0
G
	Reshape_5Reshape	Reshape_4Reshape_5/shape*
T0*
Tshape0
A
position_embedding_idsPlaceholder*
shape:*
dtype0
D
position_embedding_cumsumPlaceholder*
shape:*
dtype0
6
level1_id_8Placeholder*
dtype0*
shape:
�
varlen_gather_8/VarlenGatherVarlenGathervarlen_embedlevel1_id_8varlen_embed_offset"/device:GPU:0*	
dim*
Tindices0*
Tparams0
R
varlen_gather_8/Reshape/shapeConst*
valueB"����   *
dtype0
e
varlen_gather_8/ReshapeReshapelevel1_id_8varlen_gather_8/Reshape/shape*
T0*
Tshape0
D
varlen_gather_8/ShapeShapelevel1_id_8*
T0*
out_type0
Q
#varlen_gather_8/strided_slice/stackConst*
valueB: *
dtype0
S
%varlen_gather_8/strided_slice/stack_1Const*
valueB:*
dtype0
S
%varlen_gather_8/strided_slice/stack_2Const*
valueB:*
dtype0
�
varlen_gather_8/strided_sliceStridedSlicevarlen_gather_8/Shape#varlen_gather_8/strided_slice/stack%varlen_gather_8/strided_slice/stack_1%varlen_gather_8/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask
?
varlen_gather_8/add/yConst*
value	B :*
dtype0
Y
varlen_gather_8/addAddvarlen_gather_8/strided_slicevarlen_gather_8/add/y*
T0
E
varlen_gather_8/range/startConst*
value	B :*
dtype0
E
varlen_gather_8/range/deltaConst*
value	B :*
dtype0
y
varlen_gather_8/rangeRangevarlen_gather_8/range/startvarlen_gather_8/addvarlen_gather_8/range/delta*

Tidx0
J
varlen_gather_8/SizeSizevarlen_embed_offset*
T0*
out_type0
[
varlen_gather_8/ScatterNd/shapePackvarlen_gather_8/Size*
T0*

axis *
N
�
varlen_gather_8/ScatterNd	ScatterNdvarlen_gather_8/Reshapevarlen_gather_8/rangevarlen_gather_8/ScatterNd/shape*
Tindices0*
T0
?
varlen_gather_8/sub/yConst*
value	B :*
dtype0
U
varlen_gather_8/subSubvarlen_gather_8/ScatterNdvarlen_gather_8/sub/y*
T0
I
varlen_gather_8/ps_embed_8/yConst*
dtype0*
valueB
 *  �?
f
varlen_gather_8/ps_embed_8Mulvarlen_gather_8/VarlenGathervarlen_gather_8/ps_embed_8/y*
T0
P
&input_position_embedding/GatherV2/axisConst*
value	B : *
dtype0
�
!input_position_embedding/GatherV2GatherV2varlen_gather_8/subposition_embedding_ids&input_position_embedding/GatherV2/axis*
Tindices0*
Tparams0*
Taxis0
[
input_position_embedding/ShapeShapeposition_embedding_cumsum*
T0*
out_type0
Z
,input_position_embedding/strided_slice/stackConst*
valueB: *
dtype0
\
.input_position_embedding/strided_slice/stack_1Const*
dtype0*
valueB:
\
.input_position_embedding/strided_slice/stack_2Const*
valueB:*
dtype0
�
&input_position_embedding/strided_sliceStridedSliceinput_position_embedding/Shape,input_position_embedding/strided_slice/stack.input_position_embedding/strided_slice/stack_1.input_position_embedding/strided_slice/stack_2*
end_mask *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
H
input_position_embedding/sub/yConst*
value	B :*
dtype0
t
input_position_embedding/subSub&input_position_embedding/strided_sliceinput_position_embedding/sub/y*
T0
a
input_position_embedding/SizeSize!input_position_embedding/GatherV2*
T0*
out_type0
L
"input_position_embedding/Greater/yConst*
value	B : *
dtype0
w
 input_position_embedding/GreaterGreaterinput_position_embedding/Size"input_position_embedding/Greater/y*
T0
{
$input_position_embedding/cond/SwitchSwitch input_position_embedding/Greater input_position_embedding/Greater*
T0

c
&input_position_embedding/cond/switch_tIdentity&input_position_embedding/cond/Switch:1*
T0

a
&input_position_embedding/cond/switch_fIdentity$input_position_embedding/cond/Switch*
T0

\
%input_position_embedding/cond/pred_idIdentity input_position_embedding/Greater*
T0

�
Dinput_position_embedding/cond/make_sparse_indice/strided_slice/stackConst'^input_position_embedding/cond/switch_t*
dtype0*
valueB:
���������
�
Finput_position_embedding/cond/make_sparse_indice/strided_slice/stack_1Const'^input_position_embedding/cond/switch_t*
valueB: *
dtype0
�
Finput_position_embedding/cond/make_sparse_indice/strided_slice/stack_2Const'^input_position_embedding/cond/switch_t*
valueB:*
dtype0
�
>input_position_embedding/cond/make_sparse_indice/strided_sliceStridedSliceGinput_position_embedding/cond/make_sparse_indice/strided_slice/Switch:1Dinput_position_embedding/cond/make_sparse_indice/strided_slice/stackFinput_position_embedding/cond/make_sparse_indice/strided_slice/stack_1Finput_position_embedding/cond/make_sparse_indice/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
�
Einput_position_embedding/cond/make_sparse_indice/strided_slice/SwitchSwitchposition_embedding_cumsum%input_position_embedding/cond/pred_id*
T0*,
_class"
 loc:@position_embedding_cumsum
�
<input_position_embedding/cond/make_sparse_indice/range/startConst'^input_position_embedding/cond/switch_t*
value	B : *
dtype0
�
<input_position_embedding/cond/make_sparse_indice/range/deltaConst'^input_position_embedding/cond/switch_t*
value	B :*
dtype0
�
6input_position_embedding/cond/make_sparse_indice/rangeRange<input_position_embedding/cond/make_sparse_indice/range/start>input_position_embedding/cond/make_sparse_indice/strided_slice<input_position_embedding/cond/make_sparse_indice/range/delta*

Tidx0
�
6input_position_embedding/cond/make_sparse_indice/ShapeShapeGinput_position_embedding/cond/make_sparse_indice/strided_slice/Switch:1*
T0*
out_type0
�
Finput_position_embedding/cond/make_sparse_indice/strided_slice_1/stackConst'^input_position_embedding/cond/switch_t*
dtype0*
valueB:
���������
�
Hinput_position_embedding/cond/make_sparse_indice/strided_slice_1/stack_1Const'^input_position_embedding/cond/switch_t*
dtype0*
valueB: 
�
Hinput_position_embedding/cond/make_sparse_indice/strided_slice_1/stack_2Const'^input_position_embedding/cond/switch_t*
valueB:*
dtype0
�
@input_position_embedding/cond/make_sparse_indice/strided_slice_1StridedSlice6input_position_embedding/cond/make_sparse_indice/ShapeFinput_position_embedding/cond/make_sparse_indice/strided_slice_1/stackHinput_position_embedding/cond/make_sparse_indice/strided_slice_1/stack_1Hinput_position_embedding/cond/make_sparse_indice/strided_slice_1/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
�
8input_position_embedding/cond/make_sparse_indice/Shape_1Shape6input_position_embedding/cond/make_sparse_indice/range*
T0*
out_type0
�
Finput_position_embedding/cond/make_sparse_indice/strided_slice_2/stackConst'^input_position_embedding/cond/switch_t*
dtype0*
valueB:
���������
�
Hinput_position_embedding/cond/make_sparse_indice/strided_slice_2/stack_1Const'^input_position_embedding/cond/switch_t*
valueB: *
dtype0
�
Hinput_position_embedding/cond/make_sparse_indice/strided_slice_2/stack_2Const'^input_position_embedding/cond/switch_t*
valueB:*
dtype0
�
@input_position_embedding/cond/make_sparse_indice/strided_slice_2StridedSlice8input_position_embedding/cond/make_sparse_indice/Shape_1Finput_position_embedding/cond/make_sparse_indice/strided_slice_2/stackHinput_position_embedding/cond/make_sparse_indice/strided_slice_2/stack_1Hinput_position_embedding/cond/make_sparse_indice/strided_slice_2/stack_2*
end_mask *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
�
@input_position_embedding/cond/make_sparse_indice/Reshape/shape/0Const'^input_position_embedding/cond/switch_t*
valueB :
���������*
dtype0
�
>input_position_embedding/cond/make_sparse_indice/Reshape/shapePack@input_position_embedding/cond/make_sparse_indice/Reshape/shape/0@input_position_embedding/cond/make_sparse_indice/strided_slice_1*
T0*

axis *
N
�
8input_position_embedding/cond/make_sparse_indice/ReshapeReshapeGinput_position_embedding/cond/make_sparse_indice/strided_slice/Switch:1>input_position_embedding/cond/make_sparse_indice/Reshape/shape*
T0*
Tshape0
�
Binput_position_embedding/cond/make_sparse_indice/Reshape_1/shape/0Const'^input_position_embedding/cond/switch_t*
valueB :
���������*
dtype0
�
@input_position_embedding/cond/make_sparse_indice/Reshape_1/shapePackBinput_position_embedding/cond/make_sparse_indice/Reshape_1/shape/0@input_position_embedding/cond/make_sparse_indice/strided_slice_2*
T0*

axis *
N
�
:input_position_embedding/cond/make_sparse_indice/Reshape_1Reshape6input_position_embedding/cond/make_sparse_indice/range@input_position_embedding/cond/make_sparse_indice/Reshape_1/shape*
T0*
Tshape0
�
;input_position_embedding/cond/make_sparse_indice/UpperBound
UpperBound8input_position_embedding/cond/make_sparse_indice/Reshape:input_position_embedding/cond/make_sparse_indice/Reshape_1*
T0*
out_type0
�
8input_position_embedding/cond/make_sparse_indice/Shape_2Shape6input_position_embedding/cond/make_sparse_indice/range*
T0*
out_type0
�
:input_position_embedding/cond/make_sparse_indice/Reshape_2Reshape;input_position_embedding/cond/make_sparse_indice/UpperBound8input_position_embedding/cond/make_sparse_indice/Shape_2*
T0*
Tshape0
�
6input_position_embedding/cond/make_sparse_indice/sub/yConst'^input_position_embedding/cond/switch_t*
value	B :*
dtype0
�
4input_position_embedding/cond/make_sparse_indice/subSub:input_position_embedding/cond/make_sparse_indice/Reshape_26input_position_embedding/cond/make_sparse_indice/sub/y*
T0
~
+input_position_embedding/cond/GatherV2/axisConst'^input_position_embedding/cond/switch_t*
value	B : *
dtype0
�
&input_position_embedding/cond/GatherV2GatherV2/input_position_embedding/cond/GatherV2/Switch:11input_position_embedding/cond/GatherV2/Switch_1:1+input_position_embedding/cond/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0
�
-input_position_embedding/cond/GatherV2/SwitchSwitchvarlen_gather_8/ps_embed_8%input_position_embedding/cond/pred_id*
T0*-
_class#
!loc:@varlen_gather_8/ps_embed_8
�
/input_position_embedding/cond/GatherV2/Switch_1Switch!input_position_embedding/GatherV2%input_position_embedding/cond/pred_id*
T0*4
_class*
(&loc:@input_position_embedding/GatherV2
�
(input_position_embedding/cond/SegmentSum
SegmentSum&input_position_embedding/cond/GatherV24input_position_embedding/cond/make_sparse_indice/sub*
Tindices0*
T0
�
#input_position_embedding/cond/ShapeShapeGinput_position_embedding/cond/make_sparse_indice/strided_slice/Switch:1*
T0*
out_type0
�
1input_position_embedding/cond/strided_slice/stackConst'^input_position_embedding/cond/switch_t*
valueB: *
dtype0
�
3input_position_embedding/cond/strided_slice/stack_1Const'^input_position_embedding/cond/switch_t*
valueB:*
dtype0
�
3input_position_embedding/cond/strided_slice/stack_2Const'^input_position_embedding/cond/switch_t*
valueB:*
dtype0
�
+input_position_embedding/cond/strided_sliceStridedSlice#input_position_embedding/cond/Shape1input_position_embedding/cond/strided_slice/stack3input_position_embedding/cond/strided_slice/stack_13input_position_embedding/cond/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
v
#input_position_embedding/cond/sub/yConst'^input_position_embedding/cond/switch_t*
value	B :*
dtype0
�
!input_position_embedding/cond/subSub+input_position_embedding/cond/strided_slice#input_position_embedding/cond/sub/y*
T0
q
%input_position_embedding/cond/Shape_1Shape(input_position_embedding/cond/SegmentSum*
T0*
out_type0
�
3input_position_embedding/cond/strided_slice_1/stackConst'^input_position_embedding/cond/switch_t*
valueB: *
dtype0
�
5input_position_embedding/cond/strided_slice_1/stack_1Const'^input_position_embedding/cond/switch_t*
valueB:*
dtype0
�
5input_position_embedding/cond/strided_slice_1/stack_2Const'^input_position_embedding/cond/switch_t*
valueB:*
dtype0
�
-input_position_embedding/cond/strided_slice_1StridedSlice%input_position_embedding/cond/Shape_13input_position_embedding/cond/strided_slice_1/stack5input_position_embedding/cond/strided_slice_1/stack_15input_position_embedding/cond/strided_slice_1/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0
�
#input_position_embedding/cond/sub_1Sub!input_position_embedding/cond/sub-input_position_embedding/cond/strided_slice_1*
T0
�
.input_position_embedding/cond/Pad/paddings/0/0Const'^input_position_embedding/cond/switch_t*
dtype0*
value	B : 
�
,input_position_embedding/cond/Pad/paddings/0Pack.input_position_embedding/cond/Pad/paddings/0/0#input_position_embedding/cond/sub_1*
T0*

axis *
N
�
.input_position_embedding/cond/Pad/paddings/1_1Const'^input_position_embedding/cond/switch_t*
valueB"        *
dtype0
�
*input_position_embedding/cond/Pad/paddingsPack,input_position_embedding/cond/Pad/paddings/0.input_position_embedding/cond/Pad/paddings/1_1*
T0*

axis *
N
�
!input_position_embedding/cond/PadPad(input_position_embedding/cond/SegmentSum*input_position_embedding/cond/Pad/paddings*
T0*
	Tpaddings0
|
)input_position_embedding/cond/zeros/mul/yConst'^input_position_embedding/cond/switch_f*
value	B :*
dtype0
�
'input_position_embedding/cond/zeros/mulMul.input_position_embedding/cond/zeros/mul/Switch)input_position_embedding/cond/zeros/mul/y*
T0
�
.input_position_embedding/cond/zeros/mul/SwitchSwitchinput_position_embedding/sub%input_position_embedding/cond/pred_id*
T0*/
_class%
#!loc:@input_position_embedding/sub
~
*input_position_embedding/cond/zeros/Less/yConst'^input_position_embedding/cond/switch_f*
value
B :�*
dtype0
�
(input_position_embedding/cond/zeros/LessLess'input_position_embedding/cond/zeros/mul*input_position_embedding/cond/zeros/Less/y*
T0

,input_position_embedding/cond/zeros/packed/1Const'^input_position_embedding/cond/switch_f*
dtype0*
value	B :
�
*input_position_embedding/cond/zeros/packedPack.input_position_embedding/cond/zeros/mul/Switch,input_position_embedding/cond/zeros/packed/1*
T0*

axis *
N

)input_position_embedding/cond/zeros/ConstConst'^input_position_embedding/cond/switch_f*
valueB
 *    *
dtype0
�
#input_position_embedding/cond/zerosFill*input_position_embedding/cond/zeros/packed)input_position_embedding/cond/zeros/Const*
T0*

index_type0
�
#input_position_embedding/cond/MergeMerge#input_position_embedding/cond/zeros!input_position_embedding/cond/Pad*
T0*
N
V
kai_input_position_embeddingIdentity#input_position_embedding/cond/Merge*
T0
D
Reshape_6/shapeConst*
valueB"����   *
dtype0
Z
	Reshape_6Reshapekai_input_position_embeddingReshape_6/shape*
T0*
Tshape0
D
Reshape_7/shapeConst*
dtype0*
valueB"����   
G
	Reshape_7Reshape	Reshape_6Reshape_7/shape*
T0*
Tshape0
>
concat/axisConst*
valueB :
���������*
dtype0
i
concatConcatV2	Reshape_1	Reshape_3	Reshape_5	Reshape_7concat/axis*
N*

Tidx0*
T0
�
8expand_xtr/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"P     **
_class 
loc:@expand_xtr/dense/kernel
�
6expand_xtr/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *�-ν**
_class 
loc:@expand_xtr/dense/kernel
�
6expand_xtr/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *�-�=**
_class 
loc:@expand_xtr/dense/kernel*
dtype0
�
@expand_xtr/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform8expand_xtr/dense/kernel/Initializer/random_uniform/shape*
T0**
_class 
loc:@expand_xtr/dense/kernel*
dtype0*
seed2 *

seed 
�
6expand_xtr/dense/kernel/Initializer/random_uniform/subSub6expand_xtr/dense/kernel/Initializer/random_uniform/max6expand_xtr/dense/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@expand_xtr/dense/kernel
�
6expand_xtr/dense/kernel/Initializer/random_uniform/mulMul@expand_xtr/dense/kernel/Initializer/random_uniform/RandomUniform6expand_xtr/dense/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@expand_xtr/dense/kernel
�
2expand_xtr/dense/kernel/Initializer/random_uniformAdd6expand_xtr/dense/kernel/Initializer/random_uniform/mul6expand_xtr/dense/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@expand_xtr/dense/kernel
�
expand_xtr/dense/kernel
VariableV2*
dtype0*
	container *
shape:
��*
shared_name **
_class 
loc:@expand_xtr/dense/kernel
�
expand_xtr/dense/kernel/AssignAssignexpand_xtr/dense/kernel2expand_xtr/dense/kernel/Initializer/random_uniform*
use_locking(*
T0**
_class 
loc:@expand_xtr/dense/kernel*
validate_shape(
v
expand_xtr/dense/kernel/readIdentityexpand_xtr/dense/kernel*
T0**
_class 
loc:@expand_xtr/dense/kernel
�
'expand_xtr/dense/bias/Initializer/zerosConst*
dtype0*
valueB�*    *(
_class
loc:@expand_xtr/dense/bias
�
expand_xtr/dense/bias
VariableV2*
dtype0*
	container *
shape:�*
shared_name *(
_class
loc:@expand_xtr/dense/bias
�
expand_xtr/dense/bias/AssignAssignexpand_xtr/dense/bias'expand_xtr/dense/bias/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@expand_xtr/dense/bias*
validate_shape(
p
expand_xtr/dense/bias/readIdentityexpand_xtr/dense/bias*
T0*(
_class
loc:@expand_xtr/dense/bias
v
expand_xtr/dense/MatMulMatMulconcatexpand_xtr/dense/kernel/read*
transpose_a( *
transpose_b( *
T0
x
expand_xtr/dense/BiasAddBiasAddexpand_xtr/dense/MatMulexpand_xtr/dense/bias/read*
T0*
data_formatNHWC
M
 expand_xtr/dense/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
j
expand_xtr/dense/LeakyRelu/mulMul expand_xtr/dense/LeakyRelu/alphaexpand_xtr/dense/BiasAdd*
T0
h
expand_xtr/dense/LeakyReluMaximumexpand_xtr/dense/LeakyRelu/mulexpand_xtr/dense/BiasAdd*
T0
�
:expand_xtr/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"   �   *,
_class"
 loc:@expand_xtr/dense_1/kernel*
dtype0
�
8expand_xtr/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *   �*,
_class"
 loc:@expand_xtr/dense_1/kernel*
dtype0
�
8expand_xtr/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *   >*,
_class"
 loc:@expand_xtr/dense_1/kernel*
dtype0
�
Bexpand_xtr/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform:expand_xtr/dense_1/kernel/Initializer/random_uniform/shape*

seed *
T0*,
_class"
 loc:@expand_xtr/dense_1/kernel*
dtype0*
seed2 
�
8expand_xtr/dense_1/kernel/Initializer/random_uniform/subSub8expand_xtr/dense_1/kernel/Initializer/random_uniform/max8expand_xtr/dense_1/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@expand_xtr/dense_1/kernel
�
8expand_xtr/dense_1/kernel/Initializer/random_uniform/mulMulBexpand_xtr/dense_1/kernel/Initializer/random_uniform/RandomUniform8expand_xtr/dense_1/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@expand_xtr/dense_1/kernel
�
4expand_xtr/dense_1/kernel/Initializer/random_uniformAdd8expand_xtr/dense_1/kernel/Initializer/random_uniform/mul8expand_xtr/dense_1/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@expand_xtr/dense_1/kernel
�
expand_xtr/dense_1/kernel
VariableV2*
shared_name *,
_class"
 loc:@expand_xtr/dense_1/kernel*
dtype0*
	container *
shape:
��
�
 expand_xtr/dense_1/kernel/AssignAssignexpand_xtr/dense_1/kernel4expand_xtr/dense_1/kernel/Initializer/random_uniform*
T0*,
_class"
 loc:@expand_xtr/dense_1/kernel*
validate_shape(*
use_locking(
|
expand_xtr/dense_1/kernel/readIdentityexpand_xtr/dense_1/kernel*
T0*,
_class"
 loc:@expand_xtr/dense_1/kernel
�
)expand_xtr/dense_1/bias/Initializer/zerosConst*
valueB�*    **
_class 
loc:@expand_xtr/dense_1/bias*
dtype0
�
expand_xtr/dense_1/bias
VariableV2*
shared_name **
_class 
loc:@expand_xtr/dense_1/bias*
dtype0*
	container *
shape:�
�
expand_xtr/dense_1/bias/AssignAssignexpand_xtr/dense_1/bias)expand_xtr/dense_1/bias/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@expand_xtr/dense_1/bias*
validate_shape(
v
expand_xtr/dense_1/bias/readIdentityexpand_xtr/dense_1/bias*
T0**
_class 
loc:@expand_xtr/dense_1/bias
�
expand_xtr/dense_1/MatMulMatMulexpand_xtr/dense/LeakyReluexpand_xtr/dense_1/kernel/read*
T0*
transpose_a( *
transpose_b( 
~
expand_xtr/dense_1/BiasAddBiasAddexpand_xtr/dense_1/MatMulexpand_xtr/dense_1/bias/read*
T0*
data_formatNHWC
O
"expand_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
p
 expand_xtr/dense_1/LeakyRelu/mulMul"expand_xtr/dense_1/LeakyRelu/alphaexpand_xtr/dense_1/BiasAdd*
T0
n
expand_xtr/dense_1/LeakyReluMaximum expand_xtr/dense_1/LeakyRelu/mulexpand_xtr/dense_1/BiasAdd*
T0
�
:expand_xtr/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"�   @   *,
_class"
 loc:@expand_xtr/dense_2/kernel*
dtype0
�
8expand_xtr/dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *�5�*,
_class"
 loc:@expand_xtr/dense_2/kernel
�
8expand_xtr/dense_2/kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *�5>*,
_class"
 loc:@expand_xtr/dense_2/kernel
�
Bexpand_xtr/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform:expand_xtr/dense_2/kernel/Initializer/random_uniform/shape*
T0*,
_class"
 loc:@expand_xtr/dense_2/kernel*
dtype0*
seed2 *

seed 
�
8expand_xtr/dense_2/kernel/Initializer/random_uniform/subSub8expand_xtr/dense_2/kernel/Initializer/random_uniform/max8expand_xtr/dense_2/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@expand_xtr/dense_2/kernel
�
8expand_xtr/dense_2/kernel/Initializer/random_uniform/mulMulBexpand_xtr/dense_2/kernel/Initializer/random_uniform/RandomUniform8expand_xtr/dense_2/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@expand_xtr/dense_2/kernel
�
4expand_xtr/dense_2/kernel/Initializer/random_uniformAdd8expand_xtr/dense_2/kernel/Initializer/random_uniform/mul8expand_xtr/dense_2/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@expand_xtr/dense_2/kernel
�
expand_xtr/dense_2/kernel
VariableV2*
shape:	�@*
shared_name *,
_class"
 loc:@expand_xtr/dense_2/kernel*
dtype0*
	container 
�
 expand_xtr/dense_2/kernel/AssignAssignexpand_xtr/dense_2/kernel4expand_xtr/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@expand_xtr/dense_2/kernel
|
expand_xtr/dense_2/kernel/readIdentityexpand_xtr/dense_2/kernel*
T0*,
_class"
 loc:@expand_xtr/dense_2/kernel
�
)expand_xtr/dense_2/bias/Initializer/zerosConst*
valueB@*    **
_class 
loc:@expand_xtr/dense_2/bias*
dtype0
�
expand_xtr/dense_2/bias
VariableV2*
dtype0*
	container *
shape:@*
shared_name **
_class 
loc:@expand_xtr/dense_2/bias
�
expand_xtr/dense_2/bias/AssignAssignexpand_xtr/dense_2/bias)expand_xtr/dense_2/bias/Initializer/zeros*
validate_shape(*
use_locking(*
T0**
_class 
loc:@expand_xtr/dense_2/bias
v
expand_xtr/dense_2/bias/readIdentityexpand_xtr/dense_2/bias*
T0**
_class 
loc:@expand_xtr/dense_2/bias
�
expand_xtr/dense_2/MatMulMatMulexpand_xtr/dense_1/LeakyReluexpand_xtr/dense_2/kernel/read*
transpose_a( *
transpose_b( *
T0
~
expand_xtr/dense_2/BiasAddBiasAddexpand_xtr/dense_2/MatMulexpand_xtr/dense_2/bias/read*
T0*
data_formatNHWC
O
"expand_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
p
 expand_xtr/dense_2/LeakyRelu/mulMul"expand_xtr/dense_2/LeakyRelu/alphaexpand_xtr/dense_2/BiasAdd*
T0
n
expand_xtr/dense_2/LeakyReluMaximum expand_xtr/dense_2/LeakyRelu/mulexpand_xtr/dense_2/BiasAdd*
T0
�
:expand_xtr/dense_3/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"@      *,
_class"
 loc:@expand_xtr/dense_3/kernel
�
8expand_xtr/dense_3/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *����*,
_class"
 loc:@expand_xtr/dense_3/kernel
�
8expand_xtr/dense_3/kernel/Initializer/random_uniform/maxConst*
valueB
 *���>*,
_class"
 loc:@expand_xtr/dense_3/kernel*
dtype0
�
Bexpand_xtr/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform:expand_xtr/dense_3/kernel/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*,
_class"
 loc:@expand_xtr/dense_3/kernel
�
8expand_xtr/dense_3/kernel/Initializer/random_uniform/subSub8expand_xtr/dense_3/kernel/Initializer/random_uniform/max8expand_xtr/dense_3/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@expand_xtr/dense_3/kernel
�
8expand_xtr/dense_3/kernel/Initializer/random_uniform/mulMulBexpand_xtr/dense_3/kernel/Initializer/random_uniform/RandomUniform8expand_xtr/dense_3/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@expand_xtr/dense_3/kernel
�
4expand_xtr/dense_3/kernel/Initializer/random_uniformAdd8expand_xtr/dense_3/kernel/Initializer/random_uniform/mul8expand_xtr/dense_3/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@expand_xtr/dense_3/kernel
�
expand_xtr/dense_3/kernel
VariableV2*,
_class"
 loc:@expand_xtr/dense_3/kernel*
dtype0*
	container *
shape
:@*
shared_name 
�
 expand_xtr/dense_3/kernel/AssignAssignexpand_xtr/dense_3/kernel4expand_xtr/dense_3/kernel/Initializer/random_uniform*
T0*,
_class"
 loc:@expand_xtr/dense_3/kernel*
validate_shape(*
use_locking(
|
expand_xtr/dense_3/kernel/readIdentityexpand_xtr/dense_3/kernel*
T0*,
_class"
 loc:@expand_xtr/dense_3/kernel
�
)expand_xtr/dense_3/bias/Initializer/zerosConst*
valueB*    **
_class 
loc:@expand_xtr/dense_3/bias*
dtype0
�
expand_xtr/dense_3/bias
VariableV2**
_class 
loc:@expand_xtr/dense_3/bias*
dtype0*
	container *
shape:*
shared_name 
�
expand_xtr/dense_3/bias/AssignAssignexpand_xtr/dense_3/bias)expand_xtr/dense_3/bias/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@expand_xtr/dense_3/bias*
validate_shape(
v
expand_xtr/dense_3/bias/readIdentityexpand_xtr/dense_3/bias*
T0**
_class 
loc:@expand_xtr/dense_3/bias
�
expand_xtr/dense_3/MatMulMatMulexpand_xtr/dense_2/LeakyReluexpand_xtr/dense_3/kernel/read*
T0*
transpose_a( *
transpose_b( 
~
expand_xtr/dense_3/BiasAddBiasAddexpand_xtr/dense_3/MatMulexpand_xtr/dense_3/bias/read*
T0*
data_formatNHWC
J
expand_xtr/dense_3/SigmoidSigmoidexpand_xtr/dense_3/BiasAdd*
T0
�
6like_xtr/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"P     *(
_class
loc:@like_xtr/dense/kernel
�
4like_xtr/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *�-ν*(
_class
loc:@like_xtr/dense/kernel
�
4like_xtr/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *�-�=*(
_class
loc:@like_xtr/dense/kernel*
dtype0
�
>like_xtr/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform6like_xtr/dense/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@like_xtr/dense/kernel*
dtype0*
seed2 *

seed 
�
4like_xtr/dense/kernel/Initializer/random_uniform/subSub4like_xtr/dense/kernel/Initializer/random_uniform/max4like_xtr/dense/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@like_xtr/dense/kernel
�
4like_xtr/dense/kernel/Initializer/random_uniform/mulMul>like_xtr/dense/kernel/Initializer/random_uniform/RandomUniform4like_xtr/dense/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@like_xtr/dense/kernel
�
0like_xtr/dense/kernel/Initializer/random_uniformAdd4like_xtr/dense/kernel/Initializer/random_uniform/mul4like_xtr/dense/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@like_xtr/dense/kernel
�
like_xtr/dense/kernel
VariableV2*
shared_name *(
_class
loc:@like_xtr/dense/kernel*
dtype0*
	container *
shape:
��
�
like_xtr/dense/kernel/AssignAssignlike_xtr/dense/kernel0like_xtr/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*(
_class
loc:@like_xtr/dense/kernel*
validate_shape(
p
like_xtr/dense/kernel/readIdentitylike_xtr/dense/kernel*
T0*(
_class
loc:@like_xtr/dense/kernel

%like_xtr/dense/bias/Initializer/zerosConst*
valueB�*    *&
_class
loc:@like_xtr/dense/bias*
dtype0
�
like_xtr/dense/bias
VariableV2*
dtype0*
	container *
shape:�*
shared_name *&
_class
loc:@like_xtr/dense/bias
�
like_xtr/dense/bias/AssignAssignlike_xtr/dense/bias%like_xtr/dense/bias/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@like_xtr/dense/bias*
validate_shape(
j
like_xtr/dense/bias/readIdentitylike_xtr/dense/bias*
T0*&
_class
loc:@like_xtr/dense/bias
r
like_xtr/dense/MatMulMatMulconcatlike_xtr/dense/kernel/read*
T0*
transpose_a( *
transpose_b( 
r
like_xtr/dense/BiasAddBiasAddlike_xtr/dense/MatMullike_xtr/dense/bias/read*
T0*
data_formatNHWC
K
like_xtr/dense/LeakyRelu/alphaConst*
dtype0*
valueB
 *��L>
d
like_xtr/dense/LeakyRelu/mulMullike_xtr/dense/LeakyRelu/alphalike_xtr/dense/BiasAdd*
T0
b
like_xtr/dense/LeakyReluMaximumlike_xtr/dense/LeakyRelu/mullike_xtr/dense/BiasAdd*
T0
�
8like_xtr/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"   �   **
_class 
loc:@like_xtr/dense_1/kernel*
dtype0
�
6like_xtr/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *   �**
_class 
loc:@like_xtr/dense_1/kernel
�
6like_xtr/dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *   >**
_class 
loc:@like_xtr/dense_1/kernel
�
@like_xtr/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform8like_xtr/dense_1/kernel/Initializer/random_uniform/shape*
T0**
_class 
loc:@like_xtr/dense_1/kernel*
dtype0*
seed2 *

seed 
�
6like_xtr/dense_1/kernel/Initializer/random_uniform/subSub6like_xtr/dense_1/kernel/Initializer/random_uniform/max6like_xtr/dense_1/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@like_xtr/dense_1/kernel
�
6like_xtr/dense_1/kernel/Initializer/random_uniform/mulMul@like_xtr/dense_1/kernel/Initializer/random_uniform/RandomUniform6like_xtr/dense_1/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@like_xtr/dense_1/kernel
�
2like_xtr/dense_1/kernel/Initializer/random_uniformAdd6like_xtr/dense_1/kernel/Initializer/random_uniform/mul6like_xtr/dense_1/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@like_xtr/dense_1/kernel
�
like_xtr/dense_1/kernel
VariableV2*
shared_name **
_class 
loc:@like_xtr/dense_1/kernel*
dtype0*
	container *
shape:
��
�
like_xtr/dense_1/kernel/AssignAssignlike_xtr/dense_1/kernel2like_xtr/dense_1/kernel/Initializer/random_uniform*
T0**
_class 
loc:@like_xtr/dense_1/kernel*
validate_shape(*
use_locking(
v
like_xtr/dense_1/kernel/readIdentitylike_xtr/dense_1/kernel*
T0**
_class 
loc:@like_xtr/dense_1/kernel
�
'like_xtr/dense_1/bias/Initializer/zerosConst*
valueB�*    *(
_class
loc:@like_xtr/dense_1/bias*
dtype0
�
like_xtr/dense_1/bias
VariableV2*
dtype0*
	container *
shape:�*
shared_name *(
_class
loc:@like_xtr/dense_1/bias
�
like_xtr/dense_1/bias/AssignAssignlike_xtr/dense_1/bias'like_xtr/dense_1/bias/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@like_xtr/dense_1/bias*
validate_shape(
p
like_xtr/dense_1/bias/readIdentitylike_xtr/dense_1/bias*
T0*(
_class
loc:@like_xtr/dense_1/bias
�
like_xtr/dense_1/MatMulMatMullike_xtr/dense/LeakyRelulike_xtr/dense_1/kernel/read*
transpose_a( *
transpose_b( *
T0
x
like_xtr/dense_1/BiasAddBiasAddlike_xtr/dense_1/MatMullike_xtr/dense_1/bias/read*
T0*
data_formatNHWC
M
 like_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
j
like_xtr/dense_1/LeakyRelu/mulMul like_xtr/dense_1/LeakyRelu/alphalike_xtr/dense_1/BiasAdd*
T0
h
like_xtr/dense_1/LeakyReluMaximumlike_xtr/dense_1/LeakyRelu/mullike_xtr/dense_1/BiasAdd*
T0
�
8like_xtr/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"�   @   **
_class 
loc:@like_xtr/dense_2/kernel*
dtype0
�
6like_xtr/dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *�5�**
_class 
loc:@like_xtr/dense_2/kernel
�
6like_xtr/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *�5>**
_class 
loc:@like_xtr/dense_2/kernel*
dtype0
�
@like_xtr/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform8like_xtr/dense_2/kernel/Initializer/random_uniform/shape*

seed *
T0**
_class 
loc:@like_xtr/dense_2/kernel*
dtype0*
seed2 
�
6like_xtr/dense_2/kernel/Initializer/random_uniform/subSub6like_xtr/dense_2/kernel/Initializer/random_uniform/max6like_xtr/dense_2/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@like_xtr/dense_2/kernel
�
6like_xtr/dense_2/kernel/Initializer/random_uniform/mulMul@like_xtr/dense_2/kernel/Initializer/random_uniform/RandomUniform6like_xtr/dense_2/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@like_xtr/dense_2/kernel
�
2like_xtr/dense_2/kernel/Initializer/random_uniformAdd6like_xtr/dense_2/kernel/Initializer/random_uniform/mul6like_xtr/dense_2/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@like_xtr/dense_2/kernel
�
like_xtr/dense_2/kernel
VariableV2*
dtype0*
	container *
shape:	�@*
shared_name **
_class 
loc:@like_xtr/dense_2/kernel
�
like_xtr/dense_2/kernel/AssignAssignlike_xtr/dense_2/kernel2like_xtr/dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0**
_class 
loc:@like_xtr/dense_2/kernel*
validate_shape(
v
like_xtr/dense_2/kernel/readIdentitylike_xtr/dense_2/kernel*
T0**
_class 
loc:@like_xtr/dense_2/kernel
�
'like_xtr/dense_2/bias/Initializer/zerosConst*
valueB@*    *(
_class
loc:@like_xtr/dense_2/bias*
dtype0
�
like_xtr/dense_2/bias
VariableV2*
dtype0*
	container *
shape:@*
shared_name *(
_class
loc:@like_xtr/dense_2/bias
�
like_xtr/dense_2/bias/AssignAssignlike_xtr/dense_2/bias'like_xtr/dense_2/bias/Initializer/zeros*
validate_shape(*
use_locking(*
T0*(
_class
loc:@like_xtr/dense_2/bias
p
like_xtr/dense_2/bias/readIdentitylike_xtr/dense_2/bias*
T0*(
_class
loc:@like_xtr/dense_2/bias
�
like_xtr/dense_2/MatMulMatMullike_xtr/dense_1/LeakyRelulike_xtr/dense_2/kernel/read*
transpose_b( *
T0*
transpose_a( 
x
like_xtr/dense_2/BiasAddBiasAddlike_xtr/dense_2/MatMullike_xtr/dense_2/bias/read*
T0*
data_formatNHWC
M
 like_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
j
like_xtr/dense_2/LeakyRelu/mulMul like_xtr/dense_2/LeakyRelu/alphalike_xtr/dense_2/BiasAdd*
T0
h
like_xtr/dense_2/LeakyReluMaximumlike_xtr/dense_2/LeakyRelu/mullike_xtr/dense_2/BiasAdd*
T0
�
8like_xtr/dense_3/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"@      **
_class 
loc:@like_xtr/dense_3/kernel
�
6like_xtr/dense_3/kernel/Initializer/random_uniform/minConst*
valueB
 *����**
_class 
loc:@like_xtr/dense_3/kernel*
dtype0
�
6like_xtr/dense_3/kernel/Initializer/random_uniform/maxConst*
valueB
 *���>**
_class 
loc:@like_xtr/dense_3/kernel*
dtype0
�
@like_xtr/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform8like_xtr/dense_3/kernel/Initializer/random_uniform/shape*
T0**
_class 
loc:@like_xtr/dense_3/kernel*
dtype0*
seed2 *

seed 
�
6like_xtr/dense_3/kernel/Initializer/random_uniform/subSub6like_xtr/dense_3/kernel/Initializer/random_uniform/max6like_xtr/dense_3/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@like_xtr/dense_3/kernel
�
6like_xtr/dense_3/kernel/Initializer/random_uniform/mulMul@like_xtr/dense_3/kernel/Initializer/random_uniform/RandomUniform6like_xtr/dense_3/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@like_xtr/dense_3/kernel
�
2like_xtr/dense_3/kernel/Initializer/random_uniformAdd6like_xtr/dense_3/kernel/Initializer/random_uniform/mul6like_xtr/dense_3/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@like_xtr/dense_3/kernel
�
like_xtr/dense_3/kernel
VariableV2*
dtype0*
	container *
shape
:@*
shared_name **
_class 
loc:@like_xtr/dense_3/kernel
�
like_xtr/dense_3/kernel/AssignAssignlike_xtr/dense_3/kernel2like_xtr/dense_3/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0**
_class 
loc:@like_xtr/dense_3/kernel
v
like_xtr/dense_3/kernel/readIdentitylike_xtr/dense_3/kernel*
T0**
_class 
loc:@like_xtr/dense_3/kernel
�
'like_xtr/dense_3/bias/Initializer/zerosConst*
dtype0*
valueB*    *(
_class
loc:@like_xtr/dense_3/bias
�
like_xtr/dense_3/bias
VariableV2*
dtype0*
	container *
shape:*
shared_name *(
_class
loc:@like_xtr/dense_3/bias
�
like_xtr/dense_3/bias/AssignAssignlike_xtr/dense_3/bias'like_xtr/dense_3/bias/Initializer/zeros*
validate_shape(*
use_locking(*
T0*(
_class
loc:@like_xtr/dense_3/bias
p
like_xtr/dense_3/bias/readIdentitylike_xtr/dense_3/bias*
T0*(
_class
loc:@like_xtr/dense_3/bias
�
like_xtr/dense_3/MatMulMatMullike_xtr/dense_2/LeakyRelulike_xtr/dense_3/kernel/read*
T0*
transpose_a( *
transpose_b( 
x
like_xtr/dense_3/BiasAddBiasAddlike_xtr/dense_3/MatMullike_xtr/dense_3/bias/read*
data_formatNHWC*
T0
F
like_xtr/dense_3/SigmoidSigmoidlike_xtr/dense_3/BiasAdd*
T0
�
7reply_xtr/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"P     *)
_class
loc:@reply_xtr/dense/kernel*
dtype0
�
5reply_xtr/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *�-ν*)
_class
loc:@reply_xtr/dense/kernel*
dtype0
�
5reply_xtr/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *�-�=*)
_class
loc:@reply_xtr/dense/kernel*
dtype0
�
?reply_xtr/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7reply_xtr/dense/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@reply_xtr/dense/kernel*
dtype0*
seed2 *

seed 
�
5reply_xtr/dense/kernel/Initializer/random_uniform/subSub5reply_xtr/dense/kernel/Initializer/random_uniform/max5reply_xtr/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@reply_xtr/dense/kernel
�
5reply_xtr/dense/kernel/Initializer/random_uniform/mulMul?reply_xtr/dense/kernel/Initializer/random_uniform/RandomUniform5reply_xtr/dense/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@reply_xtr/dense/kernel
�
1reply_xtr/dense/kernel/Initializer/random_uniformAdd5reply_xtr/dense/kernel/Initializer/random_uniform/mul5reply_xtr/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@reply_xtr/dense/kernel
�
reply_xtr/dense/kernel
VariableV2*
dtype0*
	container *
shape:
��*
shared_name *)
_class
loc:@reply_xtr/dense/kernel
�
reply_xtr/dense/kernel/AssignAssignreply_xtr/dense/kernel1reply_xtr/dense/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*)
_class
loc:@reply_xtr/dense/kernel
s
reply_xtr/dense/kernel/readIdentityreply_xtr/dense/kernel*
T0*)
_class
loc:@reply_xtr/dense/kernel
�
&reply_xtr/dense/bias/Initializer/zerosConst*
valueB�*    *'
_class
loc:@reply_xtr/dense/bias*
dtype0
�
reply_xtr/dense/bias
VariableV2*
shape:�*
shared_name *'
_class
loc:@reply_xtr/dense/bias*
dtype0*
	container 
�
reply_xtr/dense/bias/AssignAssignreply_xtr/dense/bias&reply_xtr/dense/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@reply_xtr/dense/bias*
validate_shape(
m
reply_xtr/dense/bias/readIdentityreply_xtr/dense/bias*
T0*'
_class
loc:@reply_xtr/dense/bias
t
reply_xtr/dense/MatMulMatMulconcatreply_xtr/dense/kernel/read*
T0*
transpose_a( *
transpose_b( 
u
reply_xtr/dense/BiasAddBiasAddreply_xtr/dense/MatMulreply_xtr/dense/bias/read*
T0*
data_formatNHWC
L
reply_xtr/dense/LeakyRelu/alphaConst*
dtype0*
valueB
 *��L>
g
reply_xtr/dense/LeakyRelu/mulMulreply_xtr/dense/LeakyRelu/alphareply_xtr/dense/BiasAdd*
T0
e
reply_xtr/dense/LeakyReluMaximumreply_xtr/dense/LeakyRelu/mulreply_xtr/dense/BiasAdd*
T0
�
9reply_xtr/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"   �   *+
_class!
loc:@reply_xtr/dense_1/kernel*
dtype0
�
7reply_xtr/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *   �*+
_class!
loc:@reply_xtr/dense_1/kernel*
dtype0
�
7reply_xtr/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *   >*+
_class!
loc:@reply_xtr/dense_1/kernel*
dtype0
�
Areply_xtr/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform9reply_xtr/dense_1/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@reply_xtr/dense_1/kernel*
dtype0*
seed2 *

seed 
�
7reply_xtr/dense_1/kernel/Initializer/random_uniform/subSub7reply_xtr/dense_1/kernel/Initializer/random_uniform/max7reply_xtr/dense_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@reply_xtr/dense_1/kernel
�
7reply_xtr/dense_1/kernel/Initializer/random_uniform/mulMulAreply_xtr/dense_1/kernel/Initializer/random_uniform/RandomUniform7reply_xtr/dense_1/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@reply_xtr/dense_1/kernel
�
3reply_xtr/dense_1/kernel/Initializer/random_uniformAdd7reply_xtr/dense_1/kernel/Initializer/random_uniform/mul7reply_xtr/dense_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@reply_xtr/dense_1/kernel
�
reply_xtr/dense_1/kernel
VariableV2*
dtype0*
	container *
shape:
��*
shared_name *+
_class!
loc:@reply_xtr/dense_1/kernel
�
reply_xtr/dense_1/kernel/AssignAssignreply_xtr/dense_1/kernel3reply_xtr/dense_1/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*+
_class!
loc:@reply_xtr/dense_1/kernel
y
reply_xtr/dense_1/kernel/readIdentityreply_xtr/dense_1/kernel*
T0*+
_class!
loc:@reply_xtr/dense_1/kernel
�
(reply_xtr/dense_1/bias/Initializer/zerosConst*
valueB�*    *)
_class
loc:@reply_xtr/dense_1/bias*
dtype0
�
reply_xtr/dense_1/bias
VariableV2*)
_class
loc:@reply_xtr/dense_1/bias*
dtype0*
	container *
shape:�*
shared_name 
�
reply_xtr/dense_1/bias/AssignAssignreply_xtr/dense_1/bias(reply_xtr/dense_1/bias/Initializer/zeros*
validate_shape(*
use_locking(*
T0*)
_class
loc:@reply_xtr/dense_1/bias
s
reply_xtr/dense_1/bias/readIdentityreply_xtr/dense_1/bias*
T0*)
_class
loc:@reply_xtr/dense_1/bias
�
reply_xtr/dense_1/MatMulMatMulreply_xtr/dense/LeakyRelureply_xtr/dense_1/kernel/read*
transpose_a( *
transpose_b( *
T0
{
reply_xtr/dense_1/BiasAddBiasAddreply_xtr/dense_1/MatMulreply_xtr/dense_1/bias/read*
T0*
data_formatNHWC
N
!reply_xtr/dense_1/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
m
reply_xtr/dense_1/LeakyRelu/mulMul!reply_xtr/dense_1/LeakyRelu/alphareply_xtr/dense_1/BiasAdd*
T0
k
reply_xtr/dense_1/LeakyReluMaximumreply_xtr/dense_1/LeakyRelu/mulreply_xtr/dense_1/BiasAdd*
T0
�
9reply_xtr/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"�   @   *+
_class!
loc:@reply_xtr/dense_2/kernel
�
7reply_xtr/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *�5�*+
_class!
loc:@reply_xtr/dense_2/kernel*
dtype0
�
7reply_xtr/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *�5>*+
_class!
loc:@reply_xtr/dense_2/kernel*
dtype0
�
Areply_xtr/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform9reply_xtr/dense_2/kernel/Initializer/random_uniform/shape*

seed *
T0*+
_class!
loc:@reply_xtr/dense_2/kernel*
dtype0*
seed2 
�
7reply_xtr/dense_2/kernel/Initializer/random_uniform/subSub7reply_xtr/dense_2/kernel/Initializer/random_uniform/max7reply_xtr/dense_2/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@reply_xtr/dense_2/kernel
�
7reply_xtr/dense_2/kernel/Initializer/random_uniform/mulMulAreply_xtr/dense_2/kernel/Initializer/random_uniform/RandomUniform7reply_xtr/dense_2/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@reply_xtr/dense_2/kernel
�
3reply_xtr/dense_2/kernel/Initializer/random_uniformAdd7reply_xtr/dense_2/kernel/Initializer/random_uniform/mul7reply_xtr/dense_2/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@reply_xtr/dense_2/kernel
�
reply_xtr/dense_2/kernel
VariableV2*+
_class!
loc:@reply_xtr/dense_2/kernel*
dtype0*
	container *
shape:	�@*
shared_name 
�
reply_xtr/dense_2/kernel/AssignAssignreply_xtr/dense_2/kernel3reply_xtr/dense_2/kernel/Initializer/random_uniform*
T0*+
_class!
loc:@reply_xtr/dense_2/kernel*
validate_shape(*
use_locking(
y
reply_xtr/dense_2/kernel/readIdentityreply_xtr/dense_2/kernel*
T0*+
_class!
loc:@reply_xtr/dense_2/kernel
�
(reply_xtr/dense_2/bias/Initializer/zerosConst*
valueB@*    *)
_class
loc:@reply_xtr/dense_2/bias*
dtype0
�
reply_xtr/dense_2/bias
VariableV2*
shared_name *)
_class
loc:@reply_xtr/dense_2/bias*
dtype0*
	container *
shape:@
�
reply_xtr/dense_2/bias/AssignAssignreply_xtr/dense_2/bias(reply_xtr/dense_2/bias/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@reply_xtr/dense_2/bias*
validate_shape(
s
reply_xtr/dense_2/bias/readIdentityreply_xtr/dense_2/bias*
T0*)
_class
loc:@reply_xtr/dense_2/bias
�
reply_xtr/dense_2/MatMulMatMulreply_xtr/dense_1/LeakyRelureply_xtr/dense_2/kernel/read*
T0*
transpose_a( *
transpose_b( 
{
reply_xtr/dense_2/BiasAddBiasAddreply_xtr/dense_2/MatMulreply_xtr/dense_2/bias/read*
T0*
data_formatNHWC
N
!reply_xtr/dense_2/LeakyRelu/alphaConst*
valueB
 *��L>*
dtype0
m
reply_xtr/dense_2/LeakyRelu/mulMul!reply_xtr/dense_2/LeakyRelu/alphareply_xtr/dense_2/BiasAdd*
T0
k
reply_xtr/dense_2/LeakyReluMaximumreply_xtr/dense_2/LeakyRelu/mulreply_xtr/dense_2/BiasAdd*
T0
�
9reply_xtr/dense_3/kernel/Initializer/random_uniform/shapeConst*
valueB"@      *+
_class!
loc:@reply_xtr/dense_3/kernel*
dtype0
�
7reply_xtr/dense_3/kernel/Initializer/random_uniform/minConst*
valueB
 *����*+
_class!
loc:@reply_xtr/dense_3/kernel*
dtype0
�
7reply_xtr/dense_3/kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *���>*+
_class!
loc:@reply_xtr/dense_3/kernel
�
Areply_xtr/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform9reply_xtr/dense_3/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@reply_xtr/dense_3/kernel*
dtype0*
seed2 *

seed 
�
7reply_xtr/dense_3/kernel/Initializer/random_uniform/subSub7reply_xtr/dense_3/kernel/Initializer/random_uniform/max7reply_xtr/dense_3/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@reply_xtr/dense_3/kernel
�
7reply_xtr/dense_3/kernel/Initializer/random_uniform/mulMulAreply_xtr/dense_3/kernel/Initializer/random_uniform/RandomUniform7reply_xtr/dense_3/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@reply_xtr/dense_3/kernel
�
3reply_xtr/dense_3/kernel/Initializer/random_uniformAdd7reply_xtr/dense_3/kernel/Initializer/random_uniform/mul7reply_xtr/dense_3/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@reply_xtr/dense_3/kernel
�
reply_xtr/dense_3/kernel
VariableV2*+
_class!
loc:@reply_xtr/dense_3/kernel*
dtype0*
	container *
shape
:@*
shared_name 
�
reply_xtr/dense_3/kernel/AssignAssignreply_xtr/dense_3/kernel3reply_xtr/dense_3/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*+
_class!
loc:@reply_xtr/dense_3/kernel
y
reply_xtr/dense_3/kernel/readIdentityreply_xtr/dense_3/kernel*
T0*+
_class!
loc:@reply_xtr/dense_3/kernel
�
(reply_xtr/dense_3/bias/Initializer/zerosConst*
dtype0*
valueB*    *)
_class
loc:@reply_xtr/dense_3/bias
�
reply_xtr/dense_3/bias
VariableV2*)
_class
loc:@reply_xtr/dense_3/bias*
dtype0*
	container *
shape:*
shared_name 
�
reply_xtr/dense_3/bias/AssignAssignreply_xtr/dense_3/bias(reply_xtr/dense_3/bias/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@reply_xtr/dense_3/bias*
validate_shape(
s
reply_xtr/dense_3/bias/readIdentityreply_xtr/dense_3/bias*
T0*)
_class
loc:@reply_xtr/dense_3/bias
�
reply_xtr/dense_3/MatMulMatMulreply_xtr/dense_2/LeakyRelureply_xtr/dense_3/kernel/read*
transpose_b( *
T0*
transpose_a( 
{
reply_xtr/dense_3/BiasAddBiasAddreply_xtr/dense_3/MatMulreply_xtr/dense_3/bias/read*
T0*
data_formatNHWC
H
reply_xtr/dense_3/SigmoidSigmoidreply_xtr/dense_3/BiasAdd*
T0
H
strided_slice/stackConst*
dtype0*
valueB"        
J
strided_slice/stack_1Const*
valueB"       *
dtype0
J
strided_slice/stack_2Const*
valueB"      *
dtype0
�
strided_sliceStridedSlicelabelstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0
D
Reshape_8/shapeConst*
dtype0*
valueB"����   
K
	Reshape_8Reshapestrided_sliceReshape_8/shape*
T0*
Tshape0
J
strided_slice_1/stackConst*
valueB"       *
dtype0
L
strided_slice_1/stack_1Const*
valueB"       *
dtype0
L
strided_slice_1/stack_2Const*
valueB"      *
dtype0
�
strided_slice_1StridedSlicelabelstrided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
D
Reshape_9/shapeConst*
valueB"����   *
dtype0
M
	Reshape_9Reshapestrided_slice_1Reshape_9/shape*
T0*
Tshape0
J
strided_slice_2/stackConst*
valueB"       *
dtype0
L
strided_slice_2/stack_1Const*
valueB"       *
dtype0
L
strided_slice_2/stack_2Const*
valueB"      *
dtype0
�
strided_slice_2StridedSlicelabelstrided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
E
Reshape_10/shapeConst*
valueB"����   *
dtype0
O

Reshape_10Reshapestrided_slice_2Reshape_10/shape*
T0*
Tshape0
J
strided_slice_3/stackConst*
valueB"       *
dtype0
L
strided_slice_3/stack_1Const*
valueB"       *
dtype0
L
strided_slice_3/stack_2Const*
valueB"      *
dtype0
�
strided_slice_3StridedSlicelabelstrided_slice_3/stackstrided_slice_3/stack_1strided_slice_3/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
E
Reshape_11/shapeConst*
valueB"����   *
dtype0
O

Reshape_11Reshapestrided_slice_3Reshape_11/shape*
T0*
Tshape0
�
log_loss/ToFloat/xPackexpand_xtr/dense_3/Sigmoidlike_xtr/dense_3/Sigmoidreply_xtr/dense_3/Sigmoid*
T0*

axis *
N
\
log_loss/ToFloat_1/xPack	Reshape_8	Reshape_9
Reshape_10*
T0*

axis *
N
;
log_loss/add/yConst*
dtype0*
valueB
 *���3
@
log_loss/addAddlog_loss/ToFloat/xlog_loss/add/y*
T0
*
log_loss/LogLoglog_loss/add*
T0
@
log_loss/MulMullog_loss/ToFloat_1/xlog_loss/Log*
T0
*
log_loss/NegNeglog_loss/Mul*
T0
;
log_loss/sub/xConst*
valueB
 *  �?*
dtype0
B
log_loss/subSublog_loss/sub/xlog_loss/ToFloat_1/x*
T0
=
log_loss/sub_1/xConst*
valueB
 *  �?*
dtype0
D
log_loss/sub_1Sublog_loss/sub_1/xlog_loss/ToFloat/x*
T0
=
log_loss/add_1/yConst*
valueB
 *���3*
dtype0
@
log_loss/add_1Addlog_loss/sub_1log_loss/add_1/y*
T0
.
log_loss/Log_1Loglog_loss/add_1*
T0
<
log_loss/Mul_1Mullog_loss/sublog_loss/Log_1*
T0
<
log_loss/sub_2Sublog_loss/Neglog_loss/Mul_1*
T0
o
%log_loss/assert_broadcastable/weightsPack
Reshape_11
Reshape_11
Reshape_11*
T0*

axis *
N
t
+log_loss/assert_broadcastable/weights/shapeShape%log_loss/assert_broadcastable/weights*
T0*
out_type0
T
*log_loss/assert_broadcastable/weights/rankConst*
value	B :*
dtype0
\
*log_loss/assert_broadcastable/values/shapeShapelog_loss/sub_2*
T0*
out_type0
S
)log_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0
S
)log_loss/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0
�
'log_loss/assert_broadcastable/is_scalarEqual)log_loss/assert_broadcastable/is_scalar/x*log_loss/assert_broadcastable/weights/rank*
T0
�
3log_loss/assert_broadcastable/is_valid_shape/SwitchSwitch'log_loss/assert_broadcastable/is_scalar'log_loss/assert_broadcastable/is_scalar*
T0

�
5log_loss/assert_broadcastable/is_valid_shape/switch_tIdentity5log_loss/assert_broadcastable/is_valid_shape/Switch:1*
T0


5log_loss/assert_broadcastable/is_valid_shape/switch_fIdentity3log_loss/assert_broadcastable/is_valid_shape/Switch*
T0

r
4log_loss/assert_broadcastable/is_valid_shape/pred_idIdentity'log_loss/assert_broadcastable/is_scalar*
T0

�
5log_loss/assert_broadcastable/is_valid_shape/Switch_1Switch'log_loss/assert_broadcastable/is_scalar4log_loss/assert_broadcastable/is_valid_shape/pred_id*
T0
*:
_class0
.,loc:@log_loss/assert_broadcastable/is_scalar
�
Slog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualZlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch\log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0
�
Zlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitch)log_loss/assert_broadcastable/values/rank4log_loss/assert_broadcastable/is_valid_shape/pred_id*
T0*<
_class2
0.loc:@log_loss/assert_broadcastable/values/rank
�
\log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1Switch*log_loss/assert_broadcastable/weights/rank4log_loss/assert_broadcastable/is_valid_shape/pred_id*
T0*=
_class3
1/loc:@log_loss/assert_broadcastable/weights/rank
�
Mlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchSlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankSlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0

�
Olog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityOlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0

�
Olog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityMlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0

�
Nlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentitySlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0

�
flog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstP^log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
���������*
dtype0
�
blog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsmlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1flog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0
�
ilog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitch*log_loss/assert_broadcastable/values/shape4log_loss/assert_broadcastable/is_valid_shape/pred_id*
T0*=
_class3
1/loc:@log_loss/assert_broadcastable/values/shape
�
klog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switchilog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchNlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*=
_class3
1/loc:@log_loss/assert_broadcastable/values/shape
�
glog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstP^log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
valueB"      
�
glog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstP^log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0
�
alog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillglog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shapeglog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*

index_type0
�
clog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstP^log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0
�
^log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2blog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimsalog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeclog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*

Tidx0*
T0*
N
�
hlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstP^log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
���������*
dtype0
�
dlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsolog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1hlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*

Tdim0*
T0
�
klog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitch+log_loss/assert_broadcastable/weights/shape4log_loss/assert_broadcastable/is_valid_shape/pred_id*
T0*>
_class4
20loc:@log_loss/assert_broadcastable/weights/shape
�
mlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switchklog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchNlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*>
_class4
20loc:@log_loss/assert_broadcastable/weights/shape
�
plog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationdlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1^log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
T0*
validate_indices(*
set_operationa-b
�
hlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizerlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0
�
Ylog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstP^log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0
�
Wlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualYlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xhlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0
�
Olog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1SwitchSlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankNlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*f
_class\
ZXloc:@log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank
�
Llog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergeOlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Wlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N
�
2log_loss/assert_broadcastable/is_valid_shape/MergeMergeLlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge7log_loss/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N
s
#log_loss/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0
\
%log_loss/assert_broadcastable/Const_1Const*
dtype0*
valueB Bweights.shape=
u
%log_loss/assert_broadcastable/Const_2Const*8
value/B- B'log_loss/assert_broadcastable/weights:0*
dtype0
[
%log_loss/assert_broadcastable/Const_3Const*
dtype0*
valueB Bvalues.shape=
^
%log_loss/assert_broadcastable/Const_4Const*
dtype0*!
valueB Blog_loss/sub_2:0
X
%log_loss/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0
�
0log_loss/assert_broadcastable/AssertGuard/SwitchSwitch2log_loss/assert_broadcastable/is_valid_shape/Merge2log_loss/assert_broadcastable/is_valid_shape/Merge*
T0

{
2log_loss/assert_broadcastable/AssertGuard/switch_tIdentity2log_loss/assert_broadcastable/AssertGuard/Switch:1*
T0

y
2log_loss/assert_broadcastable/AssertGuard/switch_fIdentity0log_loss/assert_broadcastable/AssertGuard/Switch*
T0

z
1log_loss/assert_broadcastable/AssertGuard/pred_idIdentity2log_loss/assert_broadcastable/is_valid_shape/Merge*
T0

k
.log_loss/assert_broadcastable/AssertGuard/NoOpNoOp3^log_loss/assert_broadcastable/AssertGuard/switch_t
�
<log_loss/assert_broadcastable/AssertGuard/control_dependencyIdentity2log_loss/assert_broadcastable/AssertGuard/switch_t/^log_loss/assert_broadcastable/AssertGuard/NoOp*
T0
*E
_class;
97loc:@log_loss/assert_broadcastable/AssertGuard/switch_t
�
7log_loss/assert_broadcastable/AssertGuard/Assert/data_0Const3^log_loss/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0
�
7log_loss/assert_broadcastable/AssertGuard/Assert/data_1Const3^log_loss/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0
�
7log_loss/assert_broadcastable/AssertGuard/Assert/data_2Const3^log_loss/assert_broadcastable/AssertGuard/switch_f*
dtype0*8
value/B- B'log_loss/assert_broadcastable/weights:0
�
7log_loss/assert_broadcastable/AssertGuard/Assert/data_4Const3^log_loss/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0
�
7log_loss/assert_broadcastable/AssertGuard/Assert/data_5Const3^log_loss/assert_broadcastable/AssertGuard/switch_f*!
valueB Blog_loss/sub_2:0*
dtype0
�
7log_loss/assert_broadcastable/AssertGuard/Assert/data_7Const3^log_loss/assert_broadcastable/AssertGuard/switch_f*
dtype0*
valueB B
is_scalar=
�
0log_loss/assert_broadcastable/AssertGuard/AssertAssert7log_loss/assert_broadcastable/AssertGuard/Assert/Switch7log_loss/assert_broadcastable/AssertGuard/Assert/data_07log_loss/assert_broadcastable/AssertGuard/Assert/data_17log_loss/assert_broadcastable/AssertGuard/Assert/data_29log_loss/assert_broadcastable/AssertGuard/Assert/Switch_17log_loss/assert_broadcastable/AssertGuard/Assert/data_47log_loss/assert_broadcastable/AssertGuard/Assert/data_59log_loss/assert_broadcastable/AssertGuard/Assert/Switch_27log_loss/assert_broadcastable/AssertGuard/Assert/data_79log_loss/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
�
7log_loss/assert_broadcastable/AssertGuard/Assert/SwitchSwitch2log_loss/assert_broadcastable/is_valid_shape/Merge1log_loss/assert_broadcastable/AssertGuard/pred_id*
T0
*E
_class;
97loc:@log_loss/assert_broadcastable/is_valid_shape/Merge
�
9log_loss/assert_broadcastable/AssertGuard/Assert/Switch_1Switch+log_loss/assert_broadcastable/weights/shape1log_loss/assert_broadcastable/AssertGuard/pred_id*
T0*>
_class4
20loc:@log_loss/assert_broadcastable/weights/shape
�
9log_loss/assert_broadcastable/AssertGuard/Assert/Switch_2Switch*log_loss/assert_broadcastable/values/shape1log_loss/assert_broadcastable/AssertGuard/pred_id*
T0*=
_class3
1/loc:@log_loss/assert_broadcastable/values/shape
�
9log_loss/assert_broadcastable/AssertGuard/Assert/Switch_3Switch'log_loss/assert_broadcastable/is_scalar1log_loss/assert_broadcastable/AssertGuard/pred_id*
T0
*:
_class0
.,loc:@log_loss/assert_broadcastable/is_scalar
�
>log_loss/assert_broadcastable/AssertGuard/control_dependency_1Identity2log_loss/assert_broadcastable/AssertGuard/switch_f1^log_loss/assert_broadcastable/AssertGuard/Assert*
T0
*E
_class;
97loc:@log_loss/assert_broadcastable/AssertGuard/switch_f
�
/log_loss/assert_broadcastable/AssertGuard/MergeMerge>log_loss/assert_broadcastable/AssertGuard/control_dependency_1<log_loss/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N
�
log_loss/ToFloat_2/xPack
Reshape_11
Reshape_11
Reshape_110^log_loss/assert_broadcastable/AssertGuard/Merge*
T0*

axis *
N
D
log_loss/Mul_2Mullog_loss/sub_2log_loss/ToFloat_2/x*
T0
y
log_loss/ConstConst0^log_loss/assert_broadcastable/AssertGuard/Merge*!
valueB"          *
dtype0
Y
log_loss/SumSumlog_loss/Mul_2log_loss/Const*
T0*

Tidx0*
	keep_dims( 
8
gradients/ShapeConst*
valueB *
dtype0
@
gradients/grad_ys_0Const*
dtype0*
valueB
 *  �?
W
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0
b
)gradients/log_loss/Sum_grad/Reshape/shapeConst*!
valueB"         *
dtype0
�
#gradients/log_loss/Sum_grad/ReshapeReshapegradients/Fill)gradients/log_loss/Sum_grad/Reshape/shape*
T0*
Tshape0
S
!gradients/log_loss/Sum_grad/ShapeShapelog_loss/Mul_2*
T0*
out_type0
�
 gradients/log_loss/Sum_grad/TileTile#gradients/log_loss/Sum_grad/Reshape!gradients/log_loss/Sum_grad/Shape*

Tmultiples0*
T0
U
#gradients/log_loss/Mul_2_grad/ShapeShapelog_loss/sub_2*
T0*
out_type0
]
%gradients/log_loss/Mul_2_grad/Shape_1Shapelog_loss/ToFloat_2/x*
T0*
out_type0
�
3gradients/log_loss/Mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients/log_loss/Mul_2_grad/Shape%gradients/log_loss/Mul_2_grad/Shape_1*
T0
i
!gradients/log_loss/Mul_2_grad/MulMul gradients/log_loss/Sum_grad/Tilelog_loss/ToFloat_2/x*
T0
�
!gradients/log_loss/Mul_2_grad/SumSum!gradients/log_loss/Mul_2_grad/Mul3gradients/log_loss/Mul_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
%gradients/log_loss/Mul_2_grad/ReshapeReshape!gradients/log_loss/Mul_2_grad/Sum#gradients/log_loss/Mul_2_grad/Shape*
T0*
Tshape0
e
#gradients/log_loss/Mul_2_grad/Mul_1Mullog_loss/sub_2 gradients/log_loss/Sum_grad/Tile*
T0
�
#gradients/log_loss/Mul_2_grad/Sum_1Sum#gradients/log_loss/Mul_2_grad/Mul_15gradients/log_loss/Mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
'gradients/log_loss/Mul_2_grad/Reshape_1Reshape#gradients/log_loss/Mul_2_grad/Sum_1%gradients/log_loss/Mul_2_grad/Shape_1*
T0*
Tshape0
�
.gradients/log_loss/Mul_2_grad/tuple/group_depsNoOp&^gradients/log_loss/Mul_2_grad/Reshape(^gradients/log_loss/Mul_2_grad/Reshape_1
�
6gradients/log_loss/Mul_2_grad/tuple/control_dependencyIdentity%gradients/log_loss/Mul_2_grad/Reshape/^gradients/log_loss/Mul_2_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/log_loss/Mul_2_grad/Reshape
�
8gradients/log_loss/Mul_2_grad/tuple/control_dependency_1Identity'gradients/log_loss/Mul_2_grad/Reshape_1/^gradients/log_loss/Mul_2_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/log_loss/Mul_2_grad/Reshape_1
S
#gradients/log_loss/sub_2_grad/ShapeShapelog_loss/Neg*
T0*
out_type0
W
%gradients/log_loss/sub_2_grad/Shape_1Shapelog_loss/Mul_1*
T0*
out_type0
�
3gradients/log_loss/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients/log_loss/sub_2_grad/Shape%gradients/log_loss/sub_2_grad/Shape_1*
T0
�
!gradients/log_loss/sub_2_grad/SumSum6gradients/log_loss/Mul_2_grad/tuple/control_dependency3gradients/log_loss/sub_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
%gradients/log_loss/sub_2_grad/ReshapeReshape!gradients/log_loss/sub_2_grad/Sum#gradients/log_loss/sub_2_grad/Shape*
T0*
Tshape0
�
#gradients/log_loss/sub_2_grad/Sum_1Sum6gradients/log_loss/Mul_2_grad/tuple/control_dependency5gradients/log_loss/sub_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
V
!gradients/log_loss/sub_2_grad/NegNeg#gradients/log_loss/sub_2_grad/Sum_1*
T0
�
'gradients/log_loss/sub_2_grad/Reshape_1Reshape!gradients/log_loss/sub_2_grad/Neg%gradients/log_loss/sub_2_grad/Shape_1*
T0*
Tshape0
�
.gradients/log_loss/sub_2_grad/tuple/group_depsNoOp&^gradients/log_loss/sub_2_grad/Reshape(^gradients/log_loss/sub_2_grad/Reshape_1
�
6gradients/log_loss/sub_2_grad/tuple/control_dependencyIdentity%gradients/log_loss/sub_2_grad/Reshape/^gradients/log_loss/sub_2_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/log_loss/sub_2_grad/Reshape
�
8gradients/log_loss/sub_2_grad/tuple/control_dependency_1Identity'gradients/log_loss/sub_2_grad/Reshape_1/^gradients/log_loss/sub_2_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/log_loss/sub_2_grad/Reshape_1
g
gradients/log_loss/Neg_grad/NegNeg6gradients/log_loss/sub_2_grad/tuple/control_dependency*
T0
S
#gradients/log_loss/Mul_1_grad/ShapeShapelog_loss/sub*
T0*
out_type0
W
%gradients/log_loss/Mul_1_grad/Shape_1Shapelog_loss/Log_1*
T0*
out_type0
�
3gradients/log_loss/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients/log_loss/Mul_1_grad/Shape%gradients/log_loss/Mul_1_grad/Shape_1*
T0
{
!gradients/log_loss/Mul_1_grad/MulMul8gradients/log_loss/sub_2_grad/tuple/control_dependency_1log_loss/Log_1*
T0
�
!gradients/log_loss/Mul_1_grad/SumSum!gradients/log_loss/Mul_1_grad/Mul3gradients/log_loss/Mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
%gradients/log_loss/Mul_1_grad/ReshapeReshape!gradients/log_loss/Mul_1_grad/Sum#gradients/log_loss/Mul_1_grad/Shape*
T0*
Tshape0
{
#gradients/log_loss/Mul_1_grad/Mul_1Mullog_loss/sub8gradients/log_loss/sub_2_grad/tuple/control_dependency_1*
T0
�
#gradients/log_loss/Mul_1_grad/Sum_1Sum#gradients/log_loss/Mul_1_grad/Mul_15gradients/log_loss/Mul_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
'gradients/log_loss/Mul_1_grad/Reshape_1Reshape#gradients/log_loss/Mul_1_grad/Sum_1%gradients/log_loss/Mul_1_grad/Shape_1*
T0*
Tshape0
�
.gradients/log_loss/Mul_1_grad/tuple/group_depsNoOp&^gradients/log_loss/Mul_1_grad/Reshape(^gradients/log_loss/Mul_1_grad/Reshape_1
�
6gradients/log_loss/Mul_1_grad/tuple/control_dependencyIdentity%gradients/log_loss/Mul_1_grad/Reshape/^gradients/log_loss/Mul_1_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/log_loss/Mul_1_grad/Reshape
�
8gradients/log_loss/Mul_1_grad/tuple/control_dependency_1Identity'gradients/log_loss/Mul_1_grad/Reshape_1/^gradients/log_loss/Mul_1_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/log_loss/Mul_1_grad/Reshape_1
Y
!gradients/log_loss/Mul_grad/ShapeShapelog_loss/ToFloat_1/x*
T0*
out_type0
S
#gradients/log_loss/Mul_grad/Shape_1Shapelog_loss/Log*
T0*
out_type0
�
1gradients/log_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients/log_loss/Mul_grad/Shape#gradients/log_loss/Mul_grad/Shape_1*
T0
^
gradients/log_loss/Mul_grad/MulMulgradients/log_loss/Neg_grad/Neglog_loss/Log*
T0
�
gradients/log_loss/Mul_grad/SumSumgradients/log_loss/Mul_grad/Mul1gradients/log_loss/Mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
#gradients/log_loss/Mul_grad/ReshapeReshapegradients/log_loss/Mul_grad/Sum!gradients/log_loss/Mul_grad/Shape*
T0*
Tshape0
h
!gradients/log_loss/Mul_grad/Mul_1Mullog_loss/ToFloat_1/xgradients/log_loss/Neg_grad/Neg*
T0
�
!gradients/log_loss/Mul_grad/Sum_1Sum!gradients/log_loss/Mul_grad/Mul_13gradients/log_loss/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
%gradients/log_loss/Mul_grad/Reshape_1Reshape!gradients/log_loss/Mul_grad/Sum_1#gradients/log_loss/Mul_grad/Shape_1*
T0*
Tshape0
�
,gradients/log_loss/Mul_grad/tuple/group_depsNoOp$^gradients/log_loss/Mul_grad/Reshape&^gradients/log_loss/Mul_grad/Reshape_1
�
4gradients/log_loss/Mul_grad/tuple/control_dependencyIdentity#gradients/log_loss/Mul_grad/Reshape-^gradients/log_loss/Mul_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/log_loss/Mul_grad/Reshape
�
6gradients/log_loss/Mul_grad/tuple/control_dependency_1Identity%gradients/log_loss/Mul_grad/Reshape_1-^gradients/log_loss/Mul_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/log_loss/Mul_grad/Reshape_1
�
(gradients/log_loss/Log_1_grad/Reciprocal
Reciprocallog_loss/add_19^gradients/log_loss/Mul_1_grad/tuple/control_dependency_1*
T0
�
!gradients/log_loss/Log_1_grad/mulMul8gradients/log_loss/Mul_1_grad/tuple/control_dependency_1(gradients/log_loss/Log_1_grad/Reciprocal*
T0
�
&gradients/log_loss/Log_grad/Reciprocal
Reciprocallog_loss/add7^gradients/log_loss/Mul_grad/tuple/control_dependency_1*
T0
�
gradients/log_loss/Log_grad/mulMul6gradients/log_loss/Mul_grad/tuple/control_dependency_1&gradients/log_loss/Log_grad/Reciprocal*
T0
U
#gradients/log_loss/add_1_grad/ShapeShapelog_loss/sub_1*
T0*
out_type0
N
%gradients/log_loss/add_1_grad/Shape_1Const*
dtype0*
valueB 
�
3gradients/log_loss/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients/log_loss/add_1_grad/Shape%gradients/log_loss/add_1_grad/Shape_1*
T0
�
!gradients/log_loss/add_1_grad/SumSum!gradients/log_loss/Log_1_grad/mul3gradients/log_loss/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
%gradients/log_loss/add_1_grad/ReshapeReshape!gradients/log_loss/add_1_grad/Sum#gradients/log_loss/add_1_grad/Shape*
T0*
Tshape0
�
#gradients/log_loss/add_1_grad/Sum_1Sum!gradients/log_loss/Log_1_grad/mul5gradients/log_loss/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
'gradients/log_loss/add_1_grad/Reshape_1Reshape#gradients/log_loss/add_1_grad/Sum_1%gradients/log_loss/add_1_grad/Shape_1*
T0*
Tshape0
�
.gradients/log_loss/add_1_grad/tuple/group_depsNoOp&^gradients/log_loss/add_1_grad/Reshape(^gradients/log_loss/add_1_grad/Reshape_1
�
6gradients/log_loss/add_1_grad/tuple/control_dependencyIdentity%gradients/log_loss/add_1_grad/Reshape/^gradients/log_loss/add_1_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/log_loss/add_1_grad/Reshape
�
8gradients/log_loss/add_1_grad/tuple/control_dependency_1Identity'gradients/log_loss/add_1_grad/Reshape_1/^gradients/log_loss/add_1_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/log_loss/add_1_grad/Reshape_1
W
!gradients/log_loss/add_grad/ShapeShapelog_loss/ToFloat/x*
T0*
out_type0
L
#gradients/log_loss/add_grad/Shape_1Const*
valueB *
dtype0
�
1gradients/log_loss/add_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients/log_loss/add_grad/Shape#gradients/log_loss/add_grad/Shape_1*
T0
�
gradients/log_loss/add_grad/SumSumgradients/log_loss/Log_grad/mul1gradients/log_loss/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
#gradients/log_loss/add_grad/ReshapeReshapegradients/log_loss/add_grad/Sum!gradients/log_loss/add_grad/Shape*
T0*
Tshape0
�
!gradients/log_loss/add_grad/Sum_1Sumgradients/log_loss/Log_grad/mul3gradients/log_loss/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
%gradients/log_loss/add_grad/Reshape_1Reshape!gradients/log_loss/add_grad/Sum_1#gradients/log_loss/add_grad/Shape_1*
T0*
Tshape0
�
,gradients/log_loss/add_grad/tuple/group_depsNoOp$^gradients/log_loss/add_grad/Reshape&^gradients/log_loss/add_grad/Reshape_1
�
4gradients/log_loss/add_grad/tuple/control_dependencyIdentity#gradients/log_loss/add_grad/Reshape-^gradients/log_loss/add_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/log_loss/add_grad/Reshape
�
6gradients/log_loss/add_grad/tuple/control_dependency_1Identity%gradients/log_loss/add_grad/Reshape_1-^gradients/log_loss/add_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/log_loss/add_grad/Reshape_1
L
#gradients/log_loss/sub_1_grad/ShapeConst*
valueB *
dtype0
[
%gradients/log_loss/sub_1_grad/Shape_1Shapelog_loss/ToFloat/x*
T0*
out_type0
�
3gradients/log_loss/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients/log_loss/sub_1_grad/Shape%gradients/log_loss/sub_1_grad/Shape_1*
T0
�
!gradients/log_loss/sub_1_grad/SumSum6gradients/log_loss/add_1_grad/tuple/control_dependency3gradients/log_loss/sub_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
%gradients/log_loss/sub_1_grad/ReshapeReshape!gradients/log_loss/sub_1_grad/Sum#gradients/log_loss/sub_1_grad/Shape*
T0*
Tshape0
�
#gradients/log_loss/sub_1_grad/Sum_1Sum6gradients/log_loss/add_1_grad/tuple/control_dependency5gradients/log_loss/sub_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
V
!gradients/log_loss/sub_1_grad/NegNeg#gradients/log_loss/sub_1_grad/Sum_1*
T0
�
'gradients/log_loss/sub_1_grad/Reshape_1Reshape!gradients/log_loss/sub_1_grad/Neg%gradients/log_loss/sub_1_grad/Shape_1*
T0*
Tshape0
�
.gradients/log_loss/sub_1_grad/tuple/group_depsNoOp&^gradients/log_loss/sub_1_grad/Reshape(^gradients/log_loss/sub_1_grad/Reshape_1
�
6gradients/log_loss/sub_1_grad/tuple/control_dependencyIdentity%gradients/log_loss/sub_1_grad/Reshape/^gradients/log_loss/sub_1_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/log_loss/sub_1_grad/Reshape
�
8gradients/log_loss/sub_1_grad/tuple/control_dependency_1Identity'gradients/log_loss/sub_1_grad/Reshape_1/^gradients/log_loss/sub_1_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/log_loss/sub_1_grad/Reshape_1
�
gradients/AddNAddN4gradients/log_loss/add_grad/tuple/control_dependency8gradients/log_loss/sub_1_grad/tuple/control_dependency_1*
T0*6
_class,
*(loc:@gradients/log_loss/add_grad/Reshape*
N
c
)gradients/log_loss/ToFloat/x_grad/unstackUnpackgradients/AddN*
T0*	
num*

axis 
f
2gradients/log_loss/ToFloat/x_grad/tuple/group_depsNoOp*^gradients/log_loss/ToFloat/x_grad/unstack
�
:gradients/log_loss/ToFloat/x_grad/tuple/control_dependencyIdentity)gradients/log_loss/ToFloat/x_grad/unstack3^gradients/log_loss/ToFloat/x_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/log_loss/ToFloat/x_grad/unstack
�
<gradients/log_loss/ToFloat/x_grad/tuple/control_dependency_1Identity+gradients/log_loss/ToFloat/x_grad/unstack:13^gradients/log_loss/ToFloat/x_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/log_loss/ToFloat/x_grad/unstack
�
<gradients/log_loss/ToFloat/x_grad/tuple/control_dependency_2Identity+gradients/log_loss/ToFloat/x_grad/unstack:23^gradients/log_loss/ToFloat/x_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/log_loss/ToFloat/x_grad/unstack
�
5gradients/expand_xtr/dense_3/Sigmoid_grad/SigmoidGradSigmoidGradexpand_xtr/dense_3/Sigmoid:gradients/log_loss/ToFloat/x_grad/tuple/control_dependency*
T0
�
3gradients/like_xtr/dense_3/Sigmoid_grad/SigmoidGradSigmoidGradlike_xtr/dense_3/Sigmoid<gradients/log_loss/ToFloat/x_grad/tuple/control_dependency_1*
T0
�
4gradients/reply_xtr/dense_3/Sigmoid_grad/SigmoidGradSigmoidGradreply_xtr/dense_3/Sigmoid<gradients/log_loss/ToFloat/x_grad/tuple/control_dependency_2*
T0
�
5gradients/expand_xtr/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients/expand_xtr/dense_3/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
T0
�
:gradients/expand_xtr/dense_3/BiasAdd_grad/tuple/group_depsNoOp6^gradients/expand_xtr/dense_3/BiasAdd_grad/BiasAddGrad6^gradients/expand_xtr/dense_3/Sigmoid_grad/SigmoidGrad
�
Bgradients/expand_xtr/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity5gradients/expand_xtr/dense_3/Sigmoid_grad/SigmoidGrad;^gradients/expand_xtr/dense_3/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/expand_xtr/dense_3/Sigmoid_grad/SigmoidGrad
�
Dgradients/expand_xtr/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/expand_xtr/dense_3/BiasAdd_grad/BiasAddGrad;^gradients/expand_xtr/dense_3/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/expand_xtr/dense_3/BiasAdd_grad/BiasAddGrad
�
3gradients/like_xtr/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients/like_xtr/dense_3/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
T0
�
8gradients/like_xtr/dense_3/BiasAdd_grad/tuple/group_depsNoOp4^gradients/like_xtr/dense_3/BiasAdd_grad/BiasAddGrad4^gradients/like_xtr/dense_3/Sigmoid_grad/SigmoidGrad
�
@gradients/like_xtr/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity3gradients/like_xtr/dense_3/Sigmoid_grad/SigmoidGrad9^gradients/like_xtr/dense_3/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/like_xtr/dense_3/Sigmoid_grad/SigmoidGrad
�
Bgradients/like_xtr/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/like_xtr/dense_3/BiasAdd_grad/BiasAddGrad9^gradients/like_xtr/dense_3/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/like_xtr/dense_3/BiasAdd_grad/BiasAddGrad
�
4gradients/reply_xtr/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients/reply_xtr/dense_3/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC
�
9gradients/reply_xtr/dense_3/BiasAdd_grad/tuple/group_depsNoOp5^gradients/reply_xtr/dense_3/BiasAdd_grad/BiasAddGrad5^gradients/reply_xtr/dense_3/Sigmoid_grad/SigmoidGrad
�
Agradients/reply_xtr/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity4gradients/reply_xtr/dense_3/Sigmoid_grad/SigmoidGrad:^gradients/reply_xtr/dense_3/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/reply_xtr/dense_3/Sigmoid_grad/SigmoidGrad
�
Cgradients/reply_xtr/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity4gradients/reply_xtr/dense_3/BiasAdd_grad/BiasAddGrad:^gradients/reply_xtr/dense_3/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/reply_xtr/dense_3/BiasAdd_grad/BiasAddGrad
�
/gradients/expand_xtr/dense_3/MatMul_grad/MatMulMatMulBgradients/expand_xtr/dense_3/BiasAdd_grad/tuple/control_dependencyexpand_xtr/dense_3/kernel/read*
transpose_a( *
transpose_b(*
T0
�
1gradients/expand_xtr/dense_3/MatMul_grad/MatMul_1MatMulexpand_xtr/dense_2/LeakyReluBgradients/expand_xtr/dense_3/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
�
9gradients/expand_xtr/dense_3/MatMul_grad/tuple/group_depsNoOp0^gradients/expand_xtr/dense_3/MatMul_grad/MatMul2^gradients/expand_xtr/dense_3/MatMul_grad/MatMul_1
�
Agradients/expand_xtr/dense_3/MatMul_grad/tuple/control_dependencyIdentity/gradients/expand_xtr/dense_3/MatMul_grad/MatMul:^gradients/expand_xtr/dense_3/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/expand_xtr/dense_3/MatMul_grad/MatMul
�
Cgradients/expand_xtr/dense_3/MatMul_grad/tuple/control_dependency_1Identity1gradients/expand_xtr/dense_3/MatMul_grad/MatMul_1:^gradients/expand_xtr/dense_3/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/expand_xtr/dense_3/MatMul_grad/MatMul_1
�
-gradients/like_xtr/dense_3/MatMul_grad/MatMulMatMul@gradients/like_xtr/dense_3/BiasAdd_grad/tuple/control_dependencylike_xtr/dense_3/kernel/read*
T0*
transpose_a( *
transpose_b(
�
/gradients/like_xtr/dense_3/MatMul_grad/MatMul_1MatMullike_xtr/dense_2/LeakyRelu@gradients/like_xtr/dense_3/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
�
7gradients/like_xtr/dense_3/MatMul_grad/tuple/group_depsNoOp.^gradients/like_xtr/dense_3/MatMul_grad/MatMul0^gradients/like_xtr/dense_3/MatMul_grad/MatMul_1
�
?gradients/like_xtr/dense_3/MatMul_grad/tuple/control_dependencyIdentity-gradients/like_xtr/dense_3/MatMul_grad/MatMul8^gradients/like_xtr/dense_3/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/like_xtr/dense_3/MatMul_grad/MatMul
�
Agradients/like_xtr/dense_3/MatMul_grad/tuple/control_dependency_1Identity/gradients/like_xtr/dense_3/MatMul_grad/MatMul_18^gradients/like_xtr/dense_3/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/like_xtr/dense_3/MatMul_grad/MatMul_1
�
.gradients/reply_xtr/dense_3/MatMul_grad/MatMulMatMulAgradients/reply_xtr/dense_3/BiasAdd_grad/tuple/control_dependencyreply_xtr/dense_3/kernel/read*
transpose_a( *
transpose_b(*
T0
�
0gradients/reply_xtr/dense_3/MatMul_grad/MatMul_1MatMulreply_xtr/dense_2/LeakyReluAgradients/reply_xtr/dense_3/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
�
8gradients/reply_xtr/dense_3/MatMul_grad/tuple/group_depsNoOp/^gradients/reply_xtr/dense_3/MatMul_grad/MatMul1^gradients/reply_xtr/dense_3/MatMul_grad/MatMul_1
�
@gradients/reply_xtr/dense_3/MatMul_grad/tuple/control_dependencyIdentity.gradients/reply_xtr/dense_3/MatMul_grad/MatMul9^gradients/reply_xtr/dense_3/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/reply_xtr/dense_3/MatMul_grad/MatMul
�
Bgradients/reply_xtr/dense_3/MatMul_grad/tuple/control_dependency_1Identity0gradients/reply_xtr/dense_3/MatMul_grad/MatMul_19^gradients/reply_xtr/dense_3/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/reply_xtr/dense_3/MatMul_grad/MatMul_1
u
1gradients/expand_xtr/dense_2/LeakyRelu_grad/ShapeShape expand_xtr/dense_2/LeakyRelu/mul*
T0*
out_type0
q
3gradients/expand_xtr/dense_2/LeakyRelu_grad/Shape_1Shapeexpand_xtr/dense_2/BiasAdd*
T0*
out_type0
�
3gradients/expand_xtr/dense_2/LeakyRelu_grad/Shape_2ShapeAgradients/expand_xtr/dense_3/MatMul_grad/tuple/control_dependency*
T0*
out_type0
d
7gradients/expand_xtr/dense_2/LeakyRelu_grad/zeros/ConstConst*
dtype0*
valueB
 *    
�
1gradients/expand_xtr/dense_2/LeakyRelu_grad/zerosFill3gradients/expand_xtr/dense_2/LeakyRelu_grad/Shape_27gradients/expand_xtr/dense_2/LeakyRelu_grad/zeros/Const*
T0*

index_type0
�
8gradients/expand_xtr/dense_2/LeakyRelu_grad/GreaterEqualGreaterEqual expand_xtr/dense_2/LeakyRelu/mulexpand_xtr/dense_2/BiasAdd*
T0
�
Agradients/expand_xtr/dense_2/LeakyRelu_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/expand_xtr/dense_2/LeakyRelu_grad/Shape3gradients/expand_xtr/dense_2/LeakyRelu_grad/Shape_1*
T0
�
2gradients/expand_xtr/dense_2/LeakyRelu_grad/SelectSelect8gradients/expand_xtr/dense_2/LeakyRelu_grad/GreaterEqualAgradients/expand_xtr/dense_3/MatMul_grad/tuple/control_dependency1gradients/expand_xtr/dense_2/LeakyRelu_grad/zeros*
T0
�
4gradients/expand_xtr/dense_2/LeakyRelu_grad/Select_1Select8gradients/expand_xtr/dense_2/LeakyRelu_grad/GreaterEqual1gradients/expand_xtr/dense_2/LeakyRelu_grad/zerosAgradients/expand_xtr/dense_3/MatMul_grad/tuple/control_dependency*
T0
�
/gradients/expand_xtr/dense_2/LeakyRelu_grad/SumSum2gradients/expand_xtr/dense_2/LeakyRelu_grad/SelectAgradients/expand_xtr/dense_2/LeakyRelu_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
3gradients/expand_xtr/dense_2/LeakyRelu_grad/ReshapeReshape/gradients/expand_xtr/dense_2/LeakyRelu_grad/Sum1gradients/expand_xtr/dense_2/LeakyRelu_grad/Shape*
T0*
Tshape0
�
1gradients/expand_xtr/dense_2/LeakyRelu_grad/Sum_1Sum4gradients/expand_xtr/dense_2/LeakyRelu_grad/Select_1Cgradients/expand_xtr/dense_2/LeakyRelu_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
5gradients/expand_xtr/dense_2/LeakyRelu_grad/Reshape_1Reshape1gradients/expand_xtr/dense_2/LeakyRelu_grad/Sum_13gradients/expand_xtr/dense_2/LeakyRelu_grad/Shape_1*
T0*
Tshape0
�
<gradients/expand_xtr/dense_2/LeakyRelu_grad/tuple/group_depsNoOp4^gradients/expand_xtr/dense_2/LeakyRelu_grad/Reshape6^gradients/expand_xtr/dense_2/LeakyRelu_grad/Reshape_1
�
Dgradients/expand_xtr/dense_2/LeakyRelu_grad/tuple/control_dependencyIdentity3gradients/expand_xtr/dense_2/LeakyRelu_grad/Reshape=^gradients/expand_xtr/dense_2/LeakyRelu_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/expand_xtr/dense_2/LeakyRelu_grad/Reshape
�
Fgradients/expand_xtr/dense_2/LeakyRelu_grad/tuple/control_dependency_1Identity5gradients/expand_xtr/dense_2/LeakyRelu_grad/Reshape_1=^gradients/expand_xtr/dense_2/LeakyRelu_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/expand_xtr/dense_2/LeakyRelu_grad/Reshape_1
q
/gradients/like_xtr/dense_2/LeakyRelu_grad/ShapeShapelike_xtr/dense_2/LeakyRelu/mul*
T0*
out_type0
m
1gradients/like_xtr/dense_2/LeakyRelu_grad/Shape_1Shapelike_xtr/dense_2/BiasAdd*
T0*
out_type0
�
1gradients/like_xtr/dense_2/LeakyRelu_grad/Shape_2Shape?gradients/like_xtr/dense_3/MatMul_grad/tuple/control_dependency*
T0*
out_type0
b
5gradients/like_xtr/dense_2/LeakyRelu_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
/gradients/like_xtr/dense_2/LeakyRelu_grad/zerosFill1gradients/like_xtr/dense_2/LeakyRelu_grad/Shape_25gradients/like_xtr/dense_2/LeakyRelu_grad/zeros/Const*
T0*

index_type0
�
6gradients/like_xtr/dense_2/LeakyRelu_grad/GreaterEqualGreaterEquallike_xtr/dense_2/LeakyRelu/mullike_xtr/dense_2/BiasAdd*
T0
�
?gradients/like_xtr/dense_2/LeakyRelu_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/like_xtr/dense_2/LeakyRelu_grad/Shape1gradients/like_xtr/dense_2/LeakyRelu_grad/Shape_1*
T0
�
0gradients/like_xtr/dense_2/LeakyRelu_grad/SelectSelect6gradients/like_xtr/dense_2/LeakyRelu_grad/GreaterEqual?gradients/like_xtr/dense_3/MatMul_grad/tuple/control_dependency/gradients/like_xtr/dense_2/LeakyRelu_grad/zeros*
T0
�
2gradients/like_xtr/dense_2/LeakyRelu_grad/Select_1Select6gradients/like_xtr/dense_2/LeakyRelu_grad/GreaterEqual/gradients/like_xtr/dense_2/LeakyRelu_grad/zeros?gradients/like_xtr/dense_3/MatMul_grad/tuple/control_dependency*
T0
�
-gradients/like_xtr/dense_2/LeakyRelu_grad/SumSum0gradients/like_xtr/dense_2/LeakyRelu_grad/Select?gradients/like_xtr/dense_2/LeakyRelu_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
1gradients/like_xtr/dense_2/LeakyRelu_grad/ReshapeReshape-gradients/like_xtr/dense_2/LeakyRelu_grad/Sum/gradients/like_xtr/dense_2/LeakyRelu_grad/Shape*
T0*
Tshape0
�
/gradients/like_xtr/dense_2/LeakyRelu_grad/Sum_1Sum2gradients/like_xtr/dense_2/LeakyRelu_grad/Select_1Agradients/like_xtr/dense_2/LeakyRelu_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
3gradients/like_xtr/dense_2/LeakyRelu_grad/Reshape_1Reshape/gradients/like_xtr/dense_2/LeakyRelu_grad/Sum_11gradients/like_xtr/dense_2/LeakyRelu_grad/Shape_1*
T0*
Tshape0
�
:gradients/like_xtr/dense_2/LeakyRelu_grad/tuple/group_depsNoOp2^gradients/like_xtr/dense_2/LeakyRelu_grad/Reshape4^gradients/like_xtr/dense_2/LeakyRelu_grad/Reshape_1
�
Bgradients/like_xtr/dense_2/LeakyRelu_grad/tuple/control_dependencyIdentity1gradients/like_xtr/dense_2/LeakyRelu_grad/Reshape;^gradients/like_xtr/dense_2/LeakyRelu_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/like_xtr/dense_2/LeakyRelu_grad/Reshape
�
Dgradients/like_xtr/dense_2/LeakyRelu_grad/tuple/control_dependency_1Identity3gradients/like_xtr/dense_2/LeakyRelu_grad/Reshape_1;^gradients/like_xtr/dense_2/LeakyRelu_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/like_xtr/dense_2/LeakyRelu_grad/Reshape_1
s
0gradients/reply_xtr/dense_2/LeakyRelu_grad/ShapeShapereply_xtr/dense_2/LeakyRelu/mul*
T0*
out_type0
o
2gradients/reply_xtr/dense_2/LeakyRelu_grad/Shape_1Shapereply_xtr/dense_2/BiasAdd*
T0*
out_type0
�
2gradients/reply_xtr/dense_2/LeakyRelu_grad/Shape_2Shape@gradients/reply_xtr/dense_3/MatMul_grad/tuple/control_dependency*
T0*
out_type0
c
6gradients/reply_xtr/dense_2/LeakyRelu_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
0gradients/reply_xtr/dense_2/LeakyRelu_grad/zerosFill2gradients/reply_xtr/dense_2/LeakyRelu_grad/Shape_26gradients/reply_xtr/dense_2/LeakyRelu_grad/zeros/Const*
T0*

index_type0
�
7gradients/reply_xtr/dense_2/LeakyRelu_grad/GreaterEqualGreaterEqualreply_xtr/dense_2/LeakyRelu/mulreply_xtr/dense_2/BiasAdd*
T0
�
@gradients/reply_xtr/dense_2/LeakyRelu_grad/BroadcastGradientArgsBroadcastGradientArgs0gradients/reply_xtr/dense_2/LeakyRelu_grad/Shape2gradients/reply_xtr/dense_2/LeakyRelu_grad/Shape_1*
T0
�
1gradients/reply_xtr/dense_2/LeakyRelu_grad/SelectSelect7gradients/reply_xtr/dense_2/LeakyRelu_grad/GreaterEqual@gradients/reply_xtr/dense_3/MatMul_grad/tuple/control_dependency0gradients/reply_xtr/dense_2/LeakyRelu_grad/zeros*
T0
�
3gradients/reply_xtr/dense_2/LeakyRelu_grad/Select_1Select7gradients/reply_xtr/dense_2/LeakyRelu_grad/GreaterEqual0gradients/reply_xtr/dense_2/LeakyRelu_grad/zeros@gradients/reply_xtr/dense_3/MatMul_grad/tuple/control_dependency*
T0
�
.gradients/reply_xtr/dense_2/LeakyRelu_grad/SumSum1gradients/reply_xtr/dense_2/LeakyRelu_grad/Select@gradients/reply_xtr/dense_2/LeakyRelu_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
2gradients/reply_xtr/dense_2/LeakyRelu_grad/ReshapeReshape.gradients/reply_xtr/dense_2/LeakyRelu_grad/Sum0gradients/reply_xtr/dense_2/LeakyRelu_grad/Shape*
T0*
Tshape0
�
0gradients/reply_xtr/dense_2/LeakyRelu_grad/Sum_1Sum3gradients/reply_xtr/dense_2/LeakyRelu_grad/Select_1Bgradients/reply_xtr/dense_2/LeakyRelu_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
4gradients/reply_xtr/dense_2/LeakyRelu_grad/Reshape_1Reshape0gradients/reply_xtr/dense_2/LeakyRelu_grad/Sum_12gradients/reply_xtr/dense_2/LeakyRelu_grad/Shape_1*
T0*
Tshape0
�
;gradients/reply_xtr/dense_2/LeakyRelu_grad/tuple/group_depsNoOp3^gradients/reply_xtr/dense_2/LeakyRelu_grad/Reshape5^gradients/reply_xtr/dense_2/LeakyRelu_grad/Reshape_1
�
Cgradients/reply_xtr/dense_2/LeakyRelu_grad/tuple/control_dependencyIdentity2gradients/reply_xtr/dense_2/LeakyRelu_grad/Reshape<^gradients/reply_xtr/dense_2/LeakyRelu_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/reply_xtr/dense_2/LeakyRelu_grad/Reshape
�
Egradients/reply_xtr/dense_2/LeakyRelu_grad/tuple/control_dependency_1Identity4gradients/reply_xtr/dense_2/LeakyRelu_grad/Reshape_1<^gradients/reply_xtr/dense_2/LeakyRelu_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/reply_xtr/dense_2/LeakyRelu_grad/Reshape_1
^
5gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/ShapeConst*
dtype0*
valueB 
u
7gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/Shape_1Shapeexpand_xtr/dense_2/BiasAdd*
T0*
out_type0
�
Egradients/expand_xtr/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/Shape7gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/Shape_1*
T0
�
3gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/MulMulDgradients/expand_xtr/dense_2/LeakyRelu_grad/tuple/control_dependencyexpand_xtr/dense_2/BiasAdd*
T0
�
3gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/SumSum3gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/MulEgradients/expand_xtr/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
7gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/ReshapeReshape3gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/Sum5gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
�
5gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/Mul_1Mul"expand_xtr/dense_2/LeakyRelu/alphaDgradients/expand_xtr/dense_2/LeakyRelu_grad/tuple/control_dependency*
T0
�
5gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/Sum_1Sum5gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/Mul_1Ggradients/expand_xtr/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
9gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/Reshape_1Reshape5gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/Sum_17gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
�
@gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/tuple/group_depsNoOp8^gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/Reshape:^gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/Reshape_1
�
Hgradients/expand_xtr/dense_2/LeakyRelu/mul_grad/tuple/control_dependencyIdentity7gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/ReshapeA^gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/Reshape
�
Jgradients/expand_xtr/dense_2/LeakyRelu/mul_grad/tuple/control_dependency_1Identity9gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/Reshape_1A^gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/expand_xtr/dense_2/LeakyRelu/mul_grad/Reshape_1
\
3gradients/like_xtr/dense_2/LeakyRelu/mul_grad/ShapeConst*
valueB *
dtype0
q
5gradients/like_xtr/dense_2/LeakyRelu/mul_grad/Shape_1Shapelike_xtr/dense_2/BiasAdd*
T0*
out_type0
�
Cgradients/like_xtr/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/like_xtr/dense_2/LeakyRelu/mul_grad/Shape5gradients/like_xtr/dense_2/LeakyRelu/mul_grad/Shape_1*
T0
�
1gradients/like_xtr/dense_2/LeakyRelu/mul_grad/MulMulBgradients/like_xtr/dense_2/LeakyRelu_grad/tuple/control_dependencylike_xtr/dense_2/BiasAdd*
T0
�
1gradients/like_xtr/dense_2/LeakyRelu/mul_grad/SumSum1gradients/like_xtr/dense_2/LeakyRelu/mul_grad/MulCgradients/like_xtr/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
5gradients/like_xtr/dense_2/LeakyRelu/mul_grad/ReshapeReshape1gradients/like_xtr/dense_2/LeakyRelu/mul_grad/Sum3gradients/like_xtr/dense_2/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
�
3gradients/like_xtr/dense_2/LeakyRelu/mul_grad/Mul_1Mul like_xtr/dense_2/LeakyRelu/alphaBgradients/like_xtr/dense_2/LeakyRelu_grad/tuple/control_dependency*
T0
�
3gradients/like_xtr/dense_2/LeakyRelu/mul_grad/Sum_1Sum3gradients/like_xtr/dense_2/LeakyRelu/mul_grad/Mul_1Egradients/like_xtr/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
7gradients/like_xtr/dense_2/LeakyRelu/mul_grad/Reshape_1Reshape3gradients/like_xtr/dense_2/LeakyRelu/mul_grad/Sum_15gradients/like_xtr/dense_2/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
�
>gradients/like_xtr/dense_2/LeakyRelu/mul_grad/tuple/group_depsNoOp6^gradients/like_xtr/dense_2/LeakyRelu/mul_grad/Reshape8^gradients/like_xtr/dense_2/LeakyRelu/mul_grad/Reshape_1
�
Fgradients/like_xtr/dense_2/LeakyRelu/mul_grad/tuple/control_dependencyIdentity5gradients/like_xtr/dense_2/LeakyRelu/mul_grad/Reshape?^gradients/like_xtr/dense_2/LeakyRelu/mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/like_xtr/dense_2/LeakyRelu/mul_grad/Reshape
�
Hgradients/like_xtr/dense_2/LeakyRelu/mul_grad/tuple/control_dependency_1Identity7gradients/like_xtr/dense_2/LeakyRelu/mul_grad/Reshape_1?^gradients/like_xtr/dense_2/LeakyRelu/mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/like_xtr/dense_2/LeakyRelu/mul_grad/Reshape_1
]
4gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/ShapeConst*
valueB *
dtype0
s
6gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/Shape_1Shapereply_xtr/dense_2/BiasAdd*
T0*
out_type0
�
Dgradients/reply_xtr/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/Shape6gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/Shape_1*
T0
�
2gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/MulMulCgradients/reply_xtr/dense_2/LeakyRelu_grad/tuple/control_dependencyreply_xtr/dense_2/BiasAdd*
T0
�
2gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/SumSum2gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/MulDgradients/reply_xtr/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
6gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/ReshapeReshape2gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/Sum4gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
�
4gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/Mul_1Mul!reply_xtr/dense_2/LeakyRelu/alphaCgradients/reply_xtr/dense_2/LeakyRelu_grad/tuple/control_dependency*
T0
�
4gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/Sum_1Sum4gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/Mul_1Fgradients/reply_xtr/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
8gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/Reshape_1Reshape4gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/Sum_16gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
�
?gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/tuple/group_depsNoOp7^gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/Reshape9^gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/Reshape_1
�
Ggradients/reply_xtr/dense_2/LeakyRelu/mul_grad/tuple/control_dependencyIdentity6gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/Reshape@^gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/Reshape
�
Igradients/reply_xtr/dense_2/LeakyRelu/mul_grad/tuple/control_dependency_1Identity8gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/Reshape_1@^gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/reply_xtr/dense_2/LeakyRelu/mul_grad/Reshape_1
�
gradients/AddN_1AddNFgradients/expand_xtr/dense_2/LeakyRelu_grad/tuple/control_dependency_1Jgradients/expand_xtr/dense_2/LeakyRelu/mul_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@gradients/expand_xtr/dense_2/LeakyRelu_grad/Reshape_1*
N
v
5gradients/expand_xtr/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_1*
T0*
data_formatNHWC
�
:gradients/expand_xtr/dense_2/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_16^gradients/expand_xtr/dense_2/BiasAdd_grad/BiasAddGrad
�
Bgradients/expand_xtr/dense_2/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_1;^gradients/expand_xtr/dense_2/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/expand_xtr/dense_2/LeakyRelu_grad/Reshape_1
�
Dgradients/expand_xtr/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/expand_xtr/dense_2/BiasAdd_grad/BiasAddGrad;^gradients/expand_xtr/dense_2/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/expand_xtr/dense_2/BiasAdd_grad/BiasAddGrad
�
gradients/AddN_2AddNDgradients/like_xtr/dense_2/LeakyRelu_grad/tuple/control_dependency_1Hgradients/like_xtr/dense_2/LeakyRelu/mul_grad/tuple/control_dependency_1*
T0*F
_class<
:8loc:@gradients/like_xtr/dense_2/LeakyRelu_grad/Reshape_1*
N
t
3gradients/like_xtr/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_2*
T0*
data_formatNHWC
�
8gradients/like_xtr/dense_2/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_24^gradients/like_xtr/dense_2/BiasAdd_grad/BiasAddGrad
�
@gradients/like_xtr/dense_2/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_29^gradients/like_xtr/dense_2/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/like_xtr/dense_2/LeakyRelu_grad/Reshape_1
�
Bgradients/like_xtr/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/like_xtr/dense_2/BiasAdd_grad/BiasAddGrad9^gradients/like_xtr/dense_2/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/like_xtr/dense_2/BiasAdd_grad/BiasAddGrad
�
gradients/AddN_3AddNEgradients/reply_xtr/dense_2/LeakyRelu_grad/tuple/control_dependency_1Igradients/reply_xtr/dense_2/LeakyRelu/mul_grad/tuple/control_dependency_1*
T0*G
_class=
;9loc:@gradients/reply_xtr/dense_2/LeakyRelu_grad/Reshape_1*
N
u
4gradients/reply_xtr/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_3*
T0*
data_formatNHWC
�
9gradients/reply_xtr/dense_2/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_35^gradients/reply_xtr/dense_2/BiasAdd_grad/BiasAddGrad
�
Agradients/reply_xtr/dense_2/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_3:^gradients/reply_xtr/dense_2/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/reply_xtr/dense_2/LeakyRelu_grad/Reshape_1
�
Cgradients/reply_xtr/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity4gradients/reply_xtr/dense_2/BiasAdd_grad/BiasAddGrad:^gradients/reply_xtr/dense_2/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/reply_xtr/dense_2/BiasAdd_grad/BiasAddGrad
�
/gradients/expand_xtr/dense_2/MatMul_grad/MatMulMatMulBgradients/expand_xtr/dense_2/BiasAdd_grad/tuple/control_dependencyexpand_xtr/dense_2/kernel/read*
T0*
transpose_a( *
transpose_b(
�
1gradients/expand_xtr/dense_2/MatMul_grad/MatMul_1MatMulexpand_xtr/dense_1/LeakyReluBgradients/expand_xtr/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
�
9gradients/expand_xtr/dense_2/MatMul_grad/tuple/group_depsNoOp0^gradients/expand_xtr/dense_2/MatMul_grad/MatMul2^gradients/expand_xtr/dense_2/MatMul_grad/MatMul_1
�
Agradients/expand_xtr/dense_2/MatMul_grad/tuple/control_dependencyIdentity/gradients/expand_xtr/dense_2/MatMul_grad/MatMul:^gradients/expand_xtr/dense_2/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/expand_xtr/dense_2/MatMul_grad/MatMul
�
Cgradients/expand_xtr/dense_2/MatMul_grad/tuple/control_dependency_1Identity1gradients/expand_xtr/dense_2/MatMul_grad/MatMul_1:^gradients/expand_xtr/dense_2/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/expand_xtr/dense_2/MatMul_grad/MatMul_1
�
-gradients/like_xtr/dense_2/MatMul_grad/MatMulMatMul@gradients/like_xtr/dense_2/BiasAdd_grad/tuple/control_dependencylike_xtr/dense_2/kernel/read*
transpose_b(*
T0*
transpose_a( 
�
/gradients/like_xtr/dense_2/MatMul_grad/MatMul_1MatMullike_xtr/dense_1/LeakyRelu@gradients/like_xtr/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
�
7gradients/like_xtr/dense_2/MatMul_grad/tuple/group_depsNoOp.^gradients/like_xtr/dense_2/MatMul_grad/MatMul0^gradients/like_xtr/dense_2/MatMul_grad/MatMul_1
�
?gradients/like_xtr/dense_2/MatMul_grad/tuple/control_dependencyIdentity-gradients/like_xtr/dense_2/MatMul_grad/MatMul8^gradients/like_xtr/dense_2/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/like_xtr/dense_2/MatMul_grad/MatMul
�
Agradients/like_xtr/dense_2/MatMul_grad/tuple/control_dependency_1Identity/gradients/like_xtr/dense_2/MatMul_grad/MatMul_18^gradients/like_xtr/dense_2/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/like_xtr/dense_2/MatMul_grad/MatMul_1
�
.gradients/reply_xtr/dense_2/MatMul_grad/MatMulMatMulAgradients/reply_xtr/dense_2/BiasAdd_grad/tuple/control_dependencyreply_xtr/dense_2/kernel/read*
transpose_b(*
T0*
transpose_a( 
�
0gradients/reply_xtr/dense_2/MatMul_grad/MatMul_1MatMulreply_xtr/dense_1/LeakyReluAgradients/reply_xtr/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
�
8gradients/reply_xtr/dense_2/MatMul_grad/tuple/group_depsNoOp/^gradients/reply_xtr/dense_2/MatMul_grad/MatMul1^gradients/reply_xtr/dense_2/MatMul_grad/MatMul_1
�
@gradients/reply_xtr/dense_2/MatMul_grad/tuple/control_dependencyIdentity.gradients/reply_xtr/dense_2/MatMul_grad/MatMul9^gradients/reply_xtr/dense_2/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/reply_xtr/dense_2/MatMul_grad/MatMul
�
Bgradients/reply_xtr/dense_2/MatMul_grad/tuple/control_dependency_1Identity0gradients/reply_xtr/dense_2/MatMul_grad/MatMul_19^gradients/reply_xtr/dense_2/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/reply_xtr/dense_2/MatMul_grad/MatMul_1
u
1gradients/expand_xtr/dense_1/LeakyRelu_grad/ShapeShape expand_xtr/dense_1/LeakyRelu/mul*
T0*
out_type0
q
3gradients/expand_xtr/dense_1/LeakyRelu_grad/Shape_1Shapeexpand_xtr/dense_1/BiasAdd*
T0*
out_type0
�
3gradients/expand_xtr/dense_1/LeakyRelu_grad/Shape_2ShapeAgradients/expand_xtr/dense_2/MatMul_grad/tuple/control_dependency*
T0*
out_type0
d
7gradients/expand_xtr/dense_1/LeakyRelu_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
1gradients/expand_xtr/dense_1/LeakyRelu_grad/zerosFill3gradients/expand_xtr/dense_1/LeakyRelu_grad/Shape_27gradients/expand_xtr/dense_1/LeakyRelu_grad/zeros/Const*
T0*

index_type0
�
8gradients/expand_xtr/dense_1/LeakyRelu_grad/GreaterEqualGreaterEqual expand_xtr/dense_1/LeakyRelu/mulexpand_xtr/dense_1/BiasAdd*
T0
�
Agradients/expand_xtr/dense_1/LeakyRelu_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/expand_xtr/dense_1/LeakyRelu_grad/Shape3gradients/expand_xtr/dense_1/LeakyRelu_grad/Shape_1*
T0
�
2gradients/expand_xtr/dense_1/LeakyRelu_grad/SelectSelect8gradients/expand_xtr/dense_1/LeakyRelu_grad/GreaterEqualAgradients/expand_xtr/dense_2/MatMul_grad/tuple/control_dependency1gradients/expand_xtr/dense_1/LeakyRelu_grad/zeros*
T0
�
4gradients/expand_xtr/dense_1/LeakyRelu_grad/Select_1Select8gradients/expand_xtr/dense_1/LeakyRelu_grad/GreaterEqual1gradients/expand_xtr/dense_1/LeakyRelu_grad/zerosAgradients/expand_xtr/dense_2/MatMul_grad/tuple/control_dependency*
T0
�
/gradients/expand_xtr/dense_1/LeakyRelu_grad/SumSum2gradients/expand_xtr/dense_1/LeakyRelu_grad/SelectAgradients/expand_xtr/dense_1/LeakyRelu_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
3gradients/expand_xtr/dense_1/LeakyRelu_grad/ReshapeReshape/gradients/expand_xtr/dense_1/LeakyRelu_grad/Sum1gradients/expand_xtr/dense_1/LeakyRelu_grad/Shape*
T0*
Tshape0
�
1gradients/expand_xtr/dense_1/LeakyRelu_grad/Sum_1Sum4gradients/expand_xtr/dense_1/LeakyRelu_grad/Select_1Cgradients/expand_xtr/dense_1/LeakyRelu_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
5gradients/expand_xtr/dense_1/LeakyRelu_grad/Reshape_1Reshape1gradients/expand_xtr/dense_1/LeakyRelu_grad/Sum_13gradients/expand_xtr/dense_1/LeakyRelu_grad/Shape_1*
T0*
Tshape0
�
<gradients/expand_xtr/dense_1/LeakyRelu_grad/tuple/group_depsNoOp4^gradients/expand_xtr/dense_1/LeakyRelu_grad/Reshape6^gradients/expand_xtr/dense_1/LeakyRelu_grad/Reshape_1
�
Dgradients/expand_xtr/dense_1/LeakyRelu_grad/tuple/control_dependencyIdentity3gradients/expand_xtr/dense_1/LeakyRelu_grad/Reshape=^gradients/expand_xtr/dense_1/LeakyRelu_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/expand_xtr/dense_1/LeakyRelu_grad/Reshape
�
Fgradients/expand_xtr/dense_1/LeakyRelu_grad/tuple/control_dependency_1Identity5gradients/expand_xtr/dense_1/LeakyRelu_grad/Reshape_1=^gradients/expand_xtr/dense_1/LeakyRelu_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/expand_xtr/dense_1/LeakyRelu_grad/Reshape_1
q
/gradients/like_xtr/dense_1/LeakyRelu_grad/ShapeShapelike_xtr/dense_1/LeakyRelu/mul*
T0*
out_type0
m
1gradients/like_xtr/dense_1/LeakyRelu_grad/Shape_1Shapelike_xtr/dense_1/BiasAdd*
T0*
out_type0
�
1gradients/like_xtr/dense_1/LeakyRelu_grad/Shape_2Shape?gradients/like_xtr/dense_2/MatMul_grad/tuple/control_dependency*
T0*
out_type0
b
5gradients/like_xtr/dense_1/LeakyRelu_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
/gradients/like_xtr/dense_1/LeakyRelu_grad/zerosFill1gradients/like_xtr/dense_1/LeakyRelu_grad/Shape_25gradients/like_xtr/dense_1/LeakyRelu_grad/zeros/Const*
T0*

index_type0
�
6gradients/like_xtr/dense_1/LeakyRelu_grad/GreaterEqualGreaterEquallike_xtr/dense_1/LeakyRelu/mullike_xtr/dense_1/BiasAdd*
T0
�
?gradients/like_xtr/dense_1/LeakyRelu_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/like_xtr/dense_1/LeakyRelu_grad/Shape1gradients/like_xtr/dense_1/LeakyRelu_grad/Shape_1*
T0
�
0gradients/like_xtr/dense_1/LeakyRelu_grad/SelectSelect6gradients/like_xtr/dense_1/LeakyRelu_grad/GreaterEqual?gradients/like_xtr/dense_2/MatMul_grad/tuple/control_dependency/gradients/like_xtr/dense_1/LeakyRelu_grad/zeros*
T0
�
2gradients/like_xtr/dense_1/LeakyRelu_grad/Select_1Select6gradients/like_xtr/dense_1/LeakyRelu_grad/GreaterEqual/gradients/like_xtr/dense_1/LeakyRelu_grad/zeros?gradients/like_xtr/dense_2/MatMul_grad/tuple/control_dependency*
T0
�
-gradients/like_xtr/dense_1/LeakyRelu_grad/SumSum0gradients/like_xtr/dense_1/LeakyRelu_grad/Select?gradients/like_xtr/dense_1/LeakyRelu_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
1gradients/like_xtr/dense_1/LeakyRelu_grad/ReshapeReshape-gradients/like_xtr/dense_1/LeakyRelu_grad/Sum/gradients/like_xtr/dense_1/LeakyRelu_grad/Shape*
T0*
Tshape0
�
/gradients/like_xtr/dense_1/LeakyRelu_grad/Sum_1Sum2gradients/like_xtr/dense_1/LeakyRelu_grad/Select_1Agradients/like_xtr/dense_1/LeakyRelu_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
3gradients/like_xtr/dense_1/LeakyRelu_grad/Reshape_1Reshape/gradients/like_xtr/dense_1/LeakyRelu_grad/Sum_11gradients/like_xtr/dense_1/LeakyRelu_grad/Shape_1*
T0*
Tshape0
�
:gradients/like_xtr/dense_1/LeakyRelu_grad/tuple/group_depsNoOp2^gradients/like_xtr/dense_1/LeakyRelu_grad/Reshape4^gradients/like_xtr/dense_1/LeakyRelu_grad/Reshape_1
�
Bgradients/like_xtr/dense_1/LeakyRelu_grad/tuple/control_dependencyIdentity1gradients/like_xtr/dense_1/LeakyRelu_grad/Reshape;^gradients/like_xtr/dense_1/LeakyRelu_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/like_xtr/dense_1/LeakyRelu_grad/Reshape
�
Dgradients/like_xtr/dense_1/LeakyRelu_grad/tuple/control_dependency_1Identity3gradients/like_xtr/dense_1/LeakyRelu_grad/Reshape_1;^gradients/like_xtr/dense_1/LeakyRelu_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/like_xtr/dense_1/LeakyRelu_grad/Reshape_1
s
0gradients/reply_xtr/dense_1/LeakyRelu_grad/ShapeShapereply_xtr/dense_1/LeakyRelu/mul*
T0*
out_type0
o
2gradients/reply_xtr/dense_1/LeakyRelu_grad/Shape_1Shapereply_xtr/dense_1/BiasAdd*
T0*
out_type0
�
2gradients/reply_xtr/dense_1/LeakyRelu_grad/Shape_2Shape@gradients/reply_xtr/dense_2/MatMul_grad/tuple/control_dependency*
T0*
out_type0
c
6gradients/reply_xtr/dense_1/LeakyRelu_grad/zeros/ConstConst*
dtype0*
valueB
 *    
�
0gradients/reply_xtr/dense_1/LeakyRelu_grad/zerosFill2gradients/reply_xtr/dense_1/LeakyRelu_grad/Shape_26gradients/reply_xtr/dense_1/LeakyRelu_grad/zeros/Const*
T0*

index_type0
�
7gradients/reply_xtr/dense_1/LeakyRelu_grad/GreaterEqualGreaterEqualreply_xtr/dense_1/LeakyRelu/mulreply_xtr/dense_1/BiasAdd*
T0
�
@gradients/reply_xtr/dense_1/LeakyRelu_grad/BroadcastGradientArgsBroadcastGradientArgs0gradients/reply_xtr/dense_1/LeakyRelu_grad/Shape2gradients/reply_xtr/dense_1/LeakyRelu_grad/Shape_1*
T0
�
1gradients/reply_xtr/dense_1/LeakyRelu_grad/SelectSelect7gradients/reply_xtr/dense_1/LeakyRelu_grad/GreaterEqual@gradients/reply_xtr/dense_2/MatMul_grad/tuple/control_dependency0gradients/reply_xtr/dense_1/LeakyRelu_grad/zeros*
T0
�
3gradients/reply_xtr/dense_1/LeakyRelu_grad/Select_1Select7gradients/reply_xtr/dense_1/LeakyRelu_grad/GreaterEqual0gradients/reply_xtr/dense_1/LeakyRelu_grad/zeros@gradients/reply_xtr/dense_2/MatMul_grad/tuple/control_dependency*
T0
�
.gradients/reply_xtr/dense_1/LeakyRelu_grad/SumSum1gradients/reply_xtr/dense_1/LeakyRelu_grad/Select@gradients/reply_xtr/dense_1/LeakyRelu_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
2gradients/reply_xtr/dense_1/LeakyRelu_grad/ReshapeReshape.gradients/reply_xtr/dense_1/LeakyRelu_grad/Sum0gradients/reply_xtr/dense_1/LeakyRelu_grad/Shape*
T0*
Tshape0
�
0gradients/reply_xtr/dense_1/LeakyRelu_grad/Sum_1Sum3gradients/reply_xtr/dense_1/LeakyRelu_grad/Select_1Bgradients/reply_xtr/dense_1/LeakyRelu_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
4gradients/reply_xtr/dense_1/LeakyRelu_grad/Reshape_1Reshape0gradients/reply_xtr/dense_1/LeakyRelu_grad/Sum_12gradients/reply_xtr/dense_1/LeakyRelu_grad/Shape_1*
T0*
Tshape0
�
;gradients/reply_xtr/dense_1/LeakyRelu_grad/tuple/group_depsNoOp3^gradients/reply_xtr/dense_1/LeakyRelu_grad/Reshape5^gradients/reply_xtr/dense_1/LeakyRelu_grad/Reshape_1
�
Cgradients/reply_xtr/dense_1/LeakyRelu_grad/tuple/control_dependencyIdentity2gradients/reply_xtr/dense_1/LeakyRelu_grad/Reshape<^gradients/reply_xtr/dense_1/LeakyRelu_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/reply_xtr/dense_1/LeakyRelu_grad/Reshape
�
Egradients/reply_xtr/dense_1/LeakyRelu_grad/tuple/control_dependency_1Identity4gradients/reply_xtr/dense_1/LeakyRelu_grad/Reshape_1<^gradients/reply_xtr/dense_1/LeakyRelu_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/reply_xtr/dense_1/LeakyRelu_grad/Reshape_1
^
5gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/ShapeConst*
dtype0*
valueB 
u
7gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/Shape_1Shapeexpand_xtr/dense_1/BiasAdd*
T0*
out_type0
�
Egradients/expand_xtr/dense_1/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/Shape7gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/Shape_1*
T0
�
3gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/MulMulDgradients/expand_xtr/dense_1/LeakyRelu_grad/tuple/control_dependencyexpand_xtr/dense_1/BiasAdd*
T0
�
3gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/SumSum3gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/MulEgradients/expand_xtr/dense_1/LeakyRelu/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
7gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/ReshapeReshape3gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/Sum5gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
�
5gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/Mul_1Mul"expand_xtr/dense_1/LeakyRelu/alphaDgradients/expand_xtr/dense_1/LeakyRelu_grad/tuple/control_dependency*
T0
�
5gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/Sum_1Sum5gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/Mul_1Ggradients/expand_xtr/dense_1/LeakyRelu/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
9gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/Reshape_1Reshape5gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/Sum_17gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
�
@gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/tuple/group_depsNoOp8^gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/Reshape:^gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/Reshape_1
�
Hgradients/expand_xtr/dense_1/LeakyRelu/mul_grad/tuple/control_dependencyIdentity7gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/ReshapeA^gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/Reshape
�
Jgradients/expand_xtr/dense_1/LeakyRelu/mul_grad/tuple/control_dependency_1Identity9gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/Reshape_1A^gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/expand_xtr/dense_1/LeakyRelu/mul_grad/Reshape_1
\
3gradients/like_xtr/dense_1/LeakyRelu/mul_grad/ShapeConst*
dtype0*
valueB 
q
5gradients/like_xtr/dense_1/LeakyRelu/mul_grad/Shape_1Shapelike_xtr/dense_1/BiasAdd*
T0*
out_type0
�
Cgradients/like_xtr/dense_1/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/like_xtr/dense_1/LeakyRelu/mul_grad/Shape5gradients/like_xtr/dense_1/LeakyRelu/mul_grad/Shape_1*
T0
�
1gradients/like_xtr/dense_1/LeakyRelu/mul_grad/MulMulBgradients/like_xtr/dense_1/LeakyRelu_grad/tuple/control_dependencylike_xtr/dense_1/BiasAdd*
T0
�
1gradients/like_xtr/dense_1/LeakyRelu/mul_grad/SumSum1gradients/like_xtr/dense_1/LeakyRelu/mul_grad/MulCgradients/like_xtr/dense_1/LeakyRelu/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
5gradients/like_xtr/dense_1/LeakyRelu/mul_grad/ReshapeReshape1gradients/like_xtr/dense_1/LeakyRelu/mul_grad/Sum3gradients/like_xtr/dense_1/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
�
3gradients/like_xtr/dense_1/LeakyRelu/mul_grad/Mul_1Mul like_xtr/dense_1/LeakyRelu/alphaBgradients/like_xtr/dense_1/LeakyRelu_grad/tuple/control_dependency*
T0
�
3gradients/like_xtr/dense_1/LeakyRelu/mul_grad/Sum_1Sum3gradients/like_xtr/dense_1/LeakyRelu/mul_grad/Mul_1Egradients/like_xtr/dense_1/LeakyRelu/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
7gradients/like_xtr/dense_1/LeakyRelu/mul_grad/Reshape_1Reshape3gradients/like_xtr/dense_1/LeakyRelu/mul_grad/Sum_15gradients/like_xtr/dense_1/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
�
>gradients/like_xtr/dense_1/LeakyRelu/mul_grad/tuple/group_depsNoOp6^gradients/like_xtr/dense_1/LeakyRelu/mul_grad/Reshape8^gradients/like_xtr/dense_1/LeakyRelu/mul_grad/Reshape_1
�
Fgradients/like_xtr/dense_1/LeakyRelu/mul_grad/tuple/control_dependencyIdentity5gradients/like_xtr/dense_1/LeakyRelu/mul_grad/Reshape?^gradients/like_xtr/dense_1/LeakyRelu/mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/like_xtr/dense_1/LeakyRelu/mul_grad/Reshape
�
Hgradients/like_xtr/dense_1/LeakyRelu/mul_grad/tuple/control_dependency_1Identity7gradients/like_xtr/dense_1/LeakyRelu/mul_grad/Reshape_1?^gradients/like_xtr/dense_1/LeakyRelu/mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/like_xtr/dense_1/LeakyRelu/mul_grad/Reshape_1
]
4gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/ShapeConst*
valueB *
dtype0
s
6gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/Shape_1Shapereply_xtr/dense_1/BiasAdd*
T0*
out_type0
�
Dgradients/reply_xtr/dense_1/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/Shape6gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/Shape_1*
T0
�
2gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/MulMulCgradients/reply_xtr/dense_1/LeakyRelu_grad/tuple/control_dependencyreply_xtr/dense_1/BiasAdd*
T0
�
2gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/SumSum2gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/MulDgradients/reply_xtr/dense_1/LeakyRelu/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
6gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/ReshapeReshape2gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/Sum4gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
�
4gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/Mul_1Mul!reply_xtr/dense_1/LeakyRelu/alphaCgradients/reply_xtr/dense_1/LeakyRelu_grad/tuple/control_dependency*
T0
�
4gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/Sum_1Sum4gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/Mul_1Fgradients/reply_xtr/dense_1/LeakyRelu/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
8gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/Reshape_1Reshape4gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/Sum_16gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
�
?gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/tuple/group_depsNoOp7^gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/Reshape9^gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/Reshape_1
�
Ggradients/reply_xtr/dense_1/LeakyRelu/mul_grad/tuple/control_dependencyIdentity6gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/Reshape@^gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/Reshape
�
Igradients/reply_xtr/dense_1/LeakyRelu/mul_grad/tuple/control_dependency_1Identity8gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/Reshape_1@^gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/reply_xtr/dense_1/LeakyRelu/mul_grad/Reshape_1
�
gradients/AddN_4AddNFgradients/expand_xtr/dense_1/LeakyRelu_grad/tuple/control_dependency_1Jgradients/expand_xtr/dense_1/LeakyRelu/mul_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@gradients/expand_xtr/dense_1/LeakyRelu_grad/Reshape_1*
N
v
5gradients/expand_xtr/dense_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_4*
data_formatNHWC*
T0
�
:gradients/expand_xtr/dense_1/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_46^gradients/expand_xtr/dense_1/BiasAdd_grad/BiasAddGrad
�
Bgradients/expand_xtr/dense_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_4;^gradients/expand_xtr/dense_1/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/expand_xtr/dense_1/LeakyRelu_grad/Reshape_1
�
Dgradients/expand_xtr/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/expand_xtr/dense_1/BiasAdd_grad/BiasAddGrad;^gradients/expand_xtr/dense_1/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/expand_xtr/dense_1/BiasAdd_grad/BiasAddGrad
�
gradients/AddN_5AddNDgradients/like_xtr/dense_1/LeakyRelu_grad/tuple/control_dependency_1Hgradients/like_xtr/dense_1/LeakyRelu/mul_grad/tuple/control_dependency_1*
N*
T0*F
_class<
:8loc:@gradients/like_xtr/dense_1/LeakyRelu_grad/Reshape_1
t
3gradients/like_xtr/dense_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_5*
T0*
data_formatNHWC
�
8gradients/like_xtr/dense_1/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_54^gradients/like_xtr/dense_1/BiasAdd_grad/BiasAddGrad
�
@gradients/like_xtr/dense_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_59^gradients/like_xtr/dense_1/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/like_xtr/dense_1/LeakyRelu_grad/Reshape_1
�
Bgradients/like_xtr/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/like_xtr/dense_1/BiasAdd_grad/BiasAddGrad9^gradients/like_xtr/dense_1/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/like_xtr/dense_1/BiasAdd_grad/BiasAddGrad
�
gradients/AddN_6AddNEgradients/reply_xtr/dense_1/LeakyRelu_grad/tuple/control_dependency_1Igradients/reply_xtr/dense_1/LeakyRelu/mul_grad/tuple/control_dependency_1*
T0*G
_class=
;9loc:@gradients/reply_xtr/dense_1/LeakyRelu_grad/Reshape_1*
N
u
4gradients/reply_xtr/dense_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_6*
T0*
data_formatNHWC
�
9gradients/reply_xtr/dense_1/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_65^gradients/reply_xtr/dense_1/BiasAdd_grad/BiasAddGrad
�
Agradients/reply_xtr/dense_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_6:^gradients/reply_xtr/dense_1/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/reply_xtr/dense_1/LeakyRelu_grad/Reshape_1
�
Cgradients/reply_xtr/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity4gradients/reply_xtr/dense_1/BiasAdd_grad/BiasAddGrad:^gradients/reply_xtr/dense_1/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/reply_xtr/dense_1/BiasAdd_grad/BiasAddGrad
�
/gradients/expand_xtr/dense_1/MatMul_grad/MatMulMatMulBgradients/expand_xtr/dense_1/BiasAdd_grad/tuple/control_dependencyexpand_xtr/dense_1/kernel/read*
transpose_a( *
transpose_b(*
T0
�
1gradients/expand_xtr/dense_1/MatMul_grad/MatMul_1MatMulexpand_xtr/dense/LeakyReluBgradients/expand_xtr/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
�
9gradients/expand_xtr/dense_1/MatMul_grad/tuple/group_depsNoOp0^gradients/expand_xtr/dense_1/MatMul_grad/MatMul2^gradients/expand_xtr/dense_1/MatMul_grad/MatMul_1
�
Agradients/expand_xtr/dense_1/MatMul_grad/tuple/control_dependencyIdentity/gradients/expand_xtr/dense_1/MatMul_grad/MatMul:^gradients/expand_xtr/dense_1/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/expand_xtr/dense_1/MatMul_grad/MatMul
�
Cgradients/expand_xtr/dense_1/MatMul_grad/tuple/control_dependency_1Identity1gradients/expand_xtr/dense_1/MatMul_grad/MatMul_1:^gradients/expand_xtr/dense_1/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/expand_xtr/dense_1/MatMul_grad/MatMul_1
�
-gradients/like_xtr/dense_1/MatMul_grad/MatMulMatMul@gradients/like_xtr/dense_1/BiasAdd_grad/tuple/control_dependencylike_xtr/dense_1/kernel/read*
T0*
transpose_a( *
transpose_b(
�
/gradients/like_xtr/dense_1/MatMul_grad/MatMul_1MatMullike_xtr/dense/LeakyRelu@gradients/like_xtr/dense_1/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
�
7gradients/like_xtr/dense_1/MatMul_grad/tuple/group_depsNoOp.^gradients/like_xtr/dense_1/MatMul_grad/MatMul0^gradients/like_xtr/dense_1/MatMul_grad/MatMul_1
�
?gradients/like_xtr/dense_1/MatMul_grad/tuple/control_dependencyIdentity-gradients/like_xtr/dense_1/MatMul_grad/MatMul8^gradients/like_xtr/dense_1/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/like_xtr/dense_1/MatMul_grad/MatMul
�
Agradients/like_xtr/dense_1/MatMul_grad/tuple/control_dependency_1Identity/gradients/like_xtr/dense_1/MatMul_grad/MatMul_18^gradients/like_xtr/dense_1/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/like_xtr/dense_1/MatMul_grad/MatMul_1
�
.gradients/reply_xtr/dense_1/MatMul_grad/MatMulMatMulAgradients/reply_xtr/dense_1/BiasAdd_grad/tuple/control_dependencyreply_xtr/dense_1/kernel/read*
transpose_a( *
transpose_b(*
T0
�
0gradients/reply_xtr/dense_1/MatMul_grad/MatMul_1MatMulreply_xtr/dense/LeakyReluAgradients/reply_xtr/dense_1/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
�
8gradients/reply_xtr/dense_1/MatMul_grad/tuple/group_depsNoOp/^gradients/reply_xtr/dense_1/MatMul_grad/MatMul1^gradients/reply_xtr/dense_1/MatMul_grad/MatMul_1
�
@gradients/reply_xtr/dense_1/MatMul_grad/tuple/control_dependencyIdentity.gradients/reply_xtr/dense_1/MatMul_grad/MatMul9^gradients/reply_xtr/dense_1/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/reply_xtr/dense_1/MatMul_grad/MatMul
�
Bgradients/reply_xtr/dense_1/MatMul_grad/tuple/control_dependency_1Identity0gradients/reply_xtr/dense_1/MatMul_grad/MatMul_19^gradients/reply_xtr/dense_1/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/reply_xtr/dense_1/MatMul_grad/MatMul_1
q
/gradients/expand_xtr/dense/LeakyRelu_grad/ShapeShapeexpand_xtr/dense/LeakyRelu/mul*
T0*
out_type0
m
1gradients/expand_xtr/dense/LeakyRelu_grad/Shape_1Shapeexpand_xtr/dense/BiasAdd*
T0*
out_type0
�
1gradients/expand_xtr/dense/LeakyRelu_grad/Shape_2ShapeAgradients/expand_xtr/dense_1/MatMul_grad/tuple/control_dependency*
T0*
out_type0
b
5gradients/expand_xtr/dense/LeakyRelu_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
/gradients/expand_xtr/dense/LeakyRelu_grad/zerosFill1gradients/expand_xtr/dense/LeakyRelu_grad/Shape_25gradients/expand_xtr/dense/LeakyRelu_grad/zeros/Const*
T0*

index_type0
�
6gradients/expand_xtr/dense/LeakyRelu_grad/GreaterEqualGreaterEqualexpand_xtr/dense/LeakyRelu/mulexpand_xtr/dense/BiasAdd*
T0
�
?gradients/expand_xtr/dense/LeakyRelu_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/expand_xtr/dense/LeakyRelu_grad/Shape1gradients/expand_xtr/dense/LeakyRelu_grad/Shape_1*
T0
�
0gradients/expand_xtr/dense/LeakyRelu_grad/SelectSelect6gradients/expand_xtr/dense/LeakyRelu_grad/GreaterEqualAgradients/expand_xtr/dense_1/MatMul_grad/tuple/control_dependency/gradients/expand_xtr/dense/LeakyRelu_grad/zeros*
T0
�
2gradients/expand_xtr/dense/LeakyRelu_grad/Select_1Select6gradients/expand_xtr/dense/LeakyRelu_grad/GreaterEqual/gradients/expand_xtr/dense/LeakyRelu_grad/zerosAgradients/expand_xtr/dense_1/MatMul_grad/tuple/control_dependency*
T0
�
-gradients/expand_xtr/dense/LeakyRelu_grad/SumSum0gradients/expand_xtr/dense/LeakyRelu_grad/Select?gradients/expand_xtr/dense/LeakyRelu_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
1gradients/expand_xtr/dense/LeakyRelu_grad/ReshapeReshape-gradients/expand_xtr/dense/LeakyRelu_grad/Sum/gradients/expand_xtr/dense/LeakyRelu_grad/Shape*
T0*
Tshape0
�
/gradients/expand_xtr/dense/LeakyRelu_grad/Sum_1Sum2gradients/expand_xtr/dense/LeakyRelu_grad/Select_1Agradients/expand_xtr/dense/LeakyRelu_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
3gradients/expand_xtr/dense/LeakyRelu_grad/Reshape_1Reshape/gradients/expand_xtr/dense/LeakyRelu_grad/Sum_11gradients/expand_xtr/dense/LeakyRelu_grad/Shape_1*
T0*
Tshape0
�
:gradients/expand_xtr/dense/LeakyRelu_grad/tuple/group_depsNoOp2^gradients/expand_xtr/dense/LeakyRelu_grad/Reshape4^gradients/expand_xtr/dense/LeakyRelu_grad/Reshape_1
�
Bgradients/expand_xtr/dense/LeakyRelu_grad/tuple/control_dependencyIdentity1gradients/expand_xtr/dense/LeakyRelu_grad/Reshape;^gradients/expand_xtr/dense/LeakyRelu_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/expand_xtr/dense/LeakyRelu_grad/Reshape
�
Dgradients/expand_xtr/dense/LeakyRelu_grad/tuple/control_dependency_1Identity3gradients/expand_xtr/dense/LeakyRelu_grad/Reshape_1;^gradients/expand_xtr/dense/LeakyRelu_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/expand_xtr/dense/LeakyRelu_grad/Reshape_1
m
-gradients/like_xtr/dense/LeakyRelu_grad/ShapeShapelike_xtr/dense/LeakyRelu/mul*
T0*
out_type0
i
/gradients/like_xtr/dense/LeakyRelu_grad/Shape_1Shapelike_xtr/dense/BiasAdd*
T0*
out_type0
�
/gradients/like_xtr/dense/LeakyRelu_grad/Shape_2Shape?gradients/like_xtr/dense_1/MatMul_grad/tuple/control_dependency*
T0*
out_type0
`
3gradients/like_xtr/dense/LeakyRelu_grad/zeros/ConstConst*
dtype0*
valueB
 *    
�
-gradients/like_xtr/dense/LeakyRelu_grad/zerosFill/gradients/like_xtr/dense/LeakyRelu_grad/Shape_23gradients/like_xtr/dense/LeakyRelu_grad/zeros/Const*
T0*

index_type0
�
4gradients/like_xtr/dense/LeakyRelu_grad/GreaterEqualGreaterEquallike_xtr/dense/LeakyRelu/mullike_xtr/dense/BiasAdd*
T0
�
=gradients/like_xtr/dense/LeakyRelu_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/like_xtr/dense/LeakyRelu_grad/Shape/gradients/like_xtr/dense/LeakyRelu_grad/Shape_1*
T0
�
.gradients/like_xtr/dense/LeakyRelu_grad/SelectSelect4gradients/like_xtr/dense/LeakyRelu_grad/GreaterEqual?gradients/like_xtr/dense_1/MatMul_grad/tuple/control_dependency-gradients/like_xtr/dense/LeakyRelu_grad/zeros*
T0
�
0gradients/like_xtr/dense/LeakyRelu_grad/Select_1Select4gradients/like_xtr/dense/LeakyRelu_grad/GreaterEqual-gradients/like_xtr/dense/LeakyRelu_grad/zeros?gradients/like_xtr/dense_1/MatMul_grad/tuple/control_dependency*
T0
�
+gradients/like_xtr/dense/LeakyRelu_grad/SumSum.gradients/like_xtr/dense/LeakyRelu_grad/Select=gradients/like_xtr/dense/LeakyRelu_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
/gradients/like_xtr/dense/LeakyRelu_grad/ReshapeReshape+gradients/like_xtr/dense/LeakyRelu_grad/Sum-gradients/like_xtr/dense/LeakyRelu_grad/Shape*
T0*
Tshape0
�
-gradients/like_xtr/dense/LeakyRelu_grad/Sum_1Sum0gradients/like_xtr/dense/LeakyRelu_grad/Select_1?gradients/like_xtr/dense/LeakyRelu_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
1gradients/like_xtr/dense/LeakyRelu_grad/Reshape_1Reshape-gradients/like_xtr/dense/LeakyRelu_grad/Sum_1/gradients/like_xtr/dense/LeakyRelu_grad/Shape_1*
T0*
Tshape0
�
8gradients/like_xtr/dense/LeakyRelu_grad/tuple/group_depsNoOp0^gradients/like_xtr/dense/LeakyRelu_grad/Reshape2^gradients/like_xtr/dense/LeakyRelu_grad/Reshape_1
�
@gradients/like_xtr/dense/LeakyRelu_grad/tuple/control_dependencyIdentity/gradients/like_xtr/dense/LeakyRelu_grad/Reshape9^gradients/like_xtr/dense/LeakyRelu_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/like_xtr/dense/LeakyRelu_grad/Reshape
�
Bgradients/like_xtr/dense/LeakyRelu_grad/tuple/control_dependency_1Identity1gradients/like_xtr/dense/LeakyRelu_grad/Reshape_19^gradients/like_xtr/dense/LeakyRelu_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/like_xtr/dense/LeakyRelu_grad/Reshape_1
o
.gradients/reply_xtr/dense/LeakyRelu_grad/ShapeShapereply_xtr/dense/LeakyRelu/mul*
T0*
out_type0
k
0gradients/reply_xtr/dense/LeakyRelu_grad/Shape_1Shapereply_xtr/dense/BiasAdd*
T0*
out_type0
�
0gradients/reply_xtr/dense/LeakyRelu_grad/Shape_2Shape@gradients/reply_xtr/dense_1/MatMul_grad/tuple/control_dependency*
T0*
out_type0
a
4gradients/reply_xtr/dense/LeakyRelu_grad/zeros/ConstConst*
dtype0*
valueB
 *    
�
.gradients/reply_xtr/dense/LeakyRelu_grad/zerosFill0gradients/reply_xtr/dense/LeakyRelu_grad/Shape_24gradients/reply_xtr/dense/LeakyRelu_grad/zeros/Const*
T0*

index_type0
�
5gradients/reply_xtr/dense/LeakyRelu_grad/GreaterEqualGreaterEqualreply_xtr/dense/LeakyRelu/mulreply_xtr/dense/BiasAdd*
T0
�
>gradients/reply_xtr/dense/LeakyRelu_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/reply_xtr/dense/LeakyRelu_grad/Shape0gradients/reply_xtr/dense/LeakyRelu_grad/Shape_1*
T0
�
/gradients/reply_xtr/dense/LeakyRelu_grad/SelectSelect5gradients/reply_xtr/dense/LeakyRelu_grad/GreaterEqual@gradients/reply_xtr/dense_1/MatMul_grad/tuple/control_dependency.gradients/reply_xtr/dense/LeakyRelu_grad/zeros*
T0
�
1gradients/reply_xtr/dense/LeakyRelu_grad/Select_1Select5gradients/reply_xtr/dense/LeakyRelu_grad/GreaterEqual.gradients/reply_xtr/dense/LeakyRelu_grad/zeros@gradients/reply_xtr/dense_1/MatMul_grad/tuple/control_dependency*
T0
�
,gradients/reply_xtr/dense/LeakyRelu_grad/SumSum/gradients/reply_xtr/dense/LeakyRelu_grad/Select>gradients/reply_xtr/dense/LeakyRelu_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
0gradients/reply_xtr/dense/LeakyRelu_grad/ReshapeReshape,gradients/reply_xtr/dense/LeakyRelu_grad/Sum.gradients/reply_xtr/dense/LeakyRelu_grad/Shape*
T0*
Tshape0
�
.gradients/reply_xtr/dense/LeakyRelu_grad/Sum_1Sum1gradients/reply_xtr/dense/LeakyRelu_grad/Select_1@gradients/reply_xtr/dense/LeakyRelu_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
2gradients/reply_xtr/dense/LeakyRelu_grad/Reshape_1Reshape.gradients/reply_xtr/dense/LeakyRelu_grad/Sum_10gradients/reply_xtr/dense/LeakyRelu_grad/Shape_1*
T0*
Tshape0
�
9gradients/reply_xtr/dense/LeakyRelu_grad/tuple/group_depsNoOp1^gradients/reply_xtr/dense/LeakyRelu_grad/Reshape3^gradients/reply_xtr/dense/LeakyRelu_grad/Reshape_1
�
Agradients/reply_xtr/dense/LeakyRelu_grad/tuple/control_dependencyIdentity0gradients/reply_xtr/dense/LeakyRelu_grad/Reshape:^gradients/reply_xtr/dense/LeakyRelu_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/reply_xtr/dense/LeakyRelu_grad/Reshape
�
Cgradients/reply_xtr/dense/LeakyRelu_grad/tuple/control_dependency_1Identity2gradients/reply_xtr/dense/LeakyRelu_grad/Reshape_1:^gradients/reply_xtr/dense/LeakyRelu_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/reply_xtr/dense/LeakyRelu_grad/Reshape_1
\
3gradients/expand_xtr/dense/LeakyRelu/mul_grad/ShapeConst*
valueB *
dtype0
q
5gradients/expand_xtr/dense/LeakyRelu/mul_grad/Shape_1Shapeexpand_xtr/dense/BiasAdd*
T0*
out_type0
�
Cgradients/expand_xtr/dense/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/expand_xtr/dense/LeakyRelu/mul_grad/Shape5gradients/expand_xtr/dense/LeakyRelu/mul_grad/Shape_1*
T0
�
1gradients/expand_xtr/dense/LeakyRelu/mul_grad/MulMulBgradients/expand_xtr/dense/LeakyRelu_grad/tuple/control_dependencyexpand_xtr/dense/BiasAdd*
T0
�
1gradients/expand_xtr/dense/LeakyRelu/mul_grad/SumSum1gradients/expand_xtr/dense/LeakyRelu/mul_grad/MulCgradients/expand_xtr/dense/LeakyRelu/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
5gradients/expand_xtr/dense/LeakyRelu/mul_grad/ReshapeReshape1gradients/expand_xtr/dense/LeakyRelu/mul_grad/Sum3gradients/expand_xtr/dense/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
�
3gradients/expand_xtr/dense/LeakyRelu/mul_grad/Mul_1Mul expand_xtr/dense/LeakyRelu/alphaBgradients/expand_xtr/dense/LeakyRelu_grad/tuple/control_dependency*
T0
�
3gradients/expand_xtr/dense/LeakyRelu/mul_grad/Sum_1Sum3gradients/expand_xtr/dense/LeakyRelu/mul_grad/Mul_1Egradients/expand_xtr/dense/LeakyRelu/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
7gradients/expand_xtr/dense/LeakyRelu/mul_grad/Reshape_1Reshape3gradients/expand_xtr/dense/LeakyRelu/mul_grad/Sum_15gradients/expand_xtr/dense/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
�
>gradients/expand_xtr/dense/LeakyRelu/mul_grad/tuple/group_depsNoOp6^gradients/expand_xtr/dense/LeakyRelu/mul_grad/Reshape8^gradients/expand_xtr/dense/LeakyRelu/mul_grad/Reshape_1
�
Fgradients/expand_xtr/dense/LeakyRelu/mul_grad/tuple/control_dependencyIdentity5gradients/expand_xtr/dense/LeakyRelu/mul_grad/Reshape?^gradients/expand_xtr/dense/LeakyRelu/mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/expand_xtr/dense/LeakyRelu/mul_grad/Reshape
�
Hgradients/expand_xtr/dense/LeakyRelu/mul_grad/tuple/control_dependency_1Identity7gradients/expand_xtr/dense/LeakyRelu/mul_grad/Reshape_1?^gradients/expand_xtr/dense/LeakyRelu/mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/expand_xtr/dense/LeakyRelu/mul_grad/Reshape_1
Z
1gradients/like_xtr/dense/LeakyRelu/mul_grad/ShapeConst*
valueB *
dtype0
m
3gradients/like_xtr/dense/LeakyRelu/mul_grad/Shape_1Shapelike_xtr/dense/BiasAdd*
T0*
out_type0
�
Agradients/like_xtr/dense/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/like_xtr/dense/LeakyRelu/mul_grad/Shape3gradients/like_xtr/dense/LeakyRelu/mul_grad/Shape_1*
T0
�
/gradients/like_xtr/dense/LeakyRelu/mul_grad/MulMul@gradients/like_xtr/dense/LeakyRelu_grad/tuple/control_dependencylike_xtr/dense/BiasAdd*
T0
�
/gradients/like_xtr/dense/LeakyRelu/mul_grad/SumSum/gradients/like_xtr/dense/LeakyRelu/mul_grad/MulAgradients/like_xtr/dense/LeakyRelu/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
3gradients/like_xtr/dense/LeakyRelu/mul_grad/ReshapeReshape/gradients/like_xtr/dense/LeakyRelu/mul_grad/Sum1gradients/like_xtr/dense/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
�
1gradients/like_xtr/dense/LeakyRelu/mul_grad/Mul_1Mullike_xtr/dense/LeakyRelu/alpha@gradients/like_xtr/dense/LeakyRelu_grad/tuple/control_dependency*
T0
�
1gradients/like_xtr/dense/LeakyRelu/mul_grad/Sum_1Sum1gradients/like_xtr/dense/LeakyRelu/mul_grad/Mul_1Cgradients/like_xtr/dense/LeakyRelu/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
5gradients/like_xtr/dense/LeakyRelu/mul_grad/Reshape_1Reshape1gradients/like_xtr/dense/LeakyRelu/mul_grad/Sum_13gradients/like_xtr/dense/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
�
<gradients/like_xtr/dense/LeakyRelu/mul_grad/tuple/group_depsNoOp4^gradients/like_xtr/dense/LeakyRelu/mul_grad/Reshape6^gradients/like_xtr/dense/LeakyRelu/mul_grad/Reshape_1
�
Dgradients/like_xtr/dense/LeakyRelu/mul_grad/tuple/control_dependencyIdentity3gradients/like_xtr/dense/LeakyRelu/mul_grad/Reshape=^gradients/like_xtr/dense/LeakyRelu/mul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/like_xtr/dense/LeakyRelu/mul_grad/Reshape
�
Fgradients/like_xtr/dense/LeakyRelu/mul_grad/tuple/control_dependency_1Identity5gradients/like_xtr/dense/LeakyRelu/mul_grad/Reshape_1=^gradients/like_xtr/dense/LeakyRelu/mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/like_xtr/dense/LeakyRelu/mul_grad/Reshape_1
[
2gradients/reply_xtr/dense/LeakyRelu/mul_grad/ShapeConst*
valueB *
dtype0
o
4gradients/reply_xtr/dense/LeakyRelu/mul_grad/Shape_1Shapereply_xtr/dense/BiasAdd*
T0*
out_type0
�
Bgradients/reply_xtr/dense/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients/reply_xtr/dense/LeakyRelu/mul_grad/Shape4gradients/reply_xtr/dense/LeakyRelu/mul_grad/Shape_1*
T0
�
0gradients/reply_xtr/dense/LeakyRelu/mul_grad/MulMulAgradients/reply_xtr/dense/LeakyRelu_grad/tuple/control_dependencyreply_xtr/dense/BiasAdd*
T0
�
0gradients/reply_xtr/dense/LeakyRelu/mul_grad/SumSum0gradients/reply_xtr/dense/LeakyRelu/mul_grad/MulBgradients/reply_xtr/dense/LeakyRelu/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
4gradients/reply_xtr/dense/LeakyRelu/mul_grad/ReshapeReshape0gradients/reply_xtr/dense/LeakyRelu/mul_grad/Sum2gradients/reply_xtr/dense/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
�
2gradients/reply_xtr/dense/LeakyRelu/mul_grad/Mul_1Mulreply_xtr/dense/LeakyRelu/alphaAgradients/reply_xtr/dense/LeakyRelu_grad/tuple/control_dependency*
T0
�
2gradients/reply_xtr/dense/LeakyRelu/mul_grad/Sum_1Sum2gradients/reply_xtr/dense/LeakyRelu/mul_grad/Mul_1Dgradients/reply_xtr/dense/LeakyRelu/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
6gradients/reply_xtr/dense/LeakyRelu/mul_grad/Reshape_1Reshape2gradients/reply_xtr/dense/LeakyRelu/mul_grad/Sum_14gradients/reply_xtr/dense/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
�
=gradients/reply_xtr/dense/LeakyRelu/mul_grad/tuple/group_depsNoOp5^gradients/reply_xtr/dense/LeakyRelu/mul_grad/Reshape7^gradients/reply_xtr/dense/LeakyRelu/mul_grad/Reshape_1
�
Egradients/reply_xtr/dense/LeakyRelu/mul_grad/tuple/control_dependencyIdentity4gradients/reply_xtr/dense/LeakyRelu/mul_grad/Reshape>^gradients/reply_xtr/dense/LeakyRelu/mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/reply_xtr/dense/LeakyRelu/mul_grad/Reshape
�
Ggradients/reply_xtr/dense/LeakyRelu/mul_grad/tuple/control_dependency_1Identity6gradients/reply_xtr/dense/LeakyRelu/mul_grad/Reshape_1>^gradients/reply_xtr/dense/LeakyRelu/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/reply_xtr/dense/LeakyRelu/mul_grad/Reshape_1
�
gradients/AddN_7AddNDgradients/expand_xtr/dense/LeakyRelu_grad/tuple/control_dependency_1Hgradients/expand_xtr/dense/LeakyRelu/mul_grad/tuple/control_dependency_1*
T0*F
_class<
:8loc:@gradients/expand_xtr/dense/LeakyRelu_grad/Reshape_1*
N
t
3gradients/expand_xtr/dense/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_7*
data_formatNHWC*
T0
�
8gradients/expand_xtr/dense/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_74^gradients/expand_xtr/dense/BiasAdd_grad/BiasAddGrad
�
@gradients/expand_xtr/dense/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_79^gradients/expand_xtr/dense/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/expand_xtr/dense/LeakyRelu_grad/Reshape_1
�
Bgradients/expand_xtr/dense/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/expand_xtr/dense/BiasAdd_grad/BiasAddGrad9^gradients/expand_xtr/dense/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/expand_xtr/dense/BiasAdd_grad/BiasAddGrad
�
gradients/AddN_8AddNBgradients/like_xtr/dense/LeakyRelu_grad/tuple/control_dependency_1Fgradients/like_xtr/dense/LeakyRelu/mul_grad/tuple/control_dependency_1*
T0*D
_class:
86loc:@gradients/like_xtr/dense/LeakyRelu_grad/Reshape_1*
N
r
1gradients/like_xtr/dense/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_8*
T0*
data_formatNHWC
�
6gradients/like_xtr/dense/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_82^gradients/like_xtr/dense/BiasAdd_grad/BiasAddGrad
�
>gradients/like_xtr/dense/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_87^gradients/like_xtr/dense/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/like_xtr/dense/LeakyRelu_grad/Reshape_1
�
@gradients/like_xtr/dense/BiasAdd_grad/tuple/control_dependency_1Identity1gradients/like_xtr/dense/BiasAdd_grad/BiasAddGrad7^gradients/like_xtr/dense/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/like_xtr/dense/BiasAdd_grad/BiasAddGrad
�
gradients/AddN_9AddNCgradients/reply_xtr/dense/LeakyRelu_grad/tuple/control_dependency_1Ggradients/reply_xtr/dense/LeakyRelu/mul_grad/tuple/control_dependency_1*
T0*E
_class;
97loc:@gradients/reply_xtr/dense/LeakyRelu_grad/Reshape_1*
N
s
2gradients/reply_xtr/dense/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_9*
T0*
data_formatNHWC
�
7gradients/reply_xtr/dense/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_93^gradients/reply_xtr/dense/BiasAdd_grad/BiasAddGrad
�
?gradients/reply_xtr/dense/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_98^gradients/reply_xtr/dense/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/reply_xtr/dense/LeakyRelu_grad/Reshape_1
�
Agradients/reply_xtr/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/reply_xtr/dense/BiasAdd_grad/BiasAddGrad8^gradients/reply_xtr/dense/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/reply_xtr/dense/BiasAdd_grad/BiasAddGrad
�
-gradients/expand_xtr/dense/MatMul_grad/MatMulMatMul@gradients/expand_xtr/dense/BiasAdd_grad/tuple/control_dependencyexpand_xtr/dense/kernel/read*
T0*
transpose_a( *
transpose_b(
�
/gradients/expand_xtr/dense/MatMul_grad/MatMul_1MatMulconcat@gradients/expand_xtr/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
�
7gradients/expand_xtr/dense/MatMul_grad/tuple/group_depsNoOp.^gradients/expand_xtr/dense/MatMul_grad/MatMul0^gradients/expand_xtr/dense/MatMul_grad/MatMul_1
�
?gradients/expand_xtr/dense/MatMul_grad/tuple/control_dependencyIdentity-gradients/expand_xtr/dense/MatMul_grad/MatMul8^gradients/expand_xtr/dense/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/expand_xtr/dense/MatMul_grad/MatMul
�
Agradients/expand_xtr/dense/MatMul_grad/tuple/control_dependency_1Identity/gradients/expand_xtr/dense/MatMul_grad/MatMul_18^gradients/expand_xtr/dense/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/expand_xtr/dense/MatMul_grad/MatMul_1
�
+gradients/like_xtr/dense/MatMul_grad/MatMulMatMul>gradients/like_xtr/dense/BiasAdd_grad/tuple/control_dependencylike_xtr/dense/kernel/read*
transpose_b(*
T0*
transpose_a( 
�
-gradients/like_xtr/dense/MatMul_grad/MatMul_1MatMulconcat>gradients/like_xtr/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
�
5gradients/like_xtr/dense/MatMul_grad/tuple/group_depsNoOp,^gradients/like_xtr/dense/MatMul_grad/MatMul.^gradients/like_xtr/dense/MatMul_grad/MatMul_1
�
=gradients/like_xtr/dense/MatMul_grad/tuple/control_dependencyIdentity+gradients/like_xtr/dense/MatMul_grad/MatMul6^gradients/like_xtr/dense/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/like_xtr/dense/MatMul_grad/MatMul
�
?gradients/like_xtr/dense/MatMul_grad/tuple/control_dependency_1Identity-gradients/like_xtr/dense/MatMul_grad/MatMul_16^gradients/like_xtr/dense/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/like_xtr/dense/MatMul_grad/MatMul_1
�
,gradients/reply_xtr/dense/MatMul_grad/MatMulMatMul?gradients/reply_xtr/dense/BiasAdd_grad/tuple/control_dependencyreply_xtr/dense/kernel/read*
transpose_a( *
transpose_b(*
T0
�
.gradients/reply_xtr/dense/MatMul_grad/MatMul_1MatMulconcat?gradients/reply_xtr/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
�
6gradients/reply_xtr/dense/MatMul_grad/tuple/group_depsNoOp-^gradients/reply_xtr/dense/MatMul_grad/MatMul/^gradients/reply_xtr/dense/MatMul_grad/MatMul_1
�
>gradients/reply_xtr/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients/reply_xtr/dense/MatMul_grad/MatMul7^gradients/reply_xtr/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/reply_xtr/dense/MatMul_grad/MatMul
�
@gradients/reply_xtr/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients/reply_xtr/dense/MatMul_grad/MatMul_17^gradients/reply_xtr/dense/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/reply_xtr/dense/MatMul_grad/MatMul_1
�
1expand_xtr/dense/kernel/Adagrad/Initializer/ConstConst*
valueB
��*���=**
_class 
loc:@expand_xtr/dense/kernel*
dtype0
�
expand_xtr/dense/kernel/Adagrad
VariableV2*
shared_name **
_class 
loc:@expand_xtr/dense/kernel*
dtype0*
	container *
shape:
��
�
&expand_xtr/dense/kernel/Adagrad/AssignAssignexpand_xtr/dense/kernel/Adagrad1expand_xtr/dense/kernel/Adagrad/Initializer/Const*
use_locking(*
T0**
_class 
loc:@expand_xtr/dense/kernel*
validate_shape(
�
$expand_xtr/dense/kernel/Adagrad/readIdentityexpand_xtr/dense/kernel/Adagrad*
T0**
_class 
loc:@expand_xtr/dense/kernel
�
/expand_xtr/dense/bias/Adagrad/Initializer/ConstConst*
valueB�*���=*(
_class
loc:@expand_xtr/dense/bias*
dtype0
�
expand_xtr/dense/bias/Adagrad
VariableV2*(
_class
loc:@expand_xtr/dense/bias*
dtype0*
	container *
shape:�*
shared_name 
�
$expand_xtr/dense/bias/Adagrad/AssignAssignexpand_xtr/dense/bias/Adagrad/expand_xtr/dense/bias/Adagrad/Initializer/Const*
use_locking(*
T0*(
_class
loc:@expand_xtr/dense/bias*
validate_shape(
�
"expand_xtr/dense/bias/Adagrad/readIdentityexpand_xtr/dense/bias/Adagrad*
T0*(
_class
loc:@expand_xtr/dense/bias
�
3expand_xtr/dense_1/kernel/Adagrad/Initializer/ConstConst*
valueB
��*���=*,
_class"
 loc:@expand_xtr/dense_1/kernel*
dtype0
�
!expand_xtr/dense_1/kernel/Adagrad
VariableV2*
dtype0*
	container *
shape:
��*
shared_name *,
_class"
 loc:@expand_xtr/dense_1/kernel
�
(expand_xtr/dense_1/kernel/Adagrad/AssignAssign!expand_xtr/dense_1/kernel/Adagrad3expand_xtr/dense_1/kernel/Adagrad/Initializer/Const*
T0*,
_class"
 loc:@expand_xtr/dense_1/kernel*
validate_shape(*
use_locking(
�
&expand_xtr/dense_1/kernel/Adagrad/readIdentity!expand_xtr/dense_1/kernel/Adagrad*
T0*,
_class"
 loc:@expand_xtr/dense_1/kernel
�
1expand_xtr/dense_1/bias/Adagrad/Initializer/ConstConst*
dtype0*
valueB�*���=**
_class 
loc:@expand_xtr/dense_1/bias
�
expand_xtr/dense_1/bias/Adagrad
VariableV2**
_class 
loc:@expand_xtr/dense_1/bias*
dtype0*
	container *
shape:�*
shared_name 
�
&expand_xtr/dense_1/bias/Adagrad/AssignAssignexpand_xtr/dense_1/bias/Adagrad1expand_xtr/dense_1/bias/Adagrad/Initializer/Const*
use_locking(*
T0**
_class 
loc:@expand_xtr/dense_1/bias*
validate_shape(
�
$expand_xtr/dense_1/bias/Adagrad/readIdentityexpand_xtr/dense_1/bias/Adagrad*
T0**
_class 
loc:@expand_xtr/dense_1/bias
�
3expand_xtr/dense_2/kernel/Adagrad/Initializer/ConstConst*
valueB	�@*���=*,
_class"
 loc:@expand_xtr/dense_2/kernel*
dtype0
�
!expand_xtr/dense_2/kernel/Adagrad
VariableV2*
shared_name *,
_class"
 loc:@expand_xtr/dense_2/kernel*
dtype0*
	container *
shape:	�@
�
(expand_xtr/dense_2/kernel/Adagrad/AssignAssign!expand_xtr/dense_2/kernel/Adagrad3expand_xtr/dense_2/kernel/Adagrad/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@expand_xtr/dense_2/kernel*
validate_shape(
�
&expand_xtr/dense_2/kernel/Adagrad/readIdentity!expand_xtr/dense_2/kernel/Adagrad*
T0*,
_class"
 loc:@expand_xtr/dense_2/kernel
�
1expand_xtr/dense_2/bias/Adagrad/Initializer/ConstConst*
valueB@*���=**
_class 
loc:@expand_xtr/dense_2/bias*
dtype0
�
expand_xtr/dense_2/bias/Adagrad
VariableV2*
shape:@*
shared_name **
_class 
loc:@expand_xtr/dense_2/bias*
dtype0*
	container 
�
&expand_xtr/dense_2/bias/Adagrad/AssignAssignexpand_xtr/dense_2/bias/Adagrad1expand_xtr/dense_2/bias/Adagrad/Initializer/Const*
use_locking(*
T0**
_class 
loc:@expand_xtr/dense_2/bias*
validate_shape(
�
$expand_xtr/dense_2/bias/Adagrad/readIdentityexpand_xtr/dense_2/bias/Adagrad*
T0**
_class 
loc:@expand_xtr/dense_2/bias
�
3expand_xtr/dense_3/kernel/Adagrad/Initializer/ConstConst*
dtype0*
valueB@*���=*,
_class"
 loc:@expand_xtr/dense_3/kernel
�
!expand_xtr/dense_3/kernel/Adagrad
VariableV2*
shared_name *,
_class"
 loc:@expand_xtr/dense_3/kernel*
dtype0*
	container *
shape
:@
�
(expand_xtr/dense_3/kernel/Adagrad/AssignAssign!expand_xtr/dense_3/kernel/Adagrad3expand_xtr/dense_3/kernel/Adagrad/Initializer/Const*
T0*,
_class"
 loc:@expand_xtr/dense_3/kernel*
validate_shape(*
use_locking(
�
&expand_xtr/dense_3/kernel/Adagrad/readIdentity!expand_xtr/dense_3/kernel/Adagrad*
T0*,
_class"
 loc:@expand_xtr/dense_3/kernel
�
1expand_xtr/dense_3/bias/Adagrad/Initializer/ConstConst*
valueB*���=**
_class 
loc:@expand_xtr/dense_3/bias*
dtype0
�
expand_xtr/dense_3/bias/Adagrad
VariableV2*
shared_name **
_class 
loc:@expand_xtr/dense_3/bias*
dtype0*
	container *
shape:
�
&expand_xtr/dense_3/bias/Adagrad/AssignAssignexpand_xtr/dense_3/bias/Adagrad1expand_xtr/dense_3/bias/Adagrad/Initializer/Const*
use_locking(*
T0**
_class 
loc:@expand_xtr/dense_3/bias*
validate_shape(
�
$expand_xtr/dense_3/bias/Adagrad/readIdentityexpand_xtr/dense_3/bias/Adagrad*
T0**
_class 
loc:@expand_xtr/dense_3/bias
�
/like_xtr/dense/kernel/Adagrad/Initializer/ConstConst*
valueB
��*���=*(
_class
loc:@like_xtr/dense/kernel*
dtype0
�
like_xtr/dense/kernel/Adagrad
VariableV2*
dtype0*
	container *
shape:
��*
shared_name *(
_class
loc:@like_xtr/dense/kernel
�
$like_xtr/dense/kernel/Adagrad/AssignAssignlike_xtr/dense/kernel/Adagrad/like_xtr/dense/kernel/Adagrad/Initializer/Const*
use_locking(*
T0*(
_class
loc:@like_xtr/dense/kernel*
validate_shape(
�
"like_xtr/dense/kernel/Adagrad/readIdentitylike_xtr/dense/kernel/Adagrad*
T0*(
_class
loc:@like_xtr/dense/kernel
�
-like_xtr/dense/bias/Adagrad/Initializer/ConstConst*
valueB�*���=*&
_class
loc:@like_xtr/dense/bias*
dtype0
�
like_xtr/dense/bias/Adagrad
VariableV2*&
_class
loc:@like_xtr/dense/bias*
dtype0*
	container *
shape:�*
shared_name 
�
"like_xtr/dense/bias/Adagrad/AssignAssignlike_xtr/dense/bias/Adagrad-like_xtr/dense/bias/Adagrad/Initializer/Const*
use_locking(*
T0*&
_class
loc:@like_xtr/dense/bias*
validate_shape(
z
 like_xtr/dense/bias/Adagrad/readIdentitylike_xtr/dense/bias/Adagrad*
T0*&
_class
loc:@like_xtr/dense/bias
�
1like_xtr/dense_1/kernel/Adagrad/Initializer/ConstConst*
dtype0*
valueB
��*���=**
_class 
loc:@like_xtr/dense_1/kernel
�
like_xtr/dense_1/kernel/Adagrad
VariableV2*
shape:
��*
shared_name **
_class 
loc:@like_xtr/dense_1/kernel*
dtype0*
	container 
�
&like_xtr/dense_1/kernel/Adagrad/AssignAssignlike_xtr/dense_1/kernel/Adagrad1like_xtr/dense_1/kernel/Adagrad/Initializer/Const*
validate_shape(*
use_locking(*
T0**
_class 
loc:@like_xtr/dense_1/kernel
�
$like_xtr/dense_1/kernel/Adagrad/readIdentitylike_xtr/dense_1/kernel/Adagrad*
T0**
_class 
loc:@like_xtr/dense_1/kernel
�
/like_xtr/dense_1/bias/Adagrad/Initializer/ConstConst*
dtype0*
valueB�*���=*(
_class
loc:@like_xtr/dense_1/bias
�
like_xtr/dense_1/bias/Adagrad
VariableV2*
dtype0*
	container *
shape:�*
shared_name *(
_class
loc:@like_xtr/dense_1/bias
�
$like_xtr/dense_1/bias/Adagrad/AssignAssignlike_xtr/dense_1/bias/Adagrad/like_xtr/dense_1/bias/Adagrad/Initializer/Const*
use_locking(*
T0*(
_class
loc:@like_xtr/dense_1/bias*
validate_shape(
�
"like_xtr/dense_1/bias/Adagrad/readIdentitylike_xtr/dense_1/bias/Adagrad*
T0*(
_class
loc:@like_xtr/dense_1/bias
�
1like_xtr/dense_2/kernel/Adagrad/Initializer/ConstConst*
valueB	�@*���=**
_class 
loc:@like_xtr/dense_2/kernel*
dtype0
�
like_xtr/dense_2/kernel/Adagrad
VariableV2**
_class 
loc:@like_xtr/dense_2/kernel*
dtype0*
	container *
shape:	�@*
shared_name 
�
&like_xtr/dense_2/kernel/Adagrad/AssignAssignlike_xtr/dense_2/kernel/Adagrad1like_xtr/dense_2/kernel/Adagrad/Initializer/Const*
T0**
_class 
loc:@like_xtr/dense_2/kernel*
validate_shape(*
use_locking(
�
$like_xtr/dense_2/kernel/Adagrad/readIdentitylike_xtr/dense_2/kernel/Adagrad*
T0**
_class 
loc:@like_xtr/dense_2/kernel
�
/like_xtr/dense_2/bias/Adagrad/Initializer/ConstConst*
valueB@*���=*(
_class
loc:@like_xtr/dense_2/bias*
dtype0
�
like_xtr/dense_2/bias/Adagrad
VariableV2*
dtype0*
	container *
shape:@*
shared_name *(
_class
loc:@like_xtr/dense_2/bias
�
$like_xtr/dense_2/bias/Adagrad/AssignAssignlike_xtr/dense_2/bias/Adagrad/like_xtr/dense_2/bias/Adagrad/Initializer/Const*
T0*(
_class
loc:@like_xtr/dense_2/bias*
validate_shape(*
use_locking(
�
"like_xtr/dense_2/bias/Adagrad/readIdentitylike_xtr/dense_2/bias/Adagrad*
T0*(
_class
loc:@like_xtr/dense_2/bias
�
1like_xtr/dense_3/kernel/Adagrad/Initializer/ConstConst*
dtype0*
valueB@*���=**
_class 
loc:@like_xtr/dense_3/kernel
�
like_xtr/dense_3/kernel/Adagrad
VariableV2*
dtype0*
	container *
shape
:@*
shared_name **
_class 
loc:@like_xtr/dense_3/kernel
�
&like_xtr/dense_3/kernel/Adagrad/AssignAssignlike_xtr/dense_3/kernel/Adagrad1like_xtr/dense_3/kernel/Adagrad/Initializer/Const*
use_locking(*
T0**
_class 
loc:@like_xtr/dense_3/kernel*
validate_shape(
�
$like_xtr/dense_3/kernel/Adagrad/readIdentitylike_xtr/dense_3/kernel/Adagrad*
T0**
_class 
loc:@like_xtr/dense_3/kernel
�
/like_xtr/dense_3/bias/Adagrad/Initializer/ConstConst*
valueB*���=*(
_class
loc:@like_xtr/dense_3/bias*
dtype0
�
like_xtr/dense_3/bias/Adagrad
VariableV2*
dtype0*
	container *
shape:*
shared_name *(
_class
loc:@like_xtr/dense_3/bias
�
$like_xtr/dense_3/bias/Adagrad/AssignAssignlike_xtr/dense_3/bias/Adagrad/like_xtr/dense_3/bias/Adagrad/Initializer/Const*
use_locking(*
T0*(
_class
loc:@like_xtr/dense_3/bias*
validate_shape(
�
"like_xtr/dense_3/bias/Adagrad/readIdentitylike_xtr/dense_3/bias/Adagrad*
T0*(
_class
loc:@like_xtr/dense_3/bias
�
0reply_xtr/dense/kernel/Adagrad/Initializer/ConstConst*
valueB
��*���=*)
_class
loc:@reply_xtr/dense/kernel*
dtype0
�
reply_xtr/dense/kernel/Adagrad
VariableV2*
dtype0*
	container *
shape:
��*
shared_name *)
_class
loc:@reply_xtr/dense/kernel
�
%reply_xtr/dense/kernel/Adagrad/AssignAssignreply_xtr/dense/kernel/Adagrad0reply_xtr/dense/kernel/Adagrad/Initializer/Const*
use_locking(*
T0*)
_class
loc:@reply_xtr/dense/kernel*
validate_shape(
�
#reply_xtr/dense/kernel/Adagrad/readIdentityreply_xtr/dense/kernel/Adagrad*
T0*)
_class
loc:@reply_xtr/dense/kernel
�
.reply_xtr/dense/bias/Adagrad/Initializer/ConstConst*
valueB�*���=*'
_class
loc:@reply_xtr/dense/bias*
dtype0
�
reply_xtr/dense/bias/Adagrad
VariableV2*
dtype0*
	container *
shape:�*
shared_name *'
_class
loc:@reply_xtr/dense/bias
�
#reply_xtr/dense/bias/Adagrad/AssignAssignreply_xtr/dense/bias/Adagrad.reply_xtr/dense/bias/Adagrad/Initializer/Const*
use_locking(*
T0*'
_class
loc:@reply_xtr/dense/bias*
validate_shape(
}
!reply_xtr/dense/bias/Adagrad/readIdentityreply_xtr/dense/bias/Adagrad*
T0*'
_class
loc:@reply_xtr/dense/bias
�
2reply_xtr/dense_1/kernel/Adagrad/Initializer/ConstConst*
dtype0*
valueB
��*���=*+
_class!
loc:@reply_xtr/dense_1/kernel
�
 reply_xtr/dense_1/kernel/Adagrad
VariableV2*
shared_name *+
_class!
loc:@reply_xtr/dense_1/kernel*
dtype0*
	container *
shape:
��
�
'reply_xtr/dense_1/kernel/Adagrad/AssignAssign reply_xtr/dense_1/kernel/Adagrad2reply_xtr/dense_1/kernel/Adagrad/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@reply_xtr/dense_1/kernel*
validate_shape(
�
%reply_xtr/dense_1/kernel/Adagrad/readIdentity reply_xtr/dense_1/kernel/Adagrad*
T0*+
_class!
loc:@reply_xtr/dense_1/kernel
�
0reply_xtr/dense_1/bias/Adagrad/Initializer/ConstConst*
valueB�*���=*)
_class
loc:@reply_xtr/dense_1/bias*
dtype0
�
reply_xtr/dense_1/bias/Adagrad
VariableV2*
shape:�*
shared_name *)
_class
loc:@reply_xtr/dense_1/bias*
dtype0*
	container 
�
%reply_xtr/dense_1/bias/Adagrad/AssignAssignreply_xtr/dense_1/bias/Adagrad0reply_xtr/dense_1/bias/Adagrad/Initializer/Const*
validate_shape(*
use_locking(*
T0*)
_class
loc:@reply_xtr/dense_1/bias
�
#reply_xtr/dense_1/bias/Adagrad/readIdentityreply_xtr/dense_1/bias/Adagrad*
T0*)
_class
loc:@reply_xtr/dense_1/bias
�
2reply_xtr/dense_2/kernel/Adagrad/Initializer/ConstConst*
valueB	�@*���=*+
_class!
loc:@reply_xtr/dense_2/kernel*
dtype0
�
 reply_xtr/dense_2/kernel/Adagrad
VariableV2*
shape:	�@*
shared_name *+
_class!
loc:@reply_xtr/dense_2/kernel*
dtype0*
	container 
�
'reply_xtr/dense_2/kernel/Adagrad/AssignAssign reply_xtr/dense_2/kernel/Adagrad2reply_xtr/dense_2/kernel/Adagrad/Initializer/Const*
validate_shape(*
use_locking(*
T0*+
_class!
loc:@reply_xtr/dense_2/kernel
�
%reply_xtr/dense_2/kernel/Adagrad/readIdentity reply_xtr/dense_2/kernel/Adagrad*
T0*+
_class!
loc:@reply_xtr/dense_2/kernel
�
0reply_xtr/dense_2/bias/Adagrad/Initializer/ConstConst*
valueB@*���=*)
_class
loc:@reply_xtr/dense_2/bias*
dtype0
�
reply_xtr/dense_2/bias/Adagrad
VariableV2*
shared_name *)
_class
loc:@reply_xtr/dense_2/bias*
dtype0*
	container *
shape:@
�
%reply_xtr/dense_2/bias/Adagrad/AssignAssignreply_xtr/dense_2/bias/Adagrad0reply_xtr/dense_2/bias/Adagrad/Initializer/Const*
validate_shape(*
use_locking(*
T0*)
_class
loc:@reply_xtr/dense_2/bias
�
#reply_xtr/dense_2/bias/Adagrad/readIdentityreply_xtr/dense_2/bias/Adagrad*
T0*)
_class
loc:@reply_xtr/dense_2/bias
�
2reply_xtr/dense_3/kernel/Adagrad/Initializer/ConstConst*
valueB@*���=*+
_class!
loc:@reply_xtr/dense_3/kernel*
dtype0
�
 reply_xtr/dense_3/kernel/Adagrad
VariableV2*
shared_name *+
_class!
loc:@reply_xtr/dense_3/kernel*
dtype0*
	container *
shape
:@
�
'reply_xtr/dense_3/kernel/Adagrad/AssignAssign reply_xtr/dense_3/kernel/Adagrad2reply_xtr/dense_3/kernel/Adagrad/Initializer/Const*
validate_shape(*
use_locking(*
T0*+
_class!
loc:@reply_xtr/dense_3/kernel
�
%reply_xtr/dense_3/kernel/Adagrad/readIdentity reply_xtr/dense_3/kernel/Adagrad*
T0*+
_class!
loc:@reply_xtr/dense_3/kernel
�
0reply_xtr/dense_3/bias/Adagrad/Initializer/ConstConst*
dtype0*
valueB*���=*)
_class
loc:@reply_xtr/dense_3/bias
�
reply_xtr/dense_3/bias/Adagrad
VariableV2*
shape:*
shared_name *)
_class
loc:@reply_xtr/dense_3/bias*
dtype0*
	container 
�
%reply_xtr/dense_3/bias/Adagrad/AssignAssignreply_xtr/dense_3/bias/Adagrad0reply_xtr/dense_3/bias/Adagrad/Initializer/Const*
use_locking(*
T0*)
_class
loc:@reply_xtr/dense_3/bias*
validate_shape(
�
#reply_xtr/dense_3/bias/Adagrad/readIdentityreply_xtr/dense_3/bias/Adagrad*
T0*)
_class
loc:@reply_xtr/dense_3/bias
B
Adagrad/learning_rateConst*
valueB
 *o�:*
dtype0
�
3Adagrad/update_expand_xtr/dense/kernel/ApplyAdagradApplyAdagradexpand_xtr/dense/kernelexpand_xtr/dense/kernel/AdagradAdagrad/learning_rateAgradients/expand_xtr/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@expand_xtr/dense/kernel*
update_slots(
�
1Adagrad/update_expand_xtr/dense/bias/ApplyAdagradApplyAdagradexpand_xtr/dense/biasexpand_xtr/dense/bias/AdagradAdagrad/learning_rateBgradients/expand_xtr/dense/BiasAdd_grad/tuple/control_dependency_1*
update_slots(*
use_locking( *
T0*(
_class
loc:@expand_xtr/dense/bias
�
5Adagrad/update_expand_xtr/dense_1/kernel/ApplyAdagradApplyAdagradexpand_xtr/dense_1/kernel!expand_xtr/dense_1/kernel/AdagradAdagrad/learning_rateCgradients/expand_xtr/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@expand_xtr/dense_1/kernel*
update_slots(
�
3Adagrad/update_expand_xtr/dense_1/bias/ApplyAdagradApplyAdagradexpand_xtr/dense_1/biasexpand_xtr/dense_1/bias/AdagradAdagrad/learning_rateDgradients/expand_xtr/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@expand_xtr/dense_1/bias*
update_slots(
�
5Adagrad/update_expand_xtr/dense_2/kernel/ApplyAdagradApplyAdagradexpand_xtr/dense_2/kernel!expand_xtr/dense_2/kernel/AdagradAdagrad/learning_rateCgradients/expand_xtr/dense_2/MatMul_grad/tuple/control_dependency_1*
T0*,
_class"
 loc:@expand_xtr/dense_2/kernel*
update_slots(*
use_locking( 
�
3Adagrad/update_expand_xtr/dense_2/bias/ApplyAdagradApplyAdagradexpand_xtr/dense_2/biasexpand_xtr/dense_2/bias/AdagradAdagrad/learning_rateDgradients/expand_xtr/dense_2/BiasAdd_grad/tuple/control_dependency_1*
T0**
_class 
loc:@expand_xtr/dense_2/bias*
update_slots(*
use_locking( 
�
5Adagrad/update_expand_xtr/dense_3/kernel/ApplyAdagradApplyAdagradexpand_xtr/dense_3/kernel!expand_xtr/dense_3/kernel/AdagradAdagrad/learning_rateCgradients/expand_xtr/dense_3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@expand_xtr/dense_3/kernel*
update_slots(
�
3Adagrad/update_expand_xtr/dense_3/bias/ApplyAdagradApplyAdagradexpand_xtr/dense_3/biasexpand_xtr/dense_3/bias/AdagradAdagrad/learning_rateDgradients/expand_xtr/dense_3/BiasAdd_grad/tuple/control_dependency_1*
update_slots(*
use_locking( *
T0**
_class 
loc:@expand_xtr/dense_3/bias
�
1Adagrad/update_like_xtr/dense/kernel/ApplyAdagradApplyAdagradlike_xtr/dense/kernellike_xtr/dense/kernel/AdagradAdagrad/learning_rate?gradients/like_xtr/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*(
_class
loc:@like_xtr/dense/kernel*
update_slots(
�
/Adagrad/update_like_xtr/dense/bias/ApplyAdagradApplyAdagradlike_xtr/dense/biaslike_xtr/dense/bias/AdagradAdagrad/learning_rate@gradients/like_xtr/dense/BiasAdd_grad/tuple/control_dependency_1*
update_slots(*
use_locking( *
T0*&
_class
loc:@like_xtr/dense/bias
�
3Adagrad/update_like_xtr/dense_1/kernel/ApplyAdagradApplyAdagradlike_xtr/dense_1/kernellike_xtr/dense_1/kernel/AdagradAdagrad/learning_rateAgradients/like_xtr/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@like_xtr/dense_1/kernel*
update_slots(
�
1Adagrad/update_like_xtr/dense_1/bias/ApplyAdagradApplyAdagradlike_xtr/dense_1/biaslike_xtr/dense_1/bias/AdagradAdagrad/learning_rateBgradients/like_xtr/dense_1/BiasAdd_grad/tuple/control_dependency_1*
update_slots(*
use_locking( *
T0*(
_class
loc:@like_xtr/dense_1/bias
�
3Adagrad/update_like_xtr/dense_2/kernel/ApplyAdagradApplyAdagradlike_xtr/dense_2/kernellike_xtr/dense_2/kernel/AdagradAdagrad/learning_rateAgradients/like_xtr/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@like_xtr/dense_2/kernel*
update_slots(
�
1Adagrad/update_like_xtr/dense_2/bias/ApplyAdagradApplyAdagradlike_xtr/dense_2/biaslike_xtr/dense_2/bias/AdagradAdagrad/learning_rateBgradients/like_xtr/dense_2/BiasAdd_grad/tuple/control_dependency_1*
update_slots(*
use_locking( *
T0*(
_class
loc:@like_xtr/dense_2/bias
�
3Adagrad/update_like_xtr/dense_3/kernel/ApplyAdagradApplyAdagradlike_xtr/dense_3/kernellike_xtr/dense_3/kernel/AdagradAdagrad/learning_rateAgradients/like_xtr/dense_3/MatMul_grad/tuple/control_dependency_1*
T0**
_class 
loc:@like_xtr/dense_3/kernel*
update_slots(*
use_locking( 
�
1Adagrad/update_like_xtr/dense_3/bias/ApplyAdagradApplyAdagradlike_xtr/dense_3/biaslike_xtr/dense_3/bias/AdagradAdagrad/learning_rateBgradients/like_xtr/dense_3/BiasAdd_grad/tuple/control_dependency_1*
update_slots(*
use_locking( *
T0*(
_class
loc:@like_xtr/dense_3/bias
�
2Adagrad/update_reply_xtr/dense/kernel/ApplyAdagradApplyAdagradreply_xtr/dense/kernelreply_xtr/dense/kernel/AdagradAdagrad/learning_rate@gradients/reply_xtr/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@reply_xtr/dense/kernel*
update_slots(
�
0Adagrad/update_reply_xtr/dense/bias/ApplyAdagradApplyAdagradreply_xtr/dense/biasreply_xtr/dense/bias/AdagradAdagrad/learning_rateAgradients/reply_xtr/dense/BiasAdd_grad/tuple/control_dependency_1*
update_slots(*
use_locking( *
T0*'
_class
loc:@reply_xtr/dense/bias
�
4Adagrad/update_reply_xtr/dense_1/kernel/ApplyAdagradApplyAdagradreply_xtr/dense_1/kernel reply_xtr/dense_1/kernel/AdagradAdagrad/learning_rateBgradients/reply_xtr/dense_1/MatMul_grad/tuple/control_dependency_1*
update_slots(*
use_locking( *
T0*+
_class!
loc:@reply_xtr/dense_1/kernel
�
2Adagrad/update_reply_xtr/dense_1/bias/ApplyAdagradApplyAdagradreply_xtr/dense_1/biasreply_xtr/dense_1/bias/AdagradAdagrad/learning_rateCgradients/reply_xtr/dense_1/BiasAdd_grad/tuple/control_dependency_1*
T0*)
_class
loc:@reply_xtr/dense_1/bias*
update_slots(*
use_locking( 
�
4Adagrad/update_reply_xtr/dense_2/kernel/ApplyAdagradApplyAdagradreply_xtr/dense_2/kernel reply_xtr/dense_2/kernel/AdagradAdagrad/learning_rateBgradients/reply_xtr/dense_2/MatMul_grad/tuple/control_dependency_1*
update_slots(*
use_locking( *
T0*+
_class!
loc:@reply_xtr/dense_2/kernel
�
2Adagrad/update_reply_xtr/dense_2/bias/ApplyAdagradApplyAdagradreply_xtr/dense_2/biasreply_xtr/dense_2/bias/AdagradAdagrad/learning_rateCgradients/reply_xtr/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@reply_xtr/dense_2/bias*
update_slots(
�
4Adagrad/update_reply_xtr/dense_3/kernel/ApplyAdagradApplyAdagradreply_xtr/dense_3/kernel reply_xtr/dense_3/kernel/AdagradAdagrad/learning_rateBgradients/reply_xtr/dense_3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@reply_xtr/dense_3/kernel*
update_slots(
�
2Adagrad/update_reply_xtr/dense_3/bias/ApplyAdagradApplyAdagradreply_xtr/dense_3/biasreply_xtr/dense_3/bias/AdagradAdagrad/learning_rateCgradients/reply_xtr/dense_3/BiasAdd_grad/tuple/control_dependency_1*
T0*)
_class
loc:@reply_xtr/dense_3/bias*
update_slots(*
use_locking( 
�

AdagradNoOp2^Adagrad/update_expand_xtr/dense/bias/ApplyAdagrad4^Adagrad/update_expand_xtr/dense/kernel/ApplyAdagrad4^Adagrad/update_expand_xtr/dense_1/bias/ApplyAdagrad6^Adagrad/update_expand_xtr/dense_1/kernel/ApplyAdagrad4^Adagrad/update_expand_xtr/dense_2/bias/ApplyAdagrad6^Adagrad/update_expand_xtr/dense_2/kernel/ApplyAdagrad4^Adagrad/update_expand_xtr/dense_3/bias/ApplyAdagrad6^Adagrad/update_expand_xtr/dense_3/kernel/ApplyAdagrad0^Adagrad/update_like_xtr/dense/bias/ApplyAdagrad2^Adagrad/update_like_xtr/dense/kernel/ApplyAdagrad2^Adagrad/update_like_xtr/dense_1/bias/ApplyAdagrad4^Adagrad/update_like_xtr/dense_1/kernel/ApplyAdagrad2^Adagrad/update_like_xtr/dense_2/bias/ApplyAdagrad4^Adagrad/update_like_xtr/dense_2/kernel/ApplyAdagrad2^Adagrad/update_like_xtr/dense_3/bias/ApplyAdagrad4^Adagrad/update_like_xtr/dense_3/kernel/ApplyAdagrad1^Adagrad/update_reply_xtr/dense/bias/ApplyAdagrad3^Adagrad/update_reply_xtr/dense/kernel/ApplyAdagrad3^Adagrad/update_reply_xtr/dense_1/bias/ApplyAdagrad5^Adagrad/update_reply_xtr/dense_1/kernel/ApplyAdagrad3^Adagrad/update_reply_xtr/dense_2/bias/ApplyAdagrad5^Adagrad/update_reply_xtr/dense_2/kernel/ApplyAdagrad3^Adagrad/update_reply_xtr/dense_3/bias/ApplyAdagrad5^Adagrad/update_reply_xtr/dense_3/kernel/ApplyAdagrad
C
ShapeShapeexpand_xtr/dense_3/Sigmoid*
T0*
out_type0
C
strided_slice_4/stackConst*
valueB: *
dtype0
E
strided_slice_4/stack_1Const*
dtype0*
valueB:
E
strided_slice_4/stack_2Const*
valueB:*
dtype0
�
strided_slice_4StridedSliceShapestrided_slice_4/stackstrided_slice_4/stack_1strided_slice_4/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
3
ps_densePlaceholder*
dtype0*
shape:
S
%dense_assign_init/strided_slice/stackConst*
valueB: *
dtype0
W
'dense_assign_init/strided_slice/stack_1Const*
valueB:��*
dtype0
U
'dense_assign_init/strided_slice/stack_2Const*
dtype0*
valueB:
�
dense_assign_init/strided_sliceStridedSliceps_dense%dense_assign_init/strided_slice/stack'dense_assign_init/strided_slice/stack_1'dense_assign_init/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0
T
dense_assign_init/Reshape/shapeConst*
valueB"P     *
dtype0
}
dense_assign_init/ReshapeReshapedense_assign_init/strided_slicedense_assign_init/Reshape/shape*
T0*
Tshape0
�
dense_assign_init/AssignAssignexpand_xtr/dense/kerneldense_assign_init/Reshape*
use_locking(*
T0**
_class 
loc:@expand_xtr/dense/kernel*
validate_shape(
W
'dense_assign_init/strided_slice_1/stackConst*
valueB:��*
dtype0
Y
)dense_assign_init/strided_slice_1/stack_1Const*
valueB:��*
dtype0
W
)dense_assign_init/strided_slice_1/stack_2Const*
valueB:*
dtype0
�
!dense_assign_init/strided_slice_1StridedSliceps_dense'dense_assign_init/strided_slice_1/stack)dense_assign_init/strided_slice_1/stack_1)dense_assign_init/strided_slice_1/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask 
P
!dense_assign_init/Reshape_1/shapeConst*
valueB:�*
dtype0
�
dense_assign_init/Reshape_1Reshape!dense_assign_init/strided_slice_1!dense_assign_init/Reshape_1/shape*
T0*
Tshape0
�
dense_assign_init/Assign_1Assignexpand_xtr/dense/biasdense_assign_init/Reshape_1*
T0*(
_class
loc:@expand_xtr/dense/bias*
validate_shape(*
use_locking(
W
'dense_assign_init/strided_slice_2/stackConst*
dtype0*
valueB:��
Y
)dense_assign_init/strided_slice_2/stack_1Const*
dtype0*
valueB:��
W
)dense_assign_init/strided_slice_2/stack_2Const*
valueB:*
dtype0
�
!dense_assign_init/strided_slice_2StridedSliceps_dense'dense_assign_init/strided_slice_2/stack)dense_assign_init/strided_slice_2/stack_1)dense_assign_init/strided_slice_2/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask 
V
!dense_assign_init/Reshape_2/shapeConst*
valueB"   �   *
dtype0
�
dense_assign_init/Reshape_2Reshape!dense_assign_init/strided_slice_2!dense_assign_init/Reshape_2/shape*
T0*
Tshape0
�
dense_assign_init/Assign_2Assignexpand_xtr/dense_1/kerneldense_assign_init/Reshape_2*
use_locking(*
T0*,
_class"
 loc:@expand_xtr/dense_1/kernel*
validate_shape(
W
'dense_assign_init/strided_slice_3/stackConst*
valueB:��*
dtype0
Y
)dense_assign_init/strided_slice_3/stack_1Const*
valueB:��*
dtype0
W
)dense_assign_init/strided_slice_3/stack_2Const*
valueB:*
dtype0
�
!dense_assign_init/strided_slice_3StridedSliceps_dense'dense_assign_init/strided_slice_3/stack)dense_assign_init/strided_slice_3/stack_1)dense_assign_init/strided_slice_3/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
P
!dense_assign_init/Reshape_3/shapeConst*
dtype0*
valueB:�
�
dense_assign_init/Reshape_3Reshape!dense_assign_init/strided_slice_3!dense_assign_init/Reshape_3/shape*
T0*
Tshape0
�
dense_assign_init/Assign_3Assignexpand_xtr/dense_1/biasdense_assign_init/Reshape_3*
use_locking(*
T0**
_class 
loc:@expand_xtr/dense_1/bias*
validate_shape(
W
'dense_assign_init/strided_slice_4/stackConst*
valueB:��*
dtype0
Y
)dense_assign_init/strided_slice_4/stack_1Const*
valueB:��*
dtype0
W
)dense_assign_init/strided_slice_4/stack_2Const*
valueB:*
dtype0
�
!dense_assign_init/strided_slice_4StridedSliceps_dense'dense_assign_init/strided_slice_4/stack)dense_assign_init/strided_slice_4/stack_1)dense_assign_init/strided_slice_4/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
V
!dense_assign_init/Reshape_4/shapeConst*
dtype0*
valueB"�   @   
�
dense_assign_init/Reshape_4Reshape!dense_assign_init/strided_slice_4!dense_assign_init/Reshape_4/shape*
T0*
Tshape0
�
dense_assign_init/Assign_4Assignexpand_xtr/dense_2/kerneldense_assign_init/Reshape_4*
T0*,
_class"
 loc:@expand_xtr/dense_2/kernel*
validate_shape(*
use_locking(
W
'dense_assign_init/strided_slice_5/stackConst*
valueB:��*
dtype0
Y
)dense_assign_init/strided_slice_5/stack_1Const*
valueB:��*
dtype0
W
)dense_assign_init/strided_slice_5/stack_2Const*
valueB:*
dtype0
�
!dense_assign_init/strided_slice_5StridedSliceps_dense'dense_assign_init/strided_slice_5/stack)dense_assign_init/strided_slice_5/stack_1)dense_assign_init/strided_slice_5/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
O
!dense_assign_init/Reshape_5/shapeConst*
valueB:@*
dtype0
�
dense_assign_init/Reshape_5Reshape!dense_assign_init/strided_slice_5!dense_assign_init/Reshape_5/shape*
T0*
Tshape0
�
dense_assign_init/Assign_5Assignexpand_xtr/dense_2/biasdense_assign_init/Reshape_5*
use_locking(*
T0**
_class 
loc:@expand_xtr/dense_2/bias*
validate_shape(
W
'dense_assign_init/strided_slice_6/stackConst*
valueB:��*
dtype0
Y
)dense_assign_init/strided_slice_6/stack_1Const*
valueB:��*
dtype0
W
)dense_assign_init/strided_slice_6/stack_2Const*
valueB:*
dtype0
�
!dense_assign_init/strided_slice_6StridedSliceps_dense'dense_assign_init/strided_slice_6/stack)dense_assign_init/strided_slice_6/stack_1)dense_assign_init/strided_slice_6/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0
V
!dense_assign_init/Reshape_6/shapeConst*
dtype0*
valueB"@      
�
dense_assign_init/Reshape_6Reshape!dense_assign_init/strided_slice_6!dense_assign_init/Reshape_6/shape*
T0*
Tshape0
�
dense_assign_init/Assign_6Assignexpand_xtr/dense_3/kerneldense_assign_init/Reshape_6*
T0*,
_class"
 loc:@expand_xtr/dense_3/kernel*
validate_shape(*
use_locking(
W
'dense_assign_init/strided_slice_7/stackConst*
valueB:��*
dtype0
Y
)dense_assign_init/strided_slice_7/stack_1Const*
dtype0*
valueB:��
W
)dense_assign_init/strided_slice_7/stack_2Const*
valueB:*
dtype0
�
!dense_assign_init/strided_slice_7StridedSliceps_dense'dense_assign_init/strided_slice_7/stack)dense_assign_init/strided_slice_7/stack_1)dense_assign_init/strided_slice_7/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
O
!dense_assign_init/Reshape_7/shapeConst*
valueB:*
dtype0
�
dense_assign_init/Reshape_7Reshape!dense_assign_init/strided_slice_7!dense_assign_init/Reshape_7/shape*
T0*
Tshape0
�
dense_assign_init/Assign_7Assignexpand_xtr/dense_3/biasdense_assign_init/Reshape_7*
T0**
_class 
loc:@expand_xtr/dense_3/bias*
validate_shape(*
use_locking(
W
'dense_assign_init/strided_slice_8/stackConst*
valueB:��*
dtype0
Y
)dense_assign_init/strided_slice_8/stack_1Const*
valueB:��*
dtype0
W
)dense_assign_init/strided_slice_8/stack_2Const*
valueB:*
dtype0
�
!dense_assign_init/strided_slice_8StridedSliceps_dense'dense_assign_init/strided_slice_8/stack)dense_assign_init/strided_slice_8/stack_1)dense_assign_init/strided_slice_8/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
V
!dense_assign_init/Reshape_8/shapeConst*
valueB"P     *
dtype0
�
dense_assign_init/Reshape_8Reshape!dense_assign_init/strided_slice_8!dense_assign_init/Reshape_8/shape*
T0*
Tshape0
�
dense_assign_init/Assign_8Assignlike_xtr/dense/kerneldense_assign_init/Reshape_8*
use_locking(*
T0*(
_class
loc:@like_xtr/dense/kernel*
validate_shape(
W
'dense_assign_init/strided_slice_9/stackConst*
dtype0*
valueB:��
Y
)dense_assign_init/strided_slice_9/stack_1Const*
valueB:��*
dtype0
W
)dense_assign_init/strided_slice_9/stack_2Const*
valueB:*
dtype0
�
!dense_assign_init/strided_slice_9StridedSliceps_dense'dense_assign_init/strided_slice_9/stack)dense_assign_init/strided_slice_9/stack_1)dense_assign_init/strided_slice_9/stack_2*
end_mask *
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask 
P
!dense_assign_init/Reshape_9/shapeConst*
valueB:�*
dtype0
�
dense_assign_init/Reshape_9Reshape!dense_assign_init/strided_slice_9!dense_assign_init/Reshape_9/shape*
T0*
Tshape0
�
dense_assign_init/Assign_9Assignlike_xtr/dense/biasdense_assign_init/Reshape_9*
validate_shape(*
use_locking(*
T0*&
_class
loc:@like_xtr/dense/bias
X
(dense_assign_init/strided_slice_10/stackConst*
dtype0*
valueB:��
Z
*dense_assign_init/strided_slice_10/stack_1Const*
dtype0*
valueB:��
X
*dense_assign_init/strided_slice_10/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_10StridedSliceps_dense(dense_assign_init/strided_slice_10/stack*dense_assign_init/strided_slice_10/stack_1*dense_assign_init/strided_slice_10/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
W
"dense_assign_init/Reshape_10/shapeConst*
dtype0*
valueB"   �   
�
dense_assign_init/Reshape_10Reshape"dense_assign_init/strided_slice_10"dense_assign_init/Reshape_10/shape*
T0*
Tshape0
�
dense_assign_init/Assign_10Assignlike_xtr/dense_1/kerneldense_assign_init/Reshape_10*
use_locking(*
T0**
_class 
loc:@like_xtr/dense_1/kernel*
validate_shape(
X
(dense_assign_init/strided_slice_11/stackConst*
valueB:��*
dtype0
Z
*dense_assign_init/strided_slice_11/stack_1Const*
dtype0*
valueB:��
X
*dense_assign_init/strided_slice_11/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_11StridedSliceps_dense(dense_assign_init/strided_slice_11/stack*dense_assign_init/strided_slice_11/stack_1*dense_assign_init/strided_slice_11/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask 
Q
"dense_assign_init/Reshape_11/shapeConst*
valueB:�*
dtype0
�
dense_assign_init/Reshape_11Reshape"dense_assign_init/strided_slice_11"dense_assign_init/Reshape_11/shape*
T0*
Tshape0
�
dense_assign_init/Assign_11Assignlike_xtr/dense_1/biasdense_assign_init/Reshape_11*
validate_shape(*
use_locking(*
T0*(
_class
loc:@like_xtr/dense_1/bias
X
(dense_assign_init/strided_slice_12/stackConst*
valueB:��*
dtype0
Z
*dense_assign_init/strided_slice_12/stack_1Const*
valueB:��*
dtype0
X
*dense_assign_init/strided_slice_12/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_12StridedSliceps_dense(dense_assign_init/strided_slice_12/stack*dense_assign_init/strided_slice_12/stack_1*dense_assign_init/strided_slice_12/stack_2*
end_mask *
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask 
W
"dense_assign_init/Reshape_12/shapeConst*
dtype0*
valueB"�   @   
�
dense_assign_init/Reshape_12Reshape"dense_assign_init/strided_slice_12"dense_assign_init/Reshape_12/shape*
T0*
Tshape0
�
dense_assign_init/Assign_12Assignlike_xtr/dense_2/kerneldense_assign_init/Reshape_12*
T0**
_class 
loc:@like_xtr/dense_2/kernel*
validate_shape(*
use_locking(
X
(dense_assign_init/strided_slice_13/stackConst*
dtype0*
valueB:��
Z
*dense_assign_init/strided_slice_13/stack_1Const*
dtype0*
valueB:��
X
*dense_assign_init/strided_slice_13/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_13StridedSliceps_dense(dense_assign_init/strided_slice_13/stack*dense_assign_init/strided_slice_13/stack_1*dense_assign_init/strided_slice_13/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0
P
"dense_assign_init/Reshape_13/shapeConst*
valueB:@*
dtype0
�
dense_assign_init/Reshape_13Reshape"dense_assign_init/strided_slice_13"dense_assign_init/Reshape_13/shape*
T0*
Tshape0
�
dense_assign_init/Assign_13Assignlike_xtr/dense_2/biasdense_assign_init/Reshape_13*
validate_shape(*
use_locking(*
T0*(
_class
loc:@like_xtr/dense_2/bias
X
(dense_assign_init/strided_slice_14/stackConst*
valueB:��*
dtype0
Z
*dense_assign_init/strided_slice_14/stack_1Const*
valueB:��*
dtype0
X
*dense_assign_init/strided_slice_14/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_14StridedSliceps_dense(dense_assign_init/strided_slice_14/stack*dense_assign_init/strided_slice_14/stack_1*dense_assign_init/strided_slice_14/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
W
"dense_assign_init/Reshape_14/shapeConst*
valueB"@      *
dtype0
�
dense_assign_init/Reshape_14Reshape"dense_assign_init/strided_slice_14"dense_assign_init/Reshape_14/shape*
T0*
Tshape0
�
dense_assign_init/Assign_14Assignlike_xtr/dense_3/kerneldense_assign_init/Reshape_14*
T0**
_class 
loc:@like_xtr/dense_3/kernel*
validate_shape(*
use_locking(
X
(dense_assign_init/strided_slice_15/stackConst*
valueB:��*
dtype0
Z
*dense_assign_init/strided_slice_15/stack_1Const*
valueB:��*
dtype0
X
*dense_assign_init/strided_slice_15/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_15StridedSliceps_dense(dense_assign_init/strided_slice_15/stack*dense_assign_init/strided_slice_15/stack_1*dense_assign_init/strided_slice_15/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0
P
"dense_assign_init/Reshape_15/shapeConst*
valueB:*
dtype0
�
dense_assign_init/Reshape_15Reshape"dense_assign_init/strided_slice_15"dense_assign_init/Reshape_15/shape*
T0*
Tshape0
�
dense_assign_init/Assign_15Assignlike_xtr/dense_3/biasdense_assign_init/Reshape_15*
validate_shape(*
use_locking(*
T0*(
_class
loc:@like_xtr/dense_3/bias
X
(dense_assign_init/strided_slice_16/stackConst*
valueB:��*
dtype0
Z
*dense_assign_init/strided_slice_16/stack_1Const*
valueB:��*
dtype0
X
*dense_assign_init/strided_slice_16/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_16StridedSliceps_dense(dense_assign_init/strided_slice_16/stack*dense_assign_init/strided_slice_16/stack_1*dense_assign_init/strided_slice_16/stack_2*
end_mask *
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask 
W
"dense_assign_init/Reshape_16/shapeConst*
dtype0*
valueB"P     
�
dense_assign_init/Reshape_16Reshape"dense_assign_init/strided_slice_16"dense_assign_init/Reshape_16/shape*
T0*
Tshape0
�
dense_assign_init/Assign_16Assignreply_xtr/dense/kerneldense_assign_init/Reshape_16*
use_locking(*
T0*)
_class
loc:@reply_xtr/dense/kernel*
validate_shape(
X
(dense_assign_init/strided_slice_17/stackConst*
valueB:��*
dtype0
Z
*dense_assign_init/strided_slice_17/stack_1Const*
valueB:��*
dtype0
X
*dense_assign_init/strided_slice_17/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_17StridedSliceps_dense(dense_assign_init/strided_slice_17/stack*dense_assign_init/strided_slice_17/stack_1*dense_assign_init/strided_slice_17/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask 
Q
"dense_assign_init/Reshape_17/shapeConst*
valueB:�*
dtype0
�
dense_assign_init/Reshape_17Reshape"dense_assign_init/strided_slice_17"dense_assign_init/Reshape_17/shape*
T0*
Tshape0
�
dense_assign_init/Assign_17Assignreply_xtr/dense/biasdense_assign_init/Reshape_17*
use_locking(*
T0*'
_class
loc:@reply_xtr/dense/bias*
validate_shape(
X
(dense_assign_init/strided_slice_18/stackConst*
valueB:��*
dtype0
Z
*dense_assign_init/strided_slice_18/stack_1Const*
dtype0*
valueB:��
X
*dense_assign_init/strided_slice_18/stack_2Const*
dtype0*
valueB:
�
"dense_assign_init/strided_slice_18StridedSliceps_dense(dense_assign_init/strided_slice_18/stack*dense_assign_init/strided_slice_18/stack_1*dense_assign_init/strided_slice_18/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
W
"dense_assign_init/Reshape_18/shapeConst*
valueB"   �   *
dtype0
�
dense_assign_init/Reshape_18Reshape"dense_assign_init/strided_slice_18"dense_assign_init/Reshape_18/shape*
T0*
Tshape0
�
dense_assign_init/Assign_18Assignreply_xtr/dense_1/kerneldense_assign_init/Reshape_18*
use_locking(*
T0*+
_class!
loc:@reply_xtr/dense_1/kernel*
validate_shape(
X
(dense_assign_init/strided_slice_19/stackConst*
valueB:��*
dtype0
Z
*dense_assign_init/strided_slice_19/stack_1Const*
valueB:��*
dtype0
X
*dense_assign_init/strided_slice_19/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_19StridedSliceps_dense(dense_assign_init/strided_slice_19/stack*dense_assign_init/strided_slice_19/stack_1*dense_assign_init/strided_slice_19/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0
Q
"dense_assign_init/Reshape_19/shapeConst*
valueB:�*
dtype0
�
dense_assign_init/Reshape_19Reshape"dense_assign_init/strided_slice_19"dense_assign_init/Reshape_19/shape*
T0*
Tshape0
�
dense_assign_init/Assign_19Assignreply_xtr/dense_1/biasdense_assign_init/Reshape_19*
use_locking(*
T0*)
_class
loc:@reply_xtr/dense_1/bias*
validate_shape(
X
(dense_assign_init/strided_slice_20/stackConst*
valueB:��*
dtype0
Z
*dense_assign_init/strided_slice_20/stack_1Const*
dtype0*
valueB:��
X
*dense_assign_init/strided_slice_20/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_20StridedSliceps_dense(dense_assign_init/strided_slice_20/stack*dense_assign_init/strided_slice_20/stack_1*dense_assign_init/strided_slice_20/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask 
W
"dense_assign_init/Reshape_20/shapeConst*
valueB"�   @   *
dtype0
�
dense_assign_init/Reshape_20Reshape"dense_assign_init/strided_slice_20"dense_assign_init/Reshape_20/shape*
T0*
Tshape0
�
dense_assign_init/Assign_20Assignreply_xtr/dense_2/kerneldense_assign_init/Reshape_20*
use_locking(*
T0*+
_class!
loc:@reply_xtr/dense_2/kernel*
validate_shape(
X
(dense_assign_init/strided_slice_21/stackConst*
dtype0*
valueB:��
Z
*dense_assign_init/strided_slice_21/stack_1Const*
valueB:«*
dtype0
X
*dense_assign_init/strided_slice_21/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_21StridedSliceps_dense(dense_assign_init/strided_slice_21/stack*dense_assign_init/strided_slice_21/stack_1*dense_assign_init/strided_slice_21/stack_2*
end_mask *
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask 
P
"dense_assign_init/Reshape_21/shapeConst*
valueB:@*
dtype0
�
dense_assign_init/Reshape_21Reshape"dense_assign_init/strided_slice_21"dense_assign_init/Reshape_21/shape*
T0*
Tshape0
�
dense_assign_init/Assign_21Assignreply_xtr/dense_2/biasdense_assign_init/Reshape_21*
use_locking(*
T0*)
_class
loc:@reply_xtr/dense_2/bias*
validate_shape(
X
(dense_assign_init/strided_slice_22/stackConst*
valueB:«*
dtype0
Z
*dense_assign_init/strided_slice_22/stack_1Const*
valueB:��*
dtype0
X
*dense_assign_init/strided_slice_22/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_22StridedSliceps_dense(dense_assign_init/strided_slice_22/stack*dense_assign_init/strided_slice_22/stack_1*dense_assign_init/strided_slice_22/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
W
"dense_assign_init/Reshape_22/shapeConst*
valueB"@      *
dtype0
�
dense_assign_init/Reshape_22Reshape"dense_assign_init/strided_slice_22"dense_assign_init/Reshape_22/shape*
T0*
Tshape0
�
dense_assign_init/Assign_22Assignreply_xtr/dense_3/kerneldense_assign_init/Reshape_22*
use_locking(*
T0*+
_class!
loc:@reply_xtr/dense_3/kernel*
validate_shape(
X
(dense_assign_init/strided_slice_23/stackConst*
valueB:��*
dtype0
Z
*dense_assign_init/strided_slice_23/stack_1Const*
dtype0*
valueB:��
X
*dense_assign_init/strided_slice_23/stack_2Const*
dtype0*
valueB:
�
"dense_assign_init/strided_slice_23StridedSliceps_dense(dense_assign_init/strided_slice_23/stack*dense_assign_init/strided_slice_23/stack_1*dense_assign_init/strided_slice_23/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
P
"dense_assign_init/Reshape_23/shapeConst*
valueB:*
dtype0
�
dense_assign_init/Reshape_23Reshape"dense_assign_init/strided_slice_23"dense_assign_init/Reshape_23/shape*
T0*
Tshape0
�
dense_assign_init/Assign_23Assignreply_xtr/dense_3/biasdense_assign_init/Reshape_23*
T0*)
_class
loc:@reply_xtr/dense_3/bias*
validate_shape(*
use_locking(
X
(dense_assign_init/strided_slice_24/stackConst*
valueB:��*
dtype0
Z
*dense_assign_init/strided_slice_24/stack_1Const*
valueB:��*
dtype0
X
*dense_assign_init/strided_slice_24/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_24StridedSliceps_dense(dense_assign_init/strided_slice_24/stack*dense_assign_init/strided_slice_24/stack_1*dense_assign_init/strided_slice_24/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0
W
"dense_assign_init/Reshape_24/shapeConst*
valueB"P     *
dtype0
�
dense_assign_init/Reshape_24Reshape"dense_assign_init/strided_slice_24"dense_assign_init/Reshape_24/shape*
T0*
Tshape0
�
dense_assign_init/Assign_24Assignexpand_xtr/dense/kernel/Adagraddense_assign_init/Reshape_24*
use_locking(*
T0**
_class 
loc:@expand_xtr/dense/kernel*
validate_shape(
X
(dense_assign_init/strided_slice_25/stackConst*
valueB:��*
dtype0
Z
*dense_assign_init/strided_slice_25/stack_1Const*
valueB:��*
dtype0
X
*dense_assign_init/strided_slice_25/stack_2Const*
dtype0*
valueB:
�
"dense_assign_init/strided_slice_25StridedSliceps_dense(dense_assign_init/strided_slice_25/stack*dense_assign_init/strided_slice_25/stack_1*dense_assign_init/strided_slice_25/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
Q
"dense_assign_init/Reshape_25/shapeConst*
valueB:�*
dtype0
�
dense_assign_init/Reshape_25Reshape"dense_assign_init/strided_slice_25"dense_assign_init/Reshape_25/shape*
T0*
Tshape0
�
dense_assign_init/Assign_25Assignexpand_xtr/dense/bias/Adagraddense_assign_init/Reshape_25*
use_locking(*
T0*(
_class
loc:@expand_xtr/dense/bias*
validate_shape(
X
(dense_assign_init/strided_slice_26/stackConst*
valueB:��*
dtype0
Z
*dense_assign_init/strided_slice_26/stack_1Const*
dtype0*
valueB:��
X
*dense_assign_init/strided_slice_26/stack_2Const*
dtype0*
valueB:
�
"dense_assign_init/strided_slice_26StridedSliceps_dense(dense_assign_init/strided_slice_26/stack*dense_assign_init/strided_slice_26/stack_1*dense_assign_init/strided_slice_26/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask 
W
"dense_assign_init/Reshape_26/shapeConst*
valueB"   �   *
dtype0
�
dense_assign_init/Reshape_26Reshape"dense_assign_init/strided_slice_26"dense_assign_init/Reshape_26/shape*
T0*
Tshape0
�
dense_assign_init/Assign_26Assign!expand_xtr/dense_1/kernel/Adagraddense_assign_init/Reshape_26*
use_locking(*
T0*,
_class"
 loc:@expand_xtr/dense_1/kernel*
validate_shape(
X
(dense_assign_init/strided_slice_27/stackConst*
dtype0*
valueB:��
Z
*dense_assign_init/strided_slice_27/stack_1Const*
valueB:��*
dtype0
X
*dense_assign_init/strided_slice_27/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_27StridedSliceps_dense(dense_assign_init/strided_slice_27/stack*dense_assign_init/strided_slice_27/stack_1*dense_assign_init/strided_slice_27/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0
Q
"dense_assign_init/Reshape_27/shapeConst*
valueB:�*
dtype0
�
dense_assign_init/Reshape_27Reshape"dense_assign_init/strided_slice_27"dense_assign_init/Reshape_27/shape*
T0*
Tshape0
�
dense_assign_init/Assign_27Assignexpand_xtr/dense_1/bias/Adagraddense_assign_init/Reshape_27*
use_locking(*
T0**
_class 
loc:@expand_xtr/dense_1/bias*
validate_shape(
X
(dense_assign_init/strided_slice_28/stackConst*
valueB:��*
dtype0
Z
*dense_assign_init/strided_slice_28/stack_1Const*
valueB:��*
dtype0
X
*dense_assign_init/strided_slice_28/stack_2Const*
dtype0*
valueB:
�
"dense_assign_init/strided_slice_28StridedSliceps_dense(dense_assign_init/strided_slice_28/stack*dense_assign_init/strided_slice_28/stack_1*dense_assign_init/strided_slice_28/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
W
"dense_assign_init/Reshape_28/shapeConst*
valueB"�   @   *
dtype0
�
dense_assign_init/Reshape_28Reshape"dense_assign_init/strided_slice_28"dense_assign_init/Reshape_28/shape*
T0*
Tshape0
�
dense_assign_init/Assign_28Assign!expand_xtr/dense_2/kernel/Adagraddense_assign_init/Reshape_28*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@expand_xtr/dense_2/kernel
X
(dense_assign_init/strided_slice_29/stackConst*
valueB:��*
dtype0
Z
*dense_assign_init/strided_slice_29/stack_1Const*
dtype0*
valueB:Ï
X
*dense_assign_init/strided_slice_29/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_29StridedSliceps_dense(dense_assign_init/strided_slice_29/stack*dense_assign_init/strided_slice_29/stack_1*dense_assign_init/strided_slice_29/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
P
"dense_assign_init/Reshape_29/shapeConst*
valueB:@*
dtype0
�
dense_assign_init/Reshape_29Reshape"dense_assign_init/strided_slice_29"dense_assign_init/Reshape_29/shape*
T0*
Tshape0
�
dense_assign_init/Assign_29Assignexpand_xtr/dense_2/bias/Adagraddense_assign_init/Reshape_29*
use_locking(*
T0**
_class 
loc:@expand_xtr/dense_2/bias*
validate_shape(
X
(dense_assign_init/strided_slice_30/stackConst*
valueB:Ï*
dtype0
Z
*dense_assign_init/strided_slice_30/stack_1Const*
valueB:��*
dtype0
X
*dense_assign_init/strided_slice_30/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_30StridedSliceps_dense(dense_assign_init/strided_slice_30/stack*dense_assign_init/strided_slice_30/stack_1*dense_assign_init/strided_slice_30/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask 
W
"dense_assign_init/Reshape_30/shapeConst*
dtype0*
valueB"@      
�
dense_assign_init/Reshape_30Reshape"dense_assign_init/strided_slice_30"dense_assign_init/Reshape_30/shape*
T0*
Tshape0
�
dense_assign_init/Assign_30Assign!expand_xtr/dense_3/kernel/Adagraddense_assign_init/Reshape_30*
T0*,
_class"
 loc:@expand_xtr/dense_3/kernel*
validate_shape(*
use_locking(
X
(dense_assign_init/strided_slice_31/stackConst*
valueB:��*
dtype0
Z
*dense_assign_init/strided_slice_31/stack_1Const*
dtype0*
valueB:��
X
*dense_assign_init/strided_slice_31/stack_2Const*
dtype0*
valueB:
�
"dense_assign_init/strided_slice_31StridedSliceps_dense(dense_assign_init/strided_slice_31/stack*dense_assign_init/strided_slice_31/stack_1*dense_assign_init/strided_slice_31/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask 
P
"dense_assign_init/Reshape_31/shapeConst*
valueB:*
dtype0
�
dense_assign_init/Reshape_31Reshape"dense_assign_init/strided_slice_31"dense_assign_init/Reshape_31/shape*
T0*
Tshape0
�
dense_assign_init/Assign_31Assignexpand_xtr/dense_3/bias/Adagraddense_assign_init/Reshape_31*
T0**
_class 
loc:@expand_xtr/dense_3/bias*
validate_shape(*
use_locking(
X
(dense_assign_init/strided_slice_32/stackConst*
valueB:��*
dtype0
Z
*dense_assign_init/strided_slice_32/stack_1Const*
dtype0*
valueB:��$
X
*dense_assign_init/strided_slice_32/stack_2Const*
dtype0*
valueB:
�
"dense_assign_init/strided_slice_32StridedSliceps_dense(dense_assign_init/strided_slice_32/stack*dense_assign_init/strided_slice_32/stack_1*dense_assign_init/strided_slice_32/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask 
W
"dense_assign_init/Reshape_32/shapeConst*
valueB"P     *
dtype0
�
dense_assign_init/Reshape_32Reshape"dense_assign_init/strided_slice_32"dense_assign_init/Reshape_32/shape*
T0*
Tshape0
�
dense_assign_init/Assign_32Assignlike_xtr/dense/kernel/Adagraddense_assign_init/Reshape_32*
use_locking(*
T0*(
_class
loc:@like_xtr/dense/kernel*
validate_shape(
X
(dense_assign_init/strided_slice_33/stackConst*
valueB:��$*
dtype0
Z
*dense_assign_init/strided_slice_33/stack_1Const*
valueB:��$*
dtype0
X
*dense_assign_init/strided_slice_33/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_33StridedSliceps_dense(dense_assign_init/strided_slice_33/stack*dense_assign_init/strided_slice_33/stack_1*dense_assign_init/strided_slice_33/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
Q
"dense_assign_init/Reshape_33/shapeConst*
valueB:�*
dtype0
�
dense_assign_init/Reshape_33Reshape"dense_assign_init/strided_slice_33"dense_assign_init/Reshape_33/shape*
T0*
Tshape0
�
dense_assign_init/Assign_33Assignlike_xtr/dense/bias/Adagraddense_assign_init/Reshape_33*
T0*&
_class
loc:@like_xtr/dense/bias*
validate_shape(*
use_locking(
X
(dense_assign_init/strided_slice_34/stackConst*
valueB:��$*
dtype0
Z
*dense_assign_init/strided_slice_34/stack_1Const*
dtype0*
valueB:��&
X
*dense_assign_init/strided_slice_34/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_34StridedSliceps_dense(dense_assign_init/strided_slice_34/stack*dense_assign_init/strided_slice_34/stack_1*dense_assign_init/strided_slice_34/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
W
"dense_assign_init/Reshape_34/shapeConst*
dtype0*
valueB"   �   
�
dense_assign_init/Reshape_34Reshape"dense_assign_init/strided_slice_34"dense_assign_init/Reshape_34/shape*
T0*
Tshape0
�
dense_assign_init/Assign_34Assignlike_xtr/dense_1/kernel/Adagraddense_assign_init/Reshape_34*
validate_shape(*
use_locking(*
T0**
_class 
loc:@like_xtr/dense_1/kernel
X
(dense_assign_init/strided_slice_35/stackConst*
valueB:��&*
dtype0
Z
*dense_assign_init/strided_slice_35/stack_1Const*
valueB:��&*
dtype0
X
*dense_assign_init/strided_slice_35/stack_2Const*
dtype0*
valueB:
�
"dense_assign_init/strided_slice_35StridedSliceps_dense(dense_assign_init/strided_slice_35/stack*dense_assign_init/strided_slice_35/stack_1*dense_assign_init/strided_slice_35/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0
Q
"dense_assign_init/Reshape_35/shapeConst*
valueB:�*
dtype0
�
dense_assign_init/Reshape_35Reshape"dense_assign_init/strided_slice_35"dense_assign_init/Reshape_35/shape*
T0*
Tshape0
�
dense_assign_init/Assign_35Assignlike_xtr/dense_1/bias/Adagraddense_assign_init/Reshape_35*
validate_shape(*
use_locking(*
T0*(
_class
loc:@like_xtr/dense_1/bias
X
(dense_assign_init/strided_slice_36/stackConst*
valueB:��&*
dtype0
Z
*dense_assign_init/strided_slice_36/stack_1Const*
valueB:��&*
dtype0
X
*dense_assign_init/strided_slice_36/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_36StridedSliceps_dense(dense_assign_init/strided_slice_36/stack*dense_assign_init/strided_slice_36/stack_1*dense_assign_init/strided_slice_36/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
W
"dense_assign_init/Reshape_36/shapeConst*
valueB"�   @   *
dtype0
�
dense_assign_init/Reshape_36Reshape"dense_assign_init/strided_slice_36"dense_assign_init/Reshape_36/shape*
T0*
Tshape0
�
dense_assign_init/Assign_36Assignlike_xtr/dense_2/kernel/Adagraddense_assign_init/Reshape_36*
T0**
_class 
loc:@like_xtr/dense_2/kernel*
validate_shape(*
use_locking(
X
(dense_assign_init/strided_slice_37/stackConst*
valueB:��&*
dtype0
Z
*dense_assign_init/strided_slice_37/stack_1Const*
valueB:��&*
dtype0
X
*dense_assign_init/strided_slice_37/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_37StridedSliceps_dense(dense_assign_init/strided_slice_37/stack*dense_assign_init/strided_slice_37/stack_1*dense_assign_init/strided_slice_37/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
P
"dense_assign_init/Reshape_37/shapeConst*
valueB:@*
dtype0
�
dense_assign_init/Reshape_37Reshape"dense_assign_init/strided_slice_37"dense_assign_init/Reshape_37/shape*
T0*
Tshape0
�
dense_assign_init/Assign_37Assignlike_xtr/dense_2/bias/Adagraddense_assign_init/Reshape_37*
use_locking(*
T0*(
_class
loc:@like_xtr/dense_2/bias*
validate_shape(
X
(dense_assign_init/strided_slice_38/stackConst*
valueB:��&*
dtype0
Z
*dense_assign_init/strided_slice_38/stack_1Const*
dtype0*
valueB:��&
X
*dense_assign_init/strided_slice_38/stack_2Const*
dtype0*
valueB:
�
"dense_assign_init/strided_slice_38StridedSliceps_dense(dense_assign_init/strided_slice_38/stack*dense_assign_init/strided_slice_38/stack_1*dense_assign_init/strided_slice_38/stack_2*
end_mask *
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask 
W
"dense_assign_init/Reshape_38/shapeConst*
valueB"@      *
dtype0
�
dense_assign_init/Reshape_38Reshape"dense_assign_init/strided_slice_38"dense_assign_init/Reshape_38/shape*
T0*
Tshape0
�
dense_assign_init/Assign_38Assignlike_xtr/dense_3/kernel/Adagraddense_assign_init/Reshape_38*
use_locking(*
T0**
_class 
loc:@like_xtr/dense_3/kernel*
validate_shape(
X
(dense_assign_init/strided_slice_39/stackConst*
valueB:��&*
dtype0
Z
*dense_assign_init/strided_slice_39/stack_1Const*
valueB:��&*
dtype0
X
*dense_assign_init/strided_slice_39/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_39StridedSliceps_dense(dense_assign_init/strided_slice_39/stack*dense_assign_init/strided_slice_39/stack_1*dense_assign_init/strided_slice_39/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
P
"dense_assign_init/Reshape_39/shapeConst*
dtype0*
valueB:
�
dense_assign_init/Reshape_39Reshape"dense_assign_init/strided_slice_39"dense_assign_init/Reshape_39/shape*
T0*
Tshape0
�
dense_assign_init/Assign_39Assignlike_xtr/dense_3/bias/Adagraddense_assign_init/Reshape_39*
T0*(
_class
loc:@like_xtr/dense_3/bias*
validate_shape(*
use_locking(
X
(dense_assign_init/strided_slice_40/stackConst*
valueB:��&*
dtype0
Z
*dense_assign_init/strided_slice_40/stack_1Const*
dtype0*
valueB:��,
X
*dense_assign_init/strided_slice_40/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_40StridedSliceps_dense(dense_assign_init/strided_slice_40/stack*dense_assign_init/strided_slice_40/stack_1*dense_assign_init/strided_slice_40/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
W
"dense_assign_init/Reshape_40/shapeConst*
valueB"P     *
dtype0
�
dense_assign_init/Reshape_40Reshape"dense_assign_init/strided_slice_40"dense_assign_init/Reshape_40/shape*
T0*
Tshape0
�
dense_assign_init/Assign_40Assignreply_xtr/dense/kernel/Adagraddense_assign_init/Reshape_40*
validate_shape(*
use_locking(*
T0*)
_class
loc:@reply_xtr/dense/kernel
X
(dense_assign_init/strided_slice_41/stackConst*
valueB:��,*
dtype0
Z
*dense_assign_init/strided_slice_41/stack_1Const*
dtype0*
valueB:��,
X
*dense_assign_init/strided_slice_41/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_41StridedSliceps_dense(dense_assign_init/strided_slice_41/stack*dense_assign_init/strided_slice_41/stack_1*dense_assign_init/strided_slice_41/stack_2*
end_mask *
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask 
Q
"dense_assign_init/Reshape_41/shapeConst*
valueB:�*
dtype0
�
dense_assign_init/Reshape_41Reshape"dense_assign_init/strided_slice_41"dense_assign_init/Reshape_41/shape*
T0*
Tshape0
�
dense_assign_init/Assign_41Assignreply_xtr/dense/bias/Adagraddense_assign_init/Reshape_41*
validate_shape(*
use_locking(*
T0*'
_class
loc:@reply_xtr/dense/bias
X
(dense_assign_init/strided_slice_42/stackConst*
valueB:��,*
dtype0
Z
*dense_assign_init/strided_slice_42/stack_1Const*
valueB:��.*
dtype0
X
*dense_assign_init/strided_slice_42/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_42StridedSliceps_dense(dense_assign_init/strided_slice_42/stack*dense_assign_init/strided_slice_42/stack_1*dense_assign_init/strided_slice_42/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
W
"dense_assign_init/Reshape_42/shapeConst*
valueB"   �   *
dtype0
�
dense_assign_init/Reshape_42Reshape"dense_assign_init/strided_slice_42"dense_assign_init/Reshape_42/shape*
T0*
Tshape0
�
dense_assign_init/Assign_42Assign reply_xtr/dense_1/kernel/Adagraddense_assign_init/Reshape_42*
validate_shape(*
use_locking(*
T0*+
_class!
loc:@reply_xtr/dense_1/kernel
X
(dense_assign_init/strided_slice_43/stackConst*
valueB:��.*
dtype0
Z
*dense_assign_init/strided_slice_43/stack_1Const*
valueB:��.*
dtype0
X
*dense_assign_init/strided_slice_43/stack_2Const*
dtype0*
valueB:
�
"dense_assign_init/strided_slice_43StridedSliceps_dense(dense_assign_init/strided_slice_43/stack*dense_assign_init/strided_slice_43/stack_1*dense_assign_init/strided_slice_43/stack_2*
end_mask *
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask 
Q
"dense_assign_init/Reshape_43/shapeConst*
dtype0*
valueB:�
�
dense_assign_init/Reshape_43Reshape"dense_assign_init/strided_slice_43"dense_assign_init/Reshape_43/shape*
T0*
Tshape0
�
dense_assign_init/Assign_43Assignreply_xtr/dense_1/bias/Adagraddense_assign_init/Reshape_43*
validate_shape(*
use_locking(*
T0*)
_class
loc:@reply_xtr/dense_1/bias
X
(dense_assign_init/strided_slice_44/stackConst*
valueB:��.*
dtype0
Z
*dense_assign_init/strided_slice_44/stack_1Const*
valueB:��.*
dtype0
X
*dense_assign_init/strided_slice_44/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_44StridedSliceps_dense(dense_assign_init/strided_slice_44/stack*dense_assign_init/strided_slice_44/stack_1*dense_assign_init/strided_slice_44/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask 
W
"dense_assign_init/Reshape_44/shapeConst*
valueB"�   @   *
dtype0
�
dense_assign_init/Reshape_44Reshape"dense_assign_init/strided_slice_44"dense_assign_init/Reshape_44/shape*
T0*
Tshape0
�
dense_assign_init/Assign_44Assign reply_xtr/dense_2/kernel/Adagraddense_assign_init/Reshape_44*
use_locking(*
T0*+
_class!
loc:@reply_xtr/dense_2/kernel*
validate_shape(
X
(dense_assign_init/strided_slice_45/stackConst*
valueB:��.*
dtype0
Z
*dense_assign_init/strided_slice_45/stack_1Const*
valueB:��.*
dtype0
X
*dense_assign_init/strided_slice_45/stack_2Const*
dtype0*
valueB:
�
"dense_assign_init/strided_slice_45StridedSliceps_dense(dense_assign_init/strided_slice_45/stack*dense_assign_init/strided_slice_45/stack_1*dense_assign_init/strided_slice_45/stack_2*
end_mask *
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask 
P
"dense_assign_init/Reshape_45/shapeConst*
valueB:@*
dtype0
�
dense_assign_init/Reshape_45Reshape"dense_assign_init/strided_slice_45"dense_assign_init/Reshape_45/shape*
T0*
Tshape0
�
dense_assign_init/Assign_45Assignreply_xtr/dense_2/bias/Adagraddense_assign_init/Reshape_45*
validate_shape(*
use_locking(*
T0*)
_class
loc:@reply_xtr/dense_2/bias
X
(dense_assign_init/strided_slice_46/stackConst*
valueB:��.*
dtype0
Z
*dense_assign_init/strided_slice_46/stack_1Const*
valueB:��.*
dtype0
X
*dense_assign_init/strided_slice_46/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_46StridedSliceps_dense(dense_assign_init/strided_slice_46/stack*dense_assign_init/strided_slice_46/stack_1*dense_assign_init/strided_slice_46/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask 
W
"dense_assign_init/Reshape_46/shapeConst*
valueB"@      *
dtype0
�
dense_assign_init/Reshape_46Reshape"dense_assign_init/strided_slice_46"dense_assign_init/Reshape_46/shape*
T0*
Tshape0
�
dense_assign_init/Assign_46Assign reply_xtr/dense_3/kernel/Adagraddense_assign_init/Reshape_46*
use_locking(*
T0*+
_class!
loc:@reply_xtr/dense_3/kernel*
validate_shape(
X
(dense_assign_init/strided_slice_47/stackConst*
valueB:��.*
dtype0
Z
*dense_assign_init/strided_slice_47/stack_1Const*
valueB:��.*
dtype0
X
*dense_assign_init/strided_slice_47/stack_2Const*
valueB:*
dtype0
�
"dense_assign_init/strided_slice_47StridedSliceps_dense(dense_assign_init/strided_slice_47/stack*dense_assign_init/strided_slice_47/stack_1*dense_assign_init/strided_slice_47/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
P
"dense_assign_init/Reshape_47/shapeConst*
dtype0*
valueB:
�
dense_assign_init/Reshape_47Reshape"dense_assign_init/strided_slice_47"dense_assign_init/Reshape_47/shape*
T0*
Tshape0
�
dense_assign_init/Assign_47Assignreply_xtr/dense_3/bias/Adagraddense_assign_init/Reshape_47*
validate_shape(*
use_locking(*
T0*)
_class
loc:@reply_xtr/dense_3/bias
�
dense_assignNoOp^dense_assign_init/Assign^dense_assign_init/Assign_1^dense_assign_init/Assign_10^dense_assign_init/Assign_11^dense_assign_init/Assign_12^dense_assign_init/Assign_13^dense_assign_init/Assign_14^dense_assign_init/Assign_15^dense_assign_init/Assign_16^dense_assign_init/Assign_17^dense_assign_init/Assign_18^dense_assign_init/Assign_19^dense_assign_init/Assign_2^dense_assign_init/Assign_20^dense_assign_init/Assign_21^dense_assign_init/Assign_22^dense_assign_init/Assign_23^dense_assign_init/Assign_24^dense_assign_init/Assign_25^dense_assign_init/Assign_26^dense_assign_init/Assign_27^dense_assign_init/Assign_28^dense_assign_init/Assign_29^dense_assign_init/Assign_3^dense_assign_init/Assign_30^dense_assign_init/Assign_31^dense_assign_init/Assign_32^dense_assign_init/Assign_33^dense_assign_init/Assign_34^dense_assign_init/Assign_35^dense_assign_init/Assign_36^dense_assign_init/Assign_37^dense_assign_init/Assign_38^dense_assign_init/Assign_39^dense_assign_init/Assign_4^dense_assign_init/Assign_40^dense_assign_init/Assign_41^dense_assign_init/Assign_42^dense_assign_init/Assign_43^dense_assign_init/Assign_44^dense_assign_init/Assign_45^dense_assign_init/Assign_46^dense_assign_init/Assign_47^dense_assign_init/Assign_5^dense_assign_init/Assign_6^dense_assign_init/Assign_7^dense_assign_init/Assign_8^dense_assign_init/Assign_9
�
dense_init/initNoOp%^expand_xtr/dense/bias/Adagrad/Assign^expand_xtr/dense/bias/Assign'^expand_xtr/dense/kernel/Adagrad/Assign^expand_xtr/dense/kernel/Assign'^expand_xtr/dense_1/bias/Adagrad/Assign^expand_xtr/dense_1/bias/Assign)^expand_xtr/dense_1/kernel/Adagrad/Assign!^expand_xtr/dense_1/kernel/Assign'^expand_xtr/dense_2/bias/Adagrad/Assign^expand_xtr/dense_2/bias/Assign)^expand_xtr/dense_2/kernel/Adagrad/Assign!^expand_xtr/dense_2/kernel/Assign'^expand_xtr/dense_3/bias/Adagrad/Assign^expand_xtr/dense_3/bias/Assign)^expand_xtr/dense_3/kernel/Adagrad/Assign!^expand_xtr/dense_3/kernel/Assign#^like_xtr/dense/bias/Adagrad/Assign^like_xtr/dense/bias/Assign%^like_xtr/dense/kernel/Adagrad/Assign^like_xtr/dense/kernel/Assign%^like_xtr/dense_1/bias/Adagrad/Assign^like_xtr/dense_1/bias/Assign'^like_xtr/dense_1/kernel/Adagrad/Assign^like_xtr/dense_1/kernel/Assign%^like_xtr/dense_2/bias/Adagrad/Assign^like_xtr/dense_2/bias/Assign'^like_xtr/dense_2/kernel/Adagrad/Assign^like_xtr/dense_2/kernel/Assign%^like_xtr/dense_3/bias/Adagrad/Assign^like_xtr/dense_3/bias/Assign'^like_xtr/dense_3/kernel/Adagrad/Assign^like_xtr/dense_3/kernel/Assign$^reply_xtr/dense/bias/Adagrad/Assign^reply_xtr/dense/bias/Assign&^reply_xtr/dense/kernel/Adagrad/Assign^reply_xtr/dense/kernel/Assign&^reply_xtr/dense_1/bias/Adagrad/Assign^reply_xtr/dense_1/bias/Assign(^reply_xtr/dense_1/kernel/Adagrad/Assign ^reply_xtr/dense_1/kernel/Assign&^reply_xtr/dense_2/bias/Adagrad/Assign^reply_xtr/dense_2/bias/Assign(^reply_xtr/dense_2/kernel/Adagrad/Assign ^reply_xtr/dense_2/kernel/Assign&^reply_xtr/dense_3/bias/Adagrad/Assign^reply_xtr/dense_3/bias/Assign(^reply_xtr/dense_3/kernel/Adagrad/Assign ^reply_xtr/dense_3/kernel/Assign
O
dense_init/readIdentityexpand_xtr/dense/kernel^dense_init/init*
T0
a
dense_init/Reshape/shapeConst^dense_init/init*
valueB:
���������*
dtype0
_
dense_init/ReshapeReshapedense_init/readdense_init/Reshape/shape*
T0*
Tshape0
O
dense_init/read_1Identityexpand_xtr/dense/bias^dense_init/init*
T0
c
dense_init/Reshape_1/shapeConst^dense_init/init*
valueB:
���������*
dtype0
e
dense_init/Reshape_1Reshapedense_init/read_1dense_init/Reshape_1/shape*
T0*
Tshape0
S
dense_init/read_2Identityexpand_xtr/dense_1/kernel^dense_init/init*
T0
c
dense_init/Reshape_2/shapeConst^dense_init/init*
valueB:
���������*
dtype0
e
dense_init/Reshape_2Reshapedense_init/read_2dense_init/Reshape_2/shape*
T0*
Tshape0
Q
dense_init/read_3Identityexpand_xtr/dense_1/bias^dense_init/init*
T0
c
dense_init/Reshape_3/shapeConst^dense_init/init*
valueB:
���������*
dtype0
e
dense_init/Reshape_3Reshapedense_init/read_3dense_init/Reshape_3/shape*
T0*
Tshape0
S
dense_init/read_4Identityexpand_xtr/dense_2/kernel^dense_init/init*
T0
c
dense_init/Reshape_4/shapeConst^dense_init/init*
valueB:
���������*
dtype0
e
dense_init/Reshape_4Reshapedense_init/read_4dense_init/Reshape_4/shape*
T0*
Tshape0
Q
dense_init/read_5Identityexpand_xtr/dense_2/bias^dense_init/init*
T0
c
dense_init/Reshape_5/shapeConst^dense_init/init*
valueB:
���������*
dtype0
e
dense_init/Reshape_5Reshapedense_init/read_5dense_init/Reshape_5/shape*
T0*
Tshape0
S
dense_init/read_6Identityexpand_xtr/dense_3/kernel^dense_init/init*
T0
c
dense_init/Reshape_6/shapeConst^dense_init/init*
valueB:
���������*
dtype0
e
dense_init/Reshape_6Reshapedense_init/read_6dense_init/Reshape_6/shape*
T0*
Tshape0
Q
dense_init/read_7Identityexpand_xtr/dense_3/bias^dense_init/init*
T0
c
dense_init/Reshape_7/shapeConst^dense_init/init*
valueB:
���������*
dtype0
e
dense_init/Reshape_7Reshapedense_init/read_7dense_init/Reshape_7/shape*
T0*
Tshape0
O
dense_init/read_8Identitylike_xtr/dense/kernel^dense_init/init*
T0
c
dense_init/Reshape_8/shapeConst^dense_init/init*
valueB:
���������*
dtype0
e
dense_init/Reshape_8Reshapedense_init/read_8dense_init/Reshape_8/shape*
T0*
Tshape0
M
dense_init/read_9Identitylike_xtr/dense/bias^dense_init/init*
T0
c
dense_init/Reshape_9/shapeConst^dense_init/init*
dtype0*
valueB:
���������
e
dense_init/Reshape_9Reshapedense_init/read_9dense_init/Reshape_9/shape*
T0*
Tshape0
R
dense_init/read_10Identitylike_xtr/dense_1/kernel^dense_init/init*
T0
d
dense_init/Reshape_10/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_10Reshapedense_init/read_10dense_init/Reshape_10/shape*
T0*
Tshape0
P
dense_init/read_11Identitylike_xtr/dense_1/bias^dense_init/init*
T0
d
dense_init/Reshape_11/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_11Reshapedense_init/read_11dense_init/Reshape_11/shape*
T0*
Tshape0
R
dense_init/read_12Identitylike_xtr/dense_2/kernel^dense_init/init*
T0
d
dense_init/Reshape_12/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_12Reshapedense_init/read_12dense_init/Reshape_12/shape*
T0*
Tshape0
P
dense_init/read_13Identitylike_xtr/dense_2/bias^dense_init/init*
T0
d
dense_init/Reshape_13/shapeConst^dense_init/init*
dtype0*
valueB:
���������
h
dense_init/Reshape_13Reshapedense_init/read_13dense_init/Reshape_13/shape*
T0*
Tshape0
R
dense_init/read_14Identitylike_xtr/dense_3/kernel^dense_init/init*
T0
d
dense_init/Reshape_14/shapeConst^dense_init/init*
dtype0*
valueB:
���������
h
dense_init/Reshape_14Reshapedense_init/read_14dense_init/Reshape_14/shape*
T0*
Tshape0
P
dense_init/read_15Identitylike_xtr/dense_3/bias^dense_init/init*
T0
d
dense_init/Reshape_15/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_15Reshapedense_init/read_15dense_init/Reshape_15/shape*
T0*
Tshape0
Q
dense_init/read_16Identityreply_xtr/dense/kernel^dense_init/init*
T0
d
dense_init/Reshape_16/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_16Reshapedense_init/read_16dense_init/Reshape_16/shape*
T0*
Tshape0
O
dense_init/read_17Identityreply_xtr/dense/bias^dense_init/init*
T0
d
dense_init/Reshape_17/shapeConst^dense_init/init*
dtype0*
valueB:
���������
h
dense_init/Reshape_17Reshapedense_init/read_17dense_init/Reshape_17/shape*
T0*
Tshape0
S
dense_init/read_18Identityreply_xtr/dense_1/kernel^dense_init/init*
T0
d
dense_init/Reshape_18/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_18Reshapedense_init/read_18dense_init/Reshape_18/shape*
T0*
Tshape0
Q
dense_init/read_19Identityreply_xtr/dense_1/bias^dense_init/init*
T0
d
dense_init/Reshape_19/shapeConst^dense_init/init*
dtype0*
valueB:
���������
h
dense_init/Reshape_19Reshapedense_init/read_19dense_init/Reshape_19/shape*
T0*
Tshape0
S
dense_init/read_20Identityreply_xtr/dense_2/kernel^dense_init/init*
T0
d
dense_init/Reshape_20/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_20Reshapedense_init/read_20dense_init/Reshape_20/shape*
T0*
Tshape0
Q
dense_init/read_21Identityreply_xtr/dense_2/bias^dense_init/init*
T0
d
dense_init/Reshape_21/shapeConst^dense_init/init*
dtype0*
valueB:
���������
h
dense_init/Reshape_21Reshapedense_init/read_21dense_init/Reshape_21/shape*
T0*
Tshape0
S
dense_init/read_22Identityreply_xtr/dense_3/kernel^dense_init/init*
T0
d
dense_init/Reshape_22/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_22Reshapedense_init/read_22dense_init/Reshape_22/shape*
T0*
Tshape0
Q
dense_init/read_23Identityreply_xtr/dense_3/bias^dense_init/init*
T0
d
dense_init/Reshape_23/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_23Reshapedense_init/read_23dense_init/Reshape_23/shape*
T0*
Tshape0
Z
dense_init/read_24Identityexpand_xtr/dense/kernel/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_24/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_24Reshapedense_init/read_24dense_init/Reshape_24/shape*
T0*
Tshape0
X
dense_init/read_25Identityexpand_xtr/dense/bias/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_25/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_25Reshapedense_init/read_25dense_init/Reshape_25/shape*
T0*
Tshape0
\
dense_init/read_26Identity!expand_xtr/dense_1/kernel/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_26/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_26Reshapedense_init/read_26dense_init/Reshape_26/shape*
T0*
Tshape0
Z
dense_init/read_27Identityexpand_xtr/dense_1/bias/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_27/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_27Reshapedense_init/read_27dense_init/Reshape_27/shape*
T0*
Tshape0
\
dense_init/read_28Identity!expand_xtr/dense_2/kernel/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_28/shapeConst^dense_init/init*
dtype0*
valueB:
���������
h
dense_init/Reshape_28Reshapedense_init/read_28dense_init/Reshape_28/shape*
T0*
Tshape0
Z
dense_init/read_29Identityexpand_xtr/dense_2/bias/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_29/shapeConst^dense_init/init*
dtype0*
valueB:
���������
h
dense_init/Reshape_29Reshapedense_init/read_29dense_init/Reshape_29/shape*
T0*
Tshape0
\
dense_init/read_30Identity!expand_xtr/dense_3/kernel/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_30/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_30Reshapedense_init/read_30dense_init/Reshape_30/shape*
T0*
Tshape0
Z
dense_init/read_31Identityexpand_xtr/dense_3/bias/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_31/shapeConst^dense_init/init*
dtype0*
valueB:
���������
h
dense_init/Reshape_31Reshapedense_init/read_31dense_init/Reshape_31/shape*
T0*
Tshape0
X
dense_init/read_32Identitylike_xtr/dense/kernel/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_32/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_32Reshapedense_init/read_32dense_init/Reshape_32/shape*
T0*
Tshape0
V
dense_init/read_33Identitylike_xtr/dense/bias/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_33/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_33Reshapedense_init/read_33dense_init/Reshape_33/shape*
T0*
Tshape0
Z
dense_init/read_34Identitylike_xtr/dense_1/kernel/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_34/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_34Reshapedense_init/read_34dense_init/Reshape_34/shape*
T0*
Tshape0
X
dense_init/read_35Identitylike_xtr/dense_1/bias/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_35/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_35Reshapedense_init/read_35dense_init/Reshape_35/shape*
T0*
Tshape0
Z
dense_init/read_36Identitylike_xtr/dense_2/kernel/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_36/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_36Reshapedense_init/read_36dense_init/Reshape_36/shape*
T0*
Tshape0
X
dense_init/read_37Identitylike_xtr/dense_2/bias/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_37/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_37Reshapedense_init/read_37dense_init/Reshape_37/shape*
T0*
Tshape0
Z
dense_init/read_38Identitylike_xtr/dense_3/kernel/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_38/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_38Reshapedense_init/read_38dense_init/Reshape_38/shape*
T0*
Tshape0
X
dense_init/read_39Identitylike_xtr/dense_3/bias/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_39/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_39Reshapedense_init/read_39dense_init/Reshape_39/shape*
T0*
Tshape0
Y
dense_init/read_40Identityreply_xtr/dense/kernel/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_40/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_40Reshapedense_init/read_40dense_init/Reshape_40/shape*
T0*
Tshape0
W
dense_init/read_41Identityreply_xtr/dense/bias/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_41/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_41Reshapedense_init/read_41dense_init/Reshape_41/shape*
T0*
Tshape0
[
dense_init/read_42Identity reply_xtr/dense_1/kernel/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_42/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_42Reshapedense_init/read_42dense_init/Reshape_42/shape*
T0*
Tshape0
Y
dense_init/read_43Identityreply_xtr/dense_1/bias/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_43/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_43Reshapedense_init/read_43dense_init/Reshape_43/shape*
T0*
Tshape0
[
dense_init/read_44Identity reply_xtr/dense_2/kernel/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_44/shapeConst^dense_init/init*
dtype0*
valueB:
���������
h
dense_init/Reshape_44Reshapedense_init/read_44dense_init/Reshape_44/shape*
T0*
Tshape0
Y
dense_init/read_45Identityreply_xtr/dense_2/bias/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_45/shapeConst^dense_init/init*
dtype0*
valueB:
���������
h
dense_init/Reshape_45Reshapedense_init/read_45dense_init/Reshape_45/shape*
T0*
Tshape0
[
dense_init/read_46Identity reply_xtr/dense_3/kernel/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_46/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_46Reshapedense_init/read_46dense_init/Reshape_46/shape*
T0*
Tshape0
Y
dense_init/read_47Identityreply_xtr/dense_3/bias/Adagrad^dense_init/init*
T0
d
dense_init/Reshape_47/shapeConst^dense_init/init*
valueB:
���������*
dtype0
h
dense_init/Reshape_47Reshapedense_init/read_47dense_init/Reshape_47/shape*
T0*
Tshape0
R
dense_init/concat/axisConst^dense_init/init*
value	B : *
dtype0
�	
dense_init/concatConcatV2dense_init/Reshapedense_init/Reshape_1dense_init/Reshape_2dense_init/Reshape_3dense_init/Reshape_4dense_init/Reshape_5dense_init/Reshape_6dense_init/Reshape_7dense_init/Reshape_8dense_init/Reshape_9dense_init/Reshape_10dense_init/Reshape_11dense_init/Reshape_12dense_init/Reshape_13dense_init/Reshape_14dense_init/Reshape_15dense_init/Reshape_16dense_init/Reshape_17dense_init/Reshape_18dense_init/Reshape_19dense_init/Reshape_20dense_init/Reshape_21dense_init/Reshape_22dense_init/Reshape_23dense_init/Reshape_24dense_init/Reshape_25dense_init/Reshape_26dense_init/Reshape_27dense_init/Reshape_28dense_init/Reshape_29dense_init/Reshape_30dense_init/Reshape_31dense_init/Reshape_32dense_init/Reshape_33dense_init/Reshape_34dense_init/Reshape_35dense_init/Reshape_36dense_init/Reshape_37dense_init/Reshape_38dense_init/Reshape_39dense_init/Reshape_40dense_init/Reshape_41dense_init/Reshape_42dense_init/Reshape_43dense_init/Reshape_44dense_init/Reshape_45dense_init/Reshape_46dense_init/Reshape_47dense_init/concat/axis*

Tidx0*
T0*
N0
=
grad_scale/inputConst*
valueB
 *
ף5*
dtype0
P

grad_scalePlaceholderWithDefaultgrad_scale/input*
shape: *
dtype0
Q
sparse_grad_scalePlaceholderWithDefault
grad_scale*
dtype0*
shape: 
=
loss_scale/inputConst*
valueB
 *  �?*
dtype0
P

loss_scalePlaceholderWithDefaultloss_scale/input*
dtype0*
shape: 
3
truedivRealDiv
grad_scale
loss_scale*
T0
<
	truediv_1RealDivsparse_grad_scale
loss_scale*
T0
-
mulMul
loss_scalelog_loss/Sum*
T0
:
gradients_1/ShapeConst*
dtype0*
valueB 
B
gradients_1/grad_ys_0Const*
valueB
 *  �?*
dtype0
]
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0
H
gradients_1/mul_grad/MulMulgradients_1/Filllog_loss/Sum*
T0
H
gradients_1/mul_grad/Mul_1Mulgradients_1/Fill
loss_scale*
T0
d
+gradients_1/log_loss/Sum_grad/Reshape/shapeConst*!
valueB"         *
dtype0
�
%gradients_1/log_loss/Sum_grad/ReshapeReshapegradients_1/mul_grad/Mul_1+gradients_1/log_loss/Sum_grad/Reshape/shape*
T0*
Tshape0
U
#gradients_1/log_loss/Sum_grad/ShapeShapelog_loss/Mul_2*
T0*
out_type0
�
"gradients_1/log_loss/Sum_grad/TileTile%gradients_1/log_loss/Sum_grad/Reshape#gradients_1/log_loss/Sum_grad/Shape*

Tmultiples0*
T0
W
%gradients_1/log_loss/Mul_2_grad/ShapeShapelog_loss/sub_2*
T0*
out_type0
_
'gradients_1/log_loss/Mul_2_grad/Shape_1Shapelog_loss/ToFloat_2/x*
T0*
out_type0
�
5gradients_1/log_loss/Mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients_1/log_loss/Mul_2_grad/Shape'gradients_1/log_loss/Mul_2_grad/Shape_1*
T0
m
#gradients_1/log_loss/Mul_2_grad/MulMul"gradients_1/log_loss/Sum_grad/Tilelog_loss/ToFloat_2/x*
T0
�
#gradients_1/log_loss/Mul_2_grad/SumSum#gradients_1/log_loss/Mul_2_grad/Mul5gradients_1/log_loss/Mul_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
'gradients_1/log_loss/Mul_2_grad/ReshapeReshape#gradients_1/log_loss/Mul_2_grad/Sum%gradients_1/log_loss/Mul_2_grad/Shape*
T0*
Tshape0
i
%gradients_1/log_loss/Mul_2_grad/Mul_1Mullog_loss/sub_2"gradients_1/log_loss/Sum_grad/Tile*
T0
�
%gradients_1/log_loss/Mul_2_grad/Sum_1Sum%gradients_1/log_loss/Mul_2_grad/Mul_17gradients_1/log_loss/Mul_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
)gradients_1/log_loss/Mul_2_grad/Reshape_1Reshape%gradients_1/log_loss/Mul_2_grad/Sum_1'gradients_1/log_loss/Mul_2_grad/Shape_1*
T0*
Tshape0
U
%gradients_1/log_loss/sub_2_grad/ShapeShapelog_loss/Neg*
T0*
out_type0
Y
'gradients_1/log_loss/sub_2_grad/Shape_1Shapelog_loss/Mul_1*
T0*
out_type0
�
5gradients_1/log_loss/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients_1/log_loss/sub_2_grad/Shape'gradients_1/log_loss/sub_2_grad/Shape_1*
T0
�
#gradients_1/log_loss/sub_2_grad/SumSum'gradients_1/log_loss/Mul_2_grad/Reshape5gradients_1/log_loss/sub_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
'gradients_1/log_loss/sub_2_grad/ReshapeReshape#gradients_1/log_loss/sub_2_grad/Sum%gradients_1/log_loss/sub_2_grad/Shape*
T0*
Tshape0
�
%gradients_1/log_loss/sub_2_grad/Sum_1Sum'gradients_1/log_loss/Mul_2_grad/Reshape7gradients_1/log_loss/sub_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Z
#gradients_1/log_loss/sub_2_grad/NegNeg%gradients_1/log_loss/sub_2_grad/Sum_1*
T0
�
)gradients_1/log_loss/sub_2_grad/Reshape_1Reshape#gradients_1/log_loss/sub_2_grad/Neg'gradients_1/log_loss/sub_2_grad/Shape_1*
T0*
Tshape0
Z
!gradients_1/log_loss/Neg_grad/NegNeg'gradients_1/log_loss/sub_2_grad/Reshape*
T0
U
%gradients_1/log_loss/Mul_1_grad/ShapeShapelog_loss/sub*
T0*
out_type0
Y
'gradients_1/log_loss/Mul_1_grad/Shape_1Shapelog_loss/Log_1*
T0*
out_type0
�
5gradients_1/log_loss/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients_1/log_loss/Mul_1_grad/Shape'gradients_1/log_loss/Mul_1_grad/Shape_1*
T0
n
#gradients_1/log_loss/Mul_1_grad/MulMul)gradients_1/log_loss/sub_2_grad/Reshape_1log_loss/Log_1*
T0
�
#gradients_1/log_loss/Mul_1_grad/SumSum#gradients_1/log_loss/Mul_1_grad/Mul5gradients_1/log_loss/Mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
'gradients_1/log_loss/Mul_1_grad/ReshapeReshape#gradients_1/log_loss/Mul_1_grad/Sum%gradients_1/log_loss/Mul_1_grad/Shape*
T0*
Tshape0
n
%gradients_1/log_loss/Mul_1_grad/Mul_1Mullog_loss/sub)gradients_1/log_loss/sub_2_grad/Reshape_1*
T0
�
%gradients_1/log_loss/Mul_1_grad/Sum_1Sum%gradients_1/log_loss/Mul_1_grad/Mul_17gradients_1/log_loss/Mul_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
)gradients_1/log_loss/Mul_1_grad/Reshape_1Reshape%gradients_1/log_loss/Mul_1_grad/Sum_1'gradients_1/log_loss/Mul_1_grad/Shape_1*
T0*
Tshape0
[
#gradients_1/log_loss/Mul_grad/ShapeShapelog_loss/ToFloat_1/x*
T0*
out_type0
U
%gradients_1/log_loss/Mul_grad/Shape_1Shapelog_loss/Log*
T0*
out_type0
�
3gradients_1/log_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients_1/log_loss/Mul_grad/Shape%gradients_1/log_loss/Mul_grad/Shape_1*
T0
b
!gradients_1/log_loss/Mul_grad/MulMul!gradients_1/log_loss/Neg_grad/Neglog_loss/Log*
T0
�
!gradients_1/log_loss/Mul_grad/SumSum!gradients_1/log_loss/Mul_grad/Mul3gradients_1/log_loss/Mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
%gradients_1/log_loss/Mul_grad/ReshapeReshape!gradients_1/log_loss/Mul_grad/Sum#gradients_1/log_loss/Mul_grad/Shape*
T0*
Tshape0
l
#gradients_1/log_loss/Mul_grad/Mul_1Mullog_loss/ToFloat_1/x!gradients_1/log_loss/Neg_grad/Neg*
T0
�
#gradients_1/log_loss/Mul_grad/Sum_1Sum#gradients_1/log_loss/Mul_grad/Mul_15gradients_1/log_loss/Mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
'gradients_1/log_loss/Mul_grad/Reshape_1Reshape#gradients_1/log_loss/Mul_grad/Sum_1%gradients_1/log_loss/Mul_grad/Shape_1*
T0*
Tshape0
}
*gradients_1/log_loss/Log_1_grad/Reciprocal
Reciprocallog_loss/add_1*^gradients_1/log_loss/Mul_1_grad/Reshape_1*
T0
�
#gradients_1/log_loss/Log_1_grad/mulMul)gradients_1/log_loss/Mul_1_grad/Reshape_1*gradients_1/log_loss/Log_1_grad/Reciprocal*
T0
w
(gradients_1/log_loss/Log_grad/Reciprocal
Reciprocallog_loss/add(^gradients_1/log_loss/Mul_grad/Reshape_1*
T0
�
!gradients_1/log_loss/Log_grad/mulMul'gradients_1/log_loss/Mul_grad/Reshape_1(gradients_1/log_loss/Log_grad/Reciprocal*
T0
W
%gradients_1/log_loss/add_1_grad/ShapeShapelog_loss/sub_1*
T0*
out_type0
P
'gradients_1/log_loss/add_1_grad/Shape_1Const*
valueB *
dtype0
�
5gradients_1/log_loss/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients_1/log_loss/add_1_grad/Shape'gradients_1/log_loss/add_1_grad/Shape_1*
T0
�
#gradients_1/log_loss/add_1_grad/SumSum#gradients_1/log_loss/Log_1_grad/mul5gradients_1/log_loss/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
'gradients_1/log_loss/add_1_grad/ReshapeReshape#gradients_1/log_loss/add_1_grad/Sum%gradients_1/log_loss/add_1_grad/Shape*
T0*
Tshape0
�
%gradients_1/log_loss/add_1_grad/Sum_1Sum#gradients_1/log_loss/Log_1_grad/mul7gradients_1/log_loss/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
)gradients_1/log_loss/add_1_grad/Reshape_1Reshape%gradients_1/log_loss/add_1_grad/Sum_1'gradients_1/log_loss/add_1_grad/Shape_1*
T0*
Tshape0
Y
#gradients_1/log_loss/add_grad/ShapeShapelog_loss/ToFloat/x*
T0*
out_type0
N
%gradients_1/log_loss/add_grad/Shape_1Const*
valueB *
dtype0
�
3gradients_1/log_loss/add_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients_1/log_loss/add_grad/Shape%gradients_1/log_loss/add_grad/Shape_1*
T0
�
!gradients_1/log_loss/add_grad/SumSum!gradients_1/log_loss/Log_grad/mul3gradients_1/log_loss/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
%gradients_1/log_loss/add_grad/ReshapeReshape!gradients_1/log_loss/add_grad/Sum#gradients_1/log_loss/add_grad/Shape*
T0*
Tshape0
�
#gradients_1/log_loss/add_grad/Sum_1Sum!gradients_1/log_loss/Log_grad/mul5gradients_1/log_loss/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
'gradients_1/log_loss/add_grad/Reshape_1Reshape#gradients_1/log_loss/add_grad/Sum_1%gradients_1/log_loss/add_grad/Shape_1*
T0*
Tshape0
N
%gradients_1/log_loss/sub_1_grad/ShapeConst*
dtype0*
valueB 
]
'gradients_1/log_loss/sub_1_grad/Shape_1Shapelog_loss/ToFloat/x*
T0*
out_type0
�
5gradients_1/log_loss/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients_1/log_loss/sub_1_grad/Shape'gradients_1/log_loss/sub_1_grad/Shape_1*
T0
�
#gradients_1/log_loss/sub_1_grad/SumSum'gradients_1/log_loss/add_1_grad/Reshape5gradients_1/log_loss/sub_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
'gradients_1/log_loss/sub_1_grad/ReshapeReshape#gradients_1/log_loss/sub_1_grad/Sum%gradients_1/log_loss/sub_1_grad/Shape*
T0*
Tshape0
�
%gradients_1/log_loss/sub_1_grad/Sum_1Sum'gradients_1/log_loss/add_1_grad/Reshape7gradients_1/log_loss/sub_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
Z
#gradients_1/log_loss/sub_1_grad/NegNeg%gradients_1/log_loss/sub_1_grad/Sum_1*
T0
�
)gradients_1/log_loss/sub_1_grad/Reshape_1Reshape#gradients_1/log_loss/sub_1_grad/Neg'gradients_1/log_loss/sub_1_grad/Shape_1*
T0*
Tshape0
�
gradients_1/AddNAddN%gradients_1/log_loss/add_grad/Reshape)gradients_1/log_loss/sub_1_grad/Reshape_1*
T0*8
_class.
,*loc:@gradients_1/log_loss/add_grad/Reshape*
N
g
+gradients_1/log_loss/ToFloat/x_grad/unstackUnpackgradients_1/AddN*
T0*	
num*

axis 
�
7gradients_1/expand_xtr/dense_3/Sigmoid_grad/SigmoidGradSigmoidGradexpand_xtr/dense_3/Sigmoid+gradients_1/log_loss/ToFloat/x_grad/unstack*
T0
�
5gradients_1/like_xtr/dense_3/Sigmoid_grad/SigmoidGradSigmoidGradlike_xtr/dense_3/Sigmoid-gradients_1/log_loss/ToFloat/x_grad/unstack:1*
T0
�
6gradients_1/reply_xtr/dense_3/Sigmoid_grad/SigmoidGradSigmoidGradreply_xtr/dense_3/Sigmoid-gradients_1/log_loss/ToFloat/x_grad/unstack:2*
T0
�
7gradients_1/expand_xtr/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients_1/expand_xtr/dense_3/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
T0
�
5gradients_1/like_xtr/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients_1/like_xtr/dense_3/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC
�
6gradients_1/reply_xtr/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients_1/reply_xtr/dense_3/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC
�
1gradients_1/expand_xtr/dense_3/MatMul_grad/MatMulMatMul7gradients_1/expand_xtr/dense_3/Sigmoid_grad/SigmoidGradexpand_xtr/dense_3/kernel/read*
T0*
transpose_a( *
transpose_b(
�
3gradients_1/expand_xtr/dense_3/MatMul_grad/MatMul_1MatMulexpand_xtr/dense_2/LeakyRelu7gradients_1/expand_xtr/dense_3/Sigmoid_grad/SigmoidGrad*
transpose_a(*
transpose_b( *
T0
�
/gradients_1/like_xtr/dense_3/MatMul_grad/MatMulMatMul5gradients_1/like_xtr/dense_3/Sigmoid_grad/SigmoidGradlike_xtr/dense_3/kernel/read*
T0*
transpose_a( *
transpose_b(
�
1gradients_1/like_xtr/dense_3/MatMul_grad/MatMul_1MatMullike_xtr/dense_2/LeakyRelu5gradients_1/like_xtr/dense_3/Sigmoid_grad/SigmoidGrad*
T0*
transpose_a(*
transpose_b( 
�
0gradients_1/reply_xtr/dense_3/MatMul_grad/MatMulMatMul6gradients_1/reply_xtr/dense_3/Sigmoid_grad/SigmoidGradreply_xtr/dense_3/kernel/read*
transpose_b(*
T0*
transpose_a( 
�
2gradients_1/reply_xtr/dense_3/MatMul_grad/MatMul_1MatMulreply_xtr/dense_2/LeakyRelu6gradients_1/reply_xtr/dense_3/Sigmoid_grad/SigmoidGrad*
T0*
transpose_a(*
transpose_b( 
w
3gradients_1/expand_xtr/dense_2/LeakyRelu_grad/ShapeShape expand_xtr/dense_2/LeakyRelu/mul*
T0*
out_type0
s
5gradients_1/expand_xtr/dense_2/LeakyRelu_grad/Shape_1Shapeexpand_xtr/dense_2/BiasAdd*
T0*
out_type0
�
5gradients_1/expand_xtr/dense_2/LeakyRelu_grad/Shape_2Shape1gradients_1/expand_xtr/dense_3/MatMul_grad/MatMul*
T0*
out_type0
f
9gradients_1/expand_xtr/dense_2/LeakyRelu_grad/zeros/ConstConst*
dtype0*
valueB
 *    
�
3gradients_1/expand_xtr/dense_2/LeakyRelu_grad/zerosFill5gradients_1/expand_xtr/dense_2/LeakyRelu_grad/Shape_29gradients_1/expand_xtr/dense_2/LeakyRelu_grad/zeros/Const*
T0*

index_type0
�
:gradients_1/expand_xtr/dense_2/LeakyRelu_grad/GreaterEqualGreaterEqual expand_xtr/dense_2/LeakyRelu/mulexpand_xtr/dense_2/BiasAdd*
T0
�
Cgradients_1/expand_xtr/dense_2/LeakyRelu_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients_1/expand_xtr/dense_2/LeakyRelu_grad/Shape5gradients_1/expand_xtr/dense_2/LeakyRelu_grad/Shape_1*
T0
�
4gradients_1/expand_xtr/dense_2/LeakyRelu_grad/SelectSelect:gradients_1/expand_xtr/dense_2/LeakyRelu_grad/GreaterEqual1gradients_1/expand_xtr/dense_3/MatMul_grad/MatMul3gradients_1/expand_xtr/dense_2/LeakyRelu_grad/zeros*
T0
�
6gradients_1/expand_xtr/dense_2/LeakyRelu_grad/Select_1Select:gradients_1/expand_xtr/dense_2/LeakyRelu_grad/GreaterEqual3gradients_1/expand_xtr/dense_2/LeakyRelu_grad/zeros1gradients_1/expand_xtr/dense_3/MatMul_grad/MatMul*
T0
�
1gradients_1/expand_xtr/dense_2/LeakyRelu_grad/SumSum4gradients_1/expand_xtr/dense_2/LeakyRelu_grad/SelectCgradients_1/expand_xtr/dense_2/LeakyRelu_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
5gradients_1/expand_xtr/dense_2/LeakyRelu_grad/ReshapeReshape1gradients_1/expand_xtr/dense_2/LeakyRelu_grad/Sum3gradients_1/expand_xtr/dense_2/LeakyRelu_grad/Shape*
T0*
Tshape0
�
3gradients_1/expand_xtr/dense_2/LeakyRelu_grad/Sum_1Sum6gradients_1/expand_xtr/dense_2/LeakyRelu_grad/Select_1Egradients_1/expand_xtr/dense_2/LeakyRelu_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
7gradients_1/expand_xtr/dense_2/LeakyRelu_grad/Reshape_1Reshape3gradients_1/expand_xtr/dense_2/LeakyRelu_grad/Sum_15gradients_1/expand_xtr/dense_2/LeakyRelu_grad/Shape_1*
T0*
Tshape0
s
1gradients_1/like_xtr/dense_2/LeakyRelu_grad/ShapeShapelike_xtr/dense_2/LeakyRelu/mul*
T0*
out_type0
o
3gradients_1/like_xtr/dense_2/LeakyRelu_grad/Shape_1Shapelike_xtr/dense_2/BiasAdd*
T0*
out_type0
�
3gradients_1/like_xtr/dense_2/LeakyRelu_grad/Shape_2Shape/gradients_1/like_xtr/dense_3/MatMul_grad/MatMul*
T0*
out_type0
d
7gradients_1/like_xtr/dense_2/LeakyRelu_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
1gradients_1/like_xtr/dense_2/LeakyRelu_grad/zerosFill3gradients_1/like_xtr/dense_2/LeakyRelu_grad/Shape_27gradients_1/like_xtr/dense_2/LeakyRelu_grad/zeros/Const*
T0*

index_type0
�
8gradients_1/like_xtr/dense_2/LeakyRelu_grad/GreaterEqualGreaterEquallike_xtr/dense_2/LeakyRelu/mullike_xtr/dense_2/BiasAdd*
T0
�
Agradients_1/like_xtr/dense_2/LeakyRelu_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients_1/like_xtr/dense_2/LeakyRelu_grad/Shape3gradients_1/like_xtr/dense_2/LeakyRelu_grad/Shape_1*
T0
�
2gradients_1/like_xtr/dense_2/LeakyRelu_grad/SelectSelect8gradients_1/like_xtr/dense_2/LeakyRelu_grad/GreaterEqual/gradients_1/like_xtr/dense_3/MatMul_grad/MatMul1gradients_1/like_xtr/dense_2/LeakyRelu_grad/zeros*
T0
�
4gradients_1/like_xtr/dense_2/LeakyRelu_grad/Select_1Select8gradients_1/like_xtr/dense_2/LeakyRelu_grad/GreaterEqual1gradients_1/like_xtr/dense_2/LeakyRelu_grad/zeros/gradients_1/like_xtr/dense_3/MatMul_grad/MatMul*
T0
�
/gradients_1/like_xtr/dense_2/LeakyRelu_grad/SumSum2gradients_1/like_xtr/dense_2/LeakyRelu_grad/SelectAgradients_1/like_xtr/dense_2/LeakyRelu_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
3gradients_1/like_xtr/dense_2/LeakyRelu_grad/ReshapeReshape/gradients_1/like_xtr/dense_2/LeakyRelu_grad/Sum1gradients_1/like_xtr/dense_2/LeakyRelu_grad/Shape*
T0*
Tshape0
�
1gradients_1/like_xtr/dense_2/LeakyRelu_grad/Sum_1Sum4gradients_1/like_xtr/dense_2/LeakyRelu_grad/Select_1Cgradients_1/like_xtr/dense_2/LeakyRelu_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
5gradients_1/like_xtr/dense_2/LeakyRelu_grad/Reshape_1Reshape1gradients_1/like_xtr/dense_2/LeakyRelu_grad/Sum_13gradients_1/like_xtr/dense_2/LeakyRelu_grad/Shape_1*
T0*
Tshape0
u
2gradients_1/reply_xtr/dense_2/LeakyRelu_grad/ShapeShapereply_xtr/dense_2/LeakyRelu/mul*
T0*
out_type0
q
4gradients_1/reply_xtr/dense_2/LeakyRelu_grad/Shape_1Shapereply_xtr/dense_2/BiasAdd*
T0*
out_type0
�
4gradients_1/reply_xtr/dense_2/LeakyRelu_grad/Shape_2Shape0gradients_1/reply_xtr/dense_3/MatMul_grad/MatMul*
T0*
out_type0
e
8gradients_1/reply_xtr/dense_2/LeakyRelu_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
2gradients_1/reply_xtr/dense_2/LeakyRelu_grad/zerosFill4gradients_1/reply_xtr/dense_2/LeakyRelu_grad/Shape_28gradients_1/reply_xtr/dense_2/LeakyRelu_grad/zeros/Const*
T0*

index_type0
�
9gradients_1/reply_xtr/dense_2/LeakyRelu_grad/GreaterEqualGreaterEqualreply_xtr/dense_2/LeakyRelu/mulreply_xtr/dense_2/BiasAdd*
T0
�
Bgradients_1/reply_xtr/dense_2/LeakyRelu_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients_1/reply_xtr/dense_2/LeakyRelu_grad/Shape4gradients_1/reply_xtr/dense_2/LeakyRelu_grad/Shape_1*
T0
�
3gradients_1/reply_xtr/dense_2/LeakyRelu_grad/SelectSelect9gradients_1/reply_xtr/dense_2/LeakyRelu_grad/GreaterEqual0gradients_1/reply_xtr/dense_3/MatMul_grad/MatMul2gradients_1/reply_xtr/dense_2/LeakyRelu_grad/zeros*
T0
�
5gradients_1/reply_xtr/dense_2/LeakyRelu_grad/Select_1Select9gradients_1/reply_xtr/dense_2/LeakyRelu_grad/GreaterEqual2gradients_1/reply_xtr/dense_2/LeakyRelu_grad/zeros0gradients_1/reply_xtr/dense_3/MatMul_grad/MatMul*
T0
�
0gradients_1/reply_xtr/dense_2/LeakyRelu_grad/SumSum3gradients_1/reply_xtr/dense_2/LeakyRelu_grad/SelectBgradients_1/reply_xtr/dense_2/LeakyRelu_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
4gradients_1/reply_xtr/dense_2/LeakyRelu_grad/ReshapeReshape0gradients_1/reply_xtr/dense_2/LeakyRelu_grad/Sum2gradients_1/reply_xtr/dense_2/LeakyRelu_grad/Shape*
T0*
Tshape0
�
2gradients_1/reply_xtr/dense_2/LeakyRelu_grad/Sum_1Sum5gradients_1/reply_xtr/dense_2/LeakyRelu_grad/Select_1Dgradients_1/reply_xtr/dense_2/LeakyRelu_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
6gradients_1/reply_xtr/dense_2/LeakyRelu_grad/Reshape_1Reshape2gradients_1/reply_xtr/dense_2/LeakyRelu_grad/Sum_14gradients_1/reply_xtr/dense_2/LeakyRelu_grad/Shape_1*
T0*
Tshape0
`
7gradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/ShapeConst*
valueB *
dtype0
w
9gradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/Shape_1Shapeexpand_xtr/dense_2/BiasAdd*
T0*
out_type0
�
Ggradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/Shape9gradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/Shape_1*
T0
�
5gradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/MulMul5gradients_1/expand_xtr/dense_2/LeakyRelu_grad/Reshapeexpand_xtr/dense_2/BiasAdd*
T0
�
5gradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/SumSum5gradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/MulGgradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
9gradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/ReshapeReshape5gradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/Sum7gradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
�
7gradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/Mul_1Mul"expand_xtr/dense_2/LeakyRelu/alpha5gradients_1/expand_xtr/dense_2/LeakyRelu_grad/Reshape*
T0
�
7gradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/Sum_1Sum7gradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/Mul_1Igradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
;gradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/Reshape_1Reshape7gradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/Sum_19gradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
^
5gradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/ShapeConst*
valueB *
dtype0
s
7gradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/Shape_1Shapelike_xtr/dense_2/BiasAdd*
T0*
out_type0
�
Egradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/Shape7gradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/Shape_1*
T0
�
3gradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/MulMul3gradients_1/like_xtr/dense_2/LeakyRelu_grad/Reshapelike_xtr/dense_2/BiasAdd*
T0
�
3gradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/SumSum3gradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/MulEgradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
7gradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/ReshapeReshape3gradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/Sum5gradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
�
5gradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/Mul_1Mul like_xtr/dense_2/LeakyRelu/alpha3gradients_1/like_xtr/dense_2/LeakyRelu_grad/Reshape*
T0
�
5gradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/Sum_1Sum5gradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/Mul_1Ggradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
9gradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/Reshape_1Reshape5gradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/Sum_17gradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
_
6gradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/ShapeConst*
valueB *
dtype0
u
8gradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/Shape_1Shapereply_xtr/dense_2/BiasAdd*
T0*
out_type0
�
Fgradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/Shape8gradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/Shape_1*
T0
�
4gradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/MulMul4gradients_1/reply_xtr/dense_2/LeakyRelu_grad/Reshapereply_xtr/dense_2/BiasAdd*
T0
�
4gradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/SumSum4gradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/MulFgradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
8gradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/ReshapeReshape4gradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/Sum6gradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
�
6gradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/Mul_1Mul!reply_xtr/dense_2/LeakyRelu/alpha4gradients_1/reply_xtr/dense_2/LeakyRelu_grad/Reshape*
T0
�
6gradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/Sum_1Sum6gradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/Mul_1Hgradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
:gradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/Reshape_1Reshape6gradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/Sum_18gradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
�
gradients_1/AddN_1AddN7gradients_1/expand_xtr/dense_2/LeakyRelu_grad/Reshape_1;gradients_1/expand_xtr/dense_2/LeakyRelu/mul_grad/Reshape_1*
T0*J
_class@
><loc:@gradients_1/expand_xtr/dense_2/LeakyRelu_grad/Reshape_1*
N
z
7gradients_1/expand_xtr/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_1*
T0*
data_formatNHWC
�
gradients_1/AddN_2AddN5gradients_1/like_xtr/dense_2/LeakyRelu_grad/Reshape_19gradients_1/like_xtr/dense_2/LeakyRelu/mul_grad/Reshape_1*
T0*H
_class>
<:loc:@gradients_1/like_xtr/dense_2/LeakyRelu_grad/Reshape_1*
N
x
5gradients_1/like_xtr/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_2*
T0*
data_formatNHWC
�
gradients_1/AddN_3AddN6gradients_1/reply_xtr/dense_2/LeakyRelu_grad/Reshape_1:gradients_1/reply_xtr/dense_2/LeakyRelu/mul_grad/Reshape_1*
T0*I
_class?
=;loc:@gradients_1/reply_xtr/dense_2/LeakyRelu_grad/Reshape_1*
N
y
6gradients_1/reply_xtr/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_3*
data_formatNHWC*
T0
�
1gradients_1/expand_xtr/dense_2/MatMul_grad/MatMulMatMulgradients_1/AddN_1expand_xtr/dense_2/kernel/read*
transpose_b(*
T0*
transpose_a( 
�
3gradients_1/expand_xtr/dense_2/MatMul_grad/MatMul_1MatMulexpand_xtr/dense_1/LeakyRelugradients_1/AddN_1*
T0*
transpose_a(*
transpose_b( 
�
/gradients_1/like_xtr/dense_2/MatMul_grad/MatMulMatMulgradients_1/AddN_2like_xtr/dense_2/kernel/read*
transpose_a( *
transpose_b(*
T0
�
1gradients_1/like_xtr/dense_2/MatMul_grad/MatMul_1MatMullike_xtr/dense_1/LeakyRelugradients_1/AddN_2*
transpose_a(*
transpose_b( *
T0
�
0gradients_1/reply_xtr/dense_2/MatMul_grad/MatMulMatMulgradients_1/AddN_3reply_xtr/dense_2/kernel/read*
T0*
transpose_a( *
transpose_b(
�
2gradients_1/reply_xtr/dense_2/MatMul_grad/MatMul_1MatMulreply_xtr/dense_1/LeakyRelugradients_1/AddN_3*
transpose_b( *
T0*
transpose_a(
w
3gradients_1/expand_xtr/dense_1/LeakyRelu_grad/ShapeShape expand_xtr/dense_1/LeakyRelu/mul*
T0*
out_type0
s
5gradients_1/expand_xtr/dense_1/LeakyRelu_grad/Shape_1Shapeexpand_xtr/dense_1/BiasAdd*
T0*
out_type0
�
5gradients_1/expand_xtr/dense_1/LeakyRelu_grad/Shape_2Shape1gradients_1/expand_xtr/dense_2/MatMul_grad/MatMul*
T0*
out_type0
f
9gradients_1/expand_xtr/dense_1/LeakyRelu_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
3gradients_1/expand_xtr/dense_1/LeakyRelu_grad/zerosFill5gradients_1/expand_xtr/dense_1/LeakyRelu_grad/Shape_29gradients_1/expand_xtr/dense_1/LeakyRelu_grad/zeros/Const*
T0*

index_type0
�
:gradients_1/expand_xtr/dense_1/LeakyRelu_grad/GreaterEqualGreaterEqual expand_xtr/dense_1/LeakyRelu/mulexpand_xtr/dense_1/BiasAdd*
T0
�
Cgradients_1/expand_xtr/dense_1/LeakyRelu_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients_1/expand_xtr/dense_1/LeakyRelu_grad/Shape5gradients_1/expand_xtr/dense_1/LeakyRelu_grad/Shape_1*
T0
�
4gradients_1/expand_xtr/dense_1/LeakyRelu_grad/SelectSelect:gradients_1/expand_xtr/dense_1/LeakyRelu_grad/GreaterEqual1gradients_1/expand_xtr/dense_2/MatMul_grad/MatMul3gradients_1/expand_xtr/dense_1/LeakyRelu_grad/zeros*
T0
�
6gradients_1/expand_xtr/dense_1/LeakyRelu_grad/Select_1Select:gradients_1/expand_xtr/dense_1/LeakyRelu_grad/GreaterEqual3gradients_1/expand_xtr/dense_1/LeakyRelu_grad/zeros1gradients_1/expand_xtr/dense_2/MatMul_grad/MatMul*
T0
�
1gradients_1/expand_xtr/dense_1/LeakyRelu_grad/SumSum4gradients_1/expand_xtr/dense_1/LeakyRelu_grad/SelectCgradients_1/expand_xtr/dense_1/LeakyRelu_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
5gradients_1/expand_xtr/dense_1/LeakyRelu_grad/ReshapeReshape1gradients_1/expand_xtr/dense_1/LeakyRelu_grad/Sum3gradients_1/expand_xtr/dense_1/LeakyRelu_grad/Shape*
T0*
Tshape0
�
3gradients_1/expand_xtr/dense_1/LeakyRelu_grad/Sum_1Sum6gradients_1/expand_xtr/dense_1/LeakyRelu_grad/Select_1Egradients_1/expand_xtr/dense_1/LeakyRelu_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
7gradients_1/expand_xtr/dense_1/LeakyRelu_grad/Reshape_1Reshape3gradients_1/expand_xtr/dense_1/LeakyRelu_grad/Sum_15gradients_1/expand_xtr/dense_1/LeakyRelu_grad/Shape_1*
T0*
Tshape0
s
1gradients_1/like_xtr/dense_1/LeakyRelu_grad/ShapeShapelike_xtr/dense_1/LeakyRelu/mul*
T0*
out_type0
o
3gradients_1/like_xtr/dense_1/LeakyRelu_grad/Shape_1Shapelike_xtr/dense_1/BiasAdd*
T0*
out_type0
�
3gradients_1/like_xtr/dense_1/LeakyRelu_grad/Shape_2Shape/gradients_1/like_xtr/dense_2/MatMul_grad/MatMul*
T0*
out_type0
d
7gradients_1/like_xtr/dense_1/LeakyRelu_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
1gradients_1/like_xtr/dense_1/LeakyRelu_grad/zerosFill3gradients_1/like_xtr/dense_1/LeakyRelu_grad/Shape_27gradients_1/like_xtr/dense_1/LeakyRelu_grad/zeros/Const*
T0*

index_type0
�
8gradients_1/like_xtr/dense_1/LeakyRelu_grad/GreaterEqualGreaterEquallike_xtr/dense_1/LeakyRelu/mullike_xtr/dense_1/BiasAdd*
T0
�
Agradients_1/like_xtr/dense_1/LeakyRelu_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients_1/like_xtr/dense_1/LeakyRelu_grad/Shape3gradients_1/like_xtr/dense_1/LeakyRelu_grad/Shape_1*
T0
�
2gradients_1/like_xtr/dense_1/LeakyRelu_grad/SelectSelect8gradients_1/like_xtr/dense_1/LeakyRelu_grad/GreaterEqual/gradients_1/like_xtr/dense_2/MatMul_grad/MatMul1gradients_1/like_xtr/dense_1/LeakyRelu_grad/zeros*
T0
�
4gradients_1/like_xtr/dense_1/LeakyRelu_grad/Select_1Select8gradients_1/like_xtr/dense_1/LeakyRelu_grad/GreaterEqual1gradients_1/like_xtr/dense_1/LeakyRelu_grad/zeros/gradients_1/like_xtr/dense_2/MatMul_grad/MatMul*
T0
�
/gradients_1/like_xtr/dense_1/LeakyRelu_grad/SumSum2gradients_1/like_xtr/dense_1/LeakyRelu_grad/SelectAgradients_1/like_xtr/dense_1/LeakyRelu_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
3gradients_1/like_xtr/dense_1/LeakyRelu_grad/ReshapeReshape/gradients_1/like_xtr/dense_1/LeakyRelu_grad/Sum1gradients_1/like_xtr/dense_1/LeakyRelu_grad/Shape*
T0*
Tshape0
�
1gradients_1/like_xtr/dense_1/LeakyRelu_grad/Sum_1Sum4gradients_1/like_xtr/dense_1/LeakyRelu_grad/Select_1Cgradients_1/like_xtr/dense_1/LeakyRelu_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
5gradients_1/like_xtr/dense_1/LeakyRelu_grad/Reshape_1Reshape1gradients_1/like_xtr/dense_1/LeakyRelu_grad/Sum_13gradients_1/like_xtr/dense_1/LeakyRelu_grad/Shape_1*
T0*
Tshape0
u
2gradients_1/reply_xtr/dense_1/LeakyRelu_grad/ShapeShapereply_xtr/dense_1/LeakyRelu/mul*
T0*
out_type0
q
4gradients_1/reply_xtr/dense_1/LeakyRelu_grad/Shape_1Shapereply_xtr/dense_1/BiasAdd*
T0*
out_type0
�
4gradients_1/reply_xtr/dense_1/LeakyRelu_grad/Shape_2Shape0gradients_1/reply_xtr/dense_2/MatMul_grad/MatMul*
T0*
out_type0
e
8gradients_1/reply_xtr/dense_1/LeakyRelu_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
2gradients_1/reply_xtr/dense_1/LeakyRelu_grad/zerosFill4gradients_1/reply_xtr/dense_1/LeakyRelu_grad/Shape_28gradients_1/reply_xtr/dense_1/LeakyRelu_grad/zeros/Const*
T0*

index_type0
�
9gradients_1/reply_xtr/dense_1/LeakyRelu_grad/GreaterEqualGreaterEqualreply_xtr/dense_1/LeakyRelu/mulreply_xtr/dense_1/BiasAdd*
T0
�
Bgradients_1/reply_xtr/dense_1/LeakyRelu_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients_1/reply_xtr/dense_1/LeakyRelu_grad/Shape4gradients_1/reply_xtr/dense_1/LeakyRelu_grad/Shape_1*
T0
�
3gradients_1/reply_xtr/dense_1/LeakyRelu_grad/SelectSelect9gradients_1/reply_xtr/dense_1/LeakyRelu_grad/GreaterEqual0gradients_1/reply_xtr/dense_2/MatMul_grad/MatMul2gradients_1/reply_xtr/dense_1/LeakyRelu_grad/zeros*
T0
�
5gradients_1/reply_xtr/dense_1/LeakyRelu_grad/Select_1Select9gradients_1/reply_xtr/dense_1/LeakyRelu_grad/GreaterEqual2gradients_1/reply_xtr/dense_1/LeakyRelu_grad/zeros0gradients_1/reply_xtr/dense_2/MatMul_grad/MatMul*
T0
�
0gradients_1/reply_xtr/dense_1/LeakyRelu_grad/SumSum3gradients_1/reply_xtr/dense_1/LeakyRelu_grad/SelectBgradients_1/reply_xtr/dense_1/LeakyRelu_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
4gradients_1/reply_xtr/dense_1/LeakyRelu_grad/ReshapeReshape0gradients_1/reply_xtr/dense_1/LeakyRelu_grad/Sum2gradients_1/reply_xtr/dense_1/LeakyRelu_grad/Shape*
T0*
Tshape0
�
2gradients_1/reply_xtr/dense_1/LeakyRelu_grad/Sum_1Sum5gradients_1/reply_xtr/dense_1/LeakyRelu_grad/Select_1Dgradients_1/reply_xtr/dense_1/LeakyRelu_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
6gradients_1/reply_xtr/dense_1/LeakyRelu_grad/Reshape_1Reshape2gradients_1/reply_xtr/dense_1/LeakyRelu_grad/Sum_14gradients_1/reply_xtr/dense_1/LeakyRelu_grad/Shape_1*
T0*
Tshape0
`
7gradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/ShapeConst*
valueB *
dtype0
w
9gradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/Shape_1Shapeexpand_xtr/dense_1/BiasAdd*
T0*
out_type0
�
Ggradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/Shape9gradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/Shape_1*
T0
�
5gradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/MulMul5gradients_1/expand_xtr/dense_1/LeakyRelu_grad/Reshapeexpand_xtr/dense_1/BiasAdd*
T0
�
5gradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/SumSum5gradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/MulGgradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
9gradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/ReshapeReshape5gradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/Sum7gradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
�
7gradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/Mul_1Mul"expand_xtr/dense_1/LeakyRelu/alpha5gradients_1/expand_xtr/dense_1/LeakyRelu_grad/Reshape*
T0
�
7gradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/Sum_1Sum7gradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/Mul_1Igradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
;gradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/Reshape_1Reshape7gradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/Sum_19gradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
^
5gradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/ShapeConst*
valueB *
dtype0
s
7gradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/Shape_1Shapelike_xtr/dense_1/BiasAdd*
T0*
out_type0
�
Egradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/Shape7gradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/Shape_1*
T0
�
3gradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/MulMul3gradients_1/like_xtr/dense_1/LeakyRelu_grad/Reshapelike_xtr/dense_1/BiasAdd*
T0
�
3gradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/SumSum3gradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/MulEgradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
7gradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/ReshapeReshape3gradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/Sum5gradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
�
5gradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/Mul_1Mul like_xtr/dense_1/LeakyRelu/alpha3gradients_1/like_xtr/dense_1/LeakyRelu_grad/Reshape*
T0
�
5gradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/Sum_1Sum5gradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/Mul_1Ggradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
9gradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/Reshape_1Reshape5gradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/Sum_17gradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
_
6gradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/ShapeConst*
valueB *
dtype0
u
8gradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/Shape_1Shapereply_xtr/dense_1/BiasAdd*
T0*
out_type0
�
Fgradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/Shape8gradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/Shape_1*
T0
�
4gradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/MulMul4gradients_1/reply_xtr/dense_1/LeakyRelu_grad/Reshapereply_xtr/dense_1/BiasAdd*
T0
�
4gradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/SumSum4gradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/MulFgradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
8gradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/ReshapeReshape4gradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/Sum6gradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
�
6gradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/Mul_1Mul!reply_xtr/dense_1/LeakyRelu/alpha4gradients_1/reply_xtr/dense_1/LeakyRelu_grad/Reshape*
T0
�
6gradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/Sum_1Sum6gradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/Mul_1Hgradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
:gradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/Reshape_1Reshape6gradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/Sum_18gradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
�
gradients_1/AddN_4AddN7gradients_1/expand_xtr/dense_1/LeakyRelu_grad/Reshape_1;gradients_1/expand_xtr/dense_1/LeakyRelu/mul_grad/Reshape_1*
T0*J
_class@
><loc:@gradients_1/expand_xtr/dense_1/LeakyRelu_grad/Reshape_1*
N
z
7gradients_1/expand_xtr/dense_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_4*
T0*
data_formatNHWC
�
gradients_1/AddN_5AddN5gradients_1/like_xtr/dense_1/LeakyRelu_grad/Reshape_19gradients_1/like_xtr/dense_1/LeakyRelu/mul_grad/Reshape_1*
T0*H
_class>
<:loc:@gradients_1/like_xtr/dense_1/LeakyRelu_grad/Reshape_1*
N
x
5gradients_1/like_xtr/dense_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_5*
data_formatNHWC*
T0
�
gradients_1/AddN_6AddN6gradients_1/reply_xtr/dense_1/LeakyRelu_grad/Reshape_1:gradients_1/reply_xtr/dense_1/LeakyRelu/mul_grad/Reshape_1*
T0*I
_class?
=;loc:@gradients_1/reply_xtr/dense_1/LeakyRelu_grad/Reshape_1*
N
y
6gradients_1/reply_xtr/dense_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_6*
data_formatNHWC*
T0
�
1gradients_1/expand_xtr/dense_1/MatMul_grad/MatMulMatMulgradients_1/AddN_4expand_xtr/dense_1/kernel/read*
transpose_b(*
T0*
transpose_a( 
�
3gradients_1/expand_xtr/dense_1/MatMul_grad/MatMul_1MatMulexpand_xtr/dense/LeakyRelugradients_1/AddN_4*
T0*
transpose_a(*
transpose_b( 
�
/gradients_1/like_xtr/dense_1/MatMul_grad/MatMulMatMulgradients_1/AddN_5like_xtr/dense_1/kernel/read*
T0*
transpose_a( *
transpose_b(
�
1gradients_1/like_xtr/dense_1/MatMul_grad/MatMul_1MatMullike_xtr/dense/LeakyRelugradients_1/AddN_5*
transpose_b( *
T0*
transpose_a(
�
0gradients_1/reply_xtr/dense_1/MatMul_grad/MatMulMatMulgradients_1/AddN_6reply_xtr/dense_1/kernel/read*
T0*
transpose_a( *
transpose_b(
�
2gradients_1/reply_xtr/dense_1/MatMul_grad/MatMul_1MatMulreply_xtr/dense/LeakyRelugradients_1/AddN_6*
T0*
transpose_a(*
transpose_b( 
s
1gradients_1/expand_xtr/dense/LeakyRelu_grad/ShapeShapeexpand_xtr/dense/LeakyRelu/mul*
T0*
out_type0
o
3gradients_1/expand_xtr/dense/LeakyRelu_grad/Shape_1Shapeexpand_xtr/dense/BiasAdd*
T0*
out_type0
�
3gradients_1/expand_xtr/dense/LeakyRelu_grad/Shape_2Shape1gradients_1/expand_xtr/dense_1/MatMul_grad/MatMul*
T0*
out_type0
d
7gradients_1/expand_xtr/dense/LeakyRelu_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
1gradients_1/expand_xtr/dense/LeakyRelu_grad/zerosFill3gradients_1/expand_xtr/dense/LeakyRelu_grad/Shape_27gradients_1/expand_xtr/dense/LeakyRelu_grad/zeros/Const*
T0*

index_type0
�
8gradients_1/expand_xtr/dense/LeakyRelu_grad/GreaterEqualGreaterEqualexpand_xtr/dense/LeakyRelu/mulexpand_xtr/dense/BiasAdd*
T0
�
Agradients_1/expand_xtr/dense/LeakyRelu_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients_1/expand_xtr/dense/LeakyRelu_grad/Shape3gradients_1/expand_xtr/dense/LeakyRelu_grad/Shape_1*
T0
�
2gradients_1/expand_xtr/dense/LeakyRelu_grad/SelectSelect8gradients_1/expand_xtr/dense/LeakyRelu_grad/GreaterEqual1gradients_1/expand_xtr/dense_1/MatMul_grad/MatMul1gradients_1/expand_xtr/dense/LeakyRelu_grad/zeros*
T0
�
4gradients_1/expand_xtr/dense/LeakyRelu_grad/Select_1Select8gradients_1/expand_xtr/dense/LeakyRelu_grad/GreaterEqual1gradients_1/expand_xtr/dense/LeakyRelu_grad/zeros1gradients_1/expand_xtr/dense_1/MatMul_grad/MatMul*
T0
�
/gradients_1/expand_xtr/dense/LeakyRelu_grad/SumSum2gradients_1/expand_xtr/dense/LeakyRelu_grad/SelectAgradients_1/expand_xtr/dense/LeakyRelu_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
3gradients_1/expand_xtr/dense/LeakyRelu_grad/ReshapeReshape/gradients_1/expand_xtr/dense/LeakyRelu_grad/Sum1gradients_1/expand_xtr/dense/LeakyRelu_grad/Shape*
T0*
Tshape0
�
1gradients_1/expand_xtr/dense/LeakyRelu_grad/Sum_1Sum4gradients_1/expand_xtr/dense/LeakyRelu_grad/Select_1Cgradients_1/expand_xtr/dense/LeakyRelu_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
5gradients_1/expand_xtr/dense/LeakyRelu_grad/Reshape_1Reshape1gradients_1/expand_xtr/dense/LeakyRelu_grad/Sum_13gradients_1/expand_xtr/dense/LeakyRelu_grad/Shape_1*
T0*
Tshape0
o
/gradients_1/like_xtr/dense/LeakyRelu_grad/ShapeShapelike_xtr/dense/LeakyRelu/mul*
T0*
out_type0
k
1gradients_1/like_xtr/dense/LeakyRelu_grad/Shape_1Shapelike_xtr/dense/BiasAdd*
T0*
out_type0
�
1gradients_1/like_xtr/dense/LeakyRelu_grad/Shape_2Shape/gradients_1/like_xtr/dense_1/MatMul_grad/MatMul*
T0*
out_type0
b
5gradients_1/like_xtr/dense/LeakyRelu_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
/gradients_1/like_xtr/dense/LeakyRelu_grad/zerosFill1gradients_1/like_xtr/dense/LeakyRelu_grad/Shape_25gradients_1/like_xtr/dense/LeakyRelu_grad/zeros/Const*
T0*

index_type0
�
6gradients_1/like_xtr/dense/LeakyRelu_grad/GreaterEqualGreaterEquallike_xtr/dense/LeakyRelu/mullike_xtr/dense/BiasAdd*
T0
�
?gradients_1/like_xtr/dense/LeakyRelu_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients_1/like_xtr/dense/LeakyRelu_grad/Shape1gradients_1/like_xtr/dense/LeakyRelu_grad/Shape_1*
T0
�
0gradients_1/like_xtr/dense/LeakyRelu_grad/SelectSelect6gradients_1/like_xtr/dense/LeakyRelu_grad/GreaterEqual/gradients_1/like_xtr/dense_1/MatMul_grad/MatMul/gradients_1/like_xtr/dense/LeakyRelu_grad/zeros*
T0
�
2gradients_1/like_xtr/dense/LeakyRelu_grad/Select_1Select6gradients_1/like_xtr/dense/LeakyRelu_grad/GreaterEqual/gradients_1/like_xtr/dense/LeakyRelu_grad/zeros/gradients_1/like_xtr/dense_1/MatMul_grad/MatMul*
T0
�
-gradients_1/like_xtr/dense/LeakyRelu_grad/SumSum0gradients_1/like_xtr/dense/LeakyRelu_grad/Select?gradients_1/like_xtr/dense/LeakyRelu_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
1gradients_1/like_xtr/dense/LeakyRelu_grad/ReshapeReshape-gradients_1/like_xtr/dense/LeakyRelu_grad/Sum/gradients_1/like_xtr/dense/LeakyRelu_grad/Shape*
T0*
Tshape0
�
/gradients_1/like_xtr/dense/LeakyRelu_grad/Sum_1Sum2gradients_1/like_xtr/dense/LeakyRelu_grad/Select_1Agradients_1/like_xtr/dense/LeakyRelu_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
3gradients_1/like_xtr/dense/LeakyRelu_grad/Reshape_1Reshape/gradients_1/like_xtr/dense/LeakyRelu_grad/Sum_11gradients_1/like_xtr/dense/LeakyRelu_grad/Shape_1*
T0*
Tshape0
q
0gradients_1/reply_xtr/dense/LeakyRelu_grad/ShapeShapereply_xtr/dense/LeakyRelu/mul*
T0*
out_type0
m
2gradients_1/reply_xtr/dense/LeakyRelu_grad/Shape_1Shapereply_xtr/dense/BiasAdd*
T0*
out_type0
�
2gradients_1/reply_xtr/dense/LeakyRelu_grad/Shape_2Shape0gradients_1/reply_xtr/dense_1/MatMul_grad/MatMul*
T0*
out_type0
c
6gradients_1/reply_xtr/dense/LeakyRelu_grad/zeros/ConstConst*
dtype0*
valueB
 *    
�
0gradients_1/reply_xtr/dense/LeakyRelu_grad/zerosFill2gradients_1/reply_xtr/dense/LeakyRelu_grad/Shape_26gradients_1/reply_xtr/dense/LeakyRelu_grad/zeros/Const*
T0*

index_type0
�
7gradients_1/reply_xtr/dense/LeakyRelu_grad/GreaterEqualGreaterEqualreply_xtr/dense/LeakyRelu/mulreply_xtr/dense/BiasAdd*
T0
�
@gradients_1/reply_xtr/dense/LeakyRelu_grad/BroadcastGradientArgsBroadcastGradientArgs0gradients_1/reply_xtr/dense/LeakyRelu_grad/Shape2gradients_1/reply_xtr/dense/LeakyRelu_grad/Shape_1*
T0
�
1gradients_1/reply_xtr/dense/LeakyRelu_grad/SelectSelect7gradients_1/reply_xtr/dense/LeakyRelu_grad/GreaterEqual0gradients_1/reply_xtr/dense_1/MatMul_grad/MatMul0gradients_1/reply_xtr/dense/LeakyRelu_grad/zeros*
T0
�
3gradients_1/reply_xtr/dense/LeakyRelu_grad/Select_1Select7gradients_1/reply_xtr/dense/LeakyRelu_grad/GreaterEqual0gradients_1/reply_xtr/dense/LeakyRelu_grad/zeros0gradients_1/reply_xtr/dense_1/MatMul_grad/MatMul*
T0
�
.gradients_1/reply_xtr/dense/LeakyRelu_grad/SumSum1gradients_1/reply_xtr/dense/LeakyRelu_grad/Select@gradients_1/reply_xtr/dense/LeakyRelu_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
2gradients_1/reply_xtr/dense/LeakyRelu_grad/ReshapeReshape.gradients_1/reply_xtr/dense/LeakyRelu_grad/Sum0gradients_1/reply_xtr/dense/LeakyRelu_grad/Shape*
T0*
Tshape0
�
0gradients_1/reply_xtr/dense/LeakyRelu_grad/Sum_1Sum3gradients_1/reply_xtr/dense/LeakyRelu_grad/Select_1Bgradients_1/reply_xtr/dense/LeakyRelu_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
4gradients_1/reply_xtr/dense/LeakyRelu_grad/Reshape_1Reshape0gradients_1/reply_xtr/dense/LeakyRelu_grad/Sum_12gradients_1/reply_xtr/dense/LeakyRelu_grad/Shape_1*
T0*
Tshape0
^
5gradients_1/expand_xtr/dense/LeakyRelu/mul_grad/ShapeConst*
dtype0*
valueB 
s
7gradients_1/expand_xtr/dense/LeakyRelu/mul_grad/Shape_1Shapeexpand_xtr/dense/BiasAdd*
T0*
out_type0
�
Egradients_1/expand_xtr/dense/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients_1/expand_xtr/dense/LeakyRelu/mul_grad/Shape7gradients_1/expand_xtr/dense/LeakyRelu/mul_grad/Shape_1*
T0
�
3gradients_1/expand_xtr/dense/LeakyRelu/mul_grad/MulMul3gradients_1/expand_xtr/dense/LeakyRelu_grad/Reshapeexpand_xtr/dense/BiasAdd*
T0
�
3gradients_1/expand_xtr/dense/LeakyRelu/mul_grad/SumSum3gradients_1/expand_xtr/dense/LeakyRelu/mul_grad/MulEgradients_1/expand_xtr/dense/LeakyRelu/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
7gradients_1/expand_xtr/dense/LeakyRelu/mul_grad/ReshapeReshape3gradients_1/expand_xtr/dense/LeakyRelu/mul_grad/Sum5gradients_1/expand_xtr/dense/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
�
5gradients_1/expand_xtr/dense/LeakyRelu/mul_grad/Mul_1Mul expand_xtr/dense/LeakyRelu/alpha3gradients_1/expand_xtr/dense/LeakyRelu_grad/Reshape*
T0
�
5gradients_1/expand_xtr/dense/LeakyRelu/mul_grad/Sum_1Sum5gradients_1/expand_xtr/dense/LeakyRelu/mul_grad/Mul_1Ggradients_1/expand_xtr/dense/LeakyRelu/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
9gradients_1/expand_xtr/dense/LeakyRelu/mul_grad/Reshape_1Reshape5gradients_1/expand_xtr/dense/LeakyRelu/mul_grad/Sum_17gradients_1/expand_xtr/dense/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
\
3gradients_1/like_xtr/dense/LeakyRelu/mul_grad/ShapeConst*
dtype0*
valueB 
o
5gradients_1/like_xtr/dense/LeakyRelu/mul_grad/Shape_1Shapelike_xtr/dense/BiasAdd*
T0*
out_type0
�
Cgradients_1/like_xtr/dense/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients_1/like_xtr/dense/LeakyRelu/mul_grad/Shape5gradients_1/like_xtr/dense/LeakyRelu/mul_grad/Shape_1*
T0
�
1gradients_1/like_xtr/dense/LeakyRelu/mul_grad/MulMul1gradients_1/like_xtr/dense/LeakyRelu_grad/Reshapelike_xtr/dense/BiasAdd*
T0
�
1gradients_1/like_xtr/dense/LeakyRelu/mul_grad/SumSum1gradients_1/like_xtr/dense/LeakyRelu/mul_grad/MulCgradients_1/like_xtr/dense/LeakyRelu/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
5gradients_1/like_xtr/dense/LeakyRelu/mul_grad/ReshapeReshape1gradients_1/like_xtr/dense/LeakyRelu/mul_grad/Sum3gradients_1/like_xtr/dense/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
�
3gradients_1/like_xtr/dense/LeakyRelu/mul_grad/Mul_1Mullike_xtr/dense/LeakyRelu/alpha1gradients_1/like_xtr/dense/LeakyRelu_grad/Reshape*
T0
�
3gradients_1/like_xtr/dense/LeakyRelu/mul_grad/Sum_1Sum3gradients_1/like_xtr/dense/LeakyRelu/mul_grad/Mul_1Egradients_1/like_xtr/dense/LeakyRelu/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
7gradients_1/like_xtr/dense/LeakyRelu/mul_grad/Reshape_1Reshape3gradients_1/like_xtr/dense/LeakyRelu/mul_grad/Sum_15gradients_1/like_xtr/dense/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
]
4gradients_1/reply_xtr/dense/LeakyRelu/mul_grad/ShapeConst*
valueB *
dtype0
q
6gradients_1/reply_xtr/dense/LeakyRelu/mul_grad/Shape_1Shapereply_xtr/dense/BiasAdd*
T0*
out_type0
�
Dgradients_1/reply_xtr/dense/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients_1/reply_xtr/dense/LeakyRelu/mul_grad/Shape6gradients_1/reply_xtr/dense/LeakyRelu/mul_grad/Shape_1*
T0
�
2gradients_1/reply_xtr/dense/LeakyRelu/mul_grad/MulMul2gradients_1/reply_xtr/dense/LeakyRelu_grad/Reshapereply_xtr/dense/BiasAdd*
T0
�
2gradients_1/reply_xtr/dense/LeakyRelu/mul_grad/SumSum2gradients_1/reply_xtr/dense/LeakyRelu/mul_grad/MulDgradients_1/reply_xtr/dense/LeakyRelu/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
6gradients_1/reply_xtr/dense/LeakyRelu/mul_grad/ReshapeReshape2gradients_1/reply_xtr/dense/LeakyRelu/mul_grad/Sum4gradients_1/reply_xtr/dense/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
�
4gradients_1/reply_xtr/dense/LeakyRelu/mul_grad/Mul_1Mulreply_xtr/dense/LeakyRelu/alpha2gradients_1/reply_xtr/dense/LeakyRelu_grad/Reshape*
T0
�
4gradients_1/reply_xtr/dense/LeakyRelu/mul_grad/Sum_1Sum4gradients_1/reply_xtr/dense/LeakyRelu/mul_grad/Mul_1Fgradients_1/reply_xtr/dense/LeakyRelu/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
8gradients_1/reply_xtr/dense/LeakyRelu/mul_grad/Reshape_1Reshape4gradients_1/reply_xtr/dense/LeakyRelu/mul_grad/Sum_16gradients_1/reply_xtr/dense/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
�
gradients_1/AddN_7AddN5gradients_1/expand_xtr/dense/LeakyRelu_grad/Reshape_19gradients_1/expand_xtr/dense/LeakyRelu/mul_grad/Reshape_1*
N*
T0*H
_class>
<:loc:@gradients_1/expand_xtr/dense/LeakyRelu_grad/Reshape_1
x
5gradients_1/expand_xtr/dense/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_7*
T0*
data_formatNHWC
�
gradients_1/AddN_8AddN3gradients_1/like_xtr/dense/LeakyRelu_grad/Reshape_17gradients_1/like_xtr/dense/LeakyRelu/mul_grad/Reshape_1*
T0*F
_class<
:8loc:@gradients_1/like_xtr/dense/LeakyRelu_grad/Reshape_1*
N
v
3gradients_1/like_xtr/dense/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_8*
T0*
data_formatNHWC
�
gradients_1/AddN_9AddN4gradients_1/reply_xtr/dense/LeakyRelu_grad/Reshape_18gradients_1/reply_xtr/dense/LeakyRelu/mul_grad/Reshape_1*
T0*G
_class=
;9loc:@gradients_1/reply_xtr/dense/LeakyRelu_grad/Reshape_1*
N
w
4gradients_1/reply_xtr/dense/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_9*
T0*
data_formatNHWC
�
/gradients_1/expand_xtr/dense/MatMul_grad/MatMulMatMulgradients_1/AddN_7expand_xtr/dense/kernel/read*
T0*
transpose_a( *
transpose_b(
�
1gradients_1/expand_xtr/dense/MatMul_grad/MatMul_1MatMulconcatgradients_1/AddN_7*
T0*
transpose_a(*
transpose_b( 
�
-gradients_1/like_xtr/dense/MatMul_grad/MatMulMatMulgradients_1/AddN_8like_xtr/dense/kernel/read*
transpose_a( *
transpose_b(*
T0
�
/gradients_1/like_xtr/dense/MatMul_grad/MatMul_1MatMulconcatgradients_1/AddN_8*
T0*
transpose_a(*
transpose_b( 
�
.gradients_1/reply_xtr/dense/MatMul_grad/MatMulMatMulgradients_1/AddN_9reply_xtr/dense/kernel/read*
transpose_b(*
T0*
transpose_a( 
�
0gradients_1/reply_xtr/dense/MatMul_grad/MatMul_1MatMulconcatgradients_1/AddN_9*
T0*
transpose_a(*
transpose_b( 
�
gradients_1/AddN_10AddN/gradients_1/expand_xtr/dense/MatMul_grad/MatMul-gradients_1/like_xtr/dense/MatMul_grad/MatMul.gradients_1/reply_xtr/dense/MatMul_grad/MatMul*
N*
T0*B
_class8
64loc:@gradients_1/expand_xtr/dense/MatMul_grad/MatMul
F
gradients_1/concat_grad/RankConst*
value	B :*
dtype0
[
gradients_1/concat_grad/modFloorModconcat/axisgradients_1/concat_grad/Rank*
T0
J
gradients_1/concat_grad/ShapeShape	Reshape_1*
T0*
out_type0
v
gradients_1/concat_grad/ShapeNShapeN	Reshape_1	Reshape_3	Reshape_5	Reshape_7*
T0*
out_type0*
N
�
$gradients_1/concat_grad/ConcatOffsetConcatOffsetgradients_1/concat_grad/modgradients_1/concat_grad/ShapeN gradients_1/concat_grad/ShapeN:1 gradients_1/concat_grad/ShapeN:2 gradients_1/concat_grad/ShapeN:3*
N
�
gradients_1/concat_grad/SliceSlicegradients_1/AddN_10$gradients_1/concat_grad/ConcatOffsetgradients_1/concat_grad/ShapeN*
T0*
Index0
�
gradients_1/concat_grad/Slice_1Slicegradients_1/AddN_10&gradients_1/concat_grad/ConcatOffset:1 gradients_1/concat_grad/ShapeN:1*
T0*
Index0
�
gradients_1/concat_grad/Slice_2Slicegradients_1/AddN_10&gradients_1/concat_grad/ConcatOffset:2 gradients_1/concat_grad/ShapeN:2*
T0*
Index0
�
gradients_1/concat_grad/Slice_3Slicegradients_1/AddN_10&gradients_1/concat_grad/ConcatOffset:3 gradients_1/concat_grad/ShapeN:3*
T0*
Index0
K
 gradients_1/Reshape_1_grad/ShapeShapeReshape*
T0*
out_type0
�
"gradients_1/Reshape_1_grad/ReshapeReshapegradients_1/concat_grad/Slice gradients_1/Reshape_1_grad/Shape*
T0*
Tshape0
M
 gradients_1/Reshape_3_grad/ShapeShape	Reshape_2*
T0*
out_type0
�
"gradients_1/Reshape_3_grad/ReshapeReshapegradients_1/concat_grad/Slice_1 gradients_1/Reshape_3_grad/Shape*
T0*
Tshape0
M
 gradients_1/Reshape_5_grad/ShapeShape	Reshape_4*
T0*
out_type0
�
"gradients_1/Reshape_5_grad/ReshapeReshapegradients_1/concat_grad/Slice_2 gradients_1/Reshape_5_grad/Shape*
T0*
Tshape0
M
 gradients_1/Reshape_7_grad/ShapeShape	Reshape_6*
T0*
out_type0
�
"gradients_1/Reshape_7_grad/ReshapeReshapegradients_1/concat_grad/Slice_3 gradients_1/Reshape_7_grad/Shape*
T0*
Tshape0
Z
gradients_1/Reshape_grad/ShapeShapekai_input_user_embedding*
T0*
out_type0
�
 gradients_1/Reshape_grad/ReshapeReshape"gradients_1/Reshape_1_grad/Reshapegradients_1/Reshape_grad/Shape*
T0*
Tshape0
\
 gradients_1/Reshape_2_grad/ShapeShapekai_input_c_id_embedding*
T0*
out_type0
�
"gradients_1/Reshape_2_grad/ReshapeReshape"gradients_1/Reshape_3_grad/Reshape gradients_1/Reshape_2_grad/Shape*
T0*
Tshape0
^
 gradients_1/Reshape_4_grad/ShapeShapekai_input_c_info_embedding*
T0*
out_type0
�
"gradients_1/Reshape_4_grad/ReshapeReshape"gradients_1/Reshape_5_grad/Reshape gradients_1/Reshape_4_grad/Shape*
T0*
Tshape0
`
 gradients_1/Reshape_6_grad/ShapeShapekai_input_position_embedding*
T0*
out_type0
�
"gradients_1/Reshape_6_grad/ReshapeReshape"gradients_1/Reshape_7_grad/Reshape gradients_1/Reshape_6_grad/Shape*
T0*
Tshape0
�
:gradients_1/input_user_embedding/cond/Merge_grad/cond_gradSwitch gradients_1/Reshape_grad/Reshape!input_user_embedding/cond/pred_id*
T0*3
_class)
'%loc:@gradients_1/Reshape_grad/Reshape
�
:gradients_1/input_c_id_embedding/cond/Merge_grad/cond_gradSwitch"gradients_1/Reshape_2_grad/Reshape!input_c_id_embedding/cond/pred_id*
T0*5
_class+
)'loc:@gradients_1/Reshape_2_grad/Reshape
�
<gradients_1/input_c_info_embedding/cond/Merge_grad/cond_gradSwitch"gradients_1/Reshape_4_grad/Reshape#input_c_info_embedding/cond/pred_id*
T0*5
_class+
)'loc:@gradients_1/Reshape_4_grad/Reshape
�
>gradients_1/input_position_embedding/cond/Merge_grad/cond_gradSwitch"gradients_1/Reshape_6_grad/Reshape%input_position_embedding/cond/pred_id*
T0*5
_class+
)'loc:@gradients_1/Reshape_6_grad/Reshape
j
3gradients_1/input_user_embedding/cond/Pad_grad/RankRank$input_user_embedding/cond/SegmentSum*
T0
`
6gradients_1/input_user_embedding/cond/Pad_grad/stack/1Const*
value	B :*
dtype0
�
4gradients_1/input_user_embedding/cond/Pad_grad/stackPack3gradients_1/input_user_embedding/cond/Pad_grad/Rank6gradients_1/input_user_embedding/cond/Pad_grad/stack/1*
N*
T0*

axis 
o
:gradients_1/input_user_embedding/cond/Pad_grad/Slice/beginConst*
valueB"        *
dtype0
�
4gradients_1/input_user_embedding/cond/Pad_grad/SliceSlice&input_user_embedding/cond/Pad/paddings:gradients_1/input_user_embedding/cond/Pad_grad/Slice/begin4gradients_1/input_user_embedding/cond/Pad_grad/stack*
T0*
Index0
s
<gradients_1/input_user_embedding/cond/Pad_grad/Reshape/shapeConst*
dtype0*
valueB:
���������
�
6gradients_1/input_user_embedding/cond/Pad_grad/ReshapeReshape4gradients_1/input_user_embedding/cond/Pad_grad/Slice<gradients_1/input_user_embedding/cond/Pad_grad/Reshape/shape*
T0*
Tshape0
|
4gradients_1/input_user_embedding/cond/Pad_grad/ShapeShape$input_user_embedding/cond/SegmentSum*
T0*
out_type0
�
6gradients_1/input_user_embedding/cond/Pad_grad/Slice_1Slice<gradients_1/input_user_embedding/cond/Merge_grad/cond_grad:16gradients_1/input_user_embedding/cond/Pad_grad/Reshape4gradients_1/input_user_embedding/cond/Pad_grad/Shape*
T0*
Index0
j
3gradients_1/input_c_id_embedding/cond/Pad_grad/RankRank$input_c_id_embedding/cond/SegmentSum*
T0
`
6gradients_1/input_c_id_embedding/cond/Pad_grad/stack/1Const*
dtype0*
value	B :
�
4gradients_1/input_c_id_embedding/cond/Pad_grad/stackPack3gradients_1/input_c_id_embedding/cond/Pad_grad/Rank6gradients_1/input_c_id_embedding/cond/Pad_grad/stack/1*
T0*

axis *
N
o
:gradients_1/input_c_id_embedding/cond/Pad_grad/Slice/beginConst*
valueB"        *
dtype0
�
4gradients_1/input_c_id_embedding/cond/Pad_grad/SliceSlice&input_c_id_embedding/cond/Pad/paddings:gradients_1/input_c_id_embedding/cond/Pad_grad/Slice/begin4gradients_1/input_c_id_embedding/cond/Pad_grad/stack*
T0*
Index0
s
<gradients_1/input_c_id_embedding/cond/Pad_grad/Reshape/shapeConst*
valueB:
���������*
dtype0
�
6gradients_1/input_c_id_embedding/cond/Pad_grad/ReshapeReshape4gradients_1/input_c_id_embedding/cond/Pad_grad/Slice<gradients_1/input_c_id_embedding/cond/Pad_grad/Reshape/shape*
T0*
Tshape0
|
4gradients_1/input_c_id_embedding/cond/Pad_grad/ShapeShape$input_c_id_embedding/cond/SegmentSum*
T0*
out_type0
�
6gradients_1/input_c_id_embedding/cond/Pad_grad/Slice_1Slice<gradients_1/input_c_id_embedding/cond/Merge_grad/cond_grad:16gradients_1/input_c_id_embedding/cond/Pad_grad/Reshape4gradients_1/input_c_id_embedding/cond/Pad_grad/Shape*
T0*
Index0
n
5gradients_1/input_c_info_embedding/cond/Pad_grad/RankRank&input_c_info_embedding/cond/SegmentSum*
T0
b
8gradients_1/input_c_info_embedding/cond/Pad_grad/stack/1Const*
value	B :*
dtype0
�
6gradients_1/input_c_info_embedding/cond/Pad_grad/stackPack5gradients_1/input_c_info_embedding/cond/Pad_grad/Rank8gradients_1/input_c_info_embedding/cond/Pad_grad/stack/1*
T0*

axis *
N
q
<gradients_1/input_c_info_embedding/cond/Pad_grad/Slice/beginConst*
dtype0*
valueB"        
�
6gradients_1/input_c_info_embedding/cond/Pad_grad/SliceSlice(input_c_info_embedding/cond/Pad/paddings<gradients_1/input_c_info_embedding/cond/Pad_grad/Slice/begin6gradients_1/input_c_info_embedding/cond/Pad_grad/stack*
T0*
Index0
u
>gradients_1/input_c_info_embedding/cond/Pad_grad/Reshape/shapeConst*
dtype0*
valueB:
���������
�
8gradients_1/input_c_info_embedding/cond/Pad_grad/ReshapeReshape6gradients_1/input_c_info_embedding/cond/Pad_grad/Slice>gradients_1/input_c_info_embedding/cond/Pad_grad/Reshape/shape*
T0*
Tshape0
�
6gradients_1/input_c_info_embedding/cond/Pad_grad/ShapeShape&input_c_info_embedding/cond/SegmentSum*
T0*
out_type0
�
8gradients_1/input_c_info_embedding/cond/Pad_grad/Slice_1Slice>gradients_1/input_c_info_embedding/cond/Merge_grad/cond_grad:18gradients_1/input_c_info_embedding/cond/Pad_grad/Reshape6gradients_1/input_c_info_embedding/cond/Pad_grad/Shape*
T0*
Index0
r
7gradients_1/input_position_embedding/cond/Pad_grad/RankRank(input_position_embedding/cond/SegmentSum*
T0
d
:gradients_1/input_position_embedding/cond/Pad_grad/stack/1Const*
value	B :*
dtype0
�
8gradients_1/input_position_embedding/cond/Pad_grad/stackPack7gradients_1/input_position_embedding/cond/Pad_grad/Rank:gradients_1/input_position_embedding/cond/Pad_grad/stack/1*
T0*

axis *
N
s
>gradients_1/input_position_embedding/cond/Pad_grad/Slice/beginConst*
valueB"        *
dtype0
�
8gradients_1/input_position_embedding/cond/Pad_grad/SliceSlice*input_position_embedding/cond/Pad/paddings>gradients_1/input_position_embedding/cond/Pad_grad/Slice/begin8gradients_1/input_position_embedding/cond/Pad_grad/stack*
T0*
Index0
w
@gradients_1/input_position_embedding/cond/Pad_grad/Reshape/shapeConst*
dtype0*
valueB:
���������
�
:gradients_1/input_position_embedding/cond/Pad_grad/ReshapeReshape8gradients_1/input_position_embedding/cond/Pad_grad/Slice@gradients_1/input_position_embedding/cond/Pad_grad/Reshape/shape*
T0*
Tshape0
�
8gradients_1/input_position_embedding/cond/Pad_grad/ShapeShape(input_position_embedding/cond/SegmentSum*
T0*
out_type0
�
:gradients_1/input_position_embedding/cond/Pad_grad/Slice_1Slice@gradients_1/input_position_embedding/cond/Merge_grad/cond_grad:1:gradients_1/input_position_embedding/cond/Pad_grad/Reshape8gradients_1/input_position_embedding/cond/Pad_grad/Shape*
T0*
Index0
m
Cgradients_1/input_user_embedding/cond/SegmentSum_grad/GatherV2/axisConst*
value	B : *
dtype0
�
>gradients_1/input_user_embedding/cond/SegmentSum_grad/GatherV2GatherV26gradients_1/input_user_embedding/cond/Pad_grad/Slice_10input_user_embedding/cond/make_sparse_indice/subCgradients_1/input_user_embedding/cond/SegmentSum_grad/GatherV2/axis*
Tindices0*
Tparams0*
Taxis0
m
Cgradients_1/input_c_id_embedding/cond/SegmentSum_grad/GatherV2/axisConst*
value	B : *
dtype0
�
>gradients_1/input_c_id_embedding/cond/SegmentSum_grad/GatherV2GatherV26gradients_1/input_c_id_embedding/cond/Pad_grad/Slice_10input_c_id_embedding/cond/make_sparse_indice/subCgradients_1/input_c_id_embedding/cond/SegmentSum_grad/GatherV2/axis*
Tparams0*
Taxis0*
Tindices0
o
Egradients_1/input_c_info_embedding/cond/SegmentSum_grad/GatherV2/axisConst*
value	B : *
dtype0
�
@gradients_1/input_c_info_embedding/cond/SegmentSum_grad/GatherV2GatherV28gradients_1/input_c_info_embedding/cond/Pad_grad/Slice_12input_c_info_embedding/cond/make_sparse_indice/subEgradients_1/input_c_info_embedding/cond/SegmentSum_grad/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0
q
Ggradients_1/input_position_embedding/cond/SegmentSum_grad/GatherV2/axisConst*
value	B : *
dtype0
�
Bgradients_1/input_position_embedding/cond/SegmentSum_grad/GatherV2GatherV2:gradients_1/input_position_embedding/cond/Pad_grad/Slice_14input_position_embedding/cond/make_sparse_indice/subGgradients_1/input_position_embedding/cond/SegmentSum_grad/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0
�
9gradients_1/input_user_embedding/cond/GatherV2_grad/ShapeShape+input_user_embedding/cond/GatherV2/Switch:1*
T0*
out_type0	*-
_class#
!loc:@varlen_gather_4/ps_embed_4
�
;gradients_1/input_user_embedding/cond/GatherV2_grad/ToInt32Cast9gradients_1/input_user_embedding/cond/GatherV2_grad/Shape*
Truncate( *

DstT0*

SrcT0	*-
_class#
!loc:@varlen_gather_4/ps_embed_4
�
8gradients_1/input_user_embedding/cond/GatherV2_grad/SizeSize-input_user_embedding/cond/GatherV2/Switch_1:1*
T0*
out_type0
l
Bgradients_1/input_user_embedding/cond/GatherV2_grad/ExpandDims/dimConst*
value	B : *
dtype0
�
>gradients_1/input_user_embedding/cond/GatherV2_grad/ExpandDims
ExpandDims8gradients_1/input_user_embedding/cond/GatherV2_grad/SizeBgradients_1/input_user_embedding/cond/GatherV2_grad/ExpandDims/dim*
T0*

Tdim0
u
Ggradients_1/input_user_embedding/cond/GatherV2_grad/strided_slice/stackConst*
valueB:*
dtype0
w
Igradients_1/input_user_embedding/cond/GatherV2_grad/strided_slice/stack_1Const*
valueB: *
dtype0
w
Igradients_1/input_user_embedding/cond/GatherV2_grad/strided_slice/stack_2Const*
valueB:*
dtype0
�
Agradients_1/input_user_embedding/cond/GatherV2_grad/strided_sliceStridedSlice;gradients_1/input_user_embedding/cond/GatherV2_grad/ToInt32Ggradients_1/input_user_embedding/cond/GatherV2_grad/strided_slice/stackIgradients_1/input_user_embedding/cond/GatherV2_grad/strided_slice/stack_1Igradients_1/input_user_embedding/cond/GatherV2_grad/strided_slice/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
Index0*
T0
i
?gradients_1/input_user_embedding/cond/GatherV2_grad/concat/axisConst*
dtype0*
value	B : 
�
:gradients_1/input_user_embedding/cond/GatherV2_grad/concatConcatV2>gradients_1/input_user_embedding/cond/GatherV2_grad/ExpandDimsAgradients_1/input_user_embedding/cond/GatherV2_grad/strided_slice?gradients_1/input_user_embedding/cond/GatherV2_grad/concat/axis*

Tidx0*
T0*
N
�
;gradients_1/input_user_embedding/cond/GatherV2_grad/ReshapeReshape>gradients_1/input_user_embedding/cond/SegmentSum_grad/GatherV2:gradients_1/input_user_embedding/cond/GatherV2_grad/concat*
T0*
Tshape0
�
=gradients_1/input_user_embedding/cond/GatherV2_grad/Reshape_1Reshape-input_user_embedding/cond/GatherV2/Switch_1:1>gradients_1/input_user_embedding/cond/GatherV2_grad/ExpandDims*
T0*
Tshape0
�
9gradients_1/input_c_id_embedding/cond/GatherV2_grad/ShapeShape+input_c_id_embedding/cond/GatherV2/Switch:1*
T0*
out_type0	*/
_class%
#!loc:@varlen_gather_64/ps_embed_64
�
;gradients_1/input_c_id_embedding/cond/GatherV2_grad/ToInt32Cast9gradients_1/input_c_id_embedding/cond/GatherV2_grad/Shape*

SrcT0	*/
_class%
#!loc:@varlen_gather_64/ps_embed_64*
Truncate( *

DstT0
�
8gradients_1/input_c_id_embedding/cond/GatherV2_grad/SizeSize-input_c_id_embedding/cond/GatherV2/Switch_1:1*
T0*
out_type0
l
Bgradients_1/input_c_id_embedding/cond/GatherV2_grad/ExpandDims/dimConst*
dtype0*
value	B : 
�
>gradients_1/input_c_id_embedding/cond/GatherV2_grad/ExpandDims
ExpandDims8gradients_1/input_c_id_embedding/cond/GatherV2_grad/SizeBgradients_1/input_c_id_embedding/cond/GatherV2_grad/ExpandDims/dim*

Tdim0*
T0
u
Ggradients_1/input_c_id_embedding/cond/GatherV2_grad/strided_slice/stackConst*
valueB:*
dtype0
w
Igradients_1/input_c_id_embedding/cond/GatherV2_grad/strided_slice/stack_1Const*
dtype0*
valueB: 
w
Igradients_1/input_c_id_embedding/cond/GatherV2_grad/strided_slice/stack_2Const*
valueB:*
dtype0
�
Agradients_1/input_c_id_embedding/cond/GatherV2_grad/strided_sliceStridedSlice;gradients_1/input_c_id_embedding/cond/GatherV2_grad/ToInt32Ggradients_1/input_c_id_embedding/cond/GatherV2_grad/strided_slice/stackIgradients_1/input_c_id_embedding/cond/GatherV2_grad/strided_slice/stack_1Igradients_1/input_c_id_embedding/cond/GatherV2_grad/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
i
?gradients_1/input_c_id_embedding/cond/GatherV2_grad/concat/axisConst*
value	B : *
dtype0
�
:gradients_1/input_c_id_embedding/cond/GatherV2_grad/concatConcatV2>gradients_1/input_c_id_embedding/cond/GatherV2_grad/ExpandDimsAgradients_1/input_c_id_embedding/cond/GatherV2_grad/strided_slice?gradients_1/input_c_id_embedding/cond/GatherV2_grad/concat/axis*
N*

Tidx0*
T0
�
;gradients_1/input_c_id_embedding/cond/GatherV2_grad/ReshapeReshape>gradients_1/input_c_id_embedding/cond/SegmentSum_grad/GatherV2:gradients_1/input_c_id_embedding/cond/GatherV2_grad/concat*
T0*
Tshape0
�
=gradients_1/input_c_id_embedding/cond/GatherV2_grad/Reshape_1Reshape-input_c_id_embedding/cond/GatherV2/Switch_1:1>gradients_1/input_c_id_embedding/cond/GatherV2_grad/ExpandDims*
T0*
Tshape0
�
;gradients_1/input_c_info_embedding/cond/GatherV2_grad/ShapeShape-input_c_info_embedding/cond/GatherV2/Switch:1*
T0*
out_type0	*/
_class%
#!loc:@varlen_gather_32/ps_embed_32
�
=gradients_1/input_c_info_embedding/cond/GatherV2_grad/ToInt32Cast;gradients_1/input_c_info_embedding/cond/GatherV2_grad/Shape*

SrcT0	*/
_class%
#!loc:@varlen_gather_32/ps_embed_32*
Truncate( *

DstT0
�
:gradients_1/input_c_info_embedding/cond/GatherV2_grad/SizeSize/input_c_info_embedding/cond/GatherV2/Switch_1:1*
T0*
out_type0
n
Dgradients_1/input_c_info_embedding/cond/GatherV2_grad/ExpandDims/dimConst*
value	B : *
dtype0
�
@gradients_1/input_c_info_embedding/cond/GatherV2_grad/ExpandDims
ExpandDims:gradients_1/input_c_info_embedding/cond/GatherV2_grad/SizeDgradients_1/input_c_info_embedding/cond/GatherV2_grad/ExpandDims/dim*

Tdim0*
T0
w
Igradients_1/input_c_info_embedding/cond/GatherV2_grad/strided_slice/stackConst*
dtype0*
valueB:
y
Kgradients_1/input_c_info_embedding/cond/GatherV2_grad/strided_slice/stack_1Const*
valueB: *
dtype0
y
Kgradients_1/input_c_info_embedding/cond/GatherV2_grad/strided_slice/stack_2Const*
valueB:*
dtype0
�
Cgradients_1/input_c_info_embedding/cond/GatherV2_grad/strided_sliceStridedSlice=gradients_1/input_c_info_embedding/cond/GatherV2_grad/ToInt32Igradients_1/input_c_info_embedding/cond/GatherV2_grad/strided_slice/stackKgradients_1/input_c_info_embedding/cond/GatherV2_grad/strided_slice/stack_1Kgradients_1/input_c_info_embedding/cond/GatherV2_grad/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask
k
Agradients_1/input_c_info_embedding/cond/GatherV2_grad/concat/axisConst*
value	B : *
dtype0
�
<gradients_1/input_c_info_embedding/cond/GatherV2_grad/concatConcatV2@gradients_1/input_c_info_embedding/cond/GatherV2_grad/ExpandDimsCgradients_1/input_c_info_embedding/cond/GatherV2_grad/strided_sliceAgradients_1/input_c_info_embedding/cond/GatherV2_grad/concat/axis*

Tidx0*
T0*
N
�
=gradients_1/input_c_info_embedding/cond/GatherV2_grad/ReshapeReshape@gradients_1/input_c_info_embedding/cond/SegmentSum_grad/GatherV2<gradients_1/input_c_info_embedding/cond/GatherV2_grad/concat*
T0*
Tshape0
�
?gradients_1/input_c_info_embedding/cond/GatherV2_grad/Reshape_1Reshape/input_c_info_embedding/cond/GatherV2/Switch_1:1@gradients_1/input_c_info_embedding/cond/GatherV2_grad/ExpandDims*
T0*
Tshape0
�
=gradients_1/input_position_embedding/cond/GatherV2_grad/ShapeShape/input_position_embedding/cond/GatherV2/Switch:1*
T0*
out_type0	*-
_class#
!loc:@varlen_gather_8/ps_embed_8
�
?gradients_1/input_position_embedding/cond/GatherV2_grad/ToInt32Cast=gradients_1/input_position_embedding/cond/GatherV2_grad/Shape*

SrcT0	*-
_class#
!loc:@varlen_gather_8/ps_embed_8*
Truncate( *

DstT0
�
<gradients_1/input_position_embedding/cond/GatherV2_grad/SizeSize1input_position_embedding/cond/GatherV2/Switch_1:1*
T0*
out_type0
p
Fgradients_1/input_position_embedding/cond/GatherV2_grad/ExpandDims/dimConst*
value	B : *
dtype0
�
Bgradients_1/input_position_embedding/cond/GatherV2_grad/ExpandDims
ExpandDims<gradients_1/input_position_embedding/cond/GatherV2_grad/SizeFgradients_1/input_position_embedding/cond/GatherV2_grad/ExpandDims/dim*

Tdim0*
T0
y
Kgradients_1/input_position_embedding/cond/GatherV2_grad/strided_slice/stackConst*
valueB:*
dtype0
{
Mgradients_1/input_position_embedding/cond/GatherV2_grad/strided_slice/stack_1Const*
dtype0*
valueB: 
{
Mgradients_1/input_position_embedding/cond/GatherV2_grad/strided_slice/stack_2Const*
valueB:*
dtype0
�
Egradients_1/input_position_embedding/cond/GatherV2_grad/strided_sliceStridedSlice?gradients_1/input_position_embedding/cond/GatherV2_grad/ToInt32Kgradients_1/input_position_embedding/cond/GatherV2_grad/strided_slice/stackMgradients_1/input_position_embedding/cond/GatherV2_grad/strided_slice/stack_1Mgradients_1/input_position_embedding/cond/GatherV2_grad/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0
m
Cgradients_1/input_position_embedding/cond/GatherV2_grad/concat/axisConst*
value	B : *
dtype0
�
>gradients_1/input_position_embedding/cond/GatherV2_grad/concatConcatV2Bgradients_1/input_position_embedding/cond/GatherV2_grad/ExpandDimsEgradients_1/input_position_embedding/cond/GatherV2_grad/strided_sliceCgradients_1/input_position_embedding/cond/GatherV2_grad/concat/axis*

Tidx0*
T0*
N
�
?gradients_1/input_position_embedding/cond/GatherV2_grad/ReshapeReshapeBgradients_1/input_position_embedding/cond/SegmentSum_grad/GatherV2>gradients_1/input_position_embedding/cond/GatherV2_grad/concat*
T0*
Tshape0
�
Agradients_1/input_position_embedding/cond/GatherV2_grad/Reshape_1Reshape1input_position_embedding/cond/GatherV2/Switch_1:1Bgradients_1/input_position_embedding/cond/GatherV2_grad/ExpandDims*
T0*
Tshape0
d
gradients_1/SwitchSwitchvarlen_gather_4/ps_embed_4!input_user_embedding/cond/pred_id*
T0
=
gradients_1/IdentityIdentitygradients_1/Switch*
T0
I
gradients_1/Shape_1Shapegradients_1/Switch*
T0*
out_type0
[
gradients_1/zeros/ConstConst^gradients_1/Identity*
dtype0*
valueB
 *    
b
gradients_1/zerosFillgradients_1/Shape_1gradients_1/zeros/Const*
T0*

index_type0

Jgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/ShapeShapegradients_1/zeros*
T0*
out_type0
�
Xgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stackConst*
valueB: *
dtype0
�
Zgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stack_1Const*
valueB:*
dtype0
�
Zgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stack_2Const*
valueB:*
dtype0
�
Rgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_sliceStridedSliceJgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/ShapeXgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stackZgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stack_1Zgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0
z
Pgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/range/startConst*
dtype0*
value	B : 
z
Pgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/range/deltaConst*
value	B :*
dtype0
�
Jgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/rangeRangePgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/range/startRgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slicePgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/range/delta*

Tidx0
�
Dgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_gradMergegradients_1/zeros;gradients_1/input_user_embedding/cond/GatherV2_grad/Reshape*
T0*
N
�
Lgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/indicesMergeJgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/range=gradients_1/input_user_embedding/cond/GatherV2_grad/Reshape_1*
T0*
N
�
Pgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/dense_shapeMergeJgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/Shape;gradients_1/input_user_embedding/cond/GatherV2_grad/ToInt32*
T0*
N
h
gradients_1/Switch_1Switchvarlen_gather_64/ps_embed_64!input_c_id_embedding/cond/pred_id*
T0
A
gradients_1/Identity_1Identitygradients_1/Switch_1*
T0
K
gradients_1/Shape_2Shapegradients_1/Switch_1*
T0*
out_type0
_
gradients_1/zeros_1/ConstConst^gradients_1/Identity_1*
valueB
 *    *
dtype0
f
gradients_1/zeros_1Fillgradients_1/Shape_2gradients_1/zeros_1/Const*
T0*

index_type0
�
Jgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/ShapeShapegradients_1/zeros_1*
T0*
out_type0
�
Xgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stackConst*
dtype0*
valueB: 
�
Zgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stack_1Const*
dtype0*
valueB:
�
Zgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stack_2Const*
valueB:*
dtype0
�
Rgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_sliceStridedSliceJgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/ShapeXgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stackZgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stack_1Zgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
z
Pgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/range/startConst*
value	B : *
dtype0
z
Pgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/range/deltaConst*
value	B :*
dtype0
�
Jgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/rangeRangePgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/range/startRgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slicePgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/range/delta*

Tidx0
�
Dgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_gradMergegradients_1/zeros_1;gradients_1/input_c_id_embedding/cond/GatherV2_grad/Reshape*
N*
T0
�
Lgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/indicesMergeJgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/range=gradients_1/input_c_id_embedding/cond/GatherV2_grad/Reshape_1*
T0*
N
�
Pgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/dense_shapeMergeJgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/Shape;gradients_1/input_c_id_embedding/cond/GatherV2_grad/ToInt32*
T0*
N
j
gradients_1/Switch_2Switchvarlen_gather_32/ps_embed_32#input_c_info_embedding/cond/pred_id*
T0
A
gradients_1/Identity_2Identitygradients_1/Switch_2*
T0
K
gradients_1/Shape_3Shapegradients_1/Switch_2*
T0*
out_type0
_
gradients_1/zeros_2/ConstConst^gradients_1/Identity_2*
valueB
 *    *
dtype0
f
gradients_1/zeros_2Fillgradients_1/Shape_3gradients_1/zeros_2/Const*
T0*

index_type0
�
Lgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/ShapeShapegradients_1/zeros_2*
T0*
out_type0
�
Zgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stackConst*
valueB: *
dtype0
�
\gradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stack_1Const*
valueB:*
dtype0
�
\gradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stack_2Const*
valueB:*
dtype0
�
Tgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_sliceStridedSliceLgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/ShapeZgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stack\gradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stack_1\gradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
|
Rgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/range/startConst*
value	B : *
dtype0
|
Rgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/range/deltaConst*
value	B :*
dtype0
�
Lgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/rangeRangeRgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/range/startTgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_sliceRgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/range/delta*

Tidx0
�
Fgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_gradMergegradients_1/zeros_2=gradients_1/input_c_info_embedding/cond/GatherV2_grad/Reshape*
T0*
N
�
Ngradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/indicesMergeLgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/range?gradients_1/input_c_info_embedding/cond/GatherV2_grad/Reshape_1*
T0*
N
�
Rgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/dense_shapeMergeLgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/Shape=gradients_1/input_c_info_embedding/cond/GatherV2_grad/ToInt32*
T0*
N
j
gradients_1/Switch_3Switchvarlen_gather_8/ps_embed_8%input_position_embedding/cond/pred_id*
T0
A
gradients_1/Identity_3Identitygradients_1/Switch_3*
T0
K
gradients_1/Shape_4Shapegradients_1/Switch_3*
T0*
out_type0
_
gradients_1/zeros_3/ConstConst^gradients_1/Identity_3*
valueB
 *    *
dtype0
f
gradients_1/zeros_3Fillgradients_1/Shape_4gradients_1/zeros_3/Const*
T0*

index_type0
�
Ngradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/ShapeShapegradients_1/zeros_3*
T0*
out_type0
�
\gradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stackConst*
valueB: *
dtype0
�
^gradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stack_1Const*
valueB:*
dtype0
�
^gradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stack_2Const*
dtype0*
valueB:
�
Vgradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_sliceStridedSliceNgradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/Shape\gradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stack^gradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stack_1^gradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
~
Tgradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/range/startConst*
value	B : *
dtype0
~
Tgradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/range/deltaConst*
dtype0*
value	B :
�
Ngradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/rangeRangeTgradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/range/startVgradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/strided_sliceTgradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/range/delta*

Tidx0
�
Hgradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_gradMergegradients_1/zeros_3?gradients_1/input_position_embedding/cond/GatherV2_grad/Reshape*
N*
T0
�
Pgradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/indicesMergeNgradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/rangeAgradients_1/input_position_embedding/cond/GatherV2_grad/Reshape_1*
T0*
N
�
Tgradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/dense_shapeMergeNgradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/Shape?gradients_1/input_position_embedding/cond/GatherV2_grad/ToInt32*
T0*
N
q
1gradients_1/varlen_gather_4/ps_embed_4_grad/ShapeShapevarlen_gather_4/VarlenGather*
T0*
out_type0
\
3gradients_1/varlen_gather_4/ps_embed_4_grad/Shape_1Const*
valueB *
dtype0
�
Agradients_1/varlen_gather_4/ps_embed_4_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients_1/varlen_gather_4/ps_embed_4_grad/Shape3gradients_1/varlen_gather_4/ps_embed_4_grad/Shape_1*
T0
q
Cgradients_1/varlen_gather_4/ps_embed_4_grad/Mul/strided_slice/stackConst*
valueB: *
dtype0
s
Egradients_1/varlen_gather_4/ps_embed_4_grad/Mul/strided_slice/stack_1Const*
valueB:*
dtype0
s
Egradients_1/varlen_gather_4/ps_embed_4_grad/Mul/strided_slice/stack_2Const*
valueB:*
dtype0
�
=gradients_1/varlen_gather_4/ps_embed_4_grad/Mul/strided_sliceStridedSlicePgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/dense_shapeCgradients_1/varlen_gather_4/ps_embed_4_grad/Mul/strided_slice/stackEgradients_1/varlen_gather_4/ps_embed_4_grad/Mul/strided_slice/stack_1Egradients_1/varlen_gather_4/ps_embed_4_grad/Mul/strided_slice/stack_2*
end_mask *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
�
1gradients_1/varlen_gather_4/ps_embed_4_grad/Mul/xUnsortedSegmentSumDgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_gradLgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/indices=gradients_1/varlen_gather_4/ps_embed_4_grad/Mul/strided_slice*
T0*
Tnumsegments0*
Tindices0
�
/gradients_1/varlen_gather_4/ps_embed_4_grad/MulMul1gradients_1/varlen_gather_4/ps_embed_4_grad/Mul/xvarlen_gather_4/ps_embed_4/y*
T0
�
/gradients_1/varlen_gather_4/ps_embed_4_grad/SumSum/gradients_1/varlen_gather_4/ps_embed_4_grad/MulAgradients_1/varlen_gather_4/ps_embed_4_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
3gradients_1/varlen_gather_4/ps_embed_4_grad/ReshapeReshape/gradients_1/varlen_gather_4/ps_embed_4_grad/Sum1gradients_1/varlen_gather_4/ps_embed_4_grad/Shape*
T0*
Tshape0
s
Egradients_1/varlen_gather_4/ps_embed_4_grad/Mul_1/strided_slice/stackConst*
valueB: *
dtype0
u
Ggradients_1/varlen_gather_4/ps_embed_4_grad/Mul_1/strided_slice/stack_1Const*
valueB:*
dtype0
u
Ggradients_1/varlen_gather_4/ps_embed_4_grad/Mul_1/strided_slice/stack_2Const*
valueB:*
dtype0
�
?gradients_1/varlen_gather_4/ps_embed_4_grad/Mul_1/strided_sliceStridedSlicePgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/dense_shapeEgradients_1/varlen_gather_4/ps_embed_4_grad/Mul_1/strided_slice/stackGgradients_1/varlen_gather_4/ps_embed_4_grad/Mul_1/strided_slice/stack_1Ggradients_1/varlen_gather_4/ps_embed_4_grad/Mul_1/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0
�
3gradients_1/varlen_gather_4/ps_embed_4_grad/Mul_1/yUnsortedSegmentSumDgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_gradLgradients_1/input_user_embedding/cond/GatherV2/Switch_grad/cond_grad/indices?gradients_1/varlen_gather_4/ps_embed_4_grad/Mul_1/strided_slice*
Tnumsegments0*
Tindices0*
T0
�
1gradients_1/varlen_gather_4/ps_embed_4_grad/Mul_1Mulvarlen_gather_4/VarlenGather3gradients_1/varlen_gather_4/ps_embed_4_grad/Mul_1/y*
T0
�
1gradients_1/varlen_gather_4/ps_embed_4_grad/Sum_1Sum1gradients_1/varlen_gather_4/ps_embed_4_grad/Mul_1Cgradients_1/varlen_gather_4/ps_embed_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
5gradients_1/varlen_gather_4/ps_embed_4_grad/Reshape_1Reshape1gradients_1/varlen_gather_4/ps_embed_4_grad/Sum_13gradients_1/varlen_gather_4/ps_embed_4_grad/Shape_1*
T0*
Tshape0
t
3gradients_1/varlen_gather_64/ps_embed_64_grad/ShapeShapevarlen_gather_64/VarlenGather*
T0*
out_type0
^
5gradients_1/varlen_gather_64/ps_embed_64_grad/Shape_1Const*
valueB *
dtype0
�
Cgradients_1/varlen_gather_64/ps_embed_64_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients_1/varlen_gather_64/ps_embed_64_grad/Shape5gradients_1/varlen_gather_64/ps_embed_64_grad/Shape_1*
T0
s
Egradients_1/varlen_gather_64/ps_embed_64_grad/Mul/strided_slice/stackConst*
valueB: *
dtype0
u
Ggradients_1/varlen_gather_64/ps_embed_64_grad/Mul/strided_slice/stack_1Const*
valueB:*
dtype0
u
Ggradients_1/varlen_gather_64/ps_embed_64_grad/Mul/strided_slice/stack_2Const*
valueB:*
dtype0
�
?gradients_1/varlen_gather_64/ps_embed_64_grad/Mul/strided_sliceStridedSlicePgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/dense_shapeEgradients_1/varlen_gather_64/ps_embed_64_grad/Mul/strided_slice/stackGgradients_1/varlen_gather_64/ps_embed_64_grad/Mul/strided_slice/stack_1Ggradients_1/varlen_gather_64/ps_embed_64_grad/Mul/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
�
3gradients_1/varlen_gather_64/ps_embed_64_grad/Mul/xUnsortedSegmentSumDgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_gradLgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/indices?gradients_1/varlen_gather_64/ps_embed_64_grad/Mul/strided_slice*
Tnumsegments0*
Tindices0*
T0
�
1gradients_1/varlen_gather_64/ps_embed_64_grad/MulMul3gradients_1/varlen_gather_64/ps_embed_64_grad/Mul/xvarlen_gather_64/ps_embed_64/y*
T0
�
1gradients_1/varlen_gather_64/ps_embed_64_grad/SumSum1gradients_1/varlen_gather_64/ps_embed_64_grad/MulCgradients_1/varlen_gather_64/ps_embed_64_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
5gradients_1/varlen_gather_64/ps_embed_64_grad/ReshapeReshape1gradients_1/varlen_gather_64/ps_embed_64_grad/Sum3gradients_1/varlen_gather_64/ps_embed_64_grad/Shape*
T0*
Tshape0
u
Ggradients_1/varlen_gather_64/ps_embed_64_grad/Mul_1/strided_slice/stackConst*
dtype0*
valueB: 
w
Igradients_1/varlen_gather_64/ps_embed_64_grad/Mul_1/strided_slice/stack_1Const*
valueB:*
dtype0
w
Igradients_1/varlen_gather_64/ps_embed_64_grad/Mul_1/strided_slice/stack_2Const*
valueB:*
dtype0
�
Agradients_1/varlen_gather_64/ps_embed_64_grad/Mul_1/strided_sliceStridedSlicePgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/dense_shapeGgradients_1/varlen_gather_64/ps_embed_64_grad/Mul_1/strided_slice/stackIgradients_1/varlen_gather_64/ps_embed_64_grad/Mul_1/strided_slice/stack_1Igradients_1/varlen_gather_64/ps_embed_64_grad/Mul_1/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
�
5gradients_1/varlen_gather_64/ps_embed_64_grad/Mul_1/yUnsortedSegmentSumDgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_gradLgradients_1/input_c_id_embedding/cond/GatherV2/Switch_grad/cond_grad/indicesAgradients_1/varlen_gather_64/ps_embed_64_grad/Mul_1/strided_slice*
Tnumsegments0*
Tindices0*
T0
�
3gradients_1/varlen_gather_64/ps_embed_64_grad/Mul_1Mulvarlen_gather_64/VarlenGather5gradients_1/varlen_gather_64/ps_embed_64_grad/Mul_1/y*
T0
�
3gradients_1/varlen_gather_64/ps_embed_64_grad/Sum_1Sum3gradients_1/varlen_gather_64/ps_embed_64_grad/Mul_1Egradients_1/varlen_gather_64/ps_embed_64_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
7gradients_1/varlen_gather_64/ps_embed_64_grad/Reshape_1Reshape3gradients_1/varlen_gather_64/ps_embed_64_grad/Sum_15gradients_1/varlen_gather_64/ps_embed_64_grad/Shape_1*
T0*
Tshape0
t
3gradients_1/varlen_gather_32/ps_embed_32_grad/ShapeShapevarlen_gather_32/VarlenGather*
T0*
out_type0
^
5gradients_1/varlen_gather_32/ps_embed_32_grad/Shape_1Const*
valueB *
dtype0
�
Cgradients_1/varlen_gather_32/ps_embed_32_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients_1/varlen_gather_32/ps_embed_32_grad/Shape5gradients_1/varlen_gather_32/ps_embed_32_grad/Shape_1*
T0
s
Egradients_1/varlen_gather_32/ps_embed_32_grad/Mul/strided_slice/stackConst*
valueB: *
dtype0
u
Ggradients_1/varlen_gather_32/ps_embed_32_grad/Mul/strided_slice/stack_1Const*
valueB:*
dtype0
u
Ggradients_1/varlen_gather_32/ps_embed_32_grad/Mul/strided_slice/stack_2Const*
dtype0*
valueB:
�
?gradients_1/varlen_gather_32/ps_embed_32_grad/Mul/strided_sliceStridedSliceRgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/dense_shapeEgradients_1/varlen_gather_32/ps_embed_32_grad/Mul/strided_slice/stackGgradients_1/varlen_gather_32/ps_embed_32_grad/Mul/strided_slice/stack_1Ggradients_1/varlen_gather_32/ps_embed_32_grad/Mul/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
�
3gradients_1/varlen_gather_32/ps_embed_32_grad/Mul/xUnsortedSegmentSumFgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_gradNgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/indices?gradients_1/varlen_gather_32/ps_embed_32_grad/Mul/strided_slice*
Tnumsegments0*
Tindices0*
T0
�
1gradients_1/varlen_gather_32/ps_embed_32_grad/MulMul3gradients_1/varlen_gather_32/ps_embed_32_grad/Mul/xvarlen_gather_32/ps_embed_32/y*
T0
�
1gradients_1/varlen_gather_32/ps_embed_32_grad/SumSum1gradients_1/varlen_gather_32/ps_embed_32_grad/MulCgradients_1/varlen_gather_32/ps_embed_32_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
5gradients_1/varlen_gather_32/ps_embed_32_grad/ReshapeReshape1gradients_1/varlen_gather_32/ps_embed_32_grad/Sum3gradients_1/varlen_gather_32/ps_embed_32_grad/Shape*
T0*
Tshape0
u
Ggradients_1/varlen_gather_32/ps_embed_32_grad/Mul_1/strided_slice/stackConst*
valueB: *
dtype0
w
Igradients_1/varlen_gather_32/ps_embed_32_grad/Mul_1/strided_slice/stack_1Const*
valueB:*
dtype0
w
Igradients_1/varlen_gather_32/ps_embed_32_grad/Mul_1/strided_slice/stack_2Const*
dtype0*
valueB:
�
Agradients_1/varlen_gather_32/ps_embed_32_grad/Mul_1/strided_sliceStridedSliceRgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/dense_shapeGgradients_1/varlen_gather_32/ps_embed_32_grad/Mul_1/strided_slice/stackIgradients_1/varlen_gather_32/ps_embed_32_grad/Mul_1/strided_slice/stack_1Igradients_1/varlen_gather_32/ps_embed_32_grad/Mul_1/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
�
5gradients_1/varlen_gather_32/ps_embed_32_grad/Mul_1/yUnsortedSegmentSumFgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_gradNgradients_1/input_c_info_embedding/cond/GatherV2/Switch_grad/cond_grad/indicesAgradients_1/varlen_gather_32/ps_embed_32_grad/Mul_1/strided_slice*
Tnumsegments0*
Tindices0*
T0
�
3gradients_1/varlen_gather_32/ps_embed_32_grad/Mul_1Mulvarlen_gather_32/VarlenGather5gradients_1/varlen_gather_32/ps_embed_32_grad/Mul_1/y*
T0
�
3gradients_1/varlen_gather_32/ps_embed_32_grad/Sum_1Sum3gradients_1/varlen_gather_32/ps_embed_32_grad/Mul_1Egradients_1/varlen_gather_32/ps_embed_32_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
7gradients_1/varlen_gather_32/ps_embed_32_grad/Reshape_1Reshape3gradients_1/varlen_gather_32/ps_embed_32_grad/Sum_15gradients_1/varlen_gather_32/ps_embed_32_grad/Shape_1*
T0*
Tshape0
q
1gradients_1/varlen_gather_8/ps_embed_8_grad/ShapeShapevarlen_gather_8/VarlenGather*
T0*
out_type0
\
3gradients_1/varlen_gather_8/ps_embed_8_grad/Shape_1Const*
valueB *
dtype0
�
Agradients_1/varlen_gather_8/ps_embed_8_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients_1/varlen_gather_8/ps_embed_8_grad/Shape3gradients_1/varlen_gather_8/ps_embed_8_grad/Shape_1*
T0
q
Cgradients_1/varlen_gather_8/ps_embed_8_grad/Mul/strided_slice/stackConst*
valueB: *
dtype0
s
Egradients_1/varlen_gather_8/ps_embed_8_grad/Mul/strided_slice/stack_1Const*
valueB:*
dtype0
s
Egradients_1/varlen_gather_8/ps_embed_8_grad/Mul/strided_slice/stack_2Const*
valueB:*
dtype0
�
=gradients_1/varlen_gather_8/ps_embed_8_grad/Mul/strided_sliceStridedSliceTgradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/dense_shapeCgradients_1/varlen_gather_8/ps_embed_8_grad/Mul/strided_slice/stackEgradients_1/varlen_gather_8/ps_embed_8_grad/Mul/strided_slice/stack_1Egradients_1/varlen_gather_8/ps_embed_8_grad/Mul/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
�
1gradients_1/varlen_gather_8/ps_embed_8_grad/Mul/xUnsortedSegmentSumHgradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_gradPgradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/indices=gradients_1/varlen_gather_8/ps_embed_8_grad/Mul/strided_slice*
Tnumsegments0*
Tindices0*
T0
�
/gradients_1/varlen_gather_8/ps_embed_8_grad/MulMul1gradients_1/varlen_gather_8/ps_embed_8_grad/Mul/xvarlen_gather_8/ps_embed_8/y*
T0
�
/gradients_1/varlen_gather_8/ps_embed_8_grad/SumSum/gradients_1/varlen_gather_8/ps_embed_8_grad/MulAgradients_1/varlen_gather_8/ps_embed_8_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
3gradients_1/varlen_gather_8/ps_embed_8_grad/ReshapeReshape/gradients_1/varlen_gather_8/ps_embed_8_grad/Sum1gradients_1/varlen_gather_8/ps_embed_8_grad/Shape*
T0*
Tshape0
s
Egradients_1/varlen_gather_8/ps_embed_8_grad/Mul_1/strided_slice/stackConst*
dtype0*
valueB: 
u
Ggradients_1/varlen_gather_8/ps_embed_8_grad/Mul_1/strided_slice/stack_1Const*
dtype0*
valueB:
u
Ggradients_1/varlen_gather_8/ps_embed_8_grad/Mul_1/strided_slice/stack_2Const*
valueB:*
dtype0
�
?gradients_1/varlen_gather_8/ps_embed_8_grad/Mul_1/strided_sliceStridedSliceTgradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/dense_shapeEgradients_1/varlen_gather_8/ps_embed_8_grad/Mul_1/strided_slice/stackGgradients_1/varlen_gather_8/ps_embed_8_grad/Mul_1/strided_slice/stack_1Ggradients_1/varlen_gather_8/ps_embed_8_grad/Mul_1/strided_slice/stack_2*
end_mask *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
�
3gradients_1/varlen_gather_8/ps_embed_8_grad/Mul_1/yUnsortedSegmentSumHgradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_gradPgradients_1/input_position_embedding/cond/GatherV2/Switch_grad/cond_grad/indices?gradients_1/varlen_gather_8/ps_embed_8_grad/Mul_1/strided_slice*
Tnumsegments0*
Tindices0*
T0
�
1gradients_1/varlen_gather_8/ps_embed_8_grad/Mul_1Mulvarlen_gather_8/VarlenGather3gradients_1/varlen_gather_8/ps_embed_8_grad/Mul_1/y*
T0
�
1gradients_1/varlen_gather_8/ps_embed_8_grad/Sum_1Sum1gradients_1/varlen_gather_8/ps_embed_8_grad/Mul_1Cgradients_1/varlen_gather_8/ps_embed_8_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
5gradients_1/varlen_gather_8/ps_embed_8_grad/Reshape_1Reshape1gradients_1/varlen_gather_8/ps_embed_8_grad/Sum_13gradients_1/varlen_gather_8/ps_embed_8_grad/Shape_1*
T0*
Tshape0
U
dense_grad_merge/Reshape/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/ReshapeReshape1gradients_1/expand_xtr/dense/MatMul_grad/MatMul_1dense_grad_merge/Reshape/shape*
T0*
Tshape0
W
 dense_grad_merge/Reshape_1/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_1Reshape5gradients_1/expand_xtr/dense/BiasAdd_grad/BiasAddGrad dense_grad_merge/Reshape_1/shape*
T0*
Tshape0
W
 dense_grad_merge/Reshape_2/shapeConst*
dtype0*
valueB:
���������
�
dense_grad_merge/Reshape_2Reshape3gradients_1/expand_xtr/dense_1/MatMul_grad/MatMul_1 dense_grad_merge/Reshape_2/shape*
T0*
Tshape0
W
 dense_grad_merge/Reshape_3/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_3Reshape7gradients_1/expand_xtr/dense_1/BiasAdd_grad/BiasAddGrad dense_grad_merge/Reshape_3/shape*
T0*
Tshape0
W
 dense_grad_merge/Reshape_4/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_4Reshape3gradients_1/expand_xtr/dense_2/MatMul_grad/MatMul_1 dense_grad_merge/Reshape_4/shape*
T0*
Tshape0
W
 dense_grad_merge/Reshape_5/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_5Reshape7gradients_1/expand_xtr/dense_2/BiasAdd_grad/BiasAddGrad dense_grad_merge/Reshape_5/shape*
T0*
Tshape0
W
 dense_grad_merge/Reshape_6/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_6Reshape3gradients_1/expand_xtr/dense_3/MatMul_grad/MatMul_1 dense_grad_merge/Reshape_6/shape*
T0*
Tshape0
W
 dense_grad_merge/Reshape_7/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_7Reshape7gradients_1/expand_xtr/dense_3/BiasAdd_grad/BiasAddGrad dense_grad_merge/Reshape_7/shape*
T0*
Tshape0
W
 dense_grad_merge/Reshape_8/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_8Reshape/gradients_1/like_xtr/dense/MatMul_grad/MatMul_1 dense_grad_merge/Reshape_8/shape*
T0*
Tshape0
W
 dense_grad_merge/Reshape_9/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_9Reshape3gradients_1/like_xtr/dense/BiasAdd_grad/BiasAddGrad dense_grad_merge/Reshape_9/shape*
T0*
Tshape0
X
!dense_grad_merge/Reshape_10/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_10Reshape1gradients_1/like_xtr/dense_1/MatMul_grad/MatMul_1!dense_grad_merge/Reshape_10/shape*
T0*
Tshape0
X
!dense_grad_merge/Reshape_11/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_11Reshape5gradients_1/like_xtr/dense_1/BiasAdd_grad/BiasAddGrad!dense_grad_merge/Reshape_11/shape*
T0*
Tshape0
X
!dense_grad_merge/Reshape_12/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_12Reshape1gradients_1/like_xtr/dense_2/MatMul_grad/MatMul_1!dense_grad_merge/Reshape_12/shape*
T0*
Tshape0
X
!dense_grad_merge/Reshape_13/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_13Reshape5gradients_1/like_xtr/dense_2/BiasAdd_grad/BiasAddGrad!dense_grad_merge/Reshape_13/shape*
T0*
Tshape0
X
!dense_grad_merge/Reshape_14/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_14Reshape1gradients_1/like_xtr/dense_3/MatMul_grad/MatMul_1!dense_grad_merge/Reshape_14/shape*
T0*
Tshape0
X
!dense_grad_merge/Reshape_15/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_15Reshape5gradients_1/like_xtr/dense_3/BiasAdd_grad/BiasAddGrad!dense_grad_merge/Reshape_15/shape*
T0*
Tshape0
X
!dense_grad_merge/Reshape_16/shapeConst*
dtype0*
valueB:
���������
�
dense_grad_merge/Reshape_16Reshape0gradients_1/reply_xtr/dense/MatMul_grad/MatMul_1!dense_grad_merge/Reshape_16/shape*
T0*
Tshape0
X
!dense_grad_merge/Reshape_17/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_17Reshape4gradients_1/reply_xtr/dense/BiasAdd_grad/BiasAddGrad!dense_grad_merge/Reshape_17/shape*
T0*
Tshape0
X
!dense_grad_merge/Reshape_18/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_18Reshape2gradients_1/reply_xtr/dense_1/MatMul_grad/MatMul_1!dense_grad_merge/Reshape_18/shape*
T0*
Tshape0
X
!dense_grad_merge/Reshape_19/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_19Reshape6gradients_1/reply_xtr/dense_1/BiasAdd_grad/BiasAddGrad!dense_grad_merge/Reshape_19/shape*
T0*
Tshape0
X
!dense_grad_merge/Reshape_20/shapeConst*
dtype0*
valueB:
���������
�
dense_grad_merge/Reshape_20Reshape2gradients_1/reply_xtr/dense_2/MatMul_grad/MatMul_1!dense_grad_merge/Reshape_20/shape*
T0*
Tshape0
X
!dense_grad_merge/Reshape_21/shapeConst*
dtype0*
valueB:
���������
�
dense_grad_merge/Reshape_21Reshape6gradients_1/reply_xtr/dense_2/BiasAdd_grad/BiasAddGrad!dense_grad_merge/Reshape_21/shape*
T0*
Tshape0
X
!dense_grad_merge/Reshape_22/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_22Reshape2gradients_1/reply_xtr/dense_3/MatMul_grad/MatMul_1!dense_grad_merge/Reshape_22/shape*
T0*
Tshape0
X
!dense_grad_merge/Reshape_23/shapeConst*
dtype0*
valueB:
���������
�
dense_grad_merge/Reshape_23Reshape6gradients_1/reply_xtr/dense_3/BiasAdd_grad/BiasAddGrad!dense_grad_merge/Reshape_23/shape*
T0*
Tshape0
`
+dense_grad_merge/zeros_like/shape_as_tensorConst*
valueB"P     *
dtype0
N
!dense_grad_merge/zeros_like/ConstConst*
dtype0*
valueB
 *    
�
dense_grad_merge/zeros_likeFill+dense_grad_merge/zeros_like/shape_as_tensor!dense_grad_merge/zeros_like/Const*
T0*

index_type0
X
!dense_grad_merge/Reshape_24/shapeConst*
valueB:
���������*
dtype0
}
dense_grad_merge/Reshape_24Reshapedense_grad_merge/zeros_like!dense_grad_merge/Reshape_24/shape*
T0*
Tshape0
O
dense_grad_merge/zeros_like_1Const*
valueB�*    *
dtype0
X
!dense_grad_merge/Reshape_25/shapeConst*
dtype0*
valueB:
���������

dense_grad_merge/Reshape_25Reshapedense_grad_merge/zeros_like_1!dense_grad_merge/Reshape_25/shape*
T0*
Tshape0
b
-dense_grad_merge/zeros_like_2/shape_as_tensorConst*
valueB"   �   *
dtype0
P
#dense_grad_merge/zeros_like_2/ConstConst*
dtype0*
valueB
 *    
�
dense_grad_merge/zeros_like_2Fill-dense_grad_merge/zeros_like_2/shape_as_tensor#dense_grad_merge/zeros_like_2/Const*
T0*

index_type0
X
!dense_grad_merge/Reshape_26/shapeConst*
valueB:
���������*
dtype0

dense_grad_merge/Reshape_26Reshapedense_grad_merge/zeros_like_2!dense_grad_merge/Reshape_26/shape*
T0*
Tshape0
O
dense_grad_merge/zeros_like_3Const*
valueB�*    *
dtype0
X
!dense_grad_merge/Reshape_27/shapeConst*
valueB:
���������*
dtype0

dense_grad_merge/Reshape_27Reshapedense_grad_merge/zeros_like_3!dense_grad_merge/Reshape_27/shape*
T0*
Tshape0
b
-dense_grad_merge/zeros_like_4/shape_as_tensorConst*
valueB"�   @   *
dtype0
P
#dense_grad_merge/zeros_like_4/ConstConst*
dtype0*
valueB
 *    
�
dense_grad_merge/zeros_like_4Fill-dense_grad_merge/zeros_like_4/shape_as_tensor#dense_grad_merge/zeros_like_4/Const*
T0*

index_type0
X
!dense_grad_merge/Reshape_28/shapeConst*
valueB:
���������*
dtype0

dense_grad_merge/Reshape_28Reshapedense_grad_merge/zeros_like_4!dense_grad_merge/Reshape_28/shape*
T0*
Tshape0
N
dense_grad_merge/zeros_like_5Const*
valueB@*    *
dtype0
X
!dense_grad_merge/Reshape_29/shapeConst*
dtype0*
valueB:
���������

dense_grad_merge/Reshape_29Reshapedense_grad_merge/zeros_like_5!dense_grad_merge/Reshape_29/shape*
T0*
Tshape0
R
dense_grad_merge/zeros_like_6Const*
valueB@*    *
dtype0
X
!dense_grad_merge/Reshape_30/shapeConst*
valueB:
���������*
dtype0

dense_grad_merge/Reshape_30Reshapedense_grad_merge/zeros_like_6!dense_grad_merge/Reshape_30/shape*
T0*
Tshape0
N
dense_grad_merge/zeros_like_7Const*
valueB*    *
dtype0
X
!dense_grad_merge/Reshape_31/shapeConst*
valueB:
���������*
dtype0

dense_grad_merge/Reshape_31Reshapedense_grad_merge/zeros_like_7!dense_grad_merge/Reshape_31/shape*
T0*
Tshape0
b
-dense_grad_merge/zeros_like_8/shape_as_tensorConst*
dtype0*
valueB"P     
P
#dense_grad_merge/zeros_like_8/ConstConst*
valueB
 *    *
dtype0
�
dense_grad_merge/zeros_like_8Fill-dense_grad_merge/zeros_like_8/shape_as_tensor#dense_grad_merge/zeros_like_8/Const*
T0*

index_type0
X
!dense_grad_merge/Reshape_32/shapeConst*
valueB:
���������*
dtype0

dense_grad_merge/Reshape_32Reshapedense_grad_merge/zeros_like_8!dense_grad_merge/Reshape_32/shape*
T0*
Tshape0
O
dense_grad_merge/zeros_like_9Const*
valueB�*    *
dtype0
X
!dense_grad_merge/Reshape_33/shapeConst*
valueB:
���������*
dtype0

dense_grad_merge/Reshape_33Reshapedense_grad_merge/zeros_like_9!dense_grad_merge/Reshape_33/shape*
T0*
Tshape0
c
.dense_grad_merge/zeros_like_10/shape_as_tensorConst*
valueB"   �   *
dtype0
Q
$dense_grad_merge/zeros_like_10/ConstConst*
valueB
 *    *
dtype0
�
dense_grad_merge/zeros_like_10Fill.dense_grad_merge/zeros_like_10/shape_as_tensor$dense_grad_merge/zeros_like_10/Const*
T0*

index_type0
X
!dense_grad_merge/Reshape_34/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_34Reshapedense_grad_merge/zeros_like_10!dense_grad_merge/Reshape_34/shape*
T0*
Tshape0
P
dense_grad_merge/zeros_like_11Const*
valueB�*    *
dtype0
X
!dense_grad_merge/Reshape_35/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_35Reshapedense_grad_merge/zeros_like_11!dense_grad_merge/Reshape_35/shape*
T0*
Tshape0
c
.dense_grad_merge/zeros_like_12/shape_as_tensorConst*
valueB"�   @   *
dtype0
Q
$dense_grad_merge/zeros_like_12/ConstConst*
valueB
 *    *
dtype0
�
dense_grad_merge/zeros_like_12Fill.dense_grad_merge/zeros_like_12/shape_as_tensor$dense_grad_merge/zeros_like_12/Const*
T0*

index_type0
X
!dense_grad_merge/Reshape_36/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_36Reshapedense_grad_merge/zeros_like_12!dense_grad_merge/Reshape_36/shape*
T0*
Tshape0
O
dense_grad_merge/zeros_like_13Const*
dtype0*
valueB@*    
X
!dense_grad_merge/Reshape_37/shapeConst*
dtype0*
valueB:
���������
�
dense_grad_merge/Reshape_37Reshapedense_grad_merge/zeros_like_13!dense_grad_merge/Reshape_37/shape*
T0*
Tshape0
S
dense_grad_merge/zeros_like_14Const*
valueB@*    *
dtype0
X
!dense_grad_merge/Reshape_38/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_38Reshapedense_grad_merge/zeros_like_14!dense_grad_merge/Reshape_38/shape*
T0*
Tshape0
O
dense_grad_merge/zeros_like_15Const*
valueB*    *
dtype0
X
!dense_grad_merge/Reshape_39/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_39Reshapedense_grad_merge/zeros_like_15!dense_grad_merge/Reshape_39/shape*
T0*
Tshape0
c
.dense_grad_merge/zeros_like_16/shape_as_tensorConst*
valueB"P     *
dtype0
Q
$dense_grad_merge/zeros_like_16/ConstConst*
dtype0*
valueB
 *    
�
dense_grad_merge/zeros_like_16Fill.dense_grad_merge/zeros_like_16/shape_as_tensor$dense_grad_merge/zeros_like_16/Const*
T0*

index_type0
X
!dense_grad_merge/Reshape_40/shapeConst*
dtype0*
valueB:
���������
�
dense_grad_merge/Reshape_40Reshapedense_grad_merge/zeros_like_16!dense_grad_merge/Reshape_40/shape*
T0*
Tshape0
P
dense_grad_merge/zeros_like_17Const*
valueB�*    *
dtype0
X
!dense_grad_merge/Reshape_41/shapeConst*
dtype0*
valueB:
���������
�
dense_grad_merge/Reshape_41Reshapedense_grad_merge/zeros_like_17!dense_grad_merge/Reshape_41/shape*
T0*
Tshape0
c
.dense_grad_merge/zeros_like_18/shape_as_tensorConst*
valueB"   �   *
dtype0
Q
$dense_grad_merge/zeros_like_18/ConstConst*
valueB
 *    *
dtype0
�
dense_grad_merge/zeros_like_18Fill.dense_grad_merge/zeros_like_18/shape_as_tensor$dense_grad_merge/zeros_like_18/Const*
T0*

index_type0
X
!dense_grad_merge/Reshape_42/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_42Reshapedense_grad_merge/zeros_like_18!dense_grad_merge/Reshape_42/shape*
T0*
Tshape0
P
dense_grad_merge/zeros_like_19Const*
valueB�*    *
dtype0
X
!dense_grad_merge/Reshape_43/shapeConst*
dtype0*
valueB:
���������
�
dense_grad_merge/Reshape_43Reshapedense_grad_merge/zeros_like_19!dense_grad_merge/Reshape_43/shape*
T0*
Tshape0
c
.dense_grad_merge/zeros_like_20/shape_as_tensorConst*
dtype0*
valueB"�   @   
Q
$dense_grad_merge/zeros_like_20/ConstConst*
dtype0*
valueB
 *    
�
dense_grad_merge/zeros_like_20Fill.dense_grad_merge/zeros_like_20/shape_as_tensor$dense_grad_merge/zeros_like_20/Const*
T0*

index_type0
X
!dense_grad_merge/Reshape_44/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_44Reshapedense_grad_merge/zeros_like_20!dense_grad_merge/Reshape_44/shape*
T0*
Tshape0
O
dense_grad_merge/zeros_like_21Const*
valueB@*    *
dtype0
X
!dense_grad_merge/Reshape_45/shapeConst*
dtype0*
valueB:
���������
�
dense_grad_merge/Reshape_45Reshapedense_grad_merge/zeros_like_21!dense_grad_merge/Reshape_45/shape*
T0*
Tshape0
S
dense_grad_merge/zeros_like_22Const*
valueB@*    *
dtype0
X
!dense_grad_merge/Reshape_46/shapeConst*
valueB:
���������*
dtype0
�
dense_grad_merge/Reshape_46Reshapedense_grad_merge/zeros_like_22!dense_grad_merge/Reshape_46/shape*
T0*
Tshape0
O
dense_grad_merge/zeros_like_23Const*
valueB*    *
dtype0
X
!dense_grad_merge/Reshape_47/shapeConst*
dtype0*
valueB:
���������
�
dense_grad_merge/Reshape_47Reshapedense_grad_merge/zeros_like_23!dense_grad_merge/Reshape_47/shape*
T0*
Tshape0
F
dense_grad_merge/concat/axisConst*
value	B : *
dtype0
�
dense_grad_merge/concatConcatV2dense_grad_merge/Reshapedense_grad_merge/Reshape_1dense_grad_merge/Reshape_2dense_grad_merge/Reshape_3dense_grad_merge/Reshape_4dense_grad_merge/Reshape_5dense_grad_merge/Reshape_6dense_grad_merge/Reshape_7dense_grad_merge/Reshape_8dense_grad_merge/Reshape_9dense_grad_merge/Reshape_10dense_grad_merge/Reshape_11dense_grad_merge/Reshape_12dense_grad_merge/Reshape_13dense_grad_merge/Reshape_14dense_grad_merge/Reshape_15dense_grad_merge/Reshape_16dense_grad_merge/Reshape_17dense_grad_merge/Reshape_18dense_grad_merge/Reshape_19dense_grad_merge/Reshape_20dense_grad_merge/Reshape_21dense_grad_merge/Reshape_22dense_grad_merge/Reshape_23dense_grad_merge/Reshape_24dense_grad_merge/Reshape_25dense_grad_merge/Reshape_26dense_grad_merge/Reshape_27dense_grad_merge/Reshape_28dense_grad_merge/Reshape_29dense_grad_merge/Reshape_30dense_grad_merge/Reshape_31dense_grad_merge/Reshape_32dense_grad_merge/Reshape_33dense_grad_merge/Reshape_34dense_grad_merge/Reshape_35dense_grad_merge/Reshape_36dense_grad_merge/Reshape_37dense_grad_merge/Reshape_38dense_grad_merge/Reshape_39dense_grad_merge/Reshape_40dense_grad_merge/Reshape_41dense_grad_merge/Reshape_42dense_grad_merge/Reshape_43dense_grad_merge/Reshape_44dense_grad_merge/Reshape_45dense_grad_merge/Reshape_46dense_grad_merge/Reshape_47dense_grad_merge/concat/axis*

Tidx0*
T0*
N0
�
VarlenScatterVarlenScattervarlen_embed3gradients_1/varlen_gather_4/ps_embed_4_grad/Reshape3gradients_1/varlen_gather_8/ps_embed_8_grad/Reshape5gradients_1/varlen_gather_32/ps_embed_32_grad/Reshape5gradients_1/varlen_gather_64/ps_embed_64_grad/Reshapelevel1_id_4level1_id_8level1_id_32level1_id_64varlen_embed_offset"/device:GPU:0*
Tindices0*
Tparams0*
N
7
mul_1Muldense_grad_merge/concattruediv*
T0
/
mul_2MulVarlenScatter	truediv_1*
T0
@
dense_init_val_for_fetchIdentitydense_init/concat*
T0
0
dense_grad_for_fetchIdentitymul_1*
T0
1
sparse_grad_for_fetchIdentitymul_2*
T0
E
Shape_1Shapeexpand_xtr/dense_3/Sigmoid*
T0*
out_type0
C
strided_slice_5/stackConst*
dtype0*
valueB: 
E
strided_slice_5/stack_1Const*
dtype0*
valueB:
E
strided_slice_5/stack_2Const*
dtype0*
valueB:
�
strided_slice_5StridedSliceShape_1strided_slice_5/stackstrided_slice_5/stack_1strided_slice_5/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0
C
Shape_2Shapelike_xtr/dense_3/Sigmoid*
T0*
out_type0
C
strided_slice_6/stackConst*
dtype0*
valueB: 
E
strided_slice_6/stack_1Const*
valueB:*
dtype0
E
strided_slice_6/stack_2Const*
dtype0*
valueB:
�
strided_slice_6StridedSliceShape_2strided_slice_6/stackstrided_slice_6/stack_1strided_slice_6/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
D
Shape_3Shapereply_xtr/dense_3/Sigmoid*
T0*
out_type0
C
strided_slice_7/stackConst*
valueB: *
dtype0
E
strided_slice_7/stack_1Const*
valueB:*
dtype0
E
strided_slice_7/stack_2Const*
valueB:*
dtype0
�
strided_slice_7StridedSliceShape_3strided_slice_7/stackstrided_slice_7/stack_1strided_slice_7/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask "