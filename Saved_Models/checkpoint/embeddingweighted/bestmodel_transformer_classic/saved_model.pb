??!
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
b
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.1-0-g85c8b2a817f8٫
{
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@* 
shared_namedense_11/kernel
t
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes
:	?@*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:@*
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:@@*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:@*
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:@*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:*
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
d
momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
momentum
]
momentum/Read/ReadVariableOpReadVariableOpmomentum*
_output_shapes
: *
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
?
5token_and_position_embedding_1/embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75token_and_position_embedding_1/embedding_2/embeddings
?
Itoken_and_position_embedding_1/embedding_2/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_1/embedding_2/embeddings*
_output_shapes

: *
dtype0
?
5token_and_position_embedding_1/embedding_3/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?R *F
shared_name75token_and_position_embedding_1/embedding_3/embeddings
?
Itoken_and_position_embedding_1/embedding_3/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_1/embedding_3/embeddings*
_output_shapes
:	?R *
dtype0
?
7transformer_block_3/multi_head_attention_3/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_3/multi_head_attention_3/query/kernel
?
Ktransformer_block_3/multi_head_attention_3/query/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_3/multi_head_attention_3/query/kernel*"
_output_shapes
:  *
dtype0
?
5transformer_block_3/multi_head_attention_3/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_3/multi_head_attention_3/query/bias
?
Itransformer_block_3/multi_head_attention_3/query/bias/Read/ReadVariableOpReadVariableOp5transformer_block_3/multi_head_attention_3/query/bias*
_output_shapes

: *
dtype0
?
5transformer_block_3/multi_head_attention_3/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *F
shared_name75transformer_block_3/multi_head_attention_3/key/kernel
?
Itransformer_block_3/multi_head_attention_3/key/kernel/Read/ReadVariableOpReadVariableOp5transformer_block_3/multi_head_attention_3/key/kernel*"
_output_shapes
:  *
dtype0
?
3transformer_block_3/multi_head_attention_3/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *D
shared_name53transformer_block_3/multi_head_attention_3/key/bias
?
Gtransformer_block_3/multi_head_attention_3/key/bias/Read/ReadVariableOpReadVariableOp3transformer_block_3/multi_head_attention_3/key/bias*
_output_shapes

: *
dtype0
?
7transformer_block_3/multi_head_attention_3/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_3/multi_head_attention_3/value/kernel
?
Ktransformer_block_3/multi_head_attention_3/value/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_3/multi_head_attention_3/value/kernel*"
_output_shapes
:  *
dtype0
?
5transformer_block_3/multi_head_attention_3/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_3/multi_head_attention_3/value/bias
?
Itransformer_block_3/multi_head_attention_3/value/bias/Read/ReadVariableOpReadVariableOp5transformer_block_3/multi_head_attention_3/value/bias*
_output_shapes

: *
dtype0
?
Btransformer_block_3/multi_head_attention_3/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBtransformer_block_3/multi_head_attention_3/attention_output/kernel
?
Vtransformer_block_3/multi_head_attention_3/attention_output/kernel/Read/ReadVariableOpReadVariableOpBtransformer_block_3/multi_head_attention_3/attention_output/kernel*"
_output_shapes
:  *
dtype0
?
@transformer_block_3/multi_head_attention_3/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@transformer_block_3/multi_head_attention_3/attention_output/bias
?
Ttransformer_block_3/multi_head_attention_3/attention_output/bias/Read/ReadVariableOpReadVariableOp@transformer_block_3/multi_head_attention_3/attention_output/bias*
_output_shapes
: *
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

: @*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:@*
dtype0
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:@ *
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
: *
dtype0
?
/transformer_block_3/layer_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_3/layer_normalization_6/gamma
?
Ctransformer_block_3/layer_normalization_6/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_3/layer_normalization_6/gamma*
_output_shapes
: *
dtype0
?
.transformer_block_3/layer_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.transformer_block_3/layer_normalization_6/beta
?
Btransformer_block_3/layer_normalization_6/beta/Read/ReadVariableOpReadVariableOp.transformer_block_3/layer_normalization_6/beta*
_output_shapes
: *
dtype0
?
/transformer_block_3/layer_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_3/layer_normalization_7/gamma
?
Ctransformer_block_3/layer_normalization_7/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_3/layer_normalization_7/gamma*
_output_shapes
: *
dtype0
?
.transformer_block_3/layer_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.transformer_block_3/layer_normalization_7/beta
?
Btransformer_block_3/layer_normalization_7/beta/Read/ReadVariableOpReadVariableOp.transformer_block_3/layer_normalization_7/beta*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
SGD/dense_11/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*-
shared_nameSGD/dense_11/kernel/momentum
?
0SGD/dense_11/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_11/kernel/momentum*
_output_shapes
:	?@*
dtype0
?
SGD/dense_11/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_11/bias/momentum
?
.SGD/dense_11/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_11/bias/momentum*
_output_shapes
:@*
dtype0
?
SGD/dense_12/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*-
shared_nameSGD/dense_12/kernel/momentum
?
0SGD/dense_12/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_12/kernel/momentum*
_output_shapes

:@@*
dtype0
?
SGD/dense_12/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_12/bias/momentum
?
.SGD/dense_12/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_12/bias/momentum*
_output_shapes
:@*
dtype0
?
SGD/dense_13/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*-
shared_nameSGD/dense_13/kernel/momentum
?
0SGD/dense_13/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_13/kernel/momentum*
_output_shapes

:@*
dtype0
?
SGD/dense_13/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/dense_13/bias/momentum
?
.SGD/dense_13/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_13/bias/momentum*
_output_shapes
:*
dtype0
?
BSGD/token_and_position_embedding_1/embedding_2/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/token_and_position_embedding_1/embedding_2/embeddings/momentum
?
VSGD/token_and_position_embedding_1/embedding_2/embeddings/momentum/Read/ReadVariableOpReadVariableOpBSGD/token_and_position_embedding_1/embedding_2/embeddings/momentum*
_output_shapes

: *
dtype0
?
BSGD/token_and_position_embedding_1/embedding_3/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?R *S
shared_nameDBSGD/token_and_position_embedding_1/embedding_3/embeddings/momentum
?
VSGD/token_and_position_embedding_1/embedding_3/embeddings/momentum/Read/ReadVariableOpReadVariableOpBSGD/token_and_position_embedding_1/embedding_3/embeddings/momentum*
_output_shapes
:	?R *
dtype0
?
DSGD/transformer_block_3/multi_head_attention_3/query/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_3/multi_head_attention_3/query/kernel/momentum
?
XSGD/transformer_block_3/multi_head_attention_3/query/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_3/multi_head_attention_3/query/kernel/momentum*"
_output_shapes
:  *
dtype0
?
BSGD/transformer_block_3/multi_head_attention_3/query/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_3/multi_head_attention_3/query/bias/momentum
?
VSGD/transformer_block_3/multi_head_attention_3/query/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_3/multi_head_attention_3/query/bias/momentum*
_output_shapes

: *
dtype0
?
BSGD/transformer_block_3/multi_head_attention_3/key/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBSGD/transformer_block_3/multi_head_attention_3/key/kernel/momentum
?
VSGD/transformer_block_3/multi_head_attention_3/key/kernel/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_3/multi_head_attention_3/key/kernel/momentum*"
_output_shapes
:  *
dtype0
?
@SGD/transformer_block_3/multi_head_attention_3/key/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *Q
shared_nameB@SGD/transformer_block_3/multi_head_attention_3/key/bias/momentum
?
TSGD/transformer_block_3/multi_head_attention_3/key/bias/momentum/Read/ReadVariableOpReadVariableOp@SGD/transformer_block_3/multi_head_attention_3/key/bias/momentum*
_output_shapes

: *
dtype0
?
DSGD/transformer_block_3/multi_head_attention_3/value/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_3/multi_head_attention_3/value/kernel/momentum
?
XSGD/transformer_block_3/multi_head_attention_3/value/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_3/multi_head_attention_3/value/kernel/momentum*"
_output_shapes
:  *
dtype0
?
BSGD/transformer_block_3/multi_head_attention_3/value/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_3/multi_head_attention_3/value/bias/momentum
?
VSGD/transformer_block_3/multi_head_attention_3/value/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_3/multi_head_attention_3/value/bias/momentum*
_output_shapes

: *
dtype0
?
OSGD/transformer_block_3/multi_head_attention_3/attention_output/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *`
shared_nameQOSGD/transformer_block_3/multi_head_attention_3/attention_output/kernel/momentum
?
cSGD/transformer_block_3/multi_head_attention_3/attention_output/kernel/momentum/Read/ReadVariableOpReadVariableOpOSGD/transformer_block_3/multi_head_attention_3/attention_output/kernel/momentum*"
_output_shapes
:  *
dtype0
?
MSGD/transformer_block_3/multi_head_attention_3/attention_output/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *^
shared_nameOMSGD/transformer_block_3/multi_head_attention_3/attention_output/bias/momentum
?
aSGD/transformer_block_3/multi_head_attention_3/attention_output/bias/momentum/Read/ReadVariableOpReadVariableOpMSGD/transformer_block_3/multi_head_attention_3/attention_output/bias/momentum*
_output_shapes
: *
dtype0
?
SGD/dense_9/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*,
shared_nameSGD/dense_9/kernel/momentum
?
/SGD/dense_9/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_9/kernel/momentum*
_output_shapes

: @*
dtype0
?
SGD/dense_9/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameSGD/dense_9/bias/momentum
?
-SGD/dense_9/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_9/bias/momentum*
_output_shapes
:@*
dtype0
?
SGD/dense_10/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *-
shared_nameSGD/dense_10/kernel/momentum
?
0SGD/dense_10/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_10/kernel/momentum*
_output_shapes

:@ *
dtype0
?
SGD/dense_10/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/dense_10/bias/momentum
?
.SGD/dense_10/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_10/bias/momentum*
_output_shapes
: *
dtype0
?
<SGD/transformer_block_3/layer_normalization_6/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_3/layer_normalization_6/gamma/momentum
?
PSGD/transformer_block_3/layer_normalization_6/gamma/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_3/layer_normalization_6/gamma/momentum*
_output_shapes
: *
dtype0
?
;SGD/transformer_block_3/layer_normalization_6/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;SGD/transformer_block_3/layer_normalization_6/beta/momentum
?
OSGD/transformer_block_3/layer_normalization_6/beta/momentum/Read/ReadVariableOpReadVariableOp;SGD/transformer_block_3/layer_normalization_6/beta/momentum*
_output_shapes
: *
dtype0
?
<SGD/transformer_block_3/layer_normalization_7/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_3/layer_normalization_7/gamma/momentum
?
PSGD/transformer_block_3/layer_normalization_7/gamma/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_3/layer_normalization_7/gamma/momentum*
_output_shapes
: *
dtype0
?
;SGD/transformer_block_3/layer_normalization_7/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;SGD/transformer_block_3/layer_normalization_7/beta/momentum
?
OSGD/transformer_block_3/layer_normalization_7/beta/momentum/Read/ReadVariableOpReadVariableOp;SGD/transformer_block_3/layer_normalization_7/beta/momentum*
_output_shapes
: *
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ل
value΄Bʄ B
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
n
	token_emb
pos_emb
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
?
att
ffn

layernorm1

layernorm2
dropout1
 dropout2
!regularization_losses
"	variables
#trainable_variables
$	keras_api
R
%regularization_losses
&	variables
'trainable_variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
R
/regularization_losses
0	variables
1trainable_variables
2	keras_api
h

3kernel
4bias
5regularization_losses
6	variables
7trainable_variables
8	keras_api
R
9regularization_losses
:	variables
;trainable_variables
<	keras_api
h

=kernel
>bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
?
	Cdecay
Dlearning_rate
Emomentum
Fiter)momentum?*momentum?3momentum?4momentum?=momentum?>momentum?Gmomentum?Hmomentum?Imomentum?Jmomentum?Kmomentum?Lmomentum?Mmomentum?Nmomentum?Omomentum?Pmomentum?Qmomentum?Rmomentum?Smomentum?Tmomentum?Umomentum?Vmomentum?Wmomentum?Xmomentum?
 
?
G0
H1
I2
J3
K4
L5
M6
N7
O8
P9
Q10
R11
S12
T13
U14
V15
W16
X17
)18
*19
320
421
=22
>23
?
G0
H1
I2
J3
K4
L5
M6
N7
O8
P9
Q10
R11
S12
T13
U14
V15
W16
X17
)18
*19
320
421
=22
>23
?
Ynon_trainable_variables
Zmetrics
regularization_losses
	variables

[layers
\layer_regularization_losses
]layer_metrics
trainable_variables
 
b
G
embeddings
^regularization_losses
_	variables
`trainable_variables
a	keras_api
b
H
embeddings
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
 

G0
H1

G0
H1
?
fnon_trainable_variables
gmetrics
regularization_losses
	variables

hlayers
ilayer_regularization_losses
jlayer_metrics
trainable_variables
 
 
 
?
knon_trainable_variables
lmetrics
regularization_losses
	variables

mlayers
nlayer_regularization_losses
olayer_metrics
trainable_variables
?
p_query_dense
q
_key_dense
r_value_dense
s_softmax
t_dropout_layer
u_output_dense
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
?
zlayer_with_weights-0
zlayer-0
{layer_with_weights-1
{layer-1
|regularization_losses
}	variables
~trainable_variables
	keras_api
v
	?axis
	Ugamma
Vbeta
?regularization_losses
?	variables
?trainable_variables
?	keras_api
v
	?axis
	Wgamma
Xbeta
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 
v
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15
v
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15
?
?non_trainable_variables
?metrics
!regularization_losses
"	variables
?layers
 ?layer_regularization_losses
?layer_metrics
#trainable_variables
 
 
 
?
?non_trainable_variables
?metrics
%regularization_losses
&	variables
?layers
 ?layer_regularization_losses
?layer_metrics
'trainable_variables
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_11/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
?
?non_trainable_variables
?metrics
+regularization_losses
,	variables
?layers
 ?layer_regularization_losses
?layer_metrics
-trainable_variables
 
 
 
?
?non_trainable_variables
?metrics
/regularization_losses
0	variables
?layers
 ?layer_regularization_losses
?layer_metrics
1trainable_variables
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

30
41
?
?non_trainable_variables
?metrics
5regularization_losses
6	variables
?layers
 ?layer_regularization_losses
?layer_metrics
7trainable_variables
 
 
 
?
?non_trainable_variables
?metrics
9regularization_losses
:	variables
?layers
 ?layer_regularization_losses
?layer_metrics
;trainable_variables
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1

=0
>1
?
?non_trainable_variables
?metrics
?regularization_losses
@	variables
?layers
 ?layer_regularization_losses
?layer_metrics
Atrainable_variables
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5token_and_position_embedding_1/embedding_2/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5token_and_position_embedding_1/embedding_3/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE7transformer_block_3/multi_head_attention_3/query/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5transformer_block_3/multi_head_attention_3/query/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5transformer_block_3/multi_head_attention_3/key/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3transformer_block_3/multi_head_attention_3/key/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE7transformer_block_3/multi_head_attention_3/value/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5transformer_block_3/multi_head_attention_3/value/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEBtransformer_block_3/multi_head_attention_3/attention_output/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE@transformer_block_3/multi_head_attention_3/attention_output/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_9/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_9/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_10/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_10/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_3/layer_normalization_6/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.transformer_block_3/layer_normalization_6/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_3/layer_normalization_7/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.transformer_block_3/layer_normalization_7/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE
 

?0
F
0
1
2
3
4
5
6
7
	8

9
 
 
 

G0

G0
?
?non_trainable_variables
?metrics
^regularization_losses
_	variables
?layers
 ?layer_regularization_losses
?layer_metrics
`trainable_variables
 

H0

H0
?
?non_trainable_variables
?metrics
bregularization_losses
c	variables
?layers
 ?layer_regularization_losses
?layer_metrics
dtrainable_variables
 
 

0
1
 
 
 
 
 
 
 
?
?partial_output_shape
?full_output_shape

Ikernel
Jbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
?partial_output_shape
?full_output_shape

Kkernel
Lbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
?partial_output_shape
?full_output_shape

Mkernel
Nbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
?partial_output_shape
?full_output_shape

Okernel
Pbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 
8
I0
J1
K2
L3
M4
N5
O6
P7
8
I0
J1
K2
L3
M4
N5
O6
P7
?
?non_trainable_variables
?metrics
vregularization_losses
w	variables
?layers
 ?layer_regularization_losses
?layer_metrics
xtrainable_variables
l

Qkernel
Rbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
l

Skernel
Tbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 

Q0
R1
S2
T3

Q0
R1
S2
T3
?
?non_trainable_variables
?metrics
|regularization_losses
}	variables
?layers
 ?layer_regularization_losses
?layer_metrics
~trainable_variables
 
 

U0
V1

U0
V1
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
 
 

W0
X1

W0
X1
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
 
 
 
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
 
 
 
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
 
 
*
0
1
2
3
4
 5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 

I0
J1

I0
J1
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
 
 
 

K0
L1

K0
L1
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
 
 
 

M0
N1

M0
N1
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
 
 
 
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
 
 
 
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
 
 
 

O0
P1

O0
P1
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
 
 
*
p0
q1
r2
s3
t4
u5
 
 
 

Q0
R1

Q0
R1
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
 

S0
T1

S0
T1
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
 
 

z0
{1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
??
VARIABLE_VALUESGD/dense_11/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_11/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_12/kernel/momentumYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_12/bias/momentumWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_13/kernel/momentumYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_13/bias/momentumWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBSGD/token_and_position_embedding_1/embedding_2/embeddings/momentumIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBSGD/token_and_position_embedding_1/embedding_3/embeddings/momentumIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDSGD/transformer_block_3/multi_head_attention_3/query/kernel/momentumIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBSGD/transformer_block_3/multi_head_attention_3/query/bias/momentumIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBSGD/transformer_block_3/multi_head_attention_3/key/kernel/momentumIvariables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@SGD/transformer_block_3/multi_head_attention_3/key/bias/momentumIvariables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDSGD/transformer_block_3/multi_head_attention_3/value/kernel/momentumIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBSGD/transformer_block_3/multi_head_attention_3/value/bias/momentumIvariables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEOSGD/transformer_block_3/multi_head_attention_3/attention_output/kernel/momentumIvariables/8/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEMSGD/transformer_block_3/multi_head_attention_3/attention_output/bias/momentumIvariables/9/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUESGD/dense_9/kernel/momentumJvariables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUESGD/dense_9/bias/momentumJvariables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUESGD/dense_10/kernel/momentumJvariables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUESGD/dense_10/bias/momentumJvariables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE<SGD/transformer_block_3/layer_normalization_6/gamma/momentumJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE;SGD/transformer_block_3/layer_normalization_6/beta/momentumJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE<SGD/transformer_block_3/layer_normalization_7/gamma/momentumJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE;SGD/transformer_block_3/layer_normalization_7/beta/momentumJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_2Placeholder*(
_output_shapes
:??????????R*
dtype0*
shape:??????????R
?

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_25token_and_position_embedding_1/embedding_3/embeddings5token_and_position_embedding_1/embedding_2/embeddings7transformer_block_3/multi_head_attention_3/query/kernel5transformer_block_3/multi_head_attention_3/query/bias5transformer_block_3/multi_head_attention_3/key/kernel3transformer_block_3/multi_head_attention_3/key/bias7transformer_block_3/multi_head_attention_3/value/kernel5transformer_block_3/multi_head_attention_3/value/biasBtransformer_block_3/multi_head_attention_3/attention_output/kernel@transformer_block_3/multi_head_attention_3/attention_output/bias/transformer_block_3/layer_normalization_6/gamma.transformer_block_3/layer_normalization_6/betadense_9/kerneldense_9/biasdense_10/kerneldense_10/bias/transformer_block_3/layer_normalization_7/gamma.transformer_block_3/layer_normalization_7/betadense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_13224
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpItoken_and_position_embedding_1/embedding_2/embeddings/Read/ReadVariableOpItoken_and_position_embedding_1/embedding_3/embeddings/Read/ReadVariableOpKtransformer_block_3/multi_head_attention_3/query/kernel/Read/ReadVariableOpItransformer_block_3/multi_head_attention_3/query/bias/Read/ReadVariableOpItransformer_block_3/multi_head_attention_3/key/kernel/Read/ReadVariableOpGtransformer_block_3/multi_head_attention_3/key/bias/Read/ReadVariableOpKtransformer_block_3/multi_head_attention_3/value/kernel/Read/ReadVariableOpItransformer_block_3/multi_head_attention_3/value/bias/Read/ReadVariableOpVtransformer_block_3/multi_head_attention_3/attention_output/kernel/Read/ReadVariableOpTtransformer_block_3/multi_head_attention_3/attention_output/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOpCtransformer_block_3/layer_normalization_6/gamma/Read/ReadVariableOpBtransformer_block_3/layer_normalization_6/beta/Read/ReadVariableOpCtransformer_block_3/layer_normalization_7/gamma/Read/ReadVariableOpBtransformer_block_3/layer_normalization_7/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp0SGD/dense_11/kernel/momentum/Read/ReadVariableOp.SGD/dense_11/bias/momentum/Read/ReadVariableOp0SGD/dense_12/kernel/momentum/Read/ReadVariableOp.SGD/dense_12/bias/momentum/Read/ReadVariableOp0SGD/dense_13/kernel/momentum/Read/ReadVariableOp.SGD/dense_13/bias/momentum/Read/ReadVariableOpVSGD/token_and_position_embedding_1/embedding_2/embeddings/momentum/Read/ReadVariableOpVSGD/token_and_position_embedding_1/embedding_3/embeddings/momentum/Read/ReadVariableOpXSGD/transformer_block_3/multi_head_attention_3/query/kernel/momentum/Read/ReadVariableOpVSGD/transformer_block_3/multi_head_attention_3/query/bias/momentum/Read/ReadVariableOpVSGD/transformer_block_3/multi_head_attention_3/key/kernel/momentum/Read/ReadVariableOpTSGD/transformer_block_3/multi_head_attention_3/key/bias/momentum/Read/ReadVariableOpXSGD/transformer_block_3/multi_head_attention_3/value/kernel/momentum/Read/ReadVariableOpVSGD/transformer_block_3/multi_head_attention_3/value/bias/momentum/Read/ReadVariableOpcSGD/transformer_block_3/multi_head_attention_3/attention_output/kernel/momentum/Read/ReadVariableOpaSGD/transformer_block_3/multi_head_attention_3/attention_output/bias/momentum/Read/ReadVariableOp/SGD/dense_9/kernel/momentum/Read/ReadVariableOp-SGD/dense_9/bias/momentum/Read/ReadVariableOp0SGD/dense_10/kernel/momentum/Read/ReadVariableOp.SGD/dense_10/bias/momentum/Read/ReadVariableOpPSGD/transformer_block_3/layer_normalization_6/gamma/momentum/Read/ReadVariableOpOSGD/transformer_block_3/layer_normalization_6/beta/momentum/Read/ReadVariableOpPSGD/transformer_block_3/layer_normalization_7/gamma/momentum/Read/ReadVariableOpOSGD/transformer_block_3/layer_normalization_7/beta/momentum/Read/ReadVariableOpConst*C
Tin<
:28	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_14625
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdecaylearning_ratemomentumSGD/iter5token_and_position_embedding_1/embedding_2/embeddings5token_and_position_embedding_1/embedding_3/embeddings7transformer_block_3/multi_head_attention_3/query/kernel5transformer_block_3/multi_head_attention_3/query/bias5transformer_block_3/multi_head_attention_3/key/kernel3transformer_block_3/multi_head_attention_3/key/bias7transformer_block_3/multi_head_attention_3/value/kernel5transformer_block_3/multi_head_attention_3/value/biasBtransformer_block_3/multi_head_attention_3/attention_output/kernel@transformer_block_3/multi_head_attention_3/attention_output/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/bias/transformer_block_3/layer_normalization_6/gamma.transformer_block_3/layer_normalization_6/beta/transformer_block_3/layer_normalization_7/gamma.transformer_block_3/layer_normalization_7/betatotalcountSGD/dense_11/kernel/momentumSGD/dense_11/bias/momentumSGD/dense_12/kernel/momentumSGD/dense_12/bias/momentumSGD/dense_13/kernel/momentumSGD/dense_13/bias/momentumBSGD/token_and_position_embedding_1/embedding_2/embeddings/momentumBSGD/token_and_position_embedding_1/embedding_3/embeddings/momentumDSGD/transformer_block_3/multi_head_attention_3/query/kernel/momentumBSGD/transformer_block_3/multi_head_attention_3/query/bias/momentumBSGD/transformer_block_3/multi_head_attention_3/key/kernel/momentum@SGD/transformer_block_3/multi_head_attention_3/key/bias/momentumDSGD/transformer_block_3/multi_head_attention_3/value/kernel/momentumBSGD/transformer_block_3/multi_head_attention_3/value/bias/momentumOSGD/transformer_block_3/multi_head_attention_3/attention_output/kernel/momentumMSGD/transformer_block_3/multi_head_attention_3/attention_output/bias/momentumSGD/dense_9/kernel/momentumSGD/dense_9/bias/momentumSGD/dense_10/kernel/momentumSGD/dense_10/bias/momentum<SGD/transformer_block_3/layer_normalization_6/gamma/momentum;SGD/transformer_block_3/layer_normalization_6/beta/momentum<SGD/transformer_block_3/layer_normalization_7/gamma/momentum;SGD/transformer_block_3/layer_normalization_7/beta/momentum*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_14797??
?
c
*__inference_dropout_11_layer_call_fn_14197

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_128282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
c
E__inference_dropout_11_layer_call_and_return_conditional_losses_12833

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
C__inference_dense_13_layer_call_and_return_conditional_losses_14212

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_13715

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_131122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesv
t:??????????R::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????R
 
_user_specified_nameinputs
??
?
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_14023

inputsF
Bmulti_head_attention_3_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_3_query_add_readvariableop_resourceD
@multi_head_attention_3_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_3_key_add_readvariableop_resourceF
Bmulti_head_attention_3_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_3_value_add_readvariableop_resourceQ
Mmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_3_attention_output_add_readvariableop_resource?
;layer_normalization_6_batchnorm_mul_readvariableop_resource;
7layer_normalization_6_batchnorm_readvariableop_resource:
6sequential_3_dense_9_tensordot_readvariableop_resource8
4sequential_3_dense_9_biasadd_readvariableop_resource;
7sequential_3_dense_10_tensordot_readvariableop_resource9
5sequential_3_dense_10_biasadd_readvariableop_resource?
;layer_normalization_7_batchnorm_mul_readvariableop_resource;
7layer_normalization_7_batchnorm_readvariableop_resource
identity??.layer_normalization_6/batchnorm/ReadVariableOp?2layer_normalization_6/batchnorm/mul/ReadVariableOp?.layer_normalization_7/batchnorm/ReadVariableOp?2layer_normalization_7/batchnorm/mul/ReadVariableOp?:multi_head_attention_3/attention_output/add/ReadVariableOp?Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp?-multi_head_attention_3/key/add/ReadVariableOp?7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp?/multi_head_attention_3/query/add/ReadVariableOp?9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp?/multi_head_attention_3/value/add/ReadVariableOp?9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp?,sequential_3/dense_10/BiasAdd/ReadVariableOp?.sequential_3/dense_10/Tensordot/ReadVariableOp?+sequential_3/dense_9/BiasAdd/ReadVariableOp?-sequential_3/dense_9/Tensordot/ReadVariableOp?
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp?
*multi_head_attention_3/query/einsum/EinsumEinsuminputsAmulti_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2,
*multi_head_attention_3/query/einsum/Einsum?
/multi_head_attention_3/query/add/ReadVariableOpReadVariableOp8multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/query/add/ReadVariableOp?
 multi_head_attention_3/query/addAddV23multi_head_attention_3/query/einsum/Einsum:output:07multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2"
 multi_head_attention_3/query/add?
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp?
(multi_head_attention_3/key/einsum/EinsumEinsuminputs?multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2*
(multi_head_attention_3/key/einsum/Einsum?
-multi_head_attention_3/key/add/ReadVariableOpReadVariableOp6multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_3/key/add/ReadVariableOp?
multi_head_attention_3/key/addAddV21multi_head_attention_3/key/einsum/Einsum:output:05multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2 
multi_head_attention_3/key/add?
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp?
*multi_head_attention_3/value/einsum/EinsumEinsuminputsAmulti_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2,
*multi_head_attention_3/value/einsum/Einsum?
/multi_head_attention_3/value/add/ReadVariableOpReadVariableOp8multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/value/add/ReadVariableOp?
 multi_head_attention_3/value/addAddV23multi_head_attention_3/value/einsum/Einsum:output:07multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2"
 multi_head_attention_3/value/add?
multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>2
multi_head_attention_3/Mul/y?
multi_head_attention_3/MulMul$multi_head_attention_3/query/add:z:0%multi_head_attention_3/Mul/y:output:0*
T0*/
_output_shapes
:?????????# 2
multi_head_attention_3/Mul?
$multi_head_attention_3/einsum/EinsumEinsum"multi_head_attention_3/key/add:z:0multi_head_attention_3/Mul:z:0*
N*
T0*/
_output_shapes
:?????????##*
equationaecd,abcd->acbe2&
$multi_head_attention_3/einsum/Einsum?
&multi_head_attention_3/softmax/SoftmaxSoftmax-multi_head_attention_3/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????##2(
&multi_head_attention_3/softmax/Softmax?
'multi_head_attention_3/dropout/IdentityIdentity0multi_head_attention_3/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????##2)
'multi_head_attention_3/dropout/Identity?
&multi_head_attention_3/einsum_1/EinsumEinsum0multi_head_attention_3/dropout/Identity:output:0$multi_head_attention_3/value/add:z:0*
N*
T0*/
_output_shapes
:?????????# *
equationacbe,aecd->abcd2(
&multi_head_attention_3/einsum_1/Einsum?
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp?
5multi_head_attention_3/attention_output/einsum/EinsumEinsum/multi_head_attention_3/einsum_1/Einsum:output:0Lmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????# *
equationabcd,cde->abe27
5multi_head_attention_3/attention_output/einsum/Einsum?
:multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_3/attention_output/add/ReadVariableOp?
+multi_head_attention_3/attention_output/addAddV2>multi_head_attention_3/attention_output/einsum/Einsum:output:0Bmulti_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2-
+multi_head_attention_3/attention_output/add?
dropout_8/IdentityIdentity/multi_head_attention_3/attention_output/add:z:0*
T0*+
_output_shapes
:?????????# 2
dropout_8/Identityn
addAddV2inputsdropout_8/Identity:output:0*
T0*+
_output_shapes
:?????????# 2
add?
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_6/moments/mean/reduction_indices?
"layer_normalization_6/moments/meanMeanadd:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2$
"layer_normalization_6/moments/mean?
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2,
*layer_normalization_6/moments/StopGradient?
/layer_normalization_6/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 21
/layer_normalization_6/moments/SquaredDifference?
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_6/moments/variance/reduction_indices?
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2(
&layer_normalization_6/moments/variance?
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_6/batchnorm/add/y?
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2%
#layer_normalization_6/batchnorm/add?
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2'
%layer_normalization_6/batchnorm/Rsqrt?
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_6/batchnorm/mul/ReadVariableOp?
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2%
#layer_normalization_6/batchnorm/mul?
%layer_normalization_6/batchnorm/mul_1Muladd:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_6/batchnorm/mul_1?
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_6/batchnorm/mul_2?
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_6/batchnorm/ReadVariableOp?
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2%
#layer_normalization_6/batchnorm/sub?
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_6/batchnorm/add_1?
-sequential_3/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_3_dense_9_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02/
-sequential_3/dense_9/Tensordot/ReadVariableOp?
#sequential_3/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_3/dense_9/Tensordot/axes?
#sequential_3/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_3/dense_9/Tensordot/free?
$sequential_3/dense_9/Tensordot/ShapeShape)layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_3/dense_9/Tensordot/Shape?
,sequential_3/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_3/dense_9/Tensordot/GatherV2/axis?
'sequential_3/dense_9/Tensordot/GatherV2GatherV2-sequential_3/dense_9/Tensordot/Shape:output:0,sequential_3/dense_9/Tensordot/free:output:05sequential_3/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_3/dense_9/Tensordot/GatherV2?
.sequential_3/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_3/dense_9/Tensordot/GatherV2_1/axis?
)sequential_3/dense_9/Tensordot/GatherV2_1GatherV2-sequential_3/dense_9/Tensordot/Shape:output:0,sequential_3/dense_9/Tensordot/axes:output:07sequential_3/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_3/dense_9/Tensordot/GatherV2_1?
$sequential_3/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_3/dense_9/Tensordot/Const?
#sequential_3/dense_9/Tensordot/ProdProd0sequential_3/dense_9/Tensordot/GatherV2:output:0-sequential_3/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_3/dense_9/Tensordot/Prod?
&sequential_3/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_3/dense_9/Tensordot/Const_1?
%sequential_3/dense_9/Tensordot/Prod_1Prod2sequential_3/dense_9/Tensordot/GatherV2_1:output:0/sequential_3/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_3/dense_9/Tensordot/Prod_1?
*sequential_3/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_3/dense_9/Tensordot/concat/axis?
%sequential_3/dense_9/Tensordot/concatConcatV2,sequential_3/dense_9/Tensordot/free:output:0,sequential_3/dense_9/Tensordot/axes:output:03sequential_3/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_9/Tensordot/concat?
$sequential_3/dense_9/Tensordot/stackPack,sequential_3/dense_9/Tensordot/Prod:output:0.sequential_3/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_3/dense_9/Tensordot/stack?
(sequential_3/dense_9/Tensordot/transpose	Transpose)layer_normalization_6/batchnorm/add_1:z:0.sequential_3/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????# 2*
(sequential_3/dense_9/Tensordot/transpose?
&sequential_3/dense_9/Tensordot/ReshapeReshape,sequential_3/dense_9/Tensordot/transpose:y:0-sequential_3/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&sequential_3/dense_9/Tensordot/Reshape?
%sequential_3/dense_9/Tensordot/MatMulMatMul/sequential_3/dense_9/Tensordot/Reshape:output:05sequential_3/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2'
%sequential_3/dense_9/Tensordot/MatMul?
&sequential_3/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2(
&sequential_3/dense_9/Tensordot/Const_2?
,sequential_3/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_3/dense_9/Tensordot/concat_1/axis?
'sequential_3/dense_9/Tensordot/concat_1ConcatV20sequential_3/dense_9/Tensordot/GatherV2:output:0/sequential_3/dense_9/Tensordot/Const_2:output:05sequential_3/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_3/dense_9/Tensordot/concat_1?
sequential_3/dense_9/TensordotReshape/sequential_3/dense_9/Tensordot/MatMul:product:00sequential_3/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@2 
sequential_3/dense_9/Tensordot?
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp?
sequential_3/dense_9/BiasAddBiasAdd'sequential_3/dense_9/Tensordot:output:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@2
sequential_3/dense_9/BiasAdd?
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#@2
sequential_3/dense_9/Relu?
.sequential_3/dense_10/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_10_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_3/dense_10/Tensordot/ReadVariableOp?
$sequential_3/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_3/dense_10/Tensordot/axes?
$sequential_3/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_3/dense_10/Tensordot/free?
%sequential_3/dense_10/Tensordot/ShapeShape'sequential_3/dense_9/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/Shape?
-sequential_3/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/GatherV2/axis?
(sequential_3/dense_10/Tensordot/GatherV2GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/free:output:06sequential_3/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/GatherV2?
/sequential_3/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_3/dense_10/Tensordot/GatherV2_1/axis?
*sequential_3/dense_10/Tensordot/GatherV2_1GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/axes:output:08sequential_3/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_3/dense_10/Tensordot/GatherV2_1?
%sequential_3/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_3/dense_10/Tensordot/Const?
$sequential_3/dense_10/Tensordot/ProdProd1sequential_3/dense_10/Tensordot/GatherV2:output:0.sequential_3/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_3/dense_10/Tensordot/Prod?
'sequential_3/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_1?
&sequential_3/dense_10/Tensordot/Prod_1Prod3sequential_3/dense_10/Tensordot/GatherV2_1:output:00sequential_3/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_3/dense_10/Tensordot/Prod_1?
+sequential_3/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_3/dense_10/Tensordot/concat/axis?
&sequential_3/dense_10/Tensordot/concatConcatV2-sequential_3/dense_10/Tensordot/free:output:0-sequential_3/dense_10/Tensordot/axes:output:04sequential_3/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_3/dense_10/Tensordot/concat?
%sequential_3/dense_10/Tensordot/stackPack-sequential_3/dense_10/Tensordot/Prod:output:0/sequential_3/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/stack?
)sequential_3/dense_10/Tensordot/transpose	Transpose'sequential_3/dense_9/Relu:activations:0/sequential_3/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2+
)sequential_3/dense_10/Tensordot/transpose?
'sequential_3/dense_10/Tensordot/ReshapeReshape-sequential_3/dense_10/Tensordot/transpose:y:0.sequential_3/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_3/dense_10/Tensordot/Reshape?
&sequential_3/dense_10/Tensordot/MatMulMatMul0sequential_3/dense_10/Tensordot/Reshape:output:06sequential_3/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_3/dense_10/Tensordot/MatMul?
'sequential_3/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_2?
-sequential_3/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/concat_1/axis?
(sequential_3/dense_10/Tensordot/concat_1ConcatV21sequential_3/dense_10/Tensordot/GatherV2:output:00sequential_3/dense_10/Tensordot/Const_2:output:06sequential_3/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/concat_1?
sequential_3/dense_10/TensordotReshape0sequential_3/dense_10/Tensordot/MatMul:product:01sequential_3/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 2!
sequential_3/dense_10/Tensordot?
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp?
sequential_3/dense_10/BiasAddBiasAdd(sequential_3/dense_10/Tensordot:output:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2
sequential_3/dense_10/BiasAdd?
dropout_9/IdentityIdentity&sequential_3/dense_10/BiasAdd:output:0*
T0*+
_output_shapes
:?????????# 2
dropout_9/Identity?
add_1AddV2)layer_normalization_6/batchnorm/add_1:z:0dropout_9/Identity:output:0*
T0*+
_output_shapes
:?????????# 2
add_1?
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_7/moments/mean/reduction_indices?
"layer_normalization_7/moments/meanMean	add_1:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2$
"layer_normalization_7/moments/mean?
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2,
*layer_normalization_7/moments/StopGradient?
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 21
/layer_normalization_7/moments/SquaredDifference?
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_7/moments/variance/reduction_indices?
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2(
&layer_normalization_7/moments/variance?
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_7/batchnorm/add/y?
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2%
#layer_normalization_7/batchnorm/add?
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2'
%layer_normalization_7/batchnorm/Rsqrt?
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_7/batchnorm/mul/ReadVariableOp?
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2%
#layer_normalization_7/batchnorm/mul?
%layer_normalization_7/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_7/batchnorm/mul_1?
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_7/batchnorm/mul_2?
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_7/batchnorm/ReadVariableOp?
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2%
#layer_normalization_7/batchnorm/sub?
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_7/batchnorm/add_1?
IdentityIdentity)layer_normalization_7/batchnorm/add_1:z:0/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp;^multi_head_attention_3/attention_output/add/ReadVariableOpE^multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_3/key/add/ReadVariableOp8^multi_head_attention_3/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/query/add/ReadVariableOp:^multi_head_attention_3/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/value/add/ReadVariableOp:^multi_head_attention_3/value/einsum/Einsum/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp/^sequential_3/dense_10/Tensordot/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp.^sequential_3/dense_9/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:?????????# ::::::::::::::::2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_3/attention_output/add/ReadVariableOp:multi_head_attention_3/attention_output/add/ReadVariableOp2?
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_3/key/add/ReadVariableOp-multi_head_attention_3/key/add/ReadVariableOp2r
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/query/add/ReadVariableOp/multi_head_attention_3/query/add/ReadVariableOp2v
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/value/add/ReadVariableOp/multi_head_attention_3/value/add/ReadVariableOp2v
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2`
.sequential_3/dense_10/Tensordot/ReadVariableOp.sequential_3/dense_10/Tensordot/ReadVariableOp2Z
+sequential_3/dense_9/BiasAdd/ReadVariableOp+sequential_3/dense_9/BiasAdd/ReadVariableOp2^
-sequential_3/dense_9/Tensordot/ReadVariableOp-sequential_3/dense_9/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_13434

inputsE
Atoken_and_position_embedding_1_embedding_3_embedding_lookup_13235E
Atoken_and_position_embedding_1_embedding_2_embedding_lookup_13241Z
Vtransformer_block_3_multi_head_attention_3_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_3_multi_head_attention_3_query_add_readvariableop_resourceX
Ttransformer_block_3_multi_head_attention_3_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_3_multi_head_attention_3_key_add_readvariableop_resourceZ
Vtransformer_block_3_multi_head_attention_3_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_3_multi_head_attention_3_value_add_readvariableop_resourcee
atransformer_block_3_multi_head_attention_3_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_3_multi_head_attention_3_attention_output_add_readvariableop_resourceS
Otransformer_block_3_layer_normalization_6_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_3_layer_normalization_6_batchnorm_readvariableop_resourceN
Jtransformer_block_3_sequential_3_dense_9_tensordot_readvariableop_resourceL
Htransformer_block_3_sequential_3_dense_9_biasadd_readvariableop_resourceO
Ktransformer_block_3_sequential_3_dense_10_tensordot_readvariableop_resourceM
Itransformer_block_3_sequential_3_dense_10_biasadd_readvariableop_resourceS
Otransformer_block_3_layer_normalization_7_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_3_layer_normalization_7_batchnorm_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource
identity??dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?;token_and_position_embedding_1/embedding_2/embedding_lookup?;token_and_position_embedding_1/embedding_3/embedding_lookup?Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp?Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp?Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp?Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp?Ntransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp?Xtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp?Atransformer_block_3/multi_head_attention_3/key/add/ReadVariableOp?Ktransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp?Ctransformer_block_3/multi_head_attention_3/query/add/ReadVariableOp?Mtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp?Ctransformer_block_3/multi_head_attention_3/value/add/ReadVariableOp?Mtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp?@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp?Btransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp??transformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOp?Atransformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOp?
$token_and_position_embedding_1/ShapeShapeinputs*
T0*
_output_shapes
:2&
$token_and_position_embedding_1/Shape?
2token_and_position_embedding_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????24
2token_and_position_embedding_1/strided_slice/stack?
4token_and_position_embedding_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_1/strided_slice/stack_1?
4token_and_position_embedding_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_1/strided_slice/stack_2?
,token_and_position_embedding_1/strided_sliceStridedSlice-token_and_position_embedding_1/Shape:output:0;token_and_position_embedding_1/strided_slice/stack:output:0=token_and_position_embedding_1/strided_slice/stack_1:output:0=token_and_position_embedding_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_1/strided_slice?
*token_and_position_embedding_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_1/range/start?
*token_and_position_embedding_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_1/range/delta?
$token_and_position_embedding_1/rangeRange3token_and_position_embedding_1/range/start:output:05token_and_position_embedding_1/strided_slice:output:03token_and_position_embedding_1/range/delta:output:0*#
_output_shapes
:?????????2&
$token_and_position_embedding_1/range?
;token_and_position_embedding_1/embedding_3/embedding_lookupResourceGatherAtoken_and_position_embedding_1_embedding_3_embedding_lookup_13235-token_and_position_embedding_1/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@token_and_position_embedding_1/embedding_3/embedding_lookup/13235*'
_output_shapes
:????????? *
dtype02=
;token_and_position_embedding_1/embedding_3/embedding_lookup?
Dtoken_and_position_embedding_1/embedding_3/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_1/embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@token_and_position_embedding_1/embedding_3/embedding_lookup/13235*'
_output_shapes
:????????? 2F
Dtoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity?
Ftoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2H
Ftoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1?
/token_and_position_embedding_1/embedding_2/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:??????????R21
/token_and_position_embedding_1/embedding_2/Cast?
;token_and_position_embedding_1/embedding_2/embedding_lookupResourceGatherAtoken_and_position_embedding_1_embedding_2_embedding_lookup_132413token_and_position_embedding_1/embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@token_and_position_embedding_1/embedding_2/embedding_lookup/13241*,
_output_shapes
:??????????R *
dtype02=
;token_and_position_embedding_1/embedding_2/embedding_lookup?
Dtoken_and_position_embedding_1/embedding_2/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_1/embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@token_and_position_embedding_1/embedding_2/embedding_lookup/13241*,
_output_shapes
:??????????R 2F
Dtoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity?
Ftoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????R 2H
Ftoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1?
"token_and_position_embedding_1/addAddV2Otoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????R 2$
"token_and_position_embedding_1/add?
"average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_1/ExpandDims/dim?
average_pooling1d_1/ExpandDims
ExpandDims&token_and_position_embedding_1/add:z:0+average_pooling1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????R 2 
average_pooling1d_1/ExpandDims?
average_pooling1d_1/AvgPoolAvgPool'average_pooling1d_1/ExpandDims:output:0*
T0*/
_output_shapes
:?????????# *
ksize	
?*
paddingVALID*
strides	
?2
average_pooling1d_1/AvgPool?
average_pooling1d_1/SqueezeSqueeze$average_pooling1d_1/AvgPool:output:0*
T0*+
_output_shapes
:?????????# *
squeeze_dims
2
average_pooling1d_1/Squeeze?
Mtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_3_multi_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp?
>transformer_block_3/multi_head_attention_3/query/einsum/EinsumEinsum$average_pooling1d_1/Squeeze:output:0Utransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2@
>transformer_block_3/multi_head_attention_3/query/einsum/Einsum?
Ctransformer_block_3/multi_head_attention_3/query/add/ReadVariableOpReadVariableOpLtransformer_block_3_multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_3/multi_head_attention_3/query/add/ReadVariableOp?
4transformer_block_3/multi_head_attention_3/query/addAddV2Gtransformer_block_3/multi_head_attention_3/query/einsum/Einsum:output:0Ktransformer_block_3/multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 26
4transformer_block_3/multi_head_attention_3/query/add?
Ktransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_3_multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp?
<transformer_block_3/multi_head_attention_3/key/einsum/EinsumEinsum$average_pooling1d_1/Squeeze:output:0Stransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2>
<transformer_block_3/multi_head_attention_3/key/einsum/Einsum?
Atransformer_block_3/multi_head_attention_3/key/add/ReadVariableOpReadVariableOpJtransformer_block_3_multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_3/multi_head_attention_3/key/add/ReadVariableOp?
2transformer_block_3/multi_head_attention_3/key/addAddV2Etransformer_block_3/multi_head_attention_3/key/einsum/Einsum:output:0Itransformer_block_3/multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 24
2transformer_block_3/multi_head_attention_3/key/add?
Mtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_3_multi_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp?
>transformer_block_3/multi_head_attention_3/value/einsum/EinsumEinsum$average_pooling1d_1/Squeeze:output:0Utransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2@
>transformer_block_3/multi_head_attention_3/value/einsum/Einsum?
Ctransformer_block_3/multi_head_attention_3/value/add/ReadVariableOpReadVariableOpLtransformer_block_3_multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_3/multi_head_attention_3/value/add/ReadVariableOp?
4transformer_block_3/multi_head_attention_3/value/addAddV2Gtransformer_block_3/multi_head_attention_3/value/einsum/Einsum:output:0Ktransformer_block_3/multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 26
4transformer_block_3/multi_head_attention_3/value/add?
0transformer_block_3/multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>22
0transformer_block_3/multi_head_attention_3/Mul/y?
.transformer_block_3/multi_head_attention_3/MulMul8transformer_block_3/multi_head_attention_3/query/add:z:09transformer_block_3/multi_head_attention_3/Mul/y:output:0*
T0*/
_output_shapes
:?????????# 20
.transformer_block_3/multi_head_attention_3/Mul?
8transformer_block_3/multi_head_attention_3/einsum/EinsumEinsum6transformer_block_3/multi_head_attention_3/key/add:z:02transformer_block_3/multi_head_attention_3/Mul:z:0*
N*
T0*/
_output_shapes
:?????????##*
equationaecd,abcd->acbe2:
8transformer_block_3/multi_head_attention_3/einsum/Einsum?
:transformer_block_3/multi_head_attention_3/softmax/SoftmaxSoftmaxAtransformer_block_3/multi_head_attention_3/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????##2<
:transformer_block_3/multi_head_attention_3/softmax/Softmax?
@transformer_block_3/multi_head_attention_3/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2B
@transformer_block_3/multi_head_attention_3/dropout/dropout/Const?
>transformer_block_3/multi_head_attention_3/dropout/dropout/MulMulDtransformer_block_3/multi_head_attention_3/softmax/Softmax:softmax:0Itransformer_block_3/multi_head_attention_3/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????##2@
>transformer_block_3/multi_head_attention_3/dropout/dropout/Mul?
@transformer_block_3/multi_head_attention_3/dropout/dropout/ShapeShapeDtransformer_block_3/multi_head_attention_3/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2B
@transformer_block_3/multi_head_attention_3/dropout/dropout/Shape?
Wtransformer_block_3/multi_head_attention_3/dropout/dropout/random_uniform/RandomUniformRandomUniformItransformer_block_3/multi_head_attention_3/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????##*
dtype02Y
Wtransformer_block_3/multi_head_attention_3/dropout/dropout/random_uniform/RandomUniform?
Itransformer_block_3/multi_head_attention_3/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2K
Itransformer_block_3/multi_head_attention_3/dropout/dropout/GreaterEqual/y?
Gtransformer_block_3/multi_head_attention_3/dropout/dropout/GreaterEqualGreaterEqual`transformer_block_3/multi_head_attention_3/dropout/dropout/random_uniform/RandomUniform:output:0Rtransformer_block_3/multi_head_attention_3/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????##2I
Gtransformer_block_3/multi_head_attention_3/dropout/dropout/GreaterEqual?
?transformer_block_3/multi_head_attention_3/dropout/dropout/CastCastKtransformer_block_3/multi_head_attention_3/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????##2A
?transformer_block_3/multi_head_attention_3/dropout/dropout/Cast?
@transformer_block_3/multi_head_attention_3/dropout/dropout/Mul_1MulBtransformer_block_3/multi_head_attention_3/dropout/dropout/Mul:z:0Ctransformer_block_3/multi_head_attention_3/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????##2B
@transformer_block_3/multi_head_attention_3/dropout/dropout/Mul_1?
:transformer_block_3/multi_head_attention_3/einsum_1/EinsumEinsumDtransformer_block_3/multi_head_attention_3/dropout/dropout/Mul_1:z:08transformer_block_3/multi_head_attention_3/value/add:z:0*
N*
T0*/
_output_shapes
:?????????# *
equationacbe,aecd->abcd2<
:transformer_block_3/multi_head_attention_3/einsum_1/Einsum?
Xtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_3_multi_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp?
Itransformer_block_3/multi_head_attention_3/attention_output/einsum/EinsumEinsumCtransformer_block_3/multi_head_attention_3/einsum_1/Einsum:output:0`transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????# *
equationabcd,cde->abe2K
Itransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum?
Ntransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_3_multi_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp?
?transformer_block_3/multi_head_attention_3/attention_output/addAddV2Rtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum:output:0Vtransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2A
?transformer_block_3/multi_head_attention_3/attention_output/add?
+transformer_block_3/dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2-
+transformer_block_3/dropout_8/dropout/Const?
)transformer_block_3/dropout_8/dropout/MulMulCtransformer_block_3/multi_head_attention_3/attention_output/add:z:04transformer_block_3/dropout_8/dropout/Const:output:0*
T0*+
_output_shapes
:?????????# 2+
)transformer_block_3/dropout_8/dropout/Mul?
+transformer_block_3/dropout_8/dropout/ShapeShapeCtransformer_block_3/multi_head_attention_3/attention_output/add:z:0*
T0*
_output_shapes
:2-
+transformer_block_3/dropout_8/dropout/Shape?
Btransformer_block_3/dropout_8/dropout/random_uniform/RandomUniformRandomUniform4transformer_block_3/dropout_8/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????# *
dtype02D
Btransformer_block_3/dropout_8/dropout/random_uniform/RandomUniform?
4transformer_block_3/dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=26
4transformer_block_3/dropout_8/dropout/GreaterEqual/y?
2transformer_block_3/dropout_8/dropout/GreaterEqualGreaterEqualKtransformer_block_3/dropout_8/dropout/random_uniform/RandomUniform:output:0=transformer_block_3/dropout_8/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????# 24
2transformer_block_3/dropout_8/dropout/GreaterEqual?
*transformer_block_3/dropout_8/dropout/CastCast6transformer_block_3/dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????# 2,
*transformer_block_3/dropout_8/dropout/Cast?
+transformer_block_3/dropout_8/dropout/Mul_1Mul-transformer_block_3/dropout_8/dropout/Mul:z:0.transformer_block_3/dropout_8/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????# 2-
+transformer_block_3/dropout_8/dropout/Mul_1?
transformer_block_3/addAddV2$average_pooling1d_1/Squeeze:output:0/transformer_block_3/dropout_8/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????# 2
transformer_block_3/add?
Htransformer_block_3/layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_3/layer_normalization_6/moments/mean/reduction_indices?
6transformer_block_3/layer_normalization_6/moments/meanMeantransformer_block_3/add:z:0Qtransformer_block_3/layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(28
6transformer_block_3/layer_normalization_6/moments/mean?
>transformer_block_3/layer_normalization_6/moments/StopGradientStopGradient?transformer_block_3/layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2@
>transformer_block_3/layer_normalization_6/moments/StopGradient?
Ctransformer_block_3/layer_normalization_6/moments/SquaredDifferenceSquaredDifferencetransformer_block_3/add:z:0Gtransformer_block_3/layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 2E
Ctransformer_block_3/layer_normalization_6/moments/SquaredDifference?
Ltransformer_block_3/layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_3/layer_normalization_6/moments/variance/reduction_indices?
:transformer_block_3/layer_normalization_6/moments/varianceMeanGtransformer_block_3/layer_normalization_6/moments/SquaredDifference:z:0Utransformer_block_3/layer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2<
:transformer_block_3/layer_normalization_6/moments/variance?
9transformer_block_3/layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52;
9transformer_block_3/layer_normalization_6/batchnorm/add/y?
7transformer_block_3/layer_normalization_6/batchnorm/addAddV2Ctransformer_block_3/layer_normalization_6/moments/variance:output:0Btransformer_block_3/layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#29
7transformer_block_3/layer_normalization_6/batchnorm/add?
9transformer_block_3/layer_normalization_6/batchnorm/RsqrtRsqrt;transformer_block_3/layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2;
9transformer_block_3/layer_normalization_6/batchnorm/Rsqrt?
Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_3_layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp?
7transformer_block_3/layer_normalization_6/batchnorm/mulMul=transformer_block_3/layer_normalization_6/batchnorm/Rsqrt:y:0Ntransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 29
7transformer_block_3/layer_normalization_6/batchnorm/mul?
9transformer_block_3/layer_normalization_6/batchnorm/mul_1Multransformer_block_3/add:z:0;transformer_block_3/layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2;
9transformer_block_3/layer_normalization_6/batchnorm/mul_1?
9transformer_block_3/layer_normalization_6/batchnorm/mul_2Mul?transformer_block_3/layer_normalization_6/moments/mean:output:0;transformer_block_3/layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2;
9transformer_block_3/layer_normalization_6/batchnorm/mul_2?
Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_3_layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp?
7transformer_block_3/layer_normalization_6/batchnorm/subSubJtransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp:value:0=transformer_block_3/layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 29
7transformer_block_3/layer_normalization_6/batchnorm/sub?
9transformer_block_3/layer_normalization_6/batchnorm/add_1AddV2=transformer_block_3/layer_normalization_6/batchnorm/mul_1:z:0;transformer_block_3/layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2;
9transformer_block_3/layer_normalization_6/batchnorm/add_1?
Atransformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_3_sequential_3_dense_9_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02C
Atransformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOp?
7transformer_block_3/sequential_3/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_3/sequential_3/dense_9/Tensordot/axes?
7transformer_block_3/sequential_3/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_3/sequential_3/dense_9/Tensordot/free?
8transformer_block_3/sequential_3/dense_9/Tensordot/ShapeShape=transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:2:
8transformer_block_3/sequential_3/dense_9/Tensordot/Shape?
@transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2/axis?
;transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2GatherV2Atransformer_block_3/sequential_3/dense_9/Tensordot/Shape:output:0@transformer_block_3/sequential_3/dense_9/Tensordot/free:output:0Itransformer_block_3/sequential_3/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2?
Btransformer_block_3/sequential_3/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_3/sequential_3/dense_9/Tensordot/GatherV2_1/axis?
=transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2_1GatherV2Atransformer_block_3/sequential_3/dense_9/Tensordot/Shape:output:0@transformer_block_3/sequential_3/dense_9/Tensordot/axes:output:0Ktransformer_block_3/sequential_3/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2_1?
8transformer_block_3/sequential_3/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_3/sequential_3/dense_9/Tensordot/Const?
7transformer_block_3/sequential_3/dense_9/Tensordot/ProdProdDtransformer_block_3/sequential_3/dense_9/Tensordot/GatherV2:output:0Atransformer_block_3/sequential_3/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_3/sequential_3/dense_9/Tensordot/Prod?
:transformer_block_3/sequential_3/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_3/sequential_3/dense_9/Tensordot/Const_1?
9transformer_block_3/sequential_3/dense_9/Tensordot/Prod_1ProdFtransformer_block_3/sequential_3/dense_9/Tensordot/GatherV2_1:output:0Ctransformer_block_3/sequential_3/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_3/sequential_3/dense_9/Tensordot/Prod_1?
>transformer_block_3/sequential_3/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_3/sequential_3/dense_9/Tensordot/concat/axis?
9transformer_block_3/sequential_3/dense_9/Tensordot/concatConcatV2@transformer_block_3/sequential_3/dense_9/Tensordot/free:output:0@transformer_block_3/sequential_3/dense_9/Tensordot/axes:output:0Gtransformer_block_3/sequential_3/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_3/sequential_3/dense_9/Tensordot/concat?
8transformer_block_3/sequential_3/dense_9/Tensordot/stackPack@transformer_block_3/sequential_3/dense_9/Tensordot/Prod:output:0Btransformer_block_3/sequential_3/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_3/sequential_3/dense_9/Tensordot/stack?
<transformer_block_3/sequential_3/dense_9/Tensordot/transpose	Transpose=transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0Btransformer_block_3/sequential_3/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????# 2>
<transformer_block_3/sequential_3/dense_9/Tensordot/transpose?
:transformer_block_3/sequential_3/dense_9/Tensordot/ReshapeReshape@transformer_block_3/sequential_3/dense_9/Tensordot/transpose:y:0Atransformer_block_3/sequential_3/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block_3/sequential_3/dense_9/Tensordot/Reshape?
9transformer_block_3/sequential_3/dense_9/Tensordot/MatMulMatMulCtransformer_block_3/sequential_3/dense_9/Tensordot/Reshape:output:0Itransformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2;
9transformer_block_3/sequential_3/dense_9/Tensordot/MatMul?
:transformer_block_3/sequential_3/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2<
:transformer_block_3/sequential_3/dense_9/Tensordot/Const_2?
@transformer_block_3/sequential_3/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_3/sequential_3/dense_9/Tensordot/concat_1/axis?
;transformer_block_3/sequential_3/dense_9/Tensordot/concat_1ConcatV2Dtransformer_block_3/sequential_3/dense_9/Tensordot/GatherV2:output:0Ctransformer_block_3/sequential_3/dense_9/Tensordot/Const_2:output:0Itransformer_block_3/sequential_3/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_3/sequential_3/dense_9/Tensordot/concat_1?
2transformer_block_3/sequential_3/dense_9/TensordotReshapeCtransformer_block_3/sequential_3/dense_9/Tensordot/MatMul:product:0Dtransformer_block_3/sequential_3/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@24
2transformer_block_3/sequential_3/dense_9/Tensordot?
?transformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_3_sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02A
?transformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOp?
0transformer_block_3/sequential_3/dense_9/BiasAddBiasAdd;transformer_block_3/sequential_3/dense_9/Tensordot:output:0Gtransformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@22
0transformer_block_3/sequential_3/dense_9/BiasAdd?
-transformer_block_3/sequential_3/dense_9/ReluRelu9transformer_block_3/sequential_3/dense_9/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#@2/
-transformer_block_3/sequential_3/dense_9/Relu?
Btransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_3_sequential_3_dense_10_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02D
Btransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp?
8transformer_block_3/sequential_3/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_3/sequential_3/dense_10/Tensordot/axes?
8transformer_block_3/sequential_3/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_3/sequential_3/dense_10/Tensordot/free?
9transformer_block_3/sequential_3/dense_10/Tensordot/ShapeShape;transformer_block_3/sequential_3/dense_9/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_3/sequential_3/dense_10/Tensordot/Shape?
Atransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2/axis?
<transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2GatherV2Btransformer_block_3/sequential_3/dense_10/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_10/Tensordot/free:output:0Jtransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2?
Ctransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1/axis?
>transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1GatherV2Btransformer_block_3/sequential_3/dense_10/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_10/Tensordot/axes:output:0Ltransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1?
9transformer_block_3/sequential_3/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_3/sequential_3/dense_10/Tensordot/Const?
8transformer_block_3/sequential_3/dense_10/Tensordot/ProdProdEtransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2:output:0Btransformer_block_3/sequential_3/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_3/sequential_3/dense_10/Tensordot/Prod?
;transformer_block_3/sequential_3/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_3/sequential_3/dense_10/Tensordot/Const_1?
:transformer_block_3/sequential_3/dense_10/Tensordot/Prod_1ProdGtransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1:output:0Dtransformer_block_3/sequential_3/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_3/sequential_3/dense_10/Tensordot/Prod_1?
?transformer_block_3/sequential_3/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_3/sequential_3/dense_10/Tensordot/concat/axis?
:transformer_block_3/sequential_3/dense_10/Tensordot/concatConcatV2Atransformer_block_3/sequential_3/dense_10/Tensordot/free:output:0Atransformer_block_3/sequential_3/dense_10/Tensordot/axes:output:0Htransformer_block_3/sequential_3/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_3/sequential_3/dense_10/Tensordot/concat?
9transformer_block_3/sequential_3/dense_10/Tensordot/stackPackAtransformer_block_3/sequential_3/dense_10/Tensordot/Prod:output:0Ctransformer_block_3/sequential_3/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_3/sequential_3/dense_10/Tensordot/stack?
=transformer_block_3/sequential_3/dense_10/Tensordot/transpose	Transpose;transformer_block_3/sequential_3/dense_9/Relu:activations:0Ctransformer_block_3/sequential_3/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2?
=transformer_block_3/sequential_3/dense_10/Tensordot/transpose?
;transformer_block_3/sequential_3/dense_10/Tensordot/ReshapeReshapeAtransformer_block_3/sequential_3/dense_10/Tensordot/transpose:y:0Btransformer_block_3/sequential_3/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2=
;transformer_block_3/sequential_3/dense_10/Tensordot/Reshape?
:transformer_block_3/sequential_3/dense_10/Tensordot/MatMulMatMulDtransformer_block_3/sequential_3/dense_10/Tensordot/Reshape:output:0Jtransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2<
:transformer_block_3/sequential_3/dense_10/Tensordot/MatMul?
;transformer_block_3/sequential_3/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_3/sequential_3/dense_10/Tensordot/Const_2?
Atransformer_block_3/sequential_3/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_3/sequential_3/dense_10/Tensordot/concat_1/axis?
<transformer_block_3/sequential_3/dense_10/Tensordot/concat_1ConcatV2Etransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2:output:0Dtransformer_block_3/sequential_3/dense_10/Tensordot/Const_2:output:0Jtransformer_block_3/sequential_3/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_3/sequential_3/dense_10/Tensordot/concat_1?
3transformer_block_3/sequential_3/dense_10/TensordotReshapeDtransformer_block_3/sequential_3/dense_10/Tensordot/MatMul:product:0Etransformer_block_3/sequential_3/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 25
3transformer_block_3/sequential_3/dense_10/Tensordot?
@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_3_sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp?
1transformer_block_3/sequential_3/dense_10/BiasAddBiasAdd<transformer_block_3/sequential_3/dense_10/Tensordot:output:0Htransformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 23
1transformer_block_3/sequential_3/dense_10/BiasAdd?
+transformer_block_3/dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2-
+transformer_block_3/dropout_9/dropout/Const?
)transformer_block_3/dropout_9/dropout/MulMul:transformer_block_3/sequential_3/dense_10/BiasAdd:output:04transformer_block_3/dropout_9/dropout/Const:output:0*
T0*+
_output_shapes
:?????????# 2+
)transformer_block_3/dropout_9/dropout/Mul?
+transformer_block_3/dropout_9/dropout/ShapeShape:transformer_block_3/sequential_3/dense_10/BiasAdd:output:0*
T0*
_output_shapes
:2-
+transformer_block_3/dropout_9/dropout/Shape?
Btransformer_block_3/dropout_9/dropout/random_uniform/RandomUniformRandomUniform4transformer_block_3/dropout_9/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????# *
dtype02D
Btransformer_block_3/dropout_9/dropout/random_uniform/RandomUniform?
4transformer_block_3/dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=26
4transformer_block_3/dropout_9/dropout/GreaterEqual/y?
2transformer_block_3/dropout_9/dropout/GreaterEqualGreaterEqualKtransformer_block_3/dropout_9/dropout/random_uniform/RandomUniform:output:0=transformer_block_3/dropout_9/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????# 24
2transformer_block_3/dropout_9/dropout/GreaterEqual?
*transformer_block_3/dropout_9/dropout/CastCast6transformer_block_3/dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????# 2,
*transformer_block_3/dropout_9/dropout/Cast?
+transformer_block_3/dropout_9/dropout/Mul_1Mul-transformer_block_3/dropout_9/dropout/Mul:z:0.transformer_block_3/dropout_9/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????# 2-
+transformer_block_3/dropout_9/dropout/Mul_1?
transformer_block_3/add_1AddV2=transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0/transformer_block_3/dropout_9/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????# 2
transformer_block_3/add_1?
Htransformer_block_3/layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_3/layer_normalization_7/moments/mean/reduction_indices?
6transformer_block_3/layer_normalization_7/moments/meanMeantransformer_block_3/add_1:z:0Qtransformer_block_3/layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(28
6transformer_block_3/layer_normalization_7/moments/mean?
>transformer_block_3/layer_normalization_7/moments/StopGradientStopGradient?transformer_block_3/layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2@
>transformer_block_3/layer_normalization_7/moments/StopGradient?
Ctransformer_block_3/layer_normalization_7/moments/SquaredDifferenceSquaredDifferencetransformer_block_3/add_1:z:0Gtransformer_block_3/layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 2E
Ctransformer_block_3/layer_normalization_7/moments/SquaredDifference?
Ltransformer_block_3/layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_3/layer_normalization_7/moments/variance/reduction_indices?
:transformer_block_3/layer_normalization_7/moments/varianceMeanGtransformer_block_3/layer_normalization_7/moments/SquaredDifference:z:0Utransformer_block_3/layer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2<
:transformer_block_3/layer_normalization_7/moments/variance?
9transformer_block_3/layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52;
9transformer_block_3/layer_normalization_7/batchnorm/add/y?
7transformer_block_3/layer_normalization_7/batchnorm/addAddV2Ctransformer_block_3/layer_normalization_7/moments/variance:output:0Btransformer_block_3/layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#29
7transformer_block_3/layer_normalization_7/batchnorm/add?
9transformer_block_3/layer_normalization_7/batchnorm/RsqrtRsqrt;transformer_block_3/layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2;
9transformer_block_3/layer_normalization_7/batchnorm/Rsqrt?
Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_3_layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp?
7transformer_block_3/layer_normalization_7/batchnorm/mulMul=transformer_block_3/layer_normalization_7/batchnorm/Rsqrt:y:0Ntransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 29
7transformer_block_3/layer_normalization_7/batchnorm/mul?
9transformer_block_3/layer_normalization_7/batchnorm/mul_1Multransformer_block_3/add_1:z:0;transformer_block_3/layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2;
9transformer_block_3/layer_normalization_7/batchnorm/mul_1?
9transformer_block_3/layer_normalization_7/batchnorm/mul_2Mul?transformer_block_3/layer_normalization_7/moments/mean:output:0;transformer_block_3/layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2;
9transformer_block_3/layer_normalization_7/batchnorm/mul_2?
Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_3_layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp?
7transformer_block_3/layer_normalization_7/batchnorm/subSubJtransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp:value:0=transformer_block_3/layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 29
7transformer_block_3/layer_normalization_7/batchnorm/sub?
9transformer_block_3/layer_normalization_7/batchnorm/add_1AddV2=transformer_block_3/layer_normalization_7/batchnorm/mul_1:z:0;transformer_block_3/layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2;
9transformer_block_3/layer_normalization_7/batchnorm/add_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
flatten_1/Const?
flatten_1/ReshapeReshape=transformer_block_3/layer_normalization_7/batchnorm/add_1:z:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshape?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
dense_11/MatMul/ReadVariableOp?
dense_11/MatMulMatMulflatten_1/Reshape:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_11/MatMul?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_11/BiasAdds
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_11/Reluy
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_10/dropout/Const?
dropout_10/dropout/MulMuldense_11/Relu:activations:0!dropout_10/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout_10/dropout/Mul
dropout_10/dropout/ShapeShapedense_11/Relu:activations:0*
T0*
_output_shapes
:2
dropout_10/dropout/Shape?
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype021
/dropout_10/dropout/random_uniform/RandomUniform?
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_10/dropout/GreaterEqual/y?
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2!
dropout_10/dropout/GreaterEqual?
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout_10/dropout/Cast?
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout_10/dropout/Mul_1?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMuldropout_10/dropout/Mul_1:z:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_12/Reluy
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_11/dropout/Const?
dropout_11/dropout/MulMuldense_12/Relu:activations:0!dropout_11/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout_11/dropout/Mul
dropout_11/dropout/ShapeShapedense_12/Relu:activations:0*
T0*
_output_shapes
:2
dropout_11/dropout/Shape?
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype021
/dropout_11/dropout/random_uniform/RandomUniform?
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_11/dropout/GreaterEqual/y?
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2!
dropout_11/dropout/GreaterEqual?
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout_11/dropout/Cast?
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout_11/dropout/Mul_1?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMuldropout_11/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/BiasAdd?
IdentityIdentitydense_13/BiasAdd:output:0 ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp<^token_and_position_embedding_1/embedding_2/embedding_lookup<^token_and_position_embedding_1/embedding_3/embedding_lookupC^transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpG^transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpC^transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpG^transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpO^transformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpY^transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_3/multi_head_attention_3/key/add/ReadVariableOpL^transformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpD^transformer_block_3/multi_head_attention_3/query/add/ReadVariableOpN^transformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpD^transformer_block_3/multi_head_attention_3/value/add/ReadVariableOpN^transformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpA^transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOpC^transformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp@^transformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOpB^transformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesv
t:??????????R::::::::::::::::::::::::2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2z
;token_and_position_embedding_1/embedding_2/embedding_lookup;token_and_position_embedding_1/embedding_2/embedding_lookup2z
;token_and_position_embedding_1/embedding_3/embedding_lookup;token_and_position_embedding_1/embedding_3/embedding_lookup2?
Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpBtransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp2?
Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpFtransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp2?
Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpBtransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp2?
Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpFtransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp2?
Ntransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpNtransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp2?
Xtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp2?
Atransformer_block_3/multi_head_attention_3/key/add/ReadVariableOpAtransformer_block_3/multi_head_attention_3/key/add/ReadVariableOp2?
Ktransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpKtransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp2?
Ctransformer_block_3/multi_head_attention_3/query/add/ReadVariableOpCtransformer_block_3/multi_head_attention_3/query/add/ReadVariableOp2?
Mtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpMtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp2?
Ctransformer_block_3/multi_head_attention_3/value/add/ReadVariableOpCtransformer_block_3/multi_head_attention_3/value/add/ReadVariableOp2?
Mtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpMtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp2?
@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp2?
Btransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOpBtransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp2?
?transformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOp?transformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOp2?
Atransformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOpAtransformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOp:P L
(
_output_shapes
:??????????R
 
_user_specified_nameinputs
?1
?
B__inference_model_1_layer_call_and_return_conditional_losses_12934
input_2(
$token_and_position_embedding_1_12876(
$token_and_position_embedding_1_12878
transformer_block_3_12882
transformer_block_3_12884
transformer_block_3_12886
transformer_block_3_12888
transformer_block_3_12890
transformer_block_3_12892
transformer_block_3_12894
transformer_block_3_12896
transformer_block_3_12898
transformer_block_3_12900
transformer_block_3_12902
transformer_block_3_12904
transformer_block_3_12906
transformer_block_3_12908
transformer_block_3_12910
transformer_block_3_12912
dense_11_12916
dense_11_12918
dense_12_12922
dense_12_12924
dense_13_12928
dense_13_12930
identity?? dense_11/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?6token_and_position_embedding_1/StatefulPartitionedCall?+transformer_block_3/StatefulPartitionedCall?
6token_and_position_embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_2$token_and_position_embedding_1_12876$token_and_position_embedding_1_12878*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *b
f]R[
Y__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_1231728
6token_and_position_embedding_1/StatefulPartitionedCall?
#average_pooling1d_1/PartitionedCallPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_121162%
#average_pooling1d_1/PartitionedCall?
+transformer_block_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_1/PartitionedCall:output:0transformer_block_3_12882transformer_block_3_12884transformer_block_3_12886transformer_block_3_12888transformer_block_3_12890transformer_block_3_12892transformer_block_3_12894transformer_block_3_12896transformer_block_3_12898transformer_block_3_12900transformer_block_3_12902transformer_block_3_12904transformer_block_3_12906transformer_block_3_12908transformer_block_3_12910transformer_block_3_12912*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_126092-
+transformer_block_3/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall4transformer_block_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_127242
flatten_1/PartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_11_12916dense_11_12918*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_127432"
 dense_11/StatefulPartitionedCall?
dropout_10/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_127762
dropout_10/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0dense_12_12922dense_12_12924*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_128002"
 dense_12/StatefulPartitionedCall?
dropout_11/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_128332
dropout_11/PartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_13_12928dense_13_12930*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_128562"
 dense_13/StatefulPartitionedCall?
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall7^token_and_position_embedding_1/StatefulPartitionedCall,^transformer_block_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesv
t:??????????R::::::::::::::::::::::::2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2p
6token_and_position_embedding_1/StatefulPartitionedCall6token_and_position_embedding_1/StatefulPartitionedCall2Z
+transformer_block_3/StatefulPartitionedCall+transformer_block_3/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????R
!
_user_specified_name	input_2
?
?
Y__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_12317
x&
"embedding_3_embedding_lookup_12304&
"embedding_2_embedding_lookup_12310
identity??embedding_2/embedding_lookup?embedding_3/embedding_lookup?
ShapeShapex*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/delta?
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:?????????2
range?
embedding_3/embedding_lookupResourceGather"embedding_3_embedding_lookup_12304range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_3/embedding_lookup/12304*'
_output_shapes
:????????? *
dtype02
embedding_3/embedding_lookup?
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_3/embedding_lookup/12304*'
_output_shapes
:????????? 2'
%embedding_3/embedding_lookup/Identity?
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2)
'embedding_3/embedding_lookup/Identity_1q
embedding_2/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:??????????R2
embedding_2/Cast?
embedding_2/embedding_lookupResourceGather"embedding_2_embedding_lookup_12310embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_2/embedding_lookup/12310*,
_output_shapes
:??????????R *
dtype02
embedding_2/embedding_lookup?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_2/embedding_lookup/12310*,
_output_shapes
:??????????R 2'
%embedding_2/embedding_lookup/Identity?
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????R 2)
'embedding_2/embedding_lookup/Identity_1?
addAddV20embedding_2/embedding_lookup/Identity_1:output:00embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????R 2
add?
IdentityIdentityadd:z:0^embedding_2/embedding_lookup^embedding_3/embedding_lookup*
T0*,
_output_shapes
:??????????R 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????R::2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2<
embedding_3/embedding_lookupembedding_3/embedding_lookup:K G
(
_output_shapes
:??????????R

_user_specified_namex
?
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_14103

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????# :S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
?
c
E__inference_dropout_10_layer_call_and_return_conditional_losses_12776

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
c
E__inference_dropout_10_layer_call_and_return_conditional_losses_14145

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
>__inference_token_and_position_embedding_1_layer_call_fn_13748
x
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *b
f]R[
Y__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_123172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????R 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????R::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????R

_user_specified_namex
? 
?
B__inference_dense_9_layer_call_and_return_conditional_losses_14392

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????# 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????#@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????#@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????# ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
?
}
(__inference_dense_10_layer_call_fn_14440

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_122032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????#@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????#@
 
_user_specified_nameinputs
?
?
C__inference_dense_10_layer_call_and_return_conditional_losses_14431

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????#@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????#@
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_13224
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_121072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesv
t:??????????R::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????R
!
_user_specified_name	input_2
?
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_12724

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????# :S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
?	
?
C__inference_dense_12_layer_call_and_return_conditional_losses_12800

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_12251

inputs
dense_9_12240
dense_9_12242
dense_10_12245
dense_10_12247
identity?? dense_10/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCallinputsdense_9_12240dense_9_12242*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_121572!
dense_9/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_12245dense_10_12247*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_122032"
 dense_10/StatefulPartitionedCall?
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
?4
?
B__inference_model_1_layer_call_and_return_conditional_losses_12998

inputs(
$token_and_position_embedding_1_12940(
$token_and_position_embedding_1_12942
transformer_block_3_12946
transformer_block_3_12948
transformer_block_3_12950
transformer_block_3_12952
transformer_block_3_12954
transformer_block_3_12956
transformer_block_3_12958
transformer_block_3_12960
transformer_block_3_12962
transformer_block_3_12964
transformer_block_3_12966
transformer_block_3_12968
transformer_block_3_12970
transformer_block_3_12972
transformer_block_3_12974
transformer_block_3_12976
dense_11_12980
dense_11_12982
dense_12_12986
dense_12_12988
dense_13_12992
dense_13_12994
identity?? dense_11/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?"dropout_10/StatefulPartitionedCall?"dropout_11/StatefulPartitionedCall?6token_and_position_embedding_1/StatefulPartitionedCall?+transformer_block_3/StatefulPartitionedCall?
6token_and_position_embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs$token_and_position_embedding_1_12940$token_and_position_embedding_1_12942*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *b
f]R[
Y__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_1231728
6token_and_position_embedding_1/StatefulPartitionedCall?
#average_pooling1d_1/PartitionedCallPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_121162%
#average_pooling1d_1/PartitionedCall?
+transformer_block_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_1/PartitionedCall:output:0transformer_block_3_12946transformer_block_3_12948transformer_block_3_12950transformer_block_3_12952transformer_block_3_12954transformer_block_3_12956transformer_block_3_12958transformer_block_3_12960transformer_block_3_12962transformer_block_3_12964transformer_block_3_12966transformer_block_3_12968transformer_block_3_12970transformer_block_3_12972transformer_block_3_12974transformer_block_3_12976*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_124822-
+transformer_block_3/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall4transformer_block_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_127242
flatten_1/PartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_11_12980dense_11_12982*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_127432"
 dense_11/StatefulPartitionedCall?
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_127712$
"dropout_10/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0dense_12_12986dense_12_12988*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_128002"
 dense_12/StatefulPartitionedCall?
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_128282$
"dropout_11/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_13_12992dense_13_12994*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_128562"
 dense_13/StatefulPartitionedCall?
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall7^token_and_position_embedding_1/StatefulPartitionedCall,^transformer_block_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesv
t:??????????R::::::::::::::::::::::::2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2p
6token_and_position_embedding_1/StatefulPartitionedCall6token_and_position_embedding_1/StatefulPartitionedCall2Z
+transformer_block_3/StatefulPartitionedCall+transformer_block_3/StatefulPartitionedCall:P L
(
_output_shapes
:??????????R
 
_user_specified_nameinputs
??
?
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_12482

inputsF
Bmulti_head_attention_3_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_3_query_add_readvariableop_resourceD
@multi_head_attention_3_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_3_key_add_readvariableop_resourceF
Bmulti_head_attention_3_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_3_value_add_readvariableop_resourceQ
Mmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_3_attention_output_add_readvariableop_resource?
;layer_normalization_6_batchnorm_mul_readvariableop_resource;
7layer_normalization_6_batchnorm_readvariableop_resource:
6sequential_3_dense_9_tensordot_readvariableop_resource8
4sequential_3_dense_9_biasadd_readvariableop_resource;
7sequential_3_dense_10_tensordot_readvariableop_resource9
5sequential_3_dense_10_biasadd_readvariableop_resource?
;layer_normalization_7_batchnorm_mul_readvariableop_resource;
7layer_normalization_7_batchnorm_readvariableop_resource
identity??.layer_normalization_6/batchnorm/ReadVariableOp?2layer_normalization_6/batchnorm/mul/ReadVariableOp?.layer_normalization_7/batchnorm/ReadVariableOp?2layer_normalization_7/batchnorm/mul/ReadVariableOp?:multi_head_attention_3/attention_output/add/ReadVariableOp?Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp?-multi_head_attention_3/key/add/ReadVariableOp?7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp?/multi_head_attention_3/query/add/ReadVariableOp?9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp?/multi_head_attention_3/value/add/ReadVariableOp?9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp?,sequential_3/dense_10/BiasAdd/ReadVariableOp?.sequential_3/dense_10/Tensordot/ReadVariableOp?+sequential_3/dense_9/BiasAdd/ReadVariableOp?-sequential_3/dense_9/Tensordot/ReadVariableOp?
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp?
*multi_head_attention_3/query/einsum/EinsumEinsuminputsAmulti_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2,
*multi_head_attention_3/query/einsum/Einsum?
/multi_head_attention_3/query/add/ReadVariableOpReadVariableOp8multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/query/add/ReadVariableOp?
 multi_head_attention_3/query/addAddV23multi_head_attention_3/query/einsum/Einsum:output:07multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2"
 multi_head_attention_3/query/add?
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp?
(multi_head_attention_3/key/einsum/EinsumEinsuminputs?multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2*
(multi_head_attention_3/key/einsum/Einsum?
-multi_head_attention_3/key/add/ReadVariableOpReadVariableOp6multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_3/key/add/ReadVariableOp?
multi_head_attention_3/key/addAddV21multi_head_attention_3/key/einsum/Einsum:output:05multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2 
multi_head_attention_3/key/add?
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp?
*multi_head_attention_3/value/einsum/EinsumEinsuminputsAmulti_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2,
*multi_head_attention_3/value/einsum/Einsum?
/multi_head_attention_3/value/add/ReadVariableOpReadVariableOp8multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/value/add/ReadVariableOp?
 multi_head_attention_3/value/addAddV23multi_head_attention_3/value/einsum/Einsum:output:07multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2"
 multi_head_attention_3/value/add?
multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>2
multi_head_attention_3/Mul/y?
multi_head_attention_3/MulMul$multi_head_attention_3/query/add:z:0%multi_head_attention_3/Mul/y:output:0*
T0*/
_output_shapes
:?????????# 2
multi_head_attention_3/Mul?
$multi_head_attention_3/einsum/EinsumEinsum"multi_head_attention_3/key/add:z:0multi_head_attention_3/Mul:z:0*
N*
T0*/
_output_shapes
:?????????##*
equationaecd,abcd->acbe2&
$multi_head_attention_3/einsum/Einsum?
&multi_head_attention_3/softmax/SoftmaxSoftmax-multi_head_attention_3/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????##2(
&multi_head_attention_3/softmax/Softmax?
,multi_head_attention_3/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,multi_head_attention_3/dropout/dropout/Const?
*multi_head_attention_3/dropout/dropout/MulMul0multi_head_attention_3/softmax/Softmax:softmax:05multi_head_attention_3/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????##2,
*multi_head_attention_3/dropout/dropout/Mul?
,multi_head_attention_3/dropout/dropout/ShapeShape0multi_head_attention_3/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_3/dropout/dropout/Shape?
Cmulti_head_attention_3/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_3/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????##*
dtype02E
Cmulti_head_attention_3/dropout/dropout/random_uniform/RandomUniform?
5multi_head_attention_3/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_3/dropout/dropout/GreaterEqual/y?
3multi_head_attention_3/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_3/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_3/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????##25
3multi_head_attention_3/dropout/dropout/GreaterEqual?
+multi_head_attention_3/dropout/dropout/CastCast7multi_head_attention_3/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????##2-
+multi_head_attention_3/dropout/dropout/Cast?
,multi_head_attention_3/dropout/dropout/Mul_1Mul.multi_head_attention_3/dropout/dropout/Mul:z:0/multi_head_attention_3/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????##2.
,multi_head_attention_3/dropout/dropout/Mul_1?
&multi_head_attention_3/einsum_1/EinsumEinsum0multi_head_attention_3/dropout/dropout/Mul_1:z:0$multi_head_attention_3/value/add:z:0*
N*
T0*/
_output_shapes
:?????????# *
equationacbe,aecd->abcd2(
&multi_head_attention_3/einsum_1/Einsum?
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp?
5multi_head_attention_3/attention_output/einsum/EinsumEinsum/multi_head_attention_3/einsum_1/Einsum:output:0Lmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????# *
equationabcd,cde->abe27
5multi_head_attention_3/attention_output/einsum/Einsum?
:multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_3/attention_output/add/ReadVariableOp?
+multi_head_attention_3/attention_output/addAddV2>multi_head_attention_3/attention_output/einsum/Einsum:output:0Bmulti_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2-
+multi_head_attention_3/attention_output/addw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_8/dropout/Const?
dropout_8/dropout/MulMul/multi_head_attention_3/attention_output/add:z:0 dropout_8/dropout/Const:output:0*
T0*+
_output_shapes
:?????????# 2
dropout_8/dropout/Mul?
dropout_8/dropout/ShapeShape/multi_head_attention_3/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape?
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????# *
dtype020
.dropout_8/dropout/random_uniform/RandomUniform?
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_8/dropout/GreaterEqual/y?
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????# 2 
dropout_8/dropout/GreaterEqual?
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????# 2
dropout_8/dropout/Cast?
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????# 2
dropout_8/dropout/Mul_1n
addAddV2inputsdropout_8/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????# 2
add?
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_6/moments/mean/reduction_indices?
"layer_normalization_6/moments/meanMeanadd:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2$
"layer_normalization_6/moments/mean?
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2,
*layer_normalization_6/moments/StopGradient?
/layer_normalization_6/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 21
/layer_normalization_6/moments/SquaredDifference?
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_6/moments/variance/reduction_indices?
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2(
&layer_normalization_6/moments/variance?
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_6/batchnorm/add/y?
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2%
#layer_normalization_6/batchnorm/add?
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2'
%layer_normalization_6/batchnorm/Rsqrt?
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_6/batchnorm/mul/ReadVariableOp?
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2%
#layer_normalization_6/batchnorm/mul?
%layer_normalization_6/batchnorm/mul_1Muladd:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_6/batchnorm/mul_1?
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_6/batchnorm/mul_2?
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_6/batchnorm/ReadVariableOp?
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2%
#layer_normalization_6/batchnorm/sub?
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_6/batchnorm/add_1?
-sequential_3/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_3_dense_9_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02/
-sequential_3/dense_9/Tensordot/ReadVariableOp?
#sequential_3/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_3/dense_9/Tensordot/axes?
#sequential_3/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_3/dense_9/Tensordot/free?
$sequential_3/dense_9/Tensordot/ShapeShape)layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_3/dense_9/Tensordot/Shape?
,sequential_3/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_3/dense_9/Tensordot/GatherV2/axis?
'sequential_3/dense_9/Tensordot/GatherV2GatherV2-sequential_3/dense_9/Tensordot/Shape:output:0,sequential_3/dense_9/Tensordot/free:output:05sequential_3/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_3/dense_9/Tensordot/GatherV2?
.sequential_3/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_3/dense_9/Tensordot/GatherV2_1/axis?
)sequential_3/dense_9/Tensordot/GatherV2_1GatherV2-sequential_3/dense_9/Tensordot/Shape:output:0,sequential_3/dense_9/Tensordot/axes:output:07sequential_3/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_3/dense_9/Tensordot/GatherV2_1?
$sequential_3/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_3/dense_9/Tensordot/Const?
#sequential_3/dense_9/Tensordot/ProdProd0sequential_3/dense_9/Tensordot/GatherV2:output:0-sequential_3/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_3/dense_9/Tensordot/Prod?
&sequential_3/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_3/dense_9/Tensordot/Const_1?
%sequential_3/dense_9/Tensordot/Prod_1Prod2sequential_3/dense_9/Tensordot/GatherV2_1:output:0/sequential_3/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_3/dense_9/Tensordot/Prod_1?
*sequential_3/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_3/dense_9/Tensordot/concat/axis?
%sequential_3/dense_9/Tensordot/concatConcatV2,sequential_3/dense_9/Tensordot/free:output:0,sequential_3/dense_9/Tensordot/axes:output:03sequential_3/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_9/Tensordot/concat?
$sequential_3/dense_9/Tensordot/stackPack,sequential_3/dense_9/Tensordot/Prod:output:0.sequential_3/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_3/dense_9/Tensordot/stack?
(sequential_3/dense_9/Tensordot/transpose	Transpose)layer_normalization_6/batchnorm/add_1:z:0.sequential_3/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????# 2*
(sequential_3/dense_9/Tensordot/transpose?
&sequential_3/dense_9/Tensordot/ReshapeReshape,sequential_3/dense_9/Tensordot/transpose:y:0-sequential_3/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&sequential_3/dense_9/Tensordot/Reshape?
%sequential_3/dense_9/Tensordot/MatMulMatMul/sequential_3/dense_9/Tensordot/Reshape:output:05sequential_3/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2'
%sequential_3/dense_9/Tensordot/MatMul?
&sequential_3/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2(
&sequential_3/dense_9/Tensordot/Const_2?
,sequential_3/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_3/dense_9/Tensordot/concat_1/axis?
'sequential_3/dense_9/Tensordot/concat_1ConcatV20sequential_3/dense_9/Tensordot/GatherV2:output:0/sequential_3/dense_9/Tensordot/Const_2:output:05sequential_3/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_3/dense_9/Tensordot/concat_1?
sequential_3/dense_9/TensordotReshape/sequential_3/dense_9/Tensordot/MatMul:product:00sequential_3/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@2 
sequential_3/dense_9/Tensordot?
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp?
sequential_3/dense_9/BiasAddBiasAdd'sequential_3/dense_9/Tensordot:output:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@2
sequential_3/dense_9/BiasAdd?
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#@2
sequential_3/dense_9/Relu?
.sequential_3/dense_10/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_10_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_3/dense_10/Tensordot/ReadVariableOp?
$sequential_3/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_3/dense_10/Tensordot/axes?
$sequential_3/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_3/dense_10/Tensordot/free?
%sequential_3/dense_10/Tensordot/ShapeShape'sequential_3/dense_9/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/Shape?
-sequential_3/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/GatherV2/axis?
(sequential_3/dense_10/Tensordot/GatherV2GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/free:output:06sequential_3/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/GatherV2?
/sequential_3/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_3/dense_10/Tensordot/GatherV2_1/axis?
*sequential_3/dense_10/Tensordot/GatherV2_1GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/axes:output:08sequential_3/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_3/dense_10/Tensordot/GatherV2_1?
%sequential_3/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_3/dense_10/Tensordot/Const?
$sequential_3/dense_10/Tensordot/ProdProd1sequential_3/dense_10/Tensordot/GatherV2:output:0.sequential_3/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_3/dense_10/Tensordot/Prod?
'sequential_3/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_1?
&sequential_3/dense_10/Tensordot/Prod_1Prod3sequential_3/dense_10/Tensordot/GatherV2_1:output:00sequential_3/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_3/dense_10/Tensordot/Prod_1?
+sequential_3/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_3/dense_10/Tensordot/concat/axis?
&sequential_3/dense_10/Tensordot/concatConcatV2-sequential_3/dense_10/Tensordot/free:output:0-sequential_3/dense_10/Tensordot/axes:output:04sequential_3/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_3/dense_10/Tensordot/concat?
%sequential_3/dense_10/Tensordot/stackPack-sequential_3/dense_10/Tensordot/Prod:output:0/sequential_3/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/stack?
)sequential_3/dense_10/Tensordot/transpose	Transpose'sequential_3/dense_9/Relu:activations:0/sequential_3/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2+
)sequential_3/dense_10/Tensordot/transpose?
'sequential_3/dense_10/Tensordot/ReshapeReshape-sequential_3/dense_10/Tensordot/transpose:y:0.sequential_3/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_3/dense_10/Tensordot/Reshape?
&sequential_3/dense_10/Tensordot/MatMulMatMul0sequential_3/dense_10/Tensordot/Reshape:output:06sequential_3/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_3/dense_10/Tensordot/MatMul?
'sequential_3/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_2?
-sequential_3/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/concat_1/axis?
(sequential_3/dense_10/Tensordot/concat_1ConcatV21sequential_3/dense_10/Tensordot/GatherV2:output:00sequential_3/dense_10/Tensordot/Const_2:output:06sequential_3/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/concat_1?
sequential_3/dense_10/TensordotReshape0sequential_3/dense_10/Tensordot/MatMul:product:01sequential_3/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 2!
sequential_3/dense_10/Tensordot?
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp?
sequential_3/dense_10/BiasAddBiasAdd(sequential_3/dense_10/Tensordot:output:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2
sequential_3/dense_10/BiasAddw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_9/dropout/Const?
dropout_9/dropout/MulMul&sequential_3/dense_10/BiasAdd:output:0 dropout_9/dropout/Const:output:0*
T0*+
_output_shapes
:?????????# 2
dropout_9/dropout/Mul?
dropout_9/dropout/ShapeShape&sequential_3/dense_10/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape?
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????# *
dtype020
.dropout_9/dropout/random_uniform/RandomUniform?
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_9/dropout/GreaterEqual/y?
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????# 2 
dropout_9/dropout/GreaterEqual?
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????# 2
dropout_9/dropout/Cast?
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????# 2
dropout_9/dropout/Mul_1?
add_1AddV2)layer_normalization_6/batchnorm/add_1:z:0dropout_9/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????# 2
add_1?
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_7/moments/mean/reduction_indices?
"layer_normalization_7/moments/meanMean	add_1:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2$
"layer_normalization_7/moments/mean?
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2,
*layer_normalization_7/moments/StopGradient?
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 21
/layer_normalization_7/moments/SquaredDifference?
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_7/moments/variance/reduction_indices?
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2(
&layer_normalization_7/moments/variance?
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_7/batchnorm/add/y?
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2%
#layer_normalization_7/batchnorm/add?
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2'
%layer_normalization_7/batchnorm/Rsqrt?
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_7/batchnorm/mul/ReadVariableOp?
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2%
#layer_normalization_7/batchnorm/mul?
%layer_normalization_7/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_7/batchnorm/mul_1?
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_7/batchnorm/mul_2?
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_7/batchnorm/ReadVariableOp?
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2%
#layer_normalization_7/batchnorm/sub?
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_7/batchnorm/add_1?
IdentityIdentity)layer_normalization_7/batchnorm/add_1:z:0/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp;^multi_head_attention_3/attention_output/add/ReadVariableOpE^multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_3/key/add/ReadVariableOp8^multi_head_attention_3/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/query/add/ReadVariableOp:^multi_head_attention_3/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/value/add/ReadVariableOp:^multi_head_attention_3/value/einsum/Einsum/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp/^sequential_3/dense_10/Tensordot/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp.^sequential_3/dense_9/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:?????????# ::::::::::::::::2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_3/attention_output/add/ReadVariableOp:multi_head_attention_3/attention_output/add/ReadVariableOp2?
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_3/key/add/ReadVariableOp-multi_head_attention_3/key/add/ReadVariableOp2r
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/query/add/ReadVariableOp/multi_head_attention_3/query/add/ReadVariableOp2v
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/value/add/ReadVariableOp/multi_head_attention_3/value/add/ReadVariableOp2v
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2`
.sequential_3/dense_10/Tensordot/ReadVariableOp.sequential_3/dense_10/Tensordot/ReadVariableOp2Z
+sequential_3/dense_9/BiasAdd/ReadVariableOp+sequential_3/dense_9/BiasAdd/ReadVariableOp2^
-sequential_3/dense_9/Tensordot/ReadVariableOp-sequential_3/dense_9/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
?	
?
C__inference_dense_13_layer_call_and_return_conditional_losses_12856

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_13163
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_131122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesv
t:??????????R::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????R
!
_user_specified_name	input_2
?

?
3__inference_transformer_block_3_layer_call_fn_14097

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_126092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:?????????# ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
?
d
E__inference_dropout_11_layer_call_and_return_conditional_losses_14187

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
E__inference_dropout_10_layer_call_and_return_conditional_losses_12771

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
Y__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_13739
x&
"embedding_3_embedding_lookup_13726&
"embedding_2_embedding_lookup_13732
identity??embedding_2/embedding_lookup?embedding_3/embedding_lookup?
ShapeShapex*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/delta?
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:?????????2
range?
embedding_3/embedding_lookupResourceGather"embedding_3_embedding_lookup_13726range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_3/embedding_lookup/13726*'
_output_shapes
:????????? *
dtype02
embedding_3/embedding_lookup?
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_3/embedding_lookup/13726*'
_output_shapes
:????????? 2'
%embedding_3/embedding_lookup/Identity?
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2)
'embedding_3/embedding_lookup/Identity_1q
embedding_2/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:??????????R2
embedding_2/Cast?
embedding_2/embedding_lookupResourceGather"embedding_2_embedding_lookup_13732embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_2/embedding_lookup/13732*,
_output_shapes
:??????????R *
dtype02
embedding_2/embedding_lookup?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_2/embedding_lookup/13732*,
_output_shapes
:??????????R 2'
%embedding_2/embedding_lookup/Identity?
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????R 2)
'embedding_2/embedding_lookup/Identity_1?
addAddV20embedding_2/embedding_lookup/Identity_1:output:00embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????R 2
add?
IdentityIdentityadd:z:0^embedding_2/embedding_lookup^embedding_3/embedding_lookup*
T0*,
_output_shapes
:??????????R 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????R::2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2<
embedding_3/embedding_lookupembedding_3/embedding_lookup:K G
(
_output_shapes
:??????????R

_user_specified_namex
?
d
E__inference_dropout_11_layer_call_and_return_conditional_losses_12828

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?{
?
__inference__traced_save_14625
file_prefix.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop'
#savev2_momentum_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	T
Psavev2_token_and_position_embedding_1_embedding_2_embeddings_read_readvariableopT
Psavev2_token_and_position_embedding_1_embedding_3_embeddings_read_readvariableopV
Rsavev2_transformer_block_3_multi_head_attention_3_query_kernel_read_readvariableopT
Psavev2_transformer_block_3_multi_head_attention_3_query_bias_read_readvariableopT
Psavev2_transformer_block_3_multi_head_attention_3_key_kernel_read_readvariableopR
Nsavev2_transformer_block_3_multi_head_attention_3_key_bias_read_readvariableopV
Rsavev2_transformer_block_3_multi_head_attention_3_value_kernel_read_readvariableopT
Psavev2_transformer_block_3_multi_head_attention_3_value_bias_read_readvariableopa
]savev2_transformer_block_3_multi_head_attention_3_attention_output_kernel_read_readvariableop_
[savev2_transformer_block_3_multi_head_attention_3_attention_output_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableopN
Jsavev2_transformer_block_3_layer_normalization_6_gamma_read_readvariableopM
Isavev2_transformer_block_3_layer_normalization_6_beta_read_readvariableopN
Jsavev2_transformer_block_3_layer_normalization_7_gamma_read_readvariableopM
Isavev2_transformer_block_3_layer_normalization_7_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop;
7savev2_sgd_dense_11_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_11_bias_momentum_read_readvariableop;
7savev2_sgd_dense_12_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_12_bias_momentum_read_readvariableop;
7savev2_sgd_dense_13_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_13_bias_momentum_read_readvariableopa
]savev2_sgd_token_and_position_embedding_1_embedding_2_embeddings_momentum_read_readvariableopa
]savev2_sgd_token_and_position_embedding_1_embedding_3_embeddings_momentum_read_readvariableopc
_savev2_sgd_transformer_block_3_multi_head_attention_3_query_kernel_momentum_read_readvariableopa
]savev2_sgd_transformer_block_3_multi_head_attention_3_query_bias_momentum_read_readvariableopa
]savev2_sgd_transformer_block_3_multi_head_attention_3_key_kernel_momentum_read_readvariableop_
[savev2_sgd_transformer_block_3_multi_head_attention_3_key_bias_momentum_read_readvariableopc
_savev2_sgd_transformer_block_3_multi_head_attention_3_value_kernel_momentum_read_readvariableopa
]savev2_sgd_transformer_block_3_multi_head_attention_3_value_bias_momentum_read_readvariableopn
jsavev2_sgd_transformer_block_3_multi_head_attention_3_attention_output_kernel_momentum_read_readvariableopl
hsavev2_sgd_transformer_block_3_multi_head_attention_3_attention_output_bias_momentum_read_readvariableop:
6savev2_sgd_dense_9_kernel_momentum_read_readvariableop8
4savev2_sgd_dense_9_bias_momentum_read_readvariableop;
7savev2_sgd_dense_10_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_10_bias_momentum_read_readvariableop[
Wsavev2_sgd_transformer_block_3_layer_normalization_6_gamma_momentum_read_readvariableopZ
Vsavev2_sgd_transformer_block_3_layer_normalization_6_beta_momentum_read_readvariableop[
Wsavev2_sgd_transformer_block_3_layer_normalization_7_gamma_momentum_read_readvariableopZ
Vsavev2_sgd_transformer_block_3_layer_normalization_7_beta_momentum_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
value?B?7B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/8/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/9/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableopPsavev2_token_and_position_embedding_1_embedding_2_embeddings_read_readvariableopPsavev2_token_and_position_embedding_1_embedding_3_embeddings_read_readvariableopRsavev2_transformer_block_3_multi_head_attention_3_query_kernel_read_readvariableopPsavev2_transformer_block_3_multi_head_attention_3_query_bias_read_readvariableopPsavev2_transformer_block_3_multi_head_attention_3_key_kernel_read_readvariableopNsavev2_transformer_block_3_multi_head_attention_3_key_bias_read_readvariableopRsavev2_transformer_block_3_multi_head_attention_3_value_kernel_read_readvariableopPsavev2_transformer_block_3_multi_head_attention_3_value_bias_read_readvariableop]savev2_transformer_block_3_multi_head_attention_3_attention_output_kernel_read_readvariableop[savev2_transformer_block_3_multi_head_attention_3_attention_output_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableopJsavev2_transformer_block_3_layer_normalization_6_gamma_read_readvariableopIsavev2_transformer_block_3_layer_normalization_6_beta_read_readvariableopJsavev2_transformer_block_3_layer_normalization_7_gamma_read_readvariableopIsavev2_transformer_block_3_layer_normalization_7_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop7savev2_sgd_dense_11_kernel_momentum_read_readvariableop5savev2_sgd_dense_11_bias_momentum_read_readvariableop7savev2_sgd_dense_12_kernel_momentum_read_readvariableop5savev2_sgd_dense_12_bias_momentum_read_readvariableop7savev2_sgd_dense_13_kernel_momentum_read_readvariableop5savev2_sgd_dense_13_bias_momentum_read_readvariableop]savev2_sgd_token_and_position_embedding_1_embedding_2_embeddings_momentum_read_readvariableop]savev2_sgd_token_and_position_embedding_1_embedding_3_embeddings_momentum_read_readvariableop_savev2_sgd_transformer_block_3_multi_head_attention_3_query_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_3_multi_head_attention_3_query_bias_momentum_read_readvariableop]savev2_sgd_transformer_block_3_multi_head_attention_3_key_kernel_momentum_read_readvariableop[savev2_sgd_transformer_block_3_multi_head_attention_3_key_bias_momentum_read_readvariableop_savev2_sgd_transformer_block_3_multi_head_attention_3_value_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_3_multi_head_attention_3_value_bias_momentum_read_readvariableopjsavev2_sgd_transformer_block_3_multi_head_attention_3_attention_output_kernel_momentum_read_readvariableophsavev2_sgd_transformer_block_3_multi_head_attention_3_attention_output_bias_momentum_read_readvariableop6savev2_sgd_dense_9_kernel_momentum_read_readvariableop4savev2_sgd_dense_9_bias_momentum_read_readvariableop7savev2_sgd_dense_10_kernel_momentum_read_readvariableop5savev2_sgd_dense_10_bias_momentum_read_readvariableopWsavev2_sgd_transformer_block_3_layer_normalization_6_gamma_momentum_read_readvariableopVsavev2_sgd_transformer_block_3_layer_normalization_6_beta_momentum_read_readvariableopWsavev2_sgd_transformer_block_3_layer_normalization_7_gamma_momentum_read_readvariableopVsavev2_sgd_transformer_block_3_layer_normalization_7_beta_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *E
dtypes;
927	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?@:@:@@:@:@:: : : : : :	?R :  : :  : :  : :  : : @:@:@ : : : : : : : :	?@:@:@@:@:@:: :	?R :  : :  : :  : :  : : @:@:@ : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :$ 

_output_shapes

: :%!

_output_shapes
:	?R :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?@:  

_output_shapes
:@:$! 

_output_shapes

:@@: "

_output_shapes
:@:$# 

_output_shapes

:@: $

_output_shapes
::$% 

_output_shapes

: :%&!

_output_shapes
:	?R :('$
"
_output_shapes
:  :$( 

_output_shapes

: :()$
"
_output_shapes
:  :$* 

_output_shapes

: :(+$
"
_output_shapes
:  :$, 

_output_shapes

: :(-$
"
_output_shapes
:  : .

_output_shapes
: :$/ 

_output_shapes

: @: 0

_output_shapes
:@:$1 

_output_shapes

:@ : 2

_output_shapes
: : 3

_output_shapes
: : 4

_output_shapes
: : 5

_output_shapes
: : 6

_output_shapes
: :7

_output_shapes
: 
?
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_12234
dense_9_input
dense_9_12223
dense_9_12225
dense_10_12228
dense_10_12230
identity?? dense_10/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCalldense_9_inputdense_9_12223dense_9_12225*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_121572!
dense_9/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_12228dense_10_12230*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_122032"
 dense_10/StatefulPartitionedCall?
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????# 
'
_user_specified_namedense_9_input
?
F
*__inference_dropout_11_layer_call_fn_14202

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_128332
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?%
!__inference__traced_restore_14797
file_prefix$
 assignvariableop_dense_11_kernel$
 assignvariableop_1_dense_11_bias&
"assignvariableop_2_dense_12_kernel$
 assignvariableop_3_dense_12_bias&
"assignvariableop_4_dense_13_kernel$
 assignvariableop_5_dense_13_bias
assignvariableop_6_decay$
 assignvariableop_7_learning_rate
assignvariableop_8_momentum
assignvariableop_9_sgd_iterM
Iassignvariableop_10_token_and_position_embedding_1_embedding_2_embeddingsM
Iassignvariableop_11_token_and_position_embedding_1_embedding_3_embeddingsO
Kassignvariableop_12_transformer_block_3_multi_head_attention_3_query_kernelM
Iassignvariableop_13_transformer_block_3_multi_head_attention_3_query_biasM
Iassignvariableop_14_transformer_block_3_multi_head_attention_3_key_kernelK
Gassignvariableop_15_transformer_block_3_multi_head_attention_3_key_biasO
Kassignvariableop_16_transformer_block_3_multi_head_attention_3_value_kernelM
Iassignvariableop_17_transformer_block_3_multi_head_attention_3_value_biasZ
Vassignvariableop_18_transformer_block_3_multi_head_attention_3_attention_output_kernelX
Tassignvariableop_19_transformer_block_3_multi_head_attention_3_attention_output_bias&
"assignvariableop_20_dense_9_kernel$
 assignvariableop_21_dense_9_bias'
#assignvariableop_22_dense_10_kernel%
!assignvariableop_23_dense_10_biasG
Cassignvariableop_24_transformer_block_3_layer_normalization_6_gammaF
Bassignvariableop_25_transformer_block_3_layer_normalization_6_betaG
Cassignvariableop_26_transformer_block_3_layer_normalization_7_gammaF
Bassignvariableop_27_transformer_block_3_layer_normalization_7_beta
assignvariableop_28_total
assignvariableop_29_count4
0assignvariableop_30_sgd_dense_11_kernel_momentum2
.assignvariableop_31_sgd_dense_11_bias_momentum4
0assignvariableop_32_sgd_dense_12_kernel_momentum2
.assignvariableop_33_sgd_dense_12_bias_momentum4
0assignvariableop_34_sgd_dense_13_kernel_momentum2
.assignvariableop_35_sgd_dense_13_bias_momentumZ
Vassignvariableop_36_sgd_token_and_position_embedding_1_embedding_2_embeddings_momentumZ
Vassignvariableop_37_sgd_token_and_position_embedding_1_embedding_3_embeddings_momentum\
Xassignvariableop_38_sgd_transformer_block_3_multi_head_attention_3_query_kernel_momentumZ
Vassignvariableop_39_sgd_transformer_block_3_multi_head_attention_3_query_bias_momentumZ
Vassignvariableop_40_sgd_transformer_block_3_multi_head_attention_3_key_kernel_momentumX
Tassignvariableop_41_sgd_transformer_block_3_multi_head_attention_3_key_bias_momentum\
Xassignvariableop_42_sgd_transformer_block_3_multi_head_attention_3_value_kernel_momentumZ
Vassignvariableop_43_sgd_transformer_block_3_multi_head_attention_3_value_bias_momentumg
cassignvariableop_44_sgd_transformer_block_3_multi_head_attention_3_attention_output_kernel_momentume
aassignvariableop_45_sgd_transformer_block_3_multi_head_attention_3_attention_output_bias_momentum3
/assignvariableop_46_sgd_dense_9_kernel_momentum1
-assignvariableop_47_sgd_dense_9_bias_momentum4
0assignvariableop_48_sgd_dense_10_kernel_momentum2
.assignvariableop_49_sgd_dense_10_bias_momentumT
Passignvariableop_50_sgd_transformer_block_3_layer_normalization_6_gamma_momentumS
Oassignvariableop_51_sgd_transformer_block_3_layer_normalization_6_beta_momentumT
Passignvariableop_52_sgd_transformer_block_3_layer_normalization_7_gamma_momentumS
Oassignvariableop_53_sgd_transformer_block_3_layer_normalization_7_beta_momentum
identity_55??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
value?B?7B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/8/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/9/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_11_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_11_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_12_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_12_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_13_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_13_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_momentumIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_iterIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpIassignvariableop_10_token_and_position_embedding_1_embedding_2_embeddingsIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpIassignvariableop_11_token_and_position_embedding_1_embedding_3_embeddingsIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpKassignvariableop_12_transformer_block_3_multi_head_attention_3_query_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpIassignvariableop_13_transformer_block_3_multi_head_attention_3_query_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpIassignvariableop_14_transformer_block_3_multi_head_attention_3_key_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpGassignvariableop_15_transformer_block_3_multi_head_attention_3_key_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpKassignvariableop_16_transformer_block_3_multi_head_attention_3_value_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpIassignvariableop_17_transformer_block_3_multi_head_attention_3_value_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpVassignvariableop_18_transformer_block_3_multi_head_attention_3_attention_output_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpTassignvariableop_19_transformer_block_3_multi_head_attention_3_attention_output_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_9_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_9_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp#assignvariableop_22_dense_10_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp!assignvariableop_23_dense_10_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpCassignvariableop_24_transformer_block_3_layer_normalization_6_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpBassignvariableop_25_transformer_block_3_layer_normalization_6_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpCassignvariableop_26_transformer_block_3_layer_normalization_7_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpBassignvariableop_27_transformer_block_3_layer_normalization_7_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp0assignvariableop_30_sgd_dense_11_kernel_momentumIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp.assignvariableop_31_sgd_dense_11_bias_momentumIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp0assignvariableop_32_sgd_dense_12_kernel_momentumIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp.assignvariableop_33_sgd_dense_12_bias_momentumIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp0assignvariableop_34_sgd_dense_13_kernel_momentumIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp.assignvariableop_35_sgd_dense_13_bias_momentumIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpVassignvariableop_36_sgd_token_and_position_embedding_1_embedding_2_embeddings_momentumIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpVassignvariableop_37_sgd_token_and_position_embedding_1_embedding_3_embeddings_momentumIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpXassignvariableop_38_sgd_transformer_block_3_multi_head_attention_3_query_kernel_momentumIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpVassignvariableop_39_sgd_transformer_block_3_multi_head_attention_3_query_bias_momentumIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpVassignvariableop_40_sgd_transformer_block_3_multi_head_attention_3_key_kernel_momentumIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpTassignvariableop_41_sgd_transformer_block_3_multi_head_attention_3_key_bias_momentumIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpXassignvariableop_42_sgd_transformer_block_3_multi_head_attention_3_value_kernel_momentumIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpVassignvariableop_43_sgd_transformer_block_3_multi_head_attention_3_value_bias_momentumIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpcassignvariableop_44_sgd_transformer_block_3_multi_head_attention_3_attention_output_kernel_momentumIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpaassignvariableop_45_sgd_transformer_block_3_multi_head_attention_3_attention_output_bias_momentumIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp/assignvariableop_46_sgd_dense_9_kernel_momentumIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp-assignvariableop_47_sgd_dense_9_bias_momentumIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp0assignvariableop_48_sgd_dense_10_kernel_momentumIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp.assignvariableop_49_sgd_dense_10_bias_momentumIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpPassignvariableop_50_sgd_transformer_block_3_layer_normalization_6_gamma_momentumIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpOassignvariableop_51_sgd_transformer_block_3_layer_normalization_6_beta_momentumIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpPassignvariableop_52_sgd_transformer_block_3_layer_normalization_7_gamma_momentumIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpOassignvariableop_53_sgd_transformer_block_3_layer_normalization_7_beta_momentumIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_539
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_54?	
Identity_55IdentityIdentity_54:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_55"#
identity_55Identity_55:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
O
3__inference_average_pooling1d_1_layer_call_fn_12122

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_121162
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
}
(__inference_dense_12_layer_call_fn_14175

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_128002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_13662

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_129982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesv
t:??????????R::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????R
 
_user_specified_nameinputs
?1
?
B__inference_model_1_layer_call_and_return_conditional_losses_13112

inputs(
$token_and_position_embedding_1_13054(
$token_and_position_embedding_1_13056
transformer_block_3_13060
transformer_block_3_13062
transformer_block_3_13064
transformer_block_3_13066
transformer_block_3_13068
transformer_block_3_13070
transformer_block_3_13072
transformer_block_3_13074
transformer_block_3_13076
transformer_block_3_13078
transformer_block_3_13080
transformer_block_3_13082
transformer_block_3_13084
transformer_block_3_13086
transformer_block_3_13088
transformer_block_3_13090
dense_11_13094
dense_11_13096
dense_12_13100
dense_12_13102
dense_13_13106
dense_13_13108
identity?? dense_11/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?6token_and_position_embedding_1/StatefulPartitionedCall?+transformer_block_3/StatefulPartitionedCall?
6token_and_position_embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs$token_and_position_embedding_1_13054$token_and_position_embedding_1_13056*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *b
f]R[
Y__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_1231728
6token_and_position_embedding_1/StatefulPartitionedCall?
#average_pooling1d_1/PartitionedCallPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_121162%
#average_pooling1d_1/PartitionedCall?
+transformer_block_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_1/PartitionedCall:output:0transformer_block_3_13060transformer_block_3_13062transformer_block_3_13064transformer_block_3_13066transformer_block_3_13068transformer_block_3_13070transformer_block_3_13072transformer_block_3_13074transformer_block_3_13076transformer_block_3_13078transformer_block_3_13080transformer_block_3_13082transformer_block_3_13084transformer_block_3_13086transformer_block_3_13088transformer_block_3_13090*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_126092-
+transformer_block_3/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall4transformer_block_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_127242
flatten_1/PartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_11_13094dense_11_13096*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_127432"
 dense_11/StatefulPartitionedCall?
dropout_10/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_127762
dropout_10/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0dense_12_13100dense_12_13102*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_128002"
 dense_12/StatefulPartitionedCall?
dropout_11/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_128332
dropout_11/PartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_13_13106dense_13_13108*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_128562"
 dense_13/StatefulPartitionedCall?
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall7^token_and_position_embedding_1/StatefulPartitionedCall,^transformer_block_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesv
t:??????????R::::::::::::::::::::::::2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2p
6token_and_position_embedding_1/StatefulPartitionedCall6token_and_position_embedding_1/StatefulPartitionedCall2Z
+transformer_block_3/StatefulPartitionedCall+transformer_block_3/StatefulPartitionedCall:P L
(
_output_shapes
:??????????R
 
_user_specified_nameinputs
??
?
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_12609

inputsF
Bmulti_head_attention_3_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_3_query_add_readvariableop_resourceD
@multi_head_attention_3_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_3_key_add_readvariableop_resourceF
Bmulti_head_attention_3_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_3_value_add_readvariableop_resourceQ
Mmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_3_attention_output_add_readvariableop_resource?
;layer_normalization_6_batchnorm_mul_readvariableop_resource;
7layer_normalization_6_batchnorm_readvariableop_resource:
6sequential_3_dense_9_tensordot_readvariableop_resource8
4sequential_3_dense_9_biasadd_readvariableop_resource;
7sequential_3_dense_10_tensordot_readvariableop_resource9
5sequential_3_dense_10_biasadd_readvariableop_resource?
;layer_normalization_7_batchnorm_mul_readvariableop_resource;
7layer_normalization_7_batchnorm_readvariableop_resource
identity??.layer_normalization_6/batchnorm/ReadVariableOp?2layer_normalization_6/batchnorm/mul/ReadVariableOp?.layer_normalization_7/batchnorm/ReadVariableOp?2layer_normalization_7/batchnorm/mul/ReadVariableOp?:multi_head_attention_3/attention_output/add/ReadVariableOp?Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp?-multi_head_attention_3/key/add/ReadVariableOp?7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp?/multi_head_attention_3/query/add/ReadVariableOp?9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp?/multi_head_attention_3/value/add/ReadVariableOp?9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp?,sequential_3/dense_10/BiasAdd/ReadVariableOp?.sequential_3/dense_10/Tensordot/ReadVariableOp?+sequential_3/dense_9/BiasAdd/ReadVariableOp?-sequential_3/dense_9/Tensordot/ReadVariableOp?
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp?
*multi_head_attention_3/query/einsum/EinsumEinsuminputsAmulti_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2,
*multi_head_attention_3/query/einsum/Einsum?
/multi_head_attention_3/query/add/ReadVariableOpReadVariableOp8multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/query/add/ReadVariableOp?
 multi_head_attention_3/query/addAddV23multi_head_attention_3/query/einsum/Einsum:output:07multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2"
 multi_head_attention_3/query/add?
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp?
(multi_head_attention_3/key/einsum/EinsumEinsuminputs?multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2*
(multi_head_attention_3/key/einsum/Einsum?
-multi_head_attention_3/key/add/ReadVariableOpReadVariableOp6multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_3/key/add/ReadVariableOp?
multi_head_attention_3/key/addAddV21multi_head_attention_3/key/einsum/Einsum:output:05multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2 
multi_head_attention_3/key/add?
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp?
*multi_head_attention_3/value/einsum/EinsumEinsuminputsAmulti_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2,
*multi_head_attention_3/value/einsum/Einsum?
/multi_head_attention_3/value/add/ReadVariableOpReadVariableOp8multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/value/add/ReadVariableOp?
 multi_head_attention_3/value/addAddV23multi_head_attention_3/value/einsum/Einsum:output:07multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2"
 multi_head_attention_3/value/add?
multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>2
multi_head_attention_3/Mul/y?
multi_head_attention_3/MulMul$multi_head_attention_3/query/add:z:0%multi_head_attention_3/Mul/y:output:0*
T0*/
_output_shapes
:?????????# 2
multi_head_attention_3/Mul?
$multi_head_attention_3/einsum/EinsumEinsum"multi_head_attention_3/key/add:z:0multi_head_attention_3/Mul:z:0*
N*
T0*/
_output_shapes
:?????????##*
equationaecd,abcd->acbe2&
$multi_head_attention_3/einsum/Einsum?
&multi_head_attention_3/softmax/SoftmaxSoftmax-multi_head_attention_3/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????##2(
&multi_head_attention_3/softmax/Softmax?
'multi_head_attention_3/dropout/IdentityIdentity0multi_head_attention_3/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????##2)
'multi_head_attention_3/dropout/Identity?
&multi_head_attention_3/einsum_1/EinsumEinsum0multi_head_attention_3/dropout/Identity:output:0$multi_head_attention_3/value/add:z:0*
N*
T0*/
_output_shapes
:?????????# *
equationacbe,aecd->abcd2(
&multi_head_attention_3/einsum_1/Einsum?
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp?
5multi_head_attention_3/attention_output/einsum/EinsumEinsum/multi_head_attention_3/einsum_1/Einsum:output:0Lmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????# *
equationabcd,cde->abe27
5multi_head_attention_3/attention_output/einsum/Einsum?
:multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_3/attention_output/add/ReadVariableOp?
+multi_head_attention_3/attention_output/addAddV2>multi_head_attention_3/attention_output/einsum/Einsum:output:0Bmulti_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2-
+multi_head_attention_3/attention_output/add?
dropout_8/IdentityIdentity/multi_head_attention_3/attention_output/add:z:0*
T0*+
_output_shapes
:?????????# 2
dropout_8/Identityn
addAddV2inputsdropout_8/Identity:output:0*
T0*+
_output_shapes
:?????????# 2
add?
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_6/moments/mean/reduction_indices?
"layer_normalization_6/moments/meanMeanadd:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2$
"layer_normalization_6/moments/mean?
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2,
*layer_normalization_6/moments/StopGradient?
/layer_normalization_6/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 21
/layer_normalization_6/moments/SquaredDifference?
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_6/moments/variance/reduction_indices?
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2(
&layer_normalization_6/moments/variance?
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_6/batchnorm/add/y?
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2%
#layer_normalization_6/batchnorm/add?
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2'
%layer_normalization_6/batchnorm/Rsqrt?
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_6/batchnorm/mul/ReadVariableOp?
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2%
#layer_normalization_6/batchnorm/mul?
%layer_normalization_6/batchnorm/mul_1Muladd:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_6/batchnorm/mul_1?
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_6/batchnorm/mul_2?
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_6/batchnorm/ReadVariableOp?
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2%
#layer_normalization_6/batchnorm/sub?
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_6/batchnorm/add_1?
-sequential_3/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_3_dense_9_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02/
-sequential_3/dense_9/Tensordot/ReadVariableOp?
#sequential_3/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_3/dense_9/Tensordot/axes?
#sequential_3/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_3/dense_9/Tensordot/free?
$sequential_3/dense_9/Tensordot/ShapeShape)layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_3/dense_9/Tensordot/Shape?
,sequential_3/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_3/dense_9/Tensordot/GatherV2/axis?
'sequential_3/dense_9/Tensordot/GatherV2GatherV2-sequential_3/dense_9/Tensordot/Shape:output:0,sequential_3/dense_9/Tensordot/free:output:05sequential_3/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_3/dense_9/Tensordot/GatherV2?
.sequential_3/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_3/dense_9/Tensordot/GatherV2_1/axis?
)sequential_3/dense_9/Tensordot/GatherV2_1GatherV2-sequential_3/dense_9/Tensordot/Shape:output:0,sequential_3/dense_9/Tensordot/axes:output:07sequential_3/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_3/dense_9/Tensordot/GatherV2_1?
$sequential_3/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_3/dense_9/Tensordot/Const?
#sequential_3/dense_9/Tensordot/ProdProd0sequential_3/dense_9/Tensordot/GatherV2:output:0-sequential_3/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_3/dense_9/Tensordot/Prod?
&sequential_3/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_3/dense_9/Tensordot/Const_1?
%sequential_3/dense_9/Tensordot/Prod_1Prod2sequential_3/dense_9/Tensordot/GatherV2_1:output:0/sequential_3/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_3/dense_9/Tensordot/Prod_1?
*sequential_3/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_3/dense_9/Tensordot/concat/axis?
%sequential_3/dense_9/Tensordot/concatConcatV2,sequential_3/dense_9/Tensordot/free:output:0,sequential_3/dense_9/Tensordot/axes:output:03sequential_3/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_9/Tensordot/concat?
$sequential_3/dense_9/Tensordot/stackPack,sequential_3/dense_9/Tensordot/Prod:output:0.sequential_3/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_3/dense_9/Tensordot/stack?
(sequential_3/dense_9/Tensordot/transpose	Transpose)layer_normalization_6/batchnorm/add_1:z:0.sequential_3/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????# 2*
(sequential_3/dense_9/Tensordot/transpose?
&sequential_3/dense_9/Tensordot/ReshapeReshape,sequential_3/dense_9/Tensordot/transpose:y:0-sequential_3/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&sequential_3/dense_9/Tensordot/Reshape?
%sequential_3/dense_9/Tensordot/MatMulMatMul/sequential_3/dense_9/Tensordot/Reshape:output:05sequential_3/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2'
%sequential_3/dense_9/Tensordot/MatMul?
&sequential_3/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2(
&sequential_3/dense_9/Tensordot/Const_2?
,sequential_3/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_3/dense_9/Tensordot/concat_1/axis?
'sequential_3/dense_9/Tensordot/concat_1ConcatV20sequential_3/dense_9/Tensordot/GatherV2:output:0/sequential_3/dense_9/Tensordot/Const_2:output:05sequential_3/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_3/dense_9/Tensordot/concat_1?
sequential_3/dense_9/TensordotReshape/sequential_3/dense_9/Tensordot/MatMul:product:00sequential_3/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@2 
sequential_3/dense_9/Tensordot?
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp?
sequential_3/dense_9/BiasAddBiasAdd'sequential_3/dense_9/Tensordot:output:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@2
sequential_3/dense_9/BiasAdd?
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#@2
sequential_3/dense_9/Relu?
.sequential_3/dense_10/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_10_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_3/dense_10/Tensordot/ReadVariableOp?
$sequential_3/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_3/dense_10/Tensordot/axes?
$sequential_3/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_3/dense_10/Tensordot/free?
%sequential_3/dense_10/Tensordot/ShapeShape'sequential_3/dense_9/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/Shape?
-sequential_3/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/GatherV2/axis?
(sequential_3/dense_10/Tensordot/GatherV2GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/free:output:06sequential_3/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/GatherV2?
/sequential_3/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_3/dense_10/Tensordot/GatherV2_1/axis?
*sequential_3/dense_10/Tensordot/GatherV2_1GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/axes:output:08sequential_3/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_3/dense_10/Tensordot/GatherV2_1?
%sequential_3/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_3/dense_10/Tensordot/Const?
$sequential_3/dense_10/Tensordot/ProdProd1sequential_3/dense_10/Tensordot/GatherV2:output:0.sequential_3/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_3/dense_10/Tensordot/Prod?
'sequential_3/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_1?
&sequential_3/dense_10/Tensordot/Prod_1Prod3sequential_3/dense_10/Tensordot/GatherV2_1:output:00sequential_3/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_3/dense_10/Tensordot/Prod_1?
+sequential_3/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_3/dense_10/Tensordot/concat/axis?
&sequential_3/dense_10/Tensordot/concatConcatV2-sequential_3/dense_10/Tensordot/free:output:0-sequential_3/dense_10/Tensordot/axes:output:04sequential_3/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_3/dense_10/Tensordot/concat?
%sequential_3/dense_10/Tensordot/stackPack-sequential_3/dense_10/Tensordot/Prod:output:0/sequential_3/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/stack?
)sequential_3/dense_10/Tensordot/transpose	Transpose'sequential_3/dense_9/Relu:activations:0/sequential_3/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2+
)sequential_3/dense_10/Tensordot/transpose?
'sequential_3/dense_10/Tensordot/ReshapeReshape-sequential_3/dense_10/Tensordot/transpose:y:0.sequential_3/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_3/dense_10/Tensordot/Reshape?
&sequential_3/dense_10/Tensordot/MatMulMatMul0sequential_3/dense_10/Tensordot/Reshape:output:06sequential_3/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_3/dense_10/Tensordot/MatMul?
'sequential_3/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_2?
-sequential_3/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/concat_1/axis?
(sequential_3/dense_10/Tensordot/concat_1ConcatV21sequential_3/dense_10/Tensordot/GatherV2:output:00sequential_3/dense_10/Tensordot/Const_2:output:06sequential_3/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/concat_1?
sequential_3/dense_10/TensordotReshape0sequential_3/dense_10/Tensordot/MatMul:product:01sequential_3/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 2!
sequential_3/dense_10/Tensordot?
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp?
sequential_3/dense_10/BiasAddBiasAdd(sequential_3/dense_10/Tensordot:output:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2
sequential_3/dense_10/BiasAdd?
dropout_9/IdentityIdentity&sequential_3/dense_10/BiasAdd:output:0*
T0*+
_output_shapes
:?????????# 2
dropout_9/Identity?
add_1AddV2)layer_normalization_6/batchnorm/add_1:z:0dropout_9/Identity:output:0*
T0*+
_output_shapes
:?????????# 2
add_1?
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_7/moments/mean/reduction_indices?
"layer_normalization_7/moments/meanMean	add_1:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2$
"layer_normalization_7/moments/mean?
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2,
*layer_normalization_7/moments/StopGradient?
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 21
/layer_normalization_7/moments/SquaredDifference?
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_7/moments/variance/reduction_indices?
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2(
&layer_normalization_7/moments/variance?
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_7/batchnorm/add/y?
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2%
#layer_normalization_7/batchnorm/add?
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2'
%layer_normalization_7/batchnorm/Rsqrt?
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_7/batchnorm/mul/ReadVariableOp?
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2%
#layer_normalization_7/batchnorm/mul?
%layer_normalization_7/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_7/batchnorm/mul_1?
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_7/batchnorm/mul_2?
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_7/batchnorm/ReadVariableOp?
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2%
#layer_normalization_7/batchnorm/sub?
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_7/batchnorm/add_1?
IdentityIdentity)layer_normalization_7/batchnorm/add_1:z:0/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp;^multi_head_attention_3/attention_output/add/ReadVariableOpE^multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_3/key/add/ReadVariableOp8^multi_head_attention_3/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/query/add/ReadVariableOp:^multi_head_attention_3/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/value/add/ReadVariableOp:^multi_head_attention_3/value/einsum/Einsum/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp/^sequential_3/dense_10/Tensordot/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp.^sequential_3/dense_9/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:?????????# ::::::::::::::::2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_3/attention_output/add/ReadVariableOp:multi_head_attention_3/attention_output/add/ReadVariableOp2?
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_3/key/add/ReadVariableOp-multi_head_attention_3/key/add/ReadVariableOp2r
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/query/add/ReadVariableOp/multi_head_attention_3/query/add/ReadVariableOp2v
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/value/add/ReadVariableOp/multi_head_attention_3/value/add/ReadVariableOp2v
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2`
.sequential_3/dense_10/Tensordot/ReadVariableOp.sequential_3/dense_10/Tensordot/ReadVariableOp2Z
+sequential_3/dense_9/BiasAdd/ReadVariableOp+sequential_3/dense_9/BiasAdd/ReadVariableOp2^
-sequential_3/dense_9/Tensordot/ReadVariableOp-sequential_3/dense_9/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
?
?
,__inference_sequential_3_layer_call_fn_12262
dense_9_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_9_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_122512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????# 
'
_user_specified_namedense_9_input
?	
?
C__inference_dense_11_layer_call_and_return_conditional_losses_12743

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
}
(__inference_dense_11_layer_call_fn_14128

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_127432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_12278

inputs
dense_9_12267
dense_9_12269
dense_10_12272
dense_10_12274
identity?? dense_10/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCallinputsdense_9_12267dense_9_12269*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_121572!
dense_9/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_12272dense_10_12274*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_122032"
 dense_10/StatefulPartitionedCall?
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
? 
?
B__inference_dense_9_layer_call_and_return_conditional_losses_12157

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????# 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????#@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????#@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????# ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
?	
?
C__inference_dense_11_layer_call_and_return_conditional_losses_14119

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_10_layer_call_fn_14150

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_127712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
}
(__inference_dense_13_layer_call_fn_14221

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_128562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
,__inference_sequential_3_layer_call_fn_12289
dense_9_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_9_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_122782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????# 
'
_user_specified_namedense_9_input
?
?
,__inference_sequential_3_layer_call_fn_14348

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_122512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_12107
input_2M
Imodel_1_token_and_position_embedding_1_embedding_3_embedding_lookup_11943M
Imodel_1_token_and_position_embedding_1_embedding_2_embedding_lookup_11949b
^model_1_transformer_block_3_multi_head_attention_3_query_einsum_einsum_readvariableop_resourceX
Tmodel_1_transformer_block_3_multi_head_attention_3_query_add_readvariableop_resource`
\model_1_transformer_block_3_multi_head_attention_3_key_einsum_einsum_readvariableop_resourceV
Rmodel_1_transformer_block_3_multi_head_attention_3_key_add_readvariableop_resourceb
^model_1_transformer_block_3_multi_head_attention_3_value_einsum_einsum_readvariableop_resourceX
Tmodel_1_transformer_block_3_multi_head_attention_3_value_add_readvariableop_resourcem
imodel_1_transformer_block_3_multi_head_attention_3_attention_output_einsum_einsum_readvariableop_resourcec
_model_1_transformer_block_3_multi_head_attention_3_attention_output_add_readvariableop_resource[
Wmodel_1_transformer_block_3_layer_normalization_6_batchnorm_mul_readvariableop_resourceW
Smodel_1_transformer_block_3_layer_normalization_6_batchnorm_readvariableop_resourceV
Rmodel_1_transformer_block_3_sequential_3_dense_9_tensordot_readvariableop_resourceT
Pmodel_1_transformer_block_3_sequential_3_dense_9_biasadd_readvariableop_resourceW
Smodel_1_transformer_block_3_sequential_3_dense_10_tensordot_readvariableop_resourceU
Qmodel_1_transformer_block_3_sequential_3_dense_10_biasadd_readvariableop_resource[
Wmodel_1_transformer_block_3_layer_normalization_7_batchnorm_mul_readvariableop_resourceW
Smodel_1_transformer_block_3_layer_normalization_7_batchnorm_readvariableop_resource3
/model_1_dense_11_matmul_readvariableop_resource4
0model_1_dense_11_biasadd_readvariableop_resource3
/model_1_dense_12_matmul_readvariableop_resource4
0model_1_dense_12_biasadd_readvariableop_resource3
/model_1_dense_13_matmul_readvariableop_resource4
0model_1_dense_13_biasadd_readvariableop_resource
identity??'model_1/dense_11/BiasAdd/ReadVariableOp?&model_1/dense_11/MatMul/ReadVariableOp?'model_1/dense_12/BiasAdd/ReadVariableOp?&model_1/dense_12/MatMul/ReadVariableOp?'model_1/dense_13/BiasAdd/ReadVariableOp?&model_1/dense_13/MatMul/ReadVariableOp?Cmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup?Cmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup?Jmodel_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp?Nmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp?Jmodel_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp?Nmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp?Vmodel_1/transformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp?`model_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp?Imodel_1/transformer_block_3/multi_head_attention_3/key/add/ReadVariableOp?Smodel_1/transformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp?Kmodel_1/transformer_block_3/multi_head_attention_3/query/add/ReadVariableOp?Umodel_1/transformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp?Kmodel_1/transformer_block_3/multi_head_attention_3/value/add/ReadVariableOp?Umodel_1/transformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp?Hmodel_1/transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp?Jmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp?Gmodel_1/transformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOp?Imodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOp?
,model_1/token_and_position_embedding_1/ShapeShapeinput_2*
T0*
_output_shapes
:2.
,model_1/token_and_position_embedding_1/Shape?
:model_1/token_and_position_embedding_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2<
:model_1/token_and_position_embedding_1/strided_slice/stack?
<model_1/token_and_position_embedding_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<model_1/token_and_position_embedding_1/strided_slice/stack_1?
<model_1/token_and_position_embedding_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model_1/token_and_position_embedding_1/strided_slice/stack_2?
4model_1/token_and_position_embedding_1/strided_sliceStridedSlice5model_1/token_and_position_embedding_1/Shape:output:0Cmodel_1/token_and_position_embedding_1/strided_slice/stack:output:0Emodel_1/token_and_position_embedding_1/strided_slice/stack_1:output:0Emodel_1/token_and_position_embedding_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4model_1/token_and_position_embedding_1/strided_slice?
2model_1/token_and_position_embedding_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 24
2model_1/token_and_position_embedding_1/range/start?
2model_1/token_and_position_embedding_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :24
2model_1/token_and_position_embedding_1/range/delta?
,model_1/token_and_position_embedding_1/rangeRange;model_1/token_and_position_embedding_1/range/start:output:0=model_1/token_and_position_embedding_1/strided_slice:output:0;model_1/token_and_position_embedding_1/range/delta:output:0*#
_output_shapes
:?????????2.
,model_1/token_and_position_embedding_1/range?
Cmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookupResourceGatherImodel_1_token_and_position_embedding_1_embedding_3_embedding_lookup_119435model_1/token_and_position_embedding_1/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*\
_classR
PNloc:@model_1/token_and_position_embedding_1/embedding_3/embedding_lookup/11943*'
_output_shapes
:????????? *
dtype02E
Cmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup?
Lmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup/IdentityIdentityLmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*\
_classR
PNloc:@model_1/token_and_position_embedding_1/embedding_3/embedding_lookup/11943*'
_output_shapes
:????????? 2N
Lmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup/Identity?
Nmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1IdentityUmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2P
Nmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1?
7model_1/token_and_position_embedding_1/embedding_2/CastCastinput_2*

DstT0*

SrcT0*(
_output_shapes
:??????????R29
7model_1/token_and_position_embedding_1/embedding_2/Cast?
Cmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookupResourceGatherImodel_1_token_and_position_embedding_1_embedding_2_embedding_lookup_11949;model_1/token_and_position_embedding_1/embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*\
_classR
PNloc:@model_1/token_and_position_embedding_1/embedding_2/embedding_lookup/11949*,
_output_shapes
:??????????R *
dtype02E
Cmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup?
Lmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup/IdentityIdentityLmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*\
_classR
PNloc:@model_1/token_and_position_embedding_1/embedding_2/embedding_lookup/11949*,
_output_shapes
:??????????R 2N
Lmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup/Identity?
Nmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1IdentityUmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????R 2P
Nmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1?
*model_1/token_and_position_embedding_1/addAddV2Wmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1:output:0Wmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????R 2,
*model_1/token_and_position_embedding_1/add?
*model_1/average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_1/average_pooling1d_1/ExpandDims/dim?
&model_1/average_pooling1d_1/ExpandDims
ExpandDims.model_1/token_and_position_embedding_1/add:z:03model_1/average_pooling1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????R 2(
&model_1/average_pooling1d_1/ExpandDims?
#model_1/average_pooling1d_1/AvgPoolAvgPool/model_1/average_pooling1d_1/ExpandDims:output:0*
T0*/
_output_shapes
:?????????# *
ksize	
?*
paddingVALID*
strides	
?2%
#model_1/average_pooling1d_1/AvgPool?
#model_1/average_pooling1d_1/SqueezeSqueeze,model_1/average_pooling1d_1/AvgPool:output:0*
T0*+
_output_shapes
:?????????# *
squeeze_dims
2%
#model_1/average_pooling1d_1/Squeeze?
Umodel_1/transformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOp^model_1_transformer_block_3_multi_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_1/transformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp?
Fmodel_1/transformer_block_3/multi_head_attention_3/query/einsum/EinsumEinsum,model_1/average_pooling1d_1/Squeeze:output:0]model_1/transformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2H
Fmodel_1/transformer_block_3/multi_head_attention_3/query/einsum/Einsum?
Kmodel_1/transformer_block_3/multi_head_attention_3/query/add/ReadVariableOpReadVariableOpTmodel_1_transformer_block_3_multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_1/transformer_block_3/multi_head_attention_3/query/add/ReadVariableOp?
<model_1/transformer_block_3/multi_head_attention_3/query/addAddV2Omodel_1/transformer_block_3/multi_head_attention_3/query/einsum/Einsum:output:0Smodel_1/transformer_block_3/multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2>
<model_1/transformer_block_3/multi_head_attention_3/query/add?
Smodel_1/transformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOp\model_1_transformer_block_3_multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02U
Smodel_1/transformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp?
Dmodel_1/transformer_block_3/multi_head_attention_3/key/einsum/EinsumEinsum,model_1/average_pooling1d_1/Squeeze:output:0[model_1/transformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2F
Dmodel_1/transformer_block_3/multi_head_attention_3/key/einsum/Einsum?
Imodel_1/transformer_block_3/multi_head_attention_3/key/add/ReadVariableOpReadVariableOpRmodel_1_transformer_block_3_multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

: *
dtype02K
Imodel_1/transformer_block_3/multi_head_attention_3/key/add/ReadVariableOp?
:model_1/transformer_block_3/multi_head_attention_3/key/addAddV2Mmodel_1/transformer_block_3/multi_head_attention_3/key/einsum/Einsum:output:0Qmodel_1/transformer_block_3/multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2<
:model_1/transformer_block_3/multi_head_attention_3/key/add?
Umodel_1/transformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOp^model_1_transformer_block_3_multi_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_1/transformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp?
Fmodel_1/transformer_block_3/multi_head_attention_3/value/einsum/EinsumEinsum,model_1/average_pooling1d_1/Squeeze:output:0]model_1/transformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2H
Fmodel_1/transformer_block_3/multi_head_attention_3/value/einsum/Einsum?
Kmodel_1/transformer_block_3/multi_head_attention_3/value/add/ReadVariableOpReadVariableOpTmodel_1_transformer_block_3_multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_1/transformer_block_3/multi_head_attention_3/value/add/ReadVariableOp?
<model_1/transformer_block_3/multi_head_attention_3/value/addAddV2Omodel_1/transformer_block_3/multi_head_attention_3/value/einsum/Einsum:output:0Smodel_1/transformer_block_3/multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2>
<model_1/transformer_block_3/multi_head_attention_3/value/add?
8model_1/transformer_block_3/multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>2:
8model_1/transformer_block_3/multi_head_attention_3/Mul/y?
6model_1/transformer_block_3/multi_head_attention_3/MulMul@model_1/transformer_block_3/multi_head_attention_3/query/add:z:0Amodel_1/transformer_block_3/multi_head_attention_3/Mul/y:output:0*
T0*/
_output_shapes
:?????????# 28
6model_1/transformer_block_3/multi_head_attention_3/Mul?
@model_1/transformer_block_3/multi_head_attention_3/einsum/EinsumEinsum>model_1/transformer_block_3/multi_head_attention_3/key/add:z:0:model_1/transformer_block_3/multi_head_attention_3/Mul:z:0*
N*
T0*/
_output_shapes
:?????????##*
equationaecd,abcd->acbe2B
@model_1/transformer_block_3/multi_head_attention_3/einsum/Einsum?
Bmodel_1/transformer_block_3/multi_head_attention_3/softmax/SoftmaxSoftmaxImodel_1/transformer_block_3/multi_head_attention_3/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????##2D
Bmodel_1/transformer_block_3/multi_head_attention_3/softmax/Softmax?
Cmodel_1/transformer_block_3/multi_head_attention_3/dropout/IdentityIdentityLmodel_1/transformer_block_3/multi_head_attention_3/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????##2E
Cmodel_1/transformer_block_3/multi_head_attention_3/dropout/Identity?
Bmodel_1/transformer_block_3/multi_head_attention_3/einsum_1/EinsumEinsumLmodel_1/transformer_block_3/multi_head_attention_3/dropout/Identity:output:0@model_1/transformer_block_3/multi_head_attention_3/value/add:z:0*
N*
T0*/
_output_shapes
:?????????# *
equationacbe,aecd->abcd2D
Bmodel_1/transformer_block_3/multi_head_attention_3/einsum_1/Einsum?
`model_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpimodel_1_transformer_block_3_multi_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02b
`model_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp?
Qmodel_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/EinsumEinsumKmodel_1/transformer_block_3/multi_head_attention_3/einsum_1/Einsum:output:0hmodel_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????# *
equationabcd,cde->abe2S
Qmodel_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum?
Vmodel_1/transformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOp_model_1_transformer_block_3_multi_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02X
Vmodel_1/transformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp?
Gmodel_1/transformer_block_3/multi_head_attention_3/attention_output/addAddV2Zmodel_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum:output:0^model_1/transformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2I
Gmodel_1/transformer_block_3/multi_head_attention_3/attention_output/add?
.model_1/transformer_block_3/dropout_8/IdentityIdentityKmodel_1/transformer_block_3/multi_head_attention_3/attention_output/add:z:0*
T0*+
_output_shapes
:?????????# 20
.model_1/transformer_block_3/dropout_8/Identity?
model_1/transformer_block_3/addAddV2,model_1/average_pooling1d_1/Squeeze:output:07model_1/transformer_block_3/dropout_8/Identity:output:0*
T0*+
_output_shapes
:?????????# 2!
model_1/transformer_block_3/add?
Pmodel_1/transformer_block_3/layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2R
Pmodel_1/transformer_block_3/layer_normalization_6/moments/mean/reduction_indices?
>model_1/transformer_block_3/layer_normalization_6/moments/meanMean#model_1/transformer_block_3/add:z:0Ymodel_1/transformer_block_3/layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2@
>model_1/transformer_block_3/layer_normalization_6/moments/mean?
Fmodel_1/transformer_block_3/layer_normalization_6/moments/StopGradientStopGradientGmodel_1/transformer_block_3/layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2H
Fmodel_1/transformer_block_3/layer_normalization_6/moments/StopGradient?
Kmodel_1/transformer_block_3/layer_normalization_6/moments/SquaredDifferenceSquaredDifference#model_1/transformer_block_3/add:z:0Omodel_1/transformer_block_3/layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 2M
Kmodel_1/transformer_block_3/layer_normalization_6/moments/SquaredDifference?
Tmodel_1/transformer_block_3/layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2V
Tmodel_1/transformer_block_3/layer_normalization_6/moments/variance/reduction_indices?
Bmodel_1/transformer_block_3/layer_normalization_6/moments/varianceMeanOmodel_1/transformer_block_3/layer_normalization_6/moments/SquaredDifference:z:0]model_1/transformer_block_3/layer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2D
Bmodel_1/transformer_block_3/layer_normalization_6/moments/variance?
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52C
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/add/y?
?model_1/transformer_block_3/layer_normalization_6/batchnorm/addAddV2Kmodel_1/transformer_block_3/layer_normalization_6/moments/variance:output:0Jmodel_1/transformer_block_3/layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2A
?model_1/transformer_block_3/layer_normalization_6/batchnorm/add?
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/RsqrtRsqrtCmodel_1/transformer_block_3/layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2C
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/Rsqrt?
Nmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpWmodel_1_transformer_block_3_layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02P
Nmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp?
?model_1/transformer_block_3/layer_normalization_6/batchnorm/mulMulEmodel_1/transformer_block_3/layer_normalization_6/batchnorm/Rsqrt:y:0Vmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2A
?model_1/transformer_block_3/layer_normalization_6/batchnorm/mul?
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul_1Mul#model_1/transformer_block_3/add:z:0Cmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2C
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul_1?
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul_2MulGmodel_1/transformer_block_3/layer_normalization_6/moments/mean:output:0Cmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2C
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul_2?
Jmodel_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpReadVariableOpSmodel_1_transformer_block_3_layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02L
Jmodel_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp?
?model_1/transformer_block_3/layer_normalization_6/batchnorm/subSubRmodel_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp:value:0Emodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2A
?model_1/transformer_block_3/layer_normalization_6/batchnorm/sub?
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/add_1AddV2Emodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul_1:z:0Cmodel_1/transformer_block_3/layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2C
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/add_1?
Imodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOpReadVariableOpRmodel_1_transformer_block_3_sequential_3_dense_9_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02K
Imodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOp?
?model_1/transformer_block_3/sequential_3/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2A
?model_1/transformer_block_3/sequential_3/dense_9/Tensordot/axes?
?model_1/transformer_block_3/sequential_3/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2A
?model_1/transformer_block_3/sequential_3/dense_9/Tensordot/free?
@model_1/transformer_block_3/sequential_3/dense_9/Tensordot/ShapeShapeEmodel_1/transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:2B
@model_1/transformer_block_3/sequential_3/dense_9/Tensordot/Shape?
Hmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2/axis?
Cmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2GatherV2Imodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/Shape:output:0Hmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/free:output:0Qmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2?
Jmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2_1/axis?
Emodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2_1GatherV2Imodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/Shape:output:0Hmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/axes:output:0Smodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2G
Emodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2_1?
@model_1/transformer_block_3/sequential_3/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2B
@model_1/transformer_block_3/sequential_3/dense_9/Tensordot/Const?
?model_1/transformer_block_3/sequential_3/dense_9/Tensordot/ProdProdLmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2:output:0Imodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2A
?model_1/transformer_block_3/sequential_3/dense_9/Tensordot/Prod?
Bmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2D
Bmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/Const_1?
Amodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/Prod_1ProdNmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2_1:output:0Kmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2C
Amodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/Prod_1?
Fmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/concat/axis?
Amodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/concatConcatV2Hmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/free:output:0Hmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/axes:output:0Omodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/concat?
@model_1/transformer_block_3/sequential_3/dense_9/Tensordot/stackPackHmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/Prod:output:0Jmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2B
@model_1/transformer_block_3/sequential_3/dense_9/Tensordot/stack?
Dmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/transpose	TransposeEmodel_1/transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0Jmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????# 2F
Dmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/transpose?
Bmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/ReshapeReshapeHmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/transpose:y:0Imodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2D
Bmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/Reshape?
Amodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/MatMulMatMulKmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/Reshape:output:0Qmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2C
Amodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/MatMul?
Bmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2D
Bmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/Const_2?
Hmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/concat_1/axis?
Cmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/concat_1ConcatV2Lmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2:output:0Kmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/Const_2:output:0Qmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2E
Cmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/concat_1?
:model_1/transformer_block_3/sequential_3/dense_9/TensordotReshapeKmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/MatMul:product:0Lmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@2<
:model_1/transformer_block_3/sequential_3/dense_9/Tensordot?
Gmodel_1/transformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOpPmodel_1_transformer_block_3_sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gmodel_1/transformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOp?
8model_1/transformer_block_3/sequential_3/dense_9/BiasAddBiasAddCmodel_1/transformer_block_3/sequential_3/dense_9/Tensordot:output:0Omodel_1/transformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@2:
8model_1/transformer_block_3/sequential_3/dense_9/BiasAdd?
5model_1/transformer_block_3/sequential_3/dense_9/ReluReluAmodel_1/transformer_block_3/sequential_3/dense_9/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#@27
5model_1/transformer_block_3/sequential_3/dense_9/Relu?
Jmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOpReadVariableOpSmodel_1_transformer_block_3_sequential_3_dense_10_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02L
Jmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp?
@model_1/transformer_block_3/sequential_3/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_1/transformer_block_3/sequential_3/dense_10/Tensordot/axes?
@model_1/transformer_block_3/sequential_3/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_1/transformer_block_3/sequential_3/dense_10/Tensordot/free?
Amodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ShapeShapeCmodel_1/transformer_block_3/sequential_3/dense_9/Relu:activations:0*
T0*
_output_shapes
:2C
Amodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Shape?
Imodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2/axis?
Dmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2GatherV2Jmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Shape:output:0Imodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/free:output:0Rmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2?
Kmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1/axis?
Fmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1GatherV2Jmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Shape:output:0Imodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/axes:output:0Tmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1?
Amodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Const?
@model_1/transformer_block_3/sequential_3/dense_10/Tensordot/ProdProdMmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2:output:0Jmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_1/transformer_block_3/sequential_3/dense_10/Tensordot/Prod?
Cmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Const_1?
Bmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Prod_1ProdOmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1:output:0Lmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Prod_1?
Gmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat/axis?
Bmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concatConcatV2Imodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/free:output:0Imodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/axes:output:0Pmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat?
Amodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/stackPackImodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Prod:output:0Kmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/stack?
Emodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/transpose	TransposeCmodel_1/transformer_block_3/sequential_3/dense_9/Relu:activations:0Kmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2G
Emodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/transpose?
Cmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ReshapeReshapeImodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/transpose:y:0Jmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2E
Cmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Reshape?
Bmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/MatMulMatMulLmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Reshape:output:0Rmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2D
Bmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/MatMul?
Cmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Const_2?
Imodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat_1/axis?
Dmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat_1ConcatV2Mmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2:output:0Lmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Const_2:output:0Rmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat_1?
;model_1/transformer_block_3/sequential_3/dense_10/TensordotReshapeLmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/MatMul:product:0Mmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 2=
;model_1/transformer_block_3/sequential_3/dense_10/Tensordot?
Hmodel_1/transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOpQmodel_1_transformer_block_3_sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel_1/transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp?
9model_1/transformer_block_3/sequential_3/dense_10/BiasAddBiasAddDmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot:output:0Pmodel_1/transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2;
9model_1/transformer_block_3/sequential_3/dense_10/BiasAdd?
.model_1/transformer_block_3/dropout_9/IdentityIdentityBmodel_1/transformer_block_3/sequential_3/dense_10/BiasAdd:output:0*
T0*+
_output_shapes
:?????????# 20
.model_1/transformer_block_3/dropout_9/Identity?
!model_1/transformer_block_3/add_1AddV2Emodel_1/transformer_block_3/layer_normalization_6/batchnorm/add_1:z:07model_1/transformer_block_3/dropout_9/Identity:output:0*
T0*+
_output_shapes
:?????????# 2#
!model_1/transformer_block_3/add_1?
Pmodel_1/transformer_block_3/layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2R
Pmodel_1/transformer_block_3/layer_normalization_7/moments/mean/reduction_indices?
>model_1/transformer_block_3/layer_normalization_7/moments/meanMean%model_1/transformer_block_3/add_1:z:0Ymodel_1/transformer_block_3/layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2@
>model_1/transformer_block_3/layer_normalization_7/moments/mean?
Fmodel_1/transformer_block_3/layer_normalization_7/moments/StopGradientStopGradientGmodel_1/transformer_block_3/layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2H
Fmodel_1/transformer_block_3/layer_normalization_7/moments/StopGradient?
Kmodel_1/transformer_block_3/layer_normalization_7/moments/SquaredDifferenceSquaredDifference%model_1/transformer_block_3/add_1:z:0Omodel_1/transformer_block_3/layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 2M
Kmodel_1/transformer_block_3/layer_normalization_7/moments/SquaredDifference?
Tmodel_1/transformer_block_3/layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2V
Tmodel_1/transformer_block_3/layer_normalization_7/moments/variance/reduction_indices?
Bmodel_1/transformer_block_3/layer_normalization_7/moments/varianceMeanOmodel_1/transformer_block_3/layer_normalization_7/moments/SquaredDifference:z:0]model_1/transformer_block_3/layer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2D
Bmodel_1/transformer_block_3/layer_normalization_7/moments/variance?
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52C
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/add/y?
?model_1/transformer_block_3/layer_normalization_7/batchnorm/addAddV2Kmodel_1/transformer_block_3/layer_normalization_7/moments/variance:output:0Jmodel_1/transformer_block_3/layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2A
?model_1/transformer_block_3/layer_normalization_7/batchnorm/add?
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/RsqrtRsqrtCmodel_1/transformer_block_3/layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2C
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/Rsqrt?
Nmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpWmodel_1_transformer_block_3_layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02P
Nmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp?
?model_1/transformer_block_3/layer_normalization_7/batchnorm/mulMulEmodel_1/transformer_block_3/layer_normalization_7/batchnorm/Rsqrt:y:0Vmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2A
?model_1/transformer_block_3/layer_normalization_7/batchnorm/mul?
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul_1Mul%model_1/transformer_block_3/add_1:z:0Cmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2C
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul_1?
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul_2MulGmodel_1/transformer_block_3/layer_normalization_7/moments/mean:output:0Cmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2C
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul_2?
Jmodel_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpReadVariableOpSmodel_1_transformer_block_3_layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02L
Jmodel_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp?
?model_1/transformer_block_3/layer_normalization_7/batchnorm/subSubRmodel_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp:value:0Emodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2A
?model_1/transformer_block_3/layer_normalization_7/batchnorm/sub?
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/add_1AddV2Emodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul_1:z:0Cmodel_1/transformer_block_3/layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2C
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/add_1?
model_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
model_1/flatten_1/Const?
model_1/flatten_1/ReshapeReshapeEmodel_1/transformer_block_3/layer_normalization_7/batchnorm/add_1:z:0 model_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
model_1/flatten_1/Reshape?
&model_1/dense_11/MatMul/ReadVariableOpReadVariableOp/model_1_dense_11_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02(
&model_1/dense_11/MatMul/ReadVariableOp?
model_1/dense_11/MatMulMatMul"model_1/flatten_1/Reshape:output:0.model_1/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_1/dense_11/MatMul?
'model_1/dense_11/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/dense_11/BiasAdd/ReadVariableOp?
model_1/dense_11/BiasAddBiasAdd!model_1/dense_11/MatMul:product:0/model_1/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_1/dense_11/BiasAdd?
model_1/dense_11/ReluRelu!model_1/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_1/dense_11/Relu?
model_1/dropout_10/IdentityIdentity#model_1/dense_11/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
model_1/dropout_10/Identity?
&model_1/dense_12/MatMul/ReadVariableOpReadVariableOp/model_1_dense_12_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02(
&model_1/dense_12/MatMul/ReadVariableOp?
model_1/dense_12/MatMulMatMul$model_1/dropout_10/Identity:output:0.model_1/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_1/dense_12/MatMul?
'model_1/dense_12/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/dense_12/BiasAdd/ReadVariableOp?
model_1/dense_12/BiasAddBiasAdd!model_1/dense_12/MatMul:product:0/model_1/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_1/dense_12/BiasAdd?
model_1/dense_12/ReluRelu!model_1/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_1/dense_12/Relu?
model_1/dropout_11/IdentityIdentity#model_1/dense_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
model_1/dropout_11/Identity?
&model_1/dense_13/MatMul/ReadVariableOpReadVariableOp/model_1_dense_13_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&model_1/dense_13/MatMul/ReadVariableOp?
model_1/dense_13/MatMulMatMul$model_1/dropout_11/Identity:output:0.model_1/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_13/MatMul?
'model_1/dense_13/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/dense_13/BiasAdd/ReadVariableOp?
model_1/dense_13/BiasAddBiasAdd!model_1/dense_13/MatMul:product:0/model_1/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_13/BiasAdd?
IdentityIdentity!model_1/dense_13/BiasAdd:output:0(^model_1/dense_11/BiasAdd/ReadVariableOp'^model_1/dense_11/MatMul/ReadVariableOp(^model_1/dense_12/BiasAdd/ReadVariableOp'^model_1/dense_12/MatMul/ReadVariableOp(^model_1/dense_13/BiasAdd/ReadVariableOp'^model_1/dense_13/MatMul/ReadVariableOpD^model_1/token_and_position_embedding_1/embedding_2/embedding_lookupD^model_1/token_and_position_embedding_1/embedding_3/embedding_lookupK^model_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpO^model_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpK^model_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpO^model_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpW^model_1/transformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpa^model_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpJ^model_1/transformer_block_3/multi_head_attention_3/key/add/ReadVariableOpT^model_1/transformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpL^model_1/transformer_block_3/multi_head_attention_3/query/add/ReadVariableOpV^model_1/transformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpL^model_1/transformer_block_3/multi_head_attention_3/value/add/ReadVariableOpV^model_1/transformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpI^model_1/transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOpK^model_1/transformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOpH^model_1/transformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOpJ^model_1/transformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesv
t:??????????R::::::::::::::::::::::::2R
'model_1/dense_11/BiasAdd/ReadVariableOp'model_1/dense_11/BiasAdd/ReadVariableOp2P
&model_1/dense_11/MatMul/ReadVariableOp&model_1/dense_11/MatMul/ReadVariableOp2R
'model_1/dense_12/BiasAdd/ReadVariableOp'model_1/dense_12/BiasAdd/ReadVariableOp2P
&model_1/dense_12/MatMul/ReadVariableOp&model_1/dense_12/MatMul/ReadVariableOp2R
'model_1/dense_13/BiasAdd/ReadVariableOp'model_1/dense_13/BiasAdd/ReadVariableOp2P
&model_1/dense_13/MatMul/ReadVariableOp&model_1/dense_13/MatMul/ReadVariableOp2?
Cmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookupCmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup2?
Cmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookupCmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup2?
Jmodel_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpJmodel_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp2?
Nmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpNmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp2?
Jmodel_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpJmodel_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp2?
Nmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpNmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp2?
Vmodel_1/transformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpVmodel_1/transformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp2?
`model_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp`model_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp2?
Imodel_1/transformer_block_3/multi_head_attention_3/key/add/ReadVariableOpImodel_1/transformer_block_3/multi_head_attention_3/key/add/ReadVariableOp2?
Smodel_1/transformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpSmodel_1/transformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp2?
Kmodel_1/transformer_block_3/multi_head_attention_3/query/add/ReadVariableOpKmodel_1/transformer_block_3/multi_head_attention_3/query/add/ReadVariableOp2?
Umodel_1/transformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpUmodel_1/transformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp2?
Kmodel_1/transformer_block_3/multi_head_attention_3/value/add/ReadVariableOpKmodel_1/transformer_block_3/multi_head_attention_3/value/add/ReadVariableOp2?
Umodel_1/transformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpUmodel_1/transformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp2?
Hmodel_1/transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOpHmodel_1/transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp2?
Jmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOpJmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp2?
Gmodel_1/transformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOpGmodel_1/transformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOp2?
Imodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOpImodel_1/transformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOp:Q M
(
_output_shapes
:??????????R
!
_user_specified_name	input_2
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_13609

inputsE
Atoken_and_position_embedding_1_embedding_3_embedding_lookup_13445E
Atoken_and_position_embedding_1_embedding_2_embedding_lookup_13451Z
Vtransformer_block_3_multi_head_attention_3_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_3_multi_head_attention_3_query_add_readvariableop_resourceX
Ttransformer_block_3_multi_head_attention_3_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_3_multi_head_attention_3_key_add_readvariableop_resourceZ
Vtransformer_block_3_multi_head_attention_3_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_3_multi_head_attention_3_value_add_readvariableop_resourcee
atransformer_block_3_multi_head_attention_3_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_3_multi_head_attention_3_attention_output_add_readvariableop_resourceS
Otransformer_block_3_layer_normalization_6_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_3_layer_normalization_6_batchnorm_readvariableop_resourceN
Jtransformer_block_3_sequential_3_dense_9_tensordot_readvariableop_resourceL
Htransformer_block_3_sequential_3_dense_9_biasadd_readvariableop_resourceO
Ktransformer_block_3_sequential_3_dense_10_tensordot_readvariableop_resourceM
Itransformer_block_3_sequential_3_dense_10_biasadd_readvariableop_resourceS
Otransformer_block_3_layer_normalization_7_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_3_layer_normalization_7_batchnorm_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource
identity??dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?;token_and_position_embedding_1/embedding_2/embedding_lookup?;token_and_position_embedding_1/embedding_3/embedding_lookup?Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp?Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp?Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp?Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp?Ntransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp?Xtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp?Atransformer_block_3/multi_head_attention_3/key/add/ReadVariableOp?Ktransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp?Ctransformer_block_3/multi_head_attention_3/query/add/ReadVariableOp?Mtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp?Ctransformer_block_3/multi_head_attention_3/value/add/ReadVariableOp?Mtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp?@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp?Btransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp??transformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOp?Atransformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOp?
$token_and_position_embedding_1/ShapeShapeinputs*
T0*
_output_shapes
:2&
$token_and_position_embedding_1/Shape?
2token_and_position_embedding_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????24
2token_and_position_embedding_1/strided_slice/stack?
4token_and_position_embedding_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_1/strided_slice/stack_1?
4token_and_position_embedding_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_1/strided_slice/stack_2?
,token_and_position_embedding_1/strided_sliceStridedSlice-token_and_position_embedding_1/Shape:output:0;token_and_position_embedding_1/strided_slice/stack:output:0=token_and_position_embedding_1/strided_slice/stack_1:output:0=token_and_position_embedding_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_1/strided_slice?
*token_and_position_embedding_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_1/range/start?
*token_and_position_embedding_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_1/range/delta?
$token_and_position_embedding_1/rangeRange3token_and_position_embedding_1/range/start:output:05token_and_position_embedding_1/strided_slice:output:03token_and_position_embedding_1/range/delta:output:0*#
_output_shapes
:?????????2&
$token_and_position_embedding_1/range?
;token_and_position_embedding_1/embedding_3/embedding_lookupResourceGatherAtoken_and_position_embedding_1_embedding_3_embedding_lookup_13445-token_and_position_embedding_1/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@token_and_position_embedding_1/embedding_3/embedding_lookup/13445*'
_output_shapes
:????????? *
dtype02=
;token_and_position_embedding_1/embedding_3/embedding_lookup?
Dtoken_and_position_embedding_1/embedding_3/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_1/embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@token_and_position_embedding_1/embedding_3/embedding_lookup/13445*'
_output_shapes
:????????? 2F
Dtoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity?
Ftoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2H
Ftoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1?
/token_and_position_embedding_1/embedding_2/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:??????????R21
/token_and_position_embedding_1/embedding_2/Cast?
;token_and_position_embedding_1/embedding_2/embedding_lookupResourceGatherAtoken_and_position_embedding_1_embedding_2_embedding_lookup_134513token_and_position_embedding_1/embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@token_and_position_embedding_1/embedding_2/embedding_lookup/13451*,
_output_shapes
:??????????R *
dtype02=
;token_and_position_embedding_1/embedding_2/embedding_lookup?
Dtoken_and_position_embedding_1/embedding_2/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_1/embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@token_and_position_embedding_1/embedding_2/embedding_lookup/13451*,
_output_shapes
:??????????R 2F
Dtoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity?
Ftoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????R 2H
Ftoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1?
"token_and_position_embedding_1/addAddV2Otoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????R 2$
"token_and_position_embedding_1/add?
"average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_1/ExpandDims/dim?
average_pooling1d_1/ExpandDims
ExpandDims&token_and_position_embedding_1/add:z:0+average_pooling1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????R 2 
average_pooling1d_1/ExpandDims?
average_pooling1d_1/AvgPoolAvgPool'average_pooling1d_1/ExpandDims:output:0*
T0*/
_output_shapes
:?????????# *
ksize	
?*
paddingVALID*
strides	
?2
average_pooling1d_1/AvgPool?
average_pooling1d_1/SqueezeSqueeze$average_pooling1d_1/AvgPool:output:0*
T0*+
_output_shapes
:?????????# *
squeeze_dims
2
average_pooling1d_1/Squeeze?
Mtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_3_multi_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp?
>transformer_block_3/multi_head_attention_3/query/einsum/EinsumEinsum$average_pooling1d_1/Squeeze:output:0Utransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2@
>transformer_block_3/multi_head_attention_3/query/einsum/Einsum?
Ctransformer_block_3/multi_head_attention_3/query/add/ReadVariableOpReadVariableOpLtransformer_block_3_multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_3/multi_head_attention_3/query/add/ReadVariableOp?
4transformer_block_3/multi_head_attention_3/query/addAddV2Gtransformer_block_3/multi_head_attention_3/query/einsum/Einsum:output:0Ktransformer_block_3/multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 26
4transformer_block_3/multi_head_attention_3/query/add?
Ktransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_3_multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp?
<transformer_block_3/multi_head_attention_3/key/einsum/EinsumEinsum$average_pooling1d_1/Squeeze:output:0Stransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2>
<transformer_block_3/multi_head_attention_3/key/einsum/Einsum?
Atransformer_block_3/multi_head_attention_3/key/add/ReadVariableOpReadVariableOpJtransformer_block_3_multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_3/multi_head_attention_3/key/add/ReadVariableOp?
2transformer_block_3/multi_head_attention_3/key/addAddV2Etransformer_block_3/multi_head_attention_3/key/einsum/Einsum:output:0Itransformer_block_3/multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 24
2transformer_block_3/multi_head_attention_3/key/add?
Mtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_3_multi_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp?
>transformer_block_3/multi_head_attention_3/value/einsum/EinsumEinsum$average_pooling1d_1/Squeeze:output:0Utransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2@
>transformer_block_3/multi_head_attention_3/value/einsum/Einsum?
Ctransformer_block_3/multi_head_attention_3/value/add/ReadVariableOpReadVariableOpLtransformer_block_3_multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_3/multi_head_attention_3/value/add/ReadVariableOp?
4transformer_block_3/multi_head_attention_3/value/addAddV2Gtransformer_block_3/multi_head_attention_3/value/einsum/Einsum:output:0Ktransformer_block_3/multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 26
4transformer_block_3/multi_head_attention_3/value/add?
0transformer_block_3/multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>22
0transformer_block_3/multi_head_attention_3/Mul/y?
.transformer_block_3/multi_head_attention_3/MulMul8transformer_block_3/multi_head_attention_3/query/add:z:09transformer_block_3/multi_head_attention_3/Mul/y:output:0*
T0*/
_output_shapes
:?????????# 20
.transformer_block_3/multi_head_attention_3/Mul?
8transformer_block_3/multi_head_attention_3/einsum/EinsumEinsum6transformer_block_3/multi_head_attention_3/key/add:z:02transformer_block_3/multi_head_attention_3/Mul:z:0*
N*
T0*/
_output_shapes
:?????????##*
equationaecd,abcd->acbe2:
8transformer_block_3/multi_head_attention_3/einsum/Einsum?
:transformer_block_3/multi_head_attention_3/softmax/SoftmaxSoftmaxAtransformer_block_3/multi_head_attention_3/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????##2<
:transformer_block_3/multi_head_attention_3/softmax/Softmax?
;transformer_block_3/multi_head_attention_3/dropout/IdentityIdentityDtransformer_block_3/multi_head_attention_3/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????##2=
;transformer_block_3/multi_head_attention_3/dropout/Identity?
:transformer_block_3/multi_head_attention_3/einsum_1/EinsumEinsumDtransformer_block_3/multi_head_attention_3/dropout/Identity:output:08transformer_block_3/multi_head_attention_3/value/add:z:0*
N*
T0*/
_output_shapes
:?????????# *
equationacbe,aecd->abcd2<
:transformer_block_3/multi_head_attention_3/einsum_1/Einsum?
Xtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_3_multi_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp?
Itransformer_block_3/multi_head_attention_3/attention_output/einsum/EinsumEinsumCtransformer_block_3/multi_head_attention_3/einsum_1/Einsum:output:0`transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????# *
equationabcd,cde->abe2K
Itransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum?
Ntransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_3_multi_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp?
?transformer_block_3/multi_head_attention_3/attention_output/addAddV2Rtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum:output:0Vtransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2A
?transformer_block_3/multi_head_attention_3/attention_output/add?
&transformer_block_3/dropout_8/IdentityIdentityCtransformer_block_3/multi_head_attention_3/attention_output/add:z:0*
T0*+
_output_shapes
:?????????# 2(
&transformer_block_3/dropout_8/Identity?
transformer_block_3/addAddV2$average_pooling1d_1/Squeeze:output:0/transformer_block_3/dropout_8/Identity:output:0*
T0*+
_output_shapes
:?????????# 2
transformer_block_3/add?
Htransformer_block_3/layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_3/layer_normalization_6/moments/mean/reduction_indices?
6transformer_block_3/layer_normalization_6/moments/meanMeantransformer_block_3/add:z:0Qtransformer_block_3/layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(28
6transformer_block_3/layer_normalization_6/moments/mean?
>transformer_block_3/layer_normalization_6/moments/StopGradientStopGradient?transformer_block_3/layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2@
>transformer_block_3/layer_normalization_6/moments/StopGradient?
Ctransformer_block_3/layer_normalization_6/moments/SquaredDifferenceSquaredDifferencetransformer_block_3/add:z:0Gtransformer_block_3/layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 2E
Ctransformer_block_3/layer_normalization_6/moments/SquaredDifference?
Ltransformer_block_3/layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_3/layer_normalization_6/moments/variance/reduction_indices?
:transformer_block_3/layer_normalization_6/moments/varianceMeanGtransformer_block_3/layer_normalization_6/moments/SquaredDifference:z:0Utransformer_block_3/layer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2<
:transformer_block_3/layer_normalization_6/moments/variance?
9transformer_block_3/layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52;
9transformer_block_3/layer_normalization_6/batchnorm/add/y?
7transformer_block_3/layer_normalization_6/batchnorm/addAddV2Ctransformer_block_3/layer_normalization_6/moments/variance:output:0Btransformer_block_3/layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#29
7transformer_block_3/layer_normalization_6/batchnorm/add?
9transformer_block_3/layer_normalization_6/batchnorm/RsqrtRsqrt;transformer_block_3/layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2;
9transformer_block_3/layer_normalization_6/batchnorm/Rsqrt?
Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_3_layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp?
7transformer_block_3/layer_normalization_6/batchnorm/mulMul=transformer_block_3/layer_normalization_6/batchnorm/Rsqrt:y:0Ntransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 29
7transformer_block_3/layer_normalization_6/batchnorm/mul?
9transformer_block_3/layer_normalization_6/batchnorm/mul_1Multransformer_block_3/add:z:0;transformer_block_3/layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2;
9transformer_block_3/layer_normalization_6/batchnorm/mul_1?
9transformer_block_3/layer_normalization_6/batchnorm/mul_2Mul?transformer_block_3/layer_normalization_6/moments/mean:output:0;transformer_block_3/layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2;
9transformer_block_3/layer_normalization_6/batchnorm/mul_2?
Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_3_layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp?
7transformer_block_3/layer_normalization_6/batchnorm/subSubJtransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp:value:0=transformer_block_3/layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 29
7transformer_block_3/layer_normalization_6/batchnorm/sub?
9transformer_block_3/layer_normalization_6/batchnorm/add_1AddV2=transformer_block_3/layer_normalization_6/batchnorm/mul_1:z:0;transformer_block_3/layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2;
9transformer_block_3/layer_normalization_6/batchnorm/add_1?
Atransformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_3_sequential_3_dense_9_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02C
Atransformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOp?
7transformer_block_3/sequential_3/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_3/sequential_3/dense_9/Tensordot/axes?
7transformer_block_3/sequential_3/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_3/sequential_3/dense_9/Tensordot/free?
8transformer_block_3/sequential_3/dense_9/Tensordot/ShapeShape=transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:2:
8transformer_block_3/sequential_3/dense_9/Tensordot/Shape?
@transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2/axis?
;transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2GatherV2Atransformer_block_3/sequential_3/dense_9/Tensordot/Shape:output:0@transformer_block_3/sequential_3/dense_9/Tensordot/free:output:0Itransformer_block_3/sequential_3/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2?
Btransformer_block_3/sequential_3/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_3/sequential_3/dense_9/Tensordot/GatherV2_1/axis?
=transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2_1GatherV2Atransformer_block_3/sequential_3/dense_9/Tensordot/Shape:output:0@transformer_block_3/sequential_3/dense_9/Tensordot/axes:output:0Ktransformer_block_3/sequential_3/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_3/sequential_3/dense_9/Tensordot/GatherV2_1?
8transformer_block_3/sequential_3/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_3/sequential_3/dense_9/Tensordot/Const?
7transformer_block_3/sequential_3/dense_9/Tensordot/ProdProdDtransformer_block_3/sequential_3/dense_9/Tensordot/GatherV2:output:0Atransformer_block_3/sequential_3/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_3/sequential_3/dense_9/Tensordot/Prod?
:transformer_block_3/sequential_3/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_3/sequential_3/dense_9/Tensordot/Const_1?
9transformer_block_3/sequential_3/dense_9/Tensordot/Prod_1ProdFtransformer_block_3/sequential_3/dense_9/Tensordot/GatherV2_1:output:0Ctransformer_block_3/sequential_3/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_3/sequential_3/dense_9/Tensordot/Prod_1?
>transformer_block_3/sequential_3/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_3/sequential_3/dense_9/Tensordot/concat/axis?
9transformer_block_3/sequential_3/dense_9/Tensordot/concatConcatV2@transformer_block_3/sequential_3/dense_9/Tensordot/free:output:0@transformer_block_3/sequential_3/dense_9/Tensordot/axes:output:0Gtransformer_block_3/sequential_3/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_3/sequential_3/dense_9/Tensordot/concat?
8transformer_block_3/sequential_3/dense_9/Tensordot/stackPack@transformer_block_3/sequential_3/dense_9/Tensordot/Prod:output:0Btransformer_block_3/sequential_3/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_3/sequential_3/dense_9/Tensordot/stack?
<transformer_block_3/sequential_3/dense_9/Tensordot/transpose	Transpose=transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0Btransformer_block_3/sequential_3/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????# 2>
<transformer_block_3/sequential_3/dense_9/Tensordot/transpose?
:transformer_block_3/sequential_3/dense_9/Tensordot/ReshapeReshape@transformer_block_3/sequential_3/dense_9/Tensordot/transpose:y:0Atransformer_block_3/sequential_3/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block_3/sequential_3/dense_9/Tensordot/Reshape?
9transformer_block_3/sequential_3/dense_9/Tensordot/MatMulMatMulCtransformer_block_3/sequential_3/dense_9/Tensordot/Reshape:output:0Itransformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2;
9transformer_block_3/sequential_3/dense_9/Tensordot/MatMul?
:transformer_block_3/sequential_3/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2<
:transformer_block_3/sequential_3/dense_9/Tensordot/Const_2?
@transformer_block_3/sequential_3/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_3/sequential_3/dense_9/Tensordot/concat_1/axis?
;transformer_block_3/sequential_3/dense_9/Tensordot/concat_1ConcatV2Dtransformer_block_3/sequential_3/dense_9/Tensordot/GatherV2:output:0Ctransformer_block_3/sequential_3/dense_9/Tensordot/Const_2:output:0Itransformer_block_3/sequential_3/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_3/sequential_3/dense_9/Tensordot/concat_1?
2transformer_block_3/sequential_3/dense_9/TensordotReshapeCtransformer_block_3/sequential_3/dense_9/Tensordot/MatMul:product:0Dtransformer_block_3/sequential_3/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@24
2transformer_block_3/sequential_3/dense_9/Tensordot?
?transformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_3_sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02A
?transformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOp?
0transformer_block_3/sequential_3/dense_9/BiasAddBiasAdd;transformer_block_3/sequential_3/dense_9/Tensordot:output:0Gtransformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@22
0transformer_block_3/sequential_3/dense_9/BiasAdd?
-transformer_block_3/sequential_3/dense_9/ReluRelu9transformer_block_3/sequential_3/dense_9/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#@2/
-transformer_block_3/sequential_3/dense_9/Relu?
Btransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_3_sequential_3_dense_10_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02D
Btransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp?
8transformer_block_3/sequential_3/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_3/sequential_3/dense_10/Tensordot/axes?
8transformer_block_3/sequential_3/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_3/sequential_3/dense_10/Tensordot/free?
9transformer_block_3/sequential_3/dense_10/Tensordot/ShapeShape;transformer_block_3/sequential_3/dense_9/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_3/sequential_3/dense_10/Tensordot/Shape?
Atransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2/axis?
<transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2GatherV2Btransformer_block_3/sequential_3/dense_10/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_10/Tensordot/free:output:0Jtransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2?
Ctransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1/axis?
>transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1GatherV2Btransformer_block_3/sequential_3/dense_10/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_10/Tensordot/axes:output:0Ltransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1?
9transformer_block_3/sequential_3/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_3/sequential_3/dense_10/Tensordot/Const?
8transformer_block_3/sequential_3/dense_10/Tensordot/ProdProdEtransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2:output:0Btransformer_block_3/sequential_3/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_3/sequential_3/dense_10/Tensordot/Prod?
;transformer_block_3/sequential_3/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_3/sequential_3/dense_10/Tensordot/Const_1?
:transformer_block_3/sequential_3/dense_10/Tensordot/Prod_1ProdGtransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1:output:0Dtransformer_block_3/sequential_3/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_3/sequential_3/dense_10/Tensordot/Prod_1?
?transformer_block_3/sequential_3/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_3/sequential_3/dense_10/Tensordot/concat/axis?
:transformer_block_3/sequential_3/dense_10/Tensordot/concatConcatV2Atransformer_block_3/sequential_3/dense_10/Tensordot/free:output:0Atransformer_block_3/sequential_3/dense_10/Tensordot/axes:output:0Htransformer_block_3/sequential_3/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_3/sequential_3/dense_10/Tensordot/concat?
9transformer_block_3/sequential_3/dense_10/Tensordot/stackPackAtransformer_block_3/sequential_3/dense_10/Tensordot/Prod:output:0Ctransformer_block_3/sequential_3/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_3/sequential_3/dense_10/Tensordot/stack?
=transformer_block_3/sequential_3/dense_10/Tensordot/transpose	Transpose;transformer_block_3/sequential_3/dense_9/Relu:activations:0Ctransformer_block_3/sequential_3/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2?
=transformer_block_3/sequential_3/dense_10/Tensordot/transpose?
;transformer_block_3/sequential_3/dense_10/Tensordot/ReshapeReshapeAtransformer_block_3/sequential_3/dense_10/Tensordot/transpose:y:0Btransformer_block_3/sequential_3/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2=
;transformer_block_3/sequential_3/dense_10/Tensordot/Reshape?
:transformer_block_3/sequential_3/dense_10/Tensordot/MatMulMatMulDtransformer_block_3/sequential_3/dense_10/Tensordot/Reshape:output:0Jtransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2<
:transformer_block_3/sequential_3/dense_10/Tensordot/MatMul?
;transformer_block_3/sequential_3/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_3/sequential_3/dense_10/Tensordot/Const_2?
Atransformer_block_3/sequential_3/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_3/sequential_3/dense_10/Tensordot/concat_1/axis?
<transformer_block_3/sequential_3/dense_10/Tensordot/concat_1ConcatV2Etransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2:output:0Dtransformer_block_3/sequential_3/dense_10/Tensordot/Const_2:output:0Jtransformer_block_3/sequential_3/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_3/sequential_3/dense_10/Tensordot/concat_1?
3transformer_block_3/sequential_3/dense_10/TensordotReshapeDtransformer_block_3/sequential_3/dense_10/Tensordot/MatMul:product:0Etransformer_block_3/sequential_3/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 25
3transformer_block_3/sequential_3/dense_10/Tensordot?
@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_3_sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp?
1transformer_block_3/sequential_3/dense_10/BiasAddBiasAdd<transformer_block_3/sequential_3/dense_10/Tensordot:output:0Htransformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 23
1transformer_block_3/sequential_3/dense_10/BiasAdd?
&transformer_block_3/dropout_9/IdentityIdentity:transformer_block_3/sequential_3/dense_10/BiasAdd:output:0*
T0*+
_output_shapes
:?????????# 2(
&transformer_block_3/dropout_9/Identity?
transformer_block_3/add_1AddV2=transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0/transformer_block_3/dropout_9/Identity:output:0*
T0*+
_output_shapes
:?????????# 2
transformer_block_3/add_1?
Htransformer_block_3/layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_3/layer_normalization_7/moments/mean/reduction_indices?
6transformer_block_3/layer_normalization_7/moments/meanMeantransformer_block_3/add_1:z:0Qtransformer_block_3/layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(28
6transformer_block_3/layer_normalization_7/moments/mean?
>transformer_block_3/layer_normalization_7/moments/StopGradientStopGradient?transformer_block_3/layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2@
>transformer_block_3/layer_normalization_7/moments/StopGradient?
Ctransformer_block_3/layer_normalization_7/moments/SquaredDifferenceSquaredDifferencetransformer_block_3/add_1:z:0Gtransformer_block_3/layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 2E
Ctransformer_block_3/layer_normalization_7/moments/SquaredDifference?
Ltransformer_block_3/layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_3/layer_normalization_7/moments/variance/reduction_indices?
:transformer_block_3/layer_normalization_7/moments/varianceMeanGtransformer_block_3/layer_normalization_7/moments/SquaredDifference:z:0Utransformer_block_3/layer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2<
:transformer_block_3/layer_normalization_7/moments/variance?
9transformer_block_3/layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52;
9transformer_block_3/layer_normalization_7/batchnorm/add/y?
7transformer_block_3/layer_normalization_7/batchnorm/addAddV2Ctransformer_block_3/layer_normalization_7/moments/variance:output:0Btransformer_block_3/layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#29
7transformer_block_3/layer_normalization_7/batchnorm/add?
9transformer_block_3/layer_normalization_7/batchnorm/RsqrtRsqrt;transformer_block_3/layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2;
9transformer_block_3/layer_normalization_7/batchnorm/Rsqrt?
Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_3_layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp?
7transformer_block_3/layer_normalization_7/batchnorm/mulMul=transformer_block_3/layer_normalization_7/batchnorm/Rsqrt:y:0Ntransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 29
7transformer_block_3/layer_normalization_7/batchnorm/mul?
9transformer_block_3/layer_normalization_7/batchnorm/mul_1Multransformer_block_3/add_1:z:0;transformer_block_3/layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2;
9transformer_block_3/layer_normalization_7/batchnorm/mul_1?
9transformer_block_3/layer_normalization_7/batchnorm/mul_2Mul?transformer_block_3/layer_normalization_7/moments/mean:output:0;transformer_block_3/layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2;
9transformer_block_3/layer_normalization_7/batchnorm/mul_2?
Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_3_layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp?
7transformer_block_3/layer_normalization_7/batchnorm/subSubJtransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp:value:0=transformer_block_3/layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 29
7transformer_block_3/layer_normalization_7/batchnorm/sub?
9transformer_block_3/layer_normalization_7/batchnorm/add_1AddV2=transformer_block_3/layer_normalization_7/batchnorm/mul_1:z:0;transformer_block_3/layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2;
9transformer_block_3/layer_normalization_7/batchnorm/add_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
flatten_1/Const?
flatten_1/ReshapeReshape=transformer_block_3/layer_normalization_7/batchnorm/add_1:z:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshape?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
dense_11/MatMul/ReadVariableOp?
dense_11/MatMulMatMulflatten_1/Reshape:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_11/MatMul?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_11/BiasAdds
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_11/Relu?
dropout_10/IdentityIdentitydense_11/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
dropout_10/Identity?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMuldropout_10/Identity:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_12/Relu?
dropout_11/IdentityIdentitydense_12/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
dropout_11/Identity?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMuldropout_11/Identity:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/BiasAdd?
IdentityIdentitydense_13/BiasAdd:output:0 ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp<^token_and_position_embedding_1/embedding_2/embedding_lookup<^token_and_position_embedding_1/embedding_3/embedding_lookupC^transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpG^transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpC^transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpG^transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpO^transformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpY^transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_3/multi_head_attention_3/key/add/ReadVariableOpL^transformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpD^transformer_block_3/multi_head_attention_3/query/add/ReadVariableOpN^transformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpD^transformer_block_3/multi_head_attention_3/value/add/ReadVariableOpN^transformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpA^transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOpC^transformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp@^transformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOpB^transformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesv
t:??????????R::::::::::::::::::::::::2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2z
;token_and_position_embedding_1/embedding_2/embedding_lookup;token_and_position_embedding_1/embedding_2/embedding_lookup2z
;token_and_position_embedding_1/embedding_3/embedding_lookup;token_and_position_embedding_1/embedding_3/embedding_lookup2?
Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpBtransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp2?
Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpFtransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp2?
Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpBtransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp2?
Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpFtransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp2?
Ntransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpNtransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp2?
Xtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp2?
Atransformer_block_3/multi_head_attention_3/key/add/ReadVariableOpAtransformer_block_3/multi_head_attention_3/key/add/ReadVariableOp2?
Ktransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpKtransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp2?
Ctransformer_block_3/multi_head_attention_3/query/add/ReadVariableOpCtransformer_block_3/multi_head_attention_3/query/add/ReadVariableOp2?
Mtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpMtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp2?
Ctransformer_block_3/multi_head_attention_3/value/add/ReadVariableOpCtransformer_block_3/multi_head_attention_3/value/add/ReadVariableOp2?
Mtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpMtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp2?
@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp2?
Btransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOpBtransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp2?
?transformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOp?transformer_block_3/sequential_3/dense_9/BiasAdd/ReadVariableOp2?
Atransformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOpAtransformer_block_3/sequential_3/dense_9/Tensordot/ReadVariableOp:P L
(
_output_shapes
:??????????R
 
_user_specified_nameinputs
?
?
C__inference_dense_10_layer_call_and_return_conditional_losses_12203

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????#@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????#@
 
_user_specified_nameinputs
?
c
E__inference_dropout_11_layer_call_and_return_conditional_losses_14192

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
3__inference_transformer_block_3_layer_call_fn_14060

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_124822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:?????????# ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
?I
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_14335

inputs-
)dense_9_tensordot_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource.
*dense_10_tensordot_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource
identity??dense_10/BiasAdd/ReadVariableOp?!dense_10/Tensordot/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp? dense_9/Tensordot/ReadVariableOp?
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02"
 dense_9/Tensordot/ReadVariableOpz
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/axes?
dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_9/Tensordot/freeh
dense_9/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_9/Tensordot/Shape?
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/GatherV2/axis?
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2?
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_9/Tensordot/GatherV2_1/axis?
dense_9/Tensordot/GatherV2_1GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/axes:output:0*dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2_1|
dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const?
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod?
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_1?
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod_1?
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_9/Tensordot/concat/axis?
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat?
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/stack?
dense_9/Tensordot/transpose	Transposeinputs!dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????# 2
dense_9/Tensordot/transpose?
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_9/Tensordot/Reshape?
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_9/Tensordot/MatMul?
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_9/Tensordot/Const_2?
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/concat_1/axis?
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat_1?
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@2
dense_9/Tensordot?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@2
dense_9/BiasAddt
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#@2
dense_9/Relu?
!dense_10/Tensordot/ReadVariableOpReadVariableOp*dense_10_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02#
!dense_10/Tensordot/ReadVariableOp|
dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/axes?
dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_10/Tensordot/free~
dense_10/Tensordot/ShapeShapedense_9/Relu:activations:0*
T0*
_output_shapes
:2
dense_10/Tensordot/Shape?
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/GatherV2/axis?
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2?
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_10/Tensordot/GatherV2_1/axis?
dense_10/Tensordot/GatherV2_1GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/axes:output:0+dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2_1~
dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const?
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod?
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_1?
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod_1?
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_10/Tensordot/concat/axis?
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat?
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/stack?
dense_10/Tensordot/transpose	Transposedense_9/Relu:activations:0"dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2
dense_10/Tensordot/transpose?
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_10/Tensordot/Reshape?
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_10/Tensordot/MatMul?
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_2?
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/concat_1/axis?
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat_1?
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 2
dense_10/Tensordot?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2
dense_10/BiasAdd?
IdentityIdentitydense_10/BiasAdd:output:0 ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
?
?
,__inference_sequential_3_layer_call_fn_14361

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_122782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
?4
?
B__inference_model_1_layer_call_and_return_conditional_losses_12873
input_2(
$token_and_position_embedding_1_12328(
$token_and_position_embedding_1_12330
transformer_block_3_12685
transformer_block_3_12687
transformer_block_3_12689
transformer_block_3_12691
transformer_block_3_12693
transformer_block_3_12695
transformer_block_3_12697
transformer_block_3_12699
transformer_block_3_12701
transformer_block_3_12703
transformer_block_3_12705
transformer_block_3_12707
transformer_block_3_12709
transformer_block_3_12711
transformer_block_3_12713
transformer_block_3_12715
dense_11_12754
dense_11_12756
dense_12_12811
dense_12_12813
dense_13_12867
dense_13_12869
identity?? dense_11/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?"dropout_10/StatefulPartitionedCall?"dropout_11/StatefulPartitionedCall?6token_and_position_embedding_1/StatefulPartitionedCall?+transformer_block_3/StatefulPartitionedCall?
6token_and_position_embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_2$token_and_position_embedding_1_12328$token_and_position_embedding_1_12330*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *b
f]R[
Y__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_1231728
6token_and_position_embedding_1/StatefulPartitionedCall?
#average_pooling1d_1/PartitionedCallPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_121162%
#average_pooling1d_1/PartitionedCall?
+transformer_block_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_1/PartitionedCall:output:0transformer_block_3_12685transformer_block_3_12687transformer_block_3_12689transformer_block_3_12691transformer_block_3_12693transformer_block_3_12695transformer_block_3_12697transformer_block_3_12699transformer_block_3_12701transformer_block_3_12703transformer_block_3_12705transformer_block_3_12707transformer_block_3_12709transformer_block_3_12711transformer_block_3_12713transformer_block_3_12715*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_124822-
+transformer_block_3/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall4transformer_block_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_127242
flatten_1/PartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_11_12754dense_11_12756*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_127432"
 dense_11/StatefulPartitionedCall?
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_127712$
"dropout_10/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0dense_12_12811dense_12_12813*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_128002"
 dense_12/StatefulPartitionedCall?
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_128282$
"dropout_11/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_13_12867dense_13_12869*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_128562"
 dense_13/StatefulPartitionedCall?
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall7^token_and_position_embedding_1/StatefulPartitionedCall,^transformer_block_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesv
t:??????????R::::::::::::::::::::::::2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2p
6token_and_position_embedding_1/StatefulPartitionedCall6token_and_position_embedding_1/StatefulPartitionedCall2Z
+transformer_block_3/StatefulPartitionedCall+transformer_block_3/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????R
!
_user_specified_name	input_2
?
j
N__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_12116

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+???????????????????????????*
ksize	
?*
paddingVALID*
strides	
?2	
AvgPool?
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_12220
dense_9_input
dense_9_12168
dense_9_12170
dense_10_12214
dense_10_12216
identity?? dense_10/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCalldense_9_inputdense_9_12168dense_9_12170*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_121572!
dense_9/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_12214dense_10_12216*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_122032"
 dense_10/StatefulPartitionedCall?
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????# 
'
_user_specified_namedense_9_input
??
?
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_13896

inputsF
Bmulti_head_attention_3_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_3_query_add_readvariableop_resourceD
@multi_head_attention_3_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_3_key_add_readvariableop_resourceF
Bmulti_head_attention_3_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_3_value_add_readvariableop_resourceQ
Mmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_3_attention_output_add_readvariableop_resource?
;layer_normalization_6_batchnorm_mul_readvariableop_resource;
7layer_normalization_6_batchnorm_readvariableop_resource:
6sequential_3_dense_9_tensordot_readvariableop_resource8
4sequential_3_dense_9_biasadd_readvariableop_resource;
7sequential_3_dense_10_tensordot_readvariableop_resource9
5sequential_3_dense_10_biasadd_readvariableop_resource?
;layer_normalization_7_batchnorm_mul_readvariableop_resource;
7layer_normalization_7_batchnorm_readvariableop_resource
identity??.layer_normalization_6/batchnorm/ReadVariableOp?2layer_normalization_6/batchnorm/mul/ReadVariableOp?.layer_normalization_7/batchnorm/ReadVariableOp?2layer_normalization_7/batchnorm/mul/ReadVariableOp?:multi_head_attention_3/attention_output/add/ReadVariableOp?Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp?-multi_head_attention_3/key/add/ReadVariableOp?7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp?/multi_head_attention_3/query/add/ReadVariableOp?9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp?/multi_head_attention_3/value/add/ReadVariableOp?9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp?,sequential_3/dense_10/BiasAdd/ReadVariableOp?.sequential_3/dense_10/Tensordot/ReadVariableOp?+sequential_3/dense_9/BiasAdd/ReadVariableOp?-sequential_3/dense_9/Tensordot/ReadVariableOp?
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp?
*multi_head_attention_3/query/einsum/EinsumEinsuminputsAmulti_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2,
*multi_head_attention_3/query/einsum/Einsum?
/multi_head_attention_3/query/add/ReadVariableOpReadVariableOp8multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/query/add/ReadVariableOp?
 multi_head_attention_3/query/addAddV23multi_head_attention_3/query/einsum/Einsum:output:07multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2"
 multi_head_attention_3/query/add?
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp?
(multi_head_attention_3/key/einsum/EinsumEinsuminputs?multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2*
(multi_head_attention_3/key/einsum/Einsum?
-multi_head_attention_3/key/add/ReadVariableOpReadVariableOp6multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_3/key/add/ReadVariableOp?
multi_head_attention_3/key/addAddV21multi_head_attention_3/key/einsum/Einsum:output:05multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2 
multi_head_attention_3/key/add?
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp?
*multi_head_attention_3/value/einsum/EinsumEinsuminputsAmulti_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2,
*multi_head_attention_3/value/einsum/Einsum?
/multi_head_attention_3/value/add/ReadVariableOpReadVariableOp8multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/value/add/ReadVariableOp?
 multi_head_attention_3/value/addAddV23multi_head_attention_3/value/einsum/Einsum:output:07multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2"
 multi_head_attention_3/value/add?
multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>2
multi_head_attention_3/Mul/y?
multi_head_attention_3/MulMul$multi_head_attention_3/query/add:z:0%multi_head_attention_3/Mul/y:output:0*
T0*/
_output_shapes
:?????????# 2
multi_head_attention_3/Mul?
$multi_head_attention_3/einsum/EinsumEinsum"multi_head_attention_3/key/add:z:0multi_head_attention_3/Mul:z:0*
N*
T0*/
_output_shapes
:?????????##*
equationaecd,abcd->acbe2&
$multi_head_attention_3/einsum/Einsum?
&multi_head_attention_3/softmax/SoftmaxSoftmax-multi_head_attention_3/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????##2(
&multi_head_attention_3/softmax/Softmax?
,multi_head_attention_3/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,multi_head_attention_3/dropout/dropout/Const?
*multi_head_attention_3/dropout/dropout/MulMul0multi_head_attention_3/softmax/Softmax:softmax:05multi_head_attention_3/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????##2,
*multi_head_attention_3/dropout/dropout/Mul?
,multi_head_attention_3/dropout/dropout/ShapeShape0multi_head_attention_3/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_3/dropout/dropout/Shape?
Cmulti_head_attention_3/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_3/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????##*
dtype02E
Cmulti_head_attention_3/dropout/dropout/random_uniform/RandomUniform?
5multi_head_attention_3/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_3/dropout/dropout/GreaterEqual/y?
3multi_head_attention_3/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_3/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_3/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????##25
3multi_head_attention_3/dropout/dropout/GreaterEqual?
+multi_head_attention_3/dropout/dropout/CastCast7multi_head_attention_3/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????##2-
+multi_head_attention_3/dropout/dropout/Cast?
,multi_head_attention_3/dropout/dropout/Mul_1Mul.multi_head_attention_3/dropout/dropout/Mul:z:0/multi_head_attention_3/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????##2.
,multi_head_attention_3/dropout/dropout/Mul_1?
&multi_head_attention_3/einsum_1/EinsumEinsum0multi_head_attention_3/dropout/dropout/Mul_1:z:0$multi_head_attention_3/value/add:z:0*
N*
T0*/
_output_shapes
:?????????# *
equationacbe,aecd->abcd2(
&multi_head_attention_3/einsum_1/Einsum?
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp?
5multi_head_attention_3/attention_output/einsum/EinsumEinsum/multi_head_attention_3/einsum_1/Einsum:output:0Lmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????# *
equationabcd,cde->abe27
5multi_head_attention_3/attention_output/einsum/Einsum?
:multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_3/attention_output/add/ReadVariableOp?
+multi_head_attention_3/attention_output/addAddV2>multi_head_attention_3/attention_output/einsum/Einsum:output:0Bmulti_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2-
+multi_head_attention_3/attention_output/addw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_8/dropout/Const?
dropout_8/dropout/MulMul/multi_head_attention_3/attention_output/add:z:0 dropout_8/dropout/Const:output:0*
T0*+
_output_shapes
:?????????# 2
dropout_8/dropout/Mul?
dropout_8/dropout/ShapeShape/multi_head_attention_3/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape?
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????# *
dtype020
.dropout_8/dropout/random_uniform/RandomUniform?
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_8/dropout/GreaterEqual/y?
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????# 2 
dropout_8/dropout/GreaterEqual?
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????# 2
dropout_8/dropout/Cast?
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????# 2
dropout_8/dropout/Mul_1n
addAddV2inputsdropout_8/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????# 2
add?
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_6/moments/mean/reduction_indices?
"layer_normalization_6/moments/meanMeanadd:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2$
"layer_normalization_6/moments/mean?
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2,
*layer_normalization_6/moments/StopGradient?
/layer_normalization_6/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 21
/layer_normalization_6/moments/SquaredDifference?
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_6/moments/variance/reduction_indices?
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2(
&layer_normalization_6/moments/variance?
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_6/batchnorm/add/y?
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2%
#layer_normalization_6/batchnorm/add?
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2'
%layer_normalization_6/batchnorm/Rsqrt?
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_6/batchnorm/mul/ReadVariableOp?
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2%
#layer_normalization_6/batchnorm/mul?
%layer_normalization_6/batchnorm/mul_1Muladd:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_6/batchnorm/mul_1?
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_6/batchnorm/mul_2?
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_6/batchnorm/ReadVariableOp?
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2%
#layer_normalization_6/batchnorm/sub?
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_6/batchnorm/add_1?
-sequential_3/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_3_dense_9_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02/
-sequential_3/dense_9/Tensordot/ReadVariableOp?
#sequential_3/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_3/dense_9/Tensordot/axes?
#sequential_3/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_3/dense_9/Tensordot/free?
$sequential_3/dense_9/Tensordot/ShapeShape)layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_3/dense_9/Tensordot/Shape?
,sequential_3/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_3/dense_9/Tensordot/GatherV2/axis?
'sequential_3/dense_9/Tensordot/GatherV2GatherV2-sequential_3/dense_9/Tensordot/Shape:output:0,sequential_3/dense_9/Tensordot/free:output:05sequential_3/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_3/dense_9/Tensordot/GatherV2?
.sequential_3/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_3/dense_9/Tensordot/GatherV2_1/axis?
)sequential_3/dense_9/Tensordot/GatherV2_1GatherV2-sequential_3/dense_9/Tensordot/Shape:output:0,sequential_3/dense_9/Tensordot/axes:output:07sequential_3/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_3/dense_9/Tensordot/GatherV2_1?
$sequential_3/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_3/dense_9/Tensordot/Const?
#sequential_3/dense_9/Tensordot/ProdProd0sequential_3/dense_9/Tensordot/GatherV2:output:0-sequential_3/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_3/dense_9/Tensordot/Prod?
&sequential_3/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_3/dense_9/Tensordot/Const_1?
%sequential_3/dense_9/Tensordot/Prod_1Prod2sequential_3/dense_9/Tensordot/GatherV2_1:output:0/sequential_3/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_3/dense_9/Tensordot/Prod_1?
*sequential_3/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_3/dense_9/Tensordot/concat/axis?
%sequential_3/dense_9/Tensordot/concatConcatV2,sequential_3/dense_9/Tensordot/free:output:0,sequential_3/dense_9/Tensordot/axes:output:03sequential_3/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_9/Tensordot/concat?
$sequential_3/dense_9/Tensordot/stackPack,sequential_3/dense_9/Tensordot/Prod:output:0.sequential_3/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_3/dense_9/Tensordot/stack?
(sequential_3/dense_9/Tensordot/transpose	Transpose)layer_normalization_6/batchnorm/add_1:z:0.sequential_3/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????# 2*
(sequential_3/dense_9/Tensordot/transpose?
&sequential_3/dense_9/Tensordot/ReshapeReshape,sequential_3/dense_9/Tensordot/transpose:y:0-sequential_3/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&sequential_3/dense_9/Tensordot/Reshape?
%sequential_3/dense_9/Tensordot/MatMulMatMul/sequential_3/dense_9/Tensordot/Reshape:output:05sequential_3/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2'
%sequential_3/dense_9/Tensordot/MatMul?
&sequential_3/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2(
&sequential_3/dense_9/Tensordot/Const_2?
,sequential_3/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_3/dense_9/Tensordot/concat_1/axis?
'sequential_3/dense_9/Tensordot/concat_1ConcatV20sequential_3/dense_9/Tensordot/GatherV2:output:0/sequential_3/dense_9/Tensordot/Const_2:output:05sequential_3/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_3/dense_9/Tensordot/concat_1?
sequential_3/dense_9/TensordotReshape/sequential_3/dense_9/Tensordot/MatMul:product:00sequential_3/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@2 
sequential_3/dense_9/Tensordot?
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOp?
sequential_3/dense_9/BiasAddBiasAdd'sequential_3/dense_9/Tensordot:output:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@2
sequential_3/dense_9/BiasAdd?
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#@2
sequential_3/dense_9/Relu?
.sequential_3/dense_10/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_10_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_3/dense_10/Tensordot/ReadVariableOp?
$sequential_3/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_3/dense_10/Tensordot/axes?
$sequential_3/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_3/dense_10/Tensordot/free?
%sequential_3/dense_10/Tensordot/ShapeShape'sequential_3/dense_9/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/Shape?
-sequential_3/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/GatherV2/axis?
(sequential_3/dense_10/Tensordot/GatherV2GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/free:output:06sequential_3/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/GatherV2?
/sequential_3/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_3/dense_10/Tensordot/GatherV2_1/axis?
*sequential_3/dense_10/Tensordot/GatherV2_1GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/axes:output:08sequential_3/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_3/dense_10/Tensordot/GatherV2_1?
%sequential_3/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_3/dense_10/Tensordot/Const?
$sequential_3/dense_10/Tensordot/ProdProd1sequential_3/dense_10/Tensordot/GatherV2:output:0.sequential_3/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_3/dense_10/Tensordot/Prod?
'sequential_3/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_1?
&sequential_3/dense_10/Tensordot/Prod_1Prod3sequential_3/dense_10/Tensordot/GatherV2_1:output:00sequential_3/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_3/dense_10/Tensordot/Prod_1?
+sequential_3/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_3/dense_10/Tensordot/concat/axis?
&sequential_3/dense_10/Tensordot/concatConcatV2-sequential_3/dense_10/Tensordot/free:output:0-sequential_3/dense_10/Tensordot/axes:output:04sequential_3/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_3/dense_10/Tensordot/concat?
%sequential_3/dense_10/Tensordot/stackPack-sequential_3/dense_10/Tensordot/Prod:output:0/sequential_3/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/stack?
)sequential_3/dense_10/Tensordot/transpose	Transpose'sequential_3/dense_9/Relu:activations:0/sequential_3/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2+
)sequential_3/dense_10/Tensordot/transpose?
'sequential_3/dense_10/Tensordot/ReshapeReshape-sequential_3/dense_10/Tensordot/transpose:y:0.sequential_3/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_3/dense_10/Tensordot/Reshape?
&sequential_3/dense_10/Tensordot/MatMulMatMul0sequential_3/dense_10/Tensordot/Reshape:output:06sequential_3/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_3/dense_10/Tensordot/MatMul?
'sequential_3/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_2?
-sequential_3/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/concat_1/axis?
(sequential_3/dense_10/Tensordot/concat_1ConcatV21sequential_3/dense_10/Tensordot/GatherV2:output:00sequential_3/dense_10/Tensordot/Const_2:output:06sequential_3/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/concat_1?
sequential_3/dense_10/TensordotReshape0sequential_3/dense_10/Tensordot/MatMul:product:01sequential_3/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 2!
sequential_3/dense_10/Tensordot?
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOp?
sequential_3/dense_10/BiasAddBiasAdd(sequential_3/dense_10/Tensordot:output:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2
sequential_3/dense_10/BiasAddw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_9/dropout/Const?
dropout_9/dropout/MulMul&sequential_3/dense_10/BiasAdd:output:0 dropout_9/dropout/Const:output:0*
T0*+
_output_shapes
:?????????# 2
dropout_9/dropout/Mul?
dropout_9/dropout/ShapeShape&sequential_3/dense_10/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape?
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????# *
dtype020
.dropout_9/dropout/random_uniform/RandomUniform?
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_9/dropout/GreaterEqual/y?
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????# 2 
dropout_9/dropout/GreaterEqual?
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????# 2
dropout_9/dropout/Cast?
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????# 2
dropout_9/dropout/Mul_1?
add_1AddV2)layer_normalization_6/batchnorm/add_1:z:0dropout_9/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????# 2
add_1?
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_7/moments/mean/reduction_indices?
"layer_normalization_7/moments/meanMean	add_1:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2$
"layer_normalization_7/moments/mean?
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2,
*layer_normalization_7/moments/StopGradient?
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 21
/layer_normalization_7/moments/SquaredDifference?
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_7/moments/variance/reduction_indices?
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2(
&layer_normalization_7/moments/variance?
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_7/batchnorm/add/y?
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2%
#layer_normalization_7/batchnorm/add?
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2'
%layer_normalization_7/batchnorm/Rsqrt?
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_7/batchnorm/mul/ReadVariableOp?
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2%
#layer_normalization_7/batchnorm/mul?
%layer_normalization_7/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_7/batchnorm/mul_1?
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_7/batchnorm/mul_2?
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_7/batchnorm/ReadVariableOp?
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2%
#layer_normalization_7/batchnorm/sub?
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2'
%layer_normalization_7/batchnorm/add_1?
IdentityIdentity)layer_normalization_7/batchnorm/add_1:z:0/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp;^multi_head_attention_3/attention_output/add/ReadVariableOpE^multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_3/key/add/ReadVariableOp8^multi_head_attention_3/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/query/add/ReadVariableOp:^multi_head_attention_3/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/value/add/ReadVariableOp:^multi_head_attention_3/value/einsum/Einsum/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp/^sequential_3/dense_10/Tensordot/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp.^sequential_3/dense_9/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:?????????# ::::::::::::::::2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_3/attention_output/add/ReadVariableOp:multi_head_attention_3/attention_output/add/ReadVariableOp2?
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_3/key/add/ReadVariableOp-multi_head_attention_3/key/add/ReadVariableOp2r
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/query/add/ReadVariableOp/multi_head_attention_3/query/add/ReadVariableOp2v
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/value/add/ReadVariableOp/multi_head_attention_3/value/add/ReadVariableOp2v
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2`
.sequential_3/dense_10/Tensordot/ReadVariableOp.sequential_3/dense_10/Tensordot/ReadVariableOp2Z
+sequential_3/dense_9/BiasAdd/ReadVariableOp+sequential_3/dense_9/BiasAdd/ReadVariableOp2^
-sequential_3/dense_9/Tensordot/ReadVariableOp-sequential_3/dense_9/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
?
E
)__inference_flatten_1_layer_call_fn_14108

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_127242
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????# :S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
?
|
'__inference_dense_9_layer_call_fn_14401

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_121572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????#@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????# ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
?
d
E__inference_dropout_10_layer_call_and_return_conditional_losses_14140

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?I
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_14278

inputs-
)dense_9_tensordot_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource.
*dense_10_tensordot_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource
identity??dense_10/BiasAdd/ReadVariableOp?!dense_10/Tensordot/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp? dense_9/Tensordot/ReadVariableOp?
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02"
 dense_9/Tensordot/ReadVariableOpz
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/axes?
dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_9/Tensordot/freeh
dense_9/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_9/Tensordot/Shape?
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/GatherV2/axis?
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2?
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_9/Tensordot/GatherV2_1/axis?
dense_9/Tensordot/GatherV2_1GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/axes:output:0*dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2_1|
dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const?
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod?
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_1?
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod_1?
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_9/Tensordot/concat/axis?
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat?
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/stack?
dense_9/Tensordot/transpose	Transposeinputs!dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????# 2
dense_9/Tensordot/transpose?
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_9/Tensordot/Reshape?
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_9/Tensordot/MatMul?
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_9/Tensordot/Const_2?
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/concat_1/axis?
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat_1?
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@2
dense_9/Tensordot?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@2
dense_9/BiasAddt
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#@2
dense_9/Relu?
!dense_10/Tensordot/ReadVariableOpReadVariableOp*dense_10_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02#
!dense_10/Tensordot/ReadVariableOp|
dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/axes?
dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_10/Tensordot/free~
dense_10/Tensordot/ShapeShapedense_9/Relu:activations:0*
T0*
_output_shapes
:2
dense_10/Tensordot/Shape?
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/GatherV2/axis?
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2?
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_10/Tensordot/GatherV2_1/axis?
dense_10/Tensordot/GatherV2_1GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/axes:output:0+dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2_1~
dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const?
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod?
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_1?
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod_1?
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_10/Tensordot/concat/axis?
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat?
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/stack?
dense_10/Tensordot/transpose	Transposedense_9/Relu:activations:0"dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2
dense_10/Tensordot/transpose?
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_10/Tensordot/Reshape?
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_10/Tensordot/MatMul?
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_2?
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/concat_1/axis?
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat_1?
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 2
dense_10/Tensordot?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2
dense_10/BiasAdd?
IdentityIdentitydense_10/BiasAdd:output:0 ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_13049
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_129982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesv
t:??????????R::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????R
!
_user_specified_name	input_2
?
F
*__inference_dropout_10_layer_call_fn_14155

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_127762
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
C__inference_dense_12_layer_call_and_return_conditional_losses_14166

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input_21
serving_default_input_2:0??????????R<
dense_130
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?#
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?
_tf_keras_network?{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}, "name": "token_and_position_embedding_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_1", "inbound_nodes": [[["token_and_position_embedding_1", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block_3", "inbound_nodes": [[["average_pooling1d_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["transformer_block_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_10", "inbound_nodes": [[["dense_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["dropout_10", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_11", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["dropout_11", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_13", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 10500]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.0010000000474974513, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?
	token_emb
pos_emb
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TokenAndPositionEmbedding", "name": "token_and_position_embedding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "AveragePooling1D", "name": "average_pooling1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
att
ffn

layernorm1

layernorm2
dropout1
 dropout2
!regularization_losses
"	variables
#trainable_variables
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TransformerBlock", "name": "transformer_block_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
%regularization_losses
&	variables
'trainable_variables
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1120]}}
?
/regularization_losses
0	variables
1trainable_variables
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?

3kernel
4bias
5regularization_losses
6	variables
7trainable_variables
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
9regularization_losses
:	variables
;trainable_variables
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?

=kernel
>bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
	Cdecay
Dlearning_rate
Emomentum
Fiter)momentum?*momentum?3momentum?4momentum?=momentum?>momentum?Gmomentum?Hmomentum?Imomentum?Jmomentum?Kmomentum?Lmomentum?Mmomentum?Nmomentum?Omomentum?Pmomentum?Qmomentum?Rmomentum?Smomentum?Tmomentum?Umomentum?Vmomentum?Wmomentum?Xmomentum?"
	optimizer
 "
trackable_list_wrapper
?
G0
H1
I2
J3
K4
L5
M6
N7
O8
P9
Q10
R11
S12
T13
U14
V15
W16
X17
)18
*19
320
421
=22
>23"
trackable_list_wrapper
?
G0
H1
I2
J3
K4
L5
M6
N7
O8
P9
Q10
R11
S12
T13
U14
V15
W16
X17
)18
*19
320
421
=22
>23"
trackable_list_wrapper
?
Ynon_trainable_variables
Zmetrics
regularization_losses
	variables

[layers
\layer_regularization_losses
]layer_metrics
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
G
embeddings
^regularization_losses
_	variables
`trainable_variables
a	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500]}}
?
H
embeddings
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 10500, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
?
fnon_trainable_variables
gmetrics
regularization_losses
	variables

hlayers
ilayer_regularization_losses
jlayer_metrics
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
knon_trainable_variables
lmetrics
regularization_losses
	variables

mlayers
nlayer_regularization_losses
olayer_metrics
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
p_query_dense
q
_key_dense
r_value_dense
s_softmax
t_dropout_layer
u_output_dense
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MultiHeadAttention", "name": "multi_head_attention_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multi_head_attention_3", "trainable": true, "dtype": "float32", "num_heads": 1, "key_dim": 32, "value_dim": 32, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}
?
zlayer_with_weights-0
zlayer-0
{layer_with_weights-1
{layer-1
|regularization_losses
}	variables
~trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_9_input"}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_9_input"}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
?
	?axis
	Ugamma
Vbeta
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LayerNormalization", "name": "layer_normalization_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_6", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
?
	?axis
	Wgamma
Xbeta
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LayerNormalization", "name": "layer_normalization_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
 "
trackable_list_wrapper
?
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15"
trackable_list_wrapper
?
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
!regularization_losses
"	variables
?layers
 ?layer_regularization_losses
?layer_metrics
#trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
%regularization_losses
&	variables
?layers
 ?layer_regularization_losses
?layer_metrics
'trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?@2dense_11/kernel
:@2dense_11/bias
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
+regularization_losses
,	variables
?layers
 ?layer_regularization_losses
?layer_metrics
-trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
/regularization_losses
0	variables
?layers
 ?layer_regularization_losses
?layer_metrics
1trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_12/kernel
:@2dense_12/bias
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
5regularization_losses
6	variables
?layers
 ?layer_regularization_losses
?layer_metrics
7trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
9regularization_losses
:	variables
?layers
 ?layer_regularization_losses
?layer_metrics
;trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_13/kernel
:2dense_13/bias
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?regularization_losses
@	variables
?layers
 ?layer_regularization_losses
?layer_metrics
Atrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
G:E 25token_and_position_embedding_1/embedding_2/embeddings
H:F	?R 25token_and_position_embedding_1/embedding_3/embeddings
M:K  27transformer_block_3/multi_head_attention_3/query/kernel
G:E 25transformer_block_3/multi_head_attention_3/query/bias
K:I  25transformer_block_3/multi_head_attention_3/key/kernel
E:C 23transformer_block_3/multi_head_attention_3/key/bias
M:K  27transformer_block_3/multi_head_attention_3/value/kernel
G:E 25transformer_block_3/multi_head_attention_3/value/bias
X:V  2Btransformer_block_3/multi_head_attention_3/attention_output/kernel
N:L 2@transformer_block_3/multi_head_attention_3/attention_output/bias
 : @2dense_9/kernel
:@2dense_9/bias
!:@ 2dense_10/kernel
: 2dense_10/bias
=:; 2/transformer_block_3/layer_normalization_6/gamma
<:: 2.transformer_block_3/layer_normalization_6/beta
=:; 2/transformer_block_3/layer_normalization_7/gamma
<:: 2.transformer_block_3/layer_normalization_7/beta
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
G0"
trackable_list_wrapper
'
G0"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
^regularization_losses
_	variables
?layers
 ?layer_regularization_losses
?layer_metrics
`trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
H0"
trackable_list_wrapper
'
H0"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
bregularization_losses
c	variables
?layers
 ?layer_regularization_losses
?layer_metrics
dtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
?partial_output_shape
?full_output_shape

Ikernel
Jbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "EinsumDense", "name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
?
?partial_output_shape
?full_output_shape

Kkernel
Lbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "EinsumDense", "name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
?
?partial_output_shape
?full_output_shape

Mkernel
Nbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "EinsumDense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
?
?partial_output_shape
?full_output_shape

Okernel
Pbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "EinsumDense", "name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 32], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 1, 32]}}
 "
trackable_list_wrapper
X
I0
J1
K2
L3
M4
N5
O6
P7"
trackable_list_wrapper
X
I0
J1
K2
L3
M4
N5
O6
P7"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
vregularization_losses
w	variables
?layers
 ?layer_regularization_losses
?layer_metrics
xtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Qkernel
Rbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
?

Skernel
Tbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 64]}}
 "
trackable_list_wrapper
<
Q0
R1
S2
T3"
trackable_list_wrapper
<
Q0
R1
S2
T3"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
|regularization_losses
}	variables
?layers
 ?layer_regularization_losses
?layer_metrics
~trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
 5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
p0
q1
r2
s3
t4
u5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?regularization_losses
?	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
-:+	?@2SGD/dense_11/kernel/momentum
&:$@2SGD/dense_11/bias/momentum
,:*@@2SGD/dense_12/kernel/momentum
&:$@2SGD/dense_12/bias/momentum
,:*@2SGD/dense_13/kernel/momentum
&:$2SGD/dense_13/bias/momentum
R:P 2BSGD/token_and_position_embedding_1/embedding_2/embeddings/momentum
S:Q	?R 2BSGD/token_and_position_embedding_1/embedding_3/embeddings/momentum
X:V  2DSGD/transformer_block_3/multi_head_attention_3/query/kernel/momentum
R:P 2BSGD/transformer_block_3/multi_head_attention_3/query/bias/momentum
V:T  2BSGD/transformer_block_3/multi_head_attention_3/key/kernel/momentum
P:N 2@SGD/transformer_block_3/multi_head_attention_3/key/bias/momentum
X:V  2DSGD/transformer_block_3/multi_head_attention_3/value/kernel/momentum
R:P 2BSGD/transformer_block_3/multi_head_attention_3/value/bias/momentum
c:a  2OSGD/transformer_block_3/multi_head_attention_3/attention_output/kernel/momentum
Y:W 2MSGD/transformer_block_3/multi_head_attention_3/attention_output/bias/momentum
+:) @2SGD/dense_9/kernel/momentum
%:#@2SGD/dense_9/bias/momentum
,:*@ 2SGD/dense_10/kernel/momentum
&:$ 2SGD/dense_10/bias/momentum
H:F 2<SGD/transformer_block_3/layer_normalization_6/gamma/momentum
G:E 2;SGD/transformer_block_3/layer_normalization_6/beta/momentum
H:F 2<SGD/transformer_block_3/layer_normalization_7/gamma/momentum
G:E 2;SGD/transformer_block_3/layer_normalization_7/beta/momentum
?2?
'__inference_model_1_layer_call_fn_13715
'__inference_model_1_layer_call_fn_13049
'__inference_model_1_layer_call_fn_13163
'__inference_model_1_layer_call_fn_13662?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_model_1_layer_call_and_return_conditional_losses_13434
B__inference_model_1_layer_call_and_return_conditional_losses_12934
B__inference_model_1_layer_call_and_return_conditional_losses_13609
B__inference_model_1_layer_call_and_return_conditional_losses_12873?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_12107?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
input_2??????????R
?2?
>__inference_token_and_position_embedding_1_layer_call_fn_13748?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Y__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_13739?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_average_pooling1d_1_layer_call_fn_12122?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
N__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_12116?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
3__inference_transformer_block_3_layer_call_fn_14097
3__inference_transformer_block_3_layer_call_fn_14060?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_13896
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_14023?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_flatten_1_layer_call_fn_14108?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_1_layer_call_and_return_conditional_losses_14103?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_11_layer_call_fn_14128?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_11_layer_call_and_return_conditional_losses_14119?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_10_layer_call_fn_14155
*__inference_dropout_10_layer_call_fn_14150?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_10_layer_call_and_return_conditional_losses_14145
E__inference_dropout_10_layer_call_and_return_conditional_losses_14140?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_12_layer_call_fn_14175?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_12_layer_call_and_return_conditional_losses_14166?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_11_layer_call_fn_14202
*__inference_dropout_11_layer_call_fn_14197?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_11_layer_call_and_return_conditional_losses_14192
E__inference_dropout_11_layer_call_and_return_conditional_losses_14187?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_13_layer_call_fn_14221?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_13_layer_call_and_return_conditional_losses_14212?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_13224input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpece
args]?Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults?

 

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpece
args]?Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults?

 

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_sequential_3_layer_call_fn_12262
,__inference_sequential_3_layer_call_fn_14348
,__inference_sequential_3_layer_call_fn_12289
,__inference_sequential_3_layer_call_fn_14361?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_3_layer_call_and_return_conditional_losses_14278
G__inference_sequential_3_layer_call_and_return_conditional_losses_12220
G__inference_sequential_3_layer_call_and_return_conditional_losses_14335
G__inference_sequential_3_layer_call_and_return_conditional_losses_12234?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_9_layer_call_fn_14401?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_9_layer_call_and_return_conditional_losses_14392?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_10_layer_call_fn_14440?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_10_layer_call_and_return_conditional_losses_14431?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_12107?HGIJKLMNOPUVQRSTWX)*34=>1?.
'?$
"?
input_2??????????R
? "3?0
.
dense_13"?
dense_13??????????
N__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_12116?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
3__inference_average_pooling1d_1_layer_call_fn_12122wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
C__inference_dense_10_layer_call_and_return_conditional_losses_14431dST3?0
)?&
$?!
inputs?????????#@
? ")?&
?
0?????????# 
? ?
(__inference_dense_10_layer_call_fn_14440WST3?0
)?&
$?!
inputs?????????#@
? "??????????# ?
C__inference_dense_11_layer_call_and_return_conditional_losses_14119])*0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? |
(__inference_dense_11_layer_call_fn_14128P)*0?-
&?#
!?
inputs??????????
? "??????????@?
C__inference_dense_12_layer_call_and_return_conditional_losses_14166\34/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? {
(__inference_dense_12_layer_call_fn_14175O34/?,
%?"
 ?
inputs?????????@
? "??????????@?
C__inference_dense_13_layer_call_and_return_conditional_losses_14212\=>/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? {
(__inference_dense_13_layer_call_fn_14221O=>/?,
%?"
 ?
inputs?????????@
? "???????????
B__inference_dense_9_layer_call_and_return_conditional_losses_14392dQR3?0
)?&
$?!
inputs?????????# 
? ")?&
?
0?????????#@
? ?
'__inference_dense_9_layer_call_fn_14401WQR3?0
)?&
$?!
inputs?????????# 
? "??????????#@?
E__inference_dropout_10_layer_call_and_return_conditional_losses_14140\3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? ?
E__inference_dropout_10_layer_call_and_return_conditional_losses_14145\3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? }
*__inference_dropout_10_layer_call_fn_14150O3?0
)?&
 ?
inputs?????????@
p
? "??????????@}
*__inference_dropout_10_layer_call_fn_14155O3?0
)?&
 ?
inputs?????????@
p 
? "??????????@?
E__inference_dropout_11_layer_call_and_return_conditional_losses_14187\3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? ?
E__inference_dropout_11_layer_call_and_return_conditional_losses_14192\3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? }
*__inference_dropout_11_layer_call_fn_14197O3?0
)?&
 ?
inputs?????????@
p
? "??????????@}
*__inference_dropout_11_layer_call_fn_14202O3?0
)?&
 ?
inputs?????????@
p 
? "??????????@?
D__inference_flatten_1_layer_call_and_return_conditional_losses_14103]3?0
)?&
$?!
inputs?????????# 
? "&?#
?
0??????????
? }
)__inference_flatten_1_layer_call_fn_14108P3?0
)?&
$?!
inputs?????????# 
? "????????????
B__inference_model_1_layer_call_and_return_conditional_losses_12873|HGIJKLMNOPUVQRSTWX)*34=>9?6
/?,
"?
input_2??????????R
p

 
? "%?"
?
0?????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_12934|HGIJKLMNOPUVQRSTWX)*34=>9?6
/?,
"?
input_2??????????R
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_13434{HGIJKLMNOPUVQRSTWX)*34=>8?5
.?+
!?
inputs??????????R
p

 
? "%?"
?
0?????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_13609{HGIJKLMNOPUVQRSTWX)*34=>8?5
.?+
!?
inputs??????????R
p 

 
? "%?"
?
0?????????
? ?
'__inference_model_1_layer_call_fn_13049oHGIJKLMNOPUVQRSTWX)*34=>9?6
/?,
"?
input_2??????????R
p

 
? "???????????
'__inference_model_1_layer_call_fn_13163oHGIJKLMNOPUVQRSTWX)*34=>9?6
/?,
"?
input_2??????????R
p 

 
? "???????????
'__inference_model_1_layer_call_fn_13662nHGIJKLMNOPUVQRSTWX)*34=>8?5
.?+
!?
inputs??????????R
p

 
? "???????????
'__inference_model_1_layer_call_fn_13715nHGIJKLMNOPUVQRSTWX)*34=>8?5
.?+
!?
inputs??????????R
p 

 
? "???????????
G__inference_sequential_3_layer_call_and_return_conditional_losses_12220uQRSTB??
8?5
+?(
dense_9_input?????????# 
p

 
? ")?&
?
0?????????# 
? ?
G__inference_sequential_3_layer_call_and_return_conditional_losses_12234uQRSTB??
8?5
+?(
dense_9_input?????????# 
p 

 
? ")?&
?
0?????????# 
? ?
G__inference_sequential_3_layer_call_and_return_conditional_losses_14278nQRST;?8
1?.
$?!
inputs?????????# 
p

 
? ")?&
?
0?????????# 
? ?
G__inference_sequential_3_layer_call_and_return_conditional_losses_14335nQRST;?8
1?.
$?!
inputs?????????# 
p 

 
? ")?&
?
0?????????# 
? ?
,__inference_sequential_3_layer_call_fn_12262hQRSTB??
8?5
+?(
dense_9_input?????????# 
p

 
? "??????????# ?
,__inference_sequential_3_layer_call_fn_12289hQRSTB??
8?5
+?(
dense_9_input?????????# 
p 

 
? "??????????# ?
,__inference_sequential_3_layer_call_fn_14348aQRST;?8
1?.
$?!
inputs?????????# 
p

 
? "??????????# ?
,__inference_sequential_3_layer_call_fn_14361aQRST;?8
1?.
$?!
inputs?????????# 
p 

 
? "??????????# ?
#__inference_signature_wrapper_13224?HGIJKLMNOPUVQRSTWX)*34=><?9
? 
2?/
-
input_2"?
input_2??????????R"3?0
.
dense_13"?
dense_13??????????
Y__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_13739]HG+?(
!?
?
x??????????R
? "*?'
 ?
0??????????R 
? ?
>__inference_token_and_position_embedding_1_layer_call_fn_13748PHG+?(
!?
?
x??????????R
? "???????????R ?
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_13896vIJKLMNOPUVQRSTWX7?4
-?*
$?!
inputs?????????# 
p
? ")?&
?
0?????????# 
? ?
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_14023vIJKLMNOPUVQRSTWX7?4
-?*
$?!
inputs?????????# 
p 
? ")?&
?
0?????????# 
? ?
3__inference_transformer_block_3_layer_call_fn_14060iIJKLMNOPUVQRSTWX7?4
-?*
$?!
inputs?????????# 
p
? "??????????# ?
3__inference_transformer_block_3_layer_call_fn_14097iIJKLMNOPUVQRSTWX7?4
-?*
$?!
inputs?????????# 
p 
? "??????????# 