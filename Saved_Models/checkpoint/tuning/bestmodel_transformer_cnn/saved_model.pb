Ïú1
¿!!
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
¼
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

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

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
­
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

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
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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

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
dtypetype
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
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
list(type)(0
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

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
¾
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
executor_typestring 
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
ö
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.1-0-g85c8b2a817f8°¬+

conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv1d_2/kernel
x
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*#
_output_shapes
: *
dtype0
s
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_2/bias
l
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes	
:*
dtype0

conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 * 
shared_nameconv1d_3/kernel
x
#conv1d_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_3/kernel*#
_output_shapes
:	 *
dtype0
r
conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_3/bias
k
!conv1d_3/bias/Read/ReadVariableOpReadVariableOpconv1d_3/bias*
_output_shapes
: *
dtype0

batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_2/gamma

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
: *
dtype0

batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
: *
dtype0

!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
: *
dtype0
¢
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
: *
dtype0

batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_3/gamma

/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
: *
dtype0

batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_3/beta

.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
: *
dtype0

!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_3/moving_mean

5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
: *
dtype0
¢
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_3/moving_variance

9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0
{
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	è@* 
shared_namedense_12/kernel
t
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes
:	è@*
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
:@@* 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:@@*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:@*
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

:@@*
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:@*
dtype0
z
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_15/kernel
s
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes

:@*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
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
Æ
5token_and_position_embedding_1/embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75token_and_position_embedding_1/embedding_2/embeddings
¿
Itoken_and_position_embedding_1/embedding_2/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_1/embedding_2/embeddings*
_output_shapes

: *
dtype0
Ç
5token_and_position_embedding_1/embedding_3/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	R *F
shared_name75token_and_position_embedding_1/embedding_3/embeddings
À
Itoken_and_position_embedding_1/embedding_3/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_1/embedding_3/embeddings*
_output_shapes
:	R *
dtype0
Î
7transformer_block_3/multi_head_attention_3/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_3/multi_head_attention_3/query/kernel
Ç
Ktransformer_block_3/multi_head_attention_3/query/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_3/multi_head_attention_3/query/kernel*"
_output_shapes
:  *
dtype0
Æ
5transformer_block_3/multi_head_attention_3/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_3/multi_head_attention_3/query/bias
¿
Itransformer_block_3/multi_head_attention_3/query/bias/Read/ReadVariableOpReadVariableOp5transformer_block_3/multi_head_attention_3/query/bias*
_output_shapes

: *
dtype0
Ê
5transformer_block_3/multi_head_attention_3/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *F
shared_name75transformer_block_3/multi_head_attention_3/key/kernel
Ã
Itransformer_block_3/multi_head_attention_3/key/kernel/Read/ReadVariableOpReadVariableOp5transformer_block_3/multi_head_attention_3/key/kernel*"
_output_shapes
:  *
dtype0
Â
3transformer_block_3/multi_head_attention_3/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *D
shared_name53transformer_block_3/multi_head_attention_3/key/bias
»
Gtransformer_block_3/multi_head_attention_3/key/bias/Read/ReadVariableOpReadVariableOp3transformer_block_3/multi_head_attention_3/key/bias*
_output_shapes

: *
dtype0
Î
7transformer_block_3/multi_head_attention_3/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_3/multi_head_attention_3/value/kernel
Ç
Ktransformer_block_3/multi_head_attention_3/value/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_3/multi_head_attention_3/value/kernel*"
_output_shapes
:  *
dtype0
Æ
5transformer_block_3/multi_head_attention_3/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_3/multi_head_attention_3/value/bias
¿
Itransformer_block_3/multi_head_attention_3/value/bias/Read/ReadVariableOpReadVariableOp5transformer_block_3/multi_head_attention_3/value/bias*
_output_shapes

: *
dtype0
ä
Btransformer_block_3/multi_head_attention_3/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBtransformer_block_3/multi_head_attention_3/attention_output/kernel
Ý
Vtransformer_block_3/multi_head_attention_3/attention_output/kernel/Read/ReadVariableOpReadVariableOpBtransformer_block_3/multi_head_attention_3/attention_output/kernel*"
_output_shapes
:  *
dtype0
Ø
@transformer_block_3/multi_head_attention_3/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@transformer_block_3/multi_head_attention_3/attention_output/bias
Ñ
Ttransformer_block_3/multi_head_attention_3/attention_output/bias/Read/ReadVariableOpReadVariableOp@transformer_block_3/multi_head_attention_3/attention_output/bias*
_output_shapes
: *
dtype0
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @* 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

: @*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:@*
dtype0
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:@ *
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
: *
dtype0
¶
/transformer_block_3/layer_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_3/layer_normalization_6/gamma
¯
Ctransformer_block_3/layer_normalization_6/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_3/layer_normalization_6/gamma*
_output_shapes
: *
dtype0
´
.transformer_block_3/layer_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.transformer_block_3/layer_normalization_6/beta
­
Btransformer_block_3/layer_normalization_6/beta/Read/ReadVariableOpReadVariableOp.transformer_block_3/layer_normalization_6/beta*
_output_shapes
: *
dtype0
¶
/transformer_block_3/layer_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_3/layer_normalization_7/gamma
¯
Ctransformer_block_3/layer_normalization_7/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_3/layer_normalization_7/gamma*
_output_shapes
: *
dtype0
´
.transformer_block_3/layer_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.transformer_block_3/layer_normalization_7/beta
­
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

SGD/conv1d_2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameSGD/conv1d_2/kernel/momentum

0SGD/conv1d_2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_2/kernel/momentum*#
_output_shapes
: *
dtype0

SGD/conv1d_2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/conv1d_2/bias/momentum

.SGD/conv1d_2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_2/bias/momentum*
_output_shapes	
:*
dtype0

SGD/conv1d_3/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *-
shared_nameSGD/conv1d_3/kernel/momentum

0SGD/conv1d_3/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_3/kernel/momentum*#
_output_shapes
:	 *
dtype0

SGD/conv1d_3/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/conv1d_3/bias/momentum

.SGD/conv1d_3/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_3/bias/momentum*
_output_shapes
: *
dtype0
¨
(SGD/batch_normalization_2/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(SGD/batch_normalization_2/gamma/momentum
¡
<SGD/batch_normalization_2/gamma/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_2/gamma/momentum*
_output_shapes
: *
dtype0
¦
'SGD/batch_normalization_2/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'SGD/batch_normalization_2/beta/momentum

;SGD/batch_normalization_2/beta/momentum/Read/ReadVariableOpReadVariableOp'SGD/batch_normalization_2/beta/momentum*
_output_shapes
: *
dtype0
¨
(SGD/batch_normalization_3/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(SGD/batch_normalization_3/gamma/momentum
¡
<SGD/batch_normalization_3/gamma/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_3/gamma/momentum*
_output_shapes
: *
dtype0
¦
'SGD/batch_normalization_3/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'SGD/batch_normalization_3/beta/momentum

;SGD/batch_normalization_3/beta/momentum/Read/ReadVariableOpReadVariableOp'SGD/batch_normalization_3/beta/momentum*
_output_shapes
: *
dtype0

SGD/dense_12/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	è@*-
shared_nameSGD/dense_12/kernel/momentum

0SGD/dense_12/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_12/kernel/momentum*
_output_shapes
:	è@*
dtype0

SGD/dense_12/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_12/bias/momentum

.SGD/dense_12/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_12/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_13/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*-
shared_nameSGD/dense_13/kernel/momentum

0SGD/dense_13/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_13/kernel/momentum*
_output_shapes

:@@*
dtype0

SGD/dense_13/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_13/bias/momentum

.SGD/dense_13/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_13/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_14/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*-
shared_nameSGD/dense_14/kernel/momentum

0SGD/dense_14/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_14/kernel/momentum*
_output_shapes

:@@*
dtype0

SGD/dense_14/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_14/bias/momentum

.SGD/dense_14/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_14/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_15/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*-
shared_nameSGD/dense_15/kernel/momentum

0SGD/dense_15/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_15/kernel/momentum*
_output_shapes

:@*
dtype0

SGD/dense_15/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/dense_15/bias/momentum

.SGD/dense_15/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_15/bias/momentum*
_output_shapes
:*
dtype0
à
BSGD/token_and_position_embedding_1/embedding_2/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/token_and_position_embedding_1/embedding_2/embeddings/momentum
Ù
VSGD/token_and_position_embedding_1/embedding_2/embeddings/momentum/Read/ReadVariableOpReadVariableOpBSGD/token_and_position_embedding_1/embedding_2/embeddings/momentum*
_output_shapes

: *
dtype0
á
BSGD/token_and_position_embedding_1/embedding_3/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	R *S
shared_nameDBSGD/token_and_position_embedding_1/embedding_3/embeddings/momentum
Ú
VSGD/token_and_position_embedding_1/embedding_3/embeddings/momentum/Read/ReadVariableOpReadVariableOpBSGD/token_and_position_embedding_1/embedding_3/embeddings/momentum*
_output_shapes
:	R *
dtype0
è
DSGD/transformer_block_3/multi_head_attention_3/query/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_3/multi_head_attention_3/query/kernel/momentum
á
XSGD/transformer_block_3/multi_head_attention_3/query/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_3/multi_head_attention_3/query/kernel/momentum*"
_output_shapes
:  *
dtype0
à
BSGD/transformer_block_3/multi_head_attention_3/query/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_3/multi_head_attention_3/query/bias/momentum
Ù
VSGD/transformer_block_3/multi_head_attention_3/query/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_3/multi_head_attention_3/query/bias/momentum*
_output_shapes

: *
dtype0
ä
BSGD/transformer_block_3/multi_head_attention_3/key/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBSGD/transformer_block_3/multi_head_attention_3/key/kernel/momentum
Ý
VSGD/transformer_block_3/multi_head_attention_3/key/kernel/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_3/multi_head_attention_3/key/kernel/momentum*"
_output_shapes
:  *
dtype0
Ü
@SGD/transformer_block_3/multi_head_attention_3/key/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *Q
shared_nameB@SGD/transformer_block_3/multi_head_attention_3/key/bias/momentum
Õ
TSGD/transformer_block_3/multi_head_attention_3/key/bias/momentum/Read/ReadVariableOpReadVariableOp@SGD/transformer_block_3/multi_head_attention_3/key/bias/momentum*
_output_shapes

: *
dtype0
è
DSGD/transformer_block_3/multi_head_attention_3/value/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_3/multi_head_attention_3/value/kernel/momentum
á
XSGD/transformer_block_3/multi_head_attention_3/value/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_3/multi_head_attention_3/value/kernel/momentum*"
_output_shapes
:  *
dtype0
à
BSGD/transformer_block_3/multi_head_attention_3/value/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_3/multi_head_attention_3/value/bias/momentum
Ù
VSGD/transformer_block_3/multi_head_attention_3/value/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_3/multi_head_attention_3/value/bias/momentum*
_output_shapes

: *
dtype0
þ
OSGD/transformer_block_3/multi_head_attention_3/attention_output/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *`
shared_nameQOSGD/transformer_block_3/multi_head_attention_3/attention_output/kernel/momentum
÷
cSGD/transformer_block_3/multi_head_attention_3/attention_output/kernel/momentum/Read/ReadVariableOpReadVariableOpOSGD/transformer_block_3/multi_head_attention_3/attention_output/kernel/momentum*"
_output_shapes
:  *
dtype0
ò
MSGD/transformer_block_3/multi_head_attention_3/attention_output/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *^
shared_nameOMSGD/transformer_block_3/multi_head_attention_3/attention_output/bias/momentum
ë
aSGD/transformer_block_3/multi_head_attention_3/attention_output/bias/momentum/Read/ReadVariableOpReadVariableOpMSGD/transformer_block_3/multi_head_attention_3/attention_output/bias/momentum*
_output_shapes
: *
dtype0

SGD/dense_10/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*-
shared_nameSGD/dense_10/kernel/momentum

0SGD/dense_10/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_10/kernel/momentum*
_output_shapes

: @*
dtype0

SGD/dense_10/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_10/bias/momentum

.SGD/dense_10/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_10/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_11/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *-
shared_nameSGD/dense_11/kernel/momentum

0SGD/dense_11/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_11/kernel/momentum*
_output_shapes

:@ *
dtype0

SGD/dense_11/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/dense_11/bias/momentum

.SGD/dense_11/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_11/bias/momentum*
_output_shapes
: *
dtype0
Ð
<SGD/transformer_block_3/layer_normalization_6/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_3/layer_normalization_6/gamma/momentum
É
PSGD/transformer_block_3/layer_normalization_6/gamma/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_3/layer_normalization_6/gamma/momentum*
_output_shapes
: *
dtype0
Î
;SGD/transformer_block_3/layer_normalization_6/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;SGD/transformer_block_3/layer_normalization_6/beta/momentum
Ç
OSGD/transformer_block_3/layer_normalization_6/beta/momentum/Read/ReadVariableOpReadVariableOp;SGD/transformer_block_3/layer_normalization_6/beta/momentum*
_output_shapes
: *
dtype0
Ð
<SGD/transformer_block_3/layer_normalization_7/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_3/layer_normalization_7/gamma/momentum
É
PSGD/transformer_block_3/layer_normalization_7/gamma/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_3/layer_normalization_7/gamma/momentum*
_output_shapes
: *
dtype0
Î
;SGD/transformer_block_3/layer_normalization_7/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;SGD/transformer_block_3/layer_normalization_7/beta/momentum
Ç
OSGD/transformer_block_3/layer_normalization_7/beta/momentum/Read/ReadVariableOpReadVariableOp;SGD/transformer_block_3/layer_normalization_7/beta/momentum*
_output_shapes
: *
dtype0

NoOpNoOp
½
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*È¼
value½¼B¹¼ B±¼

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
layer-19
layer_with_weights-9
layer-20
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
n
	token_emb
pos_emb
	variables
regularization_losses
 trainable_variables
!	keras_api
h

"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
R
(	variables
)regularization_losses
*trainable_variables
+	keras_api
h

,kernel
-bias
.	variables
/regularization_losses
0trainable_variables
1	keras_api
R
2	variables
3regularization_losses
4trainable_variables
5	keras_api
R
6	variables
7regularization_losses
8trainable_variables
9	keras_api

:axis
	;gamma
<beta
=moving_mean
>moving_variance
?	variables
@regularization_losses
Atrainable_variables
B	keras_api

Caxis
	Dgamma
Ebeta
Fmoving_mean
Gmoving_variance
H	variables
Iregularization_losses
Jtrainable_variables
K	keras_api
R
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
 
Patt
Qffn
R
layernorm1
S
layernorm2
Tdropout1
Udropout2
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
R
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
 
R
^	variables
_regularization_losses
`trainable_variables
a	keras_api
h

bkernel
cbias
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
R
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
h

lkernel
mbias
n	variables
oregularization_losses
ptrainable_variables
q	keras_api
R
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
h

vkernel
wbias
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
R
|	variables
}regularization_losses
~trainable_variables
	keras_api
n
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api


decay
learning_rate
momentum
	iter"momentum§#momentum¨,momentum©-momentumª;momentum«<momentum¬Dmomentum­Emomentum®bmomentum¯cmomentum°lmomentum±mmomentum²vmomentum³wmomentum´momentumµmomentum¶momentum·momentum¸momentum¹momentumºmomentum»momentum¼momentum½momentum¾momentum¿momentumÀmomentumÁmomentumÂmomentumÃmomentumÄmomentumÅmomentumÆmomentumÇmomentumÈ
º
0
1
"2
#3
,4
-5
;6
<7
=8
>9
D10
E11
F12
G13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
b30
c31
l32
m33
v34
w35
36
37
 

0
1
"2
#3
,4
-5
;6
<7
D8
E9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
b26
c27
l28
m29
v30
w31
32
33
²
layers
	variables
regularization_losses
non_trainable_variables
metrics
trainable_variables
 layer_regularization_losses
 layer_metrics
 
g

embeddings
¡	variables
¢regularization_losses
£trainable_variables
¤	keras_api
g

embeddings
¥	variables
¦regularization_losses
§trainable_variables
¨	keras_api

0
1
 

0
1
²
©layers
	variables
regularization_losses
ªnon_trainable_variables
«metrics
 trainable_variables
 ¬layer_regularization_losses
­layer_metrics
[Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
²
®layers
$	variables
%regularization_losses
¯non_trainable_variables
°metrics
&trainable_variables
 ±layer_regularization_losses
²layer_metrics
 
 
 
²
³layers
(	variables
)regularization_losses
´non_trainable_variables
µmetrics
*trainable_variables
 ¶layer_regularization_losses
·layer_metrics
[Y
VARIABLE_VALUEconv1d_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1
 

,0
-1
²
¸layers
.	variables
/regularization_losses
¹non_trainable_variables
ºmetrics
0trainable_variables
 »layer_regularization_losses
¼layer_metrics
 
 
 
²
½layers
2	variables
3regularization_losses
¾non_trainable_variables
¿metrics
4trainable_variables
 Àlayer_regularization_losses
Álayer_metrics
 
 
 
²
Âlayers
6	variables
7regularization_losses
Ãnon_trainable_variables
Ämetrics
8trainable_variables
 Ålayer_regularization_losses
Ælayer_metrics
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

;0
<1
=2
>3
 

;0
<1
²
Çlayers
?	variables
@regularization_losses
Ènon_trainable_variables
Émetrics
Atrainable_variables
 Êlayer_regularization_losses
Ëlayer_metrics
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

D0
E1
F2
G3
 

D0
E1
²
Ìlayers
H	variables
Iregularization_losses
Ínon_trainable_variables
Îmetrics
Jtrainable_variables
 Ïlayer_regularization_losses
Ðlayer_metrics
 
 
 
²
Ñlayers
L	variables
Mregularization_losses
Ònon_trainable_variables
Ómetrics
Ntrainable_variables
 Ôlayer_regularization_losses
Õlayer_metrics
Å
Ö_query_dense
×
_key_dense
Ø_value_dense
Ù_softmax
Ú_dropout_layer
Û_output_dense
Ü	variables
Ýregularization_losses
Þtrainable_variables
ß	keras_api
¨
àlayer_with_weights-0
àlayer-0
álayer_with_weights-1
álayer-1
â	variables
ãregularization_losses
ätrainable_variables
å	keras_api
x
	æaxis

gamma
	beta
ç	variables
èregularization_losses
étrainable_variables
ê	keras_api
x
	ëaxis

gamma
	beta
ì	variables
íregularization_losses
îtrainable_variables
ï	keras_api
V
ð	variables
ñregularization_losses
òtrainable_variables
ó	keras_api
V
ô	variables
õregularization_losses
ötrainable_variables
÷	keras_api

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
 

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
²
ølayers
V	variables
Wregularization_losses
ùnon_trainable_variables
úmetrics
Xtrainable_variables
 ûlayer_regularization_losses
ülayer_metrics
 
 
 
²
ýlayers
Z	variables
[regularization_losses
þnon_trainable_variables
ÿmetrics
\trainable_variables
 layer_regularization_losses
layer_metrics
 
 
 
²
layers
^	variables
_regularization_losses
non_trainable_variables
metrics
`trainable_variables
 layer_regularization_losses
layer_metrics
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

b0
c1
 

b0
c1
²
layers
d	variables
eregularization_losses
non_trainable_variables
metrics
ftrainable_variables
 layer_regularization_losses
layer_metrics
 
 
 
²
layers
h	variables
iregularization_losses
non_trainable_variables
metrics
jtrainable_variables
 layer_regularization_losses
layer_metrics
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

l0
m1
 

l0
m1
²
layers
n	variables
oregularization_losses
non_trainable_variables
metrics
ptrainable_variables
 layer_regularization_losses
layer_metrics
 
 
 
²
layers
r	variables
sregularization_losses
non_trainable_variables
metrics
ttrainable_variables
 layer_regularization_losses
layer_metrics
[Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_14/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

v0
w1
 

v0
w1
²
layers
x	variables
yregularization_losses
non_trainable_variables
metrics
ztrainable_variables
 layer_regularization_losses
layer_metrics
 
 
 
²
 layers
|	variables
}regularization_losses
¡non_trainable_variables
¢metrics
~trainable_variables
 £layer_regularization_losses
¤layer_metrics
[Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_15/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
µ
¥layers
	variables
regularization_losses
¦non_trainable_variables
§metrics
trainable_variables
 ¨layer_regularization_losses
©layer_metrics
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
tr
VARIABLE_VALUE7transformer_block_3/multi_head_attention_3/query/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_3/multi_head_attention_3/query/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_3/multi_head_attention_3/key/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE3transformer_block_3/multi_head_attention_3/key/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7transformer_block_3/multi_head_attention_3/value/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_3/multi_head_attention_3/value/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEBtransformer_block_3/multi_head_attention_3/attention_output/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE@transformer_block_3/multi_head_attention_3/attention_output/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_10/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_10/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_11/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_11/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_3/layer_normalization_6/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.transformer_block_3/layer_normalization_6/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_3/layer_normalization_7/gamma'variables/28/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.transformer_block_3/layer_normalization_7/beta'variables/29/.ATTRIBUTES/VARIABLE_VALUE

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
10
11
12
13
14
15
16
17
18
19
20

=0
>1
F2
G3

ª0
 
 

0
 

0
µ
«layers
¡	variables
¢regularization_losses
¬non_trainable_variables
­metrics
£trainable_variables
 ®layer_regularization_losses
¯layer_metrics

0
 

0
µ
°layers
¥	variables
¦regularization_losses
±non_trainable_variables
²metrics
§trainable_variables
 ³layer_regularization_losses
´layer_metrics

0
1
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

=0
>1
 
 
 
 

F0
G1
 
 
 
 
 
 
 
 
¡
µpartial_output_shape
¶full_output_shape
kernel
	bias
·	variables
¸regularization_losses
¹trainable_variables
º	keras_api
¡
»partial_output_shape
¼full_output_shape
kernel
	bias
½	variables
¾regularization_losses
¿trainable_variables
À	keras_api
¡
Ápartial_output_shape
Âfull_output_shape
kernel
	bias
Ã	variables
Äregularization_losses
Åtrainable_variables
Æ	keras_api
V
Ç	variables
Èregularization_losses
Étrainable_variables
Ê	keras_api
V
Ë	variables
Ìregularization_losses
Ítrainable_variables
Î	keras_api
¡
Ïpartial_output_shape
Ðfull_output_shape
kernel
	bias
Ñ	variables
Òregularization_losses
Ótrainable_variables
Ô	keras_api
@
0
1
2
3
4
5
6
7
 
@
0
1
2
3
4
5
6
7
µ
Õlayers
Ü	variables
Ýregularization_losses
Önon_trainable_variables
×metrics
Þtrainable_variables
 Ølayer_regularization_losses
Ùlayer_metrics
n
kernel
	bias
Ú	variables
Ûregularization_losses
Ütrainable_variables
Ý	keras_api
n
kernel
	bias
Þ	variables
ßregularization_losses
àtrainable_variables
á	keras_api
 
0
1
2
3
 
 
0
1
2
3
µ
âlayers
â	variables
ãregularization_losses
ãnon_trainable_variables
ämetrics
ätrainable_variables
 ålayer_regularization_losses
ælayer_metrics
 

0
1
 

0
1
µ
çlayers
ç	variables
èregularization_losses
ènon_trainable_variables
émetrics
étrainable_variables
 êlayer_regularization_losses
ëlayer_metrics
 

0
1
 

0
1
µ
ìlayers
ì	variables
íregularization_losses
ínon_trainable_variables
îmetrics
îtrainable_variables
 ïlayer_regularization_losses
ðlayer_metrics
 
 
 
µ
ñlayers
ð	variables
ñregularization_losses
ònon_trainable_variables
ómetrics
òtrainable_variables
 ôlayer_regularization_losses
õlayer_metrics
 
 
 
µ
ölayers
ô	variables
õregularization_losses
÷non_trainable_variables
ømetrics
ötrainable_variables
 ùlayer_regularization_losses
úlayer_metrics
*
P0
Q1
R2
S3
T4
U5
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

ûtotal

ücount
ý	variables
þ	keras_api
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

0
1
 

0
1
µ
ÿlayers
·	variables
¸regularization_losses
non_trainable_variables
metrics
¹trainable_variables
 layer_regularization_losses
layer_metrics
 
 

0
1
 

0
1
µ
layers
½	variables
¾regularization_losses
non_trainable_variables
metrics
¿trainable_variables
 layer_regularization_losses
layer_metrics
 
 

0
1
 

0
1
µ
layers
Ã	variables
Äregularization_losses
non_trainable_variables
metrics
Åtrainable_variables
 layer_regularization_losses
layer_metrics
 
 
 
µ
layers
Ç	variables
Èregularization_losses
non_trainable_variables
metrics
Étrainable_variables
 layer_regularization_losses
layer_metrics
 
 
 
µ
layers
Ë	variables
Ìregularization_losses
non_trainable_variables
metrics
Ítrainable_variables
 layer_regularization_losses
layer_metrics
 
 

0
1
 

0
1
µ
layers
Ñ	variables
Òregularization_losses
non_trainable_variables
metrics
Ótrainable_variables
 layer_regularization_losses
layer_metrics
0
Ö0
×1
Ø2
Ù3
Ú4
Û5
 
 
 
 

0
1
 

0
1
µ
layers
Ú	variables
Ûregularization_losses
non_trainable_variables
metrics
Ütrainable_variables
  layer_regularization_losses
¡layer_metrics

0
1
 

0
1
µ
¢layers
Þ	variables
ßregularization_losses
£non_trainable_variables
¤metrics
àtrainable_variables
 ¥layer_regularization_losses
¦layer_metrics

à0
á1
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

û0
ü1

ý	variables
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

VARIABLE_VALUESGD/conv1d_2/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv1d_2/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv1d_3/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv1d_3/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/batch_normalization_2/gamma/momentumXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'SGD/batch_normalization_2/beta/momentumWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/batch_normalization_3/gamma/momentumXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'SGD/batch_normalization_3/beta/momentumWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_12/kernel/momentumYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_12/bias/momentumWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_13/kernel/momentumYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_13/bias/momentumWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_14/kernel/momentumYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_14/bias/momentumWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_15/kernel/momentumYlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_15/bias/momentumWlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¢
VARIABLE_VALUEBSGD/token_and_position_embedding_1/embedding_2/embeddings/momentumIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¢
VARIABLE_VALUEBSGD/token_and_position_embedding_1/embedding_3/embeddings/momentumIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¥¢
VARIABLE_VALUEDSGD/transformer_block_3/multi_head_attention_3/query/kernel/momentumJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEBSGD/transformer_block_3/multi_head_attention_3/query/bias/momentumJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEBSGD/transformer_block_3/multi_head_attention_3/key/kernel/momentumJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¡
VARIABLE_VALUE@SGD/transformer_block_3/multi_head_attention_3/key/bias/momentumJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¥¢
VARIABLE_VALUEDSGD/transformer_block_3/multi_head_attention_3/value/kernel/momentumJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEBSGD/transformer_block_3/multi_head_attention_3/value/bias/momentumJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
°­
VARIABLE_VALUEOSGD/transformer_block_3/multi_head_attention_3/attention_output/kernel/momentumJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
®«
VARIABLE_VALUEMSGD/transformer_block_3/multi_head_attention_3/attention_output/bias/momentumJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUESGD/dense_10/kernel/momentumJvariables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUESGD/dense_10/bias/momentumJvariables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUESGD/dense_11/kernel/momentumJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUESGD/dense_11/bias/momentumJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<SGD/transformer_block_3/layer_normalization_6/gamma/momentumJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE;SGD/transformer_block_3/layer_normalization_6/beta/momentumJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<SGD/transformer_block_3/layer_normalization_7/gamma/momentumJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE;SGD/transformer_block_3/layer_normalization_7/beta/momentumJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_3Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿR
z
serving_default_input_4Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
£
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3serving_default_input_45token_and_position_embedding_1/embedding_3/embeddings5token_and_position_embedding_1/embedding_2/embeddingsconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/beta%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/beta7transformer_block_3/multi_head_attention_3/query/kernel5transformer_block_3/multi_head_attention_3/query/bias5transformer_block_3/multi_head_attention_3/key/kernel3transformer_block_3/multi_head_attention_3/key/bias7transformer_block_3/multi_head_attention_3/value/kernel5transformer_block_3/multi_head_attention_3/value/biasBtransformer_block_3/multi_head_attention_3/attention_output/kernel@transformer_block_3/multi_head_attention_3/attention_output/bias/transformer_block_3/layer_normalization_6/gamma.transformer_block_3/layer_normalization_6/betadense_10/kerneldense_10/biasdense_11/kerneldense_11/bias/transformer_block_3/layer_normalization_7/gamma.transformer_block_3/layer_normalization_7/betadense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/bias*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&'*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_548010
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ò%
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp#conv1d_3/kernel/Read/ReadVariableOp!conv1d_3/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpItoken_and_position_embedding_1/embedding_2/embeddings/Read/ReadVariableOpItoken_and_position_embedding_1/embedding_3/embeddings/Read/ReadVariableOpKtransformer_block_3/multi_head_attention_3/query/kernel/Read/ReadVariableOpItransformer_block_3/multi_head_attention_3/query/bias/Read/ReadVariableOpItransformer_block_3/multi_head_attention_3/key/kernel/Read/ReadVariableOpGtransformer_block_3/multi_head_attention_3/key/bias/Read/ReadVariableOpKtransformer_block_3/multi_head_attention_3/value/kernel/Read/ReadVariableOpItransformer_block_3/multi_head_attention_3/value/bias/Read/ReadVariableOpVtransformer_block_3/multi_head_attention_3/attention_output/kernel/Read/ReadVariableOpTtransformer_block_3/multi_head_attention_3/attention_output/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOpCtransformer_block_3/layer_normalization_6/gamma/Read/ReadVariableOpBtransformer_block_3/layer_normalization_6/beta/Read/ReadVariableOpCtransformer_block_3/layer_normalization_7/gamma/Read/ReadVariableOpBtransformer_block_3/layer_normalization_7/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp0SGD/conv1d_2/kernel/momentum/Read/ReadVariableOp.SGD/conv1d_2/bias/momentum/Read/ReadVariableOp0SGD/conv1d_3/kernel/momentum/Read/ReadVariableOp.SGD/conv1d_3/bias/momentum/Read/ReadVariableOp<SGD/batch_normalization_2/gamma/momentum/Read/ReadVariableOp;SGD/batch_normalization_2/beta/momentum/Read/ReadVariableOp<SGD/batch_normalization_3/gamma/momentum/Read/ReadVariableOp;SGD/batch_normalization_3/beta/momentum/Read/ReadVariableOp0SGD/dense_12/kernel/momentum/Read/ReadVariableOp.SGD/dense_12/bias/momentum/Read/ReadVariableOp0SGD/dense_13/kernel/momentum/Read/ReadVariableOp.SGD/dense_13/bias/momentum/Read/ReadVariableOp0SGD/dense_14/kernel/momentum/Read/ReadVariableOp.SGD/dense_14/bias/momentum/Read/ReadVariableOp0SGD/dense_15/kernel/momentum/Read/ReadVariableOp.SGD/dense_15/bias/momentum/Read/ReadVariableOpVSGD/token_and_position_embedding_1/embedding_2/embeddings/momentum/Read/ReadVariableOpVSGD/token_and_position_embedding_1/embedding_3/embeddings/momentum/Read/ReadVariableOpXSGD/transformer_block_3/multi_head_attention_3/query/kernel/momentum/Read/ReadVariableOpVSGD/transformer_block_3/multi_head_attention_3/query/bias/momentum/Read/ReadVariableOpVSGD/transformer_block_3/multi_head_attention_3/key/kernel/momentum/Read/ReadVariableOpTSGD/transformer_block_3/multi_head_attention_3/key/bias/momentum/Read/ReadVariableOpXSGD/transformer_block_3/multi_head_attention_3/value/kernel/momentum/Read/ReadVariableOpVSGD/transformer_block_3/multi_head_attention_3/value/bias/momentum/Read/ReadVariableOpcSGD/transformer_block_3/multi_head_attention_3/attention_output/kernel/momentum/Read/ReadVariableOpaSGD/transformer_block_3/multi_head_attention_3/attention_output/bias/momentum/Read/ReadVariableOp0SGD/dense_10/kernel/momentum/Read/ReadVariableOp.SGD/dense_10/bias/momentum/Read/ReadVariableOp0SGD/dense_11/kernel/momentum/Read/ReadVariableOp.SGD/dense_11/bias/momentum/Read/ReadVariableOpPSGD/transformer_block_3/layer_normalization_6/gamma/momentum/Read/ReadVariableOpOSGD/transformer_block_3/layer_normalization_6/beta/momentum/Read/ReadVariableOpPSGD/transformer_block_3/layer_normalization_7/gamma/momentum/Read/ReadVariableOpOSGD/transformer_block_3/layer_normalization_7/beta/momentum/Read/ReadVariableOpConst*[
TinT
R2P	*
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_550183
Õ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancebatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/biasdecaylearning_ratemomentumSGD/iter5token_and_position_embedding_1/embedding_2/embeddings5token_and_position_embedding_1/embedding_3/embeddings7transformer_block_3/multi_head_attention_3/query/kernel5transformer_block_3/multi_head_attention_3/query/bias5transformer_block_3/multi_head_attention_3/key/kernel3transformer_block_3/multi_head_attention_3/key/bias7transformer_block_3/multi_head_attention_3/value/kernel5transformer_block_3/multi_head_attention_3/value/biasBtransformer_block_3/multi_head_attention_3/attention_output/kernel@transformer_block_3/multi_head_attention_3/attention_output/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/bias/transformer_block_3/layer_normalization_6/gamma.transformer_block_3/layer_normalization_6/beta/transformer_block_3/layer_normalization_7/gamma.transformer_block_3/layer_normalization_7/betatotalcountSGD/conv1d_2/kernel/momentumSGD/conv1d_2/bias/momentumSGD/conv1d_3/kernel/momentumSGD/conv1d_3/bias/momentum(SGD/batch_normalization_2/gamma/momentum'SGD/batch_normalization_2/beta/momentum(SGD/batch_normalization_3/gamma/momentum'SGD/batch_normalization_3/beta/momentumSGD/dense_12/kernel/momentumSGD/dense_12/bias/momentumSGD/dense_13/kernel/momentumSGD/dense_13/bias/momentumSGD/dense_14/kernel/momentumSGD/dense_14/bias/momentumSGD/dense_15/kernel/momentumSGD/dense_15/bias/momentumBSGD/token_and_position_embedding_1/embedding_2/embeddings/momentumBSGD/token_and_position_embedding_1/embedding_3/embeddings/momentumDSGD/transformer_block_3/multi_head_attention_3/query/kernel/momentumBSGD/transformer_block_3/multi_head_attention_3/query/bias/momentumBSGD/transformer_block_3/multi_head_attention_3/key/kernel/momentum@SGD/transformer_block_3/multi_head_attention_3/key/bias/momentumDSGD/transformer_block_3/multi_head_attention_3/value/kernel/momentumBSGD/transformer_block_3/multi_head_attention_3/value/bias/momentumOSGD/transformer_block_3/multi_head_attention_3/attention_output/kernel/momentumMSGD/transformer_block_3/multi_head_attention_3/attention_output/bias/momentumSGD/dense_10/kernel/momentumSGD/dense_10/bias/momentumSGD/dense_11/kernel/momentumSGD/dense_11/bias/momentum<SGD/transformer_block_3/layer_normalization_6/gamma/momentum;SGD/transformer_block_3/layer_normalization_6/beta/momentum<SGD/transformer_block_3/layer_normalization_7/gamma/momentum;SGD/transformer_block_3/layer_normalization_7/beta/momentum*Z
TinS
Q2O*
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_550427Ü'

*
!__inference__wrapped_model_546042
input_3
input_4N
Jmodel_1_token_and_position_embedding_1_embedding_3_embedding_lookup_545803N
Jmodel_1_token_and_position_embedding_1_embedding_2_embedding_lookup_545809@
<model_1_conv1d_2_conv1d_expanddims_1_readvariableop_resource4
0model_1_conv1d_2_biasadd_readvariableop_resource@
<model_1_conv1d_3_conv1d_expanddims_1_readvariableop_resource4
0model_1_conv1d_3_biasadd_readvariableop_resourceC
?model_1_batch_normalization_2_batchnorm_readvariableop_resourceG
Cmodel_1_batch_normalization_2_batchnorm_mul_readvariableop_resourceE
Amodel_1_batch_normalization_2_batchnorm_readvariableop_1_resourceE
Amodel_1_batch_normalization_2_batchnorm_readvariableop_2_resourceC
?model_1_batch_normalization_3_batchnorm_readvariableop_resourceG
Cmodel_1_batch_normalization_3_batchnorm_mul_readvariableop_resourceE
Amodel_1_batch_normalization_3_batchnorm_readvariableop_1_resourceE
Amodel_1_batch_normalization_3_batchnorm_readvariableop_2_resourceb
^model_1_transformer_block_3_multi_head_attention_3_query_einsum_einsum_readvariableop_resourceX
Tmodel_1_transformer_block_3_multi_head_attention_3_query_add_readvariableop_resource`
\model_1_transformer_block_3_multi_head_attention_3_key_einsum_einsum_readvariableop_resourceV
Rmodel_1_transformer_block_3_multi_head_attention_3_key_add_readvariableop_resourceb
^model_1_transformer_block_3_multi_head_attention_3_value_einsum_einsum_readvariableop_resourceX
Tmodel_1_transformer_block_3_multi_head_attention_3_value_add_readvariableop_resourcem
imodel_1_transformer_block_3_multi_head_attention_3_attention_output_einsum_einsum_readvariableop_resourcec
_model_1_transformer_block_3_multi_head_attention_3_attention_output_add_readvariableop_resource[
Wmodel_1_transformer_block_3_layer_normalization_6_batchnorm_mul_readvariableop_resourceW
Smodel_1_transformer_block_3_layer_normalization_6_batchnorm_readvariableop_resourceW
Smodel_1_transformer_block_3_sequential_3_dense_10_tensordot_readvariableop_resourceU
Qmodel_1_transformer_block_3_sequential_3_dense_10_biasadd_readvariableop_resourceW
Smodel_1_transformer_block_3_sequential_3_dense_11_tensordot_readvariableop_resourceU
Qmodel_1_transformer_block_3_sequential_3_dense_11_biasadd_readvariableop_resource[
Wmodel_1_transformer_block_3_layer_normalization_7_batchnorm_mul_readvariableop_resourceW
Smodel_1_transformer_block_3_layer_normalization_7_batchnorm_readvariableop_resource3
/model_1_dense_12_matmul_readvariableop_resource4
0model_1_dense_12_biasadd_readvariableop_resource3
/model_1_dense_13_matmul_readvariableop_resource4
0model_1_dense_13_biasadd_readvariableop_resource3
/model_1_dense_14_matmul_readvariableop_resource4
0model_1_dense_14_biasadd_readvariableop_resource3
/model_1_dense_15_matmul_readvariableop_resource4
0model_1_dense_15_biasadd_readvariableop_resource
identity¢6model_1/batch_normalization_2/batchnorm/ReadVariableOp¢8model_1/batch_normalization_2/batchnorm/ReadVariableOp_1¢8model_1/batch_normalization_2/batchnorm/ReadVariableOp_2¢:model_1/batch_normalization_2/batchnorm/mul/ReadVariableOp¢6model_1/batch_normalization_3/batchnorm/ReadVariableOp¢8model_1/batch_normalization_3/batchnorm/ReadVariableOp_1¢8model_1/batch_normalization_3/batchnorm/ReadVariableOp_2¢:model_1/batch_normalization_3/batchnorm/mul/ReadVariableOp¢'model_1/conv1d_2/BiasAdd/ReadVariableOp¢3model_1/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp¢'model_1/conv1d_3/BiasAdd/ReadVariableOp¢3model_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp¢'model_1/dense_12/BiasAdd/ReadVariableOp¢&model_1/dense_12/MatMul/ReadVariableOp¢'model_1/dense_13/BiasAdd/ReadVariableOp¢&model_1/dense_13/MatMul/ReadVariableOp¢'model_1/dense_14/BiasAdd/ReadVariableOp¢&model_1/dense_14/MatMul/ReadVariableOp¢'model_1/dense_15/BiasAdd/ReadVariableOp¢&model_1/dense_15/MatMul/ReadVariableOp¢Cmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup¢Cmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup¢Jmodel_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp¢Nmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp¢Jmodel_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp¢Nmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp¢Vmodel_1/transformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp¢`model_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp¢Imodel_1/transformer_block_3/multi_head_attention_3/key/add/ReadVariableOp¢Smodel_1/transformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp¢Kmodel_1/transformer_block_3/multi_head_attention_3/query/add/ReadVariableOp¢Umodel_1/transformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp¢Kmodel_1/transformer_block_3/multi_head_attention_3/value/add/ReadVariableOp¢Umodel_1/transformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp¢Hmodel_1/transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp¢Jmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp¢Hmodel_1/transformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOp¢Jmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOp
,model_1/token_and_position_embedding_1/ShapeShapeinput_3*
T0*
_output_shapes
:2.
,model_1/token_and_position_embedding_1/ShapeË
:model_1/token_and_position_embedding_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2<
:model_1/token_and_position_embedding_1/strided_slice/stackÆ
<model_1/token_and_position_embedding_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<model_1/token_and_position_embedding_1/strided_slice/stack_1Æ
<model_1/token_and_position_embedding_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model_1/token_and_position_embedding_1/strided_slice/stack_2Ì
4model_1/token_and_position_embedding_1/strided_sliceStridedSlice5model_1/token_and_position_embedding_1/Shape:output:0Cmodel_1/token_and_position_embedding_1/strided_slice/stack:output:0Emodel_1/token_and_position_embedding_1/strided_slice/stack_1:output:0Emodel_1/token_and_position_embedding_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4model_1/token_and_position_embedding_1/strided_sliceª
2model_1/token_and_position_embedding_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 24
2model_1/token_and_position_embedding_1/range/startª
2model_1/token_and_position_embedding_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :24
2model_1/token_and_position_embedding_1/range/deltaÃ
,model_1/token_and_position_embedding_1/rangeRange;model_1/token_and_position_embedding_1/range/start:output:0=model_1/token_and_position_embedding_1/strided_slice:output:0;model_1/token_and_position_embedding_1/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,model_1/token_and_position_embedding_1/rangeò
Cmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookupResourceGatherJmodel_1_token_and_position_embedding_1_embedding_3_embedding_lookup_5458035model_1/token_and_position_embedding_1/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*]
_classS
QOloc:@model_1/token_and_position_embedding_1/embedding_3/embedding_lookup/545803*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02E
Cmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookupµ
Lmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup/IdentityIdentityLmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*]
_classS
QOloc:@model_1/token_and_position_embedding_1/embedding_3/embedding_lookup/545803*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2N
Lmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup/Identityµ
Nmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1IdentityUmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2P
Nmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1Å
7model_1/token_and_position_embedding_1/embedding_2/CastCastinput_3*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR29
7model_1/token_and_position_embedding_1/embedding_2/Castý
Cmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookupResourceGatherJmodel_1_token_and_position_embedding_1_embedding_2_embedding_lookup_545809;model_1/token_and_position_embedding_1/embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*]
_classS
QOloc:@model_1/token_and_position_embedding_1/embedding_2/embedding_lookup/545809*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02E
Cmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookupº
Lmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup/IdentityIdentityLmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*]
_classS
QOloc:@model_1/token_and_position_embedding_1/embedding_2/embedding_lookup/545809*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2N
Lmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup/Identityº
Nmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1IdentityUmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2P
Nmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1Ê
*model_1/token_and_position_embedding_1/addAddV2Wmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1:output:0Wmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2,
*model_1/token_and_position_embedding_1/add
&model_1/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2(
&model_1/conv1d_2/conv1d/ExpandDims/dimò
"model_1/conv1d_2/conv1d/ExpandDims
ExpandDims.model_1/token_and_position_embedding_1/add:z:0/model_1/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2$
"model_1/conv1d_2/conv1d/ExpandDimsì
3model_1/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<model_1_conv1d_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: *
dtype025
3model_1/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp
(model_1/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_1/conv1d_2/conv1d/ExpandDims_1/dimü
$model_1/conv1d_2/conv1d/ExpandDims_1
ExpandDims;model_1/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:01model_1/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: 2&
$model_1/conv1d_2/conv1d/ExpandDims_1ü
model_1/conv1d_2/conv1dConv2D+model_1/conv1d_2/conv1d/ExpandDims:output:0-model_1/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*
paddingSAME*
strides
2
model_1/conv1d_2/conv1dÇ
model_1/conv1d_2/conv1d/SqueezeSqueeze model_1/conv1d_2/conv1d:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2!
model_1/conv1d_2/conv1d/SqueezeÀ
'model_1/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv1d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'model_1/conv1d_2/BiasAdd/ReadVariableOpÒ
model_1/conv1d_2/BiasAddBiasAdd(model_1/conv1d_2/conv1d/Squeeze:output:0/model_1/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2
model_1/conv1d_2/BiasAdd
model_1/conv1d_2/ReluRelu!model_1/conv1d_2/BiasAdd:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2
model_1/conv1d_2/Relu
*model_1/average_pooling1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_1/average_pooling1d_3/ExpandDims/dimô
&model_1/average_pooling1d_3/ExpandDims
ExpandDims#model_1/conv1d_2/Relu:activations:03model_1/average_pooling1d_3/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2(
&model_1/average_pooling1d_3/ExpandDimsþ
#model_1/average_pooling1d_3/AvgPoolAvgPool/model_1/average_pooling1d_3/ExpandDims:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ*
ksize
*
paddingVALID*
strides
2%
#model_1/average_pooling1d_3/AvgPoolÒ
#model_1/average_pooling1d_3/SqueezeSqueeze,model_1/average_pooling1d_3/AvgPool:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ*
squeeze_dims
2%
#model_1/average_pooling1d_3/Squeeze
&model_1/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2(
&model_1/conv1d_3/conv1d/ExpandDims/dimñ
"model_1/conv1d_3/conv1d/ExpandDims
ExpandDims,model_1/average_pooling1d_3/Squeeze:output:0/model_1/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2$
"model_1/conv1d_3/conv1d/ExpandDimsì
3model_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<model_1_conv1d_3_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:	 *
dtype025
3model_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp
(model_1/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_1/conv1d_3/conv1d/ExpandDims_1/dimü
$model_1/conv1d_3/conv1d/ExpandDims_1
ExpandDims;model_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:01model_1/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:	 2&
$model_1/conv1d_3/conv1d/ExpandDims_1û
model_1/conv1d_3/conv1dConv2D+model_1/conv1d_3/conv1d/ExpandDims:output:0-model_1/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
paddingSAME*
strides
2
model_1/conv1d_3/conv1dÆ
model_1/conv1d_3/conv1d/SqueezeSqueeze model_1/conv1d_3/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2!
model_1/conv1d_3/conv1d/Squeeze¿
'model_1/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_1/conv1d_3/BiasAdd/ReadVariableOpÑ
model_1/conv1d_3/BiasAddBiasAdd(model_1/conv1d_3/conv1d/Squeeze:output:0/model_1/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
model_1/conv1d_3/BiasAdd
model_1/conv1d_3/ReluRelu!model_1/conv1d_3/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
model_1/conv1d_3/Relu
*model_1/average_pooling1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_1/average_pooling1d_5/ExpandDims/dimþ
&model_1/average_pooling1d_5/ExpandDims
ExpandDims.model_1/token_and_position_embedding_1/add:z:03model_1/average_pooling1d_5/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2(
&model_1/average_pooling1d_5/ExpandDimsþ
#model_1/average_pooling1d_5/AvgPoolAvgPool/model_1/average_pooling1d_5/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize	
¬*
paddingVALID*
strides	
¬2%
#model_1/average_pooling1d_5/AvgPoolÐ
#model_1/average_pooling1d_5/SqueezeSqueeze,model_1/average_pooling1d_5/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2%
#model_1/average_pooling1d_5/Squeeze
*model_1/average_pooling1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_1/average_pooling1d_4/ExpandDims/dimó
&model_1/average_pooling1d_4/ExpandDims
ExpandDims#model_1/conv1d_3/Relu:activations:03model_1/average_pooling1d_4/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2(
&model_1/average_pooling1d_4/ExpandDimsü
#model_1/average_pooling1d_4/AvgPoolAvgPool/model_1/average_pooling1d_4/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize

*
paddingVALID*
strides

2%
#model_1/average_pooling1d_4/AvgPoolÐ
#model_1/average_pooling1d_4/SqueezeSqueeze,model_1/average_pooling1d_4/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2%
#model_1/average_pooling1d_4/Squeezeì
6model_1/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp?model_1_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype028
6model_1/batch_normalization_2/batchnorm/ReadVariableOp£
-model_1/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-model_1/batch_normalization_2/batchnorm/add/y
+model_1/batch_normalization_2/batchnorm/addAddV2>model_1/batch_normalization_2/batchnorm/ReadVariableOp:value:06model_1/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2-
+model_1/batch_normalization_2/batchnorm/add½
-model_1/batch_normalization_2/batchnorm/RsqrtRsqrt/model_1/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-model_1/batch_normalization_2/batchnorm/Rsqrtø
:model_1/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_1_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:model_1/batch_normalization_2/batchnorm/mul/ReadVariableOpý
+model_1/batch_normalization_2/batchnorm/mulMul1model_1/batch_normalization_2/batchnorm/Rsqrt:y:0Bmodel_1/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+model_1/batch_normalization_2/batchnorm/mulú
-model_1/batch_normalization_2/batchnorm/mul_1Mul,model_1/average_pooling1d_4/Squeeze:output:0/model_1/batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2/
-model_1/batch_normalization_2/batchnorm/mul_1ò
8model_1/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_1_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model_1/batch_normalization_2/batchnorm/ReadVariableOp_1ý
-model_1/batch_normalization_2/batchnorm/mul_2Mul@model_1/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0/model_1/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-model_1/batch_normalization_2/batchnorm/mul_2ò
8model_1/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_1_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8model_1/batch_normalization_2/batchnorm/ReadVariableOp_2û
+model_1/batch_normalization_2/batchnorm/subSub@model_1/batch_normalization_2/batchnorm/ReadVariableOp_2:value:01model_1/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+model_1/batch_normalization_2/batchnorm/sub
-model_1/batch_normalization_2/batchnorm/add_1AddV21model_1/batch_normalization_2/batchnorm/mul_1:z:0/model_1/batch_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2/
-model_1/batch_normalization_2/batchnorm/add_1ì
6model_1/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp?model_1_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype028
6model_1/batch_normalization_3/batchnorm/ReadVariableOp£
-model_1/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-model_1/batch_normalization_3/batchnorm/add/y
+model_1/batch_normalization_3/batchnorm/addAddV2>model_1/batch_normalization_3/batchnorm/ReadVariableOp:value:06model_1/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2-
+model_1/batch_normalization_3/batchnorm/add½
-model_1/batch_normalization_3/batchnorm/RsqrtRsqrt/model_1/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-model_1/batch_normalization_3/batchnorm/Rsqrtø
:model_1/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_1_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:model_1/batch_normalization_3/batchnorm/mul/ReadVariableOpý
+model_1/batch_normalization_3/batchnorm/mulMul1model_1/batch_normalization_3/batchnorm/Rsqrt:y:0Bmodel_1/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+model_1/batch_normalization_3/batchnorm/mulú
-model_1/batch_normalization_3/batchnorm/mul_1Mul,model_1/average_pooling1d_5/Squeeze:output:0/model_1/batch_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2/
-model_1/batch_normalization_3/batchnorm/mul_1ò
8model_1/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_1_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model_1/batch_normalization_3/batchnorm/ReadVariableOp_1ý
-model_1/batch_normalization_3/batchnorm/mul_2Mul@model_1/batch_normalization_3/batchnorm/ReadVariableOp_1:value:0/model_1/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-model_1/batch_normalization_3/batchnorm/mul_2ò
8model_1/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_1_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8model_1/batch_normalization_3/batchnorm/ReadVariableOp_2û
+model_1/batch_normalization_3/batchnorm/subSub@model_1/batch_normalization_3/batchnorm/ReadVariableOp_2:value:01model_1/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+model_1/batch_normalization_3/batchnorm/sub
-model_1/batch_normalization_3/batchnorm/add_1AddV21model_1/batch_normalization_3/batchnorm/mul_1:z:0/model_1/batch_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2/
-model_1/batch_normalization_3/batchnorm/add_1Ë
model_1/add_1/addAddV21model_1/batch_normalization_2/batchnorm/add_1:z:01model_1/batch_normalization_3/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
model_1/add_1/addÑ
Umodel_1/transformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOp^model_1_transformer_block_3_multi_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_1/transformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpð
Fmodel_1/transformer_block_3/multi_head_attention_3/query/einsum/EinsumEinsummodel_1/add_1/add:z:0]model_1/transformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2H
Fmodel_1/transformer_block_3/multi_head_attention_3/query/einsum/Einsum¯
Kmodel_1/transformer_block_3/multi_head_attention_3/query/add/ReadVariableOpReadVariableOpTmodel_1_transformer_block_3_multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_1/transformer_block_3/multi_head_attention_3/query/add/ReadVariableOpå
<model_1/transformer_block_3/multi_head_attention_3/query/addAddV2Omodel_1/transformer_block_3/multi_head_attention_3/query/einsum/Einsum:output:0Smodel_1/transformer_block_3/multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2>
<model_1/transformer_block_3/multi_head_attention_3/query/addË
Smodel_1/transformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOp\model_1_transformer_block_3_multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02U
Smodel_1/transformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpê
Dmodel_1/transformer_block_3/multi_head_attention_3/key/einsum/EinsumEinsummodel_1/add_1/add:z:0[model_1/transformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2F
Dmodel_1/transformer_block_3/multi_head_attention_3/key/einsum/Einsum©
Imodel_1/transformer_block_3/multi_head_attention_3/key/add/ReadVariableOpReadVariableOpRmodel_1_transformer_block_3_multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

: *
dtype02K
Imodel_1/transformer_block_3/multi_head_attention_3/key/add/ReadVariableOpÝ
:model_1/transformer_block_3/multi_head_attention_3/key/addAddV2Mmodel_1/transformer_block_3/multi_head_attention_3/key/einsum/Einsum:output:0Qmodel_1/transformer_block_3/multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:model_1/transformer_block_3/multi_head_attention_3/key/addÑ
Umodel_1/transformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOp^model_1_transformer_block_3_multi_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_1/transformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpð
Fmodel_1/transformer_block_3/multi_head_attention_3/value/einsum/EinsumEinsummodel_1/add_1/add:z:0]model_1/transformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2H
Fmodel_1/transformer_block_3/multi_head_attention_3/value/einsum/Einsum¯
Kmodel_1/transformer_block_3/multi_head_attention_3/value/add/ReadVariableOpReadVariableOpTmodel_1_transformer_block_3_multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_1/transformer_block_3/multi_head_attention_3/value/add/ReadVariableOpå
<model_1/transformer_block_3/multi_head_attention_3/value/addAddV2Omodel_1/transformer_block_3/multi_head_attention_3/value/einsum/Einsum:output:0Smodel_1/transformer_block_3/multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2>
<model_1/transformer_block_3/multi_head_attention_3/value/add¹
8model_1/transformer_block_3/multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2:
8model_1/transformer_block_3/multi_head_attention_3/Mul/y¶
6model_1/transformer_block_3/multi_head_attention_3/MulMul@model_1/transformer_block_3/multi_head_attention_3/query/add:z:0Amodel_1/transformer_block_3/multi_head_attention_3/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 28
6model_1/transformer_block_3/multi_head_attention_3/Mulì
@model_1/transformer_block_3/multi_head_attention_3/einsum/EinsumEinsum>model_1/transformer_block_3/multi_head_attention_3/key/add:z:0:model_1/transformer_block_3/multi_head_attention_3/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2B
@model_1/transformer_block_3/multi_head_attention_3/einsum/Einsum
Bmodel_1/transformer_block_3/multi_head_attention_3/softmax/SoftmaxSoftmaxImodel_1/transformer_block_3/multi_head_attention_3/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2D
Bmodel_1/transformer_block_3/multi_head_attention_3/softmax/Softmax
Cmodel_1/transformer_block_3/multi_head_attention_3/dropout/IdentityIdentityLmodel_1/transformer_block_3/multi_head_attention_3/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2E
Cmodel_1/transformer_block_3/multi_head_attention_3/dropout/Identity
Bmodel_1/transformer_block_3/multi_head_attention_3/einsum_1/EinsumEinsumLmodel_1/transformer_block_3/multi_head_attention_3/dropout/Identity:output:0@model_1/transformer_block_3/multi_head_attention_3/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2D
Bmodel_1/transformer_block_3/multi_head_attention_3/einsum_1/Einsumò
`model_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpimodel_1_transformer_block_3_multi_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02b
`model_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpÃ
Qmodel_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/EinsumEinsumKmodel_1/transformer_block_3/multi_head_attention_3/einsum_1/Einsum:output:0hmodel_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe2S
Qmodel_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/EinsumÌ
Vmodel_1/transformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOp_model_1_transformer_block_3_multi_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02X
Vmodel_1/transformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp
Gmodel_1/transformer_block_3/multi_head_attention_3/attention_output/addAddV2Zmodel_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum:output:0^model_1/transformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2I
Gmodel_1/transformer_block_3/multi_head_attention_3/attention_output/addï
.model_1/transformer_block_3/dropout_9/IdentityIdentityKmodel_1/transformer_block_3/multi_head_attention_3/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 20
.model_1/transformer_block_3/dropout_9/IdentityÑ
model_1/transformer_block_3/addAddV2model_1/add_1/add:z:07model_1/transformer_block_3/dropout_9/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
model_1/transformer_block_3/addî
Pmodel_1/transformer_block_3/layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2R
Pmodel_1/transformer_block_3/layer_normalization_6/moments/mean/reduction_indicesÏ
>model_1/transformer_block_3/layer_normalization_6/moments/meanMean#model_1/transformer_block_3/add:z:0Ymodel_1/transformer_block_3/layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2@
>model_1/transformer_block_3/layer_normalization_6/moments/mean
Fmodel_1/transformer_block_3/layer_normalization_6/moments/StopGradientStopGradientGmodel_1/transformer_block_3/layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2H
Fmodel_1/transformer_block_3/layer_normalization_6/moments/StopGradientÛ
Kmodel_1/transformer_block_3/layer_normalization_6/moments/SquaredDifferenceSquaredDifference#model_1/transformer_block_3/add:z:0Omodel_1/transformer_block_3/layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2M
Kmodel_1/transformer_block_3/layer_normalization_6/moments/SquaredDifferenceö
Tmodel_1/transformer_block_3/layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2V
Tmodel_1/transformer_block_3/layer_normalization_6/moments/variance/reduction_indices
Bmodel_1/transformer_block_3/layer_normalization_6/moments/varianceMeanOmodel_1/transformer_block_3/layer_normalization_6/moments/SquaredDifference:z:0]model_1/transformer_block_3/layer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2D
Bmodel_1/transformer_block_3/layer_normalization_6/moments/varianceË
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752C
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/add/yÚ
?model_1/transformer_block_3/layer_normalization_6/batchnorm/addAddV2Kmodel_1/transformer_block_3/layer_normalization_6/moments/variance:output:0Jmodel_1/transformer_block_3/layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2A
?model_1/transformer_block_3/layer_normalization_6/batchnorm/add
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/RsqrtRsqrtCmodel_1/transformer_block_3/layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2C
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/Rsqrt´
Nmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpWmodel_1_transformer_block_3_layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02P
Nmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpÞ
?model_1/transformer_block_3/layer_normalization_6/batchnorm/mulMulEmodel_1/transformer_block_3/layer_normalization_6/batchnorm/Rsqrt:y:0Vmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?model_1/transformer_block_3/layer_normalization_6/batchnorm/mul­
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul_1Mul#model_1/transformer_block_3/add:z:0Cmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2C
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul_1Ñ
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul_2MulGmodel_1/transformer_block_3/layer_normalization_6/moments/mean:output:0Cmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2C
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul_2¨
Jmodel_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpReadVariableOpSmodel_1_transformer_block_3_layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02L
Jmodel_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpÚ
?model_1/transformer_block_3/layer_normalization_6/batchnorm/subSubRmodel_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp:value:0Emodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?model_1/transformer_block_3/layer_normalization_6/batchnorm/subÑ
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/add_1AddV2Emodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul_1:z:0Cmodel_1/transformer_block_3/layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2C
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/add_1¬
Jmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOpReadVariableOpSmodel_1_transformer_block_3_sequential_3_dense_10_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02L
Jmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOpÎ
@model_1/transformer_block_3/sequential_3/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_1/transformer_block_3/sequential_3/dense_10/Tensordot/axesÕ
@model_1/transformer_block_3/sequential_3/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_1/transformer_block_3/sequential_3/dense_10/Tensordot/freeû
Amodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ShapeShapeEmodel_1/transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:2C
Amodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ShapeØ
Imodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2/axisË
Dmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2GatherV2Jmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Shape:output:0Imodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/free:output:0Rmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2Ü
Kmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1/axisÑ
Fmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1GatherV2Jmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Shape:output:0Imodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/axes:output:0Tmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1Ð
Amodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ConstÈ
@model_1/transformer_block_3/sequential_3/dense_10/Tensordot/ProdProdMmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2:output:0Jmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_1/transformer_block_3/sequential_3/dense_10/Tensordot/ProdÔ
Cmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Const_1Ð
Bmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Prod_1ProdOmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1:output:0Lmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Prod_1Ô
Gmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat/axisª
Bmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concatConcatV2Imodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/free:output:0Imodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/axes:output:0Pmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concatÔ
Amodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/stackPackImodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Prod:output:0Kmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/stackå
Emodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/transpose	TransposeEmodel_1/transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0Kmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2G
Emodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/transposeç
Cmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ReshapeReshapeImodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/transpose:y:0Jmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2E
Cmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Reshapeæ
Bmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/MatMulMatMulLmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Reshape:output:0Rmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2D
Bmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/MatMulÔ
Cmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2E
Cmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Const_2Ø
Imodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat_1/axis·
Dmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat_1ConcatV2Mmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2:output:0Lmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/Const_2:output:0Rmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat_1Ø
;model_1/transformer_block_3/sequential_3/dense_10/TensordotReshapeLmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/MatMul:product:0Mmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2=
;model_1/transformer_block_3/sequential_3/dense_10/Tensordot¢
Hmodel_1/transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOpQmodel_1_transformer_block_3_sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02J
Hmodel_1/transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOpÏ
9model_1/transformer_block_3/sequential_3/dense_10/BiasAddBiasAddDmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot:output:0Pmodel_1/transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2;
9model_1/transformer_block_3/sequential_3/dense_10/BiasAddò
6model_1/transformer_block_3/sequential_3/dense_10/ReluReluBmodel_1/transformer_block_3/sequential_3/dense_10/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@28
6model_1/transformer_block_3/sequential_3/dense_10/Relu¬
Jmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOpReadVariableOpSmodel_1_transformer_block_3_sequential_3_dense_11_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02L
Jmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOpÎ
@model_1/transformer_block_3/sequential_3/dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_1/transformer_block_3/sequential_3/dense_11/Tensordot/axesÕ
@model_1/transformer_block_3/sequential_3/dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_1/transformer_block_3/sequential_3/dense_11/Tensordot/freeú
Amodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/ShapeShapeDmodel_1/transformer_block_3/sequential_3/dense_10/Relu:activations:0*
T0*
_output_shapes
:2C
Amodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/ShapeØ
Imodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2/axisË
Dmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2GatherV2Jmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/Shape:output:0Imodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/free:output:0Rmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2Ü
Kmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2_1/axisÑ
Fmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2_1GatherV2Jmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/Shape:output:0Imodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/axes:output:0Tmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2_1Ð
Amodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/ConstÈ
@model_1/transformer_block_3/sequential_3/dense_11/Tensordot/ProdProdMmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2:output:0Jmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_1/transformer_block_3/sequential_3/dense_11/Tensordot/ProdÔ
Cmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/Const_1Ð
Bmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/Prod_1ProdOmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2_1:output:0Lmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/Prod_1Ô
Gmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/concat/axisª
Bmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/concatConcatV2Imodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/free:output:0Imodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/axes:output:0Pmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/concatÔ
Amodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/stackPackImodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/Prod:output:0Kmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/stackä
Emodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/transpose	TransposeDmodel_1/transformer_block_3/sequential_3/dense_10/Relu:activations:0Kmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2G
Emodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/transposeç
Cmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/ReshapeReshapeImodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/transpose:y:0Jmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2E
Cmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/Reshapeæ
Bmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/MatMulMatMulLmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/Reshape:output:0Rmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2D
Bmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/MatMulÔ
Cmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/Const_2Ø
Imodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/concat_1/axis·
Dmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/concat_1ConcatV2Mmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2:output:0Lmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/Const_2:output:0Rmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/concat_1Ø
;model_1/transformer_block_3/sequential_3/dense_11/TensordotReshapeLmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/MatMul:product:0Mmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2=
;model_1/transformer_block_3/sequential_3/dense_11/Tensordot¢
Hmodel_1/transformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOpReadVariableOpQmodel_1_transformer_block_3_sequential_3_dense_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel_1/transformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOpÏ
9model_1/transformer_block_3/sequential_3/dense_11/BiasAddBiasAddDmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot:output:0Pmodel_1/transformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9model_1/transformer_block_3/sequential_3/dense_11/BiasAddè
/model_1/transformer_block_3/dropout_10/IdentityIdentityBmodel_1/transformer_block_3/sequential_3/dense_11/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/model_1/transformer_block_3/dropout_10/Identity
!model_1/transformer_block_3/add_1AddV2Emodel_1/transformer_block_3/layer_normalization_6/batchnorm/add_1:z:08model_1/transformer_block_3/dropout_10/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2#
!model_1/transformer_block_3/add_1î
Pmodel_1/transformer_block_3/layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2R
Pmodel_1/transformer_block_3/layer_normalization_7/moments/mean/reduction_indicesÑ
>model_1/transformer_block_3/layer_normalization_7/moments/meanMean%model_1/transformer_block_3/add_1:z:0Ymodel_1/transformer_block_3/layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2@
>model_1/transformer_block_3/layer_normalization_7/moments/mean
Fmodel_1/transformer_block_3/layer_normalization_7/moments/StopGradientStopGradientGmodel_1/transformer_block_3/layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2H
Fmodel_1/transformer_block_3/layer_normalization_7/moments/StopGradientÝ
Kmodel_1/transformer_block_3/layer_normalization_7/moments/SquaredDifferenceSquaredDifference%model_1/transformer_block_3/add_1:z:0Omodel_1/transformer_block_3/layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2M
Kmodel_1/transformer_block_3/layer_normalization_7/moments/SquaredDifferenceö
Tmodel_1/transformer_block_3/layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2V
Tmodel_1/transformer_block_3/layer_normalization_7/moments/variance/reduction_indices
Bmodel_1/transformer_block_3/layer_normalization_7/moments/varianceMeanOmodel_1/transformer_block_3/layer_normalization_7/moments/SquaredDifference:z:0]model_1/transformer_block_3/layer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2D
Bmodel_1/transformer_block_3/layer_normalization_7/moments/varianceË
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752C
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/add/yÚ
?model_1/transformer_block_3/layer_normalization_7/batchnorm/addAddV2Kmodel_1/transformer_block_3/layer_normalization_7/moments/variance:output:0Jmodel_1/transformer_block_3/layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2A
?model_1/transformer_block_3/layer_normalization_7/batchnorm/add
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/RsqrtRsqrtCmodel_1/transformer_block_3/layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2C
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/Rsqrt´
Nmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpWmodel_1_transformer_block_3_layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02P
Nmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpÞ
?model_1/transformer_block_3/layer_normalization_7/batchnorm/mulMulEmodel_1/transformer_block_3/layer_normalization_7/batchnorm/Rsqrt:y:0Vmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?model_1/transformer_block_3/layer_normalization_7/batchnorm/mul¯
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul_1Mul%model_1/transformer_block_3/add_1:z:0Cmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2C
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul_1Ñ
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul_2MulGmodel_1/transformer_block_3/layer_normalization_7/moments/mean:output:0Cmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2C
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul_2¨
Jmodel_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpReadVariableOpSmodel_1_transformer_block_3_layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02L
Jmodel_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpÚ
?model_1/transformer_block_3/layer_normalization_7/batchnorm/subSubRmodel_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp:value:0Emodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?model_1/transformer_block_3/layer_normalization_7/batchnorm/subÑ
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/add_1AddV2Emodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul_1:z:0Cmodel_1/transformer_block_3/layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2C
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/add_1
model_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`  2
model_1/flatten_1/ConstÝ
model_1/flatten_1/ReshapeReshapeEmodel_1/transformer_block_3/layer_normalization_7/batchnorm/add_1:z:0 model_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
model_1/flatten_1/Reshape
!model_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_1/concat/axisÝ
model_1/concatenate_1/concatConcatV2"model_1/flatten_1/Reshape:output:0input_4*model_1/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
model_1/concatenate_1/concatÁ
&model_1/dense_12/MatMul/ReadVariableOpReadVariableOp/model_1_dense_12_matmul_readvariableop_resource*
_output_shapes
:	è@*
dtype02(
&model_1/dense_12/MatMul/ReadVariableOpÅ
model_1/dense_12/MatMulMatMul%model_1/concatenate_1/concat:output:0.model_1/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/dense_12/MatMul¿
'model_1/dense_12/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/dense_12/BiasAdd/ReadVariableOpÅ
model_1/dense_12/BiasAddBiasAdd!model_1/dense_12/MatMul:product:0/model_1/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/dense_12/BiasAdd
model_1/dense_12/ReluRelu!model_1/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/dense_12/Relu
model_1/dropout_11/IdentityIdentity#model_1/dense_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/dropout_11/IdentityÀ
&model_1/dense_13/MatMul/ReadVariableOpReadVariableOp/model_1_dense_13_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02(
&model_1/dense_13/MatMul/ReadVariableOpÄ
model_1/dense_13/MatMulMatMul$model_1/dropout_11/Identity:output:0.model_1/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/dense_13/MatMul¿
'model_1/dense_13/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/dense_13/BiasAdd/ReadVariableOpÅ
model_1/dense_13/BiasAddBiasAdd!model_1/dense_13/MatMul:product:0/model_1/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/dense_13/BiasAdd
model_1/dense_13/ReluRelu!model_1/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/dense_13/Relu
model_1/dropout_12/IdentityIdentity#model_1/dense_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/dropout_12/IdentityÀ
&model_1/dense_14/MatMul/ReadVariableOpReadVariableOp/model_1_dense_14_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02(
&model_1/dense_14/MatMul/ReadVariableOpÄ
model_1/dense_14/MatMulMatMul$model_1/dropout_12/Identity:output:0.model_1/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/dense_14/MatMul¿
'model_1/dense_14/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/dense_14/BiasAdd/ReadVariableOpÅ
model_1/dense_14/BiasAddBiasAdd!model_1/dense_14/MatMul:product:0/model_1/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/dense_14/BiasAdd
model_1/dense_14/ReluRelu!model_1/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/dense_14/Relu
model_1/dropout_13/IdentityIdentity#model_1/dense_14/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_1/dropout_13/IdentityÀ
&model_1/dense_15/MatMul/ReadVariableOpReadVariableOp/model_1_dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&model_1/dense_15/MatMul/ReadVariableOpÄ
model_1/dense_15/MatMulMatMul$model_1/dropout_13/Identity:output:0.model_1/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense_15/MatMul¿
'model_1/dense_15/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/dense_15/BiasAdd/ReadVariableOpÅ
model_1/dense_15/BiasAddBiasAdd!model_1/dense_15/MatMul:product:0/model_1/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense_15/BiasAddû
IdentityIdentity!model_1/dense_15/BiasAdd:output:07^model_1/batch_normalization_2/batchnorm/ReadVariableOp9^model_1/batch_normalization_2/batchnorm/ReadVariableOp_19^model_1/batch_normalization_2/batchnorm/ReadVariableOp_2;^model_1/batch_normalization_2/batchnorm/mul/ReadVariableOp7^model_1/batch_normalization_3/batchnorm/ReadVariableOp9^model_1/batch_normalization_3/batchnorm/ReadVariableOp_19^model_1/batch_normalization_3/batchnorm/ReadVariableOp_2;^model_1/batch_normalization_3/batchnorm/mul/ReadVariableOp(^model_1/conv1d_2/BiasAdd/ReadVariableOp4^model_1/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp(^model_1/conv1d_3/BiasAdd/ReadVariableOp4^model_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp(^model_1/dense_12/BiasAdd/ReadVariableOp'^model_1/dense_12/MatMul/ReadVariableOp(^model_1/dense_13/BiasAdd/ReadVariableOp'^model_1/dense_13/MatMul/ReadVariableOp(^model_1/dense_14/BiasAdd/ReadVariableOp'^model_1/dense_14/MatMul/ReadVariableOp(^model_1/dense_15/BiasAdd/ReadVariableOp'^model_1/dense_15/MatMul/ReadVariableOpD^model_1/token_and_position_embedding_1/embedding_2/embedding_lookupD^model_1/token_and_position_embedding_1/embedding_3/embedding_lookupK^model_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpO^model_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpK^model_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpO^model_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpW^model_1/transformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpa^model_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpJ^model_1/transformer_block_3/multi_head_attention_3/key/add/ReadVariableOpT^model_1/transformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpL^model_1/transformer_block_3/multi_head_attention_3/query/add/ReadVariableOpV^model_1/transformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpL^model_1/transformer_block_3/multi_head_attention_3/value/add/ReadVariableOpV^model_1/transformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpI^model_1/transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOpK^model_1/transformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOpI^model_1/transformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOpK^model_1/transformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ô
_input_shapesÂ
¿:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::2p
6model_1/batch_normalization_2/batchnorm/ReadVariableOp6model_1/batch_normalization_2/batchnorm/ReadVariableOp2t
8model_1/batch_normalization_2/batchnorm/ReadVariableOp_18model_1/batch_normalization_2/batchnorm/ReadVariableOp_12t
8model_1/batch_normalization_2/batchnorm/ReadVariableOp_28model_1/batch_normalization_2/batchnorm/ReadVariableOp_22x
:model_1/batch_normalization_2/batchnorm/mul/ReadVariableOp:model_1/batch_normalization_2/batchnorm/mul/ReadVariableOp2p
6model_1/batch_normalization_3/batchnorm/ReadVariableOp6model_1/batch_normalization_3/batchnorm/ReadVariableOp2t
8model_1/batch_normalization_3/batchnorm/ReadVariableOp_18model_1/batch_normalization_3/batchnorm/ReadVariableOp_12t
8model_1/batch_normalization_3/batchnorm/ReadVariableOp_28model_1/batch_normalization_3/batchnorm/ReadVariableOp_22x
:model_1/batch_normalization_3/batchnorm/mul/ReadVariableOp:model_1/batch_normalization_3/batchnorm/mul/ReadVariableOp2R
'model_1/conv1d_2/BiasAdd/ReadVariableOp'model_1/conv1d_2/BiasAdd/ReadVariableOp2j
3model_1/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp3model_1/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2R
'model_1/conv1d_3/BiasAdd/ReadVariableOp'model_1/conv1d_3/BiasAdd/ReadVariableOp2j
3model_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp3model_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp2R
'model_1/dense_12/BiasAdd/ReadVariableOp'model_1/dense_12/BiasAdd/ReadVariableOp2P
&model_1/dense_12/MatMul/ReadVariableOp&model_1/dense_12/MatMul/ReadVariableOp2R
'model_1/dense_13/BiasAdd/ReadVariableOp'model_1/dense_13/BiasAdd/ReadVariableOp2P
&model_1/dense_13/MatMul/ReadVariableOp&model_1/dense_13/MatMul/ReadVariableOp2R
'model_1/dense_14/BiasAdd/ReadVariableOp'model_1/dense_14/BiasAdd/ReadVariableOp2P
&model_1/dense_14/MatMul/ReadVariableOp&model_1/dense_14/MatMul/ReadVariableOp2R
'model_1/dense_15/BiasAdd/ReadVariableOp'model_1/dense_15/BiasAdd/ReadVariableOp2P
&model_1/dense_15/MatMul/ReadVariableOp&model_1/dense_15/MatMul/ReadVariableOp2
Cmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookupCmodel_1/token_and_position_embedding_1/embedding_2/embedding_lookup2
Cmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookupCmodel_1/token_and_position_embedding_1/embedding_3/embedding_lookup2
Jmodel_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpJmodel_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp2 
Nmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpNmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp2
Jmodel_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpJmodel_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp2 
Nmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpNmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp2°
Vmodel_1/transformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpVmodel_1/transformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp2Ä
`model_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp`model_1/transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp2
Imodel_1/transformer_block_3/multi_head_attention_3/key/add/ReadVariableOpImodel_1/transformer_block_3/multi_head_attention_3/key/add/ReadVariableOp2ª
Smodel_1/transformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpSmodel_1/transformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp2
Kmodel_1/transformer_block_3/multi_head_attention_3/query/add/ReadVariableOpKmodel_1/transformer_block_3/multi_head_attention_3/query/add/ReadVariableOp2®
Umodel_1/transformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpUmodel_1/transformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp2
Kmodel_1/transformer_block_3/multi_head_attention_3/value/add/ReadVariableOpKmodel_1/transformer_block_3/multi_head_attention_3/value/add/ReadVariableOp2®
Umodel_1/transformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpUmodel_1/transformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp2
Hmodel_1/transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOpHmodel_1/transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp2
Jmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOpJmodel_1/transformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp2
Hmodel_1/transformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOpHmodel_1/transformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOp2
Jmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOpJmodel_1/transformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOp:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_3:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
Êå
â%
C__inference_model_1_layer_call_and_return_conditional_losses_548586
inputs_0
inputs_1F
Btoken_and_position_embedding_1_embedding_3_embedding_lookup_548347F
Btoken_and_position_embedding_1_embedding_2_embedding_lookup_5483538
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource,
(conv1d_3_biasadd_readvariableop_resource;
7batch_normalization_2_batchnorm_readvariableop_resource?
;batch_normalization_2_batchnorm_mul_readvariableop_resource=
9batch_normalization_2_batchnorm_readvariableop_1_resource=
9batch_normalization_2_batchnorm_readvariableop_2_resource;
7batch_normalization_3_batchnorm_readvariableop_resource?
;batch_normalization_3_batchnorm_mul_readvariableop_resource=
9batch_normalization_3_batchnorm_readvariableop_1_resource=
9batch_normalization_3_batchnorm_readvariableop_2_resourceZ
Vtransformer_block_3_multi_head_attention_3_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_3_multi_head_attention_3_query_add_readvariableop_resourceX
Ttransformer_block_3_multi_head_attention_3_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_3_multi_head_attention_3_key_add_readvariableop_resourceZ
Vtransformer_block_3_multi_head_attention_3_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_3_multi_head_attention_3_value_add_readvariableop_resourcee
atransformer_block_3_multi_head_attention_3_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_3_multi_head_attention_3_attention_output_add_readvariableop_resourceS
Otransformer_block_3_layer_normalization_6_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_3_layer_normalization_6_batchnorm_readvariableop_resourceO
Ktransformer_block_3_sequential_3_dense_10_tensordot_readvariableop_resourceM
Itransformer_block_3_sequential_3_dense_10_biasadd_readvariableop_resourceO
Ktransformer_block_3_sequential_3_dense_11_tensordot_readvariableop_resourceM
Itransformer_block_3_sequential_3_dense_11_biasadd_readvariableop_resourceS
Otransformer_block_3_layer_normalization_7_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_3_layer_normalization_7_batchnorm_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource
identity¢.batch_normalization_2/batchnorm/ReadVariableOp¢0batch_normalization_2/batchnorm/ReadVariableOp_1¢0batch_normalization_2/batchnorm/ReadVariableOp_2¢2batch_normalization_2/batchnorm/mul/ReadVariableOp¢.batch_normalization_3/batchnorm/ReadVariableOp¢0batch_normalization_3/batchnorm/ReadVariableOp_1¢0batch_normalization_3/batchnorm/ReadVariableOp_2¢2batch_normalization_3/batchnorm/mul/ReadVariableOp¢conv1d_2/BiasAdd/ReadVariableOp¢+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp¢conv1d_3/BiasAdd/ReadVariableOp¢+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp¢dense_14/BiasAdd/ReadVariableOp¢dense_14/MatMul/ReadVariableOp¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¢;token_and_position_embedding_1/embedding_2/embedding_lookup¢;token_and_position_embedding_1/embedding_3/embedding_lookup¢Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp¢Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp¢Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp¢Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp¢Ntransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp¢Xtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp¢Atransformer_block_3/multi_head_attention_3/key/add/ReadVariableOp¢Ktransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp¢Ctransformer_block_3/multi_head_attention_3/query/add/ReadVariableOp¢Mtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp¢Ctransformer_block_3/multi_head_attention_3/value/add/ReadVariableOp¢Mtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp¢@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp¢Btransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp¢@transformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOp¢Btransformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOp
$token_and_position_embedding_1/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_1/Shape»
2token_and_position_embedding_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ24
2token_and_position_embedding_1/strided_slice/stack¶
4token_and_position_embedding_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_1/strided_slice/stack_1¶
4token_and_position_embedding_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_1/strided_slice/stack_2
,token_and_position_embedding_1/strided_sliceStridedSlice-token_and_position_embedding_1/Shape:output:0;token_and_position_embedding_1/strided_slice/stack:output:0=token_and_position_embedding_1/strided_slice/stack_1:output:0=token_and_position_embedding_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_1/strided_slice
*token_and_position_embedding_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_1/range/start
*token_and_position_embedding_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_1/range/delta
$token_and_position_embedding_1/rangeRange3token_and_position_embedding_1/range/start:output:05token_and_position_embedding_1/strided_slice:output:03token_and_position_embedding_1/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$token_and_position_embedding_1/rangeÊ
;token_and_position_embedding_1/embedding_3/embedding_lookupResourceGatherBtoken_and_position_embedding_1_embedding_3_embedding_lookup_548347-token_and_position_embedding_1/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_1/embedding_3/embedding_lookup/548347*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02=
;token_and_position_embedding_1/embedding_3/embedding_lookup
Dtoken_and_position_embedding_1/embedding_3/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_1/embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_1/embedding_3/embedding_lookup/548347*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2F
Dtoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity
Ftoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2H
Ftoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1¶
/token_and_position_embedding_1/embedding_2/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR21
/token_and_position_embedding_1/embedding_2/CastÕ
;token_and_position_embedding_1/embedding_2/embedding_lookupResourceGatherBtoken_and_position_embedding_1_embedding_2_embedding_lookup_5483533token_and_position_embedding_1/embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_1/embedding_2/embedding_lookup/548353*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02=
;token_and_position_embedding_1/embedding_2/embedding_lookup
Dtoken_and_position_embedding_1/embedding_2/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_1/embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_1/embedding_2/embedding_lookup/548353*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2F
Dtoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity¢
Ftoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2H
Ftoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1ª
"token_and_position_embedding_1/addAddV2Otoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2$
"token_and_position_embedding_1/add
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_2/conv1d/ExpandDims/dimÒ
conv1d_2/conv1d/ExpandDims
ExpandDims&token_and_position_embedding_1/add:z:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_2/conv1d/ExpandDimsÔ
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: *
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dimÜ
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: 2
conv1d_2/conv1d/ExpandDims_1Ü
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*
paddingSAME*
strides
2
conv1d_2/conv1d¯
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_2/conv1d/Squeeze¨
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp²
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2
conv1d_2/BiasAddy
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2
conv1d_2/Relu
"average_pooling1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_3/ExpandDims/dimÔ
average_pooling1d_3/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0+average_pooling1d_3/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2 
average_pooling1d_3/ExpandDimsæ
average_pooling1d_3/AvgPoolAvgPool'average_pooling1d_3/ExpandDims:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ*
ksize
*
paddingVALID*
strides
2
average_pooling1d_3/AvgPoolº
average_pooling1d_3/SqueezeSqueeze$average_pooling1d_3/AvgPool:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ*
squeeze_dims
2
average_pooling1d_3/Squeeze
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_3/conv1d/ExpandDims/dimÑ
conv1d_3/conv1d/ExpandDims
ExpandDims$average_pooling1d_3/Squeeze:output:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
conv1d_3/conv1d/ExpandDimsÔ
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:	 *
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dimÜ
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:	 2
conv1d_3/conv1d/ExpandDims_1Û
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
paddingSAME*
strides
2
conv1d_3/conv1d®
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_3/conv1d/Squeeze§
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_3/BiasAdd/ReadVariableOp±
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_3/BiasAddx
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_3/Relu
"average_pooling1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_5/ExpandDims/dimÞ
average_pooling1d_5/ExpandDims
ExpandDims&token_and_position_embedding_1/add:z:0+average_pooling1d_5/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2 
average_pooling1d_5/ExpandDimsæ
average_pooling1d_5/AvgPoolAvgPool'average_pooling1d_5/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize	
¬*
paddingVALID*
strides	
¬2
average_pooling1d_5/AvgPool¸
average_pooling1d_5/SqueezeSqueeze$average_pooling1d_5/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_5/Squeeze
"average_pooling1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_4/ExpandDims/dimÓ
average_pooling1d_4/ExpandDims
ExpandDimsconv1d_3/Relu:activations:0+average_pooling1d_4/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2 
average_pooling1d_4/ExpandDimsä
average_pooling1d_4/AvgPoolAvgPool'average_pooling1d_4/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize

*
paddingVALID*
strides

2
average_pooling1d_4/AvgPool¸
average_pooling1d_4/SqueezeSqueeze$average_pooling1d_4/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_4/SqueezeÔ
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_2/batchnorm/add/yà
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_2/batchnorm/add¥
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_2/batchnorm/Rsqrtà
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_2/batchnorm/mulÚ
%batch_normalization_2/batchnorm/mul_1Mul$average_pooling1d_4/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_2/batchnorm/mul_1Ú
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1Ý
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_2/batchnorm/mul_2Ú
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2Û
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_2/batchnorm/subá
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_2/batchnorm/add_1Ô
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_3/batchnorm/add/yà
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_3/batchnorm/add¥
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_3/batchnorm/Rsqrtà
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_3/batchnorm/mulÚ
%batch_normalization_3/batchnorm/mul_1Mul$average_pooling1d_5/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_3/batchnorm/mul_1Ú
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1Ý
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_3/batchnorm/mul_2Ú
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2Û
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_3/batchnorm/subá
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_3/batchnorm/add_1«
	add_1/addAddV2)batch_normalization_2/batchnorm/add_1:z:0)batch_normalization_3/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
	add_1/add¹
Mtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_3_multi_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpÐ
>transformer_block_3/multi_head_attention_3/query/einsum/EinsumEinsumadd_1/add:z:0Utransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2@
>transformer_block_3/multi_head_attention_3/query/einsum/Einsum
Ctransformer_block_3/multi_head_attention_3/query/add/ReadVariableOpReadVariableOpLtransformer_block_3_multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_3/multi_head_attention_3/query/add/ReadVariableOpÅ
4transformer_block_3/multi_head_attention_3/query/addAddV2Gtransformer_block_3/multi_head_attention_3/query/einsum/Einsum:output:0Ktransformer_block_3/multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_3/multi_head_attention_3/query/add³
Ktransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_3_multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpÊ
<transformer_block_3/multi_head_attention_3/key/einsum/EinsumEinsumadd_1/add:z:0Stransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2>
<transformer_block_3/multi_head_attention_3/key/einsum/Einsum
Atransformer_block_3/multi_head_attention_3/key/add/ReadVariableOpReadVariableOpJtransformer_block_3_multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_3/multi_head_attention_3/key/add/ReadVariableOp½
2transformer_block_3/multi_head_attention_3/key/addAddV2Etransformer_block_3/multi_head_attention_3/key/einsum/Einsum:output:0Itransformer_block_3/multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 24
2transformer_block_3/multi_head_attention_3/key/add¹
Mtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_3_multi_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpÐ
>transformer_block_3/multi_head_attention_3/value/einsum/EinsumEinsumadd_1/add:z:0Utransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2@
>transformer_block_3/multi_head_attention_3/value/einsum/Einsum
Ctransformer_block_3/multi_head_attention_3/value/add/ReadVariableOpReadVariableOpLtransformer_block_3_multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_3/multi_head_attention_3/value/add/ReadVariableOpÅ
4transformer_block_3/multi_head_attention_3/value/addAddV2Gtransformer_block_3/multi_head_attention_3/value/einsum/Einsum:output:0Ktransformer_block_3/multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_3/multi_head_attention_3/value/add©
0transformer_block_3/multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>22
0transformer_block_3/multi_head_attention_3/Mul/y
.transformer_block_3/multi_head_attention_3/MulMul8transformer_block_3/multi_head_attention_3/query/add:z:09transformer_block_3/multi_head_attention_3/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 20
.transformer_block_3/multi_head_attention_3/MulÌ
8transformer_block_3/multi_head_attention_3/einsum/EinsumEinsum6transformer_block_3/multi_head_attention_3/key/add:z:02transformer_block_3/multi_head_attention_3/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2:
8transformer_block_3/multi_head_attention_3/einsum/Einsum
:transformer_block_3/multi_head_attention_3/softmax/SoftmaxSoftmaxAtransformer_block_3/multi_head_attention_3/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2<
:transformer_block_3/multi_head_attention_3/softmax/Softmax
;transformer_block_3/multi_head_attention_3/dropout/IdentityIdentityDtransformer_block_3/multi_head_attention_3/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2=
;transformer_block_3/multi_head_attention_3/dropout/Identityä
:transformer_block_3/multi_head_attention_3/einsum_1/EinsumEinsumDtransformer_block_3/multi_head_attention_3/dropout/Identity:output:08transformer_block_3/multi_head_attention_3/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2<
:transformer_block_3/multi_head_attention_3/einsum_1/EinsumÚ
Xtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_3_multi_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp£
Itransformer_block_3/multi_head_attention_3/attention_output/einsum/EinsumEinsumCtransformer_block_3/multi_head_attention_3/einsum_1/Einsum:output:0`transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe2K
Itransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum´
Ntransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_3_multi_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpí
?transformer_block_3/multi_head_attention_3/attention_output/addAddV2Rtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum:output:0Vtransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?transformer_block_3/multi_head_attention_3/attention_output/add×
&transformer_block_3/dropout_9/IdentityIdentityCtransformer_block_3/multi_head_attention_3/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&transformer_block_3/dropout_9/Identity±
transformer_block_3/addAddV2add_1/add:z:0/transformer_block_3/dropout_9/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_3/addÞ
Htransformer_block_3/layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_3/layer_normalization_6/moments/mean/reduction_indices¯
6transformer_block_3/layer_normalization_6/moments/meanMeantransformer_block_3/add:z:0Qtransformer_block_3/layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(28
6transformer_block_3/layer_normalization_6/moments/mean
>transformer_block_3/layer_normalization_6/moments/StopGradientStopGradient?transformer_block_3/layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2@
>transformer_block_3/layer_normalization_6/moments/StopGradient»
Ctransformer_block_3/layer_normalization_6/moments/SquaredDifferenceSquaredDifferencetransformer_block_3/add:z:0Gtransformer_block_3/layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2E
Ctransformer_block_3/layer_normalization_6/moments/SquaredDifferenceæ
Ltransformer_block_3/layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_3/layer_normalization_6/moments/variance/reduction_indicesç
:transformer_block_3/layer_normalization_6/moments/varianceMeanGtransformer_block_3/layer_normalization_6/moments/SquaredDifference:z:0Utransformer_block_3/layer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2<
:transformer_block_3/layer_normalization_6/moments/variance»
9transformer_block_3/layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752;
9transformer_block_3/layer_normalization_6/batchnorm/add/yº
7transformer_block_3/layer_normalization_6/batchnorm/addAddV2Ctransformer_block_3/layer_normalization_6/moments/variance:output:0Btransformer_block_3/layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#29
7transformer_block_3/layer_normalization_6/batchnorm/addò
9transformer_block_3/layer_normalization_6/batchnorm/RsqrtRsqrt;transformer_block_3/layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2;
9transformer_block_3/layer_normalization_6/batchnorm/Rsqrt
Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_3_layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp¾
7transformer_block_3/layer_normalization_6/batchnorm/mulMul=transformer_block_3/layer_normalization_6/batchnorm/Rsqrt:y:0Ntransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 29
7transformer_block_3/layer_normalization_6/batchnorm/mul
9transformer_block_3/layer_normalization_6/batchnorm/mul_1Multransformer_block_3/add:z:0;transformer_block_3/layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_3/layer_normalization_6/batchnorm/mul_1±
9transformer_block_3/layer_normalization_6/batchnorm/mul_2Mul?transformer_block_3/layer_normalization_6/moments/mean:output:0;transformer_block_3/layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_3/layer_normalization_6/batchnorm/mul_2
Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_3_layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpº
7transformer_block_3/layer_normalization_6/batchnorm/subSubJtransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp:value:0=transformer_block_3/layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 29
7transformer_block_3/layer_normalization_6/batchnorm/sub±
9transformer_block_3/layer_normalization_6/batchnorm/add_1AddV2=transformer_block_3/layer_normalization_6/batchnorm/mul_1:z:0;transformer_block_3/layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_3/layer_normalization_6/batchnorm/add_1
Btransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_3_sequential_3_dense_10_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02D
Btransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp¾
8transformer_block_3/sequential_3/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_3/sequential_3/dense_10/Tensordot/axesÅ
8transformer_block_3/sequential_3/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_3/sequential_3/dense_10/Tensordot/freeã
9transformer_block_3/sequential_3/dense_10/Tensordot/ShapeShape=transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_3/sequential_3/dense_10/Tensordot/ShapeÈ
Atransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2/axis£
<transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2GatherV2Btransformer_block_3/sequential_3/dense_10/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_10/Tensordot/free:output:0Jtransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2Ì
Ctransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1/axis©
>transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1GatherV2Btransformer_block_3/sequential_3/dense_10/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_10/Tensordot/axes:output:0Ltransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1À
9transformer_block_3/sequential_3/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_3/sequential_3/dense_10/Tensordot/Const¨
8transformer_block_3/sequential_3/dense_10/Tensordot/ProdProdEtransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2:output:0Btransformer_block_3/sequential_3/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_3/sequential_3/dense_10/Tensordot/ProdÄ
;transformer_block_3/sequential_3/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_3/sequential_3/dense_10/Tensordot/Const_1°
:transformer_block_3/sequential_3/dense_10/Tensordot/Prod_1ProdGtransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1:output:0Dtransformer_block_3/sequential_3/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_3/sequential_3/dense_10/Tensordot/Prod_1Ä
?transformer_block_3/sequential_3/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_3/sequential_3/dense_10/Tensordot/concat/axis
:transformer_block_3/sequential_3/dense_10/Tensordot/concatConcatV2Atransformer_block_3/sequential_3/dense_10/Tensordot/free:output:0Atransformer_block_3/sequential_3/dense_10/Tensordot/axes:output:0Htransformer_block_3/sequential_3/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_3/sequential_3/dense_10/Tensordot/concat´
9transformer_block_3/sequential_3/dense_10/Tensordot/stackPackAtransformer_block_3/sequential_3/dense_10/Tensordot/Prod:output:0Ctransformer_block_3/sequential_3/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_3/sequential_3/dense_10/Tensordot/stackÅ
=transformer_block_3/sequential_3/dense_10/Tensordot/transpose	Transpose=transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0Ctransformer_block_3/sequential_3/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2?
=transformer_block_3/sequential_3/dense_10/Tensordot/transposeÇ
;transformer_block_3/sequential_3/dense_10/Tensordot/ReshapeReshapeAtransformer_block_3/sequential_3/dense_10/Tensordot/transpose:y:0Btransformer_block_3/sequential_3/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2=
;transformer_block_3/sequential_3/dense_10/Tensordot/ReshapeÆ
:transformer_block_3/sequential_3/dense_10/Tensordot/MatMulMatMulDtransformer_block_3/sequential_3/dense_10/Tensordot/Reshape:output:0Jtransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2<
:transformer_block_3/sequential_3/dense_10/Tensordot/MatMulÄ
;transformer_block_3/sequential_3/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2=
;transformer_block_3/sequential_3/dense_10/Tensordot/Const_2È
Atransformer_block_3/sequential_3/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_3/sequential_3/dense_10/Tensordot/concat_1/axis
<transformer_block_3/sequential_3/dense_10/Tensordot/concat_1ConcatV2Etransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2:output:0Dtransformer_block_3/sequential_3/dense_10/Tensordot/Const_2:output:0Jtransformer_block_3/sequential_3/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_3/sequential_3/dense_10/Tensordot/concat_1¸
3transformer_block_3/sequential_3/dense_10/TensordotReshapeDtransformer_block_3/sequential_3/dense_10/Tensordot/MatMul:product:0Etransformer_block_3/sequential_3/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@25
3transformer_block_3/sequential_3/dense_10/Tensordot
@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_3_sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp¯
1transformer_block_3/sequential_3/dense_10/BiasAddBiasAdd<transformer_block_3/sequential_3/dense_10/Tensordot:output:0Htransformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@23
1transformer_block_3/sequential_3/dense_10/BiasAddÚ
.transformer_block_3/sequential_3/dense_10/ReluRelu:transformer_block_3/sequential_3/dense_10/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@20
.transformer_block_3/sequential_3/dense_10/Relu
Btransformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_3_sequential_3_dense_11_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02D
Btransformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOp¾
8transformer_block_3/sequential_3/dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_3/sequential_3/dense_11/Tensordot/axesÅ
8transformer_block_3/sequential_3/dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_3/sequential_3/dense_11/Tensordot/freeâ
9transformer_block_3/sequential_3/dense_11/Tensordot/ShapeShape<transformer_block_3/sequential_3/dense_10/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_3/sequential_3/dense_11/Tensordot/ShapeÈ
Atransformer_block_3/sequential_3/dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_3/sequential_3/dense_11/Tensordot/GatherV2/axis£
<transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2GatherV2Btransformer_block_3/sequential_3/dense_11/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_11/Tensordot/free:output:0Jtransformer_block_3/sequential_3/dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2Ì
Ctransformer_block_3/sequential_3/dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_3/sequential_3/dense_11/Tensordot/GatherV2_1/axis©
>transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2_1GatherV2Btransformer_block_3/sequential_3/dense_11/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_11/Tensordot/axes:output:0Ltransformer_block_3/sequential_3/dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2_1À
9transformer_block_3/sequential_3/dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_3/sequential_3/dense_11/Tensordot/Const¨
8transformer_block_3/sequential_3/dense_11/Tensordot/ProdProdEtransformer_block_3/sequential_3/dense_11/Tensordot/GatherV2:output:0Btransformer_block_3/sequential_3/dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_3/sequential_3/dense_11/Tensordot/ProdÄ
;transformer_block_3/sequential_3/dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_3/sequential_3/dense_11/Tensordot/Const_1°
:transformer_block_3/sequential_3/dense_11/Tensordot/Prod_1ProdGtransformer_block_3/sequential_3/dense_11/Tensordot/GatherV2_1:output:0Dtransformer_block_3/sequential_3/dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_3/sequential_3/dense_11/Tensordot/Prod_1Ä
?transformer_block_3/sequential_3/dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_3/sequential_3/dense_11/Tensordot/concat/axis
:transformer_block_3/sequential_3/dense_11/Tensordot/concatConcatV2Atransformer_block_3/sequential_3/dense_11/Tensordot/free:output:0Atransformer_block_3/sequential_3/dense_11/Tensordot/axes:output:0Htransformer_block_3/sequential_3/dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_3/sequential_3/dense_11/Tensordot/concat´
9transformer_block_3/sequential_3/dense_11/Tensordot/stackPackAtransformer_block_3/sequential_3/dense_11/Tensordot/Prod:output:0Ctransformer_block_3/sequential_3/dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_3/sequential_3/dense_11/Tensordot/stackÄ
=transformer_block_3/sequential_3/dense_11/Tensordot/transpose	Transpose<transformer_block_3/sequential_3/dense_10/Relu:activations:0Ctransformer_block_3/sequential_3/dense_11/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2?
=transformer_block_3/sequential_3/dense_11/Tensordot/transposeÇ
;transformer_block_3/sequential_3/dense_11/Tensordot/ReshapeReshapeAtransformer_block_3/sequential_3/dense_11/Tensordot/transpose:y:0Btransformer_block_3/sequential_3/dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2=
;transformer_block_3/sequential_3/dense_11/Tensordot/ReshapeÆ
:transformer_block_3/sequential_3/dense_11/Tensordot/MatMulMatMulDtransformer_block_3/sequential_3/dense_11/Tensordot/Reshape:output:0Jtransformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2<
:transformer_block_3/sequential_3/dense_11/Tensordot/MatMulÄ
;transformer_block_3/sequential_3/dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_3/sequential_3/dense_11/Tensordot/Const_2È
Atransformer_block_3/sequential_3/dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_3/sequential_3/dense_11/Tensordot/concat_1/axis
<transformer_block_3/sequential_3/dense_11/Tensordot/concat_1ConcatV2Etransformer_block_3/sequential_3/dense_11/Tensordot/GatherV2:output:0Dtransformer_block_3/sequential_3/dense_11/Tensordot/Const_2:output:0Jtransformer_block_3/sequential_3/dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_3/sequential_3/dense_11/Tensordot/concat_1¸
3transformer_block_3/sequential_3/dense_11/TensordotReshapeDtransformer_block_3/sequential_3/dense_11/Tensordot/MatMul:product:0Etransformer_block_3/sequential_3/dense_11/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 25
3transformer_block_3/sequential_3/dense_11/Tensordot
@transformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_3_sequential_3_dense_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOp¯
1transformer_block_3/sequential_3/dense_11/BiasAddBiasAdd<transformer_block_3/sequential_3/dense_11/Tensordot:output:0Htransformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 23
1transformer_block_3/sequential_3/dense_11/BiasAddÐ
'transformer_block_3/dropout_10/IdentityIdentity:transformer_block_3/sequential_3/dense_11/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2)
'transformer_block_3/dropout_10/Identityæ
transformer_block_3/add_1AddV2=transformer_block_3/layer_normalization_6/batchnorm/add_1:z:00transformer_block_3/dropout_10/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_3/add_1Þ
Htransformer_block_3/layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_3/layer_normalization_7/moments/mean/reduction_indices±
6transformer_block_3/layer_normalization_7/moments/meanMeantransformer_block_3/add_1:z:0Qtransformer_block_3/layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(28
6transformer_block_3/layer_normalization_7/moments/mean
>transformer_block_3/layer_normalization_7/moments/StopGradientStopGradient?transformer_block_3/layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2@
>transformer_block_3/layer_normalization_7/moments/StopGradient½
Ctransformer_block_3/layer_normalization_7/moments/SquaredDifferenceSquaredDifferencetransformer_block_3/add_1:z:0Gtransformer_block_3/layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2E
Ctransformer_block_3/layer_normalization_7/moments/SquaredDifferenceæ
Ltransformer_block_3/layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_3/layer_normalization_7/moments/variance/reduction_indicesç
:transformer_block_3/layer_normalization_7/moments/varianceMeanGtransformer_block_3/layer_normalization_7/moments/SquaredDifference:z:0Utransformer_block_3/layer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2<
:transformer_block_3/layer_normalization_7/moments/variance»
9transformer_block_3/layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752;
9transformer_block_3/layer_normalization_7/batchnorm/add/yº
7transformer_block_3/layer_normalization_7/batchnorm/addAddV2Ctransformer_block_3/layer_normalization_7/moments/variance:output:0Btransformer_block_3/layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#29
7transformer_block_3/layer_normalization_7/batchnorm/addò
9transformer_block_3/layer_normalization_7/batchnorm/RsqrtRsqrt;transformer_block_3/layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2;
9transformer_block_3/layer_normalization_7/batchnorm/Rsqrt
Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_3_layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp¾
7transformer_block_3/layer_normalization_7/batchnorm/mulMul=transformer_block_3/layer_normalization_7/batchnorm/Rsqrt:y:0Ntransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 29
7transformer_block_3/layer_normalization_7/batchnorm/mul
9transformer_block_3/layer_normalization_7/batchnorm/mul_1Multransformer_block_3/add_1:z:0;transformer_block_3/layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_3/layer_normalization_7/batchnorm/mul_1±
9transformer_block_3/layer_normalization_7/batchnorm/mul_2Mul?transformer_block_3/layer_normalization_7/moments/mean:output:0;transformer_block_3/layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_3/layer_normalization_7/batchnorm/mul_2
Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_3_layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpº
7transformer_block_3/layer_normalization_7/batchnorm/subSubJtransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp:value:0=transformer_block_3/layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 29
7transformer_block_3/layer_normalization_7/batchnorm/sub±
9transformer_block_3/layer_normalization_7/batchnorm/add_1AddV2=transformer_block_3/layer_normalization_7/batchnorm/mul_1:z:0;transformer_block_3/layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_3/layer_normalization_7/batchnorm/add_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`  2
flatten_1/Const½
flatten_1/ReshapeReshape=transformer_block_3/layer_normalization_7/batchnorm/add_1:z:0flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
flatten_1/Reshapex
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis¾
concatenate_1/concatConcatV2flatten_1/Reshape:output:0inputs_1"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
concatenate_1/concat©
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	è@*
dtype02 
dense_12/MatMul/ReadVariableOp¥
dense_12/MatMulMatMulconcatenate_1/concat:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_12/MatMul§
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_12/BiasAdd/ReadVariableOp¥
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_12/Relu
dropout_11/IdentityIdentitydense_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_11/Identity¨
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_13/MatMul/ReadVariableOp¤
dense_13/MatMulMatMuldropout_11/Identity:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_13/MatMul§
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_13/BiasAdd/ReadVariableOp¥
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_13/BiasAdds
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_13/Relu
dropout_12/IdentityIdentitydense_13/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_12/Identity¨
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_14/MatMul/ReadVariableOp¤
dense_14/MatMulMatMuldropout_12/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_14/MatMul§
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_14/BiasAdd/ReadVariableOp¥
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_14/BiasAdds
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_14/Relu
dropout_13/IdentityIdentitydense_14/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_13/Identity¨
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_15/MatMul/ReadVariableOp¤
dense_15/MatMulMatMuldropout_13/Identity:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/MatMul§
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp¥
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/BiasAddÃ
IdentityIdentitydense_15/BiasAdd:output:0/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/conv1d/ExpandDims_1/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp<^token_and_position_embedding_1/embedding_2/embedding_lookup<^token_and_position_embedding_1/embedding_3/embedding_lookupC^transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpG^transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpC^transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpG^transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpO^transformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpY^transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_3/multi_head_attention_3/key/add/ReadVariableOpL^transformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpD^transformer_block_3/multi_head_attention_3/query/add/ReadVariableOpN^transformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpD^transformer_block_3/multi_head_attention_3/value/add/ReadVariableOpN^transformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpA^transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOpC^transformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOpA^transformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOpC^transformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ô
_input_shapesÂ
¿:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2d
0batch_normalization_3/batchnorm/ReadVariableOp_10batch_normalization_3/batchnorm/ReadVariableOp_12d
0batch_normalization_3/batchnorm/ReadVariableOp_20batch_normalization_3/batchnorm/ReadVariableOp_22h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2z
;token_and_position_embedding_1/embedding_2/embedding_lookup;token_and_position_embedding_1/embedding_2/embedding_lookup2z
;token_and_position_embedding_1/embedding_3/embedding_lookup;token_and_position_embedding_1/embedding_3/embedding_lookup2
Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpBtransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp2
Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpFtransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp2
Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpBtransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp2
Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpFtransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp2 
Ntransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpNtransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp2´
Xtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp2
Atransformer_block_3/multi_head_attention_3/key/add/ReadVariableOpAtransformer_block_3/multi_head_attention_3/key/add/ReadVariableOp2
Ktransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpKtransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp2
Ctransformer_block_3/multi_head_attention_3/query/add/ReadVariableOpCtransformer_block_3/multi_head_attention_3/query/add/ReadVariableOp2
Mtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpMtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp2
Ctransformer_block_3/multi_head_attention_3/value/add/ReadVariableOpCtransformer_block_3/multi_head_attention_3/value/add/ReadVariableOp2
Mtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpMtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp2
@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp2
Btransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOpBtransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp2
@transformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOp@transformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOp2
Btransformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOpBtransformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOp:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
¥
d
+__inference_dropout_11_layer_call_fn_549588

inputs
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_5472962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¼0
È
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_546681

inputs
assignmovingavg_546656
assignmovingavg_1_546662)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/546656*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_546656*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/546656*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/546656*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_546656AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/546656*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/546662*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_546662*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/546662*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/546662*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_546662AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/546662*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
ï
~
)__inference_dense_10_layer_call_fn_549886

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_5464022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ# ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
¼0
È
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_549115

inputs
assignmovingavg_549090
assignmovingavg_1_549096)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/549090*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_549090*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/549090*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/549090*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_549090AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/549090*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/549096*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_549096*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/549096*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/549096*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_549096AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/549096*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs


H__inference_sequential_3_layer_call_and_return_conditional_losses_546479
dense_10_input
dense_10_546468
dense_10_546470
dense_11_546473
dense_11_546475
identity¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall£
 dense_10/StatefulPartitionedCallStatefulPartitionedCalldense_10_inputdense_10_546468dense_10_546470*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_5464022"
 dense_10/StatefulPartitionedCall¾
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_546473dense_11_546475*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_5464482"
 dense_11/StatefulPartitionedCallÇ
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
(
_user_specified_namedense_10_input
ó0
È
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_549033

inputs
assignmovingavg_549008
assignmovingavg_1_549014)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/549008*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_549008*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/549008*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/549008*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_549008AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/549008*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/549014*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_549014*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/549014*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/549014*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_549014AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/549014*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1À
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
É
d
F__inference_dropout_13_layer_call_and_return_conditional_losses_549677

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
î	
Ý
D__inference_dense_14_layer_call_and_return_conditional_losses_549651

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

e
F__inference_dropout_13_layer_call_and_return_conditional_losses_549672

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÀ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*

seedY2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_548889

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1è
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ñ
ã
D__inference_dense_11_layer_call_and_return_conditional_losses_546448

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
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
Tensordot/GatherV2/axisÑ
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
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ#@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@
 
_user_specified_nameinputs
µ
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_549528

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ# :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
ºþ
Ú
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_546991

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
7layer_normalization_6_batchnorm_readvariableop_resource;
7sequential_3_dense_10_tensordot_readvariableop_resource9
5sequential_3_dense_10_biasadd_readvariableop_resource;
7sequential_3_dense_11_tensordot_readvariableop_resource9
5sequential_3_dense_11_biasadd_readvariableop_resource?
;layer_normalization_7_batchnorm_mul_readvariableop_resource;
7layer_normalization_7_batchnorm_readvariableop_resource
identity¢.layer_normalization_6/batchnorm/ReadVariableOp¢2layer_normalization_6/batchnorm/mul/ReadVariableOp¢.layer_normalization_7/batchnorm/ReadVariableOp¢2layer_normalization_7/batchnorm/mul/ReadVariableOp¢:multi_head_attention_3/attention_output/add/ReadVariableOp¢Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_3/key/add/ReadVariableOp¢7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_3/query/add/ReadVariableOp¢9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_3/value/add/ReadVariableOp¢9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp¢,sequential_3/dense_10/BiasAdd/ReadVariableOp¢.sequential_3/dense_10/Tensordot/ReadVariableOp¢,sequential_3/dense_11/BiasAdd/ReadVariableOp¢.sequential_3/dense_11/Tensordot/ReadVariableOpý
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_3/query/einsum/EinsumEinsuminputsAmulti_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_3/query/einsum/EinsumÛ
/multi_head_attention_3/query/add/ReadVariableOpReadVariableOp8multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/query/add/ReadVariableOpõ
 multi_head_attention_3/query/addAddV23multi_head_attention_3/query/einsum/Einsum:output:07multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_3/query/add÷
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_3/key/einsum/EinsumEinsuminputs?multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2*
(multi_head_attention_3/key/einsum/EinsumÕ
-multi_head_attention_3/key/add/ReadVariableOpReadVariableOp6multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_3/key/add/ReadVariableOpí
multi_head_attention_3/key/addAddV21multi_head_attention_3/key/einsum/Einsum:output:05multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
multi_head_attention_3/key/addý
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_3/value/einsum/EinsumEinsuminputsAmulti_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_3/value/einsum/EinsumÛ
/multi_head_attention_3/value/add/ReadVariableOpReadVariableOp8multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/value/add/ReadVariableOpõ
 multi_head_attention_3/value/addAddV23multi_head_attention_3/value/einsum/Einsum:output:07multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_3/value/add
multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_3/Mul/yÆ
multi_head_attention_3/MulMul$multi_head_attention_3/query/add:z:0%multi_head_attention_3/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_3/Mulü
$multi_head_attention_3/einsum/EinsumEinsum"multi_head_attention_3/key/add:z:0multi_head_attention_3/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2&
$multi_head_attention_3/einsum/EinsumÄ
&multi_head_attention_3/softmax/SoftmaxSoftmax-multi_head_attention_3/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2(
&multi_head_attention_3/softmax/Softmax¡
,multi_head_attention_3/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,multi_head_attention_3/dropout/dropout/Const
*multi_head_attention_3/dropout/dropout/MulMul0multi_head_attention_3/softmax/Softmax:softmax:05multi_head_attention_3/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2,
*multi_head_attention_3/dropout/dropout/Mul¼
,multi_head_attention_3/dropout/dropout/ShapeShape0multi_head_attention_3/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_3/dropout/dropout/Shape¥
Cmulti_head_attention_3/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_3/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
dtype0*

seedY2E
Cmulti_head_attention_3/dropout/dropout/random_uniform/RandomUniform³
5multi_head_attention_3/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_3/dropout/dropout/GreaterEqual/yÂ
3multi_head_attention_3/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_3/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_3/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##25
3multi_head_attention_3/dropout/dropout/GreaterEqualä
+multi_head_attention_3/dropout/dropout/CastCast7multi_head_attention_3/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2-
+multi_head_attention_3/dropout/dropout/Castþ
,multi_head_attention_3/dropout/dropout/Mul_1Mul.multi_head_attention_3/dropout/dropout/Mul:z:0/multi_head_attention_3/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2.
,multi_head_attention_3/dropout/dropout/Mul_1
&multi_head_attention_3/einsum_1/EinsumEinsum0multi_head_attention_3/dropout/dropout/Mul_1:z:0$multi_head_attention_3/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2(
&multi_head_attention_3/einsum_1/Einsum
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_3/attention_output/einsum/EinsumEinsum/multi_head_attention_3/einsum_1/Einsum:output:0Lmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe27
5multi_head_attention_3/attention_output/einsum/Einsumø
:multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_3/attention_output/add/ReadVariableOp
+multi_head_attention_3/attention_output/addAddV2>multi_head_attention_3/attention_output/einsum/Einsum:output:0Bmulti_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+multi_head_attention_3/attention_output/addw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_9/dropout/Const¾
dropout_9/dropout/MulMul/multi_head_attention_3/attention_output/add:z:0 dropout_9/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_9/dropout/Mul
dropout_9/dropout/ShapeShape/multi_head_attention_3/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shapeï
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype0*

seedY*
seed220
.dropout_9/dropout/random_uniform/RandomUniform
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_9/dropout/GreaterEqual/yê
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
dropout_9/dropout/GreaterEqual¡
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_9/dropout/Cast¦
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_9/dropout/Mul_1n
addAddV2inputsdropout_9/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¶
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_6/moments/mean/reduction_indicesß
"layer_normalization_6/moments/meanMeanadd:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2$
"layer_normalization_6/moments/meanË
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2,
*layer_normalization_6/moments/StopGradientë
/layer_normalization_6/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/layer_normalization_6/moments/SquaredDifference¾
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_6/moments/variance/reduction_indices
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2(
&layer_normalization_6/moments/variance
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752'
%layer_normalization_6/batchnorm/add/yê
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#layer_normalization_6/batchnorm/add¶
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2'
%layer_normalization_6/batchnorm/Rsqrtà
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_6/batchnorm/mul/ReadVariableOpî
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_6/batchnorm/mul½
%layer_normalization_6/batchnorm/mul_1Muladd:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_6/batchnorm/mul_1á
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_6/batchnorm/mul_2Ô
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_6/batchnorm/ReadVariableOpê
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_6/batchnorm/subá
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_6/batchnorm/add_1Ø
.sequential_3/dense_10/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_10_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_3/dense_10/Tensordot/ReadVariableOp
$sequential_3/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_3/dense_10/Tensordot/axes
$sequential_3/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_3/dense_10/Tensordot/free§
%sequential_3/dense_10/Tensordot/ShapeShape)layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/Shape 
-sequential_3/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/GatherV2/axis¿
(sequential_3/dense_10/Tensordot/GatherV2GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/free:output:06sequential_3/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/GatherV2¤
/sequential_3/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_3/dense_10/Tensordot/GatherV2_1/axisÅ
*sequential_3/dense_10/Tensordot/GatherV2_1GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/axes:output:08sequential_3/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_3/dense_10/Tensordot/GatherV2_1
%sequential_3/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_3/dense_10/Tensordot/ConstØ
$sequential_3/dense_10/Tensordot/ProdProd1sequential_3/dense_10/Tensordot/GatherV2:output:0.sequential_3/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_3/dense_10/Tensordot/Prod
'sequential_3/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_1à
&sequential_3/dense_10/Tensordot/Prod_1Prod3sequential_3/dense_10/Tensordot/GatherV2_1:output:00sequential_3/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_3/dense_10/Tensordot/Prod_1
+sequential_3/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_3/dense_10/Tensordot/concat/axis
&sequential_3/dense_10/Tensordot/concatConcatV2-sequential_3/dense_10/Tensordot/free:output:0-sequential_3/dense_10/Tensordot/axes:output:04sequential_3/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_3/dense_10/Tensordot/concatä
%sequential_3/dense_10/Tensordot/stackPack-sequential_3/dense_10/Tensordot/Prod:output:0/sequential_3/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/stackõ
)sequential_3/dense_10/Tensordot/transpose	Transpose)layer_normalization_6/batchnorm/add_1:z:0/sequential_3/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2+
)sequential_3/dense_10/Tensordot/transpose÷
'sequential_3/dense_10/Tensordot/ReshapeReshape-sequential_3/dense_10/Tensordot/transpose:y:0.sequential_3/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_3/dense_10/Tensordot/Reshapeö
&sequential_3/dense_10/Tensordot/MatMulMatMul0sequential_3/dense_10/Tensordot/Reshape:output:06sequential_3/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential_3/dense_10/Tensordot/MatMul
'sequential_3/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_3/dense_10/Tensordot/Const_2 
-sequential_3/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/concat_1/axis«
(sequential_3/dense_10/Tensordot/concat_1ConcatV21sequential_3/dense_10/Tensordot/GatherV2:output:00sequential_3/dense_10/Tensordot/Const_2:output:06sequential_3/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/concat_1è
sequential_3/dense_10/TensordotReshape0sequential_3/dense_10/Tensordot/MatMul:product:01sequential_3/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2!
sequential_3/dense_10/TensordotÎ
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOpß
sequential_3/dense_10/BiasAddBiasAdd(sequential_3/dense_10/Tensordot:output:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_3/dense_10/BiasAdd
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_3/dense_10/ReluØ
.sequential_3/dense_11/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_11_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_3/dense_11/Tensordot/ReadVariableOp
$sequential_3/dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_3/dense_11/Tensordot/axes
$sequential_3/dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_3/dense_11/Tensordot/free¦
%sequential_3/dense_11/Tensordot/ShapeShape(sequential_3/dense_10/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dense_11/Tensordot/Shape 
-sequential_3/dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_11/Tensordot/GatherV2/axis¿
(sequential_3/dense_11/Tensordot/GatherV2GatherV2.sequential_3/dense_11/Tensordot/Shape:output:0-sequential_3/dense_11/Tensordot/free:output:06sequential_3/dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_3/dense_11/Tensordot/GatherV2¤
/sequential_3/dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_3/dense_11/Tensordot/GatherV2_1/axisÅ
*sequential_3/dense_11/Tensordot/GatherV2_1GatherV2.sequential_3/dense_11/Tensordot/Shape:output:0-sequential_3/dense_11/Tensordot/axes:output:08sequential_3/dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_3/dense_11/Tensordot/GatherV2_1
%sequential_3/dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_3/dense_11/Tensordot/ConstØ
$sequential_3/dense_11/Tensordot/ProdProd1sequential_3/dense_11/Tensordot/GatherV2:output:0.sequential_3/dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_3/dense_11/Tensordot/Prod
'sequential_3/dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_11/Tensordot/Const_1à
&sequential_3/dense_11/Tensordot/Prod_1Prod3sequential_3/dense_11/Tensordot/GatherV2_1:output:00sequential_3/dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_3/dense_11/Tensordot/Prod_1
+sequential_3/dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_3/dense_11/Tensordot/concat/axis
&sequential_3/dense_11/Tensordot/concatConcatV2-sequential_3/dense_11/Tensordot/free:output:0-sequential_3/dense_11/Tensordot/axes:output:04sequential_3/dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_3/dense_11/Tensordot/concatä
%sequential_3/dense_11/Tensordot/stackPack-sequential_3/dense_11/Tensordot/Prod:output:0/sequential_3/dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_11/Tensordot/stackô
)sequential_3/dense_11/Tensordot/transpose	Transpose(sequential_3/dense_10/Relu:activations:0/sequential_3/dense_11/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2+
)sequential_3/dense_11/Tensordot/transpose÷
'sequential_3/dense_11/Tensordot/ReshapeReshape-sequential_3/dense_11/Tensordot/transpose:y:0.sequential_3/dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_3/dense_11/Tensordot/Reshapeö
&sequential_3/dense_11/Tensordot/MatMulMatMul0sequential_3/dense_11/Tensordot/Reshape:output:06sequential_3/dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_3/dense_11/Tensordot/MatMul
'sequential_3/dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_11/Tensordot/Const_2 
-sequential_3/dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_11/Tensordot/concat_1/axis«
(sequential_3/dense_11/Tensordot/concat_1ConcatV21sequential_3/dense_11/Tensordot/GatherV2:output:00sequential_3/dense_11/Tensordot/Const_2:output:06sequential_3/dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_3/dense_11/Tensordot/concat_1è
sequential_3/dense_11/TensordotReshape0sequential_3/dense_11/Tensordot/MatMul:product:01sequential_3/dense_11/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
sequential_3/dense_11/TensordotÎ
,sequential_3/dense_11/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/dense_11/BiasAdd/ReadVariableOpß
sequential_3/dense_11/BiasAddBiasAdd(sequential_3/dense_11/Tensordot:output:04sequential_3/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
sequential_3/dense_11/BiasAddy
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_10/dropout/Const¸
dropout_10/dropout/MulMul&sequential_3/dense_11/BiasAdd:output:0!dropout_10/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_10/dropout/Mul
dropout_10/dropout/ShapeShape&sequential_3/dense_11/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_10/dropout/Shapeò
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype0*

seedY*
seed221
/dropout_10/dropout/random_uniform/RandomUniform
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_10/dropout/GreaterEqual/yî
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
dropout_10/dropout/GreaterEqual¤
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_10/dropout/Castª
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_10/dropout/Mul_1
add_1AddV2)layer_normalization_6/batchnorm/add_1:z:0dropout_10/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¶
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_7/moments/mean/reduction_indicesá
"layer_normalization_7/moments/meanMean	add_1:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2$
"layer_normalization_7/moments/meanË
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2,
*layer_normalization_7/moments/StopGradientí
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/layer_normalization_7/moments/SquaredDifference¾
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_7/moments/variance/reduction_indices
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2(
&layer_normalization_7/moments/variance
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752'
%layer_normalization_7/batchnorm/add/yê
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#layer_normalization_7/batchnorm/add¶
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2'
%layer_normalization_7/batchnorm/Rsqrtà
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_7/batchnorm/mul/ReadVariableOpî
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_7/batchnorm/mul¿
%layer_normalization_7/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_7/batchnorm/mul_1á
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_7/batchnorm/mul_2Ô
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_7/batchnorm/ReadVariableOpê
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_7/batchnorm/subá
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_7/batchnorm/add_1×
IdentityIdentity)layer_normalization_7/batchnorm/add_1:z:0/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp;^multi_head_attention_3/attention_output/add/ReadVariableOpE^multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_3/key/add/ReadVariableOp8^multi_head_attention_3/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/query/add/ReadVariableOp:^multi_head_attention_3/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/value/add/ReadVariableOp:^multi_head_attention_3/value/einsum/Einsum/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp/^sequential_3/dense_10/Tensordot/ReadVariableOp-^sequential_3/dense_11/BiasAdd/ReadVariableOp/^sequential_3/dense_11/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_3/attention_output/add/ReadVariableOp:multi_head_attention_3/attention_output/add/ReadVariableOp2
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_3/key/add/ReadVariableOp-multi_head_attention_3/key/add/ReadVariableOp2r
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/query/add/ReadVariableOp/multi_head_attention_3/query/add/ReadVariableOp2v
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/value/add/ReadVariableOp/multi_head_attention_3/value/add/ReadVariableOp2v
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2`
.sequential_3/dense_10/Tensordot/ReadVariableOp.sequential_3/dense_10/Tensordot/ReadVariableOp2\
,sequential_3/dense_11/BiasAdd/ReadVariableOp,sequential_3/dense_11/BiasAdd/ReadVariableOp2`
.sequential_3/dense_11/Tensordot/ReadVariableOp.sequential_3/dense_11/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs

e
F__inference_dropout_12_layer_call_and_return_conditional_losses_549625

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÀ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*

seedY2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
è

Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_549135

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs

G
+__inference_dropout_13_layer_call_fn_549687

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_5474152
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

÷
D__inference_conv1d_3_layer_call_and_return_conditional_losses_548824

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
conv1d/ExpandDims¹
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:	 *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¸
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:	 2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÞ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
 
_user_specified_nameinputs

Â
(__inference_model_1_layer_call_fn_547920
input_3
input_4
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

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&'*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_5478412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ô
_input_shapesÂ
¿:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_3:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
¸Ý
Ú
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_547118

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
7layer_normalization_6_batchnorm_readvariableop_resource;
7sequential_3_dense_10_tensordot_readvariableop_resource9
5sequential_3_dense_10_biasadd_readvariableop_resource;
7sequential_3_dense_11_tensordot_readvariableop_resource9
5sequential_3_dense_11_biasadd_readvariableop_resource?
;layer_normalization_7_batchnorm_mul_readvariableop_resource;
7layer_normalization_7_batchnorm_readvariableop_resource
identity¢.layer_normalization_6/batchnorm/ReadVariableOp¢2layer_normalization_6/batchnorm/mul/ReadVariableOp¢.layer_normalization_7/batchnorm/ReadVariableOp¢2layer_normalization_7/batchnorm/mul/ReadVariableOp¢:multi_head_attention_3/attention_output/add/ReadVariableOp¢Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_3/key/add/ReadVariableOp¢7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_3/query/add/ReadVariableOp¢9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_3/value/add/ReadVariableOp¢9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp¢,sequential_3/dense_10/BiasAdd/ReadVariableOp¢.sequential_3/dense_10/Tensordot/ReadVariableOp¢,sequential_3/dense_11/BiasAdd/ReadVariableOp¢.sequential_3/dense_11/Tensordot/ReadVariableOpý
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_3/query/einsum/EinsumEinsuminputsAmulti_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_3/query/einsum/EinsumÛ
/multi_head_attention_3/query/add/ReadVariableOpReadVariableOp8multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/query/add/ReadVariableOpõ
 multi_head_attention_3/query/addAddV23multi_head_attention_3/query/einsum/Einsum:output:07multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_3/query/add÷
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_3/key/einsum/EinsumEinsuminputs?multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2*
(multi_head_attention_3/key/einsum/EinsumÕ
-multi_head_attention_3/key/add/ReadVariableOpReadVariableOp6multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_3/key/add/ReadVariableOpí
multi_head_attention_3/key/addAddV21multi_head_attention_3/key/einsum/Einsum:output:05multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
multi_head_attention_3/key/addý
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_3/value/einsum/EinsumEinsuminputsAmulti_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_3/value/einsum/EinsumÛ
/multi_head_attention_3/value/add/ReadVariableOpReadVariableOp8multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/value/add/ReadVariableOpõ
 multi_head_attention_3/value/addAddV23multi_head_attention_3/value/einsum/Einsum:output:07multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_3/value/add
multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_3/Mul/yÆ
multi_head_attention_3/MulMul$multi_head_attention_3/query/add:z:0%multi_head_attention_3/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_3/Mulü
$multi_head_attention_3/einsum/EinsumEinsum"multi_head_attention_3/key/add:z:0multi_head_attention_3/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2&
$multi_head_attention_3/einsum/EinsumÄ
&multi_head_attention_3/softmax/SoftmaxSoftmax-multi_head_attention_3/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2(
&multi_head_attention_3/softmax/SoftmaxÊ
'multi_head_attention_3/dropout/IdentityIdentity0multi_head_attention_3/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2)
'multi_head_attention_3/dropout/Identity
&multi_head_attention_3/einsum_1/EinsumEinsum0multi_head_attention_3/dropout/Identity:output:0$multi_head_attention_3/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2(
&multi_head_attention_3/einsum_1/Einsum
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_3/attention_output/einsum/EinsumEinsum/multi_head_attention_3/einsum_1/Einsum:output:0Lmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe27
5multi_head_attention_3/attention_output/einsum/Einsumø
:multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_3/attention_output/add/ReadVariableOp
+multi_head_attention_3/attention_output/addAddV2>multi_head_attention_3/attention_output/einsum/Einsum:output:0Bmulti_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+multi_head_attention_3/attention_output/add
dropout_9/IdentityIdentity/multi_head_attention_3/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_9/Identityn
addAddV2inputsdropout_9/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¶
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_6/moments/mean/reduction_indicesß
"layer_normalization_6/moments/meanMeanadd:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2$
"layer_normalization_6/moments/meanË
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2,
*layer_normalization_6/moments/StopGradientë
/layer_normalization_6/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/layer_normalization_6/moments/SquaredDifference¾
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_6/moments/variance/reduction_indices
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2(
&layer_normalization_6/moments/variance
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752'
%layer_normalization_6/batchnorm/add/yê
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#layer_normalization_6/batchnorm/add¶
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2'
%layer_normalization_6/batchnorm/Rsqrtà
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_6/batchnorm/mul/ReadVariableOpî
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_6/batchnorm/mul½
%layer_normalization_6/batchnorm/mul_1Muladd:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_6/batchnorm/mul_1á
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_6/batchnorm/mul_2Ô
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_6/batchnorm/ReadVariableOpê
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_6/batchnorm/subá
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_6/batchnorm/add_1Ø
.sequential_3/dense_10/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_10_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_3/dense_10/Tensordot/ReadVariableOp
$sequential_3/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_3/dense_10/Tensordot/axes
$sequential_3/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_3/dense_10/Tensordot/free§
%sequential_3/dense_10/Tensordot/ShapeShape)layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/Shape 
-sequential_3/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/GatherV2/axis¿
(sequential_3/dense_10/Tensordot/GatherV2GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/free:output:06sequential_3/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/GatherV2¤
/sequential_3/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_3/dense_10/Tensordot/GatherV2_1/axisÅ
*sequential_3/dense_10/Tensordot/GatherV2_1GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/axes:output:08sequential_3/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_3/dense_10/Tensordot/GatherV2_1
%sequential_3/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_3/dense_10/Tensordot/ConstØ
$sequential_3/dense_10/Tensordot/ProdProd1sequential_3/dense_10/Tensordot/GatherV2:output:0.sequential_3/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_3/dense_10/Tensordot/Prod
'sequential_3/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_1à
&sequential_3/dense_10/Tensordot/Prod_1Prod3sequential_3/dense_10/Tensordot/GatherV2_1:output:00sequential_3/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_3/dense_10/Tensordot/Prod_1
+sequential_3/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_3/dense_10/Tensordot/concat/axis
&sequential_3/dense_10/Tensordot/concatConcatV2-sequential_3/dense_10/Tensordot/free:output:0-sequential_3/dense_10/Tensordot/axes:output:04sequential_3/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_3/dense_10/Tensordot/concatä
%sequential_3/dense_10/Tensordot/stackPack-sequential_3/dense_10/Tensordot/Prod:output:0/sequential_3/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/stackõ
)sequential_3/dense_10/Tensordot/transpose	Transpose)layer_normalization_6/batchnorm/add_1:z:0/sequential_3/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2+
)sequential_3/dense_10/Tensordot/transpose÷
'sequential_3/dense_10/Tensordot/ReshapeReshape-sequential_3/dense_10/Tensordot/transpose:y:0.sequential_3/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_3/dense_10/Tensordot/Reshapeö
&sequential_3/dense_10/Tensordot/MatMulMatMul0sequential_3/dense_10/Tensordot/Reshape:output:06sequential_3/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential_3/dense_10/Tensordot/MatMul
'sequential_3/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_3/dense_10/Tensordot/Const_2 
-sequential_3/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/concat_1/axis«
(sequential_3/dense_10/Tensordot/concat_1ConcatV21sequential_3/dense_10/Tensordot/GatherV2:output:00sequential_3/dense_10/Tensordot/Const_2:output:06sequential_3/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/concat_1è
sequential_3/dense_10/TensordotReshape0sequential_3/dense_10/Tensordot/MatMul:product:01sequential_3/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2!
sequential_3/dense_10/TensordotÎ
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOpß
sequential_3/dense_10/BiasAddBiasAdd(sequential_3/dense_10/Tensordot:output:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_3/dense_10/BiasAdd
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_3/dense_10/ReluØ
.sequential_3/dense_11/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_11_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_3/dense_11/Tensordot/ReadVariableOp
$sequential_3/dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_3/dense_11/Tensordot/axes
$sequential_3/dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_3/dense_11/Tensordot/free¦
%sequential_3/dense_11/Tensordot/ShapeShape(sequential_3/dense_10/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dense_11/Tensordot/Shape 
-sequential_3/dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_11/Tensordot/GatherV2/axis¿
(sequential_3/dense_11/Tensordot/GatherV2GatherV2.sequential_3/dense_11/Tensordot/Shape:output:0-sequential_3/dense_11/Tensordot/free:output:06sequential_3/dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_3/dense_11/Tensordot/GatherV2¤
/sequential_3/dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_3/dense_11/Tensordot/GatherV2_1/axisÅ
*sequential_3/dense_11/Tensordot/GatherV2_1GatherV2.sequential_3/dense_11/Tensordot/Shape:output:0-sequential_3/dense_11/Tensordot/axes:output:08sequential_3/dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_3/dense_11/Tensordot/GatherV2_1
%sequential_3/dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_3/dense_11/Tensordot/ConstØ
$sequential_3/dense_11/Tensordot/ProdProd1sequential_3/dense_11/Tensordot/GatherV2:output:0.sequential_3/dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_3/dense_11/Tensordot/Prod
'sequential_3/dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_11/Tensordot/Const_1à
&sequential_3/dense_11/Tensordot/Prod_1Prod3sequential_3/dense_11/Tensordot/GatherV2_1:output:00sequential_3/dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_3/dense_11/Tensordot/Prod_1
+sequential_3/dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_3/dense_11/Tensordot/concat/axis
&sequential_3/dense_11/Tensordot/concatConcatV2-sequential_3/dense_11/Tensordot/free:output:0-sequential_3/dense_11/Tensordot/axes:output:04sequential_3/dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_3/dense_11/Tensordot/concatä
%sequential_3/dense_11/Tensordot/stackPack-sequential_3/dense_11/Tensordot/Prod:output:0/sequential_3/dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_11/Tensordot/stackô
)sequential_3/dense_11/Tensordot/transpose	Transpose(sequential_3/dense_10/Relu:activations:0/sequential_3/dense_11/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2+
)sequential_3/dense_11/Tensordot/transpose÷
'sequential_3/dense_11/Tensordot/ReshapeReshape-sequential_3/dense_11/Tensordot/transpose:y:0.sequential_3/dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_3/dense_11/Tensordot/Reshapeö
&sequential_3/dense_11/Tensordot/MatMulMatMul0sequential_3/dense_11/Tensordot/Reshape:output:06sequential_3/dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_3/dense_11/Tensordot/MatMul
'sequential_3/dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_11/Tensordot/Const_2 
-sequential_3/dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_11/Tensordot/concat_1/axis«
(sequential_3/dense_11/Tensordot/concat_1ConcatV21sequential_3/dense_11/Tensordot/GatherV2:output:00sequential_3/dense_11/Tensordot/Const_2:output:06sequential_3/dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_3/dense_11/Tensordot/concat_1è
sequential_3/dense_11/TensordotReshape0sequential_3/dense_11/Tensordot/MatMul:product:01sequential_3/dense_11/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
sequential_3/dense_11/TensordotÎ
,sequential_3/dense_11/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/dense_11/BiasAdd/ReadVariableOpß
sequential_3/dense_11/BiasAddBiasAdd(sequential_3/dense_11/Tensordot:output:04sequential_3/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
sequential_3/dense_11/BiasAdd
dropout_10/IdentityIdentity&sequential_3/dense_11/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_10/Identity
add_1AddV2)layer_normalization_6/batchnorm/add_1:z:0dropout_10/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¶
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_7/moments/mean/reduction_indicesá
"layer_normalization_7/moments/meanMean	add_1:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2$
"layer_normalization_7/moments/meanË
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2,
*layer_normalization_7/moments/StopGradientí
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/layer_normalization_7/moments/SquaredDifference¾
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_7/moments/variance/reduction_indices
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2(
&layer_normalization_7/moments/variance
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752'
%layer_normalization_7/batchnorm/add/yê
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#layer_normalization_7/batchnorm/add¶
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2'
%layer_normalization_7/batchnorm/Rsqrtà
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_7/batchnorm/mul/ReadVariableOpî
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_7/batchnorm/mul¿
%layer_normalization_7/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_7/batchnorm/mul_1á
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_7/batchnorm/mul_2Ô
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_7/batchnorm/ReadVariableOpê
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_7/batchnorm/subá
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_7/batchnorm/add_1×
IdentityIdentity)layer_normalization_7/batchnorm/add_1:z:0/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp;^multi_head_attention_3/attention_output/add/ReadVariableOpE^multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_3/key/add/ReadVariableOp8^multi_head_attention_3/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/query/add/ReadVariableOp:^multi_head_attention_3/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/value/add/ReadVariableOp:^multi_head_attention_3/value/einsum/Einsum/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp/^sequential_3/dense_10/Tensordot/ReadVariableOp-^sequential_3/dense_11/BiasAdd/ReadVariableOp/^sequential_3/dense_11/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_3/attention_output/add/ReadVariableOp:multi_head_attention_3/attention_output/add/ReadVariableOp2
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_3/key/add/ReadVariableOp-multi_head_attention_3/key/add/ReadVariableOp2r
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/query/add/ReadVariableOp/multi_head_attention_3/query/add/ReadVariableOp2v
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/value/add/ReadVariableOp/multi_head_attention_3/value/add/ReadVariableOp2v
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2`
.sequential_3/dense_10/Tensordot/ReadVariableOp.sequential_3/dense_10/Tensordot/ReadVariableOp2\
,sequential_3/dense_11/BiasAdd/ReadVariableOp,sequential_3/dense_11/BiasAdd/ReadVariableOp2`
.sequential_3/dense_11/Tensordot/ReadVariableOp.sequential_3/dense_11/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
	
Ý
D__inference_dense_15_layer_call_and_return_conditional_losses_547438

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
º
s
I__inference_concatenate_1_layer_call_and_return_conditional_losses_547248

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿà:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼0
È
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_548951

inputs
assignmovingavg_548926
assignmovingavg_1_548932)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/548926*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_548926*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/548926*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/548926*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_548926AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/548926*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/548932*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_548932*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/548932*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/548932*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_548932AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/548932*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
Ù`
ë
C__inference_model_1_layer_call_and_return_conditional_losses_547841

inputs
inputs_1)
%token_and_position_embedding_1_547745)
%token_and_position_embedding_1_547747
conv1d_2_547750
conv1d_2_547752
conv1d_3_547756
conv1d_3_547758 
batch_normalization_2_547763 
batch_normalization_2_547765 
batch_normalization_2_547767 
batch_normalization_2_547769 
batch_normalization_3_547772 
batch_normalization_3_547774 
batch_normalization_3_547776 
batch_normalization_3_547778
transformer_block_3_547782
transformer_block_3_547784
transformer_block_3_547786
transformer_block_3_547788
transformer_block_3_547790
transformer_block_3_547792
transformer_block_3_547794
transformer_block_3_547796
transformer_block_3_547798
transformer_block_3_547800
transformer_block_3_547802
transformer_block_3_547804
transformer_block_3_547806
transformer_block_3_547808
transformer_block_3_547810
transformer_block_3_547812
dense_12_547817
dense_12_547819
dense_13_547823
dense_13_547825
dense_14_547829
dense_14_547831
dense_15_547835
dense_15_547837
identity¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCall¢ conv1d_3/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢6token_and_position_embedding_1/StatefulPartitionedCall¢+transformer_block_3/StatefulPartitionedCall
6token_and_position_embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_1_547745%token_and_position_embedding_1_547747*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_54656328
6token_and_position_embedding_1/StatefulPartitionedCallÖ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0conv1d_2_547750conv1d_2_547752*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_5465952"
 conv1d_2/StatefulPartitionedCall¡
#average_pooling1d_3/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_5460512%
#average_pooling1d_3/PartitionedCallÂ
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_3/PartitionedCall:output:0conv1d_3_547756conv1d_3_547758*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_5466282"
 conv1d_3/StatefulPartitionedCallµ
#average_pooling1d_5/PartitionedCallPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_5460812%
#average_pooling1d_5/PartitionedCall
#average_pooling1d_4/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_5460662%
#average_pooling1d_4/PartitionedCallÂ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_4/PartitionedCall:output:0batch_normalization_2_547763batch_normalization_2_547765batch_normalization_2_547767batch_normalization_2_547769*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5467012/
-batch_normalization_2/StatefulPartitionedCallÂ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_5/PartitionedCall:output:0batch_normalization_3_547772batch_normalization_3_547774batch_normalization_3_547776batch_normalization_3_547778*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5467922/
-batch_normalization_3/StatefulPartitionedCall»
add_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:06batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_5468342
add_1/PartitionedCall
+transformer_block_3/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0transformer_block_3_547782transformer_block_3_547784transformer_block_3_547786transformer_block_3_547788transformer_block_3_547790transformer_block_3_547792transformer_block_3_547794transformer_block_3_547796transformer_block_3_547798transformer_block_3_547800transformer_block_3_547802transformer_block_3_547804transformer_block_3_547806transformer_block_3_547808transformer_block_3_547810transformer_block_3_547812*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_5471182-
+transformer_block_3/StatefulPartitionedCall
flatten_1/PartitionedCallPartitionedCall4transformer_block_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_5472332
flatten_1/PartitionedCall
concatenate_1/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_5472482
concatenate_1/PartitionedCall·
 dense_12/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_12_547817dense_12_547819*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_5472682"
 dense_12/StatefulPartitionedCall
dropout_11/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_5473012
dropout_11/PartitionedCall´
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_13_547823dense_13_547825*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_5473252"
 dense_13/StatefulPartitionedCall
dropout_12/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_5473582
dropout_12/PartitionedCall´
 dense_14/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0dense_14_547829dense_14_547831*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_5473822"
 dense_14/StatefulPartitionedCall
dropout_13/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_5474152
dropout_13/PartitionedCall´
 dense_15/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_15_547835dense_15_547837*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_5474382"
 dense_15/StatefulPartitionedCall
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall7^token_and_position_embedding_1/StatefulPartitionedCall,^transformer_block_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ô
_input_shapesÂ
¿:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2p
6token_and_position_embedding_1/StatefulPartitionedCall6token_and_position_embedding_1/StatefulPartitionedCall2Z
+transformer_block_3/StatefulPartitionedCall+transformer_block_3/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
u
I__inference_concatenate_1_layer_call_and_return_conditional_losses_549540
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿà:ÿÿÿÿÿÿÿÿÿ:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
¡
F
*__inference_flatten_1_layer_call_fn_549533

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_5472332
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ# :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs

e
F__inference_dropout_12_layer_call_and_return_conditional_losses_547353

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÀ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*

seedY2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¸
 
-__inference_sequential_3_layer_call_fn_549833

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_5464962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs

Ä
(__inference_model_1_layer_call_fn_548750
inputs_0
inputs_1
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

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&'*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_5478412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ô
_input_shapesÂ
¿:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1

e
F__inference_dropout_11_layer_call_and_return_conditional_losses_547296

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÀ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*

seedY2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_549053

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1è
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

G
+__inference_dropout_12_layer_call_fn_549640

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_5473582
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
é

H__inference_sequential_3_layer_call_and_return_conditional_losses_546523

inputs
dense_10_546512
dense_10_546514
dense_11_546517
dense_11_546519
identity¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinputsdense_10_546512dense_10_546514*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_5464022"
 dense_10/StatefulPartitionedCall¾
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_546517dense_11_546519*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_5464482"
 dense_11/StatefulPartitionedCallÇ
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
Ð
¨
-__inference_sequential_3_layer_call_fn_546507
dense_10_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCalldense_10_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_5464962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
(
_user_specified_namedense_10_input
¸Ý
Ú
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_549448

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
7layer_normalization_6_batchnorm_readvariableop_resource;
7sequential_3_dense_10_tensordot_readvariableop_resource9
5sequential_3_dense_10_biasadd_readvariableop_resource;
7sequential_3_dense_11_tensordot_readvariableop_resource9
5sequential_3_dense_11_biasadd_readvariableop_resource?
;layer_normalization_7_batchnorm_mul_readvariableop_resource;
7layer_normalization_7_batchnorm_readvariableop_resource
identity¢.layer_normalization_6/batchnorm/ReadVariableOp¢2layer_normalization_6/batchnorm/mul/ReadVariableOp¢.layer_normalization_7/batchnorm/ReadVariableOp¢2layer_normalization_7/batchnorm/mul/ReadVariableOp¢:multi_head_attention_3/attention_output/add/ReadVariableOp¢Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_3/key/add/ReadVariableOp¢7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_3/query/add/ReadVariableOp¢9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_3/value/add/ReadVariableOp¢9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp¢,sequential_3/dense_10/BiasAdd/ReadVariableOp¢.sequential_3/dense_10/Tensordot/ReadVariableOp¢,sequential_3/dense_11/BiasAdd/ReadVariableOp¢.sequential_3/dense_11/Tensordot/ReadVariableOpý
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_3/query/einsum/EinsumEinsuminputsAmulti_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_3/query/einsum/EinsumÛ
/multi_head_attention_3/query/add/ReadVariableOpReadVariableOp8multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/query/add/ReadVariableOpõ
 multi_head_attention_3/query/addAddV23multi_head_attention_3/query/einsum/Einsum:output:07multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_3/query/add÷
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_3/key/einsum/EinsumEinsuminputs?multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2*
(multi_head_attention_3/key/einsum/EinsumÕ
-multi_head_attention_3/key/add/ReadVariableOpReadVariableOp6multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_3/key/add/ReadVariableOpí
multi_head_attention_3/key/addAddV21multi_head_attention_3/key/einsum/Einsum:output:05multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
multi_head_attention_3/key/addý
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_3/value/einsum/EinsumEinsuminputsAmulti_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_3/value/einsum/EinsumÛ
/multi_head_attention_3/value/add/ReadVariableOpReadVariableOp8multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/value/add/ReadVariableOpõ
 multi_head_attention_3/value/addAddV23multi_head_attention_3/value/einsum/Einsum:output:07multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_3/value/add
multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_3/Mul/yÆ
multi_head_attention_3/MulMul$multi_head_attention_3/query/add:z:0%multi_head_attention_3/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_3/Mulü
$multi_head_attention_3/einsum/EinsumEinsum"multi_head_attention_3/key/add:z:0multi_head_attention_3/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2&
$multi_head_attention_3/einsum/EinsumÄ
&multi_head_attention_3/softmax/SoftmaxSoftmax-multi_head_attention_3/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2(
&multi_head_attention_3/softmax/SoftmaxÊ
'multi_head_attention_3/dropout/IdentityIdentity0multi_head_attention_3/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2)
'multi_head_attention_3/dropout/Identity
&multi_head_attention_3/einsum_1/EinsumEinsum0multi_head_attention_3/dropout/Identity:output:0$multi_head_attention_3/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2(
&multi_head_attention_3/einsum_1/Einsum
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_3/attention_output/einsum/EinsumEinsum/multi_head_attention_3/einsum_1/Einsum:output:0Lmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe27
5multi_head_attention_3/attention_output/einsum/Einsumø
:multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_3/attention_output/add/ReadVariableOp
+multi_head_attention_3/attention_output/addAddV2>multi_head_attention_3/attention_output/einsum/Einsum:output:0Bmulti_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+multi_head_attention_3/attention_output/add
dropout_9/IdentityIdentity/multi_head_attention_3/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_9/Identityn
addAddV2inputsdropout_9/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¶
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_6/moments/mean/reduction_indicesß
"layer_normalization_6/moments/meanMeanadd:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2$
"layer_normalization_6/moments/meanË
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2,
*layer_normalization_6/moments/StopGradientë
/layer_normalization_6/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/layer_normalization_6/moments/SquaredDifference¾
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_6/moments/variance/reduction_indices
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2(
&layer_normalization_6/moments/variance
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752'
%layer_normalization_6/batchnorm/add/yê
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#layer_normalization_6/batchnorm/add¶
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2'
%layer_normalization_6/batchnorm/Rsqrtà
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_6/batchnorm/mul/ReadVariableOpî
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_6/batchnorm/mul½
%layer_normalization_6/batchnorm/mul_1Muladd:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_6/batchnorm/mul_1á
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_6/batchnorm/mul_2Ô
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_6/batchnorm/ReadVariableOpê
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_6/batchnorm/subá
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_6/batchnorm/add_1Ø
.sequential_3/dense_10/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_10_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_3/dense_10/Tensordot/ReadVariableOp
$sequential_3/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_3/dense_10/Tensordot/axes
$sequential_3/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_3/dense_10/Tensordot/free§
%sequential_3/dense_10/Tensordot/ShapeShape)layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/Shape 
-sequential_3/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/GatherV2/axis¿
(sequential_3/dense_10/Tensordot/GatherV2GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/free:output:06sequential_3/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/GatherV2¤
/sequential_3/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_3/dense_10/Tensordot/GatherV2_1/axisÅ
*sequential_3/dense_10/Tensordot/GatherV2_1GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/axes:output:08sequential_3/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_3/dense_10/Tensordot/GatherV2_1
%sequential_3/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_3/dense_10/Tensordot/ConstØ
$sequential_3/dense_10/Tensordot/ProdProd1sequential_3/dense_10/Tensordot/GatherV2:output:0.sequential_3/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_3/dense_10/Tensordot/Prod
'sequential_3/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_1à
&sequential_3/dense_10/Tensordot/Prod_1Prod3sequential_3/dense_10/Tensordot/GatherV2_1:output:00sequential_3/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_3/dense_10/Tensordot/Prod_1
+sequential_3/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_3/dense_10/Tensordot/concat/axis
&sequential_3/dense_10/Tensordot/concatConcatV2-sequential_3/dense_10/Tensordot/free:output:0-sequential_3/dense_10/Tensordot/axes:output:04sequential_3/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_3/dense_10/Tensordot/concatä
%sequential_3/dense_10/Tensordot/stackPack-sequential_3/dense_10/Tensordot/Prod:output:0/sequential_3/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/stackõ
)sequential_3/dense_10/Tensordot/transpose	Transpose)layer_normalization_6/batchnorm/add_1:z:0/sequential_3/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2+
)sequential_3/dense_10/Tensordot/transpose÷
'sequential_3/dense_10/Tensordot/ReshapeReshape-sequential_3/dense_10/Tensordot/transpose:y:0.sequential_3/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_3/dense_10/Tensordot/Reshapeö
&sequential_3/dense_10/Tensordot/MatMulMatMul0sequential_3/dense_10/Tensordot/Reshape:output:06sequential_3/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential_3/dense_10/Tensordot/MatMul
'sequential_3/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_3/dense_10/Tensordot/Const_2 
-sequential_3/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/concat_1/axis«
(sequential_3/dense_10/Tensordot/concat_1ConcatV21sequential_3/dense_10/Tensordot/GatherV2:output:00sequential_3/dense_10/Tensordot/Const_2:output:06sequential_3/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/concat_1è
sequential_3/dense_10/TensordotReshape0sequential_3/dense_10/Tensordot/MatMul:product:01sequential_3/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2!
sequential_3/dense_10/TensordotÎ
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOpß
sequential_3/dense_10/BiasAddBiasAdd(sequential_3/dense_10/Tensordot:output:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_3/dense_10/BiasAdd
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_3/dense_10/ReluØ
.sequential_3/dense_11/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_11_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_3/dense_11/Tensordot/ReadVariableOp
$sequential_3/dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_3/dense_11/Tensordot/axes
$sequential_3/dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_3/dense_11/Tensordot/free¦
%sequential_3/dense_11/Tensordot/ShapeShape(sequential_3/dense_10/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dense_11/Tensordot/Shape 
-sequential_3/dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_11/Tensordot/GatherV2/axis¿
(sequential_3/dense_11/Tensordot/GatherV2GatherV2.sequential_3/dense_11/Tensordot/Shape:output:0-sequential_3/dense_11/Tensordot/free:output:06sequential_3/dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_3/dense_11/Tensordot/GatherV2¤
/sequential_3/dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_3/dense_11/Tensordot/GatherV2_1/axisÅ
*sequential_3/dense_11/Tensordot/GatherV2_1GatherV2.sequential_3/dense_11/Tensordot/Shape:output:0-sequential_3/dense_11/Tensordot/axes:output:08sequential_3/dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_3/dense_11/Tensordot/GatherV2_1
%sequential_3/dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_3/dense_11/Tensordot/ConstØ
$sequential_3/dense_11/Tensordot/ProdProd1sequential_3/dense_11/Tensordot/GatherV2:output:0.sequential_3/dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_3/dense_11/Tensordot/Prod
'sequential_3/dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_11/Tensordot/Const_1à
&sequential_3/dense_11/Tensordot/Prod_1Prod3sequential_3/dense_11/Tensordot/GatherV2_1:output:00sequential_3/dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_3/dense_11/Tensordot/Prod_1
+sequential_3/dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_3/dense_11/Tensordot/concat/axis
&sequential_3/dense_11/Tensordot/concatConcatV2-sequential_3/dense_11/Tensordot/free:output:0-sequential_3/dense_11/Tensordot/axes:output:04sequential_3/dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_3/dense_11/Tensordot/concatä
%sequential_3/dense_11/Tensordot/stackPack-sequential_3/dense_11/Tensordot/Prod:output:0/sequential_3/dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_11/Tensordot/stackô
)sequential_3/dense_11/Tensordot/transpose	Transpose(sequential_3/dense_10/Relu:activations:0/sequential_3/dense_11/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2+
)sequential_3/dense_11/Tensordot/transpose÷
'sequential_3/dense_11/Tensordot/ReshapeReshape-sequential_3/dense_11/Tensordot/transpose:y:0.sequential_3/dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_3/dense_11/Tensordot/Reshapeö
&sequential_3/dense_11/Tensordot/MatMulMatMul0sequential_3/dense_11/Tensordot/Reshape:output:06sequential_3/dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_3/dense_11/Tensordot/MatMul
'sequential_3/dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_11/Tensordot/Const_2 
-sequential_3/dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_11/Tensordot/concat_1/axis«
(sequential_3/dense_11/Tensordot/concat_1ConcatV21sequential_3/dense_11/Tensordot/GatherV2:output:00sequential_3/dense_11/Tensordot/Const_2:output:06sequential_3/dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_3/dense_11/Tensordot/concat_1è
sequential_3/dense_11/TensordotReshape0sequential_3/dense_11/Tensordot/MatMul:product:01sequential_3/dense_11/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
sequential_3/dense_11/TensordotÎ
,sequential_3/dense_11/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/dense_11/BiasAdd/ReadVariableOpß
sequential_3/dense_11/BiasAddBiasAdd(sequential_3/dense_11/Tensordot:output:04sequential_3/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
sequential_3/dense_11/BiasAdd
dropout_10/IdentityIdentity&sequential_3/dense_11/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_10/Identity
add_1AddV2)layer_normalization_6/batchnorm/add_1:z:0dropout_10/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¶
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_7/moments/mean/reduction_indicesá
"layer_normalization_7/moments/meanMean	add_1:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2$
"layer_normalization_7/moments/meanË
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2,
*layer_normalization_7/moments/StopGradientí
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/layer_normalization_7/moments/SquaredDifference¾
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_7/moments/variance/reduction_indices
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2(
&layer_normalization_7/moments/variance
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752'
%layer_normalization_7/batchnorm/add/yê
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#layer_normalization_7/batchnorm/add¶
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2'
%layer_normalization_7/batchnorm/Rsqrtà
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_7/batchnorm/mul/ReadVariableOpî
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_7/batchnorm/mul¿
%layer_normalization_7/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_7/batchnorm/mul_1á
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_7/batchnorm/mul_2Ô
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_7/batchnorm/ReadVariableOpê
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_7/batchnorm/subá
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_7/batchnorm/add_1×
IdentityIdentity)layer_normalization_7/batchnorm/add_1:z:0/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp;^multi_head_attention_3/attention_output/add/ReadVariableOpE^multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_3/key/add/ReadVariableOp8^multi_head_attention_3/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/query/add/ReadVariableOp:^multi_head_attention_3/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/value/add/ReadVariableOp:^multi_head_attention_3/value/einsum/Einsum/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp/^sequential_3/dense_10/Tensordot/ReadVariableOp-^sequential_3/dense_11/BiasAdd/ReadVariableOp/^sequential_3/dense_11/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_3/attention_output/add/ReadVariableOp:multi_head_attention_3/attention_output/add/ReadVariableOp2
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_3/key/add/ReadVariableOp-multi_head_attention_3/key/add/ReadVariableOp2r
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/query/add/ReadVariableOp/multi_head_attention_3/query/add/ReadVariableOp2v
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/value/add/ReadVariableOp/multi_head_attention_3/value/add/ReadVariableOp2v
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2`
.sequential_3/dense_10/Tensordot/ReadVariableOp.sequential_3/dense_10/Tensordot/ReadVariableOp2\
,sequential_3/dense_11/BiasAdd/ReadVariableOp,sequential_3/dense_11/BiasAdd/ReadVariableOp2`
.sequential_3/dense_11/Tensordot/ReadVariableOp.sequential_3/dense_11/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
¢
÷
D__inference_conv1d_2_layer_call_and_return_conditional_losses_546595

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d/ExpandDims¹
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¸
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: 2
conv1d/ExpandDims_1¸
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2	
BiasAdd^
ReluReluBiasAdd:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2
Reluª
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿR ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 
 
_user_specified_nameinputs
¨
Z
.__inference_concatenate_1_layer_call_fn_549546
inputs_0
inputs_1
identityØ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_5472482
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿà:ÿÿÿÿÿÿÿÿÿ:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
µ
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_547233

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ# :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
è

Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_548971

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
ô

Z__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_548774
x'
#embedding_3_embedding_lookup_548761'
#embedding_2_embedding_lookup_548767
identity¢embedding_2/embedding_lookup¢embedding_3/embedding_lookup?
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
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice/stack_2â
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
range/delta
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
range¯
embedding_3/embedding_lookupResourceGather#embedding_3_embedding_lookup_548761range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_3/embedding_lookup/548761*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02
embedding_3/embedding_lookup
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_3/embedding_lookup/548761*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%embedding_3/embedding_lookup/IdentityÀ
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'embedding_3/embedding_lookup/Identity_1q
embedding_2/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2
embedding_2/Castº
embedding_2/embedding_lookupResourceGather#embedding_2_embedding_lookup_548767embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_2/embedding_lookup/548767*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02
embedding_2/embedding_lookup
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_2/embedding_lookup/548767*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2'
%embedding_2/embedding_lookup/IdentityÅ
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2)
'embedding_2/embedding_lookup/Identity_1®
addAddV20embedding_2/embedding_lookup/Identity_1:output:00embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
add
IdentityIdentityadd:z:0^embedding_2/embedding_lookup^embedding_3/embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿR::2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2<
embedding_3/embedding_lookupembedding_3/embedding_lookup:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR

_user_specified_namex


?__inference_token_and_position_embedding_1_layer_call_fn_548783
x
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_5465632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿR::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR

_user_specified_namex

G
+__inference_dropout_11_layer_call_fn_549593

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_5473012
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
õ
k
O__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_546066

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDimsº
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize

*
paddingVALID*
strides

2	
AvgPool
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
~
)__inference_dense_15_layer_call_fn_549706

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_5474382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¸
 
-__inference_sequential_3_layer_call_fn_549846

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_5465232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs


Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_546216

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1è
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ð

à
4__inference_transformer_block_3_layer_call_fn_549485

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
identity¢StatefulPartitionedCallÁ
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
:ÿÿÿÿÿÿÿÿÿ# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_5469912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
ß
~
)__inference_dense_14_layer_call_fn_549660

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_5473822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ì
©
6__inference_batch_normalization_2_layer_call_fn_548902

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5461832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

P
4__inference_average_pooling1d_4_layer_call_fn_546072

inputs
identityæ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_5460662
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î	
Ý
D__inference_dense_14_layer_call_and_return_conditional_losses_547382

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
õ
k
O__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_546051

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDimsº
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
AvgPool
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó0
È
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_548869

inputs
assignmovingavg_548844
assignmovingavg_1_548850)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/548844*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_548844*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/548844*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/548844*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_548844AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/548844*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/548850*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_548850*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/548850*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/548850*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_548850AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/548850*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1À
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
õ
~
)__inference_conv1d_3_layer_call_fn_548833

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_5466282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÞ::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
 
_user_specified_nameinputs
J
¯
H__inference_sequential_3_layer_call_and_return_conditional_losses_549820

inputs.
*dense_10_tensordot_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource.
*dense_11_tensordot_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity¢dense_10/BiasAdd/ReadVariableOp¢!dense_10/Tensordot/ReadVariableOp¢dense_11/BiasAdd/ReadVariableOp¢!dense_11/Tensordot/ReadVariableOp±
!dense_10/Tensordot/ReadVariableOpReadVariableOp*dense_10_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02#
!dense_10/Tensordot/ReadVariableOp|
dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/axes
dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_10/Tensordot/freej
dense_10/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_10/Tensordot/Shape
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/GatherV2/axisþ
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_10/Tensordot/GatherV2_1/axis
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
dense_10/Tensordot/Const¤
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_1¬
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod_1
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_10/Tensordot/concat/axisÝ
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat°
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/stack«
dense_10/Tensordot/transpose	Transposeinputs"dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_10/Tensordot/transposeÃ
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_10/Tensordot/ReshapeÂ
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_10/Tensordot/MatMul
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_10/Tensordot/Const_2
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/concat_1/axisê
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat_1´
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_10/Tensordot§
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_10/BiasAdd/ReadVariableOp«
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_10/BiasAddw
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_10/Relu±
!dense_11/Tensordot/ReadVariableOpReadVariableOp*dense_11_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02#
!dense_11/Tensordot/ReadVariableOp|
dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_11/Tensordot/axes
dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_11/Tensordot/free
dense_11/Tensordot/ShapeShapedense_10/Relu:activations:0*
T0*
_output_shapes
:2
dense_11/Tensordot/Shape
 dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_11/Tensordot/GatherV2/axisþ
dense_11/Tensordot/GatherV2GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/free:output:0)dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_11/Tensordot/GatherV2
"dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_11/Tensordot/GatherV2_1/axis
dense_11/Tensordot/GatherV2_1GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/axes:output:0+dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_11/Tensordot/GatherV2_1~
dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_11/Tensordot/Const¤
dense_11/Tensordot/ProdProd$dense_11/Tensordot/GatherV2:output:0!dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_11/Tensordot/Prod
dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_11/Tensordot/Const_1¬
dense_11/Tensordot/Prod_1Prod&dense_11/Tensordot/GatherV2_1:output:0#dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_11/Tensordot/Prod_1
dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_11/Tensordot/concat/axisÝ
dense_11/Tensordot/concatConcatV2 dense_11/Tensordot/free:output:0 dense_11/Tensordot/axes:output:0'dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/concat°
dense_11/Tensordot/stackPack dense_11/Tensordot/Prod:output:0"dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/stackÀ
dense_11/Tensordot/transpose	Transposedense_10/Relu:activations:0"dense_11/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_11/Tensordot/transposeÃ
dense_11/Tensordot/ReshapeReshape dense_11/Tensordot/transpose:y:0!dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_11/Tensordot/ReshapeÂ
dense_11/Tensordot/MatMulMatMul#dense_11/Tensordot/Reshape:output:0)dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_11/Tensordot/MatMul
dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_11/Tensordot/Const_2
 dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_11/Tensordot/concat_1/axisê
dense_11/Tensordot/concat_1ConcatV2$dense_11/Tensordot/GatherV2:output:0#dense_11/Tensordot/Const_2:output:0)dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/concat_1´
dense_11/TensordotReshape#dense_11/Tensordot/MatMul:product:0$dense_11/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_11/Tensordot§
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_11/BiasAdd/ReadVariableOp«
dense_11/BiasAddBiasAdddense_11/Tensordot:output:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_11/BiasAddý
IdentityIdentitydense_11/BiasAdd:output:0 ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp"^dense_11/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2F
!dense_11/Tensordot/ReadVariableOp!dense_11/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
î
©
6__inference_batch_normalization_2_layer_call_fn_548915

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5462162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ô

Z__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_546563
x'
#embedding_3_embedding_lookup_546550'
#embedding_2_embedding_lookup_546556
identity¢embedding_2/embedding_lookup¢embedding_3/embedding_lookup?
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
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice/stack_2â
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
range/delta
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
range¯
embedding_3/embedding_lookupResourceGather#embedding_3_embedding_lookup_546550range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_3/embedding_lookup/546550*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02
embedding_3/embedding_lookup
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_3/embedding_lookup/546550*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%embedding_3/embedding_lookup/IdentityÀ
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'embedding_3/embedding_lookup/Identity_1q
embedding_2/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2
embedding_2/Castº
embedding_2/embedding_lookupResourceGather#embedding_2_embedding_lookup_546556embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_2/embedding_lookup/546556*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02
embedding_2/embedding_lookup
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_2/embedding_lookup/546556*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2'
%embedding_2/embedding_lookup/IdentityÅ
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2)
'embedding_2/embedding_lookup/Identity_1®
addAddV20embedding_2/embedding_lookup/Identity_1:output:00embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
add
IdentityIdentityadd:z:0^embedding_2/embedding_lookup^embedding_3/embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿR::2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2<
embedding_3/embedding_lookupembedding_3/embedding_lookup:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR

_user_specified_namex
É
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_549630

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¹§
Ì*
__inference__traced_save_550183
file_prefix.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop,
(savev2_conv1d_3_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop$
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
[savev2_transformer_block_3_multi_head_attention_3_attention_output_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableopN
Jsavev2_transformer_block_3_layer_normalization_6_gamma_read_readvariableopM
Isavev2_transformer_block_3_layer_normalization_6_beta_read_readvariableopN
Jsavev2_transformer_block_3_layer_normalization_7_gamma_read_readvariableopM
Isavev2_transformer_block_3_layer_normalization_7_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop;
7savev2_sgd_conv1d_2_kernel_momentum_read_readvariableop9
5savev2_sgd_conv1d_2_bias_momentum_read_readvariableop;
7savev2_sgd_conv1d_3_kernel_momentum_read_readvariableop9
5savev2_sgd_conv1d_3_bias_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_2_gamma_momentum_read_readvariableopF
Bsavev2_sgd_batch_normalization_2_beta_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_3_gamma_momentum_read_readvariableopF
Bsavev2_sgd_batch_normalization_3_beta_momentum_read_readvariableop;
7savev2_sgd_dense_12_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_12_bias_momentum_read_readvariableop;
7savev2_sgd_dense_13_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_13_bias_momentum_read_readvariableop;
7savev2_sgd_dense_14_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_14_bias_momentum_read_readvariableop;
7savev2_sgd_dense_15_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_15_bias_momentum_read_readvariableopa
]savev2_sgd_token_and_position_embedding_1_embedding_2_embeddings_momentum_read_readvariableopa
]savev2_sgd_token_and_position_embedding_1_embedding_3_embeddings_momentum_read_readvariableopc
_savev2_sgd_transformer_block_3_multi_head_attention_3_query_kernel_momentum_read_readvariableopa
]savev2_sgd_transformer_block_3_multi_head_attention_3_query_bias_momentum_read_readvariableopa
]savev2_sgd_transformer_block_3_multi_head_attention_3_key_kernel_momentum_read_readvariableop_
[savev2_sgd_transformer_block_3_multi_head_attention_3_key_bias_momentum_read_readvariableopc
_savev2_sgd_transformer_block_3_multi_head_attention_3_value_kernel_momentum_read_readvariableopa
]savev2_sgd_transformer_block_3_multi_head_attention_3_value_bias_momentum_read_readvariableopn
jsavev2_sgd_transformer_block_3_multi_head_attention_3_attention_output_kernel_momentum_read_readvariableopl
hsavev2_sgd_transformer_block_3_multi_head_attention_3_attention_output_bias_momentum_read_readvariableop;
7savev2_sgd_dense_10_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_10_bias_momentum_read_readvariableop;
7savev2_sgd_dense_11_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_11_bias_momentum_read_readvariableop[
Wsavev2_sgd_transformer_block_3_layer_normalization_6_gamma_momentum_read_readvariableopZ
Vsavev2_sgd_transformer_block_3_layer_normalization_6_beta_momentum_read_readvariableop[
Wsavev2_sgd_transformer_block_3_layer_normalization_7_gamma_momentum_read_readvariableopZ
Vsavev2_sgd_transformer_block_3_layer_normalization_7_beta_momentum_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename(
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*'
value'B'OB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names©
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*³
value©B¦OB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices°)
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableopPsavev2_token_and_position_embedding_1_embedding_2_embeddings_read_readvariableopPsavev2_token_and_position_embedding_1_embedding_3_embeddings_read_readvariableopRsavev2_transformer_block_3_multi_head_attention_3_query_kernel_read_readvariableopPsavev2_transformer_block_3_multi_head_attention_3_query_bias_read_readvariableopPsavev2_transformer_block_3_multi_head_attention_3_key_kernel_read_readvariableopNsavev2_transformer_block_3_multi_head_attention_3_key_bias_read_readvariableopRsavev2_transformer_block_3_multi_head_attention_3_value_kernel_read_readvariableopPsavev2_transformer_block_3_multi_head_attention_3_value_bias_read_readvariableop]savev2_transformer_block_3_multi_head_attention_3_attention_output_kernel_read_readvariableop[savev2_transformer_block_3_multi_head_attention_3_attention_output_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableopJsavev2_transformer_block_3_layer_normalization_6_gamma_read_readvariableopIsavev2_transformer_block_3_layer_normalization_6_beta_read_readvariableopJsavev2_transformer_block_3_layer_normalization_7_gamma_read_readvariableopIsavev2_transformer_block_3_layer_normalization_7_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop7savev2_sgd_conv1d_2_kernel_momentum_read_readvariableop5savev2_sgd_conv1d_2_bias_momentum_read_readvariableop7savev2_sgd_conv1d_3_kernel_momentum_read_readvariableop5savev2_sgd_conv1d_3_bias_momentum_read_readvariableopCsavev2_sgd_batch_normalization_2_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_2_beta_momentum_read_readvariableopCsavev2_sgd_batch_normalization_3_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_3_beta_momentum_read_readvariableop7savev2_sgd_dense_12_kernel_momentum_read_readvariableop5savev2_sgd_dense_12_bias_momentum_read_readvariableop7savev2_sgd_dense_13_kernel_momentum_read_readvariableop5savev2_sgd_dense_13_bias_momentum_read_readvariableop7savev2_sgd_dense_14_kernel_momentum_read_readvariableop5savev2_sgd_dense_14_bias_momentum_read_readvariableop7savev2_sgd_dense_15_kernel_momentum_read_readvariableop5savev2_sgd_dense_15_bias_momentum_read_readvariableop]savev2_sgd_token_and_position_embedding_1_embedding_2_embeddings_momentum_read_readvariableop]savev2_sgd_token_and_position_embedding_1_embedding_3_embeddings_momentum_read_readvariableop_savev2_sgd_transformer_block_3_multi_head_attention_3_query_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_3_multi_head_attention_3_query_bias_momentum_read_readvariableop]savev2_sgd_transformer_block_3_multi_head_attention_3_key_kernel_momentum_read_readvariableop[savev2_sgd_transformer_block_3_multi_head_attention_3_key_bias_momentum_read_readvariableop_savev2_sgd_transformer_block_3_multi_head_attention_3_value_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_3_multi_head_attention_3_value_bias_momentum_read_readvariableopjsavev2_sgd_transformer_block_3_multi_head_attention_3_attention_output_kernel_momentum_read_readvariableophsavev2_sgd_transformer_block_3_multi_head_attention_3_attention_output_bias_momentum_read_readvariableop7savev2_sgd_dense_10_kernel_momentum_read_readvariableop5savev2_sgd_dense_10_bias_momentum_read_readvariableop7savev2_sgd_dense_11_kernel_momentum_read_readvariableop5savev2_sgd_dense_11_bias_momentum_read_readvariableopWsavev2_sgd_transformer_block_3_layer_normalization_6_gamma_momentum_read_readvariableopVsavev2_sgd_transformer_block_3_layer_normalization_6_beta_momentum_read_readvariableopWsavev2_sgd_transformer_block_3_layer_normalization_7_gamma_momentum_read_readvariableopVsavev2_sgd_transformer_block_3_layer_normalization_7_beta_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *]
dtypesS
Q2O	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*
_input_shapes
: : ::	 : : : : : : : : : :	è@:@:@@:@:@@:@:@:: : : : : :	R :  : :  : :  : :  : : @:@:@ : : : : : : : : ::	 : : : : : :	è@:@:@@:@:@@:@:@:: :	R :  : :  : :  : :  : : @:@:@ : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_output_shapes
: :!

_output_shapes	
::)%
#
_output_shapes
:	 : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	è@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: :%!

_output_shapes
:	R :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$  

_output_shapes

: :(!$
"
_output_shapes
:  : "

_output_shapes
: :$# 

_output_shapes

: @: $

_output_shapes
:@:$% 

_output_shapes

:@ : &

_output_shapes
: : '

_output_shapes
: : (

_output_shapes
: : )

_output_shapes
: : *

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :)-%
#
_output_shapes
: :!.

_output_shapes	
::)/%
#
_output_shapes
:	 : 0

_output_shapes
: : 1

_output_shapes
: : 2

_output_shapes
: : 3

_output_shapes
: : 4

_output_shapes
: :%5!

_output_shapes
:	è@: 6

_output_shapes
:@:$7 

_output_shapes

:@@: 8

_output_shapes
:@:$9 

_output_shapes

:@@: :

_output_shapes
:@:$; 

_output_shapes

:@: <

_output_shapes
::$= 

_output_shapes

: :%>!

_output_shapes
:	R :(?$
"
_output_shapes
:  :$@ 

_output_shapes

: :(A$
"
_output_shapes
:  :$B 

_output_shapes

: :(C$
"
_output_shapes
:  :$D 

_output_shapes

: :(E$
"
_output_shapes
:  : F

_output_shapes
: :$G 

_output_shapes

: @: H

_output_shapes
:@:$I 

_output_shapes

:@ : J

_output_shapes
: : K

_output_shapes
: : L

_output_shapes
: : M

_output_shapes
: : N

_output_shapes
: :O

_output_shapes
: 
¥
d
+__inference_dropout_12_layer_call_fn_549635

inputs
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_5473532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¿
m
A__inference_add_1_layer_call_and_return_conditional_losses_549167
inputs_0
inputs_1
identity]
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ# :ÿÿÿÿÿÿÿÿÿ# :U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
"
_user_specified_name
inputs/1
Ñ
ã
D__inference_dense_11_layer_call_and_return_conditional_losses_549916

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
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
Tensordot/GatherV2/axisÑ
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
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ#@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@
 
_user_specified_nameinputs
è

Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_546701

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
è

Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_546792

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
¢
÷
D__inference_conv1d_2_layer_call_and_return_conditional_losses_548799

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d/ExpandDims¹
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¸
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: 2
conv1d/ExpandDims_1¸
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2	
BiasAdd^
ReluReluBiasAdd:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2
Reluª
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿR ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 
 
_user_specified_nameinputs
± 
ã
D__inference_dense_10_layer_call_and_return_conditional_losses_549877

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
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
Tensordot/GatherV2/axisÑ
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
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ# ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
¥
d
+__inference_dropout_13_layer_call_fn_549682

inputs
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_5474102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
± 
ã
D__inference_dense_10_layer_call_and_return_conditional_losses_546402

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
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
Tensordot/GatherV2/axisÑ
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
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ# ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
ó0
È
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_546323

inputs
assignmovingavg_546298
assignmovingavg_1_546304)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/546298*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_546298*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/546298*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/546298*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_546298AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/546298*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/546304*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_546304*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/546304*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/546304*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_546304AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/546304*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1À
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ï
~
)__inference_dense_11_layer_call_fn_549925

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_5464482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ#@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@
 
_user_specified_nameinputs
ñ	
Ý
D__inference_dense_12_layer_call_and_return_conditional_losses_549557

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	è@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿè::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
 
_user_specified_nameinputs
È
©
6__inference_batch_normalization_3_layer_call_fn_549148

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5467722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
î	
Ý
D__inference_dense_13_layer_call_and_return_conditional_losses_547325

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

÷
D__inference_conv1d_3_layer_call_and_return_conditional_losses_546628

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
conv1d/ExpandDims¹
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:	 *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¸
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:	 2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÞ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
 
_user_specified_nameinputs
Û`
ë
C__inference_model_1_layer_call_and_return_conditional_losses_547555
input_3
input_4)
%token_and_position_embedding_1_547459)
%token_and_position_embedding_1_547461
conv1d_2_547464
conv1d_2_547466
conv1d_3_547470
conv1d_3_547472 
batch_normalization_2_547477 
batch_normalization_2_547479 
batch_normalization_2_547481 
batch_normalization_2_547483 
batch_normalization_3_547486 
batch_normalization_3_547488 
batch_normalization_3_547490 
batch_normalization_3_547492
transformer_block_3_547496
transformer_block_3_547498
transformer_block_3_547500
transformer_block_3_547502
transformer_block_3_547504
transformer_block_3_547506
transformer_block_3_547508
transformer_block_3_547510
transformer_block_3_547512
transformer_block_3_547514
transformer_block_3_547516
transformer_block_3_547518
transformer_block_3_547520
transformer_block_3_547522
transformer_block_3_547524
transformer_block_3_547526
dense_12_547531
dense_12_547533
dense_13_547537
dense_13_547539
dense_14_547543
dense_14_547545
dense_15_547549
dense_15_547551
identity¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCall¢ conv1d_3/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢6token_and_position_embedding_1/StatefulPartitionedCall¢+transformer_block_3/StatefulPartitionedCall
6token_and_position_embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_3%token_and_position_embedding_1_547459%token_and_position_embedding_1_547461*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_54656328
6token_and_position_embedding_1/StatefulPartitionedCallÖ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0conv1d_2_547464conv1d_2_547466*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_5465952"
 conv1d_2/StatefulPartitionedCall¡
#average_pooling1d_3/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_5460512%
#average_pooling1d_3/PartitionedCallÂ
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_3/PartitionedCall:output:0conv1d_3_547470conv1d_3_547472*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_5466282"
 conv1d_3/StatefulPartitionedCallµ
#average_pooling1d_5/PartitionedCallPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_5460812%
#average_pooling1d_5/PartitionedCall
#average_pooling1d_4/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_5460662%
#average_pooling1d_4/PartitionedCallÂ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_4/PartitionedCall:output:0batch_normalization_2_547477batch_normalization_2_547479batch_normalization_2_547481batch_normalization_2_547483*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5467012/
-batch_normalization_2/StatefulPartitionedCallÂ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_5/PartitionedCall:output:0batch_normalization_3_547486batch_normalization_3_547488batch_normalization_3_547490batch_normalization_3_547492*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5467922/
-batch_normalization_3/StatefulPartitionedCall»
add_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:06batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_5468342
add_1/PartitionedCall
+transformer_block_3/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0transformer_block_3_547496transformer_block_3_547498transformer_block_3_547500transformer_block_3_547502transformer_block_3_547504transformer_block_3_547506transformer_block_3_547508transformer_block_3_547510transformer_block_3_547512transformer_block_3_547514transformer_block_3_547516transformer_block_3_547518transformer_block_3_547520transformer_block_3_547522transformer_block_3_547524transformer_block_3_547526*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_5471182-
+transformer_block_3/StatefulPartitionedCall
flatten_1/PartitionedCallPartitionedCall4transformer_block_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_5472332
flatten_1/PartitionedCall
concatenate_1/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0input_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_5472482
concatenate_1/PartitionedCall·
 dense_12/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_12_547531dense_12_547533*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_5472682"
 dense_12/StatefulPartitionedCall
dropout_11/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_5473012
dropout_11/PartitionedCall´
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_13_547537dense_13_547539*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_5473252"
 dense_13/StatefulPartitionedCall
dropout_12/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_5473582
dropout_12/PartitionedCall´
 dense_14/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0dense_14_547543dense_14_547545*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_5473822"
 dense_14/StatefulPartitionedCall
dropout_13/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_5474152
dropout_13/PartitionedCall´
 dense_15/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_15_547549dense_15_547551*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_5474382"
 dense_15/StatefulPartitionedCall
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall7^token_and_position_embedding_1/StatefulPartitionedCall,^transformer_block_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ô
_input_shapesÂ
¿:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2p
6token_and_position_embedding_1/StatefulPartitionedCall6token_and_position_embedding_1/StatefulPartitionedCall2Z
+transformer_block_3/StatefulPartitionedCall+transformer_block_3/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_3:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
ß
~
)__inference_dense_13_layer_call_fn_549613

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_5473252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ì
©
6__inference_batch_normalization_3_layer_call_fn_549066

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5463232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
üý
º'
C__inference_model_1_layer_call_and_return_conditional_losses_548335
inputs_0
inputs_1F
Btoken_and_position_embedding_1_embedding_3_embedding_lookup_548022F
Btoken_and_position_embedding_1_embedding_2_embedding_lookup_5480288
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource,
(conv1d_3_biasadd_readvariableop_resource0
,batch_normalization_2_assignmovingavg_5480782
.batch_normalization_2_assignmovingavg_1_548084?
;batch_normalization_2_batchnorm_mul_readvariableop_resource;
7batch_normalization_2_batchnorm_readvariableop_resource0
,batch_normalization_3_assignmovingavg_5481102
.batch_normalization_3_assignmovingavg_1_548116?
;batch_normalization_3_batchnorm_mul_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resourceZ
Vtransformer_block_3_multi_head_attention_3_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_3_multi_head_attention_3_query_add_readvariableop_resourceX
Ttransformer_block_3_multi_head_attention_3_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_3_multi_head_attention_3_key_add_readvariableop_resourceZ
Vtransformer_block_3_multi_head_attention_3_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_3_multi_head_attention_3_value_add_readvariableop_resourcee
atransformer_block_3_multi_head_attention_3_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_3_multi_head_attention_3_attention_output_add_readvariableop_resourceS
Otransformer_block_3_layer_normalization_6_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_3_layer_normalization_6_batchnorm_readvariableop_resourceO
Ktransformer_block_3_sequential_3_dense_10_tensordot_readvariableop_resourceM
Itransformer_block_3_sequential_3_dense_10_biasadd_readvariableop_resourceO
Ktransformer_block_3_sequential_3_dense_11_tensordot_readvariableop_resourceM
Itransformer_block_3_sequential_3_dense_11_biasadd_readvariableop_resourceS
Otransformer_block_3_layer_normalization_7_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_3_layer_normalization_7_batchnorm_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource
identity¢9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_2/AssignMovingAvg/ReadVariableOp¢;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_2/batchnorm/ReadVariableOp¢2batch_normalization_2/batchnorm/mul/ReadVariableOp¢9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_3/AssignMovingAvg/ReadVariableOp¢;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_3/batchnorm/ReadVariableOp¢2batch_normalization_3/batchnorm/mul/ReadVariableOp¢conv1d_2/BiasAdd/ReadVariableOp¢+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp¢conv1d_3/BiasAdd/ReadVariableOp¢+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp¢dense_14/BiasAdd/ReadVariableOp¢dense_14/MatMul/ReadVariableOp¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¢;token_and_position_embedding_1/embedding_2/embedding_lookup¢;token_and_position_embedding_1/embedding_3/embedding_lookup¢Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp¢Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp¢Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp¢Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp¢Ntransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp¢Xtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp¢Atransformer_block_3/multi_head_attention_3/key/add/ReadVariableOp¢Ktransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp¢Ctransformer_block_3/multi_head_attention_3/query/add/ReadVariableOp¢Mtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp¢Ctransformer_block_3/multi_head_attention_3/value/add/ReadVariableOp¢Mtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp¢@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp¢Btransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp¢@transformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOp¢Btransformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOp
$token_and_position_embedding_1/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_1/Shape»
2token_and_position_embedding_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ24
2token_and_position_embedding_1/strided_slice/stack¶
4token_and_position_embedding_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_1/strided_slice/stack_1¶
4token_and_position_embedding_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_1/strided_slice/stack_2
,token_and_position_embedding_1/strided_sliceStridedSlice-token_and_position_embedding_1/Shape:output:0;token_and_position_embedding_1/strided_slice/stack:output:0=token_and_position_embedding_1/strided_slice/stack_1:output:0=token_and_position_embedding_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_1/strided_slice
*token_and_position_embedding_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_1/range/start
*token_and_position_embedding_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_1/range/delta
$token_and_position_embedding_1/rangeRange3token_and_position_embedding_1/range/start:output:05token_and_position_embedding_1/strided_slice:output:03token_and_position_embedding_1/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$token_and_position_embedding_1/rangeÊ
;token_and_position_embedding_1/embedding_3/embedding_lookupResourceGatherBtoken_and_position_embedding_1_embedding_3_embedding_lookup_548022-token_and_position_embedding_1/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_1/embedding_3/embedding_lookup/548022*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02=
;token_and_position_embedding_1/embedding_3/embedding_lookup
Dtoken_and_position_embedding_1/embedding_3/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_1/embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_1/embedding_3/embedding_lookup/548022*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2F
Dtoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity
Ftoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2H
Ftoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1¶
/token_and_position_embedding_1/embedding_2/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR21
/token_and_position_embedding_1/embedding_2/CastÕ
;token_and_position_embedding_1/embedding_2/embedding_lookupResourceGatherBtoken_and_position_embedding_1_embedding_2_embedding_lookup_5480283token_and_position_embedding_1/embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_1/embedding_2/embedding_lookup/548028*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02=
;token_and_position_embedding_1/embedding_2/embedding_lookup
Dtoken_and_position_embedding_1/embedding_2/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_1/embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_1/embedding_2/embedding_lookup/548028*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2F
Dtoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity¢
Ftoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2H
Ftoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1ª
"token_and_position_embedding_1/addAddV2Otoken_and_position_embedding_1/embedding_2/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_1/embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2$
"token_and_position_embedding_1/add
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_2/conv1d/ExpandDims/dimÒ
conv1d_2/conv1d/ExpandDims
ExpandDims&token_and_position_embedding_1/add:z:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_2/conv1d/ExpandDimsÔ
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: *
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dimÜ
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: 2
conv1d_2/conv1d/ExpandDims_1Ü
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*
paddingSAME*
strides
2
conv1d_2/conv1d¯
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_2/conv1d/Squeeze¨
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp²
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2
conv1d_2/BiasAddy
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2
conv1d_2/Relu
"average_pooling1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_3/ExpandDims/dimÔ
average_pooling1d_3/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0+average_pooling1d_3/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2 
average_pooling1d_3/ExpandDimsæ
average_pooling1d_3/AvgPoolAvgPool'average_pooling1d_3/ExpandDims:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ*
ksize
*
paddingVALID*
strides
2
average_pooling1d_3/AvgPoolº
average_pooling1d_3/SqueezeSqueeze$average_pooling1d_3/AvgPool:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ*
squeeze_dims
2
average_pooling1d_3/Squeeze
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_3/conv1d/ExpandDims/dimÑ
conv1d_3/conv1d/ExpandDims
ExpandDims$average_pooling1d_3/Squeeze:output:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
conv1d_3/conv1d/ExpandDimsÔ
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:	 *
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dimÜ
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:	 2
conv1d_3/conv1d/ExpandDims_1Û
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
paddingSAME*
strides
2
conv1d_3/conv1d®
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_3/conv1d/Squeeze§
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_3/BiasAdd/ReadVariableOp±
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_3/BiasAddx
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_3/Relu
"average_pooling1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_5/ExpandDims/dimÞ
average_pooling1d_5/ExpandDims
ExpandDims&token_and_position_embedding_1/add:z:0+average_pooling1d_5/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2 
average_pooling1d_5/ExpandDimsæ
average_pooling1d_5/AvgPoolAvgPool'average_pooling1d_5/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize	
¬*
paddingVALID*
strides	
¬2
average_pooling1d_5/AvgPool¸
average_pooling1d_5/SqueezeSqueeze$average_pooling1d_5/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_5/Squeeze
"average_pooling1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_4/ExpandDims/dimÓ
average_pooling1d_4/ExpandDims
ExpandDimsconv1d_3/Relu:activations:0+average_pooling1d_4/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2 
average_pooling1d_4/ExpandDimsä
average_pooling1d_4/AvgPoolAvgPool'average_pooling1d_4/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize

*
paddingVALID*
strides

2
average_pooling1d_4/AvgPool¸
average_pooling1d_4/SqueezeSqueeze$average_pooling1d_4/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_4/Squeeze½
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_2/moments/mean/reduction_indicesó
"batch_normalization_2/moments/meanMean$average_pooling1d_4/Squeeze:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_2/moments/meanÂ
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_2/moments/StopGradient
/batch_normalization_2/moments/SquaredDifferenceSquaredDifference$average_pooling1d_4/Squeeze:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/batch_normalization_2/moments/SquaredDifferenceÅ
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_2/moments/variance/reduction_indices
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2(
&batch_normalization_2/moments/varianceÃ
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_2/moments/SqueezeË
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1
+batch_normalization_2/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/548078*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_2/AssignMovingAvg/decayÕ
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_2_assignmovingavg_548078*
_output_shapes
: *
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOpß
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/548078*
_output_shapes
: 2+
)batch_normalization_2/AssignMovingAvg/subÖ
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/548078*
_output_shapes
: 2+
)batch_normalization_2/AssignMovingAvg/mul³
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_2_assignmovingavg_548078-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/548078*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_2/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/548084*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_2/AssignMovingAvg_1/decayÛ
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_2_assignmovingavg_1_548084*
_output_shapes
: *
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpé
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/548084*
_output_shapes
: 2-
+batch_normalization_2/AssignMovingAvg_1/subà
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/548084*
_output_shapes
: 2-
+batch_normalization_2/AssignMovingAvg_1/mul¿
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_2_assignmovingavg_1_548084/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/548084*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_2/batchnorm/add/yÚ
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_2/batchnorm/add¥
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_2/batchnorm/Rsqrtà
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_2/batchnorm/mulÚ
%batch_normalization_2/batchnorm/mul_1Mul$average_pooling1d_4/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_2/batchnorm/mul_1Ó
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_2/batchnorm/mul_2Ô
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_2/batchnorm/ReadVariableOpÙ
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_2/batchnorm/subá
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_2/batchnorm/add_1½
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_3/moments/mean/reduction_indicesó
"batch_normalization_3/moments/meanMean$average_pooling1d_5/Squeeze:output:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_3/moments/meanÂ
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_3/moments/StopGradient
/batch_normalization_3/moments/SquaredDifferenceSquaredDifference$average_pooling1d_5/Squeeze:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/batch_normalization_3/moments/SquaredDifferenceÅ
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_3/moments/variance/reduction_indices
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2(
&batch_normalization_3/moments/varianceÃ
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_3/moments/SqueezeË
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1
+batch_normalization_3/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/548110*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_3/AssignMovingAvg/decayÕ
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_3_assignmovingavg_548110*
_output_shapes
: *
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOpß
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/548110*
_output_shapes
: 2+
)batch_normalization_3/AssignMovingAvg/subÖ
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/548110*
_output_shapes
: 2+
)batch_normalization_3/AssignMovingAvg/mul³
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_3_assignmovingavg_548110-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/548110*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_3/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/548116*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_3/AssignMovingAvg_1/decayÛ
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_3_assignmovingavg_1_548116*
_output_shapes
: *
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpé
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/548116*
_output_shapes
: 2-
+batch_normalization_3/AssignMovingAvg_1/subà
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/548116*
_output_shapes
: 2-
+batch_normalization_3/AssignMovingAvg_1/mul¿
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_3_assignmovingavg_1_548116/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/548116*
_output_shapes
 *
dtype02=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_3/batchnorm/add/yÚ
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_3/batchnorm/add¥
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_3/batchnorm/Rsqrtà
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_3/batchnorm/mulÚ
%batch_normalization_3/batchnorm/mul_1Mul$average_pooling1d_5/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_3/batchnorm/mul_1Ó
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_3/batchnorm/mul_2Ô
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpÙ
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_3/batchnorm/subá
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_3/batchnorm/add_1«
	add_1/addAddV2)batch_normalization_2/batchnorm/add_1:z:0)batch_normalization_3/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
	add_1/add¹
Mtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_3_multi_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpÐ
>transformer_block_3/multi_head_attention_3/query/einsum/EinsumEinsumadd_1/add:z:0Utransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2@
>transformer_block_3/multi_head_attention_3/query/einsum/Einsum
Ctransformer_block_3/multi_head_attention_3/query/add/ReadVariableOpReadVariableOpLtransformer_block_3_multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_3/multi_head_attention_3/query/add/ReadVariableOpÅ
4transformer_block_3/multi_head_attention_3/query/addAddV2Gtransformer_block_3/multi_head_attention_3/query/einsum/Einsum:output:0Ktransformer_block_3/multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_3/multi_head_attention_3/query/add³
Ktransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_3_multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpÊ
<transformer_block_3/multi_head_attention_3/key/einsum/EinsumEinsumadd_1/add:z:0Stransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2>
<transformer_block_3/multi_head_attention_3/key/einsum/Einsum
Atransformer_block_3/multi_head_attention_3/key/add/ReadVariableOpReadVariableOpJtransformer_block_3_multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_3/multi_head_attention_3/key/add/ReadVariableOp½
2transformer_block_3/multi_head_attention_3/key/addAddV2Etransformer_block_3/multi_head_attention_3/key/einsum/Einsum:output:0Itransformer_block_3/multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 24
2transformer_block_3/multi_head_attention_3/key/add¹
Mtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_3_multi_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpÐ
>transformer_block_3/multi_head_attention_3/value/einsum/EinsumEinsumadd_1/add:z:0Utransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2@
>transformer_block_3/multi_head_attention_3/value/einsum/Einsum
Ctransformer_block_3/multi_head_attention_3/value/add/ReadVariableOpReadVariableOpLtransformer_block_3_multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_3/multi_head_attention_3/value/add/ReadVariableOpÅ
4transformer_block_3/multi_head_attention_3/value/addAddV2Gtransformer_block_3/multi_head_attention_3/value/einsum/Einsum:output:0Ktransformer_block_3/multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_3/multi_head_attention_3/value/add©
0transformer_block_3/multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>22
0transformer_block_3/multi_head_attention_3/Mul/y
.transformer_block_3/multi_head_attention_3/MulMul8transformer_block_3/multi_head_attention_3/query/add:z:09transformer_block_3/multi_head_attention_3/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 20
.transformer_block_3/multi_head_attention_3/MulÌ
8transformer_block_3/multi_head_attention_3/einsum/EinsumEinsum6transformer_block_3/multi_head_attention_3/key/add:z:02transformer_block_3/multi_head_attention_3/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2:
8transformer_block_3/multi_head_attention_3/einsum/Einsum
:transformer_block_3/multi_head_attention_3/softmax/SoftmaxSoftmaxAtransformer_block_3/multi_head_attention_3/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2<
:transformer_block_3/multi_head_attention_3/softmax/SoftmaxÉ
@transformer_block_3/multi_head_attention_3/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2B
@transformer_block_3/multi_head_attention_3/dropout/dropout/ConstÒ
>transformer_block_3/multi_head_attention_3/dropout/dropout/MulMulDtransformer_block_3/multi_head_attention_3/softmax/Softmax:softmax:0Itransformer_block_3/multi_head_attention_3/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2@
>transformer_block_3/multi_head_attention_3/dropout/dropout/Mulø
@transformer_block_3/multi_head_attention_3/dropout/dropout/ShapeShapeDtransformer_block_3/multi_head_attention_3/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2B
@transformer_block_3/multi_head_attention_3/dropout/dropout/Shapeá
Wtransformer_block_3/multi_head_attention_3/dropout/dropout/random_uniform/RandomUniformRandomUniformItransformer_block_3/multi_head_attention_3/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
dtype0*

seedY2Y
Wtransformer_block_3/multi_head_attention_3/dropout/dropout/random_uniform/RandomUniformÛ
Itransformer_block_3/multi_head_attention_3/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2K
Itransformer_block_3/multi_head_attention_3/dropout/dropout/GreaterEqual/y
Gtransformer_block_3/multi_head_attention_3/dropout/dropout/GreaterEqualGreaterEqual`transformer_block_3/multi_head_attention_3/dropout/dropout/random_uniform/RandomUniform:output:0Rtransformer_block_3/multi_head_attention_3/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2I
Gtransformer_block_3/multi_head_attention_3/dropout/dropout/GreaterEqual 
?transformer_block_3/multi_head_attention_3/dropout/dropout/CastCastKtransformer_block_3/multi_head_attention_3/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2A
?transformer_block_3/multi_head_attention_3/dropout/dropout/CastÎ
@transformer_block_3/multi_head_attention_3/dropout/dropout/Mul_1MulBtransformer_block_3/multi_head_attention_3/dropout/dropout/Mul:z:0Ctransformer_block_3/multi_head_attention_3/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2B
@transformer_block_3/multi_head_attention_3/dropout/dropout/Mul_1ä
:transformer_block_3/multi_head_attention_3/einsum_1/EinsumEinsumDtransformer_block_3/multi_head_attention_3/dropout/dropout/Mul_1:z:08transformer_block_3/multi_head_attention_3/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2<
:transformer_block_3/multi_head_attention_3/einsum_1/EinsumÚ
Xtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_3_multi_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp£
Itransformer_block_3/multi_head_attention_3/attention_output/einsum/EinsumEinsumCtransformer_block_3/multi_head_attention_3/einsum_1/Einsum:output:0`transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe2K
Itransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum´
Ntransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_3_multi_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpí
?transformer_block_3/multi_head_attention_3/attention_output/addAddV2Rtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum:output:0Vtransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?transformer_block_3/multi_head_attention_3/attention_output/add
+transformer_block_3/dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2-
+transformer_block_3/dropout_9/dropout/Const
)transformer_block_3/dropout_9/dropout/MulMulCtransformer_block_3/multi_head_attention_3/attention_output/add:z:04transformer_block_3/dropout_9/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2+
)transformer_block_3/dropout_9/dropout/MulÍ
+transformer_block_3/dropout_9/dropout/ShapeShapeCtransformer_block_3/multi_head_attention_3/attention_output/add:z:0*
T0*
_output_shapes
:2-
+transformer_block_3/dropout_9/dropout/Shape«
Btransformer_block_3/dropout_9/dropout/random_uniform/RandomUniformRandomUniform4transformer_block_3/dropout_9/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype0*

seedY*
seed22D
Btransformer_block_3/dropout_9/dropout/random_uniform/RandomUniform±
4transformer_block_3/dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=26
4transformer_block_3/dropout_9/dropout/GreaterEqual/yº
2transformer_block_3/dropout_9/dropout/GreaterEqualGreaterEqualKtransformer_block_3/dropout_9/dropout/random_uniform/RandomUniform:output:0=transformer_block_3/dropout_9/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 24
2transformer_block_3/dropout_9/dropout/GreaterEqualÝ
*transformer_block_3/dropout_9/dropout/CastCast6transformer_block_3/dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2,
*transformer_block_3/dropout_9/dropout/Castö
+transformer_block_3/dropout_9/dropout/Mul_1Mul-transformer_block_3/dropout_9/dropout/Mul:z:0.transformer_block_3/dropout_9/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+transformer_block_3/dropout_9/dropout/Mul_1±
transformer_block_3/addAddV2add_1/add:z:0/transformer_block_3/dropout_9/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_3/addÞ
Htransformer_block_3/layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_3/layer_normalization_6/moments/mean/reduction_indices¯
6transformer_block_3/layer_normalization_6/moments/meanMeantransformer_block_3/add:z:0Qtransformer_block_3/layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(28
6transformer_block_3/layer_normalization_6/moments/mean
>transformer_block_3/layer_normalization_6/moments/StopGradientStopGradient?transformer_block_3/layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2@
>transformer_block_3/layer_normalization_6/moments/StopGradient»
Ctransformer_block_3/layer_normalization_6/moments/SquaredDifferenceSquaredDifferencetransformer_block_3/add:z:0Gtransformer_block_3/layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2E
Ctransformer_block_3/layer_normalization_6/moments/SquaredDifferenceæ
Ltransformer_block_3/layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_3/layer_normalization_6/moments/variance/reduction_indicesç
:transformer_block_3/layer_normalization_6/moments/varianceMeanGtransformer_block_3/layer_normalization_6/moments/SquaredDifference:z:0Utransformer_block_3/layer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2<
:transformer_block_3/layer_normalization_6/moments/variance»
9transformer_block_3/layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752;
9transformer_block_3/layer_normalization_6/batchnorm/add/yº
7transformer_block_3/layer_normalization_6/batchnorm/addAddV2Ctransformer_block_3/layer_normalization_6/moments/variance:output:0Btransformer_block_3/layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#29
7transformer_block_3/layer_normalization_6/batchnorm/addò
9transformer_block_3/layer_normalization_6/batchnorm/RsqrtRsqrt;transformer_block_3/layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2;
9transformer_block_3/layer_normalization_6/batchnorm/Rsqrt
Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_3_layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp¾
7transformer_block_3/layer_normalization_6/batchnorm/mulMul=transformer_block_3/layer_normalization_6/batchnorm/Rsqrt:y:0Ntransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 29
7transformer_block_3/layer_normalization_6/batchnorm/mul
9transformer_block_3/layer_normalization_6/batchnorm/mul_1Multransformer_block_3/add:z:0;transformer_block_3/layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_3/layer_normalization_6/batchnorm/mul_1±
9transformer_block_3/layer_normalization_6/batchnorm/mul_2Mul?transformer_block_3/layer_normalization_6/moments/mean:output:0;transformer_block_3/layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_3/layer_normalization_6/batchnorm/mul_2
Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_3_layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpº
7transformer_block_3/layer_normalization_6/batchnorm/subSubJtransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp:value:0=transformer_block_3/layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 29
7transformer_block_3/layer_normalization_6/batchnorm/sub±
9transformer_block_3/layer_normalization_6/batchnorm/add_1AddV2=transformer_block_3/layer_normalization_6/batchnorm/mul_1:z:0;transformer_block_3/layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_3/layer_normalization_6/batchnorm/add_1
Btransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_3_sequential_3_dense_10_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02D
Btransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp¾
8transformer_block_3/sequential_3/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_3/sequential_3/dense_10/Tensordot/axesÅ
8transformer_block_3/sequential_3/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_3/sequential_3/dense_10/Tensordot/freeã
9transformer_block_3/sequential_3/dense_10/Tensordot/ShapeShape=transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_3/sequential_3/dense_10/Tensordot/ShapeÈ
Atransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2/axis£
<transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2GatherV2Btransformer_block_3/sequential_3/dense_10/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_10/Tensordot/free:output:0Jtransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2Ì
Ctransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1/axis©
>transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1GatherV2Btransformer_block_3/sequential_3/dense_10/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_10/Tensordot/axes:output:0Ltransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1À
9transformer_block_3/sequential_3/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_3/sequential_3/dense_10/Tensordot/Const¨
8transformer_block_3/sequential_3/dense_10/Tensordot/ProdProdEtransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2:output:0Btransformer_block_3/sequential_3/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_3/sequential_3/dense_10/Tensordot/ProdÄ
;transformer_block_3/sequential_3/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_3/sequential_3/dense_10/Tensordot/Const_1°
:transformer_block_3/sequential_3/dense_10/Tensordot/Prod_1ProdGtransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2_1:output:0Dtransformer_block_3/sequential_3/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_3/sequential_3/dense_10/Tensordot/Prod_1Ä
?transformer_block_3/sequential_3/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_3/sequential_3/dense_10/Tensordot/concat/axis
:transformer_block_3/sequential_3/dense_10/Tensordot/concatConcatV2Atransformer_block_3/sequential_3/dense_10/Tensordot/free:output:0Atransformer_block_3/sequential_3/dense_10/Tensordot/axes:output:0Htransformer_block_3/sequential_3/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_3/sequential_3/dense_10/Tensordot/concat´
9transformer_block_3/sequential_3/dense_10/Tensordot/stackPackAtransformer_block_3/sequential_3/dense_10/Tensordot/Prod:output:0Ctransformer_block_3/sequential_3/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_3/sequential_3/dense_10/Tensordot/stackÅ
=transformer_block_3/sequential_3/dense_10/Tensordot/transpose	Transpose=transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0Ctransformer_block_3/sequential_3/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2?
=transformer_block_3/sequential_3/dense_10/Tensordot/transposeÇ
;transformer_block_3/sequential_3/dense_10/Tensordot/ReshapeReshapeAtransformer_block_3/sequential_3/dense_10/Tensordot/transpose:y:0Btransformer_block_3/sequential_3/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2=
;transformer_block_3/sequential_3/dense_10/Tensordot/ReshapeÆ
:transformer_block_3/sequential_3/dense_10/Tensordot/MatMulMatMulDtransformer_block_3/sequential_3/dense_10/Tensordot/Reshape:output:0Jtransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2<
:transformer_block_3/sequential_3/dense_10/Tensordot/MatMulÄ
;transformer_block_3/sequential_3/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2=
;transformer_block_3/sequential_3/dense_10/Tensordot/Const_2È
Atransformer_block_3/sequential_3/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_3/sequential_3/dense_10/Tensordot/concat_1/axis
<transformer_block_3/sequential_3/dense_10/Tensordot/concat_1ConcatV2Etransformer_block_3/sequential_3/dense_10/Tensordot/GatherV2:output:0Dtransformer_block_3/sequential_3/dense_10/Tensordot/Const_2:output:0Jtransformer_block_3/sequential_3/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_3/sequential_3/dense_10/Tensordot/concat_1¸
3transformer_block_3/sequential_3/dense_10/TensordotReshapeDtransformer_block_3/sequential_3/dense_10/Tensordot/MatMul:product:0Etransformer_block_3/sequential_3/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@25
3transformer_block_3/sequential_3/dense_10/Tensordot
@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_3_sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp¯
1transformer_block_3/sequential_3/dense_10/BiasAddBiasAdd<transformer_block_3/sequential_3/dense_10/Tensordot:output:0Htransformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@23
1transformer_block_3/sequential_3/dense_10/BiasAddÚ
.transformer_block_3/sequential_3/dense_10/ReluRelu:transformer_block_3/sequential_3/dense_10/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@20
.transformer_block_3/sequential_3/dense_10/Relu
Btransformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_3_sequential_3_dense_11_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02D
Btransformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOp¾
8transformer_block_3/sequential_3/dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_3/sequential_3/dense_11/Tensordot/axesÅ
8transformer_block_3/sequential_3/dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_3/sequential_3/dense_11/Tensordot/freeâ
9transformer_block_3/sequential_3/dense_11/Tensordot/ShapeShape<transformer_block_3/sequential_3/dense_10/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_3/sequential_3/dense_11/Tensordot/ShapeÈ
Atransformer_block_3/sequential_3/dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_3/sequential_3/dense_11/Tensordot/GatherV2/axis£
<transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2GatherV2Btransformer_block_3/sequential_3/dense_11/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_11/Tensordot/free:output:0Jtransformer_block_3/sequential_3/dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2Ì
Ctransformer_block_3/sequential_3/dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_3/sequential_3/dense_11/Tensordot/GatherV2_1/axis©
>transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2_1GatherV2Btransformer_block_3/sequential_3/dense_11/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_11/Tensordot/axes:output:0Ltransformer_block_3/sequential_3/dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_3/sequential_3/dense_11/Tensordot/GatherV2_1À
9transformer_block_3/sequential_3/dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_3/sequential_3/dense_11/Tensordot/Const¨
8transformer_block_3/sequential_3/dense_11/Tensordot/ProdProdEtransformer_block_3/sequential_3/dense_11/Tensordot/GatherV2:output:0Btransformer_block_3/sequential_3/dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_3/sequential_3/dense_11/Tensordot/ProdÄ
;transformer_block_3/sequential_3/dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_3/sequential_3/dense_11/Tensordot/Const_1°
:transformer_block_3/sequential_3/dense_11/Tensordot/Prod_1ProdGtransformer_block_3/sequential_3/dense_11/Tensordot/GatherV2_1:output:0Dtransformer_block_3/sequential_3/dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_3/sequential_3/dense_11/Tensordot/Prod_1Ä
?transformer_block_3/sequential_3/dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_3/sequential_3/dense_11/Tensordot/concat/axis
:transformer_block_3/sequential_3/dense_11/Tensordot/concatConcatV2Atransformer_block_3/sequential_3/dense_11/Tensordot/free:output:0Atransformer_block_3/sequential_3/dense_11/Tensordot/axes:output:0Htransformer_block_3/sequential_3/dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_3/sequential_3/dense_11/Tensordot/concat´
9transformer_block_3/sequential_3/dense_11/Tensordot/stackPackAtransformer_block_3/sequential_3/dense_11/Tensordot/Prod:output:0Ctransformer_block_3/sequential_3/dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_3/sequential_3/dense_11/Tensordot/stackÄ
=transformer_block_3/sequential_3/dense_11/Tensordot/transpose	Transpose<transformer_block_3/sequential_3/dense_10/Relu:activations:0Ctransformer_block_3/sequential_3/dense_11/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2?
=transformer_block_3/sequential_3/dense_11/Tensordot/transposeÇ
;transformer_block_3/sequential_3/dense_11/Tensordot/ReshapeReshapeAtransformer_block_3/sequential_3/dense_11/Tensordot/transpose:y:0Btransformer_block_3/sequential_3/dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2=
;transformer_block_3/sequential_3/dense_11/Tensordot/ReshapeÆ
:transformer_block_3/sequential_3/dense_11/Tensordot/MatMulMatMulDtransformer_block_3/sequential_3/dense_11/Tensordot/Reshape:output:0Jtransformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2<
:transformer_block_3/sequential_3/dense_11/Tensordot/MatMulÄ
;transformer_block_3/sequential_3/dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_3/sequential_3/dense_11/Tensordot/Const_2È
Atransformer_block_3/sequential_3/dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_3/sequential_3/dense_11/Tensordot/concat_1/axis
<transformer_block_3/sequential_3/dense_11/Tensordot/concat_1ConcatV2Etransformer_block_3/sequential_3/dense_11/Tensordot/GatherV2:output:0Dtransformer_block_3/sequential_3/dense_11/Tensordot/Const_2:output:0Jtransformer_block_3/sequential_3/dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_3/sequential_3/dense_11/Tensordot/concat_1¸
3transformer_block_3/sequential_3/dense_11/TensordotReshapeDtransformer_block_3/sequential_3/dense_11/Tensordot/MatMul:product:0Etransformer_block_3/sequential_3/dense_11/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 25
3transformer_block_3/sequential_3/dense_11/Tensordot
@transformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_3_sequential_3_dense_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOp¯
1transformer_block_3/sequential_3/dense_11/BiasAddBiasAdd<transformer_block_3/sequential_3/dense_11/Tensordot:output:0Htransformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 23
1transformer_block_3/sequential_3/dense_11/BiasAdd¡
,transformer_block_3/dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2.
,transformer_block_3/dropout_10/dropout/Const
*transformer_block_3/dropout_10/dropout/MulMul:transformer_block_3/sequential_3/dense_11/BiasAdd:output:05transformer_block_3/dropout_10/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2,
*transformer_block_3/dropout_10/dropout/MulÆ
,transformer_block_3/dropout_10/dropout/ShapeShape:transformer_block_3/sequential_3/dense_11/BiasAdd:output:0*
T0*
_output_shapes
:2.
,transformer_block_3/dropout_10/dropout/Shape®
Ctransformer_block_3/dropout_10/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_3/dropout_10/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype0*

seedY*
seed22E
Ctransformer_block_3/dropout_10/dropout/random_uniform/RandomUniform³
5transformer_block_3/dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=27
5transformer_block_3/dropout_10/dropout/GreaterEqual/y¾
3transformer_block_3/dropout_10/dropout/GreaterEqualGreaterEqualLtransformer_block_3/dropout_10/dropout/random_uniform/RandomUniform:output:0>transformer_block_3/dropout_10/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 25
3transformer_block_3/dropout_10/dropout/GreaterEqualà
+transformer_block_3/dropout_10/dropout/CastCast7transformer_block_3/dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+transformer_block_3/dropout_10/dropout/Castú
,transformer_block_3/dropout_10/dropout/Mul_1Mul.transformer_block_3/dropout_10/dropout/Mul:z:0/transformer_block_3/dropout_10/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2.
,transformer_block_3/dropout_10/dropout/Mul_1æ
transformer_block_3/add_1AddV2=transformer_block_3/layer_normalization_6/batchnorm/add_1:z:00transformer_block_3/dropout_10/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_3/add_1Þ
Htransformer_block_3/layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_3/layer_normalization_7/moments/mean/reduction_indices±
6transformer_block_3/layer_normalization_7/moments/meanMeantransformer_block_3/add_1:z:0Qtransformer_block_3/layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(28
6transformer_block_3/layer_normalization_7/moments/mean
>transformer_block_3/layer_normalization_7/moments/StopGradientStopGradient?transformer_block_3/layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2@
>transformer_block_3/layer_normalization_7/moments/StopGradient½
Ctransformer_block_3/layer_normalization_7/moments/SquaredDifferenceSquaredDifferencetransformer_block_3/add_1:z:0Gtransformer_block_3/layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2E
Ctransformer_block_3/layer_normalization_7/moments/SquaredDifferenceæ
Ltransformer_block_3/layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_3/layer_normalization_7/moments/variance/reduction_indicesç
:transformer_block_3/layer_normalization_7/moments/varianceMeanGtransformer_block_3/layer_normalization_7/moments/SquaredDifference:z:0Utransformer_block_3/layer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2<
:transformer_block_3/layer_normalization_7/moments/variance»
9transformer_block_3/layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752;
9transformer_block_3/layer_normalization_7/batchnorm/add/yº
7transformer_block_3/layer_normalization_7/batchnorm/addAddV2Ctransformer_block_3/layer_normalization_7/moments/variance:output:0Btransformer_block_3/layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#29
7transformer_block_3/layer_normalization_7/batchnorm/addò
9transformer_block_3/layer_normalization_7/batchnorm/RsqrtRsqrt;transformer_block_3/layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2;
9transformer_block_3/layer_normalization_7/batchnorm/Rsqrt
Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_3_layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp¾
7transformer_block_3/layer_normalization_7/batchnorm/mulMul=transformer_block_3/layer_normalization_7/batchnorm/Rsqrt:y:0Ntransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 29
7transformer_block_3/layer_normalization_7/batchnorm/mul
9transformer_block_3/layer_normalization_7/batchnorm/mul_1Multransformer_block_3/add_1:z:0;transformer_block_3/layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_3/layer_normalization_7/batchnorm/mul_1±
9transformer_block_3/layer_normalization_7/batchnorm/mul_2Mul?transformer_block_3/layer_normalization_7/moments/mean:output:0;transformer_block_3/layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_3/layer_normalization_7/batchnorm/mul_2
Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_3_layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpº
7transformer_block_3/layer_normalization_7/batchnorm/subSubJtransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp:value:0=transformer_block_3/layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 29
7transformer_block_3/layer_normalization_7/batchnorm/sub±
9transformer_block_3/layer_normalization_7/batchnorm/add_1AddV2=transformer_block_3/layer_normalization_7/batchnorm/mul_1:z:0;transformer_block_3/layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_3/layer_normalization_7/batchnorm/add_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`  2
flatten_1/Const½
flatten_1/ReshapeReshape=transformer_block_3/layer_normalization_7/batchnorm/add_1:z:0flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
flatten_1/Reshapex
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis¾
concatenate_1/concatConcatV2flatten_1/Reshape:output:0inputs_1"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
concatenate_1/concat©
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	è@*
dtype02 
dense_12/MatMul/ReadVariableOp¥
dense_12/MatMulMatMulconcatenate_1/concat:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_12/MatMul§
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_12/BiasAdd/ReadVariableOp¥
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_12/Reluy
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_11/dropout/Const©
dropout_11/dropout/MulMuldense_12/Relu:activations:0!dropout_11/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_11/dropout/Mul
dropout_11/dropout/ShapeShapedense_12/Relu:activations:0*
T0*
_output_shapes
:2
dropout_11/dropout/Shapeî
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*

seedY*
seed221
/dropout_11/dropout/random_uniform/RandomUniform
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_11/dropout/GreaterEqual/yê
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
dropout_11/dropout/GreaterEqual 
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_11/dropout/Cast¦
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_11/dropout/Mul_1¨
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_13/MatMul/ReadVariableOp¤
dense_13/MatMulMatMuldropout_11/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_13/MatMul§
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_13/BiasAdd/ReadVariableOp¥
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_13/BiasAdds
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_13/Reluy
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_12/dropout/Const©
dropout_12/dropout/MulMuldense_13/Relu:activations:0!dropout_12/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_12/dropout/Mul
dropout_12/dropout/ShapeShapedense_13/Relu:activations:0*
T0*
_output_shapes
:2
dropout_12/dropout/Shapeî
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*

seedY*
seed221
/dropout_12/dropout/random_uniform/RandomUniform
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_12/dropout/GreaterEqual/yê
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
dropout_12/dropout/GreaterEqual 
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_12/dropout/Cast¦
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_12/dropout/Mul_1¨
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_14/MatMul/ReadVariableOp¤
dense_14/MatMulMatMuldropout_12/dropout/Mul_1:z:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_14/MatMul§
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_14/BiasAdd/ReadVariableOp¥
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_14/BiasAdds
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_14/Reluy
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_13/dropout/Const©
dropout_13/dropout/MulMuldense_14/Relu:activations:0!dropout_13/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_13/dropout/Mul
dropout_13/dropout/ShapeShapedense_14/Relu:activations:0*
T0*
_output_shapes
:2
dropout_13/dropout/Shapeî
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*

seedY*
seed221
/dropout_13/dropout/random_uniform/RandomUniform
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_13/dropout/GreaterEqual/yê
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
dropout_13/dropout/GreaterEqual 
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_13/dropout/Cast¦
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_13/dropout/Mul_1¨
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_15/MatMul/ReadVariableOp¤
dense_15/MatMulMatMuldropout_13/dropout/Mul_1:z:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/MatMul§
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp¥
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/BiasAddË
IdentityIdentitydense_15/BiasAdd:output:0:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_2/AssignMovingAvg/ReadVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_3/AssignMovingAvg/ReadVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/conv1d/ExpandDims_1/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp<^token_and_position_embedding_1/embedding_2/embedding_lookup<^token_and_position_embedding_1/embedding_3/embedding_lookupC^transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpG^transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpC^transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpG^transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpO^transformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpY^transformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_3/multi_head_attention_3/key/add/ReadVariableOpL^transformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpD^transformer_block_3/multi_head_attention_3/query/add/ReadVariableOpN^transformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpD^transformer_block_3/multi_head_attention_3/value/add/ReadVariableOpN^transformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpA^transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOpC^transformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOpA^transformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOpC^transformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ô
_input_shapesÂ
¿:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::2v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2v
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2z
;token_and_position_embedding_1/embedding_2/embedding_lookup;token_and_position_embedding_1/embedding_2/embedding_lookup2z
;token_and_position_embedding_1/embedding_3/embedding_lookup;token_and_position_embedding_1/embedding_3/embedding_lookup2
Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpBtransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp2
Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpFtransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp2
Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpBtransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp2
Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpFtransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp2 
Ntransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOpNtransformer_block_3/multi_head_attention_3/attention_output/add/ReadVariableOp2´
Xtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_3/multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp2
Atransformer_block_3/multi_head_attention_3/key/add/ReadVariableOpAtransformer_block_3/multi_head_attention_3/key/add/ReadVariableOp2
Ktransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOpKtransformer_block_3/multi_head_attention_3/key/einsum/Einsum/ReadVariableOp2
Ctransformer_block_3/multi_head_attention_3/query/add/ReadVariableOpCtransformer_block_3/multi_head_attention_3/query/add/ReadVariableOp2
Mtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOpMtransformer_block_3/multi_head_attention_3/query/einsum/Einsum/ReadVariableOp2
Ctransformer_block_3/multi_head_attention_3/value/add/ReadVariableOpCtransformer_block_3/multi_head_attention_3/value/add/ReadVariableOp2
Mtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOpMtransformer_block_3/multi_head_attention_3/value/einsum/Einsum/ReadVariableOp2
@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp@transformer_block_3/sequential_3/dense_10/BiasAdd/ReadVariableOp2
Btransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOpBtransformer_block_3/sequential_3/dense_10/Tensordot/ReadVariableOp2
@transformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOp@transformer_block_3/sequential_3/dense_11/BiasAdd/ReadVariableOp2
Btransformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOpBtransformer_block_3/sequential_3/dense_11/Tensordot/ReadVariableOp:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
J
¯
H__inference_sequential_3_layer_call_and_return_conditional_losses_549763

inputs.
*dense_10_tensordot_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource.
*dense_11_tensordot_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity¢dense_10/BiasAdd/ReadVariableOp¢!dense_10/Tensordot/ReadVariableOp¢dense_11/BiasAdd/ReadVariableOp¢!dense_11/Tensordot/ReadVariableOp±
!dense_10/Tensordot/ReadVariableOpReadVariableOp*dense_10_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02#
!dense_10/Tensordot/ReadVariableOp|
dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/axes
dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_10/Tensordot/freej
dense_10/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_10/Tensordot/Shape
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/GatherV2/axisþ
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_10/Tensordot/GatherV2_1/axis
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
dense_10/Tensordot/Const¤
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_1¬
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod_1
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_10/Tensordot/concat/axisÝ
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat°
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/stack«
dense_10/Tensordot/transpose	Transposeinputs"dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_10/Tensordot/transposeÃ
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_10/Tensordot/ReshapeÂ
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_10/Tensordot/MatMul
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_10/Tensordot/Const_2
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/concat_1/axisê
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat_1´
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_10/Tensordot§
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_10/BiasAdd/ReadVariableOp«
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_10/BiasAddw
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_10/Relu±
!dense_11/Tensordot/ReadVariableOpReadVariableOp*dense_11_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02#
!dense_11/Tensordot/ReadVariableOp|
dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_11/Tensordot/axes
dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_11/Tensordot/free
dense_11/Tensordot/ShapeShapedense_10/Relu:activations:0*
T0*
_output_shapes
:2
dense_11/Tensordot/Shape
 dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_11/Tensordot/GatherV2/axisþ
dense_11/Tensordot/GatherV2GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/free:output:0)dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_11/Tensordot/GatherV2
"dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_11/Tensordot/GatherV2_1/axis
dense_11/Tensordot/GatherV2_1GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/axes:output:0+dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_11/Tensordot/GatherV2_1~
dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_11/Tensordot/Const¤
dense_11/Tensordot/ProdProd$dense_11/Tensordot/GatherV2:output:0!dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_11/Tensordot/Prod
dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_11/Tensordot/Const_1¬
dense_11/Tensordot/Prod_1Prod&dense_11/Tensordot/GatherV2_1:output:0#dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_11/Tensordot/Prod_1
dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_11/Tensordot/concat/axisÝ
dense_11/Tensordot/concatConcatV2 dense_11/Tensordot/free:output:0 dense_11/Tensordot/axes:output:0'dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/concat°
dense_11/Tensordot/stackPack dense_11/Tensordot/Prod:output:0"dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/stackÀ
dense_11/Tensordot/transpose	Transposedense_10/Relu:activations:0"dense_11/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_11/Tensordot/transposeÃ
dense_11/Tensordot/ReshapeReshape dense_11/Tensordot/transpose:y:0!dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_11/Tensordot/ReshapeÂ
dense_11/Tensordot/MatMulMatMul#dense_11/Tensordot/Reshape:output:0)dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_11/Tensordot/MatMul
dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_11/Tensordot/Const_2
 dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_11/Tensordot/concat_1/axisê
dense_11/Tensordot/concat_1ConcatV2$dense_11/Tensordot/GatherV2:output:0#dense_11/Tensordot/Const_2:output:0)dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/concat_1´
dense_11/TensordotReshape#dense_11/Tensordot/MatMul:product:0$dense_11/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_11/Tensordot§
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_11/BiasAdd/ReadVariableOp«
dense_11/BiasAddBiasAdddense_11/Tensordot:output:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_11/BiasAddý
IdentityIdentitydense_11/BiasAdd:output:0 ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp"^dense_11/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2F
!dense_11/Tensordot/ReadVariableOp!dense_11/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
¼0
È
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_546772

inputs
assignmovingavg_546747
assignmovingavg_1_546753)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/546747*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_546747*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/546747*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/546747*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_546747AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/546747*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/546753*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_546753*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/546753*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/546753*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_546753AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/546753*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
¬
R
&__inference_add_1_layer_call_fn_549173
inputs_0
inputs_1
identityÓ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_5468342
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ# :ÿÿÿÿÿÿÿÿÿ# :U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
"
_user_specified_name
inputs/1
õ
~
)__inference_conv1d_2_layer_call_fn_548808

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_5465952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿR ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 
 
_user_specified_nameinputs
ó0
È
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_546183

inputs
assignmovingavg_546158
assignmovingavg_1_546164)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/546158*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_546158*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/546158*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/546158*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_546158AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/546158*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/546164*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_546164*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/546164*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/546164*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_546164AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/546164*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1À
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_546356

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1è
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ê
©
6__inference_batch_normalization_3_layer_call_fn_549161

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5467922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs

Ä
(__inference_model_1_layer_call_fn_548668
inputs_0
inputs_1
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

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*D
_read_only_resource_inputs&
$"
 !"#$%&'*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_5476592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ô
_input_shapesÂ
¿:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
×
£3
"__inference__traced_restore_550427
file_prefix$
 assignvariableop_conv1d_2_kernel$
 assignvariableop_1_conv1d_2_bias&
"assignvariableop_2_conv1d_3_kernel$
 assignvariableop_3_conv1d_3_bias2
.assignvariableop_4_batch_normalization_2_gamma1
-assignvariableop_5_batch_normalization_2_beta8
4assignvariableop_6_batch_normalization_2_moving_mean<
8assignvariableop_7_batch_normalization_2_moving_variance2
.assignvariableop_8_batch_normalization_3_gamma1
-assignvariableop_9_batch_normalization_3_beta9
5assignvariableop_10_batch_normalization_3_moving_mean=
9assignvariableop_11_batch_normalization_3_moving_variance'
#assignvariableop_12_dense_12_kernel%
!assignvariableop_13_dense_12_bias'
#assignvariableop_14_dense_13_kernel%
!assignvariableop_15_dense_13_bias'
#assignvariableop_16_dense_14_kernel%
!assignvariableop_17_dense_14_bias'
#assignvariableop_18_dense_15_kernel%
!assignvariableop_19_dense_15_bias
assignvariableop_20_decay%
!assignvariableop_21_learning_rate 
assignvariableop_22_momentum 
assignvariableop_23_sgd_iterM
Iassignvariableop_24_token_and_position_embedding_1_embedding_2_embeddingsM
Iassignvariableop_25_token_and_position_embedding_1_embedding_3_embeddingsO
Kassignvariableop_26_transformer_block_3_multi_head_attention_3_query_kernelM
Iassignvariableop_27_transformer_block_3_multi_head_attention_3_query_biasM
Iassignvariableop_28_transformer_block_3_multi_head_attention_3_key_kernelK
Gassignvariableop_29_transformer_block_3_multi_head_attention_3_key_biasO
Kassignvariableop_30_transformer_block_3_multi_head_attention_3_value_kernelM
Iassignvariableop_31_transformer_block_3_multi_head_attention_3_value_biasZ
Vassignvariableop_32_transformer_block_3_multi_head_attention_3_attention_output_kernelX
Tassignvariableop_33_transformer_block_3_multi_head_attention_3_attention_output_bias'
#assignvariableop_34_dense_10_kernel%
!assignvariableop_35_dense_10_bias'
#assignvariableop_36_dense_11_kernel%
!assignvariableop_37_dense_11_biasG
Cassignvariableop_38_transformer_block_3_layer_normalization_6_gammaF
Bassignvariableop_39_transformer_block_3_layer_normalization_6_betaG
Cassignvariableop_40_transformer_block_3_layer_normalization_7_gammaF
Bassignvariableop_41_transformer_block_3_layer_normalization_7_beta
assignvariableop_42_total
assignvariableop_43_count4
0assignvariableop_44_sgd_conv1d_2_kernel_momentum2
.assignvariableop_45_sgd_conv1d_2_bias_momentum4
0assignvariableop_46_sgd_conv1d_3_kernel_momentum2
.assignvariableop_47_sgd_conv1d_3_bias_momentum@
<assignvariableop_48_sgd_batch_normalization_2_gamma_momentum?
;assignvariableop_49_sgd_batch_normalization_2_beta_momentum@
<assignvariableop_50_sgd_batch_normalization_3_gamma_momentum?
;assignvariableop_51_sgd_batch_normalization_3_beta_momentum4
0assignvariableop_52_sgd_dense_12_kernel_momentum2
.assignvariableop_53_sgd_dense_12_bias_momentum4
0assignvariableop_54_sgd_dense_13_kernel_momentum2
.assignvariableop_55_sgd_dense_13_bias_momentum4
0assignvariableop_56_sgd_dense_14_kernel_momentum2
.assignvariableop_57_sgd_dense_14_bias_momentum4
0assignvariableop_58_sgd_dense_15_kernel_momentum2
.assignvariableop_59_sgd_dense_15_bias_momentumZ
Vassignvariableop_60_sgd_token_and_position_embedding_1_embedding_2_embeddings_momentumZ
Vassignvariableop_61_sgd_token_and_position_embedding_1_embedding_3_embeddings_momentum\
Xassignvariableop_62_sgd_transformer_block_3_multi_head_attention_3_query_kernel_momentumZ
Vassignvariableop_63_sgd_transformer_block_3_multi_head_attention_3_query_bias_momentumZ
Vassignvariableop_64_sgd_transformer_block_3_multi_head_attention_3_key_kernel_momentumX
Tassignvariableop_65_sgd_transformer_block_3_multi_head_attention_3_key_bias_momentum\
Xassignvariableop_66_sgd_transformer_block_3_multi_head_attention_3_value_kernel_momentumZ
Vassignvariableop_67_sgd_transformer_block_3_multi_head_attention_3_value_bias_momentumg
cassignvariableop_68_sgd_transformer_block_3_multi_head_attention_3_attention_output_kernel_momentume
aassignvariableop_69_sgd_transformer_block_3_multi_head_attention_3_attention_output_bias_momentum4
0assignvariableop_70_sgd_dense_10_kernel_momentum2
.assignvariableop_71_sgd_dense_10_bias_momentum4
0assignvariableop_72_sgd_dense_11_kernel_momentum2
.assignvariableop_73_sgd_dense_11_bias_momentumT
Passignvariableop_74_sgd_transformer_block_3_layer_normalization_6_gamma_momentumS
Oassignvariableop_75_sgd_transformer_block_3_layer_normalization_6_beta_momentumT
Passignvariableop_76_sgd_transformer_block_3_layer_normalization_7_gamma_momentumS
Oassignvariableop_77_sgd_transformer_block_3_layer_normalization_7_beta_momentum
identity_79¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_8¢AssignVariableOp_9(
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*'
value'B'OB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¯
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*³
value©B¦OB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices¹
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ò
_output_shapes¿
¼:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*]
dtypesS
Q2O	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_conv1d_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4³
AssignVariableOp_4AssignVariableOp.assignvariableop_4_batch_normalization_2_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5²
AssignVariableOp_5AssignVariableOp-assignvariableop_5_batch_normalization_2_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¹
AssignVariableOp_6AssignVariableOp4assignvariableop_6_batch_normalization_2_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7½
AssignVariableOp_7AssignVariableOp8assignvariableop_7_batch_normalization_2_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8³
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_3_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9²
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_3_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10½
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_3_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Á
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_3_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_12_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_12_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_13_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_13_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16«
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_14_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17©
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_14_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18«
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_15_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19©
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_15_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¡
AssignVariableOp_20AssignVariableOpassignvariableop_20_decayIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21©
AssignVariableOp_21AssignVariableOp!assignvariableop_21_learning_rateIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¤
AssignVariableOp_22AssignVariableOpassignvariableop_22_momentumIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_23¤
AssignVariableOp_23AssignVariableOpassignvariableop_23_sgd_iterIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ñ
AssignVariableOp_24AssignVariableOpIassignvariableop_24_token_and_position_embedding_1_embedding_2_embeddingsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ñ
AssignVariableOp_25AssignVariableOpIassignvariableop_25_token_and_position_embedding_1_embedding_3_embeddingsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ó
AssignVariableOp_26AssignVariableOpKassignvariableop_26_transformer_block_3_multi_head_attention_3_query_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ñ
AssignVariableOp_27AssignVariableOpIassignvariableop_27_transformer_block_3_multi_head_attention_3_query_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ñ
AssignVariableOp_28AssignVariableOpIassignvariableop_28_transformer_block_3_multi_head_attention_3_key_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ï
AssignVariableOp_29AssignVariableOpGassignvariableop_29_transformer_block_3_multi_head_attention_3_key_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ó
AssignVariableOp_30AssignVariableOpKassignvariableop_30_transformer_block_3_multi_head_attention_3_value_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ñ
AssignVariableOp_31AssignVariableOpIassignvariableop_31_transformer_block_3_multi_head_attention_3_value_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Þ
AssignVariableOp_32AssignVariableOpVassignvariableop_32_transformer_block_3_multi_head_attention_3_attention_output_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ü
AssignVariableOp_33AssignVariableOpTassignvariableop_33_transformer_block_3_multi_head_attention_3_attention_output_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34«
AssignVariableOp_34AssignVariableOp#assignvariableop_34_dense_10_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35©
AssignVariableOp_35AssignVariableOp!assignvariableop_35_dense_10_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36«
AssignVariableOp_36AssignVariableOp#assignvariableop_36_dense_11_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37©
AssignVariableOp_37AssignVariableOp!assignvariableop_37_dense_11_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ë
AssignVariableOp_38AssignVariableOpCassignvariableop_38_transformer_block_3_layer_normalization_6_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Ê
AssignVariableOp_39AssignVariableOpBassignvariableop_39_transformer_block_3_layer_normalization_6_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Ë
AssignVariableOp_40AssignVariableOpCassignvariableop_40_transformer_block_3_layer_normalization_7_gammaIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Ê
AssignVariableOp_41AssignVariableOpBassignvariableop_41_transformer_block_3_layer_normalization_7_betaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42¡
AssignVariableOp_42AssignVariableOpassignvariableop_42_totalIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43¡
AssignVariableOp_43AssignVariableOpassignvariableop_43_countIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¸
AssignVariableOp_44AssignVariableOp0assignvariableop_44_sgd_conv1d_2_kernel_momentumIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45¶
AssignVariableOp_45AssignVariableOp.assignvariableop_45_sgd_conv1d_2_bias_momentumIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46¸
AssignVariableOp_46AssignVariableOp0assignvariableop_46_sgd_conv1d_3_kernel_momentumIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47¶
AssignVariableOp_47AssignVariableOp.assignvariableop_47_sgd_conv1d_3_bias_momentumIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Ä
AssignVariableOp_48AssignVariableOp<assignvariableop_48_sgd_batch_normalization_2_gamma_momentumIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Ã
AssignVariableOp_49AssignVariableOp;assignvariableop_49_sgd_batch_normalization_2_beta_momentumIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Ä
AssignVariableOp_50AssignVariableOp<assignvariableop_50_sgd_batch_normalization_3_gamma_momentumIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Ã
AssignVariableOp_51AssignVariableOp;assignvariableop_51_sgd_batch_normalization_3_beta_momentumIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52¸
AssignVariableOp_52AssignVariableOp0assignvariableop_52_sgd_dense_12_kernel_momentumIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53¶
AssignVariableOp_53AssignVariableOp.assignvariableop_53_sgd_dense_12_bias_momentumIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54¸
AssignVariableOp_54AssignVariableOp0assignvariableop_54_sgd_dense_13_kernel_momentumIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55¶
AssignVariableOp_55AssignVariableOp.assignvariableop_55_sgd_dense_13_bias_momentumIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56¸
AssignVariableOp_56AssignVariableOp0assignvariableop_56_sgd_dense_14_kernel_momentumIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57¶
AssignVariableOp_57AssignVariableOp.assignvariableop_57_sgd_dense_14_bias_momentumIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58¸
AssignVariableOp_58AssignVariableOp0assignvariableop_58_sgd_dense_15_kernel_momentumIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59¶
AssignVariableOp_59AssignVariableOp.assignvariableop_59_sgd_dense_15_bias_momentumIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Þ
AssignVariableOp_60AssignVariableOpVassignvariableop_60_sgd_token_and_position_embedding_1_embedding_2_embeddings_momentumIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Þ
AssignVariableOp_61AssignVariableOpVassignvariableop_61_sgd_token_and_position_embedding_1_embedding_3_embeddings_momentumIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62à
AssignVariableOp_62AssignVariableOpXassignvariableop_62_sgd_transformer_block_3_multi_head_attention_3_query_kernel_momentumIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63Þ
AssignVariableOp_63AssignVariableOpVassignvariableop_63_sgd_transformer_block_3_multi_head_attention_3_query_bias_momentumIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Þ
AssignVariableOp_64AssignVariableOpVassignvariableop_64_sgd_transformer_block_3_multi_head_attention_3_key_kernel_momentumIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65Ü
AssignVariableOp_65AssignVariableOpTassignvariableop_65_sgd_transformer_block_3_multi_head_attention_3_key_bias_momentumIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66à
AssignVariableOp_66AssignVariableOpXassignvariableop_66_sgd_transformer_block_3_multi_head_attention_3_value_kernel_momentumIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67Þ
AssignVariableOp_67AssignVariableOpVassignvariableop_67_sgd_transformer_block_3_multi_head_attention_3_value_bias_momentumIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68ë
AssignVariableOp_68AssignVariableOpcassignvariableop_68_sgd_transformer_block_3_multi_head_attention_3_attention_output_kernel_momentumIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69é
AssignVariableOp_69AssignVariableOpaassignvariableop_69_sgd_transformer_block_3_multi_head_attention_3_attention_output_bias_momentumIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70¸
AssignVariableOp_70AssignVariableOp0assignvariableop_70_sgd_dense_10_kernel_momentumIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71¶
AssignVariableOp_71AssignVariableOp.assignvariableop_71_sgd_dense_10_bias_momentumIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72¸
AssignVariableOp_72AssignVariableOp0assignvariableop_72_sgd_dense_11_kernel_momentumIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73¶
AssignVariableOp_73AssignVariableOp.assignvariableop_73_sgd_dense_11_bias_momentumIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74Ø
AssignVariableOp_74AssignVariableOpPassignvariableop_74_sgd_transformer_block_3_layer_normalization_6_gamma_momentumIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75×
AssignVariableOp_75AssignVariableOpOassignvariableop_75_sgd_transformer_block_3_layer_normalization_6_beta_momentumIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76Ø
AssignVariableOp_76AssignVariableOpPassignvariableop_76_sgd_transformer_block_3_layer_normalization_7_gamma_momentumIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77×
AssignVariableOp_77AssignVariableOpOassignvariableop_77_sgd_transformer_block_3_layer_normalization_7_beta_momentumIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_779
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_78Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_78
Identity_79IdentityIdentity_78:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_79"#
identity_79Identity_79:output:0*Ï
_input_shapes½
º: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
½e
Ú
C__inference_model_1_layer_call_and_return_conditional_losses_547455
input_3
input_4)
%token_and_position_embedding_1_546574)
%token_and_position_embedding_1_546576
conv1d_2_546606
conv1d_2_546608
conv1d_3_546639
conv1d_3_546641 
batch_normalization_2_546728 
batch_normalization_2_546730 
batch_normalization_2_546732 
batch_normalization_2_546734 
batch_normalization_3_546819 
batch_normalization_3_546821 
batch_normalization_3_546823 
batch_normalization_3_546825
transformer_block_3_547194
transformer_block_3_547196
transformer_block_3_547198
transformer_block_3_547200
transformer_block_3_547202
transformer_block_3_547204
transformer_block_3_547206
transformer_block_3_547208
transformer_block_3_547210
transformer_block_3_547212
transformer_block_3_547214
transformer_block_3_547216
transformer_block_3_547218
transformer_block_3_547220
transformer_block_3_547222
transformer_block_3_547224
dense_12_547279
dense_12_547281
dense_13_547336
dense_13_547338
dense_14_547393
dense_14_547395
dense_15_547449
dense_15_547451
identity¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCall¢ conv1d_3/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢"dropout_11/StatefulPartitionedCall¢"dropout_12/StatefulPartitionedCall¢"dropout_13/StatefulPartitionedCall¢6token_and_position_embedding_1/StatefulPartitionedCall¢+transformer_block_3/StatefulPartitionedCall
6token_and_position_embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_3%token_and_position_embedding_1_546574%token_and_position_embedding_1_546576*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_54656328
6token_and_position_embedding_1/StatefulPartitionedCallÖ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0conv1d_2_546606conv1d_2_546608*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_5465952"
 conv1d_2/StatefulPartitionedCall¡
#average_pooling1d_3/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_5460512%
#average_pooling1d_3/PartitionedCallÂ
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_3/PartitionedCall:output:0conv1d_3_546639conv1d_3_546641*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_5466282"
 conv1d_3/StatefulPartitionedCallµ
#average_pooling1d_5/PartitionedCallPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_5460812%
#average_pooling1d_5/PartitionedCall
#average_pooling1d_4/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_5460662%
#average_pooling1d_4/PartitionedCallÀ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_4/PartitionedCall:output:0batch_normalization_2_546728batch_normalization_2_546730batch_normalization_2_546732batch_normalization_2_546734*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5466812/
-batch_normalization_2/StatefulPartitionedCallÀ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_5/PartitionedCall:output:0batch_normalization_3_546819batch_normalization_3_546821batch_normalization_3_546823batch_normalization_3_546825*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5467722/
-batch_normalization_3/StatefulPartitionedCall»
add_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:06batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_5468342
add_1/PartitionedCall
+transformer_block_3/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0transformer_block_3_547194transformer_block_3_547196transformer_block_3_547198transformer_block_3_547200transformer_block_3_547202transformer_block_3_547204transformer_block_3_547206transformer_block_3_547208transformer_block_3_547210transformer_block_3_547212transformer_block_3_547214transformer_block_3_547216transformer_block_3_547218transformer_block_3_547220transformer_block_3_547222transformer_block_3_547224*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_5469912-
+transformer_block_3/StatefulPartitionedCall
flatten_1/PartitionedCallPartitionedCall4transformer_block_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_5472332
flatten_1/PartitionedCall
concatenate_1/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0input_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_5472482
concatenate_1/PartitionedCall·
 dense_12/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_12_547279dense_12_547281*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_5472682"
 dense_12/StatefulPartitionedCall
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_5472962$
"dropout_11/StatefulPartitionedCall¼
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_13_547336dense_13_547338*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_5473252"
 dense_13/StatefulPartitionedCall½
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0#^dropout_11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_5473532$
"dropout_12/StatefulPartitionedCall¼
 dense_14/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0dense_14_547393dense_14_547395*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_5473822"
 dense_14/StatefulPartitionedCall½
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_5474102$
"dropout_13/StatefulPartitionedCall¼
 dense_15/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_15_547449dense_15_547451*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_5474382"
 dense_15/StatefulPartitionedCall
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall7^token_and_position_embedding_1/StatefulPartitionedCall,^transformer_block_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ô
_input_shapesÂ
¿:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2p
6token_and_position_embedding_1/StatefulPartitionedCall6token_and_position_embedding_1/StatefulPartitionedCall2Z
+transformer_block_3/StatefulPartitionedCall+transformer_block_3/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_3:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4

Â
(__inference_model_1_layer_call_fn_547738
input_3
input_4
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

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*D
_read_only_resource_inputs&
$"
 !"#$%&'*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_5476592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ô
_input_shapesÂ
¿:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_3:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
É
d
F__inference_dropout_13_layer_call_and_return_conditional_losses_547415

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
Ý
D__inference_dense_15_layer_call_and_return_conditional_losses_549697

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


H__inference_sequential_3_layer_call_and_return_conditional_losses_546465
dense_10_input
dense_10_546413
dense_10_546415
dense_11_546459
dense_11_546461
identity¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall£
 dense_10/StatefulPartitionedCallStatefulPartitionedCalldense_10_inputdense_10_546413dense_10_546415*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_5464022"
 dense_10/StatefulPartitionedCall¾
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_546459dense_11_546461*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_5464482"
 dense_11/StatefulPartitionedCallÇ
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
(
_user_specified_namedense_10_input
·
k
A__inference_add_1_layer_call_and_return_conditional_losses_546834

inputs
inputs_1
identity[
addAddV2inputsinputs_1*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ# :ÿÿÿÿÿÿÿÿÿ# :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
é

H__inference_sequential_3_layer_call_and_return_conditional_losses_546496

inputs
dense_10_546485
dense_10_546487
dense_11_546490
dense_11_546492
identity¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinputsdense_10_546485dense_10_546487*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_5464022"
 dense_10/StatefulPartitionedCall¾
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_546490dense_11_546492*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_5464482"
 dense_11/StatefulPartitionedCallÇ
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs

P
4__inference_average_pooling1d_5_layer_call_fn_546087

inputs
identityæ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_5460812
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_547358

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ð
¨
-__inference_sequential_3_layer_call_fn_546534
dense_10_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCalldense_10_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_5465232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
(
_user_specified_namedense_10_input
È
©
6__inference_batch_normalization_2_layer_call_fn_548984

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5466812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
ñ	
Ý
D__inference_dense_12_layer_call_and_return_conditional_losses_547268

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	è@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿè::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
 
_user_specified_nameinputs
ºþ
Ú
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_549321

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
7layer_normalization_6_batchnorm_readvariableop_resource;
7sequential_3_dense_10_tensordot_readvariableop_resource9
5sequential_3_dense_10_biasadd_readvariableop_resource;
7sequential_3_dense_11_tensordot_readvariableop_resource9
5sequential_3_dense_11_biasadd_readvariableop_resource?
;layer_normalization_7_batchnorm_mul_readvariableop_resource;
7layer_normalization_7_batchnorm_readvariableop_resource
identity¢.layer_normalization_6/batchnorm/ReadVariableOp¢2layer_normalization_6/batchnorm/mul/ReadVariableOp¢.layer_normalization_7/batchnorm/ReadVariableOp¢2layer_normalization_7/batchnorm/mul/ReadVariableOp¢:multi_head_attention_3/attention_output/add/ReadVariableOp¢Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_3/key/add/ReadVariableOp¢7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_3/query/add/ReadVariableOp¢9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_3/value/add/ReadVariableOp¢9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp¢,sequential_3/dense_10/BiasAdd/ReadVariableOp¢.sequential_3/dense_10/Tensordot/ReadVariableOp¢,sequential_3/dense_11/BiasAdd/ReadVariableOp¢.sequential_3/dense_11/Tensordot/ReadVariableOpý
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_3/query/einsum/EinsumEinsuminputsAmulti_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_3/query/einsum/EinsumÛ
/multi_head_attention_3/query/add/ReadVariableOpReadVariableOp8multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/query/add/ReadVariableOpõ
 multi_head_attention_3/query/addAddV23multi_head_attention_3/query/einsum/Einsum:output:07multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_3/query/add÷
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_3/key/einsum/EinsumEinsuminputs?multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2*
(multi_head_attention_3/key/einsum/EinsumÕ
-multi_head_attention_3/key/add/ReadVariableOpReadVariableOp6multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_3/key/add/ReadVariableOpí
multi_head_attention_3/key/addAddV21multi_head_attention_3/key/einsum/Einsum:output:05multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
multi_head_attention_3/key/addý
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_3/value/einsum/EinsumEinsuminputsAmulti_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_3/value/einsum/EinsumÛ
/multi_head_attention_3/value/add/ReadVariableOpReadVariableOp8multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/value/add/ReadVariableOpõ
 multi_head_attention_3/value/addAddV23multi_head_attention_3/value/einsum/Einsum:output:07multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_3/value/add
multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_3/Mul/yÆ
multi_head_attention_3/MulMul$multi_head_attention_3/query/add:z:0%multi_head_attention_3/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_3/Mulü
$multi_head_attention_3/einsum/EinsumEinsum"multi_head_attention_3/key/add:z:0multi_head_attention_3/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2&
$multi_head_attention_3/einsum/EinsumÄ
&multi_head_attention_3/softmax/SoftmaxSoftmax-multi_head_attention_3/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2(
&multi_head_attention_3/softmax/Softmax¡
,multi_head_attention_3/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,multi_head_attention_3/dropout/dropout/Const
*multi_head_attention_3/dropout/dropout/MulMul0multi_head_attention_3/softmax/Softmax:softmax:05multi_head_attention_3/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2,
*multi_head_attention_3/dropout/dropout/Mul¼
,multi_head_attention_3/dropout/dropout/ShapeShape0multi_head_attention_3/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_3/dropout/dropout/Shape¥
Cmulti_head_attention_3/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_3/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
dtype0*

seedY2E
Cmulti_head_attention_3/dropout/dropout/random_uniform/RandomUniform³
5multi_head_attention_3/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_3/dropout/dropout/GreaterEqual/yÂ
3multi_head_attention_3/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_3/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_3/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##25
3multi_head_attention_3/dropout/dropout/GreaterEqualä
+multi_head_attention_3/dropout/dropout/CastCast7multi_head_attention_3/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2-
+multi_head_attention_3/dropout/dropout/Castþ
,multi_head_attention_3/dropout/dropout/Mul_1Mul.multi_head_attention_3/dropout/dropout/Mul:z:0/multi_head_attention_3/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2.
,multi_head_attention_3/dropout/dropout/Mul_1
&multi_head_attention_3/einsum_1/EinsumEinsum0multi_head_attention_3/dropout/dropout/Mul_1:z:0$multi_head_attention_3/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2(
&multi_head_attention_3/einsum_1/Einsum
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_3/attention_output/einsum/EinsumEinsum/multi_head_attention_3/einsum_1/Einsum:output:0Lmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe27
5multi_head_attention_3/attention_output/einsum/Einsumø
:multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_3/attention_output/add/ReadVariableOp
+multi_head_attention_3/attention_output/addAddV2>multi_head_attention_3/attention_output/einsum/Einsum:output:0Bmulti_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+multi_head_attention_3/attention_output/addw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_9/dropout/Const¾
dropout_9/dropout/MulMul/multi_head_attention_3/attention_output/add:z:0 dropout_9/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_9/dropout/Mul
dropout_9/dropout/ShapeShape/multi_head_attention_3/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shapeï
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype0*

seedY*
seed220
.dropout_9/dropout/random_uniform/RandomUniform
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_9/dropout/GreaterEqual/yê
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
dropout_9/dropout/GreaterEqual¡
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_9/dropout/Cast¦
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_9/dropout/Mul_1n
addAddV2inputsdropout_9/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¶
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_6/moments/mean/reduction_indicesß
"layer_normalization_6/moments/meanMeanadd:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2$
"layer_normalization_6/moments/meanË
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2,
*layer_normalization_6/moments/StopGradientë
/layer_normalization_6/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/layer_normalization_6/moments/SquaredDifference¾
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_6/moments/variance/reduction_indices
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2(
&layer_normalization_6/moments/variance
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752'
%layer_normalization_6/batchnorm/add/yê
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#layer_normalization_6/batchnorm/add¶
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2'
%layer_normalization_6/batchnorm/Rsqrtà
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_6/batchnorm/mul/ReadVariableOpî
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_6/batchnorm/mul½
%layer_normalization_6/batchnorm/mul_1Muladd:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_6/batchnorm/mul_1á
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_6/batchnorm/mul_2Ô
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_6/batchnorm/ReadVariableOpê
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_6/batchnorm/subá
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_6/batchnorm/add_1Ø
.sequential_3/dense_10/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_10_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_3/dense_10/Tensordot/ReadVariableOp
$sequential_3/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_3/dense_10/Tensordot/axes
$sequential_3/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_3/dense_10/Tensordot/free§
%sequential_3/dense_10/Tensordot/ShapeShape)layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/Shape 
-sequential_3/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/GatherV2/axis¿
(sequential_3/dense_10/Tensordot/GatherV2GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/free:output:06sequential_3/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/GatherV2¤
/sequential_3/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_3/dense_10/Tensordot/GatherV2_1/axisÅ
*sequential_3/dense_10/Tensordot/GatherV2_1GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/axes:output:08sequential_3/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_3/dense_10/Tensordot/GatherV2_1
%sequential_3/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_3/dense_10/Tensordot/ConstØ
$sequential_3/dense_10/Tensordot/ProdProd1sequential_3/dense_10/Tensordot/GatherV2:output:0.sequential_3/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_3/dense_10/Tensordot/Prod
'sequential_3/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_1à
&sequential_3/dense_10/Tensordot/Prod_1Prod3sequential_3/dense_10/Tensordot/GatherV2_1:output:00sequential_3/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_3/dense_10/Tensordot/Prod_1
+sequential_3/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_3/dense_10/Tensordot/concat/axis
&sequential_3/dense_10/Tensordot/concatConcatV2-sequential_3/dense_10/Tensordot/free:output:0-sequential_3/dense_10/Tensordot/axes:output:04sequential_3/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_3/dense_10/Tensordot/concatä
%sequential_3/dense_10/Tensordot/stackPack-sequential_3/dense_10/Tensordot/Prod:output:0/sequential_3/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/stackõ
)sequential_3/dense_10/Tensordot/transpose	Transpose)layer_normalization_6/batchnorm/add_1:z:0/sequential_3/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2+
)sequential_3/dense_10/Tensordot/transpose÷
'sequential_3/dense_10/Tensordot/ReshapeReshape-sequential_3/dense_10/Tensordot/transpose:y:0.sequential_3/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_3/dense_10/Tensordot/Reshapeö
&sequential_3/dense_10/Tensordot/MatMulMatMul0sequential_3/dense_10/Tensordot/Reshape:output:06sequential_3/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential_3/dense_10/Tensordot/MatMul
'sequential_3/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_3/dense_10/Tensordot/Const_2 
-sequential_3/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/concat_1/axis«
(sequential_3/dense_10/Tensordot/concat_1ConcatV21sequential_3/dense_10/Tensordot/GatherV2:output:00sequential_3/dense_10/Tensordot/Const_2:output:06sequential_3/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/concat_1è
sequential_3/dense_10/TensordotReshape0sequential_3/dense_10/Tensordot/MatMul:product:01sequential_3/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2!
sequential_3/dense_10/TensordotÎ
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOpß
sequential_3/dense_10/BiasAddBiasAdd(sequential_3/dense_10/Tensordot:output:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_3/dense_10/BiasAdd
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_3/dense_10/ReluØ
.sequential_3/dense_11/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_11_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_3/dense_11/Tensordot/ReadVariableOp
$sequential_3/dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_3/dense_11/Tensordot/axes
$sequential_3/dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_3/dense_11/Tensordot/free¦
%sequential_3/dense_11/Tensordot/ShapeShape(sequential_3/dense_10/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dense_11/Tensordot/Shape 
-sequential_3/dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_11/Tensordot/GatherV2/axis¿
(sequential_3/dense_11/Tensordot/GatherV2GatherV2.sequential_3/dense_11/Tensordot/Shape:output:0-sequential_3/dense_11/Tensordot/free:output:06sequential_3/dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_3/dense_11/Tensordot/GatherV2¤
/sequential_3/dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_3/dense_11/Tensordot/GatherV2_1/axisÅ
*sequential_3/dense_11/Tensordot/GatherV2_1GatherV2.sequential_3/dense_11/Tensordot/Shape:output:0-sequential_3/dense_11/Tensordot/axes:output:08sequential_3/dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_3/dense_11/Tensordot/GatherV2_1
%sequential_3/dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_3/dense_11/Tensordot/ConstØ
$sequential_3/dense_11/Tensordot/ProdProd1sequential_3/dense_11/Tensordot/GatherV2:output:0.sequential_3/dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_3/dense_11/Tensordot/Prod
'sequential_3/dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_11/Tensordot/Const_1à
&sequential_3/dense_11/Tensordot/Prod_1Prod3sequential_3/dense_11/Tensordot/GatherV2_1:output:00sequential_3/dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_3/dense_11/Tensordot/Prod_1
+sequential_3/dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_3/dense_11/Tensordot/concat/axis
&sequential_3/dense_11/Tensordot/concatConcatV2-sequential_3/dense_11/Tensordot/free:output:0-sequential_3/dense_11/Tensordot/axes:output:04sequential_3/dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_3/dense_11/Tensordot/concatä
%sequential_3/dense_11/Tensordot/stackPack-sequential_3/dense_11/Tensordot/Prod:output:0/sequential_3/dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_11/Tensordot/stackô
)sequential_3/dense_11/Tensordot/transpose	Transpose(sequential_3/dense_10/Relu:activations:0/sequential_3/dense_11/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2+
)sequential_3/dense_11/Tensordot/transpose÷
'sequential_3/dense_11/Tensordot/ReshapeReshape-sequential_3/dense_11/Tensordot/transpose:y:0.sequential_3/dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_3/dense_11/Tensordot/Reshapeö
&sequential_3/dense_11/Tensordot/MatMulMatMul0sequential_3/dense_11/Tensordot/Reshape:output:06sequential_3/dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_3/dense_11/Tensordot/MatMul
'sequential_3/dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_11/Tensordot/Const_2 
-sequential_3/dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_11/Tensordot/concat_1/axis«
(sequential_3/dense_11/Tensordot/concat_1ConcatV21sequential_3/dense_11/Tensordot/GatherV2:output:00sequential_3/dense_11/Tensordot/Const_2:output:06sequential_3/dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_3/dense_11/Tensordot/concat_1è
sequential_3/dense_11/TensordotReshape0sequential_3/dense_11/Tensordot/MatMul:product:01sequential_3/dense_11/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
sequential_3/dense_11/TensordotÎ
,sequential_3/dense_11/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/dense_11/BiasAdd/ReadVariableOpß
sequential_3/dense_11/BiasAddBiasAdd(sequential_3/dense_11/Tensordot:output:04sequential_3/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
sequential_3/dense_11/BiasAddy
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_10/dropout/Const¸
dropout_10/dropout/MulMul&sequential_3/dense_11/BiasAdd:output:0!dropout_10/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_10/dropout/Mul
dropout_10/dropout/ShapeShape&sequential_3/dense_11/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_10/dropout/Shapeò
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype0*

seedY*
seed221
/dropout_10/dropout/random_uniform/RandomUniform
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_10/dropout/GreaterEqual/yî
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
dropout_10/dropout/GreaterEqual¤
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_10/dropout/Castª
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_10/dropout/Mul_1
add_1AddV2)layer_normalization_6/batchnorm/add_1:z:0dropout_10/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¶
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_7/moments/mean/reduction_indicesá
"layer_normalization_7/moments/meanMean	add_1:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2$
"layer_normalization_7/moments/meanË
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2,
*layer_normalization_7/moments/StopGradientí
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/layer_normalization_7/moments/SquaredDifference¾
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_7/moments/variance/reduction_indices
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2(
&layer_normalization_7/moments/variance
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752'
%layer_normalization_7/batchnorm/add/yê
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#layer_normalization_7/batchnorm/add¶
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2'
%layer_normalization_7/batchnorm/Rsqrtà
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_7/batchnorm/mul/ReadVariableOpî
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_7/batchnorm/mul¿
%layer_normalization_7/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_7/batchnorm/mul_1á
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_7/batchnorm/mul_2Ô
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_7/batchnorm/ReadVariableOpê
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_7/batchnorm/subá
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_7/batchnorm/add_1×
IdentityIdentity)layer_normalization_7/batchnorm/add_1:z:0/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp;^multi_head_attention_3/attention_output/add/ReadVariableOpE^multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_3/key/add/ReadVariableOp8^multi_head_attention_3/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/query/add/ReadVariableOp:^multi_head_attention_3/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/value/add/ReadVariableOp:^multi_head_attention_3/value/einsum/Einsum/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp/^sequential_3/dense_10/Tensordot/ReadVariableOp-^sequential_3/dense_11/BiasAdd/ReadVariableOp/^sequential_3/dense_11/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_3/attention_output/add/ReadVariableOp:multi_head_attention_3/attention_output/add/ReadVariableOp2
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_3/key/add/ReadVariableOp-multi_head_attention_3/key/add/ReadVariableOp2r
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp7multi_head_attention_3/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/query/add/ReadVariableOp/multi_head_attention_3/query/add/ReadVariableOp2v
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp9multi_head_attention_3/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_3/value/add/ReadVariableOp/multi_head_attention_3/value/add/ReadVariableOp2v
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp9multi_head_attention_3/value/einsum/Einsum/ReadVariableOp2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2`
.sequential_3/dense_10/Tensordot/ReadVariableOp.sequential_3/dense_10/Tensordot/ReadVariableOp2\
,sequential_3/dense_11/BiasAdd/ReadVariableOp,sequential_3/dense_11/BiasAdd/ReadVariableOp2`
.sequential_3/dense_11/Tensordot/ReadVariableOp.sequential_3/dense_11/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
É
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_549583

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
î	
Ý
D__inference_dense_13_layer_call_and_return_conditional_losses_549604

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

P
4__inference_average_pooling1d_3_layer_call_fn_546057

inputs
identityæ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_5460512
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

e
F__inference_dropout_13_layer_call_and_return_conditional_losses_547410

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÀ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*

seedY2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
»e
Ú
C__inference_model_1_layer_call_and_return_conditional_losses_547659

inputs
inputs_1)
%token_and_position_embedding_1_547563)
%token_and_position_embedding_1_547565
conv1d_2_547568
conv1d_2_547570
conv1d_3_547574
conv1d_3_547576 
batch_normalization_2_547581 
batch_normalization_2_547583 
batch_normalization_2_547585 
batch_normalization_2_547587 
batch_normalization_3_547590 
batch_normalization_3_547592 
batch_normalization_3_547594 
batch_normalization_3_547596
transformer_block_3_547600
transformer_block_3_547602
transformer_block_3_547604
transformer_block_3_547606
transformer_block_3_547608
transformer_block_3_547610
transformer_block_3_547612
transformer_block_3_547614
transformer_block_3_547616
transformer_block_3_547618
transformer_block_3_547620
transformer_block_3_547622
transformer_block_3_547624
transformer_block_3_547626
transformer_block_3_547628
transformer_block_3_547630
dense_12_547635
dense_12_547637
dense_13_547641
dense_13_547643
dense_14_547647
dense_14_547649
dense_15_547653
dense_15_547655
identity¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCall¢ conv1d_3/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢"dropout_11/StatefulPartitionedCall¢"dropout_12/StatefulPartitionedCall¢"dropout_13/StatefulPartitionedCall¢6token_and_position_embedding_1/StatefulPartitionedCall¢+transformer_block_3/StatefulPartitionedCall
6token_and_position_embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_1_547563%token_and_position_embedding_1_547565*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_54656328
6token_and_position_embedding_1/StatefulPartitionedCallÖ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0conv1d_2_547568conv1d_2_547570*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_5465952"
 conv1d_2/StatefulPartitionedCall¡
#average_pooling1d_3/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_5460512%
#average_pooling1d_3/PartitionedCallÂ
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_3/PartitionedCall:output:0conv1d_3_547574conv1d_3_547576*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_5466282"
 conv1d_3/StatefulPartitionedCallµ
#average_pooling1d_5/PartitionedCallPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_5460812%
#average_pooling1d_5/PartitionedCall
#average_pooling1d_4/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_5460662%
#average_pooling1d_4/PartitionedCallÀ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_4/PartitionedCall:output:0batch_normalization_2_547581batch_normalization_2_547583batch_normalization_2_547585batch_normalization_2_547587*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5466812/
-batch_normalization_2/StatefulPartitionedCallÀ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_5/PartitionedCall:output:0batch_normalization_3_547590batch_normalization_3_547592batch_normalization_3_547594batch_normalization_3_547596*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5467722/
-batch_normalization_3/StatefulPartitionedCall»
add_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:06batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_5468342
add_1/PartitionedCall
+transformer_block_3/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0transformer_block_3_547600transformer_block_3_547602transformer_block_3_547604transformer_block_3_547606transformer_block_3_547608transformer_block_3_547610transformer_block_3_547612transformer_block_3_547614transformer_block_3_547616transformer_block_3_547618transformer_block_3_547620transformer_block_3_547622transformer_block_3_547624transformer_block_3_547626transformer_block_3_547628transformer_block_3_547630*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_5469912-
+transformer_block_3/StatefulPartitionedCall
flatten_1/PartitionedCallPartitionedCall4transformer_block_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_5472332
flatten_1/PartitionedCall
concatenate_1/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_5472482
concatenate_1/PartitionedCall·
 dense_12/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_12_547635dense_12_547637*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_5472682"
 dense_12/StatefulPartitionedCall
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_5472962$
"dropout_11/StatefulPartitionedCall¼
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_13_547641dense_13_547643*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_5473252"
 dense_13/StatefulPartitionedCall½
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0#^dropout_11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_5473532$
"dropout_12/StatefulPartitionedCall¼
 dense_14/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0dense_14_547647dense_14_547649*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_5473822"
 dense_14/StatefulPartitionedCall½
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_5474102$
"dropout_13/StatefulPartitionedCall¼
 dense_15/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_15_547653dense_15_547655*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_5474382"
 dense_15/StatefulPartitionedCall
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall7^token_and_position_embedding_1/StatefulPartitionedCall,^transformer_block_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ô
_input_shapesÂ
¿:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2p
6token_and_position_embedding_1/StatefulPartitionedCall6token_and_position_embedding_1/StatefulPartitionedCall2Z
+transformer_block_3/StatefulPartitionedCall+transformer_block_3/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð

à
4__inference_transformer_block_3_layer_call_fn_549522

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
identity¢StatefulPartitionedCallÁ
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
:ÿÿÿÿÿÿÿÿÿ# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_5471182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
É
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_547301

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ê
©
6__inference_batch_normalization_2_layer_call_fn_548997

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5467012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs

e
F__inference_dropout_11_layer_call_and_return_conditional_losses_549578

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÀ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*

seedY2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ò
¾
$__inference_signature_wrapper_548010
input_3
input_4
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

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&'*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_5460422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ô
_input_shapesÂ
¿:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_3:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
î
©
6__inference_batch_normalization_3_layer_call_fn_549079

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5463562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
÷
k
O__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_546081

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDims¼
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize	
¬*
paddingVALID*
strides	
¬2	
AvgPool
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
~
)__inference_dense_12_layer_call_fn_549566

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_5472682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿè::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*é
serving_defaultÕ
<
input_31
serving_default_input_3:0ÿÿÿÿÿÿÿÿÿR
;
input_40
serving_default_input_4:0ÿÿÿÿÿÿÿÿÿ<
dense_150
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:º©
ÆM
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
layer-19
layer_with_weights-9
layer-20
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
É__call__
+Ê&call_and_return_all_conditional_losses
Ë_default_save_signature"ØG
_tf_keras_network¼G{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}, "name": "token_and_position_embedding_1", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["token_and_position_embedding_1", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_3", "inbound_nodes": [[["conv1d_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_3", "inbound_nodes": [[["average_pooling1d_3", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_4", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_4", "inbound_nodes": [[["conv1d_3", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_5", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_5", "inbound_nodes": [[["token_and_position_embedding_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["average_pooling1d_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["average_pooling1d_5", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}], ["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block_3", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["transformer_block_3", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["flatten_1", 0, 0, {}], ["input_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_11", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["dropout_11", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_12", "inbound_nodes": [[["dense_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_14", "inbound_nodes": [[["dropout_12", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_13", "inbound_nodes": [[["dense_14", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_15", "inbound_nodes": [[["dropout_13", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0], ["input_4", 0, 0]], "output_layers": [["dense_15", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 10500]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 10500]}, {"class_name": "TensorShape", "items": [null, 8]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.0005000000237487257, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
ñ"î
_tf_keras_input_layerÎ{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
ç
	token_emb
pos_emb
	variables
regularization_losses
 trainable_variables
!	keras_api
Ì__call__
+Í&call_and_return_all_conditional_losses"º
_tf_keras_layer {"class_name": "TokenAndPositionEmbedding", "name": "token_and_position_embedding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
ê	

"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
Î__call__
+Ï&call_and_return_all_conditional_losses"Ã
_tf_keras_layer©{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500, 32]}}

(	variables
)regularization_losses
*trainable_variables
+	keras_api
Ð__call__
+Ñ&call_and_return_all_conditional_losses"ø
_tf_keras_layerÞ{"class_name": "AveragePooling1D", "name": "average_pooling1d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
é	

,kernel
-bias
.	variables
/regularization_losses
0trainable_variables
1	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses"Â
_tf_keras_layer¨{"class_name": "Conv1D", "name": "conv1d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 350, 128]}}

2	variables
3regularization_losses
4trainable_variables
5	keras_api
Ô__call__
+Õ&call_and_return_all_conditional_losses"ø
_tf_keras_layerÞ{"class_name": "AveragePooling1D", "name": "average_pooling1d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_4", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}

6	variables
7regularization_losses
8trainable_variables
9	keras_api
Ö__call__
+×&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "AveragePooling1D", "name": "average_pooling1d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_5", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
¸	
:axis
	;gamma
<beta
=moving_mean
>moving_variance
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
Ø__call__
+Ù&call_and_return_all_conditional_losses"â
_tf_keras_layerÈ{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
¸	
Caxis
	Dgamma
Ebeta
Fmoving_mean
Gmoving_variance
H	variables
Iregularization_losses
Jtrainable_variables
K	keras_api
Ú__call__
+Û&call_and_return_all_conditional_losses"â
_tf_keras_layerÈ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
³
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
Ü__call__
+Ý&call_and_return_all_conditional_losses"¢
_tf_keras_layer{"class_name": "Add", "name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 35, 32]}, {"class_name": "TensorShape", "items": [null, 35, 32]}]}

Patt
Qffn
R
layernorm1
S
layernorm2
Tdropout1
Udropout2
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
Þ__call__
+ß&call_and_return_all_conditional_losses"¥
_tf_keras_layer{"class_name": "TransformerBlock", "name": "transformer_block_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
è
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
à__call__
+á&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
é"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
Ð
^	variables
_regularization_losses
`trainable_variables
a	keras_api
â__call__
+ã&call_and_return_all_conditional_losses"¿
_tf_keras_layer¥{"class_name": "Concatenate", "name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1120]}, {"class_name": "TensorShape", "items": [null, 8]}]}
ø

bkernel
cbias
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
ä__call__
+å&call_and_return_all_conditional_losses"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1128]}}
é
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
æ__call__
+ç&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
ô

lkernel
mbias
n	variables
oregularization_losses
ptrainable_variables
q	keras_api
è__call__
+é&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
é
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
ê__call__
+ë&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
ô

vkernel
wbias
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
ì__call__
+í&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
é
|	variables
}regularization_losses
~trainable_variables
	keras_api
î__call__
+ï&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
û
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
ð__call__
+ñ&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
£

decay
learning_rate
momentum
	iter"momentum§#momentum¨,momentum©-momentumª;momentum«<momentum¬Dmomentum­Emomentum®bmomentum¯cmomentum°lmomentum±mmomentum²vmomentum³wmomentum´momentumµmomentum¶momentum·momentum¸momentum¹momentumºmomentum»momentum¼momentum½momentum¾momentum¿momentumÀmomentumÁmomentumÂmomentumÃmomentumÄmomentumÅmomentumÆmomentumÇmomentumÈ"
	optimizer
Ú
0
1
"2
#3
,4
-5
;6
<7
=8
>9
D10
E11
F12
G13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
b30
c31
l32
m33
v34
w35
36
37"
trackable_list_wrapper
 "
trackable_list_wrapper
º
0
1
"2
#3
,4
-5
;6
<7
D8
E9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
b26
c27
l28
m29
v30
w31
32
33"
trackable_list_wrapper
Ó
layers
	variables
regularization_losses
non_trainable_variables
metrics
trainable_variables
 layer_regularization_losses
 layer_metrics
É__call__
Ë_default_save_signature
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
-
òserving_default"
signature_map
µ

embeddings
¡	variables
¢regularization_losses
£trainable_variables
¤	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses"
_tf_keras_layerõ{"class_name": "Embedding", "name": "embedding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500]}}
²

embeddings
¥	variables
¦regularization_losses
§trainable_variables
¨	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses"
_tf_keras_layerò{"class_name": "Embedding", "name": "embedding_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 10500, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
µ
©layers
	variables
regularization_losses
ªnon_trainable_variables
«metrics
 trainable_variables
 ¬layer_regularization_losses
­layer_metrics
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
&:$ 2conv1d_2/kernel
:2conv1d_2/bias
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
µ
®layers
$	variables
%regularization_losses
¯non_trainable_variables
°metrics
&trainable_variables
 ±layer_regularization_losses
²layer_metrics
Î__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
³layers
(	variables
)regularization_losses
´non_trainable_variables
µmetrics
*trainable_variables
 ¶layer_regularization_losses
·layer_metrics
Ð__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses"
_generic_user_object
&:$	 2conv1d_3/kernel
: 2conv1d_3/bias
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
µ
¸layers
.	variables
/regularization_losses
¹non_trainable_variables
ºmetrics
0trainable_variables
 »layer_regularization_losses
¼layer_metrics
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
½layers
2	variables
3regularization_losses
¾non_trainable_variables
¿metrics
4trainable_variables
 Àlayer_regularization_losses
Álayer_metrics
Ô__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Âlayers
6	variables
7regularization_losses
Ãnon_trainable_variables
Ämetrics
8trainable_variables
 Ålayer_regularization_losses
Ælayer_metrics
Ö__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_2/gamma
(:& 2batch_normalization_2/beta
1:/  (2!batch_normalization_2/moving_mean
5:3  (2%batch_normalization_2/moving_variance
<
;0
<1
=2
>3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
µ
Çlayers
?	variables
@regularization_losses
Ènon_trainable_variables
Émetrics
Atrainable_variables
 Êlayer_regularization_losses
Ëlayer_metrics
Ø__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_3/gamma
(:& 2batch_normalization_3/beta
1:/  (2!batch_normalization_3/moving_mean
5:3  (2%batch_normalization_3/moving_variance
<
D0
E1
F2
G3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
µ
Ìlayers
H	variables
Iregularization_losses
Ínon_trainable_variables
Îmetrics
Jtrainable_variables
 Ïlayer_regularization_losses
Ðlayer_metrics
Ú__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ñlayers
L	variables
Mregularization_losses
Ònon_trainable_variables
Ómetrics
Ntrainable_variables
 Ôlayer_regularization_losses
Õlayer_metrics
Ü__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses"
_generic_user_object

Ö_query_dense
×
_key_dense
Ø_value_dense
Ù_softmax
Ú_dropout_layer
Û_output_dense
Ü	variables
Ýregularization_losses
Þtrainable_variables
ß	keras_api
÷__call__
+ø&call_and_return_all_conditional_losses"
_tf_keras_layerê{"class_name": "MultiHeadAttention", "name": "multi_head_attention_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multi_head_attention_3", "trainable": true, "dtype": "float32", "num_heads": 1, "key_dim": 32, "value_dim": 32, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}
¯
àlayer_with_weights-0
àlayer-0
álayer_with_weights-1
álayer-1
â	variables
ãregularization_losses
ätrainable_variables
å	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses"È
_tf_keras_sequential©{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_10_input"}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_10_input"}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
ê
	æaxis

gamma
	beta
ç	variables
èregularization_losses
étrainable_variables
ê	keras_api
û__call__
+ü&call_and_return_all_conditional_losses"³
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_6", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
ê
	ëaxis

gamma
	beta
ì	variables
íregularization_losses
îtrainable_variables
ï	keras_api
ý__call__
+þ&call_and_return_all_conditional_losses"³
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
ë
ð	variables
ñregularization_losses
òtrainable_variables
ó	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
í
ô	variables
õregularization_losses
ötrainable_variables
÷	keras_api
__call__
+&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
¦
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
¦
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
µ
ølayers
V	variables
Wregularization_losses
ùnon_trainable_variables
úmetrics
Xtrainable_variables
 ûlayer_regularization_losses
ülayer_metrics
Þ__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ýlayers
Z	variables
[regularization_losses
þnon_trainable_variables
ÿmetrics
\trainable_variables
 layer_regularization_losses
layer_metrics
à__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
^	variables
_regularization_losses
non_trainable_variables
metrics
`trainable_variables
 layer_regularization_losses
layer_metrics
â__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses"
_generic_user_object
": 	è@2dense_12/kernel
:@2dense_12/bias
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
µ
layers
d	variables
eregularization_losses
non_trainable_variables
metrics
ftrainable_variables
 layer_regularization_losses
layer_metrics
ä__call__
+å&call_and_return_all_conditional_losses
'å"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
h	variables
iregularization_losses
non_trainable_variables
metrics
jtrainable_variables
 layer_regularization_losses
layer_metrics
æ__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_13/kernel
:@2dense_13/bias
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
µ
layers
n	variables
oregularization_losses
non_trainable_variables
metrics
ptrainable_variables
 layer_regularization_losses
layer_metrics
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
r	variables
sregularization_losses
non_trainable_variables
metrics
ttrainable_variables
 layer_regularization_losses
layer_metrics
ê__call__
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_14/kernel
:@2dense_14/bias
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
µ
layers
x	variables
yregularization_losses
non_trainable_variables
metrics
ztrainable_variables
 layer_regularization_losses
layer_metrics
ì__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layers
|	variables
}regularization_losses
¡non_trainable_variables
¢metrics
~trainable_variables
 £layer_regularization_losses
¤layer_metrics
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_15/kernel
:2dense_15/bias
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
¥layers
	variables
regularization_losses
¦non_trainable_variables
§metrics
trainable_variables
 ¨layer_regularization_losses
©layer_metrics
ð__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
G:E 25token_and_position_embedding_1/embedding_2/embeddings
H:F	R 25token_and_position_embedding_1/embedding_3/embeddings
M:K  27transformer_block_3/multi_head_attention_3/query/kernel
G:E 25transformer_block_3/multi_head_attention_3/query/bias
K:I  25transformer_block_3/multi_head_attention_3/key/kernel
E:C 23transformer_block_3/multi_head_attention_3/key/bias
M:K  27transformer_block_3/multi_head_attention_3/value/kernel
G:E 25transformer_block_3/multi_head_attention_3/value/bias
X:V  2Btransformer_block_3/multi_head_attention_3/attention_output/kernel
N:L 2@transformer_block_3/multi_head_attention_3/attention_output/bias
!: @2dense_10/kernel
:@2dense_10/bias
!:@ 2dense_11/kernel
: 2dense_11/bias
=:; 2/transformer_block_3/layer_normalization_6/gamma
<:: 2.transformer_block_3/layer_normalization_6/beta
=:; 2/transformer_block_3/layer_normalization_7/gamma
<:: 2.transformer_block_3/layer_normalization_7/beta
¾
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
10
11
12
13
14
15
16
17
18
19
20"
trackable_list_wrapper
<
=0
>1
F2
G3"
trackable_list_wrapper
(
ª0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
¸
«layers
¡	variables
¢regularization_losses
¬non_trainable_variables
­metrics
£trainable_variables
 ®layer_regularization_losses
¯layer_metrics
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses"
_generic_user_object
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
¸
°layers
¥	variables
¦regularization_losses
±non_trainable_variables
²metrics
§trainable_variables
 ³layer_regularization_losses
´layer_metrics
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
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
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
F0
G1"
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
Ë
µpartial_output_shape
¶full_output_shape
kernel
	bias
·	variables
¸regularization_losses
¹trainable_variables
º	keras_api
__call__
+&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "EinsumDense", "name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
Ç
»partial_output_shape
¼full_output_shape
kernel
	bias
½	variables
¾regularization_losses
¿trainable_variables
À	keras_api
__call__
+&call_and_return_all_conditional_losses"ç
_tf_keras_layerÍ{"class_name": "EinsumDense", "name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
Ë
Ápartial_output_shape
Âfull_output_shape
kernel
	bias
Ã	variables
Äregularization_losses
Åtrainable_variables
Æ	keras_api
__call__
+&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "EinsumDense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
ë
Ç	variables
Èregularization_losses
Étrainable_variables
Ê	keras_api
__call__
+&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}}
ç
Ë	variables
Ìregularization_losses
Ítrainable_variables
Î	keras_api
__call__
+&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
à
Ïpartial_output_shape
Ðfull_output_shape
kernel
	bias
Ñ	variables
Òregularization_losses
Ótrainable_variables
Ô	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layeræ{"class_name": "EinsumDense", "name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 32], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 1, 32]}}
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
¸
Õlayers
Ü	variables
Ýregularization_losses
Önon_trainable_variables
×metrics
Þtrainable_variables
 Ølayer_regularization_losses
Ùlayer_metrics
÷__call__
+ø&call_and_return_all_conditional_losses
'ø"call_and_return_conditional_losses"
_generic_user_object
þ
kernel
	bias
Ú	variables
Ûregularization_losses
Ütrainable_variables
Ý	keras_api
__call__
+&call_and_return_all_conditional_losses"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}

kernel
	bias
Þ	variables
ßregularization_losses
àtrainable_variables
á	keras_api
__call__
+&call_and_return_all_conditional_losses"Ó
_tf_keras_layer¹{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 64]}}
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
¸
âlayers
â	variables
ãregularization_losses
ãnon_trainable_variables
ämetrics
ätrainable_variables
 ålayer_regularization_losses
ælayer_metrics
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
çlayers
ç	variables
èregularization_losses
ènon_trainable_variables
émetrics
étrainable_variables
 êlayer_regularization_losses
ëlayer_metrics
û__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
ìlayers
ì	variables
íregularization_losses
ínon_trainable_variables
îmetrics
îtrainable_variables
 ïlayer_regularization_losses
ðlayer_metrics
ý__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ñlayers
ð	variables
ñregularization_losses
ònon_trainable_variables
ómetrics
òtrainable_variables
 ôlayer_regularization_losses
õlayer_metrics
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ölayers
ô	variables
õregularization_losses
÷non_trainable_variables
ømetrics
ötrainable_variables
 ùlayer_regularization_losses
úlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
J
P0
Q1
R2
S3
T4
U5"
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
¿

ûtotal

ücount
ý	variables
þ	keras_api"
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
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
ÿlayers
·	variables
¸regularization_losses
non_trainable_variables
metrics
¹trainable_variables
 layer_regularization_losses
layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
layers
½	variables
¾regularization_losses
non_trainable_variables
metrics
¿trainable_variables
 layer_regularization_losses
layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
layers
Ã	variables
Äregularization_losses
non_trainable_variables
metrics
Åtrainable_variables
 layer_regularization_losses
layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
Ç	variables
Èregularization_losses
non_trainable_variables
metrics
Étrainable_variables
 layer_regularization_losses
layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
Ë	variables
Ìregularization_losses
non_trainable_variables
metrics
Ítrainable_variables
 layer_regularization_losses
layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
layers
Ñ	variables
Òregularization_losses
non_trainable_variables
metrics
Ótrainable_variables
 layer_regularization_losses
layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
P
Ö0
×1
Ø2
Ù3
Ú4
Û5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
layers
Ú	variables
Ûregularization_losses
non_trainable_variables
metrics
Ütrainable_variables
  layer_regularization_losses
¡layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
¢layers
Þ	variables
ßregularization_losses
£non_trainable_variables
¤metrics
àtrainable_variables
 ¥layer_regularization_losses
¦layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
0
à0
á1"
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
:  (2total
:  (2count
0
û0
ü1"
trackable_list_wrapper
.
ý	variables"
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
1:/ 2SGD/conv1d_2/kernel/momentum
':%2SGD/conv1d_2/bias/momentum
1:/	 2SGD/conv1d_3/kernel/momentum
&:$ 2SGD/conv1d_3/bias/momentum
4:2 2(SGD/batch_normalization_2/gamma/momentum
3:1 2'SGD/batch_normalization_2/beta/momentum
4:2 2(SGD/batch_normalization_3/gamma/momentum
3:1 2'SGD/batch_normalization_3/beta/momentum
-:+	è@2SGD/dense_12/kernel/momentum
&:$@2SGD/dense_12/bias/momentum
,:*@@2SGD/dense_13/kernel/momentum
&:$@2SGD/dense_13/bias/momentum
,:*@@2SGD/dense_14/kernel/momentum
&:$@2SGD/dense_14/bias/momentum
,:*@2SGD/dense_15/kernel/momentum
&:$2SGD/dense_15/bias/momentum
R:P 2BSGD/token_and_position_embedding_1/embedding_2/embeddings/momentum
S:Q	R 2BSGD/token_and_position_embedding_1/embedding_3/embeddings/momentum
X:V  2DSGD/transformer_block_3/multi_head_attention_3/query/kernel/momentum
R:P 2BSGD/transformer_block_3/multi_head_attention_3/query/bias/momentum
V:T  2BSGD/transformer_block_3/multi_head_attention_3/key/kernel/momentum
P:N 2@SGD/transformer_block_3/multi_head_attention_3/key/bias/momentum
X:V  2DSGD/transformer_block_3/multi_head_attention_3/value/kernel/momentum
R:P 2BSGD/transformer_block_3/multi_head_attention_3/value/bias/momentum
c:a  2OSGD/transformer_block_3/multi_head_attention_3/attention_output/kernel/momentum
Y:W 2MSGD/transformer_block_3/multi_head_attention_3/attention_output/bias/momentum
,:* @2SGD/dense_10/kernel/momentum
&:$@2SGD/dense_10/bias/momentum
,:*@ 2SGD/dense_11/kernel/momentum
&:$ 2SGD/dense_11/bias/momentum
H:F 2<SGD/transformer_block_3/layer_normalization_6/gamma/momentum
G:E 2;SGD/transformer_block_3/layer_normalization_6/beta/momentum
H:F 2<SGD/transformer_block_3/layer_normalization_7/gamma/momentum
G:E 2;SGD/transformer_block_3/layer_normalization_7/beta/momentum
î2ë
(__inference_model_1_layer_call_fn_547920
(__inference_model_1_layer_call_fn_548750
(__inference_model_1_layer_call_fn_547738
(__inference_model_1_layer_call_fn_548668À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
C__inference_model_1_layer_call_and_return_conditional_losses_548586
C__inference_model_1_layer_call_and_return_conditional_losses_548335
C__inference_model_1_layer_call_and_return_conditional_losses_547455
C__inference_model_1_layer_call_and_return_conditional_losses_547555À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
!__inference__wrapped_model_546042ß
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *O¢L
JG
"
input_3ÿÿÿÿÿÿÿÿÿR
!
input_4ÿÿÿÿÿÿÿÿÿ
ä2á
?__inference_token_and_position_embedding_1_layer_call_fn_548783
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿ2ü
Z__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_548774
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv1d_2_layer_call_fn_548808¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv1d_2_layer_call_and_return_conditional_losses_548799¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
4__inference_average_pooling1d_3_layer_call_fn_546057Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª2§
O__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_546051Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ó2Ð
)__inference_conv1d_3_layer_call_fn_548833¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv1d_3_layer_call_and_return_conditional_losses_548824¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
4__inference_average_pooling1d_4_layer_call_fn_546072Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª2§
O__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_546066Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
4__inference_average_pooling1d_5_layer_call_fn_546087Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª2§
O__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_546081Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
6__inference_batch_normalization_2_layer_call_fn_548984
6__inference_batch_normalization_2_layer_call_fn_548902
6__inference_batch_normalization_2_layer_call_fn_548915
6__inference_batch_normalization_2_layer_call_fn_548997´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_548889
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_548869
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_548951
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_548971´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
6__inference_batch_normalization_3_layer_call_fn_549079
6__inference_batch_normalization_3_layer_call_fn_549066
6__inference_batch_normalization_3_layer_call_fn_549148
6__inference_batch_normalization_3_layer_call_fn_549161´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_549053
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_549115
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_549033
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_549135´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ð2Í
&__inference_add_1_layer_call_fn_549173¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_add_1_layer_call_and_return_conditional_losses_549167¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¢2
4__inference_transformer_block_3_layer_call_fn_549522
4__inference_transformer_block_3_layer_call_fn_549485°
§²£
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_549448
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_549321°
§²£
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
*__inference_flatten_1_layer_call_fn_549533¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_flatten_1_layer_call_and_return_conditional_losses_549528¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_concatenate_1_layer_call_fn_549546¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_concatenate_1_layer_call_and_return_conditional_losses_549540¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_12_layer_call_fn_549566¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_12_layer_call_and_return_conditional_losses_549557¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
+__inference_dropout_11_layer_call_fn_549588
+__inference_dropout_11_layer_call_fn_549593´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_11_layer_call_and_return_conditional_losses_549578
F__inference_dropout_11_layer_call_and_return_conditional_losses_549583´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_dense_13_layer_call_fn_549613¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_13_layer_call_and_return_conditional_losses_549604¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
+__inference_dropout_12_layer_call_fn_549640
+__inference_dropout_12_layer_call_fn_549635´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_12_layer_call_and_return_conditional_losses_549630
F__inference_dropout_12_layer_call_and_return_conditional_losses_549625´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_dense_14_layer_call_fn_549660¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_14_layer_call_and_return_conditional_losses_549651¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
+__inference_dropout_13_layer_call_fn_549682
+__inference_dropout_13_layer_call_fn_549687´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_13_layer_call_and_return_conditional_losses_549672
F__inference_dropout_13_layer_call_and_return_conditional_losses_549677´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_dense_15_layer_call_fn_549706¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_15_layer_call_and_return_conditional_losses_549697¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÒBÏ
$__inference_signature_wrapper_548010input_3input_4"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2ÿü
ó²ï
FullArgSpece
args]Z
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
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿü
ó²ï
FullArgSpece
args]Z
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
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿ
-__inference_sequential_3_layer_call_fn_549846
-__inference_sequential_3_layer_call_fn_546507
-__inference_sequential_3_layer_call_fn_549833
-__inference_sequential_3_layer_call_fn_546534À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
H__inference_sequential_3_layer_call_and_return_conditional_losses_549763
H__inference_sequential_3_layer_call_and_return_conditional_losses_549820
H__inference_sequential_3_layer_call_and_return_conditional_losses_546465
H__inference_sequential_3_layer_call_and_return_conditional_losses_546479À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µ2²¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µ2²¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_10_layer_call_fn_549886¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_10_layer_call_and_return_conditional_losses_549877¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_11_layer_call_fn_549925¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_11_layer_call_and_return_conditional_losses_549916¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ò
!__inference__wrapped_model_546042Ì:"#,->;=<GDFEbclmvwY¢V
O¢L
JG
"
input_3ÿÿÿÿÿÿÿÿÿR
!
input_4ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_15"
dense_15ÿÿÿÿÿÿÿÿÿÕ
A__inference_add_1_layer_call_and_return_conditional_losses_549167b¢_
X¢U
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿ# 
&#
inputs/1ÿÿÿÿÿÿÿÿÿ# 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ­
&__inference_add_1_layer_call_fn_549173b¢_
X¢U
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿ# 
&#
inputs/1ÿÿÿÿÿÿÿÿÿ# 
ª "ÿÿÿÿÿÿÿÿÿ# Ø
O__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_546051E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
4__inference_average_pooling1d_3_layer_call_fn_546057wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØ
O__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_546066E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
4__inference_average_pooling1d_4_layer_call_fn_546072wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØ
O__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_546081E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
4__inference_average_pooling1d_5_layer_call_fn_546087wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÑ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_548869|=>;<@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ñ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_548889|>;=<@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ¿
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_548951j=>;<7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ¿
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_548971j>;=<7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ©
6__inference_batch_normalization_2_layer_call_fn_548902o=>;<@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ©
6__inference_batch_normalization_2_layer_call_fn_548915o>;=<@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
6__inference_batch_normalization_2_layer_call_fn_548984]=>;<7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª "ÿÿÿÿÿÿÿÿÿ# 
6__inference_batch_normalization_2_layer_call_fn_548997]>;=<7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª "ÿÿÿÿÿÿÿÿÿ# Ñ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_549033|FGDE@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ñ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_549053|GDFE@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ¿
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_549115jFGDE7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ¿
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_549135jGDFE7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ©
6__inference_batch_normalization_3_layer_call_fn_549066oFGDE@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ©
6__inference_batch_normalization_3_layer_call_fn_549079oGDFE@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
6__inference_batch_normalization_3_layer_call_fn_549148]FGDE7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª "ÿÿÿÿÿÿÿÿÿ# 
6__inference_batch_normalization_3_layer_call_fn_549161]GDFE7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª "ÿÿÿÿÿÿÿÿÿ# Ó
I__inference_concatenate_1_layer_call_and_return_conditional_losses_549540[¢X
Q¢N
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿà
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿè
 ª
.__inference_concatenate_1_layer_call_fn_549546x[¢X
Q¢N
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿà
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿè¯
D__inference_conv1d_2_layer_call_and_return_conditional_losses_548799g"#4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿR 
ª "+¢(
!
0ÿÿÿÿÿÿÿÿÿR
 
)__inference_conv1d_2_layer_call_fn_548808Z"#4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿR 
ª "ÿÿÿÿÿÿÿÿÿR¯
D__inference_conv1d_3_layer_call_and_return_conditional_losses_548824g,-5¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿÞ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÞ 
 
)__inference_conv1d_3_layer_call_fn_548833Z,-5¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿÞ
ª "ÿÿÿÿÿÿÿÿÿÞ ®
D__inference_dense_10_layer_call_and_return_conditional_losses_549877f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ#@
 
)__inference_dense_10_layer_call_fn_549886Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª "ÿÿÿÿÿÿÿÿÿ#@®
D__inference_dense_11_layer_call_and_return_conditional_losses_549916f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ#@
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 
)__inference_dense_11_layer_call_fn_549925Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ#@
ª "ÿÿÿÿÿÿÿÿÿ# ¥
D__inference_dense_12_layer_call_and_return_conditional_losses_549557]bc0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿè
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 }
)__inference_dense_12_layer_call_fn_549566Pbc0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿè
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dense_13_layer_call_and_return_conditional_losses_549604\lm/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
)__inference_dense_13_layer_call_fn_549613Olm/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dense_14_layer_call_and_return_conditional_losses_549651\vw/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
)__inference_dense_14_layer_call_fn_549660Ovw/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@¦
D__inference_dense_15_layer_call_and_return_conditional_losses_549697^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dense_15_layer_call_fn_549706Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dropout_11_layer_call_and_return_conditional_losses_549578\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¦
F__inference_dropout_11_layer_call_and_return_conditional_losses_549583\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ~
+__inference_dropout_11_layer_call_fn_549588O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@~
+__inference_dropout_11_layer_call_fn_549593O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@¦
F__inference_dropout_12_layer_call_and_return_conditional_losses_549625\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¦
F__inference_dropout_12_layer_call_and_return_conditional_losses_549630\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ~
+__inference_dropout_12_layer_call_fn_549635O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@~
+__inference_dropout_12_layer_call_fn_549640O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@¦
F__inference_dropout_13_layer_call_and_return_conditional_losses_549672\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¦
F__inference_dropout_13_layer_call_and_return_conditional_losses_549677\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ~
+__inference_dropout_13_layer_call_fn_549682O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@~
+__inference_dropout_13_layer_call_fn_549687O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@¦
E__inference_flatten_1_layer_call_and_return_conditional_losses_549528]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿà
 ~
*__inference_flatten_1_layer_call_fn_549533P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª "ÿÿÿÿÿÿÿÿÿà
C__inference_model_1_layer_call_and_return_conditional_losses_547455Æ:"#,-=>;<FGDEbclmvwa¢^
W¢T
JG
"
input_3ÿÿÿÿÿÿÿÿÿR
!
input_4ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
C__inference_model_1_layer_call_and_return_conditional_losses_547555Æ:"#,->;=<GDFEbclmvwa¢^
W¢T
JG
"
input_3ÿÿÿÿÿÿÿÿÿR
!
input_4ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
C__inference_model_1_layer_call_and_return_conditional_losses_548335È:"#,-=>;<FGDEbclmvwc¢`
Y¢V
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿR
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
C__inference_model_1_layer_call_and_return_conditional_losses_548586È:"#,->;=<GDFEbclmvwc¢`
Y¢V
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿR
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 æ
(__inference_model_1_layer_call_fn_547738¹:"#,-=>;<FGDEbclmvwa¢^
W¢T
JG
"
input_3ÿÿÿÿÿÿÿÿÿR
!
input_4ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿæ
(__inference_model_1_layer_call_fn_547920¹:"#,->;=<GDFEbclmvwa¢^
W¢T
JG
"
input_3ÿÿÿÿÿÿÿÿÿR
!
input_4ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿè
(__inference_model_1_layer_call_fn_548668»:"#,-=>;<FGDEbclmvwc¢`
Y¢V
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿR
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿè
(__inference_model_1_layer_call_fn_548750»:"#,->;=<GDFEbclmvwc¢`
Y¢V
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿR
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÆ
H__inference_sequential_3_layer_call_and_return_conditional_losses_546465zC¢@
9¢6
,)
dense_10_inputÿÿÿÿÿÿÿÿÿ# 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 Æ
H__inference_sequential_3_layer_call_and_return_conditional_losses_546479zC¢@
9¢6
,)
dense_10_inputÿÿÿÿÿÿÿÿÿ# 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ¾
H__inference_sequential_3_layer_call_and_return_conditional_losses_549763r;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ¾
H__inference_sequential_3_layer_call_and_return_conditional_losses_549820r;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 
-__inference_sequential_3_layer_call_fn_546507mC¢@
9¢6
,)
dense_10_inputÿÿÿÿÿÿÿÿÿ# 
p

 
ª "ÿÿÿÿÿÿÿÿÿ# 
-__inference_sequential_3_layer_call_fn_546534mC¢@
9¢6
,)
dense_10_inputÿÿÿÿÿÿÿÿÿ# 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ# 
-__inference_sequential_3_layer_call_fn_549833e;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p

 
ª "ÿÿÿÿÿÿÿÿÿ# 
-__inference_sequential_3_layer_call_fn_549846e;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ# 
$__inference_signature_wrapper_548010Ý:"#,->;=<GDFEbclmvwj¢g
¢ 
`ª]
-
input_3"
input_3ÿÿÿÿÿÿÿÿÿR
,
input_4!
input_4ÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_15"
dense_15ÿÿÿÿÿÿÿÿÿ½
Z__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_548774_+¢(
!¢

xÿÿÿÿÿÿÿÿÿR
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿR 
 
?__inference_token_and_position_embedding_1_layer_call_fn_548783R+¢(
!¢

xÿÿÿÿÿÿÿÿÿR
ª "ÿÿÿÿÿÿÿÿÿR Ú
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_549321 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 Ú
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_549448 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ±
4__inference_transformer_block_3_layer_call_fn_549485y 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª "ÿÿÿÿÿÿÿÿÿ# ±
4__inference_transformer_block_3_layer_call_fn_549522y 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª "ÿÿÿÿÿÿÿÿÿ# 