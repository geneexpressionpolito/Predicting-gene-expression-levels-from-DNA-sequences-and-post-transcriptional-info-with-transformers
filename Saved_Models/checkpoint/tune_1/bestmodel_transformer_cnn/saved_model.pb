É´3
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
 "serve*2.4.12v2.4.1-0-g85c8b2a817f8ñ,
~
conv1d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv1d_4/kernel
w
#conv1d_4/kernel/Read/ReadVariableOpReadVariableOpconv1d_4/kernel*"
_output_shapes
:  *
dtype0
r
conv1d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_4/bias
k
!conv1d_4/bias/Read/ReadVariableOpReadVariableOpconv1d_4/bias*
_output_shapes
: *
dtype0
~
conv1d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  * 
shared_nameconv1d_5/kernel
w
#conv1d_5/kernel/Read/ReadVariableOpReadVariableOpconv1d_5/kernel*"
_output_shapes
:	  *
dtype0
r
conv1d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_5/bias
k
!conv1d_5/bias/Read/ReadVariableOpReadVariableOpconv1d_5/bias*
_output_shapes
: *
dtype0

batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_6/gamma

/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
: *
dtype0

batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_6/beta

.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
: *
dtype0

!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_6/moving_mean

5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
: *
dtype0
¢
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_6/moving_variance

9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
: *
dtype0

batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_7/gamma

/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
: *
dtype0

batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_7/beta

.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
: *
dtype0

!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_7/moving_mean

5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
: *
dtype0
¢
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_7/moving_variance

9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
: *
dtype0

batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_8/gamma

/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
:*
dtype0

batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_8/beta

.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
:*
dtype0

!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_8/moving_mean

5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
:*
dtype0
¢
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_8/moving_variance

9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
:*
dtype0
{
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	è@* 
shared_namedense_18/kernel
t
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes
:	è@*
dtype0
r
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_18/bias
k
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes
:@*
dtype0
z
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_19/kernel
s
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes

:@@*
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes
:@*
dtype0
z
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_20/kernel
s
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes

:@*
dtype0
r
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
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
5token_and_position_embedding_2/embedding_4/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75token_and_position_embedding_2/embedding_4/embeddings
¿
Itoken_and_position_embedding_2/embedding_4/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_2/embedding_4/embeddings*
_output_shapes

: *
dtype0
Ç
5token_and_position_embedding_2/embedding_5/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	R *F
shared_name75token_and_position_embedding_2/embedding_5/embeddings
À
Itoken_and_position_embedding_2/embedding_5/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_2/embedding_5/embeddings*
_output_shapes
:	R *
dtype0
Î
7transformer_block_5/multi_head_attention_5/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_5/multi_head_attention_5/query/kernel
Ç
Ktransformer_block_5/multi_head_attention_5/query/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_5/multi_head_attention_5/query/kernel*"
_output_shapes
:  *
dtype0
Æ
5transformer_block_5/multi_head_attention_5/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_5/multi_head_attention_5/query/bias
¿
Itransformer_block_5/multi_head_attention_5/query/bias/Read/ReadVariableOpReadVariableOp5transformer_block_5/multi_head_attention_5/query/bias*
_output_shapes

: *
dtype0
Ê
5transformer_block_5/multi_head_attention_5/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *F
shared_name75transformer_block_5/multi_head_attention_5/key/kernel
Ã
Itransformer_block_5/multi_head_attention_5/key/kernel/Read/ReadVariableOpReadVariableOp5transformer_block_5/multi_head_attention_5/key/kernel*"
_output_shapes
:  *
dtype0
Â
3transformer_block_5/multi_head_attention_5/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *D
shared_name53transformer_block_5/multi_head_attention_5/key/bias
»
Gtransformer_block_5/multi_head_attention_5/key/bias/Read/ReadVariableOpReadVariableOp3transformer_block_5/multi_head_attention_5/key/bias*
_output_shapes

: *
dtype0
Î
7transformer_block_5/multi_head_attention_5/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_5/multi_head_attention_5/value/kernel
Ç
Ktransformer_block_5/multi_head_attention_5/value/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_5/multi_head_attention_5/value/kernel*"
_output_shapes
:  *
dtype0
Æ
5transformer_block_5/multi_head_attention_5/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_5/multi_head_attention_5/value/bias
¿
Itransformer_block_5/multi_head_attention_5/value/bias/Read/ReadVariableOpReadVariableOp5transformer_block_5/multi_head_attention_5/value/bias*
_output_shapes

: *
dtype0
ä
Btransformer_block_5/multi_head_attention_5/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBtransformer_block_5/multi_head_attention_5/attention_output/kernel
Ý
Vtransformer_block_5/multi_head_attention_5/attention_output/kernel/Read/ReadVariableOpReadVariableOpBtransformer_block_5/multi_head_attention_5/attention_output/kernel*"
_output_shapes
:  *
dtype0
Ø
@transformer_block_5/multi_head_attention_5/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@transformer_block_5/multi_head_attention_5/attention_output/bias
Ñ
Ttransformer_block_5/multi_head_attention_5/attention_output/bias/Read/ReadVariableOpReadVariableOp@transformer_block_5/multi_head_attention_5/attention_output/bias*
_output_shapes
: *
dtype0
z
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @* 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

: @*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:@*
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

:@ *
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
: *
dtype0
¸
0transformer_block_5/layer_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20transformer_block_5/layer_normalization_10/gamma
±
Dtransformer_block_5/layer_normalization_10/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_5/layer_normalization_10/gamma*
_output_shapes
: *
dtype0
¶
/transformer_block_5/layer_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_5/layer_normalization_10/beta
¯
Ctransformer_block_5/layer_normalization_10/beta/Read/ReadVariableOpReadVariableOp/transformer_block_5/layer_normalization_10/beta*
_output_shapes
: *
dtype0
¸
0transformer_block_5/layer_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20transformer_block_5/layer_normalization_11/gamma
±
Dtransformer_block_5/layer_normalization_11/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_5/layer_normalization_11/gamma*
_output_shapes
: *
dtype0
¶
/transformer_block_5/layer_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_5/layer_normalization_11/beta
¯
Ctransformer_block_5/layer_normalization_11/beta/Read/ReadVariableOpReadVariableOp/transformer_block_5/layer_normalization_11/beta*
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

SGD/conv1d_4/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *-
shared_nameSGD/conv1d_4/kernel/momentum

0SGD/conv1d_4/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_4/kernel/momentum*"
_output_shapes
:  *
dtype0

SGD/conv1d_4/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/conv1d_4/bias/momentum

.SGD/conv1d_4/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_4/bias/momentum*
_output_shapes
: *
dtype0

SGD/conv1d_5/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *-
shared_nameSGD/conv1d_5/kernel/momentum

0SGD/conv1d_5/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_5/kernel/momentum*"
_output_shapes
:	  *
dtype0

SGD/conv1d_5/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/conv1d_5/bias/momentum

.SGD/conv1d_5/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_5/bias/momentum*
_output_shapes
: *
dtype0
¨
(SGD/batch_normalization_6/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(SGD/batch_normalization_6/gamma/momentum
¡
<SGD/batch_normalization_6/gamma/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_6/gamma/momentum*
_output_shapes
: *
dtype0
¦
'SGD/batch_normalization_6/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'SGD/batch_normalization_6/beta/momentum

;SGD/batch_normalization_6/beta/momentum/Read/ReadVariableOpReadVariableOp'SGD/batch_normalization_6/beta/momentum*
_output_shapes
: *
dtype0
¨
(SGD/batch_normalization_7/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(SGD/batch_normalization_7/gamma/momentum
¡
<SGD/batch_normalization_7/gamma/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_7/gamma/momentum*
_output_shapes
: *
dtype0
¦
'SGD/batch_normalization_7/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'SGD/batch_normalization_7/beta/momentum

;SGD/batch_normalization_7/beta/momentum/Read/ReadVariableOpReadVariableOp'SGD/batch_normalization_7/beta/momentum*
_output_shapes
: *
dtype0
¨
(SGD/batch_normalization_8/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(SGD/batch_normalization_8/gamma/momentum
¡
<SGD/batch_normalization_8/gamma/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_8/gamma/momentum*
_output_shapes
:*
dtype0
¦
'SGD/batch_normalization_8/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'SGD/batch_normalization_8/beta/momentum

;SGD/batch_normalization_8/beta/momentum/Read/ReadVariableOpReadVariableOp'SGD/batch_normalization_8/beta/momentum*
_output_shapes
:*
dtype0

SGD/dense_18/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	è@*-
shared_nameSGD/dense_18/kernel/momentum

0SGD/dense_18/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_18/kernel/momentum*
_output_shapes
:	è@*
dtype0

SGD/dense_18/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_18/bias/momentum

.SGD/dense_18/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_18/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_19/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*-
shared_nameSGD/dense_19/kernel/momentum

0SGD/dense_19/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_19/kernel/momentum*
_output_shapes

:@@*
dtype0

SGD/dense_19/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_19/bias/momentum

.SGD/dense_19/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_19/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_20/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*-
shared_nameSGD/dense_20/kernel/momentum

0SGD/dense_20/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_20/kernel/momentum*
_output_shapes

:@*
dtype0

SGD/dense_20/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/dense_20/bias/momentum

.SGD/dense_20/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_20/bias/momentum*
_output_shapes
:*
dtype0
à
BSGD/token_and_position_embedding_2/embedding_4/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/token_and_position_embedding_2/embedding_4/embeddings/momentum
Ù
VSGD/token_and_position_embedding_2/embedding_4/embeddings/momentum/Read/ReadVariableOpReadVariableOpBSGD/token_and_position_embedding_2/embedding_4/embeddings/momentum*
_output_shapes

: *
dtype0
á
BSGD/token_and_position_embedding_2/embedding_5/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	R *S
shared_nameDBSGD/token_and_position_embedding_2/embedding_5/embeddings/momentum
Ú
VSGD/token_and_position_embedding_2/embedding_5/embeddings/momentum/Read/ReadVariableOpReadVariableOpBSGD/token_and_position_embedding_2/embedding_5/embeddings/momentum*
_output_shapes
:	R *
dtype0
è
DSGD/transformer_block_5/multi_head_attention_5/query/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_5/multi_head_attention_5/query/kernel/momentum
á
XSGD/transformer_block_5/multi_head_attention_5/query/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_5/multi_head_attention_5/query/kernel/momentum*"
_output_shapes
:  *
dtype0
à
BSGD/transformer_block_5/multi_head_attention_5/query/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_5/multi_head_attention_5/query/bias/momentum
Ù
VSGD/transformer_block_5/multi_head_attention_5/query/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_5/multi_head_attention_5/query/bias/momentum*
_output_shapes

: *
dtype0
ä
BSGD/transformer_block_5/multi_head_attention_5/key/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBSGD/transformer_block_5/multi_head_attention_5/key/kernel/momentum
Ý
VSGD/transformer_block_5/multi_head_attention_5/key/kernel/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_5/multi_head_attention_5/key/kernel/momentum*"
_output_shapes
:  *
dtype0
Ü
@SGD/transformer_block_5/multi_head_attention_5/key/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *Q
shared_nameB@SGD/transformer_block_5/multi_head_attention_5/key/bias/momentum
Õ
TSGD/transformer_block_5/multi_head_attention_5/key/bias/momentum/Read/ReadVariableOpReadVariableOp@SGD/transformer_block_5/multi_head_attention_5/key/bias/momentum*
_output_shapes

: *
dtype0
è
DSGD/transformer_block_5/multi_head_attention_5/value/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_5/multi_head_attention_5/value/kernel/momentum
á
XSGD/transformer_block_5/multi_head_attention_5/value/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_5/multi_head_attention_5/value/kernel/momentum*"
_output_shapes
:  *
dtype0
à
BSGD/transformer_block_5/multi_head_attention_5/value/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_5/multi_head_attention_5/value/bias/momentum
Ù
VSGD/transformer_block_5/multi_head_attention_5/value/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_5/multi_head_attention_5/value/bias/momentum*
_output_shapes

: *
dtype0
þ
OSGD/transformer_block_5/multi_head_attention_5/attention_output/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *`
shared_nameQOSGD/transformer_block_5/multi_head_attention_5/attention_output/kernel/momentum
÷
cSGD/transformer_block_5/multi_head_attention_5/attention_output/kernel/momentum/Read/ReadVariableOpReadVariableOpOSGD/transformer_block_5/multi_head_attention_5/attention_output/kernel/momentum*"
_output_shapes
:  *
dtype0
ò
MSGD/transformer_block_5/multi_head_attention_5/attention_output/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *^
shared_nameOMSGD/transformer_block_5/multi_head_attention_5/attention_output/bias/momentum
ë
aSGD/transformer_block_5/multi_head_attention_5/attention_output/bias/momentum/Read/ReadVariableOpReadVariableOpMSGD/transformer_block_5/multi_head_attention_5/attention_output/bias/momentum*
_output_shapes
: *
dtype0

SGD/dense_16/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*-
shared_nameSGD/dense_16/kernel/momentum

0SGD/dense_16/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_16/kernel/momentum*
_output_shapes

: @*
dtype0

SGD/dense_16/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_16/bias/momentum

.SGD/dense_16/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_16/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_17/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *-
shared_nameSGD/dense_17/kernel/momentum

0SGD/dense_17/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_17/kernel/momentum*
_output_shapes

:@ *
dtype0

SGD/dense_17/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/dense_17/bias/momentum

.SGD/dense_17/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_17/bias/momentum*
_output_shapes
: *
dtype0
Ò
=SGD/transformer_block_5/layer_normalization_10/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=SGD/transformer_block_5/layer_normalization_10/gamma/momentum
Ë
QSGD/transformer_block_5/layer_normalization_10/gamma/momentum/Read/ReadVariableOpReadVariableOp=SGD/transformer_block_5/layer_normalization_10/gamma/momentum*
_output_shapes
: *
dtype0
Ð
<SGD/transformer_block_5/layer_normalization_10/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_5/layer_normalization_10/beta/momentum
É
PSGD/transformer_block_5/layer_normalization_10/beta/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_5/layer_normalization_10/beta/momentum*
_output_shapes
: *
dtype0
Ò
=SGD/transformer_block_5/layer_normalization_11/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=SGD/transformer_block_5/layer_normalization_11/gamma/momentum
Ë
QSGD/transformer_block_5/layer_normalization_11/gamma/momentum/Read/ReadVariableOpReadVariableOp=SGD/transformer_block_5/layer_normalization_11/gamma/momentum*
_output_shapes
: *
dtype0
Ð
<SGD/transformer_block_5/layer_normalization_11/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_5/layer_normalization_11/beta/momentum
É
PSGD/transformer_block_5/layer_normalization_11/beta/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_5/layer_normalization_11/beta/momentum*
_output_shapes
: *
dtype0

NoOpNoOp
ç½
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¡½
value½B½ B½

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
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer-16
layer_with_weights-8
layer-17
layer-18
layer_with_weights-9
layer-19
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
n
	token_emb
pos_emb
trainable_variables
	variables
regularization_losses
 	keras_api
h

!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
R
'trainable_variables
(	variables
)regularization_losses
*	keras_api
h

+kernel
,bias
-trainable_variables
.	variables
/regularization_losses
0	keras_api
R
1trainable_variables
2	variables
3regularization_losses
4	keras_api
R
5trainable_variables
6	variables
7regularization_losses
8	keras_api

9axis
	:gamma
;beta
<moving_mean
=moving_variance
>trainable_variables
?	variables
@regularization_losses
A	keras_api

Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
R
Ktrainable_variables
L	variables
Mregularization_losses
N	keras_api
 
Oatt
Pffn
Q
layernorm1
R
layernorm2
Sdropout1
Tdropout2
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
 
R
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api

]axis
	^gamma
_beta
`moving_mean
amoving_variance
btrainable_variables
c	variables
dregularization_losses
e	keras_api
R
ftrainable_variables
g	variables
hregularization_losses
i	keras_api
h

jkernel
kbias
ltrainable_variables
m	variables
nregularization_losses
o	keras_api
R
ptrainable_variables
q	variables
rregularization_losses
s	keras_api
h

tkernel
ubias
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
R
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
l

~kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api


decay
learning_rate
momentum
	iter!momentum "momentum¡+momentum¢,momentum£:momentum¤;momentum¥Cmomentum¦Dmomentum§^momentum¨_momentum©jmomentumªkmomentum«tmomentum¬umomentum­~momentum®momentum¯momentum°momentum±momentum²momentum³momentum´momentumµmomentum¶momentum·momentum¸momentum¹momentumºmomentum»momentum¼momentum½momentum¾momentum¿momentumÀmomentumÁ
È
0
1
!2
"3
+4
,5
:6
;7
<8
=9
C10
D11
E12
F13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
^30
_31
`32
a33
j34
k35
t36
u37
~38
39
 

0
1
!2
"3
+4
,5
:6
;7
C8
D9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
^26
_27
j28
k29
t30
u31
~32
33
²
	variables
metrics
non_trainable_variables
 layer_regularization_losses
regularization_losses
trainable_variables
layer_metrics
layers
 
g

embeddings
trainable_variables
 	variables
¡regularization_losses
¢	keras_api
g

embeddings
£trainable_variables
¤	variables
¥regularization_losses
¦	keras_api

0
1

0
1
 
²
§layers
trainable_variables
	variables
¨metrics
regularization_losses
©non_trainable_variables
ªlayer_metrics
 «layer_regularization_losses
[Y
VARIABLE_VALUEconv1d_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
²
¬layers
#trainable_variables
$	variables
­metrics
%regularization_losses
®non_trainable_variables
¯layer_metrics
 °layer_regularization_losses
 
 
 
²
±layers
'trainable_variables
(	variables
²metrics
)regularization_losses
³non_trainable_variables
´layer_metrics
 µlayer_regularization_losses
[Y
VARIABLE_VALUEconv1d_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1

+0
,1
 
²
¶layers
-trainable_variables
.	variables
·metrics
/regularization_losses
¸non_trainable_variables
¹layer_metrics
 ºlayer_regularization_losses
 
 
 
²
»layers
1trainable_variables
2	variables
¼metrics
3regularization_losses
½non_trainable_variables
¾layer_metrics
 ¿layer_regularization_losses
 
 
 
²
Àlayers
5trainable_variables
6	variables
Ámetrics
7regularization_losses
Ânon_trainable_variables
Ãlayer_metrics
 Älayer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_6/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_6/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_6/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_6/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

:0
;1

:0
;1
<2
=3
 
²
Ålayers
>trainable_variables
?	variables
Æmetrics
@regularization_losses
Çnon_trainable_variables
Èlayer_metrics
 Élayer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_7/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_7/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_7/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_7/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

C0
D1

C0
D1
E2
F3
 
²
Êlayers
Gtrainable_variables
H	variables
Ëmetrics
Iregularization_losses
Ìnon_trainable_variables
Ílayer_metrics
 Îlayer_regularization_losses
 
 
 
²
Ïlayers
Ktrainable_variables
L	variables
Ðmetrics
Mregularization_losses
Ñnon_trainable_variables
Òlayer_metrics
 Ólayer_regularization_losses
Å
Ô_query_dense
Õ
_key_dense
Ö_value_dense
×_softmax
Ø_dropout_layer
Ù_output_dense
Útrainable_variables
Û	variables
Üregularization_losses
Ý	keras_api
¨
Þlayer_with_weights-0
Þlayer-0
ßlayer_with_weights-1
ßlayer-1
à	variables
áregularization_losses
âtrainable_variables
ã	keras_api
x
	äaxis

gamma
	beta
åtrainable_variables
æ	variables
çregularization_losses
è	keras_api
x
	éaxis

gamma
	beta
êtrainable_variables
ë	variables
ìregularization_losses
í	keras_api
V
îtrainable_variables
ï	variables
ðregularization_losses
ñ	keras_api
V
òtrainable_variables
ó	variables
ôregularization_losses
õ	keras_api

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
 
²
ölayers
Utrainable_variables
V	variables
÷metrics
Wregularization_losses
ønon_trainable_variables
ùlayer_metrics
 úlayer_regularization_losses
 
 
 
²
ûlayers
Ytrainable_variables
Z	variables
ümetrics
[regularization_losses
ýnon_trainable_variables
þlayer_metrics
 ÿlayer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_8/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_8/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_8/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_8/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

^0
_1

^0
_1
`2
a3
 
²
layers
btrainable_variables
c	variables
metrics
dregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
 
 
 
²
layers
ftrainable_variables
g	variables
metrics
hregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
[Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_18/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

j0
k1

j0
k1
 
²
layers
ltrainable_variables
m	variables
metrics
nregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
 
 
 
²
layers
ptrainable_variables
q	variables
metrics
rregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
[Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_19/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

t0
u1

t0
u1
 
²
layers
vtrainable_variables
w	variables
metrics
xregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
 
 
 
²
layers
ztrainable_variables
{	variables
metrics
|regularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
[Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_20/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

~0
1

~0
1
 
µ
layers
trainable_variables
	variables
metrics
regularization_losses
 non_trainable_variables
¡layer_metrics
 ¢layer_regularization_losses
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5token_and_position_embedding_2/embedding_4/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5token_and_position_embedding_2/embedding_5/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7transformer_block_5/multi_head_attention_5/query/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_5/multi_head_attention_5/query/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_5/multi_head_attention_5/key/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE3transformer_block_5/multi_head_attention_5/key/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7transformer_block_5/multi_head_attention_5/value/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_5/multi_head_attention_5/value/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEBtransformer_block_5/multi_head_attention_5/attention_output/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE@transformer_block_5/multi_head_attention_5/attention_output/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_16/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_16/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_17/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_17/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0transformer_block_5/layer_normalization_10/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_5/layer_normalization_10/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0transformer_block_5/layer_normalization_11/gamma'variables/28/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_5/layer_normalization_11/beta'variables/29/.ATTRIBUTES/VARIABLE_VALUE

£0
*
<0
=1
E2
F3
`4
a5
 
 

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

0

0
 
µ
¤layers
trainable_variables
 	variables
¥metrics
¡regularization_losses
¦non_trainable_variables
§layer_metrics
 ¨layer_regularization_losses

0

0
 
µ
©layers
£trainable_variables
¤	variables
ªmetrics
¥regularization_losses
«non_trainable_variables
¬layer_metrics
 ­layer_regularization_losses

0
1
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

<0
=1
 
 
 
 

E0
F1
 
 
 
 
 
 
 
¡
®partial_output_shape
¯full_output_shape
kernel
	bias
°trainable_variables
±	variables
²regularization_losses
³	keras_api
¡
´partial_output_shape
µfull_output_shape
kernel
	bias
¶trainable_variables
·	variables
¸regularization_losses
¹	keras_api
¡
ºpartial_output_shape
»full_output_shape
kernel
	bias
¼trainable_variables
½	variables
¾regularization_losses
¿	keras_api
V
Àtrainable_variables
Á	variables
Âregularization_losses
Ã	keras_api
V
Ätrainable_variables
Å	variables
Æregularization_losses
Ç	keras_api
¡
Èpartial_output_shape
Éfull_output_shape
kernel
	bias
Êtrainable_variables
Ë	variables
Ìregularization_losses
Í	keras_api
@
0
1
2
3
4
5
6
7
@
0
1
2
3
4
5
6
7
 
µ
Îlayers
Útrainable_variables
Û	variables
Ïmetrics
Üregularization_losses
Ðnon_trainable_variables
Ñlayer_metrics
 Òlayer_regularization_losses
n
kernel
	bias
Ótrainable_variables
Ô	variables
Õregularization_losses
Ö	keras_api
n
kernel
	bias
×trainable_variables
Ø	variables
Ùregularization_losses
Ú	keras_api
 
0
1
2
3
 
 
0
1
2
3
µ
à	variables
Ûmetrics
Ünon_trainable_variables
 Ýlayer_regularization_losses
áregularization_losses
âtrainable_variables
Þlayer_metrics
ßlayers
 

0
1

0
1
 
µ
àlayers
åtrainable_variables
æ	variables
ámetrics
çregularization_losses
ânon_trainable_variables
ãlayer_metrics
 älayer_regularization_losses
 

0
1

0
1
 
µ
ålayers
êtrainable_variables
ë	variables
æmetrics
ìregularization_losses
çnon_trainable_variables
èlayer_metrics
 élayer_regularization_losses
 
 
 
µ
êlayers
îtrainable_variables
ï	variables
ëmetrics
ðregularization_losses
ìnon_trainable_variables
ílayer_metrics
 îlayer_regularization_losses
 
 
 
µ
ïlayers
òtrainable_variables
ó	variables
ðmetrics
ôregularization_losses
ñnon_trainable_variables
òlayer_metrics
 ólayer_regularization_losses
*
O0
P1
Q2
R3
S4
T5
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
`0
a1
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

ôtotal

õcount
ö	variables
÷	keras_api
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
0
1

0
1
 
µ
ølayers
°trainable_variables
±	variables
ùmetrics
²regularization_losses
únon_trainable_variables
ûlayer_metrics
 ülayer_regularization_losses
 
 

0
1

0
1
 
µ
ýlayers
¶trainable_variables
·	variables
þmetrics
¸regularization_losses
ÿnon_trainable_variables
layer_metrics
 layer_regularization_losses
 
 

0
1

0
1
 
µ
layers
¼trainable_variables
½	variables
metrics
¾regularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
 
 
 
µ
layers
Àtrainable_variables
Á	variables
metrics
Âregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
 
 
 
µ
layers
Ätrainable_variables
Å	variables
metrics
Æregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
 
 

0
1

0
1
 
µ
layers
Êtrainable_variables
Ë	variables
metrics
Ìregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
0
Ô0
Õ1
Ö2
×3
Ø4
Ù5
 
 
 
 

0
1

0
1
 
µ
layers
Ótrainable_variables
Ô	variables
metrics
Õregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses

0
1

0
1
 
µ
layers
×trainable_variables
Ø	variables
metrics
Ùregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
 
 
 
 

Þ0
ß1
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
ô0
õ1

ö	variables
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
VARIABLE_VALUESGD/conv1d_4/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv1d_4/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv1d_5/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv1d_5/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/batch_normalization_6/gamma/momentumXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'SGD/batch_normalization_6/beta/momentumWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/batch_normalization_7/gamma/momentumXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'SGD/batch_normalization_7/beta/momentumWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/batch_normalization_8/gamma/momentumXlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'SGD/batch_normalization_8/beta/momentumWlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_18/kernel/momentumYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_18/bias/momentumWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_19/kernel/momentumYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_19/bias/momentumWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_20/kernel/momentumYlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_20/bias/momentumWlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¢
VARIABLE_VALUEBSGD/token_and_position_embedding_2/embedding_4/embeddings/momentumIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¢
VARIABLE_VALUEBSGD/token_and_position_embedding_2/embedding_5/embeddings/momentumIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¥¢
VARIABLE_VALUEDSGD/transformer_block_5/multi_head_attention_5/query/kernel/momentumJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEBSGD/transformer_block_5/multi_head_attention_5/query/bias/momentumJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEBSGD/transformer_block_5/multi_head_attention_5/key/kernel/momentumJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¡
VARIABLE_VALUE@SGD/transformer_block_5/multi_head_attention_5/key/bias/momentumJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¥¢
VARIABLE_VALUEDSGD/transformer_block_5/multi_head_attention_5/value/kernel/momentumJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEBSGD/transformer_block_5/multi_head_attention_5/value/bias/momentumJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
°­
VARIABLE_VALUEOSGD/transformer_block_5/multi_head_attention_5/attention_output/kernel/momentumJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
®«
VARIABLE_VALUEMSGD/transformer_block_5/multi_head_attention_5/attention_output/bias/momentumJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUESGD/dense_16/kernel/momentumJvariables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUESGD/dense_16/bias/momentumJvariables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUESGD/dense_17/kernel/momentumJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUESGD/dense_17/bias/momentumJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=SGD/transformer_block_5/layer_normalization_10/gamma/momentumJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<SGD/transformer_block_5/layer_normalization_10/beta/momentumJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=SGD/transformer_block_5/layer_normalization_11/gamma/momentumJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<SGD/transformer_block_5/layer_normalization_11/beta/momentumJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_5Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿR
z
serving_default_input_6Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5serving_default_input_65token_and_position_embedding_2/embedding_5/embeddings5token_and_position_embedding_2/embedding_4/embeddingsconv1d_4/kernelconv1d_4/biasconv1d_5/kernelconv1d_5/bias%batch_normalization_6/moving_variancebatch_normalization_6/gamma!batch_normalization_6/moving_meanbatch_normalization_6/beta%batch_normalization_7/moving_variancebatch_normalization_7/gamma!batch_normalization_7/moving_meanbatch_normalization_7/beta7transformer_block_5/multi_head_attention_5/query/kernel5transformer_block_5/multi_head_attention_5/query/bias5transformer_block_5/multi_head_attention_5/key/kernel3transformer_block_5/multi_head_attention_5/key/bias7transformer_block_5/multi_head_attention_5/value/kernel5transformer_block_5/multi_head_attention_5/value/biasBtransformer_block_5/multi_head_attention_5/attention_output/kernel@transformer_block_5/multi_head_attention_5/attention_output/bias0transformer_block_5/layer_normalization_10/gamma/transformer_block_5/layer_normalization_10/betadense_16/kerneldense_16/biasdense_17/kerneldense_17/bias0transformer_block_5/layer_normalization_11/gamma/transformer_block_5/layer_normalization_11/beta%batch_normalization_8/moving_variancebatch_normalization_8/gamma!batch_normalization_8/moving_meanbatch_normalization_8/betadense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/bias*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_423083
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
 '
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1d_4/kernel/Read/ReadVariableOp!conv1d_4/bias/Read/ReadVariableOp#conv1d_5/kernel/Read/ReadVariableOp!conv1d_5/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOp#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpItoken_and_position_embedding_2/embedding_4/embeddings/Read/ReadVariableOpItoken_and_position_embedding_2/embedding_5/embeddings/Read/ReadVariableOpKtransformer_block_5/multi_head_attention_5/query/kernel/Read/ReadVariableOpItransformer_block_5/multi_head_attention_5/query/bias/Read/ReadVariableOpItransformer_block_5/multi_head_attention_5/key/kernel/Read/ReadVariableOpGtransformer_block_5/multi_head_attention_5/key/bias/Read/ReadVariableOpKtransformer_block_5/multi_head_attention_5/value/kernel/Read/ReadVariableOpItransformer_block_5/multi_head_attention_5/value/bias/Read/ReadVariableOpVtransformer_block_5/multi_head_attention_5/attention_output/kernel/Read/ReadVariableOpTtransformer_block_5/multi_head_attention_5/attention_output/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOpDtransformer_block_5/layer_normalization_10/gamma/Read/ReadVariableOpCtransformer_block_5/layer_normalization_10/beta/Read/ReadVariableOpDtransformer_block_5/layer_normalization_11/gamma/Read/ReadVariableOpCtransformer_block_5/layer_normalization_11/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp0SGD/conv1d_4/kernel/momentum/Read/ReadVariableOp.SGD/conv1d_4/bias/momentum/Read/ReadVariableOp0SGD/conv1d_5/kernel/momentum/Read/ReadVariableOp.SGD/conv1d_5/bias/momentum/Read/ReadVariableOp<SGD/batch_normalization_6/gamma/momentum/Read/ReadVariableOp;SGD/batch_normalization_6/beta/momentum/Read/ReadVariableOp<SGD/batch_normalization_7/gamma/momentum/Read/ReadVariableOp;SGD/batch_normalization_7/beta/momentum/Read/ReadVariableOp<SGD/batch_normalization_8/gamma/momentum/Read/ReadVariableOp;SGD/batch_normalization_8/beta/momentum/Read/ReadVariableOp0SGD/dense_18/kernel/momentum/Read/ReadVariableOp.SGD/dense_18/bias/momentum/Read/ReadVariableOp0SGD/dense_19/kernel/momentum/Read/ReadVariableOp.SGD/dense_19/bias/momentum/Read/ReadVariableOp0SGD/dense_20/kernel/momentum/Read/ReadVariableOp.SGD/dense_20/bias/momentum/Read/ReadVariableOpVSGD/token_and_position_embedding_2/embedding_4/embeddings/momentum/Read/ReadVariableOpVSGD/token_and_position_embedding_2/embedding_5/embeddings/momentum/Read/ReadVariableOpXSGD/transformer_block_5/multi_head_attention_5/query/kernel/momentum/Read/ReadVariableOpVSGD/transformer_block_5/multi_head_attention_5/query/bias/momentum/Read/ReadVariableOpVSGD/transformer_block_5/multi_head_attention_5/key/kernel/momentum/Read/ReadVariableOpTSGD/transformer_block_5/multi_head_attention_5/key/bias/momentum/Read/ReadVariableOpXSGD/transformer_block_5/multi_head_attention_5/value/kernel/momentum/Read/ReadVariableOpVSGD/transformer_block_5/multi_head_attention_5/value/bias/momentum/Read/ReadVariableOpcSGD/transformer_block_5/multi_head_attention_5/attention_output/kernel/momentum/Read/ReadVariableOpaSGD/transformer_block_5/multi_head_attention_5/attention_output/bias/momentum/Read/ReadVariableOp0SGD/dense_16/kernel/momentum/Read/ReadVariableOp.SGD/dense_16/bias/momentum/Read/ReadVariableOp0SGD/dense_17/kernel/momentum/Read/ReadVariableOp.SGD/dense_17/bias/momentum/Read/ReadVariableOpQSGD/transformer_block_5/layer_normalization_10/gamma/momentum/Read/ReadVariableOpPSGD/transformer_block_5/layer_normalization_10/beta/momentum/Read/ReadVariableOpQSGD/transformer_block_5/layer_normalization_11/gamma/momentum/Read/ReadVariableOpPSGD/transformer_block_5/layer_normalization_11/beta/momentum/Read/ReadVariableOpConst*]
TinV
T2R	*
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
__inference__traced_save_425330
Û
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_4/kernelconv1d_4/biasconv1d_5/kernelconv1d_5/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancebatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancebatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variancedense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/biasdecaylearning_ratemomentumSGD/iter5token_and_position_embedding_2/embedding_4/embeddings5token_and_position_embedding_2/embedding_5/embeddings7transformer_block_5/multi_head_attention_5/query/kernel5transformer_block_5/multi_head_attention_5/query/bias5transformer_block_5/multi_head_attention_5/key/kernel3transformer_block_5/multi_head_attention_5/key/bias7transformer_block_5/multi_head_attention_5/value/kernel5transformer_block_5/multi_head_attention_5/value/biasBtransformer_block_5/multi_head_attention_5/attention_output/kernel@transformer_block_5/multi_head_attention_5/attention_output/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias0transformer_block_5/layer_normalization_10/gamma/transformer_block_5/layer_normalization_10/beta0transformer_block_5/layer_normalization_11/gamma/transformer_block_5/layer_normalization_11/betatotalcountSGD/conv1d_4/kernel/momentumSGD/conv1d_4/bias/momentumSGD/conv1d_5/kernel/momentumSGD/conv1d_5/bias/momentum(SGD/batch_normalization_6/gamma/momentum'SGD/batch_normalization_6/beta/momentum(SGD/batch_normalization_7/gamma/momentum'SGD/batch_normalization_7/beta/momentum(SGD/batch_normalization_8/gamma/momentum'SGD/batch_normalization_8/beta/momentumSGD/dense_18/kernel/momentumSGD/dense_18/bias/momentumSGD/dense_19/kernel/momentumSGD/dense_19/bias/momentumSGD/dense_20/kernel/momentumSGD/dense_20/bias/momentumBSGD/token_and_position_embedding_2/embedding_4/embeddings/momentumBSGD/token_and_position_embedding_2/embedding_5/embeddings/momentumDSGD/transformer_block_5/multi_head_attention_5/query/kernel/momentumBSGD/transformer_block_5/multi_head_attention_5/query/bias/momentumBSGD/transformer_block_5/multi_head_attention_5/key/kernel/momentum@SGD/transformer_block_5/multi_head_attention_5/key/bias/momentumDSGD/transformer_block_5/multi_head_attention_5/value/kernel/momentumBSGD/transformer_block_5/multi_head_attention_5/value/bias/momentumOSGD/transformer_block_5/multi_head_attention_5/attention_output/kernel/momentumMSGD/transformer_block_5/multi_head_attention_5/attention_output/bias/momentumSGD/dense_16/kernel/momentumSGD/dense_16/bias/momentumSGD/dense_17/kernel/momentumSGD/dense_17/bias/momentum=SGD/transformer_block_5/layer_normalization_10/gamma/momentum<SGD/transformer_block_5/layer_normalization_10/beta/momentum=SGD/transformer_block_5/layer_normalization_11/gamma/momentum<SGD/transformer_block_5/layer_normalization_11/beta/momentum*\
TinU
S2Q*
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
"__inference__traced_restore_425580¯)
î	
Ý
D__inference_dense_19_layer_call_and_return_conditional_losses_422434

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
Â
u
I__inference_concatenate_2_layer_call_and_return_conditional_losses_424728
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

e
F__inference_dropout_17_layer_call_and_return_conditional_losses_422462

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
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
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
ì
©
6__inference_batch_normalization_6_layer_call_fn_424008

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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4211172
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
`
Ö
C__inference_model_2_layer_call_and_return_conditional_losses_422610
input_5
input_6)
%token_and_position_embedding_2_422511)
%token_and_position_embedding_2_422513
conv1d_4_422516
conv1d_4_422518
conv1d_5_422522
conv1d_5_422524 
batch_normalization_6_422529 
batch_normalization_6_422531 
batch_normalization_6_422533 
batch_normalization_6_422535 
batch_normalization_7_422538 
batch_normalization_7_422540 
batch_normalization_7_422542 
batch_normalization_7_422544
transformer_block_5_422548
transformer_block_5_422550
transformer_block_5_422552
transformer_block_5_422554
transformer_block_5_422556
transformer_block_5_422558
transformer_block_5_422560
transformer_block_5_422562
transformer_block_5_422564
transformer_block_5_422566
transformer_block_5_422568
transformer_block_5_422570
transformer_block_5_422572
transformer_block_5_422574
transformer_block_5_422576
transformer_block_5_422578 
batch_normalization_8_422582 
batch_normalization_8_422584 
batch_normalization_8_422586 
batch_normalization_8_422588
dense_18_422592
dense_18_422594
dense_19_422598
dense_19_422600
dense_20_422604
dense_20_422606
identity¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢ conv1d_4/StatefulPartitionedCall¢ conv1d_5/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢ dense_20/StatefulPartitionedCall¢6token_and_position_embedding_2/StatefulPartitionedCall¢+transformer_block_5/StatefulPartitionedCall
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_5%token_and_position_embedding_2_422511%token_and_position_embedding_2_422513*
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
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_42163728
6token_and_position_embedding_2/StatefulPartitionedCallÕ
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0conv1d_4_422516conv1d_4_422518*
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
GPU2*0J 8 *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_4216692"
 conv1d_4/StatefulPartitionedCall 
#average_pooling1d_6/PartitionedCallPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_6_layer_call_and_return_conditional_losses_4209852%
#average_pooling1d_6/PartitionedCallÂ
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_6/PartitionedCall:output:0conv1d_5_422522conv1d_5_422524*
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
D__inference_conv1d_5_layer_call_and_return_conditional_losses_4217022"
 conv1d_5/StatefulPartitionedCallµ
#average_pooling1d_8/PartitionedCallPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0*
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
O__inference_average_pooling1d_8_layer_call_and_return_conditional_losses_4210152%
#average_pooling1d_8/PartitionedCall
#average_pooling1d_7/PartitionedCallPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0*
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
O__inference_average_pooling1d_7_layer_call_and_return_conditional_losses_4210002%
#average_pooling1d_7/PartitionedCallÂ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_7/PartitionedCall:output:0batch_normalization_6_422529batch_normalization_6_422531batch_normalization_6_422533batch_normalization_6_422535*
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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4217752/
-batch_normalization_6/StatefulPartitionedCallÂ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_8/PartitionedCall:output:0batch_normalization_7_422538batch_normalization_7_422540batch_normalization_7_422542batch_normalization_7_422544*
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4218662/
-batch_normalization_7/StatefulPartitionedCall»
add_2/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
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
A__inference_add_2_layer_call_and_return_conditional_losses_4219082
add_2/PartitionedCall
+transformer_block_5/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0transformer_block_5_422548transformer_block_5_422550transformer_block_5_422552transformer_block_5_422554transformer_block_5_422556transformer_block_5_422558transformer_block_5_422560transformer_block_5_422562transformer_block_5_422564transformer_block_5_422566transformer_block_5_422568transformer_block_5_422570transformer_block_5_422572transformer_block_5_422574transformer_block_5_422576transformer_block_5_422578*
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
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_4221922-
+transformer_block_5/StatefulPartitionedCall
flatten_2/PartitionedCallPartitionedCall4transformer_block_5/StatefulPartitionedCall:output:0*
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
E__inference_flatten_2_layer_call_and_return_conditional_losses_4223072
flatten_2/PartitionedCall
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCallinput_6batch_normalization_8_422582batch_normalization_8_422584batch_normalization_8_422586batch_normalization_8_422588*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4215972/
-batch_normalization_8/StatefulPartitionedCall¼
concatenate_2/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:06batch_normalization_8/StatefulPartitionedCall:output:0*
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
I__inference_concatenate_2_layer_call_and_return_conditional_losses_4223572
concatenate_2/PartitionedCall·
 dense_18/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_18_422592dense_18_422594*
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
D__inference_dense_18_layer_call_and_return_conditional_losses_4223772"
 dense_18/StatefulPartitionedCall
dropout_16/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
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
F__inference_dropout_16_layer_call_and_return_conditional_losses_4224102
dropout_16/PartitionedCall´
 dense_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0dense_19_422598dense_19_422600*
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
D__inference_dense_19_layer_call_and_return_conditional_losses_4224342"
 dense_19/StatefulPartitionedCall
dropout_17/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
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
F__inference_dropout_17_layer_call_and_return_conditional_losses_4224672
dropout_17/PartitionedCall´
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0dense_20_422604dense_20_422606*
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
D__inference_dense_20_layer_call_and_return_conditional_losses_4224902"
 dense_20/StatefulPartitionedCall£
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_5/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_5/StatefulPartitionedCall+transformer_block_5/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_5:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6
î	
Ý
D__inference_dense_19_layer_call_and_return_conditional_losses_424792

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
± 
ã
D__inference_dense_16_layer_call_and_return_conditional_losses_425018

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
	
Ý
D__inference_dense_20_layer_call_and_return_conditional_losses_424838

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
¹Þ
â
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_424554

inputsF
Bmulti_head_attention_5_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_5_query_add_readvariableop_resourceD
@multi_head_attention_5_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_5_key_add_readvariableop_resourceF
Bmulti_head_attention_5_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_5_value_add_readvariableop_resourceQ
Mmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_5_attention_output_add_readvariableop_resource@
<layer_normalization_10_batchnorm_mul_readvariableop_resource<
8layer_normalization_10_batchnorm_readvariableop_resource;
7sequential_5_dense_16_tensordot_readvariableop_resource9
5sequential_5_dense_16_biasadd_readvariableop_resource;
7sequential_5_dense_17_tensordot_readvariableop_resource9
5sequential_5_dense_17_biasadd_readvariableop_resource@
<layer_normalization_11_batchnorm_mul_readvariableop_resource<
8layer_normalization_11_batchnorm_readvariableop_resource
identity¢/layer_normalization_10/batchnorm/ReadVariableOp¢3layer_normalization_10/batchnorm/mul/ReadVariableOp¢/layer_normalization_11/batchnorm/ReadVariableOp¢3layer_normalization_11/batchnorm/mul/ReadVariableOp¢:multi_head_attention_5/attention_output/add/ReadVariableOp¢Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_5/key/add/ReadVariableOp¢7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_5/query/add/ReadVariableOp¢9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_5/value/add/ReadVariableOp¢9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp¢,sequential_5/dense_16/BiasAdd/ReadVariableOp¢.sequential_5/dense_16/Tensordot/ReadVariableOp¢,sequential_5/dense_17/BiasAdd/ReadVariableOp¢.sequential_5/dense_17/Tensordot/ReadVariableOpý
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_5/query/einsum/EinsumEinsuminputsAmulti_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_5/query/einsum/EinsumÛ
/multi_head_attention_5/query/add/ReadVariableOpReadVariableOp8multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/query/add/ReadVariableOpõ
 multi_head_attention_5/query/addAddV23multi_head_attention_5/query/einsum/Einsum:output:07multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_5/query/add÷
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_5/key/einsum/EinsumEinsuminputs?multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2*
(multi_head_attention_5/key/einsum/EinsumÕ
-multi_head_attention_5/key/add/ReadVariableOpReadVariableOp6multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_5/key/add/ReadVariableOpí
multi_head_attention_5/key/addAddV21multi_head_attention_5/key/einsum/Einsum:output:05multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
multi_head_attention_5/key/addý
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_5/value/einsum/EinsumEinsuminputsAmulti_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_5/value/einsum/EinsumÛ
/multi_head_attention_5/value/add/ReadVariableOpReadVariableOp8multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/value/add/ReadVariableOpõ
 multi_head_attention_5/value/addAddV23multi_head_attention_5/value/einsum/Einsum:output:07multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_5/value/add
multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_5/Mul/yÆ
multi_head_attention_5/MulMul$multi_head_attention_5/query/add:z:0%multi_head_attention_5/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_5/Mulü
$multi_head_attention_5/einsum/EinsumEinsum"multi_head_attention_5/key/add:z:0multi_head_attention_5/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2&
$multi_head_attention_5/einsum/EinsumÄ
&multi_head_attention_5/softmax/SoftmaxSoftmax-multi_head_attention_5/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2(
&multi_head_attention_5/softmax/SoftmaxÊ
'multi_head_attention_5/dropout/IdentityIdentity0multi_head_attention_5/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2)
'multi_head_attention_5/dropout/Identity
&multi_head_attention_5/einsum_1/EinsumEinsum0multi_head_attention_5/dropout/Identity:output:0$multi_head_attention_5/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2(
&multi_head_attention_5/einsum_1/Einsum
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_5/attention_output/einsum/EinsumEinsum/multi_head_attention_5/einsum_1/Einsum:output:0Lmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe27
5multi_head_attention_5/attention_output/einsum/Einsumø
:multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_5/attention_output/add/ReadVariableOp
+multi_head_attention_5/attention_output/addAddV2>multi_head_attention_5/attention_output/einsum/Einsum:output:0Bmulti_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+multi_head_attention_5/attention_output/add
dropout_14/IdentityIdentity/multi_head_attention_5/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_14/Identityo
addAddV2inputsdropout_14/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¸
5layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_10/moments/mean/reduction_indicesâ
#layer_normalization_10/moments/meanMeanadd:z:0>layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_10/moments/meanÎ
+layer_normalization_10/moments/StopGradientStopGradient,layer_normalization_10/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_10/moments/StopGradientî
0layer_normalization_10/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_10/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_10/moments/SquaredDifferenceÀ
9layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_10/moments/variance/reduction_indices
'layer_normalization_10/moments/varianceMean4layer_normalization_10/moments/SquaredDifference:z:0Blayer_normalization_10/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_10/moments/variance
&layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_10/batchnorm/add/yî
$layer_normalization_10/batchnorm/addAddV20layer_normalization_10/moments/variance:output:0/layer_normalization_10/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_10/batchnorm/add¹
&layer_normalization_10/batchnorm/RsqrtRsqrt(layer_normalization_10/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_10/batchnorm/Rsqrtã
3layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_10/batchnorm/mul/ReadVariableOpò
$layer_normalization_10/batchnorm/mulMul*layer_normalization_10/batchnorm/Rsqrt:y:0;layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_10/batchnorm/mulÀ
&layer_normalization_10/batchnorm/mul_1Muladd:z:0(layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_10/batchnorm/mul_1å
&layer_normalization_10/batchnorm/mul_2Mul,layer_normalization_10/moments/mean:output:0(layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_10/batchnorm/mul_2×
/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_10/batchnorm/ReadVariableOpî
$layer_normalization_10/batchnorm/subSub7layer_normalization_10/batchnorm/ReadVariableOp:value:0*layer_normalization_10/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_10/batchnorm/subå
&layer_normalization_10/batchnorm/add_1AddV2*layer_normalization_10/batchnorm/mul_1:z:0(layer_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_10/batchnorm/add_1Ø
.sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_5/dense_16/Tensordot/ReadVariableOp
$sequential_5/dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_5/dense_16/Tensordot/axes
$sequential_5/dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_5/dense_16/Tensordot/free¨
%sequential_5/dense_16/Tensordot/ShapeShape*layer_normalization_10/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_5/dense_16/Tensordot/Shape 
-sequential_5/dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_16/Tensordot/GatherV2/axis¿
(sequential_5/dense_16/Tensordot/GatherV2GatherV2.sequential_5/dense_16/Tensordot/Shape:output:0-sequential_5/dense_16/Tensordot/free:output:06sequential_5/dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_5/dense_16/Tensordot/GatherV2¤
/sequential_5/dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_5/dense_16/Tensordot/GatherV2_1/axisÅ
*sequential_5/dense_16/Tensordot/GatherV2_1GatherV2.sequential_5/dense_16/Tensordot/Shape:output:0-sequential_5/dense_16/Tensordot/axes:output:08sequential_5/dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_5/dense_16/Tensordot/GatherV2_1
%sequential_5/dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_5/dense_16/Tensordot/ConstØ
$sequential_5/dense_16/Tensordot/ProdProd1sequential_5/dense_16/Tensordot/GatherV2:output:0.sequential_5/dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_5/dense_16/Tensordot/Prod
'sequential_5/dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_16/Tensordot/Const_1à
&sequential_5/dense_16/Tensordot/Prod_1Prod3sequential_5/dense_16/Tensordot/GatherV2_1:output:00sequential_5/dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_5/dense_16/Tensordot/Prod_1
+sequential_5/dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_5/dense_16/Tensordot/concat/axis
&sequential_5/dense_16/Tensordot/concatConcatV2-sequential_5/dense_16/Tensordot/free:output:0-sequential_5/dense_16/Tensordot/axes:output:04sequential_5/dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/dense_16/Tensordot/concatä
%sequential_5/dense_16/Tensordot/stackPack-sequential_5/dense_16/Tensordot/Prod:output:0/sequential_5/dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/dense_16/Tensordot/stackö
)sequential_5/dense_16/Tensordot/transpose	Transpose*layer_normalization_10/batchnorm/add_1:z:0/sequential_5/dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2+
)sequential_5/dense_16/Tensordot/transpose÷
'sequential_5/dense_16/Tensordot/ReshapeReshape-sequential_5/dense_16/Tensordot/transpose:y:0.sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_5/dense_16/Tensordot/Reshapeö
&sequential_5/dense_16/Tensordot/MatMulMatMul0sequential_5/dense_16/Tensordot/Reshape:output:06sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential_5/dense_16/Tensordot/MatMul
'sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_5/dense_16/Tensordot/Const_2 
-sequential_5/dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_16/Tensordot/concat_1/axis«
(sequential_5/dense_16/Tensordot/concat_1ConcatV21sequential_5/dense_16/Tensordot/GatherV2:output:00sequential_5/dense_16/Tensordot/Const_2:output:06sequential_5/dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_5/dense_16/Tensordot/concat_1è
sequential_5/dense_16/TensordotReshape0sequential_5/dense_16/Tensordot/MatMul:product:01sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2!
sequential_5/dense_16/TensordotÎ
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOpß
sequential_5/dense_16/BiasAddBiasAdd(sequential_5/dense_16/Tensordot:output:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_5/dense_16/BiasAdd
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_5/dense_16/ReluØ
.sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_5/dense_17/Tensordot/ReadVariableOp
$sequential_5/dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_5/dense_17/Tensordot/axes
$sequential_5/dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_5/dense_17/Tensordot/free¦
%sequential_5/dense_17/Tensordot/ShapeShape(sequential_5/dense_16/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_5/dense_17/Tensordot/Shape 
-sequential_5/dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_17/Tensordot/GatherV2/axis¿
(sequential_5/dense_17/Tensordot/GatherV2GatherV2.sequential_5/dense_17/Tensordot/Shape:output:0-sequential_5/dense_17/Tensordot/free:output:06sequential_5/dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_5/dense_17/Tensordot/GatherV2¤
/sequential_5/dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_5/dense_17/Tensordot/GatherV2_1/axisÅ
*sequential_5/dense_17/Tensordot/GatherV2_1GatherV2.sequential_5/dense_17/Tensordot/Shape:output:0-sequential_5/dense_17/Tensordot/axes:output:08sequential_5/dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_5/dense_17/Tensordot/GatherV2_1
%sequential_5/dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_5/dense_17/Tensordot/ConstØ
$sequential_5/dense_17/Tensordot/ProdProd1sequential_5/dense_17/Tensordot/GatherV2:output:0.sequential_5/dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_5/dense_17/Tensordot/Prod
'sequential_5/dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_17/Tensordot/Const_1à
&sequential_5/dense_17/Tensordot/Prod_1Prod3sequential_5/dense_17/Tensordot/GatherV2_1:output:00sequential_5/dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_5/dense_17/Tensordot/Prod_1
+sequential_5/dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_5/dense_17/Tensordot/concat/axis
&sequential_5/dense_17/Tensordot/concatConcatV2-sequential_5/dense_17/Tensordot/free:output:0-sequential_5/dense_17/Tensordot/axes:output:04sequential_5/dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/dense_17/Tensordot/concatä
%sequential_5/dense_17/Tensordot/stackPack-sequential_5/dense_17/Tensordot/Prod:output:0/sequential_5/dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/dense_17/Tensordot/stackô
)sequential_5/dense_17/Tensordot/transpose	Transpose(sequential_5/dense_16/Relu:activations:0/sequential_5/dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2+
)sequential_5/dense_17/Tensordot/transpose÷
'sequential_5/dense_17/Tensordot/ReshapeReshape-sequential_5/dense_17/Tensordot/transpose:y:0.sequential_5/dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_5/dense_17/Tensordot/Reshapeö
&sequential_5/dense_17/Tensordot/MatMulMatMul0sequential_5/dense_17/Tensordot/Reshape:output:06sequential_5/dense_17/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_5/dense_17/Tensordot/MatMul
'sequential_5/dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_17/Tensordot/Const_2 
-sequential_5/dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_17/Tensordot/concat_1/axis«
(sequential_5/dense_17/Tensordot/concat_1ConcatV21sequential_5/dense_17/Tensordot/GatherV2:output:00sequential_5/dense_17/Tensordot/Const_2:output:06sequential_5/dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_5/dense_17/Tensordot/concat_1è
sequential_5/dense_17/TensordotReshape0sequential_5/dense_17/Tensordot/MatMul:product:01sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
sequential_5/dense_17/TensordotÎ
,sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_17/BiasAdd/ReadVariableOpß
sequential_5/dense_17/BiasAddBiasAdd(sequential_5/dense_17/Tensordot:output:04sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
sequential_5/dense_17/BiasAdd
dropout_15/IdentityIdentity&sequential_5/dense_17/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_15/Identity
add_1AddV2*layer_normalization_10/batchnorm/add_1:z:0dropout_15/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¸
5layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_11/moments/mean/reduction_indicesä
#layer_normalization_11/moments/meanMean	add_1:z:0>layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_11/moments/meanÎ
+layer_normalization_11/moments/StopGradientStopGradient,layer_normalization_11/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_11/moments/StopGradientð
0layer_normalization_11/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_11/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_11/moments/SquaredDifferenceÀ
9layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_11/moments/variance/reduction_indices
'layer_normalization_11/moments/varianceMean4layer_normalization_11/moments/SquaredDifference:z:0Blayer_normalization_11/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_11/moments/variance
&layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_11/batchnorm/add/yî
$layer_normalization_11/batchnorm/addAddV20layer_normalization_11/moments/variance:output:0/layer_normalization_11/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_11/batchnorm/add¹
&layer_normalization_11/batchnorm/RsqrtRsqrt(layer_normalization_11/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_11/batchnorm/Rsqrtã
3layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_11/batchnorm/mul/ReadVariableOpò
$layer_normalization_11/batchnorm/mulMul*layer_normalization_11/batchnorm/Rsqrt:y:0;layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_11/batchnorm/mulÂ
&layer_normalization_11/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_11/batchnorm/mul_1å
&layer_normalization_11/batchnorm/mul_2Mul,layer_normalization_11/moments/mean:output:0(layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_11/batchnorm/mul_2×
/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_11/batchnorm/ReadVariableOpî
$layer_normalization_11/batchnorm/subSub7layer_normalization_11/batchnorm/ReadVariableOp:value:0*layer_normalization_11/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_11/batchnorm/subå
&layer_normalization_11/batchnorm/add_1AddV2*layer_normalization_11/batchnorm/mul_1:z:0(layer_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_11/batchnorm/add_1Ü
IdentityIdentity*layer_normalization_11/batchnorm/add_1:z:00^layer_normalization_10/batchnorm/ReadVariableOp4^layer_normalization_10/batchnorm/mul/ReadVariableOp0^layer_normalization_11/batchnorm/ReadVariableOp4^layer_normalization_11/batchnorm/mul/ReadVariableOp;^multi_head_attention_5/attention_output/add/ReadVariableOpE^multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_5/key/add/ReadVariableOp8^multi_head_attention_5/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/query/add/ReadVariableOp:^multi_head_attention_5/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/value/add/ReadVariableOp:^multi_head_attention_5/value/einsum/Einsum/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp/^sequential_5/dense_16/Tensordot/ReadVariableOp-^sequential_5/dense_17/BiasAdd/ReadVariableOp/^sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2b
/layer_normalization_10/batchnorm/ReadVariableOp/layer_normalization_10/batchnorm/ReadVariableOp2j
3layer_normalization_10/batchnorm/mul/ReadVariableOp3layer_normalization_10/batchnorm/mul/ReadVariableOp2b
/layer_normalization_11/batchnorm/ReadVariableOp/layer_normalization_11/batchnorm/ReadVariableOp2j
3layer_normalization_11/batchnorm/mul/ReadVariableOp3layer_normalization_11/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_5/attention_output/add/ReadVariableOp:multi_head_attention_5/attention_output/add/ReadVariableOp2
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_5/key/add/ReadVariableOp-multi_head_attention_5/key/add/ReadVariableOp2r
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/query/add/ReadVariableOp/multi_head_attention_5/query/add/ReadVariableOp2v
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/value/add/ReadVariableOp/multi_head_attention_5/value/add/ReadVariableOp2v
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2`
.sequential_5/dense_16/Tensordot/ReadVariableOp.sequential_5/dense_16/Tensordot/ReadVariableOp2\
,sequential_5/dense_17/BiasAdd/ReadVariableOp,sequential_5/dense_17/BiasAdd/ReadVariableOp2`
.sequential_5/dense_17/Tensordot/ReadVariableOp.sequential_5/dense_17/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
ï
~
)__inference_dense_16_layer_call_fn_425027

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
D__inference_dense_16_layer_call_and_return_conditional_losses_4213362
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
ó0
È
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_424139

inputs
assignmovingavg_424114
assignmovingavg_1_424120)
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
loc:@AssignMovingAvg/424114*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_424114*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/424114*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/424114*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_424114AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/424114*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/424120*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_424120*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/424120*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/424120*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_424120AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/424120*
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
F__inference_dropout_17_layer_call_and_return_conditional_losses_422467

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
¹Þ
â
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_422192

inputsF
Bmulti_head_attention_5_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_5_query_add_readvariableop_resourceD
@multi_head_attention_5_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_5_key_add_readvariableop_resourceF
Bmulti_head_attention_5_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_5_value_add_readvariableop_resourceQ
Mmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_5_attention_output_add_readvariableop_resource@
<layer_normalization_10_batchnorm_mul_readvariableop_resource<
8layer_normalization_10_batchnorm_readvariableop_resource;
7sequential_5_dense_16_tensordot_readvariableop_resource9
5sequential_5_dense_16_biasadd_readvariableop_resource;
7sequential_5_dense_17_tensordot_readvariableop_resource9
5sequential_5_dense_17_biasadd_readvariableop_resource@
<layer_normalization_11_batchnorm_mul_readvariableop_resource<
8layer_normalization_11_batchnorm_readvariableop_resource
identity¢/layer_normalization_10/batchnorm/ReadVariableOp¢3layer_normalization_10/batchnorm/mul/ReadVariableOp¢/layer_normalization_11/batchnorm/ReadVariableOp¢3layer_normalization_11/batchnorm/mul/ReadVariableOp¢:multi_head_attention_5/attention_output/add/ReadVariableOp¢Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_5/key/add/ReadVariableOp¢7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_5/query/add/ReadVariableOp¢9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_5/value/add/ReadVariableOp¢9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp¢,sequential_5/dense_16/BiasAdd/ReadVariableOp¢.sequential_5/dense_16/Tensordot/ReadVariableOp¢,sequential_5/dense_17/BiasAdd/ReadVariableOp¢.sequential_5/dense_17/Tensordot/ReadVariableOpý
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_5/query/einsum/EinsumEinsuminputsAmulti_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_5/query/einsum/EinsumÛ
/multi_head_attention_5/query/add/ReadVariableOpReadVariableOp8multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/query/add/ReadVariableOpõ
 multi_head_attention_5/query/addAddV23multi_head_attention_5/query/einsum/Einsum:output:07multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_5/query/add÷
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_5/key/einsum/EinsumEinsuminputs?multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2*
(multi_head_attention_5/key/einsum/EinsumÕ
-multi_head_attention_5/key/add/ReadVariableOpReadVariableOp6multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_5/key/add/ReadVariableOpí
multi_head_attention_5/key/addAddV21multi_head_attention_5/key/einsum/Einsum:output:05multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
multi_head_attention_5/key/addý
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_5/value/einsum/EinsumEinsuminputsAmulti_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_5/value/einsum/EinsumÛ
/multi_head_attention_5/value/add/ReadVariableOpReadVariableOp8multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/value/add/ReadVariableOpõ
 multi_head_attention_5/value/addAddV23multi_head_attention_5/value/einsum/Einsum:output:07multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_5/value/add
multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_5/Mul/yÆ
multi_head_attention_5/MulMul$multi_head_attention_5/query/add:z:0%multi_head_attention_5/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_5/Mulü
$multi_head_attention_5/einsum/EinsumEinsum"multi_head_attention_5/key/add:z:0multi_head_attention_5/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2&
$multi_head_attention_5/einsum/EinsumÄ
&multi_head_attention_5/softmax/SoftmaxSoftmax-multi_head_attention_5/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2(
&multi_head_attention_5/softmax/SoftmaxÊ
'multi_head_attention_5/dropout/IdentityIdentity0multi_head_attention_5/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2)
'multi_head_attention_5/dropout/Identity
&multi_head_attention_5/einsum_1/EinsumEinsum0multi_head_attention_5/dropout/Identity:output:0$multi_head_attention_5/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2(
&multi_head_attention_5/einsum_1/Einsum
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_5/attention_output/einsum/EinsumEinsum/multi_head_attention_5/einsum_1/Einsum:output:0Lmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe27
5multi_head_attention_5/attention_output/einsum/Einsumø
:multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_5/attention_output/add/ReadVariableOp
+multi_head_attention_5/attention_output/addAddV2>multi_head_attention_5/attention_output/einsum/Einsum:output:0Bmulti_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+multi_head_attention_5/attention_output/add
dropout_14/IdentityIdentity/multi_head_attention_5/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_14/Identityo
addAddV2inputsdropout_14/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¸
5layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_10/moments/mean/reduction_indicesâ
#layer_normalization_10/moments/meanMeanadd:z:0>layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_10/moments/meanÎ
+layer_normalization_10/moments/StopGradientStopGradient,layer_normalization_10/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_10/moments/StopGradientî
0layer_normalization_10/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_10/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_10/moments/SquaredDifferenceÀ
9layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_10/moments/variance/reduction_indices
'layer_normalization_10/moments/varianceMean4layer_normalization_10/moments/SquaredDifference:z:0Blayer_normalization_10/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_10/moments/variance
&layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_10/batchnorm/add/yî
$layer_normalization_10/batchnorm/addAddV20layer_normalization_10/moments/variance:output:0/layer_normalization_10/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_10/batchnorm/add¹
&layer_normalization_10/batchnorm/RsqrtRsqrt(layer_normalization_10/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_10/batchnorm/Rsqrtã
3layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_10/batchnorm/mul/ReadVariableOpò
$layer_normalization_10/batchnorm/mulMul*layer_normalization_10/batchnorm/Rsqrt:y:0;layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_10/batchnorm/mulÀ
&layer_normalization_10/batchnorm/mul_1Muladd:z:0(layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_10/batchnorm/mul_1å
&layer_normalization_10/batchnorm/mul_2Mul,layer_normalization_10/moments/mean:output:0(layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_10/batchnorm/mul_2×
/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_10/batchnorm/ReadVariableOpî
$layer_normalization_10/batchnorm/subSub7layer_normalization_10/batchnorm/ReadVariableOp:value:0*layer_normalization_10/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_10/batchnorm/subå
&layer_normalization_10/batchnorm/add_1AddV2*layer_normalization_10/batchnorm/mul_1:z:0(layer_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_10/batchnorm/add_1Ø
.sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_5/dense_16/Tensordot/ReadVariableOp
$sequential_5/dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_5/dense_16/Tensordot/axes
$sequential_5/dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_5/dense_16/Tensordot/free¨
%sequential_5/dense_16/Tensordot/ShapeShape*layer_normalization_10/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_5/dense_16/Tensordot/Shape 
-sequential_5/dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_16/Tensordot/GatherV2/axis¿
(sequential_5/dense_16/Tensordot/GatherV2GatherV2.sequential_5/dense_16/Tensordot/Shape:output:0-sequential_5/dense_16/Tensordot/free:output:06sequential_5/dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_5/dense_16/Tensordot/GatherV2¤
/sequential_5/dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_5/dense_16/Tensordot/GatherV2_1/axisÅ
*sequential_5/dense_16/Tensordot/GatherV2_1GatherV2.sequential_5/dense_16/Tensordot/Shape:output:0-sequential_5/dense_16/Tensordot/axes:output:08sequential_5/dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_5/dense_16/Tensordot/GatherV2_1
%sequential_5/dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_5/dense_16/Tensordot/ConstØ
$sequential_5/dense_16/Tensordot/ProdProd1sequential_5/dense_16/Tensordot/GatherV2:output:0.sequential_5/dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_5/dense_16/Tensordot/Prod
'sequential_5/dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_16/Tensordot/Const_1à
&sequential_5/dense_16/Tensordot/Prod_1Prod3sequential_5/dense_16/Tensordot/GatherV2_1:output:00sequential_5/dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_5/dense_16/Tensordot/Prod_1
+sequential_5/dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_5/dense_16/Tensordot/concat/axis
&sequential_5/dense_16/Tensordot/concatConcatV2-sequential_5/dense_16/Tensordot/free:output:0-sequential_5/dense_16/Tensordot/axes:output:04sequential_5/dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/dense_16/Tensordot/concatä
%sequential_5/dense_16/Tensordot/stackPack-sequential_5/dense_16/Tensordot/Prod:output:0/sequential_5/dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/dense_16/Tensordot/stackö
)sequential_5/dense_16/Tensordot/transpose	Transpose*layer_normalization_10/batchnorm/add_1:z:0/sequential_5/dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2+
)sequential_5/dense_16/Tensordot/transpose÷
'sequential_5/dense_16/Tensordot/ReshapeReshape-sequential_5/dense_16/Tensordot/transpose:y:0.sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_5/dense_16/Tensordot/Reshapeö
&sequential_5/dense_16/Tensordot/MatMulMatMul0sequential_5/dense_16/Tensordot/Reshape:output:06sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential_5/dense_16/Tensordot/MatMul
'sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_5/dense_16/Tensordot/Const_2 
-sequential_5/dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_16/Tensordot/concat_1/axis«
(sequential_5/dense_16/Tensordot/concat_1ConcatV21sequential_5/dense_16/Tensordot/GatherV2:output:00sequential_5/dense_16/Tensordot/Const_2:output:06sequential_5/dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_5/dense_16/Tensordot/concat_1è
sequential_5/dense_16/TensordotReshape0sequential_5/dense_16/Tensordot/MatMul:product:01sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2!
sequential_5/dense_16/TensordotÎ
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOpß
sequential_5/dense_16/BiasAddBiasAdd(sequential_5/dense_16/Tensordot:output:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_5/dense_16/BiasAdd
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_5/dense_16/ReluØ
.sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_5/dense_17/Tensordot/ReadVariableOp
$sequential_5/dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_5/dense_17/Tensordot/axes
$sequential_5/dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_5/dense_17/Tensordot/free¦
%sequential_5/dense_17/Tensordot/ShapeShape(sequential_5/dense_16/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_5/dense_17/Tensordot/Shape 
-sequential_5/dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_17/Tensordot/GatherV2/axis¿
(sequential_5/dense_17/Tensordot/GatherV2GatherV2.sequential_5/dense_17/Tensordot/Shape:output:0-sequential_5/dense_17/Tensordot/free:output:06sequential_5/dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_5/dense_17/Tensordot/GatherV2¤
/sequential_5/dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_5/dense_17/Tensordot/GatherV2_1/axisÅ
*sequential_5/dense_17/Tensordot/GatherV2_1GatherV2.sequential_5/dense_17/Tensordot/Shape:output:0-sequential_5/dense_17/Tensordot/axes:output:08sequential_5/dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_5/dense_17/Tensordot/GatherV2_1
%sequential_5/dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_5/dense_17/Tensordot/ConstØ
$sequential_5/dense_17/Tensordot/ProdProd1sequential_5/dense_17/Tensordot/GatherV2:output:0.sequential_5/dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_5/dense_17/Tensordot/Prod
'sequential_5/dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_17/Tensordot/Const_1à
&sequential_5/dense_17/Tensordot/Prod_1Prod3sequential_5/dense_17/Tensordot/GatherV2_1:output:00sequential_5/dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_5/dense_17/Tensordot/Prod_1
+sequential_5/dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_5/dense_17/Tensordot/concat/axis
&sequential_5/dense_17/Tensordot/concatConcatV2-sequential_5/dense_17/Tensordot/free:output:0-sequential_5/dense_17/Tensordot/axes:output:04sequential_5/dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/dense_17/Tensordot/concatä
%sequential_5/dense_17/Tensordot/stackPack-sequential_5/dense_17/Tensordot/Prod:output:0/sequential_5/dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/dense_17/Tensordot/stackô
)sequential_5/dense_17/Tensordot/transpose	Transpose(sequential_5/dense_16/Relu:activations:0/sequential_5/dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2+
)sequential_5/dense_17/Tensordot/transpose÷
'sequential_5/dense_17/Tensordot/ReshapeReshape-sequential_5/dense_17/Tensordot/transpose:y:0.sequential_5/dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_5/dense_17/Tensordot/Reshapeö
&sequential_5/dense_17/Tensordot/MatMulMatMul0sequential_5/dense_17/Tensordot/Reshape:output:06sequential_5/dense_17/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_5/dense_17/Tensordot/MatMul
'sequential_5/dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_17/Tensordot/Const_2 
-sequential_5/dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_17/Tensordot/concat_1/axis«
(sequential_5/dense_17/Tensordot/concat_1ConcatV21sequential_5/dense_17/Tensordot/GatherV2:output:00sequential_5/dense_17/Tensordot/Const_2:output:06sequential_5/dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_5/dense_17/Tensordot/concat_1è
sequential_5/dense_17/TensordotReshape0sequential_5/dense_17/Tensordot/MatMul:product:01sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
sequential_5/dense_17/TensordotÎ
,sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_17/BiasAdd/ReadVariableOpß
sequential_5/dense_17/BiasAddBiasAdd(sequential_5/dense_17/Tensordot:output:04sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
sequential_5/dense_17/BiasAdd
dropout_15/IdentityIdentity&sequential_5/dense_17/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_15/Identity
add_1AddV2*layer_normalization_10/batchnorm/add_1:z:0dropout_15/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¸
5layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_11/moments/mean/reduction_indicesä
#layer_normalization_11/moments/meanMean	add_1:z:0>layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_11/moments/meanÎ
+layer_normalization_11/moments/StopGradientStopGradient,layer_normalization_11/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_11/moments/StopGradientð
0layer_normalization_11/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_11/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_11/moments/SquaredDifferenceÀ
9layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_11/moments/variance/reduction_indices
'layer_normalization_11/moments/varianceMean4layer_normalization_11/moments/SquaredDifference:z:0Blayer_normalization_11/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_11/moments/variance
&layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_11/batchnorm/add/yî
$layer_normalization_11/batchnorm/addAddV20layer_normalization_11/moments/variance:output:0/layer_normalization_11/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_11/batchnorm/add¹
&layer_normalization_11/batchnorm/RsqrtRsqrt(layer_normalization_11/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_11/batchnorm/Rsqrtã
3layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_11/batchnorm/mul/ReadVariableOpò
$layer_normalization_11/batchnorm/mulMul*layer_normalization_11/batchnorm/Rsqrt:y:0;layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_11/batchnorm/mulÂ
&layer_normalization_11/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_11/batchnorm/mul_1å
&layer_normalization_11/batchnorm/mul_2Mul,layer_normalization_11/moments/mean:output:0(layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_11/batchnorm/mul_2×
/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_11/batchnorm/ReadVariableOpî
$layer_normalization_11/batchnorm/subSub7layer_normalization_11/batchnorm/ReadVariableOp:value:0*layer_normalization_11/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_11/batchnorm/subå
&layer_normalization_11/batchnorm/add_1AddV2*layer_normalization_11/batchnorm/mul_1:z:0(layer_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_11/batchnorm/add_1Ü
IdentityIdentity*layer_normalization_11/batchnorm/add_1:z:00^layer_normalization_10/batchnorm/ReadVariableOp4^layer_normalization_10/batchnorm/mul/ReadVariableOp0^layer_normalization_11/batchnorm/ReadVariableOp4^layer_normalization_11/batchnorm/mul/ReadVariableOp;^multi_head_attention_5/attention_output/add/ReadVariableOpE^multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_5/key/add/ReadVariableOp8^multi_head_attention_5/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/query/add/ReadVariableOp:^multi_head_attention_5/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/value/add/ReadVariableOp:^multi_head_attention_5/value/einsum/Einsum/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp/^sequential_5/dense_16/Tensordot/ReadVariableOp-^sequential_5/dense_17/BiasAdd/ReadVariableOp/^sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2b
/layer_normalization_10/batchnorm/ReadVariableOp/layer_normalization_10/batchnorm/ReadVariableOp2j
3layer_normalization_10/batchnorm/mul/ReadVariableOp3layer_normalization_10/batchnorm/mul/ReadVariableOp2b
/layer_normalization_11/batchnorm/ReadVariableOp/layer_normalization_11/batchnorm/ReadVariableOp2j
3layer_normalization_11/batchnorm/mul/ReadVariableOp3layer_normalization_11/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_5/attention_output/add/ReadVariableOp:multi_head_attention_5/attention_output/add/ReadVariableOp2
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_5/key/add/ReadVariableOp-multi_head_attention_5/key/add/ReadVariableOp2r
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/query/add/ReadVariableOp/multi_head_attention_5/query/add/ReadVariableOp2v
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/value/add/ReadVariableOp/multi_head_attention_5/value/add/ReadVariableOp2v
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2`
.sequential_5/dense_16/Tensordot/ReadVariableOp.sequential_5/dense_16/Tensordot/ReadVariableOp2\
,sequential_5/dense_17/BiasAdd/ReadVariableOp,sequential_5/dense_17/BiasAdd/ReadVariableOp2`
.sequential_5/dense_17/Tensordot/ReadVariableOp.sequential_5/dense_17/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
ß
~
)__inference_dense_19_layer_call_fn_424801

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
D__inference_dense_19_layer_call_and_return_conditional_losses_4224342
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
ó
~
)__inference_conv1d_4_layer_call_fn_423914

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
:ÿÿÿÿÿÿÿÿÿR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_4216692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿR ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 
 
_user_specified_nameinputs
è

Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_421775

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

÷
D__inference_conv1d_4_layer_call_and_return_conditional_losses_421669

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
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
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
:ÿÿÿÿÿÿÿÿÿR 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2

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
õ
k
O__inference_average_pooling1d_7_layer_call_and_return_conditional_losses_421000

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
0
È
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_424675

inputs
assignmovingavg_424650
assignmovingavg_1_424656)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/424650*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_424650*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/424650*
_output_shapes
:2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/424650*
_output_shapes
:2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_424650AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/424650*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/424656*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_424656*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/424656*
_output_shapes
:2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/424656*
_output_shapes
:2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_424656AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/424656*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë«
,
__inference__traced_save_425330
file_prefix.
*savev2_conv1d_4_kernel_read_readvariableop,
(savev2_conv1d_4_bias_read_readvariableop.
*savev2_conv1d_5_kernel_read_readvariableop,
(savev2_conv1d_5_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop'
#savev2_momentum_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	T
Psavev2_token_and_position_embedding_2_embedding_4_embeddings_read_readvariableopT
Psavev2_token_and_position_embedding_2_embedding_5_embeddings_read_readvariableopV
Rsavev2_transformer_block_5_multi_head_attention_5_query_kernel_read_readvariableopT
Psavev2_transformer_block_5_multi_head_attention_5_query_bias_read_readvariableopT
Psavev2_transformer_block_5_multi_head_attention_5_key_kernel_read_readvariableopR
Nsavev2_transformer_block_5_multi_head_attention_5_key_bias_read_readvariableopV
Rsavev2_transformer_block_5_multi_head_attention_5_value_kernel_read_readvariableopT
Psavev2_transformer_block_5_multi_head_attention_5_value_bias_read_readvariableopa
]savev2_transformer_block_5_multi_head_attention_5_attention_output_kernel_read_readvariableop_
[savev2_transformer_block_5_multi_head_attention_5_attention_output_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableopO
Ksavev2_transformer_block_5_layer_normalization_10_gamma_read_readvariableopN
Jsavev2_transformer_block_5_layer_normalization_10_beta_read_readvariableopO
Ksavev2_transformer_block_5_layer_normalization_11_gamma_read_readvariableopN
Jsavev2_transformer_block_5_layer_normalization_11_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop;
7savev2_sgd_conv1d_4_kernel_momentum_read_readvariableop9
5savev2_sgd_conv1d_4_bias_momentum_read_readvariableop;
7savev2_sgd_conv1d_5_kernel_momentum_read_readvariableop9
5savev2_sgd_conv1d_5_bias_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_6_gamma_momentum_read_readvariableopF
Bsavev2_sgd_batch_normalization_6_beta_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_7_gamma_momentum_read_readvariableopF
Bsavev2_sgd_batch_normalization_7_beta_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_8_gamma_momentum_read_readvariableopF
Bsavev2_sgd_batch_normalization_8_beta_momentum_read_readvariableop;
7savev2_sgd_dense_18_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_18_bias_momentum_read_readvariableop;
7savev2_sgd_dense_19_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_19_bias_momentum_read_readvariableop;
7savev2_sgd_dense_20_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_20_bias_momentum_read_readvariableopa
]savev2_sgd_token_and_position_embedding_2_embedding_4_embeddings_momentum_read_readvariableopa
]savev2_sgd_token_and_position_embedding_2_embedding_5_embeddings_momentum_read_readvariableopc
_savev2_sgd_transformer_block_5_multi_head_attention_5_query_kernel_momentum_read_readvariableopa
]savev2_sgd_transformer_block_5_multi_head_attention_5_query_bias_momentum_read_readvariableopa
]savev2_sgd_transformer_block_5_multi_head_attention_5_key_kernel_momentum_read_readvariableop_
[savev2_sgd_transformer_block_5_multi_head_attention_5_key_bias_momentum_read_readvariableopc
_savev2_sgd_transformer_block_5_multi_head_attention_5_value_kernel_momentum_read_readvariableopa
]savev2_sgd_transformer_block_5_multi_head_attention_5_value_bias_momentum_read_readvariableopn
jsavev2_sgd_transformer_block_5_multi_head_attention_5_attention_output_kernel_momentum_read_readvariableopl
hsavev2_sgd_transformer_block_5_multi_head_attention_5_attention_output_bias_momentum_read_readvariableop;
7savev2_sgd_dense_16_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_16_bias_momentum_read_readvariableop;
7savev2_sgd_dense_17_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_17_bias_momentum_read_readvariableop\
Xsavev2_sgd_transformer_block_5_layer_normalization_10_gamma_momentum_read_readvariableop[
Wsavev2_sgd_transformer_block_5_layer_normalization_10_beta_momentum_read_readvariableop\
Xsavev2_sgd_transformer_block_5_layer_normalization_11_gamma_momentum_read_readvariableop[
Wsavev2_sgd_transformer_block_5_layer_normalization_11_beta_momentum_read_readvariableop
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
ShardedFilename)
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*(
value(B(QB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names­
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*·
value­BªQB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesì*
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_4_kernel_read_readvariableop(savev2_conv1d_4_bias_read_readvariableop*savev2_conv1d_5_kernel_read_readvariableop(savev2_conv1d_5_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableopPsavev2_token_and_position_embedding_2_embedding_4_embeddings_read_readvariableopPsavev2_token_and_position_embedding_2_embedding_5_embeddings_read_readvariableopRsavev2_transformer_block_5_multi_head_attention_5_query_kernel_read_readvariableopPsavev2_transformer_block_5_multi_head_attention_5_query_bias_read_readvariableopPsavev2_transformer_block_5_multi_head_attention_5_key_kernel_read_readvariableopNsavev2_transformer_block_5_multi_head_attention_5_key_bias_read_readvariableopRsavev2_transformer_block_5_multi_head_attention_5_value_kernel_read_readvariableopPsavev2_transformer_block_5_multi_head_attention_5_value_bias_read_readvariableop]savev2_transformer_block_5_multi_head_attention_5_attention_output_kernel_read_readvariableop[savev2_transformer_block_5_multi_head_attention_5_attention_output_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableopKsavev2_transformer_block_5_layer_normalization_10_gamma_read_readvariableopJsavev2_transformer_block_5_layer_normalization_10_beta_read_readvariableopKsavev2_transformer_block_5_layer_normalization_11_gamma_read_readvariableopJsavev2_transformer_block_5_layer_normalization_11_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop7savev2_sgd_conv1d_4_kernel_momentum_read_readvariableop5savev2_sgd_conv1d_4_bias_momentum_read_readvariableop7savev2_sgd_conv1d_5_kernel_momentum_read_readvariableop5savev2_sgd_conv1d_5_bias_momentum_read_readvariableopCsavev2_sgd_batch_normalization_6_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_6_beta_momentum_read_readvariableopCsavev2_sgd_batch_normalization_7_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_7_beta_momentum_read_readvariableopCsavev2_sgd_batch_normalization_8_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_8_beta_momentum_read_readvariableop7savev2_sgd_dense_18_kernel_momentum_read_readvariableop5savev2_sgd_dense_18_bias_momentum_read_readvariableop7savev2_sgd_dense_19_kernel_momentum_read_readvariableop5savev2_sgd_dense_19_bias_momentum_read_readvariableop7savev2_sgd_dense_20_kernel_momentum_read_readvariableop5savev2_sgd_dense_20_bias_momentum_read_readvariableop]savev2_sgd_token_and_position_embedding_2_embedding_4_embeddings_momentum_read_readvariableop]savev2_sgd_token_and_position_embedding_2_embedding_5_embeddings_momentum_read_readvariableop_savev2_sgd_transformer_block_5_multi_head_attention_5_query_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_5_multi_head_attention_5_query_bias_momentum_read_readvariableop]savev2_sgd_transformer_block_5_multi_head_attention_5_key_kernel_momentum_read_readvariableop[savev2_sgd_transformer_block_5_multi_head_attention_5_key_bias_momentum_read_readvariableop_savev2_sgd_transformer_block_5_multi_head_attention_5_value_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_5_multi_head_attention_5_value_bias_momentum_read_readvariableopjsavev2_sgd_transformer_block_5_multi_head_attention_5_attention_output_kernel_momentum_read_readvariableophsavev2_sgd_transformer_block_5_multi_head_attention_5_attention_output_bias_momentum_read_readvariableop7savev2_sgd_dense_16_kernel_momentum_read_readvariableop5savev2_sgd_dense_16_bias_momentum_read_readvariableop7savev2_sgd_dense_17_kernel_momentum_read_readvariableop5savev2_sgd_dense_17_bias_momentum_read_readvariableopXsavev2_sgd_transformer_block_5_layer_normalization_10_gamma_momentum_read_readvariableopWsavev2_sgd_transformer_block_5_layer_normalization_10_beta_momentum_read_readvariableopXsavev2_sgd_transformer_block_5_layer_normalization_11_gamma_momentum_read_readvariableopWsavev2_sgd_transformer_block_5_layer_normalization_11_beta_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *_
dtypesU
S2Q	2
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

identity_1Identity_1:output:0*
_input_shapes
: :  : :	  : : : : : : : : : :::::	è@:@:@@:@:@:: : : : : :	R :  : :  : :  : :  : : @:@:@ : : : : : : : :  : :	  : : : : : :::	è@:@:@@:@:@:: :	R :  : :  : :  : :  : : @:@:@ : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:  : 

_output_shapes
: :($
"
_output_shapes
:	  : 
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
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	è@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: :%!

_output_shapes
:	R :($
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
:  :$" 

_output_shapes

: :(#$
"
_output_shapes
:  : $

_output_shapes
: :$% 

_output_shapes

: @: &

_output_shapes
:@:$' 

_output_shapes

:@ : (

_output_shapes
: : )

_output_shapes
: : *

_output_shapes
: : +

_output_shapes
: : ,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :(/$
"
_output_shapes
:  : 0

_output_shapes
: :(1$
"
_output_shapes
:	  : 2
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
: : 7

_output_shapes
:: 8

_output_shapes
::%9!

_output_shapes
:	è@: :

_output_shapes
:@:$; 

_output_shapes

:@@: <

_output_shapes
:@:$= 

_output_shapes

:@: >

_output_shapes
::$? 

_output_shapes

: :%@!

_output_shapes
:	R :(A$
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
:  :$F 

_output_shapes

: :(G$
"
_output_shapes
:  : H

_output_shapes
: :$I 

_output_shapes

: @: J

_output_shapes
:@:$K 

_output_shapes

:@ : L

_output_shapes
: : M

_output_shapes
: : N

_output_shapes
: : O

_output_shapes
: : P

_output_shapes
: :Q

_output_shapes
: 
¶
Þ
$__inference_signature_wrapper_423083
input_5
input_6
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

unknown_36

unknown_37

unknown_38
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_36
unknown_37
unknown_38*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_4209762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_5:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6
`
Ö
C__inference_model_2_layer_call_and_return_conditional_losses_422906

inputs
inputs_1)
%token_and_position_embedding_2_422807)
%token_and_position_embedding_2_422809
conv1d_4_422812
conv1d_4_422814
conv1d_5_422818
conv1d_5_422820 
batch_normalization_6_422825 
batch_normalization_6_422827 
batch_normalization_6_422829 
batch_normalization_6_422831 
batch_normalization_7_422834 
batch_normalization_7_422836 
batch_normalization_7_422838 
batch_normalization_7_422840
transformer_block_5_422844
transformer_block_5_422846
transformer_block_5_422848
transformer_block_5_422850
transformer_block_5_422852
transformer_block_5_422854
transformer_block_5_422856
transformer_block_5_422858
transformer_block_5_422860
transformer_block_5_422862
transformer_block_5_422864
transformer_block_5_422866
transformer_block_5_422868
transformer_block_5_422870
transformer_block_5_422872
transformer_block_5_422874 
batch_normalization_8_422878 
batch_normalization_8_422880 
batch_normalization_8_422882 
batch_normalization_8_422884
dense_18_422888
dense_18_422890
dense_19_422894
dense_19_422896
dense_20_422900
dense_20_422902
identity¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢ conv1d_4/StatefulPartitionedCall¢ conv1d_5/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢ dense_20/StatefulPartitionedCall¢6token_and_position_embedding_2/StatefulPartitionedCall¢+transformer_block_5/StatefulPartitionedCall
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_2_422807%token_and_position_embedding_2_422809*
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
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_42163728
6token_and_position_embedding_2/StatefulPartitionedCallÕ
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0conv1d_4_422812conv1d_4_422814*
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
GPU2*0J 8 *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_4216692"
 conv1d_4/StatefulPartitionedCall 
#average_pooling1d_6/PartitionedCallPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_6_layer_call_and_return_conditional_losses_4209852%
#average_pooling1d_6/PartitionedCallÂ
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_6/PartitionedCall:output:0conv1d_5_422818conv1d_5_422820*
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
D__inference_conv1d_5_layer_call_and_return_conditional_losses_4217022"
 conv1d_5/StatefulPartitionedCallµ
#average_pooling1d_8/PartitionedCallPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0*
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
O__inference_average_pooling1d_8_layer_call_and_return_conditional_losses_4210152%
#average_pooling1d_8/PartitionedCall
#average_pooling1d_7/PartitionedCallPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0*
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
O__inference_average_pooling1d_7_layer_call_and_return_conditional_losses_4210002%
#average_pooling1d_7/PartitionedCallÂ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_7/PartitionedCall:output:0batch_normalization_6_422825batch_normalization_6_422827batch_normalization_6_422829batch_normalization_6_422831*
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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4217752/
-batch_normalization_6/StatefulPartitionedCallÂ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_8/PartitionedCall:output:0batch_normalization_7_422834batch_normalization_7_422836batch_normalization_7_422838batch_normalization_7_422840*
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4218662/
-batch_normalization_7/StatefulPartitionedCall»
add_2/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
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
A__inference_add_2_layer_call_and_return_conditional_losses_4219082
add_2/PartitionedCall
+transformer_block_5/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0transformer_block_5_422844transformer_block_5_422846transformer_block_5_422848transformer_block_5_422850transformer_block_5_422852transformer_block_5_422854transformer_block_5_422856transformer_block_5_422858transformer_block_5_422860transformer_block_5_422862transformer_block_5_422864transformer_block_5_422866transformer_block_5_422868transformer_block_5_422870transformer_block_5_422872transformer_block_5_422874*
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
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_4221922-
+transformer_block_5/StatefulPartitionedCall
flatten_2/PartitionedCallPartitionedCall4transformer_block_5/StatefulPartitionedCall:output:0*
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
E__inference_flatten_2_layer_call_and_return_conditional_losses_4223072
flatten_2/PartitionedCall
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCallinputs_1batch_normalization_8_422878batch_normalization_8_422880batch_normalization_8_422882batch_normalization_8_422884*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4215972/
-batch_normalization_8/StatefulPartitionedCall¼
concatenate_2/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:06batch_normalization_8/StatefulPartitionedCall:output:0*
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
I__inference_concatenate_2_layer_call_and_return_conditional_losses_4223572
concatenate_2/PartitionedCall·
 dense_18/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_18_422888dense_18_422890*
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
D__inference_dense_18_layer_call_and_return_conditional_losses_4223772"
 dense_18/StatefulPartitionedCall
dropout_16/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
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
F__inference_dropout_16_layer_call_and_return_conditional_losses_4224102
dropout_16/PartitionedCall´
 dense_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0dense_19_422894dense_19_422896*
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
D__inference_dense_19_layer_call_and_return_conditional_losses_4224342"
 dense_19/StatefulPartitionedCall
dropout_17/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
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
F__inference_dropout_17_layer_call_and_return_conditional_losses_4224672
dropout_17/PartitionedCall´
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0dense_20_422900dense_20_422902*
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
D__inference_dense_20_layer_call_and_return_conditional_losses_4224902"
 dense_20/StatefulPartitionedCall£
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_5/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_5/StatefulPartitionedCall+transformer_block_5/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

P
4__inference_average_pooling1d_6_layer_call_fn_420991

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
O__inference_average_pooling1d_6_layer_call_and_return_conditional_losses_4209852
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
¸
 
-__inference_sequential_5_layer_call_fn_424987

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
H__inference_sequential_5_layer_call_and_return_conditional_losses_4214572
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
J
¯
H__inference_sequential_5_layer_call_and_return_conditional_losses_424961

inputs.
*dense_16_tensordot_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource.
*dense_17_tensordot_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
identity¢dense_16/BiasAdd/ReadVariableOp¢!dense_16/Tensordot/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢!dense_17/Tensordot/ReadVariableOp±
!dense_16/Tensordot/ReadVariableOpReadVariableOp*dense_16_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02#
!dense_16/Tensordot/ReadVariableOp|
dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_16/Tensordot/axes
dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_16/Tensordot/freej
dense_16/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_16/Tensordot/Shape
 dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_16/Tensordot/GatherV2/axisþ
dense_16/Tensordot/GatherV2GatherV2!dense_16/Tensordot/Shape:output:0 dense_16/Tensordot/free:output:0)dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_16/Tensordot/GatherV2
"dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_16/Tensordot/GatherV2_1/axis
dense_16/Tensordot/GatherV2_1GatherV2!dense_16/Tensordot/Shape:output:0 dense_16/Tensordot/axes:output:0+dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_16/Tensordot/GatherV2_1~
dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_16/Tensordot/Const¤
dense_16/Tensordot/ProdProd$dense_16/Tensordot/GatherV2:output:0!dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_16/Tensordot/Prod
dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_16/Tensordot/Const_1¬
dense_16/Tensordot/Prod_1Prod&dense_16/Tensordot/GatherV2_1:output:0#dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_16/Tensordot/Prod_1
dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_16/Tensordot/concat/axisÝ
dense_16/Tensordot/concatConcatV2 dense_16/Tensordot/free:output:0 dense_16/Tensordot/axes:output:0'dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_16/Tensordot/concat°
dense_16/Tensordot/stackPack dense_16/Tensordot/Prod:output:0"dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_16/Tensordot/stack«
dense_16/Tensordot/transpose	Transposeinputs"dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_16/Tensordot/transposeÃ
dense_16/Tensordot/ReshapeReshape dense_16/Tensordot/transpose:y:0!dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_16/Tensordot/ReshapeÂ
dense_16/Tensordot/MatMulMatMul#dense_16/Tensordot/Reshape:output:0)dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_16/Tensordot/MatMul
dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_16/Tensordot/Const_2
 dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_16/Tensordot/concat_1/axisê
dense_16/Tensordot/concat_1ConcatV2$dense_16/Tensordot/GatherV2:output:0#dense_16/Tensordot/Const_2:output:0)dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_16/Tensordot/concat_1´
dense_16/TensordotReshape#dense_16/Tensordot/MatMul:product:0$dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_16/Tensordot§
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_16/BiasAdd/ReadVariableOp«
dense_16/BiasAddBiasAdddense_16/Tensordot:output:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_16/BiasAddw
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_16/Relu±
!dense_17/Tensordot/ReadVariableOpReadVariableOp*dense_17_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02#
!dense_17/Tensordot/ReadVariableOp|
dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_17/Tensordot/axes
dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_17/Tensordot/free
dense_17/Tensordot/ShapeShapedense_16/Relu:activations:0*
T0*
_output_shapes
:2
dense_17/Tensordot/Shape
 dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_17/Tensordot/GatherV2/axisþ
dense_17/Tensordot/GatherV2GatherV2!dense_17/Tensordot/Shape:output:0 dense_17/Tensordot/free:output:0)dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_17/Tensordot/GatherV2
"dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_17/Tensordot/GatherV2_1/axis
dense_17/Tensordot/GatherV2_1GatherV2!dense_17/Tensordot/Shape:output:0 dense_17/Tensordot/axes:output:0+dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_17/Tensordot/GatherV2_1~
dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_17/Tensordot/Const¤
dense_17/Tensordot/ProdProd$dense_17/Tensordot/GatherV2:output:0!dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_17/Tensordot/Prod
dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_17/Tensordot/Const_1¬
dense_17/Tensordot/Prod_1Prod&dense_17/Tensordot/GatherV2_1:output:0#dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_17/Tensordot/Prod_1
dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_17/Tensordot/concat/axisÝ
dense_17/Tensordot/concatConcatV2 dense_17/Tensordot/free:output:0 dense_17/Tensordot/axes:output:0'dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_17/Tensordot/concat°
dense_17/Tensordot/stackPack dense_17/Tensordot/Prod:output:0"dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_17/Tensordot/stackÀ
dense_17/Tensordot/transpose	Transposedense_16/Relu:activations:0"dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_17/Tensordot/transposeÃ
dense_17/Tensordot/ReshapeReshape dense_17/Tensordot/transpose:y:0!dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_17/Tensordot/ReshapeÂ
dense_17/Tensordot/MatMulMatMul#dense_17/Tensordot/Reshape:output:0)dense_17/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_17/Tensordot/MatMul
dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_17/Tensordot/Const_2
 dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_17/Tensordot/concat_1/axisê
dense_17/Tensordot/concat_1ConcatV2$dense_17/Tensordot/GatherV2:output:0#dense_17/Tensordot/Const_2:output:0)dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_17/Tensordot/concat_1´
dense_17/TensordotReshape#dense_17/Tensordot/MatMul:product:0$dense_17/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_17/Tensordot§
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_17/BiasAdd/ReadVariableOp«
dense_17/BiasAddBiasAdddense_17/Tensordot:output:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_17/BiasAddý
IdentityIdentitydense_17/BiasAdd:output:0 ^dense_16/BiasAdd/ReadVariableOp"^dense_16/Tensordot/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp"^dense_17/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2F
!dense_16/Tensordot/ReadVariableOp!dense_16/Tensordot/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2F
!dense_17/Tensordot/ReadVariableOp!dense_17/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
Ûø
(
C__inference_model_2_layer_call_and_return_conditional_losses_423684
inputs_0
inputs_1F
Btoken_and_position_embedding_2_embedding_5_embedding_lookup_423437F
Btoken_and_position_embedding_2_embedding_4_embedding_lookup_4234438
4conv1d_4_conv1d_expanddims_1_readvariableop_resource,
(conv1d_4_biasadd_readvariableop_resource8
4conv1d_5_conv1d_expanddims_1_readvariableop_resource,
(conv1d_5_biasadd_readvariableop_resource;
7batch_normalization_6_batchnorm_readvariableop_resource?
;batch_normalization_6_batchnorm_mul_readvariableop_resource=
9batch_normalization_6_batchnorm_readvariableop_1_resource=
9batch_normalization_6_batchnorm_readvariableop_2_resource;
7batch_normalization_7_batchnorm_readvariableop_resource?
;batch_normalization_7_batchnorm_mul_readvariableop_resource=
9batch_normalization_7_batchnorm_readvariableop_1_resource=
9batch_normalization_7_batchnorm_readvariableop_2_resourceZ
Vtransformer_block_5_multi_head_attention_5_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_5_multi_head_attention_5_query_add_readvariableop_resourceX
Ttransformer_block_5_multi_head_attention_5_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_5_multi_head_attention_5_key_add_readvariableop_resourceZ
Vtransformer_block_5_multi_head_attention_5_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_5_multi_head_attention_5_value_add_readvariableop_resourcee
atransformer_block_5_multi_head_attention_5_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_5_multi_head_attention_5_attention_output_add_readvariableop_resourceT
Ptransformer_block_5_layer_normalization_10_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_5_layer_normalization_10_batchnorm_readvariableop_resourceO
Ktransformer_block_5_sequential_5_dense_16_tensordot_readvariableop_resourceM
Itransformer_block_5_sequential_5_dense_16_biasadd_readvariableop_resourceO
Ktransformer_block_5_sequential_5_dense_17_tensordot_readvariableop_resourceM
Itransformer_block_5_sequential_5_dense_17_biasadd_readvariableop_resourceT
Ptransformer_block_5_layer_normalization_11_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_5_layer_normalization_11_batchnorm_readvariableop_resource;
7batch_normalization_8_batchnorm_readvariableop_resource?
;batch_normalization_8_batchnorm_mul_readvariableop_resource=
9batch_normalization_8_batchnorm_readvariableop_1_resource=
9batch_normalization_8_batchnorm_readvariableop_2_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource
identity¢.batch_normalization_6/batchnorm/ReadVariableOp¢0batch_normalization_6/batchnorm/ReadVariableOp_1¢0batch_normalization_6/batchnorm/ReadVariableOp_2¢2batch_normalization_6/batchnorm/mul/ReadVariableOp¢.batch_normalization_7/batchnorm/ReadVariableOp¢0batch_normalization_7/batchnorm/ReadVariableOp_1¢0batch_normalization_7/batchnorm/ReadVariableOp_2¢2batch_normalization_7/batchnorm/mul/ReadVariableOp¢.batch_normalization_8/batchnorm/ReadVariableOp¢0batch_normalization_8/batchnorm/ReadVariableOp_1¢0batch_normalization_8/batchnorm/ReadVariableOp_2¢2batch_normalization_8/batchnorm/mul/ReadVariableOp¢conv1d_4/BiasAdd/ReadVariableOp¢+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp¢conv1d_5/BiasAdd/ReadVariableOp¢+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp¢dense_18/BiasAdd/ReadVariableOp¢dense_18/MatMul/ReadVariableOp¢dense_19/BiasAdd/ReadVariableOp¢dense_19/MatMul/ReadVariableOp¢dense_20/BiasAdd/ReadVariableOp¢dense_20/MatMul/ReadVariableOp¢;token_and_position_embedding_2/embedding_4/embedding_lookup¢;token_and_position_embedding_2/embedding_5/embedding_lookup¢Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp¢Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp¢Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp¢Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp¢Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp¢Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp¢Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp¢Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp¢Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOp¢Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp¢Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOp¢Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp¢@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp¢Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp¢@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp¢Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp
$token_and_position_embedding_2/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_2/Shape»
2token_and_position_embedding_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ24
2token_and_position_embedding_2/strided_slice/stack¶
4token_and_position_embedding_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_2/strided_slice/stack_1¶
4token_and_position_embedding_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_2/strided_slice/stack_2
,token_and_position_embedding_2/strided_sliceStridedSlice-token_and_position_embedding_2/Shape:output:0;token_and_position_embedding_2/strided_slice/stack:output:0=token_and_position_embedding_2/strided_slice/stack_1:output:0=token_and_position_embedding_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_2/strided_slice
*token_and_position_embedding_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_2/range/start
*token_and_position_embedding_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_2/range/delta
$token_and_position_embedding_2/rangeRange3token_and_position_embedding_2/range/start:output:05token_and_position_embedding_2/strided_slice:output:03token_and_position_embedding_2/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$token_and_position_embedding_2/rangeÊ
;token_and_position_embedding_2/embedding_5/embedding_lookupResourceGatherBtoken_and_position_embedding_2_embedding_5_embedding_lookup_423437-token_and_position_embedding_2/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/423437*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02=
;token_and_position_embedding_2/embedding_5/embedding_lookup
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/423437*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2F
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2H
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1¶
/token_and_position_embedding_2/embedding_4/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR21
/token_and_position_embedding_2/embedding_4/CastÕ
;token_and_position_embedding_2/embedding_4/embedding_lookupResourceGatherBtoken_and_position_embedding_2_embedding_4_embedding_lookup_4234433token_and_position_embedding_2/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/423443*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02=
;token_and_position_embedding_2/embedding_4/embedding_lookup
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/423443*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2F
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity¢
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2H
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1ª
"token_and_position_embedding_2/addAddV2Otoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2$
"token_and_position_embedding_2/add
conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_4/conv1d/ExpandDims/dimÒ
conv1d_4/conv1d/ExpandDims
ExpandDims&token_and_position_embedding_2/add:z:0'conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_4/conv1d/ExpandDimsÓ
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02-
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_4/conv1d/ExpandDims_1/dimÛ
conv1d_4/conv1d/ExpandDims_1
ExpandDims3conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_4/conv1d/ExpandDims_1Û
conv1d_4/conv1dConv2D#conv1d_4/conv1d/ExpandDims:output:0%conv1d_4/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
paddingSAME*
strides
2
conv1d_4/conv1d®
conv1d_4/conv1d/SqueezeSqueezeconv1d_4/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_4/conv1d/Squeeze§
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_4/BiasAdd/ReadVariableOp±
conv1d_4/BiasAddBiasAdd conv1d_4/conv1d/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_4/BiasAddx
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_4/Relu
"average_pooling1d_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_6/ExpandDims/dimÓ
average_pooling1d_6/ExpandDims
ExpandDimsconv1d_4/Relu:activations:0+average_pooling1d_6/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2 
average_pooling1d_6/ExpandDimså
average_pooling1d_6/AvgPoolAvgPool'average_pooling1d_6/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_6/AvgPool¹
average_pooling1d_6/SqueezeSqueeze$average_pooling1d_6/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims
2
average_pooling1d_6/Squeeze
conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_5/conv1d/ExpandDims/dimÐ
conv1d_5/conv1d/ExpandDims
ExpandDims$average_pooling1d_6/Squeeze:output:0'conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_5/conv1d/ExpandDimsÓ
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02-
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_5/conv1d/ExpandDims_1/dimÛ
conv1d_5/conv1d/ExpandDims_1
ExpandDims3conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d_5/conv1d/ExpandDims_1Û
conv1d_5/conv1dConv2D#conv1d_5/conv1d/ExpandDims:output:0%conv1d_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
paddingSAME*
strides
2
conv1d_5/conv1d®
conv1d_5/conv1d/SqueezeSqueezeconv1d_5/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_5/conv1d/Squeeze§
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_5/BiasAdd/ReadVariableOp±
conv1d_5/BiasAddBiasAdd conv1d_5/conv1d/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_5/BiasAddx
conv1d_5/ReluReluconv1d_5/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_5/Relu
"average_pooling1d_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_8/ExpandDims/dimÞ
average_pooling1d_8/ExpandDims
ExpandDims&token_and_position_embedding_2/add:z:0+average_pooling1d_8/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2 
average_pooling1d_8/ExpandDimsæ
average_pooling1d_8/AvgPoolAvgPool'average_pooling1d_8/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize	
¬*
paddingVALID*
strides	
¬2
average_pooling1d_8/AvgPool¸
average_pooling1d_8/SqueezeSqueeze$average_pooling1d_8/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_8/Squeeze
"average_pooling1d_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_7/ExpandDims/dimÓ
average_pooling1d_7/ExpandDims
ExpandDimsconv1d_5/Relu:activations:0+average_pooling1d_7/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2 
average_pooling1d_7/ExpandDimsä
average_pooling1d_7/AvgPoolAvgPool'average_pooling1d_7/ExpandDims:output:0*
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
average_pooling1d_7/AvgPool¸
average_pooling1d_7/SqueezeSqueeze$average_pooling1d_7/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_7/SqueezeÔ
.batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_6/batchnorm/ReadVariableOp
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_6/batchnorm/add/yà
#batch_normalization_6/batchnorm/addAddV26batch_normalization_6/batchnorm/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_6/batchnorm/add¥
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_6/batchnorm/Rsqrtà
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_6/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_6/batchnorm/mulÚ
%batch_normalization_6/batchnorm/mul_1Mul$average_pooling1d_7/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_6/batchnorm/mul_1Ú
0batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_6/batchnorm/ReadVariableOp_1Ý
%batch_normalization_6/batchnorm/mul_2Mul8batch_normalization_6/batchnorm/ReadVariableOp_1:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_6/batchnorm/mul_2Ú
0batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_6/batchnorm/ReadVariableOp_2Û
#batch_normalization_6/batchnorm/subSub8batch_normalization_6/batchnorm/ReadVariableOp_2:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_6/batchnorm/subá
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_6/batchnorm/add_1Ô
.batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_7/batchnorm/ReadVariableOp
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_7/batchnorm/add/yà
#batch_normalization_7/batchnorm/addAddV26batch_normalization_7/batchnorm/ReadVariableOp:value:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_7/batchnorm/add¥
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_7/batchnorm/Rsqrtà
2batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_7/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:0:batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_7/batchnorm/mulÚ
%batch_normalization_7/batchnorm/mul_1Mul$average_pooling1d_8/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_7/batchnorm/mul_1Ú
0batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_7/batchnorm/ReadVariableOp_1Ý
%batch_normalization_7/batchnorm/mul_2Mul8batch_normalization_7/batchnorm/ReadVariableOp_1:value:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_7/batchnorm/mul_2Ú
0batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_7/batchnorm/ReadVariableOp_2Û
#batch_normalization_7/batchnorm/subSub8batch_normalization_7/batchnorm/ReadVariableOp_2:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_7/batchnorm/subá
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_7/batchnorm/add_1«
	add_2/addAddV2)batch_normalization_6/batchnorm/add_1:z:0)batch_normalization_7/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
	add_2/add¹
Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_5_multi_head_attention_5_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpÐ
>transformer_block_5/multi_head_attention_5/query/einsum/EinsumEinsumadd_2/add:z:0Utransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2@
>transformer_block_5/multi_head_attention_5/query/einsum/Einsum
Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOpReadVariableOpLtransformer_block_5_multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOpÅ
4transformer_block_5/multi_head_attention_5/query/addAddV2Gtransformer_block_5/multi_head_attention_5/query/einsum/Einsum:output:0Ktransformer_block_5/multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_5/multi_head_attention_5/query/add³
Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_5_multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpÊ
<transformer_block_5/multi_head_attention_5/key/einsum/EinsumEinsumadd_2/add:z:0Stransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2>
<transformer_block_5/multi_head_attention_5/key/einsum/Einsum
Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOpReadVariableOpJtransformer_block_5_multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp½
2transformer_block_5/multi_head_attention_5/key/addAddV2Etransformer_block_5/multi_head_attention_5/key/einsum/Einsum:output:0Itransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 24
2transformer_block_5/multi_head_attention_5/key/add¹
Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_5_multi_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpÐ
>transformer_block_5/multi_head_attention_5/value/einsum/EinsumEinsumadd_2/add:z:0Utransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2@
>transformer_block_5/multi_head_attention_5/value/einsum/Einsum
Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOpReadVariableOpLtransformer_block_5_multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOpÅ
4transformer_block_5/multi_head_attention_5/value/addAddV2Gtransformer_block_5/multi_head_attention_5/value/einsum/Einsum:output:0Ktransformer_block_5/multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_5/multi_head_attention_5/value/add©
0transformer_block_5/multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>22
0transformer_block_5/multi_head_attention_5/Mul/y
.transformer_block_5/multi_head_attention_5/MulMul8transformer_block_5/multi_head_attention_5/query/add:z:09transformer_block_5/multi_head_attention_5/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 20
.transformer_block_5/multi_head_attention_5/MulÌ
8transformer_block_5/multi_head_attention_5/einsum/EinsumEinsum6transformer_block_5/multi_head_attention_5/key/add:z:02transformer_block_5/multi_head_attention_5/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2:
8transformer_block_5/multi_head_attention_5/einsum/Einsum
:transformer_block_5/multi_head_attention_5/softmax/SoftmaxSoftmaxAtransformer_block_5/multi_head_attention_5/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2<
:transformer_block_5/multi_head_attention_5/softmax/Softmax
;transformer_block_5/multi_head_attention_5/dropout/IdentityIdentityDtransformer_block_5/multi_head_attention_5/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2=
;transformer_block_5/multi_head_attention_5/dropout/Identityä
:transformer_block_5/multi_head_attention_5/einsum_1/EinsumEinsumDtransformer_block_5/multi_head_attention_5/dropout/Identity:output:08transformer_block_5/multi_head_attention_5/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2<
:transformer_block_5/multi_head_attention_5/einsum_1/EinsumÚ
Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_5_multi_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp£
Itransformer_block_5/multi_head_attention_5/attention_output/einsum/EinsumEinsumCtransformer_block_5/multi_head_attention_5/einsum_1/Einsum:output:0`transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe2K
Itransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum´
Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_5_multi_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpí
?transformer_block_5/multi_head_attention_5/attention_output/addAddV2Rtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum:output:0Vtransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?transformer_block_5/multi_head_attention_5/attention_output/addÙ
'transformer_block_5/dropout_14/IdentityIdentityCtransformer_block_5/multi_head_attention_5/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2)
'transformer_block_5/dropout_14/Identity²
transformer_block_5/addAddV2add_2/add:z:00transformer_block_5/dropout_14/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_5/addà
Itransformer_block_5/layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_5/layer_normalization_10/moments/mean/reduction_indices²
7transformer_block_5/layer_normalization_10/moments/meanMeantransformer_block_5/add:z:0Rtransformer_block_5/layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(29
7transformer_block_5/layer_normalization_10/moments/mean
?transformer_block_5/layer_normalization_10/moments/StopGradientStopGradient@transformer_block_5/layer_normalization_10/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2A
?transformer_block_5/layer_normalization_10/moments/StopGradient¾
Dtransformer_block_5/layer_normalization_10/moments/SquaredDifferenceSquaredDifferencetransformer_block_5/add:z:0Htransformer_block_5/layer_normalization_10/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2F
Dtransformer_block_5/layer_normalization_10/moments/SquaredDifferenceè
Mtransformer_block_5/layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_5/layer_normalization_10/moments/variance/reduction_indicesë
;transformer_block_5/layer_normalization_10/moments/varianceMeanHtransformer_block_5/layer_normalization_10/moments/SquaredDifference:z:0Vtransformer_block_5/layer_normalization_10/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2=
;transformer_block_5/layer_normalization_10/moments/variance½
:transformer_block_5/layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_5/layer_normalization_10/batchnorm/add/y¾
8transformer_block_5/layer_normalization_10/batchnorm/addAddV2Dtransformer_block_5/layer_normalization_10/moments/variance:output:0Ctransformer_block_5/layer_normalization_10/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2:
8transformer_block_5/layer_normalization_10/batchnorm/addõ
:transformer_block_5/layer_normalization_10/batchnorm/RsqrtRsqrt<transformer_block_5/layer_normalization_10/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2<
:transformer_block_5/layer_normalization_10/batchnorm/Rsqrt
Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_5_layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpÂ
8transformer_block_5/layer_normalization_10/batchnorm/mulMul>transformer_block_5/layer_normalization_10/batchnorm/Rsqrt:y:0Otransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2:
8transformer_block_5/layer_normalization_10/batchnorm/mul
:transformer_block_5/layer_normalization_10/batchnorm/mul_1Multransformer_block_5/add:z:0<transformer_block_5/layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_5/layer_normalization_10/batchnorm/mul_1µ
:transformer_block_5/layer_normalization_10/batchnorm/mul_2Mul@transformer_block_5/layer_normalization_10/moments/mean:output:0<transformer_block_5/layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_5/layer_normalization_10/batchnorm/mul_2
Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_5_layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp¾
8transformer_block_5/layer_normalization_10/batchnorm/subSubKtransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp:value:0>transformer_block_5/layer_normalization_10/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2:
8transformer_block_5/layer_normalization_10/batchnorm/subµ
:transformer_block_5/layer_normalization_10/batchnorm/add_1AddV2>transformer_block_5/layer_normalization_10/batchnorm/mul_1:z:0<transformer_block_5/layer_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_5/layer_normalization_10/batchnorm/add_1
Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_5_sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02D
Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp¾
8transformer_block_5/sequential_5/dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_5/sequential_5/dense_16/Tensordot/axesÅ
8transformer_block_5/sequential_5/dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_5/sequential_5/dense_16/Tensordot/freeä
9transformer_block_5/sequential_5/dense_16/Tensordot/ShapeShape>transformer_block_5/layer_normalization_10/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_5/sequential_5/dense_16/Tensordot/ShapeÈ
Atransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2/axis£
<transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2GatherV2Btransformer_block_5/sequential_5/dense_16/Tensordot/Shape:output:0Atransformer_block_5/sequential_5/dense_16/Tensordot/free:output:0Jtransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2Ì
Ctransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1/axis©
>transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1GatherV2Btransformer_block_5/sequential_5/dense_16/Tensordot/Shape:output:0Atransformer_block_5/sequential_5/dense_16/Tensordot/axes:output:0Ltransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1À
9transformer_block_5/sequential_5/dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_5/sequential_5/dense_16/Tensordot/Const¨
8transformer_block_5/sequential_5/dense_16/Tensordot/ProdProdEtransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2:output:0Btransformer_block_5/sequential_5/dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_5/sequential_5/dense_16/Tensordot/ProdÄ
;transformer_block_5/sequential_5/dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_5/sequential_5/dense_16/Tensordot/Const_1°
:transformer_block_5/sequential_5/dense_16/Tensordot/Prod_1ProdGtransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1:output:0Dtransformer_block_5/sequential_5/dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_5/sequential_5/dense_16/Tensordot/Prod_1Ä
?transformer_block_5/sequential_5/dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_5/sequential_5/dense_16/Tensordot/concat/axis
:transformer_block_5/sequential_5/dense_16/Tensordot/concatConcatV2Atransformer_block_5/sequential_5/dense_16/Tensordot/free:output:0Atransformer_block_5/sequential_5/dense_16/Tensordot/axes:output:0Htransformer_block_5/sequential_5/dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_5/sequential_5/dense_16/Tensordot/concat´
9transformer_block_5/sequential_5/dense_16/Tensordot/stackPackAtransformer_block_5/sequential_5/dense_16/Tensordot/Prod:output:0Ctransformer_block_5/sequential_5/dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_5/sequential_5/dense_16/Tensordot/stackÆ
=transformer_block_5/sequential_5/dense_16/Tensordot/transpose	Transpose>transformer_block_5/layer_normalization_10/batchnorm/add_1:z:0Ctransformer_block_5/sequential_5/dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2?
=transformer_block_5/sequential_5/dense_16/Tensordot/transposeÇ
;transformer_block_5/sequential_5/dense_16/Tensordot/ReshapeReshapeAtransformer_block_5/sequential_5/dense_16/Tensordot/transpose:y:0Btransformer_block_5/sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2=
;transformer_block_5/sequential_5/dense_16/Tensordot/ReshapeÆ
:transformer_block_5/sequential_5/dense_16/Tensordot/MatMulMatMulDtransformer_block_5/sequential_5/dense_16/Tensordot/Reshape:output:0Jtransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2<
:transformer_block_5/sequential_5/dense_16/Tensordot/MatMulÄ
;transformer_block_5/sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2=
;transformer_block_5/sequential_5/dense_16/Tensordot/Const_2È
Atransformer_block_5/sequential_5/dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_5/sequential_5/dense_16/Tensordot/concat_1/axis
<transformer_block_5/sequential_5/dense_16/Tensordot/concat_1ConcatV2Etransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2:output:0Dtransformer_block_5/sequential_5/dense_16/Tensordot/Const_2:output:0Jtransformer_block_5/sequential_5/dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_5/sequential_5/dense_16/Tensordot/concat_1¸
3transformer_block_5/sequential_5/dense_16/TensordotReshapeDtransformer_block_5/sequential_5/dense_16/Tensordot/MatMul:product:0Etransformer_block_5/sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@25
3transformer_block_5/sequential_5/dense_16/Tensordot
@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_5_sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp¯
1transformer_block_5/sequential_5/dense_16/BiasAddBiasAdd<transformer_block_5/sequential_5/dense_16/Tensordot:output:0Htransformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@23
1transformer_block_5/sequential_5/dense_16/BiasAddÚ
.transformer_block_5/sequential_5/dense_16/ReluRelu:transformer_block_5/sequential_5/dense_16/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@20
.transformer_block_5/sequential_5/dense_16/Relu
Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_5_sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02D
Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp¾
8transformer_block_5/sequential_5/dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_5/sequential_5/dense_17/Tensordot/axesÅ
8transformer_block_5/sequential_5/dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_5/sequential_5/dense_17/Tensordot/freeâ
9transformer_block_5/sequential_5/dense_17/Tensordot/ShapeShape<transformer_block_5/sequential_5/dense_16/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_5/sequential_5/dense_17/Tensordot/ShapeÈ
Atransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2/axis£
<transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2GatherV2Btransformer_block_5/sequential_5/dense_17/Tensordot/Shape:output:0Atransformer_block_5/sequential_5/dense_17/Tensordot/free:output:0Jtransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2Ì
Ctransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1/axis©
>transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1GatherV2Btransformer_block_5/sequential_5/dense_17/Tensordot/Shape:output:0Atransformer_block_5/sequential_5/dense_17/Tensordot/axes:output:0Ltransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1À
9transformer_block_5/sequential_5/dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_5/sequential_5/dense_17/Tensordot/Const¨
8transformer_block_5/sequential_5/dense_17/Tensordot/ProdProdEtransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2:output:0Btransformer_block_5/sequential_5/dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_5/sequential_5/dense_17/Tensordot/ProdÄ
;transformer_block_5/sequential_5/dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_5/sequential_5/dense_17/Tensordot/Const_1°
:transformer_block_5/sequential_5/dense_17/Tensordot/Prod_1ProdGtransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1:output:0Dtransformer_block_5/sequential_5/dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_5/sequential_5/dense_17/Tensordot/Prod_1Ä
?transformer_block_5/sequential_5/dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_5/sequential_5/dense_17/Tensordot/concat/axis
:transformer_block_5/sequential_5/dense_17/Tensordot/concatConcatV2Atransformer_block_5/sequential_5/dense_17/Tensordot/free:output:0Atransformer_block_5/sequential_5/dense_17/Tensordot/axes:output:0Htransformer_block_5/sequential_5/dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_5/sequential_5/dense_17/Tensordot/concat´
9transformer_block_5/sequential_5/dense_17/Tensordot/stackPackAtransformer_block_5/sequential_5/dense_17/Tensordot/Prod:output:0Ctransformer_block_5/sequential_5/dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_5/sequential_5/dense_17/Tensordot/stackÄ
=transformer_block_5/sequential_5/dense_17/Tensordot/transpose	Transpose<transformer_block_5/sequential_5/dense_16/Relu:activations:0Ctransformer_block_5/sequential_5/dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2?
=transformer_block_5/sequential_5/dense_17/Tensordot/transposeÇ
;transformer_block_5/sequential_5/dense_17/Tensordot/ReshapeReshapeAtransformer_block_5/sequential_5/dense_17/Tensordot/transpose:y:0Btransformer_block_5/sequential_5/dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2=
;transformer_block_5/sequential_5/dense_17/Tensordot/ReshapeÆ
:transformer_block_5/sequential_5/dense_17/Tensordot/MatMulMatMulDtransformer_block_5/sequential_5/dense_17/Tensordot/Reshape:output:0Jtransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2<
:transformer_block_5/sequential_5/dense_17/Tensordot/MatMulÄ
;transformer_block_5/sequential_5/dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_5/sequential_5/dense_17/Tensordot/Const_2È
Atransformer_block_5/sequential_5/dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_5/sequential_5/dense_17/Tensordot/concat_1/axis
<transformer_block_5/sequential_5/dense_17/Tensordot/concat_1ConcatV2Etransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2:output:0Dtransformer_block_5/sequential_5/dense_17/Tensordot/Const_2:output:0Jtransformer_block_5/sequential_5/dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_5/sequential_5/dense_17/Tensordot/concat_1¸
3transformer_block_5/sequential_5/dense_17/TensordotReshapeDtransformer_block_5/sequential_5/dense_17/Tensordot/MatMul:product:0Etransformer_block_5/sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 25
3transformer_block_5/sequential_5/dense_17/Tensordot
@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_5_sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp¯
1transformer_block_5/sequential_5/dense_17/BiasAddBiasAdd<transformer_block_5/sequential_5/dense_17/Tensordot:output:0Htransformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 23
1transformer_block_5/sequential_5/dense_17/BiasAddÐ
'transformer_block_5/dropout_15/IdentityIdentity:transformer_block_5/sequential_5/dense_17/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2)
'transformer_block_5/dropout_15/Identityç
transformer_block_5/add_1AddV2>transformer_block_5/layer_normalization_10/batchnorm/add_1:z:00transformer_block_5/dropout_15/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_5/add_1à
Itransformer_block_5/layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_5/layer_normalization_11/moments/mean/reduction_indices´
7transformer_block_5/layer_normalization_11/moments/meanMeantransformer_block_5/add_1:z:0Rtransformer_block_5/layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(29
7transformer_block_5/layer_normalization_11/moments/mean
?transformer_block_5/layer_normalization_11/moments/StopGradientStopGradient@transformer_block_5/layer_normalization_11/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2A
?transformer_block_5/layer_normalization_11/moments/StopGradientÀ
Dtransformer_block_5/layer_normalization_11/moments/SquaredDifferenceSquaredDifferencetransformer_block_5/add_1:z:0Htransformer_block_5/layer_normalization_11/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2F
Dtransformer_block_5/layer_normalization_11/moments/SquaredDifferenceè
Mtransformer_block_5/layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_5/layer_normalization_11/moments/variance/reduction_indicesë
;transformer_block_5/layer_normalization_11/moments/varianceMeanHtransformer_block_5/layer_normalization_11/moments/SquaredDifference:z:0Vtransformer_block_5/layer_normalization_11/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2=
;transformer_block_5/layer_normalization_11/moments/variance½
:transformer_block_5/layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_5/layer_normalization_11/batchnorm/add/y¾
8transformer_block_5/layer_normalization_11/batchnorm/addAddV2Dtransformer_block_5/layer_normalization_11/moments/variance:output:0Ctransformer_block_5/layer_normalization_11/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2:
8transformer_block_5/layer_normalization_11/batchnorm/addõ
:transformer_block_5/layer_normalization_11/batchnorm/RsqrtRsqrt<transformer_block_5/layer_normalization_11/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2<
:transformer_block_5/layer_normalization_11/batchnorm/Rsqrt
Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_5_layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpÂ
8transformer_block_5/layer_normalization_11/batchnorm/mulMul>transformer_block_5/layer_normalization_11/batchnorm/Rsqrt:y:0Otransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2:
8transformer_block_5/layer_normalization_11/batchnorm/mul
:transformer_block_5/layer_normalization_11/batchnorm/mul_1Multransformer_block_5/add_1:z:0<transformer_block_5/layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_5/layer_normalization_11/batchnorm/mul_1µ
:transformer_block_5/layer_normalization_11/batchnorm/mul_2Mul@transformer_block_5/layer_normalization_11/moments/mean:output:0<transformer_block_5/layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_5/layer_normalization_11/batchnorm/mul_2
Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_5_layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp¾
8transformer_block_5/layer_normalization_11/batchnorm/subSubKtransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp:value:0>transformer_block_5/layer_normalization_11/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2:
8transformer_block_5/layer_normalization_11/batchnorm/subµ
:transformer_block_5/layer_normalization_11/batchnorm/add_1AddV2>transformer_block_5/layer_normalization_11/batchnorm/mul_1:z:0<transformer_block_5/layer_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_5/layer_normalization_11/batchnorm/add_1s
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`  2
flatten_2/Const¾
flatten_2/ReshapeReshape>transformer_block_5/layer_normalization_11/batchnorm/add_1:z:0flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
flatten_2/ReshapeÔ
.batch_normalization_8/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_8/batchnorm/ReadVariableOp
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_8/batchnorm/add/yà
#batch_normalization_8/batchnorm/addAddV26batch_normalization_8/batchnorm/ReadVariableOp:value:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/add¥
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/Rsqrtà
2batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_8/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:0:batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/mulº
%batch_normalization_8/batchnorm/mul_1Mulinputs_1'batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_8/batchnorm/mul_1Ú
0batch_normalization_8/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_8_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_8/batchnorm/ReadVariableOp_1Ý
%batch_normalization_8/batchnorm/mul_2Mul8batch_normalization_8/batchnorm/ReadVariableOp_1:value:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/mul_2Ú
0batch_normalization_8/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_8_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_8/batchnorm/ReadVariableOp_2Û
#batch_normalization_8/batchnorm/subSub8batch_normalization_8/batchnorm/ReadVariableOp_2:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/subÝ
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_8/batchnorm/add_1x
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axisß
concatenate_2/concatConcatV2flatten_2/Reshape:output:0)batch_normalization_8/batchnorm/add_1:z:0"concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
concatenate_2/concat©
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes
:	è@*
dtype02 
dense_18/MatMul/ReadVariableOp¥
dense_18/MatMulMatMulconcatenate_2/concat:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_18/MatMul§
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_18/BiasAdd/ReadVariableOp¥
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_18/BiasAdds
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_18/Relu
dropout_16/IdentityIdentitydense_18/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_16/Identity¨
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_19/MatMul/ReadVariableOp¤
dense_19/MatMulMatMuldropout_16/Identity:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_19/MatMul§
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_19/BiasAdd/ReadVariableOp¥
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_19/BiasAdds
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_19/Relu
dropout_17/IdentityIdentitydense_19/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_17/Identity¨
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_20/MatMul/ReadVariableOp¤
dense_20/MatMulMatMuldropout_17/Identity:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_20/MatMul§
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp¥
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_20/BiasAddÐ
IdentityIdentitydense_20/BiasAdd:output:0/^batch_normalization_6/batchnorm/ReadVariableOp1^batch_normalization_6/batchnorm/ReadVariableOp_11^batch_normalization_6/batchnorm/ReadVariableOp_23^batch_normalization_6/batchnorm/mul/ReadVariableOp/^batch_normalization_7/batchnorm/ReadVariableOp1^batch_normalization_7/batchnorm/ReadVariableOp_11^batch_normalization_7/batchnorm/ReadVariableOp_23^batch_normalization_7/batchnorm/mul/ReadVariableOp/^batch_normalization_8/batchnorm/ReadVariableOp1^batch_normalization_8/batchnorm/ReadVariableOp_11^batch_normalization_8/batchnorm/ReadVariableOp_23^batch_normalization_8/batchnorm/mul/ReadVariableOp ^conv1d_4/BiasAdd/ReadVariableOp,^conv1d_4/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_5/BiasAdd/ReadVariableOp,^conv1d_5/conv1d/ExpandDims_1/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp<^token_and_position_embedding_2/embedding_4/embedding_lookup<^token_and_position_embedding_2/embedding_5/embedding_lookupD^transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpH^transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpD^transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpH^transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpO^transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpY^transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_5/multi_head_attention_5/key/add/ReadVariableOpL^transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpD^transformer_block_5/multi_head_attention_5/query/add/ReadVariableOpN^transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpD^transformer_block_5/multi_head_attention_5/value/add/ReadVariableOpN^transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpA^transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpC^transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpA^transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpC^transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_6/batchnorm/ReadVariableOp.batch_normalization_6/batchnorm/ReadVariableOp2d
0batch_normalization_6/batchnorm/ReadVariableOp_10batch_normalization_6/batchnorm/ReadVariableOp_12d
0batch_normalization_6/batchnorm/ReadVariableOp_20batch_normalization_6/batchnorm/ReadVariableOp_22h
2batch_normalization_6/batchnorm/mul/ReadVariableOp2batch_normalization_6/batchnorm/mul/ReadVariableOp2`
.batch_normalization_7/batchnorm/ReadVariableOp.batch_normalization_7/batchnorm/ReadVariableOp2d
0batch_normalization_7/batchnorm/ReadVariableOp_10batch_normalization_7/batchnorm/ReadVariableOp_12d
0batch_normalization_7/batchnorm/ReadVariableOp_20batch_normalization_7/batchnorm/ReadVariableOp_22h
2batch_normalization_7/batchnorm/mul/ReadVariableOp2batch_normalization_7/batchnorm/mul/ReadVariableOp2`
.batch_normalization_8/batchnorm/ReadVariableOp.batch_normalization_8/batchnorm/ReadVariableOp2d
0batch_normalization_8/batchnorm/ReadVariableOp_10batch_normalization_8/batchnorm/ReadVariableOp_12d
0batch_normalization_8/batchnorm/ReadVariableOp_20batch_normalization_8/batchnorm/ReadVariableOp_22h
2batch_normalization_8/batchnorm/mul/ReadVariableOp2batch_normalization_8/batchnorm/mul/ReadVariableOp2B
conv1d_4/BiasAdd/ReadVariableOpconv1d_4/BiasAdd/ReadVariableOp2Z
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_5/BiasAdd/ReadVariableOpconv1d_5/BiasAdd/ReadVariableOp2Z
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2z
;token_and_position_embedding_2/embedding_4/embedding_lookup;token_and_position_embedding_2/embedding_4/embedding_lookup2z
;token_and_position_embedding_2/embedding_5/embedding_lookup;token_and_position_embedding_2/embedding_5/embedding_lookup2
Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpCtransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp2
Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpGtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp2
Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpCtransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp2
Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpGtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp2 
Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpNtransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp2´
Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp2
Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOpAtransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp2
Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpKtransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp2
Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOpCtransformer_block_5/multi_head_attention_5/query/add/ReadVariableOp2
Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpMtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp2
Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOpCtransformer_block_5/multi_head_attention_5/value/add/ReadVariableOp2
Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpMtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp2
@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp2
Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpBtransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp2
@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp2
Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOpBtransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp:R N
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

÷
D__inference_conv1d_5_layer_call_and_return_conditional_losses_421702

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
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
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
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÞ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 
 
_user_specified_nameinputs

P
4__inference_average_pooling1d_8_layer_call_fn_421021

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
O__inference_average_pooling1d_8_layer_call_and_return_conditional_losses_4210152
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
î
©
6__inference_batch_normalization_7_layer_call_fn_424185

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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4212902
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
0
È
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_421564

inputs
assignmovingavg_421539
assignmovingavg_1_421545)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/421539*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_421539*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/421539*
_output_shapes
:2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/421539*
_output_shapes
:2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_421539AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/421539*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/421545*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_421545*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/421545*
_output_shapes
:2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/421545*
_output_shapes
:2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_421545AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/421545*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

e
F__inference_dropout_16_layer_call_and_return_conditional_losses_422405

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
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
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
¼0
È
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_424057

inputs
assignmovingavg_424032
assignmovingavg_1_424038)
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
loc:@AssignMovingAvg/424032*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_424032*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/424032*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/424032*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_424032AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/424032*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/424038*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_424038*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/424038*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/424038*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_424038AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/424038*
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
º
s
I__inference_concatenate_2_layer_call_and_return_conditional_losses_422357

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
è

Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_424077

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
ï
~
)__inference_dense_17_layer_call_fn_425066

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
D__inference_dense_17_layer_call_and_return_conditional_losses_4213822
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
¸
 
-__inference_sequential_5_layer_call_fn_424974

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
H__inference_sequential_5_layer_call_and_return_conditional_losses_4214302
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
ô

Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_421637
x'
#embedding_5_embedding_lookup_421624'
#embedding_4_embedding_lookup_421630
identity¢embedding_4/embedding_lookup¢embedding_5/embedding_lookup?
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
embedding_5/embedding_lookupResourceGather#embedding_5_embedding_lookup_421624range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_5/embedding_lookup/421624*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02
embedding_5/embedding_lookup
%embedding_5/embedding_lookup/IdentityIdentity%embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_5/embedding_lookup/421624*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%embedding_5/embedding_lookup/IdentityÀ
'embedding_5/embedding_lookup/Identity_1Identity.embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'embedding_5/embedding_lookup/Identity_1q
embedding_4/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2
embedding_4/Castº
embedding_4/embedding_lookupResourceGather#embedding_4_embedding_lookup_421630embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_4/embedding_lookup/421630*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02
embedding_4/embedding_lookup
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_4/embedding_lookup/421630*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2'
%embedding_4/embedding_lookup/IdentityÅ
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2)
'embedding_4/embedding_lookup/Identity_1®
addAddV20embedding_4/embedding_lookup/Identity_1:output:00embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
add
IdentityIdentityadd:z:0^embedding_4/embedding_lookup^embedding_5/embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿR::2<
embedding_4/embedding_lookupembedding_4/embedding_lookup2<
embedding_5/embedding_lookupembedding_5/embedding_lookup:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR

_user_specified_namex
	
Ý
D__inference_dense_20_layer_call_and_return_conditional_losses_422490

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
Ê
©
6__inference_batch_normalization_7_layer_call_fn_424267

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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4218662
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
µ
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_422307

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
Ð

à
4__inference_transformer_block_5_layer_call_fn_424591

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
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_4220652
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
Ñ
ã
D__inference_dense_17_layer_call_and_return_conditional_losses_425057

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
È
©
6__inference_batch_normalization_6_layer_call_fn_424090

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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4217552
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
÷
k
O__inference_average_pooling1d_8_layer_call_and_return_conditional_losses_421015

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
®
Ø*
C__inference_model_2_layer_call_and_return_conditional_losses_423425
inputs_0
inputs_1F
Btoken_and_position_embedding_2_embedding_5_embedding_lookup_423095F
Btoken_and_position_embedding_2_embedding_4_embedding_lookup_4231018
4conv1d_4_conv1d_expanddims_1_readvariableop_resource,
(conv1d_4_biasadd_readvariableop_resource8
4conv1d_5_conv1d_expanddims_1_readvariableop_resource,
(conv1d_5_biasadd_readvariableop_resource0
,batch_normalization_6_assignmovingavg_4231512
.batch_normalization_6_assignmovingavg_1_423157?
;batch_normalization_6_batchnorm_mul_readvariableop_resource;
7batch_normalization_6_batchnorm_readvariableop_resource0
,batch_normalization_7_assignmovingavg_4231832
.batch_normalization_7_assignmovingavg_1_423189?
;batch_normalization_7_batchnorm_mul_readvariableop_resource;
7batch_normalization_7_batchnorm_readvariableop_resourceZ
Vtransformer_block_5_multi_head_attention_5_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_5_multi_head_attention_5_query_add_readvariableop_resourceX
Ttransformer_block_5_multi_head_attention_5_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_5_multi_head_attention_5_key_add_readvariableop_resourceZ
Vtransformer_block_5_multi_head_attention_5_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_5_multi_head_attention_5_value_add_readvariableop_resourcee
atransformer_block_5_multi_head_attention_5_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_5_multi_head_attention_5_attention_output_add_readvariableop_resourceT
Ptransformer_block_5_layer_normalization_10_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_5_layer_normalization_10_batchnorm_readvariableop_resourceO
Ktransformer_block_5_sequential_5_dense_16_tensordot_readvariableop_resourceM
Itransformer_block_5_sequential_5_dense_16_biasadd_readvariableop_resourceO
Ktransformer_block_5_sequential_5_dense_17_tensordot_readvariableop_resourceM
Itransformer_block_5_sequential_5_dense_17_biasadd_readvariableop_resourceT
Ptransformer_block_5_layer_normalization_11_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_5_layer_normalization_11_batchnorm_readvariableop_resource0
,batch_normalization_8_assignmovingavg_4233622
.batch_normalization_8_assignmovingavg_1_423368?
;batch_normalization_8_batchnorm_mul_readvariableop_resource;
7batch_normalization_8_batchnorm_readvariableop_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource
identity¢9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_6/AssignMovingAvg/ReadVariableOp¢;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_6/batchnorm/ReadVariableOp¢2batch_normalization_6/batchnorm/mul/ReadVariableOp¢9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_7/AssignMovingAvg/ReadVariableOp¢;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_7/batchnorm/ReadVariableOp¢2batch_normalization_7/batchnorm/mul/ReadVariableOp¢9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_8/AssignMovingAvg/ReadVariableOp¢;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_8/batchnorm/ReadVariableOp¢2batch_normalization_8/batchnorm/mul/ReadVariableOp¢conv1d_4/BiasAdd/ReadVariableOp¢+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp¢conv1d_5/BiasAdd/ReadVariableOp¢+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp¢dense_18/BiasAdd/ReadVariableOp¢dense_18/MatMul/ReadVariableOp¢dense_19/BiasAdd/ReadVariableOp¢dense_19/MatMul/ReadVariableOp¢dense_20/BiasAdd/ReadVariableOp¢dense_20/MatMul/ReadVariableOp¢;token_and_position_embedding_2/embedding_4/embedding_lookup¢;token_and_position_embedding_2/embedding_5/embedding_lookup¢Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp¢Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp¢Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp¢Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp¢Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp¢Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp¢Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp¢Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp¢Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOp¢Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp¢Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOp¢Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp¢@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp¢Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp¢@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp¢Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp
$token_and_position_embedding_2/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_2/Shape»
2token_and_position_embedding_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ24
2token_and_position_embedding_2/strided_slice/stack¶
4token_and_position_embedding_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_2/strided_slice/stack_1¶
4token_and_position_embedding_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_2/strided_slice/stack_2
,token_and_position_embedding_2/strided_sliceStridedSlice-token_and_position_embedding_2/Shape:output:0;token_and_position_embedding_2/strided_slice/stack:output:0=token_and_position_embedding_2/strided_slice/stack_1:output:0=token_and_position_embedding_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_2/strided_slice
*token_and_position_embedding_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_2/range/start
*token_and_position_embedding_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_2/range/delta
$token_and_position_embedding_2/rangeRange3token_and_position_embedding_2/range/start:output:05token_and_position_embedding_2/strided_slice:output:03token_and_position_embedding_2/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$token_and_position_embedding_2/rangeÊ
;token_and_position_embedding_2/embedding_5/embedding_lookupResourceGatherBtoken_and_position_embedding_2_embedding_5_embedding_lookup_423095-token_and_position_embedding_2/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/423095*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02=
;token_and_position_embedding_2/embedding_5/embedding_lookup
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/423095*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2F
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2H
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1¶
/token_and_position_embedding_2/embedding_4/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR21
/token_and_position_embedding_2/embedding_4/CastÕ
;token_and_position_embedding_2/embedding_4/embedding_lookupResourceGatherBtoken_and_position_embedding_2_embedding_4_embedding_lookup_4231013token_and_position_embedding_2/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/423101*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02=
;token_and_position_embedding_2/embedding_4/embedding_lookup
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/423101*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2F
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity¢
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2H
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1ª
"token_and_position_embedding_2/addAddV2Otoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2$
"token_and_position_embedding_2/add
conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_4/conv1d/ExpandDims/dimÒ
conv1d_4/conv1d/ExpandDims
ExpandDims&token_and_position_embedding_2/add:z:0'conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_4/conv1d/ExpandDimsÓ
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02-
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_4/conv1d/ExpandDims_1/dimÛ
conv1d_4/conv1d/ExpandDims_1
ExpandDims3conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_4/conv1d/ExpandDims_1Û
conv1d_4/conv1dConv2D#conv1d_4/conv1d/ExpandDims:output:0%conv1d_4/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
paddingSAME*
strides
2
conv1d_4/conv1d®
conv1d_4/conv1d/SqueezeSqueezeconv1d_4/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_4/conv1d/Squeeze§
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_4/BiasAdd/ReadVariableOp±
conv1d_4/BiasAddBiasAdd conv1d_4/conv1d/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_4/BiasAddx
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_4/Relu
"average_pooling1d_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_6/ExpandDims/dimÓ
average_pooling1d_6/ExpandDims
ExpandDimsconv1d_4/Relu:activations:0+average_pooling1d_6/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2 
average_pooling1d_6/ExpandDimså
average_pooling1d_6/AvgPoolAvgPool'average_pooling1d_6/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_6/AvgPool¹
average_pooling1d_6/SqueezeSqueeze$average_pooling1d_6/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims
2
average_pooling1d_6/Squeeze
conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_5/conv1d/ExpandDims/dimÐ
conv1d_5/conv1d/ExpandDims
ExpandDims$average_pooling1d_6/Squeeze:output:0'conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_5/conv1d/ExpandDimsÓ
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02-
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_5/conv1d/ExpandDims_1/dimÛ
conv1d_5/conv1d/ExpandDims_1
ExpandDims3conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d_5/conv1d/ExpandDims_1Û
conv1d_5/conv1dConv2D#conv1d_5/conv1d/ExpandDims:output:0%conv1d_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
paddingSAME*
strides
2
conv1d_5/conv1d®
conv1d_5/conv1d/SqueezeSqueezeconv1d_5/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_5/conv1d/Squeeze§
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_5/BiasAdd/ReadVariableOp±
conv1d_5/BiasAddBiasAdd conv1d_5/conv1d/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_5/BiasAddx
conv1d_5/ReluReluconv1d_5/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_5/Relu
"average_pooling1d_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_8/ExpandDims/dimÞ
average_pooling1d_8/ExpandDims
ExpandDims&token_and_position_embedding_2/add:z:0+average_pooling1d_8/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2 
average_pooling1d_8/ExpandDimsæ
average_pooling1d_8/AvgPoolAvgPool'average_pooling1d_8/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize	
¬*
paddingVALID*
strides	
¬2
average_pooling1d_8/AvgPool¸
average_pooling1d_8/SqueezeSqueeze$average_pooling1d_8/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_8/Squeeze
"average_pooling1d_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_7/ExpandDims/dimÓ
average_pooling1d_7/ExpandDims
ExpandDimsconv1d_5/Relu:activations:0+average_pooling1d_7/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2 
average_pooling1d_7/ExpandDimsä
average_pooling1d_7/AvgPoolAvgPool'average_pooling1d_7/ExpandDims:output:0*
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
average_pooling1d_7/AvgPool¸
average_pooling1d_7/SqueezeSqueeze$average_pooling1d_7/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_7/Squeeze½
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_6/moments/mean/reduction_indicesó
"batch_normalization_6/moments/meanMean$average_pooling1d_7/Squeeze:output:0=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_6/moments/meanÂ
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_6/moments/StopGradient
/batch_normalization_6/moments/SquaredDifferenceSquaredDifference$average_pooling1d_7/Squeeze:output:03batch_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/batch_normalization_6/moments/SquaredDifferenceÅ
8batch_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_6/moments/variance/reduction_indices
&batch_normalization_6/moments/varianceMean3batch_normalization_6/moments/SquaredDifference:z:0Abatch_normalization_6/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2(
&batch_normalization_6/moments/varianceÃ
%batch_normalization_6/moments/SqueezeSqueeze+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_6/moments/SqueezeË
'batch_normalization_6/moments/Squeeze_1Squeeze/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_6/moments/Squeeze_1
+batch_normalization_6/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg/423151*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_6/AssignMovingAvg/decayÕ
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_6_assignmovingavg_423151*
_output_shapes
: *
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOpß
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg/423151*
_output_shapes
: 2+
)batch_normalization_6/AssignMovingAvg/subÖ
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:04batch_normalization_6/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg/423151*
_output_shapes
: 2+
)batch_normalization_6/AssignMovingAvg/mul³
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_6_assignmovingavg_423151-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg/423151*
_output_shapes
 *
dtype02;
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_6/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/423157*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_6/AssignMovingAvg_1/decayÛ
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_6_assignmovingavg_1_423157*
_output_shapes
: *
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpé
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/423157*
_output_shapes
: 2-
+batch_normalization_6/AssignMovingAvg_1/subà
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:06batch_normalization_6/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/423157*
_output_shapes
: 2-
+batch_normalization_6/AssignMovingAvg_1/mul¿
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_6_assignmovingavg_1_423157/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/423157*
_output_shapes
 *
dtype02=
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_6/batchnorm/add/yÚ
#batch_normalization_6/batchnorm/addAddV20batch_normalization_6/moments/Squeeze_1:output:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_6/batchnorm/add¥
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_6/batchnorm/Rsqrtà
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_6/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_6/batchnorm/mulÚ
%batch_normalization_6/batchnorm/mul_1Mul$average_pooling1d_7/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_6/batchnorm/mul_1Ó
%batch_normalization_6/batchnorm/mul_2Mul.batch_normalization_6/moments/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_6/batchnorm/mul_2Ô
.batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_6/batchnorm/ReadVariableOpÙ
#batch_normalization_6/batchnorm/subSub6batch_normalization_6/batchnorm/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_6/batchnorm/subá
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_6/batchnorm/add_1½
4batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_7/moments/mean/reduction_indicesó
"batch_normalization_7/moments/meanMean$average_pooling1d_8/Squeeze:output:0=batch_normalization_7/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_7/moments/meanÂ
*batch_normalization_7/moments/StopGradientStopGradient+batch_normalization_7/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_7/moments/StopGradient
/batch_normalization_7/moments/SquaredDifferenceSquaredDifference$average_pooling1d_8/Squeeze:output:03batch_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/batch_normalization_7/moments/SquaredDifferenceÅ
8batch_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_7/moments/variance/reduction_indices
&batch_normalization_7/moments/varianceMean3batch_normalization_7/moments/SquaredDifference:z:0Abatch_normalization_7/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2(
&batch_normalization_7/moments/varianceÃ
%batch_normalization_7/moments/SqueezeSqueeze+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_7/moments/SqueezeË
'batch_normalization_7/moments/Squeeze_1Squeeze/batch_normalization_7/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_7/moments/Squeeze_1
+batch_normalization_7/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg/423183*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_7/AssignMovingAvg/decayÕ
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_7_assignmovingavg_423183*
_output_shapes
: *
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOpß
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg/423183*
_output_shapes
: 2+
)batch_normalization_7/AssignMovingAvg/subÖ
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:04batch_normalization_7/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg/423183*
_output_shapes
: 2+
)batch_normalization_7/AssignMovingAvg/mul³
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_7_assignmovingavg_423183-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg/423183*
_output_shapes
 *
dtype02;
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_7/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/423189*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_7/AssignMovingAvg_1/decayÛ
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_7_assignmovingavg_1_423189*
_output_shapes
: *
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpé
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/423189*
_output_shapes
: 2-
+batch_normalization_7/AssignMovingAvg_1/subà
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:06batch_normalization_7/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/423189*
_output_shapes
: 2-
+batch_normalization_7/AssignMovingAvg_1/mul¿
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_7_assignmovingavg_1_423189/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/423189*
_output_shapes
 *
dtype02=
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_7/batchnorm/add/yÚ
#batch_normalization_7/batchnorm/addAddV20batch_normalization_7/moments/Squeeze_1:output:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_7/batchnorm/add¥
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_7/batchnorm/Rsqrtà
2batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_7/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:0:batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_7/batchnorm/mulÚ
%batch_normalization_7/batchnorm/mul_1Mul$average_pooling1d_8/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_7/batchnorm/mul_1Ó
%batch_normalization_7/batchnorm/mul_2Mul.batch_normalization_7/moments/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_7/batchnorm/mul_2Ô
.batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_7/batchnorm/ReadVariableOpÙ
#batch_normalization_7/batchnorm/subSub6batch_normalization_7/batchnorm/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_7/batchnorm/subá
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_7/batchnorm/add_1«
	add_2/addAddV2)batch_normalization_6/batchnorm/add_1:z:0)batch_normalization_7/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
	add_2/add¹
Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_5_multi_head_attention_5_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpÐ
>transformer_block_5/multi_head_attention_5/query/einsum/EinsumEinsumadd_2/add:z:0Utransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2@
>transformer_block_5/multi_head_attention_5/query/einsum/Einsum
Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOpReadVariableOpLtransformer_block_5_multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOpÅ
4transformer_block_5/multi_head_attention_5/query/addAddV2Gtransformer_block_5/multi_head_attention_5/query/einsum/Einsum:output:0Ktransformer_block_5/multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_5/multi_head_attention_5/query/add³
Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_5_multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpÊ
<transformer_block_5/multi_head_attention_5/key/einsum/EinsumEinsumadd_2/add:z:0Stransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2>
<transformer_block_5/multi_head_attention_5/key/einsum/Einsum
Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOpReadVariableOpJtransformer_block_5_multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp½
2transformer_block_5/multi_head_attention_5/key/addAddV2Etransformer_block_5/multi_head_attention_5/key/einsum/Einsum:output:0Itransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 24
2transformer_block_5/multi_head_attention_5/key/add¹
Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_5_multi_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpÐ
>transformer_block_5/multi_head_attention_5/value/einsum/EinsumEinsumadd_2/add:z:0Utransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2@
>transformer_block_5/multi_head_attention_5/value/einsum/Einsum
Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOpReadVariableOpLtransformer_block_5_multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOpÅ
4transformer_block_5/multi_head_attention_5/value/addAddV2Gtransformer_block_5/multi_head_attention_5/value/einsum/Einsum:output:0Ktransformer_block_5/multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_5/multi_head_attention_5/value/add©
0transformer_block_5/multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>22
0transformer_block_5/multi_head_attention_5/Mul/y
.transformer_block_5/multi_head_attention_5/MulMul8transformer_block_5/multi_head_attention_5/query/add:z:09transformer_block_5/multi_head_attention_5/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 20
.transformer_block_5/multi_head_attention_5/MulÌ
8transformer_block_5/multi_head_attention_5/einsum/EinsumEinsum6transformer_block_5/multi_head_attention_5/key/add:z:02transformer_block_5/multi_head_attention_5/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2:
8transformer_block_5/multi_head_attention_5/einsum/Einsum
:transformer_block_5/multi_head_attention_5/softmax/SoftmaxSoftmaxAtransformer_block_5/multi_head_attention_5/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2<
:transformer_block_5/multi_head_attention_5/softmax/SoftmaxÉ
@transformer_block_5/multi_head_attention_5/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2B
@transformer_block_5/multi_head_attention_5/dropout/dropout/ConstÒ
>transformer_block_5/multi_head_attention_5/dropout/dropout/MulMulDtransformer_block_5/multi_head_attention_5/softmax/Softmax:softmax:0Itransformer_block_5/multi_head_attention_5/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2@
>transformer_block_5/multi_head_attention_5/dropout/dropout/Mulø
@transformer_block_5/multi_head_attention_5/dropout/dropout/ShapeShapeDtransformer_block_5/multi_head_attention_5/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2B
@transformer_block_5/multi_head_attention_5/dropout/dropout/ShapeÕ
Wtransformer_block_5/multi_head_attention_5/dropout/dropout/random_uniform/RandomUniformRandomUniformItransformer_block_5/multi_head_attention_5/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
dtype02Y
Wtransformer_block_5/multi_head_attention_5/dropout/dropout/random_uniform/RandomUniformÛ
Itransformer_block_5/multi_head_attention_5/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2K
Itransformer_block_5/multi_head_attention_5/dropout/dropout/GreaterEqual/y
Gtransformer_block_5/multi_head_attention_5/dropout/dropout/GreaterEqualGreaterEqual`transformer_block_5/multi_head_attention_5/dropout/dropout/random_uniform/RandomUniform:output:0Rtransformer_block_5/multi_head_attention_5/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2I
Gtransformer_block_5/multi_head_attention_5/dropout/dropout/GreaterEqual 
?transformer_block_5/multi_head_attention_5/dropout/dropout/CastCastKtransformer_block_5/multi_head_attention_5/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2A
?transformer_block_5/multi_head_attention_5/dropout/dropout/CastÎ
@transformer_block_5/multi_head_attention_5/dropout/dropout/Mul_1MulBtransformer_block_5/multi_head_attention_5/dropout/dropout/Mul:z:0Ctransformer_block_5/multi_head_attention_5/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2B
@transformer_block_5/multi_head_attention_5/dropout/dropout/Mul_1ä
:transformer_block_5/multi_head_attention_5/einsum_1/EinsumEinsumDtransformer_block_5/multi_head_attention_5/dropout/dropout/Mul_1:z:08transformer_block_5/multi_head_attention_5/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2<
:transformer_block_5/multi_head_attention_5/einsum_1/EinsumÚ
Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_5_multi_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp£
Itransformer_block_5/multi_head_attention_5/attention_output/einsum/EinsumEinsumCtransformer_block_5/multi_head_attention_5/einsum_1/Einsum:output:0`transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe2K
Itransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum´
Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_5_multi_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpí
?transformer_block_5/multi_head_attention_5/attention_output/addAddV2Rtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum:output:0Vtransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?transformer_block_5/multi_head_attention_5/attention_output/add¡
,transformer_block_5/dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2.
,transformer_block_5/dropout_14/dropout/Const
*transformer_block_5/dropout_14/dropout/MulMulCtransformer_block_5/multi_head_attention_5/attention_output/add:z:05transformer_block_5/dropout_14/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2,
*transformer_block_5/dropout_14/dropout/MulÏ
,transformer_block_5/dropout_14/dropout/ShapeShapeCtransformer_block_5/multi_head_attention_5/attention_output/add:z:0*
T0*
_output_shapes
:2.
,transformer_block_5/dropout_14/dropout/Shape
Ctransformer_block_5/dropout_14/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_5/dropout_14/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype02E
Ctransformer_block_5/dropout_14/dropout/random_uniform/RandomUniform³
5transformer_block_5/dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=27
5transformer_block_5/dropout_14/dropout/GreaterEqual/y¾
3transformer_block_5/dropout_14/dropout/GreaterEqualGreaterEqualLtransformer_block_5/dropout_14/dropout/random_uniform/RandomUniform:output:0>transformer_block_5/dropout_14/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 25
3transformer_block_5/dropout_14/dropout/GreaterEqualà
+transformer_block_5/dropout_14/dropout/CastCast7transformer_block_5/dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+transformer_block_5/dropout_14/dropout/Castú
,transformer_block_5/dropout_14/dropout/Mul_1Mul.transformer_block_5/dropout_14/dropout/Mul:z:0/transformer_block_5/dropout_14/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2.
,transformer_block_5/dropout_14/dropout/Mul_1²
transformer_block_5/addAddV2add_2/add:z:00transformer_block_5/dropout_14/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_5/addà
Itransformer_block_5/layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_5/layer_normalization_10/moments/mean/reduction_indices²
7transformer_block_5/layer_normalization_10/moments/meanMeantransformer_block_5/add:z:0Rtransformer_block_5/layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(29
7transformer_block_5/layer_normalization_10/moments/mean
?transformer_block_5/layer_normalization_10/moments/StopGradientStopGradient@transformer_block_5/layer_normalization_10/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2A
?transformer_block_5/layer_normalization_10/moments/StopGradient¾
Dtransformer_block_5/layer_normalization_10/moments/SquaredDifferenceSquaredDifferencetransformer_block_5/add:z:0Htransformer_block_5/layer_normalization_10/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2F
Dtransformer_block_5/layer_normalization_10/moments/SquaredDifferenceè
Mtransformer_block_5/layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_5/layer_normalization_10/moments/variance/reduction_indicesë
;transformer_block_5/layer_normalization_10/moments/varianceMeanHtransformer_block_5/layer_normalization_10/moments/SquaredDifference:z:0Vtransformer_block_5/layer_normalization_10/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2=
;transformer_block_5/layer_normalization_10/moments/variance½
:transformer_block_5/layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_5/layer_normalization_10/batchnorm/add/y¾
8transformer_block_5/layer_normalization_10/batchnorm/addAddV2Dtransformer_block_5/layer_normalization_10/moments/variance:output:0Ctransformer_block_5/layer_normalization_10/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2:
8transformer_block_5/layer_normalization_10/batchnorm/addõ
:transformer_block_5/layer_normalization_10/batchnorm/RsqrtRsqrt<transformer_block_5/layer_normalization_10/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2<
:transformer_block_5/layer_normalization_10/batchnorm/Rsqrt
Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_5_layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpÂ
8transformer_block_5/layer_normalization_10/batchnorm/mulMul>transformer_block_5/layer_normalization_10/batchnorm/Rsqrt:y:0Otransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2:
8transformer_block_5/layer_normalization_10/batchnorm/mul
:transformer_block_5/layer_normalization_10/batchnorm/mul_1Multransformer_block_5/add:z:0<transformer_block_5/layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_5/layer_normalization_10/batchnorm/mul_1µ
:transformer_block_5/layer_normalization_10/batchnorm/mul_2Mul@transformer_block_5/layer_normalization_10/moments/mean:output:0<transformer_block_5/layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_5/layer_normalization_10/batchnorm/mul_2
Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_5_layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp¾
8transformer_block_5/layer_normalization_10/batchnorm/subSubKtransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp:value:0>transformer_block_5/layer_normalization_10/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2:
8transformer_block_5/layer_normalization_10/batchnorm/subµ
:transformer_block_5/layer_normalization_10/batchnorm/add_1AddV2>transformer_block_5/layer_normalization_10/batchnorm/mul_1:z:0<transformer_block_5/layer_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_5/layer_normalization_10/batchnorm/add_1
Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_5_sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02D
Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp¾
8transformer_block_5/sequential_5/dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_5/sequential_5/dense_16/Tensordot/axesÅ
8transformer_block_5/sequential_5/dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_5/sequential_5/dense_16/Tensordot/freeä
9transformer_block_5/sequential_5/dense_16/Tensordot/ShapeShape>transformer_block_5/layer_normalization_10/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_5/sequential_5/dense_16/Tensordot/ShapeÈ
Atransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2/axis£
<transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2GatherV2Btransformer_block_5/sequential_5/dense_16/Tensordot/Shape:output:0Atransformer_block_5/sequential_5/dense_16/Tensordot/free:output:0Jtransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2Ì
Ctransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1/axis©
>transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1GatherV2Btransformer_block_5/sequential_5/dense_16/Tensordot/Shape:output:0Atransformer_block_5/sequential_5/dense_16/Tensordot/axes:output:0Ltransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1À
9transformer_block_5/sequential_5/dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_5/sequential_5/dense_16/Tensordot/Const¨
8transformer_block_5/sequential_5/dense_16/Tensordot/ProdProdEtransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2:output:0Btransformer_block_5/sequential_5/dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_5/sequential_5/dense_16/Tensordot/ProdÄ
;transformer_block_5/sequential_5/dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_5/sequential_5/dense_16/Tensordot/Const_1°
:transformer_block_5/sequential_5/dense_16/Tensordot/Prod_1ProdGtransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1:output:0Dtransformer_block_5/sequential_5/dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_5/sequential_5/dense_16/Tensordot/Prod_1Ä
?transformer_block_5/sequential_5/dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_5/sequential_5/dense_16/Tensordot/concat/axis
:transformer_block_5/sequential_5/dense_16/Tensordot/concatConcatV2Atransformer_block_5/sequential_5/dense_16/Tensordot/free:output:0Atransformer_block_5/sequential_5/dense_16/Tensordot/axes:output:0Htransformer_block_5/sequential_5/dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_5/sequential_5/dense_16/Tensordot/concat´
9transformer_block_5/sequential_5/dense_16/Tensordot/stackPackAtransformer_block_5/sequential_5/dense_16/Tensordot/Prod:output:0Ctransformer_block_5/sequential_5/dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_5/sequential_5/dense_16/Tensordot/stackÆ
=transformer_block_5/sequential_5/dense_16/Tensordot/transpose	Transpose>transformer_block_5/layer_normalization_10/batchnorm/add_1:z:0Ctransformer_block_5/sequential_5/dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2?
=transformer_block_5/sequential_5/dense_16/Tensordot/transposeÇ
;transformer_block_5/sequential_5/dense_16/Tensordot/ReshapeReshapeAtransformer_block_5/sequential_5/dense_16/Tensordot/transpose:y:0Btransformer_block_5/sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2=
;transformer_block_5/sequential_5/dense_16/Tensordot/ReshapeÆ
:transformer_block_5/sequential_5/dense_16/Tensordot/MatMulMatMulDtransformer_block_5/sequential_5/dense_16/Tensordot/Reshape:output:0Jtransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2<
:transformer_block_5/sequential_5/dense_16/Tensordot/MatMulÄ
;transformer_block_5/sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2=
;transformer_block_5/sequential_5/dense_16/Tensordot/Const_2È
Atransformer_block_5/sequential_5/dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_5/sequential_5/dense_16/Tensordot/concat_1/axis
<transformer_block_5/sequential_5/dense_16/Tensordot/concat_1ConcatV2Etransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2:output:0Dtransformer_block_5/sequential_5/dense_16/Tensordot/Const_2:output:0Jtransformer_block_5/sequential_5/dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_5/sequential_5/dense_16/Tensordot/concat_1¸
3transformer_block_5/sequential_5/dense_16/TensordotReshapeDtransformer_block_5/sequential_5/dense_16/Tensordot/MatMul:product:0Etransformer_block_5/sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@25
3transformer_block_5/sequential_5/dense_16/Tensordot
@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_5_sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp¯
1transformer_block_5/sequential_5/dense_16/BiasAddBiasAdd<transformer_block_5/sequential_5/dense_16/Tensordot:output:0Htransformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@23
1transformer_block_5/sequential_5/dense_16/BiasAddÚ
.transformer_block_5/sequential_5/dense_16/ReluRelu:transformer_block_5/sequential_5/dense_16/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@20
.transformer_block_5/sequential_5/dense_16/Relu
Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_5_sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02D
Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp¾
8transformer_block_5/sequential_5/dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_5/sequential_5/dense_17/Tensordot/axesÅ
8transformer_block_5/sequential_5/dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_5/sequential_5/dense_17/Tensordot/freeâ
9transformer_block_5/sequential_5/dense_17/Tensordot/ShapeShape<transformer_block_5/sequential_5/dense_16/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_5/sequential_5/dense_17/Tensordot/ShapeÈ
Atransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2/axis£
<transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2GatherV2Btransformer_block_5/sequential_5/dense_17/Tensordot/Shape:output:0Atransformer_block_5/sequential_5/dense_17/Tensordot/free:output:0Jtransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2Ì
Ctransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1/axis©
>transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1GatherV2Btransformer_block_5/sequential_5/dense_17/Tensordot/Shape:output:0Atransformer_block_5/sequential_5/dense_17/Tensordot/axes:output:0Ltransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1À
9transformer_block_5/sequential_5/dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_5/sequential_5/dense_17/Tensordot/Const¨
8transformer_block_5/sequential_5/dense_17/Tensordot/ProdProdEtransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2:output:0Btransformer_block_5/sequential_5/dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_5/sequential_5/dense_17/Tensordot/ProdÄ
;transformer_block_5/sequential_5/dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_5/sequential_5/dense_17/Tensordot/Const_1°
:transformer_block_5/sequential_5/dense_17/Tensordot/Prod_1ProdGtransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1:output:0Dtransformer_block_5/sequential_5/dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_5/sequential_5/dense_17/Tensordot/Prod_1Ä
?transformer_block_5/sequential_5/dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_5/sequential_5/dense_17/Tensordot/concat/axis
:transformer_block_5/sequential_5/dense_17/Tensordot/concatConcatV2Atransformer_block_5/sequential_5/dense_17/Tensordot/free:output:0Atransformer_block_5/sequential_5/dense_17/Tensordot/axes:output:0Htransformer_block_5/sequential_5/dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_5/sequential_5/dense_17/Tensordot/concat´
9transformer_block_5/sequential_5/dense_17/Tensordot/stackPackAtransformer_block_5/sequential_5/dense_17/Tensordot/Prod:output:0Ctransformer_block_5/sequential_5/dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_5/sequential_5/dense_17/Tensordot/stackÄ
=transformer_block_5/sequential_5/dense_17/Tensordot/transpose	Transpose<transformer_block_5/sequential_5/dense_16/Relu:activations:0Ctransformer_block_5/sequential_5/dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2?
=transformer_block_5/sequential_5/dense_17/Tensordot/transposeÇ
;transformer_block_5/sequential_5/dense_17/Tensordot/ReshapeReshapeAtransformer_block_5/sequential_5/dense_17/Tensordot/transpose:y:0Btransformer_block_5/sequential_5/dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2=
;transformer_block_5/sequential_5/dense_17/Tensordot/ReshapeÆ
:transformer_block_5/sequential_5/dense_17/Tensordot/MatMulMatMulDtransformer_block_5/sequential_5/dense_17/Tensordot/Reshape:output:0Jtransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2<
:transformer_block_5/sequential_5/dense_17/Tensordot/MatMulÄ
;transformer_block_5/sequential_5/dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_5/sequential_5/dense_17/Tensordot/Const_2È
Atransformer_block_5/sequential_5/dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_5/sequential_5/dense_17/Tensordot/concat_1/axis
<transformer_block_5/sequential_5/dense_17/Tensordot/concat_1ConcatV2Etransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2:output:0Dtransformer_block_5/sequential_5/dense_17/Tensordot/Const_2:output:0Jtransformer_block_5/sequential_5/dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_5/sequential_5/dense_17/Tensordot/concat_1¸
3transformer_block_5/sequential_5/dense_17/TensordotReshapeDtransformer_block_5/sequential_5/dense_17/Tensordot/MatMul:product:0Etransformer_block_5/sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 25
3transformer_block_5/sequential_5/dense_17/Tensordot
@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_5_sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp¯
1transformer_block_5/sequential_5/dense_17/BiasAddBiasAdd<transformer_block_5/sequential_5/dense_17/Tensordot:output:0Htransformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 23
1transformer_block_5/sequential_5/dense_17/BiasAdd¡
,transformer_block_5/dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2.
,transformer_block_5/dropout_15/dropout/Const
*transformer_block_5/dropout_15/dropout/MulMul:transformer_block_5/sequential_5/dense_17/BiasAdd:output:05transformer_block_5/dropout_15/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2,
*transformer_block_5/dropout_15/dropout/MulÆ
,transformer_block_5/dropout_15/dropout/ShapeShape:transformer_block_5/sequential_5/dense_17/BiasAdd:output:0*
T0*
_output_shapes
:2.
,transformer_block_5/dropout_15/dropout/Shape
Ctransformer_block_5/dropout_15/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_5/dropout_15/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype02E
Ctransformer_block_5/dropout_15/dropout/random_uniform/RandomUniform³
5transformer_block_5/dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=27
5transformer_block_5/dropout_15/dropout/GreaterEqual/y¾
3transformer_block_5/dropout_15/dropout/GreaterEqualGreaterEqualLtransformer_block_5/dropout_15/dropout/random_uniform/RandomUniform:output:0>transformer_block_5/dropout_15/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 25
3transformer_block_5/dropout_15/dropout/GreaterEqualà
+transformer_block_5/dropout_15/dropout/CastCast7transformer_block_5/dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+transformer_block_5/dropout_15/dropout/Castú
,transformer_block_5/dropout_15/dropout/Mul_1Mul.transformer_block_5/dropout_15/dropout/Mul:z:0/transformer_block_5/dropout_15/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2.
,transformer_block_5/dropout_15/dropout/Mul_1ç
transformer_block_5/add_1AddV2>transformer_block_5/layer_normalization_10/batchnorm/add_1:z:00transformer_block_5/dropout_15/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_5/add_1à
Itransformer_block_5/layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_5/layer_normalization_11/moments/mean/reduction_indices´
7transformer_block_5/layer_normalization_11/moments/meanMeantransformer_block_5/add_1:z:0Rtransformer_block_5/layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(29
7transformer_block_5/layer_normalization_11/moments/mean
?transformer_block_5/layer_normalization_11/moments/StopGradientStopGradient@transformer_block_5/layer_normalization_11/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2A
?transformer_block_5/layer_normalization_11/moments/StopGradientÀ
Dtransformer_block_5/layer_normalization_11/moments/SquaredDifferenceSquaredDifferencetransformer_block_5/add_1:z:0Htransformer_block_5/layer_normalization_11/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2F
Dtransformer_block_5/layer_normalization_11/moments/SquaredDifferenceè
Mtransformer_block_5/layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_5/layer_normalization_11/moments/variance/reduction_indicesë
;transformer_block_5/layer_normalization_11/moments/varianceMeanHtransformer_block_5/layer_normalization_11/moments/SquaredDifference:z:0Vtransformer_block_5/layer_normalization_11/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2=
;transformer_block_5/layer_normalization_11/moments/variance½
:transformer_block_5/layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_5/layer_normalization_11/batchnorm/add/y¾
8transformer_block_5/layer_normalization_11/batchnorm/addAddV2Dtransformer_block_5/layer_normalization_11/moments/variance:output:0Ctransformer_block_5/layer_normalization_11/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2:
8transformer_block_5/layer_normalization_11/batchnorm/addõ
:transformer_block_5/layer_normalization_11/batchnorm/RsqrtRsqrt<transformer_block_5/layer_normalization_11/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2<
:transformer_block_5/layer_normalization_11/batchnorm/Rsqrt
Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_5_layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpÂ
8transformer_block_5/layer_normalization_11/batchnorm/mulMul>transformer_block_5/layer_normalization_11/batchnorm/Rsqrt:y:0Otransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2:
8transformer_block_5/layer_normalization_11/batchnorm/mul
:transformer_block_5/layer_normalization_11/batchnorm/mul_1Multransformer_block_5/add_1:z:0<transformer_block_5/layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_5/layer_normalization_11/batchnorm/mul_1µ
:transformer_block_5/layer_normalization_11/batchnorm/mul_2Mul@transformer_block_5/layer_normalization_11/moments/mean:output:0<transformer_block_5/layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_5/layer_normalization_11/batchnorm/mul_2
Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_5_layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp¾
8transformer_block_5/layer_normalization_11/batchnorm/subSubKtransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp:value:0>transformer_block_5/layer_normalization_11/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2:
8transformer_block_5/layer_normalization_11/batchnorm/subµ
:transformer_block_5/layer_normalization_11/batchnorm/add_1AddV2>transformer_block_5/layer_normalization_11/batchnorm/mul_1:z:0<transformer_block_5/layer_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_5/layer_normalization_11/batchnorm/add_1s
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`  2
flatten_2/Const¾
flatten_2/ReshapeReshape>transformer_block_5/layer_normalization_11/batchnorm/add_1:z:0flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
flatten_2/Reshape¶
4batch_normalization_8/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_8/moments/mean/reduction_indicesÓ
"batch_normalization_8/moments/meanMeaninputs_1=batch_normalization_8/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_8/moments/mean¾
*batch_normalization_8/moments/StopGradientStopGradient+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_8/moments/StopGradientè
/batch_normalization_8/moments/SquaredDifferenceSquaredDifferenceinputs_13batch_normalization_8/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/batch_normalization_8/moments/SquaredDifference¾
8batch_normalization_8/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_8/moments/variance/reduction_indices
&batch_normalization_8/moments/varianceMean3batch_normalization_8/moments/SquaredDifference:z:0Abatch_normalization_8/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_8/moments/varianceÂ
%batch_normalization_8/moments/SqueezeSqueeze+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_8/moments/SqueezeÊ
'batch_normalization_8/moments/Squeeze_1Squeeze/batch_normalization_8/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_8/moments/Squeeze_1
+batch_normalization_8/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_8/AssignMovingAvg/423362*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_8/AssignMovingAvg/decayÕ
4batch_normalization_8/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_8_assignmovingavg_423362*
_output_shapes
:*
dtype026
4batch_normalization_8/AssignMovingAvg/ReadVariableOpß
)batch_normalization_8/AssignMovingAvg/subSub<batch_normalization_8/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_8/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_8/AssignMovingAvg/423362*
_output_shapes
:2+
)batch_normalization_8/AssignMovingAvg/subÖ
)batch_normalization_8/AssignMovingAvg/mulMul-batch_normalization_8/AssignMovingAvg/sub:z:04batch_normalization_8/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_8/AssignMovingAvg/423362*
_output_shapes
:2+
)batch_normalization_8/AssignMovingAvg/mul³
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_8_assignmovingavg_423362-batch_normalization_8/AssignMovingAvg/mul:z:05^batch_normalization_8/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_8/AssignMovingAvg/423362*
_output_shapes
 *
dtype02;
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_8/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg_1/423368*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_8/AssignMovingAvg_1/decayÛ
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_8_assignmovingavg_1_423368*
_output_shapes
:*
dtype028
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpé
+batch_normalization_8/AssignMovingAvg_1/subSub>batch_normalization_8/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_8/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg_1/423368*
_output_shapes
:2-
+batch_normalization_8/AssignMovingAvg_1/subà
+batch_normalization_8/AssignMovingAvg_1/mulMul/batch_normalization_8/AssignMovingAvg_1/sub:z:06batch_normalization_8/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg_1/423368*
_output_shapes
:2-
+batch_normalization_8/AssignMovingAvg_1/mul¿
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_8_assignmovingavg_1_423368/batch_normalization_8/AssignMovingAvg_1/mul:z:07^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg_1/423368*
_output_shapes
 *
dtype02=
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_8/batchnorm/add/yÚ
#batch_normalization_8/batchnorm/addAddV20batch_normalization_8/moments/Squeeze_1:output:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/add¥
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/Rsqrtà
2batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_8/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:0:batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/mulº
%batch_normalization_8/batchnorm/mul_1Mulinputs_1'batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_8/batchnorm/mul_1Ó
%batch_normalization_8/batchnorm/mul_2Mul.batch_normalization_8/moments/Squeeze:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/mul_2Ô
.batch_normalization_8/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_8/batchnorm/ReadVariableOpÙ
#batch_normalization_8/batchnorm/subSub6batch_normalization_8/batchnorm/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/subÝ
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_8/batchnorm/add_1x
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axisß
concatenate_2/concatConcatV2flatten_2/Reshape:output:0)batch_normalization_8/batchnorm/add_1:z:0"concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
concatenate_2/concat©
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes
:	è@*
dtype02 
dense_18/MatMul/ReadVariableOp¥
dense_18/MatMulMatMulconcatenate_2/concat:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_18/MatMul§
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_18/BiasAdd/ReadVariableOp¥
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_18/BiasAdds
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_18/Reluy
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_16/dropout/Const©
dropout_16/dropout/MulMuldense_18/Relu:activations:0!dropout_16/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_16/dropout/Mul
dropout_16/dropout/ShapeShapedense_18/Relu:activations:0*
T0*
_output_shapes
:2
dropout_16/dropout/ShapeÕ
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype021
/dropout_16/dropout/random_uniform/RandomUniform
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_16/dropout/GreaterEqual/yê
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
dropout_16/dropout/GreaterEqual 
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_16/dropout/Cast¦
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_16/dropout/Mul_1¨
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_19/MatMul/ReadVariableOp¤
dense_19/MatMulMatMuldropout_16/dropout/Mul_1:z:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_19/MatMul§
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_19/BiasAdd/ReadVariableOp¥
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_19/BiasAdds
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_19/Reluy
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_17/dropout/Const©
dropout_17/dropout/MulMuldense_19/Relu:activations:0!dropout_17/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_17/dropout/Mul
dropout_17/dropout/ShapeShapedense_19/Relu:activations:0*
T0*
_output_shapes
:2
dropout_17/dropout/ShapeÕ
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype021
/dropout_17/dropout/random_uniform/RandomUniform
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_17/dropout/GreaterEqual/yê
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
dropout_17/dropout/GreaterEqual 
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_17/dropout/Cast¦
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_17/dropout/Mul_1¨
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_20/MatMul/ReadVariableOp¤
dense_20/MatMulMatMuldropout_17/dropout/Mul_1:z:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_20/MatMul§
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp¥
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_20/BiasAddÜ
IdentityIdentitydense_20/BiasAdd:output:0:^batch_normalization_6/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_6/AssignMovingAvg/ReadVariableOp<^batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_6/batchnorm/ReadVariableOp3^batch_normalization_6/batchnorm/mul/ReadVariableOp:^batch_normalization_7/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_7/AssignMovingAvg/ReadVariableOp<^batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_7/batchnorm/ReadVariableOp3^batch_normalization_7/batchnorm/mul/ReadVariableOp:^batch_normalization_8/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_8/AssignMovingAvg/ReadVariableOp<^batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_8/batchnorm/ReadVariableOp3^batch_normalization_8/batchnorm/mul/ReadVariableOp ^conv1d_4/BiasAdd/ReadVariableOp,^conv1d_4/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_5/BiasAdd/ReadVariableOp,^conv1d_5/conv1d/ExpandDims_1/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp<^token_and_position_embedding_2/embedding_4/embedding_lookup<^token_and_position_embedding_2/embedding_5/embedding_lookupD^transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpH^transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpD^transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpH^transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpO^transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpY^transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_5/multi_head_attention_5/key/add/ReadVariableOpL^transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpD^transformer_block_5/multi_head_attention_5/query/add/ReadVariableOpN^transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpD^transformer_block_5/multi_head_attention_5/value/add/ReadVariableOpN^transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpA^transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpC^transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpA^transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpC^transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::2v
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_6/AssignMovingAvg/ReadVariableOp4batch_normalization_6/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_6/batchnorm/ReadVariableOp.batch_normalization_6/batchnorm/ReadVariableOp2h
2batch_normalization_6/batchnorm/mul/ReadVariableOp2batch_normalization_6/batchnorm/mul/ReadVariableOp2v
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_7/AssignMovingAvg/ReadVariableOp4batch_normalization_7/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_7/batchnorm/ReadVariableOp.batch_normalization_7/batchnorm/ReadVariableOp2h
2batch_normalization_7/batchnorm/mul/ReadVariableOp2batch_normalization_7/batchnorm/mul/ReadVariableOp2v
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_8/AssignMovingAvg/ReadVariableOp4batch_normalization_8/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_8/batchnorm/ReadVariableOp.batch_normalization_8/batchnorm/ReadVariableOp2h
2batch_normalization_8/batchnorm/mul/ReadVariableOp2batch_normalization_8/batchnorm/mul/ReadVariableOp2B
conv1d_4/BiasAdd/ReadVariableOpconv1d_4/BiasAdd/ReadVariableOp2Z
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_5/BiasAdd/ReadVariableOpconv1d_5/BiasAdd/ReadVariableOp2Z
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2z
;token_and_position_embedding_2/embedding_4/embedding_lookup;token_and_position_embedding_2/embedding_4/embedding_lookup2z
;token_and_position_embedding_2/embedding_5/embedding_lookup;token_and_position_embedding_2/embedding_5/embedding_lookup2
Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpCtransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp2
Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpGtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp2
Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpCtransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp2
Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpGtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp2 
Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpNtransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp2´
Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp2
Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOpAtransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp2
Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpKtransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp2
Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOpCtransformer_block_5/multi_head_attention_5/query/add/ReadVariableOp2
Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpMtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp2
Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOpCtransformer_block_5/multi_head_attention_5/value/add/ReadVariableOp2
Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpMtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp2
@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp2
Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpBtransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp2
@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp2
Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOpBtransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp:R N
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
¿
m
A__inference_add_2_layer_call_and_return_conditional_losses_424273
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

e
F__inference_dropout_16_layer_call_and_return_conditional_losses_424766

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
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
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
É
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_424771

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
é

H__inference_sequential_5_layer_call_and_return_conditional_losses_421457

inputs
dense_16_421446
dense_16_421448
dense_17_421451
dense_17_421453
identity¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall
 dense_16/StatefulPartitionedCallStatefulPartitionedCallinputsdense_16_421446dense_16_421448*
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
D__inference_dense_16_layer_call_and_return_conditional_losses_4213362"
 dense_16/StatefulPartitionedCall¾
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_421451dense_17_421453*
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
D__inference_dense_17_layer_call_and_return_conditional_losses_4213822"
 dense_17/StatefulPartitionedCallÇ
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
c
 
C__inference_model_2_layer_call_and_return_conditional_losses_422717

inputs
inputs_1)
%token_and_position_embedding_2_422618)
%token_and_position_embedding_2_422620
conv1d_4_422623
conv1d_4_422625
conv1d_5_422629
conv1d_5_422631 
batch_normalization_6_422636 
batch_normalization_6_422638 
batch_normalization_6_422640 
batch_normalization_6_422642 
batch_normalization_7_422645 
batch_normalization_7_422647 
batch_normalization_7_422649 
batch_normalization_7_422651
transformer_block_5_422655
transformer_block_5_422657
transformer_block_5_422659
transformer_block_5_422661
transformer_block_5_422663
transformer_block_5_422665
transformer_block_5_422667
transformer_block_5_422669
transformer_block_5_422671
transformer_block_5_422673
transformer_block_5_422675
transformer_block_5_422677
transformer_block_5_422679
transformer_block_5_422681
transformer_block_5_422683
transformer_block_5_422685 
batch_normalization_8_422689 
batch_normalization_8_422691 
batch_normalization_8_422693 
batch_normalization_8_422695
dense_18_422699
dense_18_422701
dense_19_422705
dense_19_422707
dense_20_422711
dense_20_422713
identity¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢ conv1d_4/StatefulPartitionedCall¢ conv1d_5/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢ dense_20/StatefulPartitionedCall¢"dropout_16/StatefulPartitionedCall¢"dropout_17/StatefulPartitionedCall¢6token_and_position_embedding_2/StatefulPartitionedCall¢+transformer_block_5/StatefulPartitionedCall
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_2_422618%token_and_position_embedding_2_422620*
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
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_42163728
6token_and_position_embedding_2/StatefulPartitionedCallÕ
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0conv1d_4_422623conv1d_4_422625*
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
GPU2*0J 8 *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_4216692"
 conv1d_4/StatefulPartitionedCall 
#average_pooling1d_6/PartitionedCallPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_6_layer_call_and_return_conditional_losses_4209852%
#average_pooling1d_6/PartitionedCallÂ
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_6/PartitionedCall:output:0conv1d_5_422629conv1d_5_422631*
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
D__inference_conv1d_5_layer_call_and_return_conditional_losses_4217022"
 conv1d_5/StatefulPartitionedCallµ
#average_pooling1d_8/PartitionedCallPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0*
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
O__inference_average_pooling1d_8_layer_call_and_return_conditional_losses_4210152%
#average_pooling1d_8/PartitionedCall
#average_pooling1d_7/PartitionedCallPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0*
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
O__inference_average_pooling1d_7_layer_call_and_return_conditional_losses_4210002%
#average_pooling1d_7/PartitionedCallÀ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_7/PartitionedCall:output:0batch_normalization_6_422636batch_normalization_6_422638batch_normalization_6_422640batch_normalization_6_422642*
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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4217552/
-batch_normalization_6/StatefulPartitionedCallÀ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_8/PartitionedCall:output:0batch_normalization_7_422645batch_normalization_7_422647batch_normalization_7_422649batch_normalization_7_422651*
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4218462/
-batch_normalization_7/StatefulPartitionedCall»
add_2/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
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
A__inference_add_2_layer_call_and_return_conditional_losses_4219082
add_2/PartitionedCall
+transformer_block_5/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0transformer_block_5_422655transformer_block_5_422657transformer_block_5_422659transformer_block_5_422661transformer_block_5_422663transformer_block_5_422665transformer_block_5_422667transformer_block_5_422669transformer_block_5_422671transformer_block_5_422673transformer_block_5_422675transformer_block_5_422677transformer_block_5_422679transformer_block_5_422681transformer_block_5_422683transformer_block_5_422685*
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
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_4220652-
+transformer_block_5/StatefulPartitionedCall
flatten_2/PartitionedCallPartitionedCall4transformer_block_5/StatefulPartitionedCall:output:0*
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
E__inference_flatten_2_layer_call_and_return_conditional_losses_4223072
flatten_2/PartitionedCall
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCallinputs_1batch_normalization_8_422689batch_normalization_8_422691batch_normalization_8_422693batch_normalization_8_422695*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4215642/
-batch_normalization_8/StatefulPartitionedCall¼
concatenate_2/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:06batch_normalization_8/StatefulPartitionedCall:output:0*
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
I__inference_concatenate_2_layer_call_and_return_conditional_losses_4223572
concatenate_2/PartitionedCall·
 dense_18/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_18_422699dense_18_422701*
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
D__inference_dense_18_layer_call_and_return_conditional_losses_4223772"
 dense_18/StatefulPartitionedCall
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
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
F__inference_dropout_16_layer_call_and_return_conditional_losses_4224052$
"dropout_16/StatefulPartitionedCall¼
 dense_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0dense_19_422705dense_19_422707*
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
D__inference_dense_19_layer_call_and_return_conditional_losses_4224342"
 dense_19/StatefulPartitionedCall½
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
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
F__inference_dropout_17_layer_call_and_return_conditional_losses_4224622$
"dropout_17/StatefulPartitionedCall¼
 dense_20/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_20_422711dense_20_422713*
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
D__inference_dense_20_layer_call_and_return_conditional_losses_4224902"
 dense_20/StatefulPartitionedCallí
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_5/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_5/StatefulPartitionedCall+transformer_block_5/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_sequential_5_layer_call_and_return_conditional_losses_421399
dense_16_input
dense_16_421347
dense_16_421349
dense_17_421393
dense_17_421395
identity¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall£
 dense_16/StatefulPartitionedCallStatefulPartitionedCalldense_16_inputdense_16_421347dense_16_421349*
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
D__inference_dense_16_layer_call_and_return_conditional_losses_4213362"
 dense_16/StatefulPartitionedCall¾
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_421393dense_17_421395*
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
D__inference_dense_17_layer_call_and_return_conditional_losses_4213822"
 dense_17/StatefulPartitionedCallÇ
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
(
_user_specified_namedense_16_input
õ
k
O__inference_average_pooling1d_6_layer_call_and_return_conditional_losses_420985

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
É
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_422410

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
ó0
È
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_421117

inputs
assignmovingavg_421092
assignmovingavg_1_421098)
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
loc:@AssignMovingAvg/421092*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_421092*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/421092*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/421092*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_421092AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/421092*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/421098*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_421098*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/421098*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/421098*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_421098AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/421098*
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_424159

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
É
d
F__inference_dropout_17_layer_call_and_return_conditional_losses_424818

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

÷
D__inference_conv1d_4_layer_call_and_return_conditional_losses_423905

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
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
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
:ÿÿÿÿÿÿÿÿÿR 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2

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


H__inference_sequential_5_layer_call_and_return_conditional_losses_421413
dense_16_input
dense_16_421402
dense_16_421404
dense_17_421407
dense_17_421409
identity¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall£
 dense_16/StatefulPartitionedCallStatefulPartitionedCalldense_16_inputdense_16_421402dense_16_421404*
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
D__inference_dense_16_layer_call_and_return_conditional_losses_4213362"
 dense_16/StatefulPartitionedCall¾
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_421407dense_17_421409*
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
D__inference_dense_17_layer_call_and_return_conditional_losses_4213822"
 dense_17/StatefulPartitionedCallÇ
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
(
_user_specified_namedense_16_input
Ü
â
(__inference_model_2_layer_call_fn_422989
input_5
input_6
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

unknown_36

unknown_37

unknown_38
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_36
unknown_37
unknown_38*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_4229062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_5:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6
¼0
È
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_421755

inputs
assignmovingavg_421730
assignmovingavg_1_421736)
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
loc:@AssignMovingAvg/421730*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_421730*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/421730*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/421730*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_421730AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/421730*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/421736*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_421736*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/421736*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/421736*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_421736AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/421736*
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
±
ð,
!__inference__wrapped_model_420976
input_5
input_6N
Jmodel_2_token_and_position_embedding_2_embedding_5_embedding_lookup_420729N
Jmodel_2_token_and_position_embedding_2_embedding_4_embedding_lookup_420735@
<model_2_conv1d_4_conv1d_expanddims_1_readvariableop_resource4
0model_2_conv1d_4_biasadd_readvariableop_resource@
<model_2_conv1d_5_conv1d_expanddims_1_readvariableop_resource4
0model_2_conv1d_5_biasadd_readvariableop_resourceC
?model_2_batch_normalization_6_batchnorm_readvariableop_resourceG
Cmodel_2_batch_normalization_6_batchnorm_mul_readvariableop_resourceE
Amodel_2_batch_normalization_6_batchnorm_readvariableop_1_resourceE
Amodel_2_batch_normalization_6_batchnorm_readvariableop_2_resourceC
?model_2_batch_normalization_7_batchnorm_readvariableop_resourceG
Cmodel_2_batch_normalization_7_batchnorm_mul_readvariableop_resourceE
Amodel_2_batch_normalization_7_batchnorm_readvariableop_1_resourceE
Amodel_2_batch_normalization_7_batchnorm_readvariableop_2_resourceb
^model_2_transformer_block_5_multi_head_attention_5_query_einsum_einsum_readvariableop_resourceX
Tmodel_2_transformer_block_5_multi_head_attention_5_query_add_readvariableop_resource`
\model_2_transformer_block_5_multi_head_attention_5_key_einsum_einsum_readvariableop_resourceV
Rmodel_2_transformer_block_5_multi_head_attention_5_key_add_readvariableop_resourceb
^model_2_transformer_block_5_multi_head_attention_5_value_einsum_einsum_readvariableop_resourceX
Tmodel_2_transformer_block_5_multi_head_attention_5_value_add_readvariableop_resourcem
imodel_2_transformer_block_5_multi_head_attention_5_attention_output_einsum_einsum_readvariableop_resourcec
_model_2_transformer_block_5_multi_head_attention_5_attention_output_add_readvariableop_resource\
Xmodel_2_transformer_block_5_layer_normalization_10_batchnorm_mul_readvariableop_resourceX
Tmodel_2_transformer_block_5_layer_normalization_10_batchnorm_readvariableop_resourceW
Smodel_2_transformer_block_5_sequential_5_dense_16_tensordot_readvariableop_resourceU
Qmodel_2_transformer_block_5_sequential_5_dense_16_biasadd_readvariableop_resourceW
Smodel_2_transformer_block_5_sequential_5_dense_17_tensordot_readvariableop_resourceU
Qmodel_2_transformer_block_5_sequential_5_dense_17_biasadd_readvariableop_resource\
Xmodel_2_transformer_block_5_layer_normalization_11_batchnorm_mul_readvariableop_resourceX
Tmodel_2_transformer_block_5_layer_normalization_11_batchnorm_readvariableop_resourceC
?model_2_batch_normalization_8_batchnorm_readvariableop_resourceG
Cmodel_2_batch_normalization_8_batchnorm_mul_readvariableop_resourceE
Amodel_2_batch_normalization_8_batchnorm_readvariableop_1_resourceE
Amodel_2_batch_normalization_8_batchnorm_readvariableop_2_resource3
/model_2_dense_18_matmul_readvariableop_resource4
0model_2_dense_18_biasadd_readvariableop_resource3
/model_2_dense_19_matmul_readvariableop_resource4
0model_2_dense_19_biasadd_readvariableop_resource3
/model_2_dense_20_matmul_readvariableop_resource4
0model_2_dense_20_biasadd_readvariableop_resource
identity¢6model_2/batch_normalization_6/batchnorm/ReadVariableOp¢8model_2/batch_normalization_6/batchnorm/ReadVariableOp_1¢8model_2/batch_normalization_6/batchnorm/ReadVariableOp_2¢:model_2/batch_normalization_6/batchnorm/mul/ReadVariableOp¢6model_2/batch_normalization_7/batchnorm/ReadVariableOp¢8model_2/batch_normalization_7/batchnorm/ReadVariableOp_1¢8model_2/batch_normalization_7/batchnorm/ReadVariableOp_2¢:model_2/batch_normalization_7/batchnorm/mul/ReadVariableOp¢6model_2/batch_normalization_8/batchnorm/ReadVariableOp¢8model_2/batch_normalization_8/batchnorm/ReadVariableOp_1¢8model_2/batch_normalization_8/batchnorm/ReadVariableOp_2¢:model_2/batch_normalization_8/batchnorm/mul/ReadVariableOp¢'model_2/conv1d_4/BiasAdd/ReadVariableOp¢3model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp¢'model_2/conv1d_5/BiasAdd/ReadVariableOp¢3model_2/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp¢'model_2/dense_18/BiasAdd/ReadVariableOp¢&model_2/dense_18/MatMul/ReadVariableOp¢'model_2/dense_19/BiasAdd/ReadVariableOp¢&model_2/dense_19/MatMul/ReadVariableOp¢'model_2/dense_20/BiasAdd/ReadVariableOp¢&model_2/dense_20/MatMul/ReadVariableOp¢Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup¢Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup¢Kmodel_2/transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp¢Omodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp¢Kmodel_2/transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp¢Omodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp¢Vmodel_2/transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp¢`model_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp¢Imodel_2/transformer_block_5/multi_head_attention_5/key/add/ReadVariableOp¢Smodel_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp¢Kmodel_2/transformer_block_5/multi_head_attention_5/query/add/ReadVariableOp¢Umodel_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp¢Kmodel_2/transformer_block_5/multi_head_attention_5/value/add/ReadVariableOp¢Umodel_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp¢Hmodel_2/transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp¢Jmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp¢Hmodel_2/transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp¢Jmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp
,model_2/token_and_position_embedding_2/ShapeShapeinput_5*
T0*
_output_shapes
:2.
,model_2/token_and_position_embedding_2/ShapeË
:model_2/token_and_position_embedding_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2<
:model_2/token_and_position_embedding_2/strided_slice/stackÆ
<model_2/token_and_position_embedding_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<model_2/token_and_position_embedding_2/strided_slice/stack_1Æ
<model_2/token_and_position_embedding_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model_2/token_and_position_embedding_2/strided_slice/stack_2Ì
4model_2/token_and_position_embedding_2/strided_sliceStridedSlice5model_2/token_and_position_embedding_2/Shape:output:0Cmodel_2/token_and_position_embedding_2/strided_slice/stack:output:0Emodel_2/token_and_position_embedding_2/strided_slice/stack_1:output:0Emodel_2/token_and_position_embedding_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4model_2/token_and_position_embedding_2/strided_sliceª
2model_2/token_and_position_embedding_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : 24
2model_2/token_and_position_embedding_2/range/startª
2model_2/token_and_position_embedding_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :24
2model_2/token_and_position_embedding_2/range/deltaÃ
,model_2/token_and_position_embedding_2/rangeRange;model_2/token_and_position_embedding_2/range/start:output:0=model_2/token_and_position_embedding_2/strided_slice:output:0;model_2/token_and_position_embedding_2/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,model_2/token_and_position_embedding_2/rangeò
Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookupResourceGatherJmodel_2_token_and_position_embedding_2_embedding_5_embedding_lookup_4207295model_2/token_and_position_embedding_2/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*]
_classS
QOloc:@model_2/token_and_position_embedding_2/embedding_5/embedding_lookup/420729*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02E
Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookupµ
Lmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/IdentityIdentityLmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*]
_classS
QOloc:@model_2/token_and_position_embedding_2/embedding_5/embedding_lookup/420729*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2N
Lmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identityµ
Nmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1IdentityUmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2P
Nmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1Å
7model_2/token_and_position_embedding_2/embedding_4/CastCastinput_5*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR29
7model_2/token_and_position_embedding_2/embedding_4/Castý
Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookupResourceGatherJmodel_2_token_and_position_embedding_2_embedding_4_embedding_lookup_420735;model_2/token_and_position_embedding_2/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*]
_classS
QOloc:@model_2/token_and_position_embedding_2/embedding_4/embedding_lookup/420735*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02E
Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookupº
Lmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/IdentityIdentityLmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*]
_classS
QOloc:@model_2/token_and_position_embedding_2/embedding_4/embedding_lookup/420735*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2N
Lmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identityº
Nmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1IdentityUmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2P
Nmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1Ê
*model_2/token_and_position_embedding_2/addAddV2Wmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1:output:0Wmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2,
*model_2/token_and_position_embedding_2/add
&model_2/conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2(
&model_2/conv1d_4/conv1d/ExpandDims/dimò
"model_2/conv1d_4/conv1d/ExpandDims
ExpandDims.model_2/token_and_position_embedding_2/add:z:0/model_2/conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2$
"model_2/conv1d_4/conv1d/ExpandDimsë
3model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<model_2_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype025
3model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp
(model_2/conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_2/conv1d_4/conv1d/ExpandDims_1/dimû
$model_2/conv1d_4/conv1d/ExpandDims_1
ExpandDims;model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:01model_2/conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2&
$model_2/conv1d_4/conv1d/ExpandDims_1û
model_2/conv1d_4/conv1dConv2D+model_2/conv1d_4/conv1d/ExpandDims:output:0-model_2/conv1d_4/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
paddingSAME*
strides
2
model_2/conv1d_4/conv1dÆ
model_2/conv1d_4/conv1d/SqueezeSqueeze model_2/conv1d_4/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2!
model_2/conv1d_4/conv1d/Squeeze¿
'model_2/conv1d_4/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_2/conv1d_4/BiasAdd/ReadVariableOpÑ
model_2/conv1d_4/BiasAddBiasAdd(model_2/conv1d_4/conv1d/Squeeze:output:0/model_2/conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
model_2/conv1d_4/BiasAdd
model_2/conv1d_4/ReluRelu!model_2/conv1d_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
model_2/conv1d_4/Relu
*model_2/average_pooling1d_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_2/average_pooling1d_6/ExpandDims/dimó
&model_2/average_pooling1d_6/ExpandDims
ExpandDims#model_2/conv1d_4/Relu:activations:03model_2/average_pooling1d_6/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2(
&model_2/average_pooling1d_6/ExpandDimsý
#model_2/average_pooling1d_6/AvgPoolAvgPool/model_2/average_pooling1d_6/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
ksize
*
paddingVALID*
strides
2%
#model_2/average_pooling1d_6/AvgPoolÑ
#model_2/average_pooling1d_6/SqueezeSqueeze,model_2/average_pooling1d_6/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims
2%
#model_2/average_pooling1d_6/Squeeze
&model_2/conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2(
&model_2/conv1d_5/conv1d/ExpandDims/dimð
"model_2/conv1d_5/conv1d/ExpandDims
ExpandDims,model_2/average_pooling1d_6/Squeeze:output:0/model_2/conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2$
"model_2/conv1d_5/conv1d/ExpandDimsë
3model_2/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<model_2_conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype025
3model_2/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp
(model_2/conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_2/conv1d_5/conv1d/ExpandDims_1/dimû
$model_2/conv1d_5/conv1d/ExpandDims_1
ExpandDims;model_2/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:01model_2/conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2&
$model_2/conv1d_5/conv1d/ExpandDims_1û
model_2/conv1d_5/conv1dConv2D+model_2/conv1d_5/conv1d/ExpandDims:output:0-model_2/conv1d_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
paddingSAME*
strides
2
model_2/conv1d_5/conv1dÆ
model_2/conv1d_5/conv1d/SqueezeSqueeze model_2/conv1d_5/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2!
model_2/conv1d_5/conv1d/Squeeze¿
'model_2/conv1d_5/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv1d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_2/conv1d_5/BiasAdd/ReadVariableOpÑ
model_2/conv1d_5/BiasAddBiasAdd(model_2/conv1d_5/conv1d/Squeeze:output:0/model_2/conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
model_2/conv1d_5/BiasAdd
model_2/conv1d_5/ReluRelu!model_2/conv1d_5/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
model_2/conv1d_5/Relu
*model_2/average_pooling1d_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_2/average_pooling1d_8/ExpandDims/dimþ
&model_2/average_pooling1d_8/ExpandDims
ExpandDims.model_2/token_and_position_embedding_2/add:z:03model_2/average_pooling1d_8/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2(
&model_2/average_pooling1d_8/ExpandDimsþ
#model_2/average_pooling1d_8/AvgPoolAvgPool/model_2/average_pooling1d_8/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize	
¬*
paddingVALID*
strides	
¬2%
#model_2/average_pooling1d_8/AvgPoolÐ
#model_2/average_pooling1d_8/SqueezeSqueeze,model_2/average_pooling1d_8/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2%
#model_2/average_pooling1d_8/Squeeze
*model_2/average_pooling1d_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_2/average_pooling1d_7/ExpandDims/dimó
&model_2/average_pooling1d_7/ExpandDims
ExpandDims#model_2/conv1d_5/Relu:activations:03model_2/average_pooling1d_7/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2(
&model_2/average_pooling1d_7/ExpandDimsü
#model_2/average_pooling1d_7/AvgPoolAvgPool/model_2/average_pooling1d_7/ExpandDims:output:0*
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
#model_2/average_pooling1d_7/AvgPoolÐ
#model_2/average_pooling1d_7/SqueezeSqueeze,model_2/average_pooling1d_7/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2%
#model_2/average_pooling1d_7/Squeezeì
6model_2/batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp?model_2_batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype028
6model_2/batch_normalization_6/batchnorm/ReadVariableOp£
-model_2/batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-model_2/batch_normalization_6/batchnorm/add/y
+model_2/batch_normalization_6/batchnorm/addAddV2>model_2/batch_normalization_6/batchnorm/ReadVariableOp:value:06model_2/batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2-
+model_2/batch_normalization_6/batchnorm/add½
-model_2/batch_normalization_6/batchnorm/RsqrtRsqrt/model_2/batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-model_2/batch_normalization_6/batchnorm/Rsqrtø
:model_2/batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_2_batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:model_2/batch_normalization_6/batchnorm/mul/ReadVariableOpý
+model_2/batch_normalization_6/batchnorm/mulMul1model_2/batch_normalization_6/batchnorm/Rsqrt:y:0Bmodel_2/batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+model_2/batch_normalization_6/batchnorm/mulú
-model_2/batch_normalization_6/batchnorm/mul_1Mul,model_2/average_pooling1d_7/Squeeze:output:0/model_2/batch_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2/
-model_2/batch_normalization_6/batchnorm/mul_1ò
8model_2/batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_2_batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model_2/batch_normalization_6/batchnorm/ReadVariableOp_1ý
-model_2/batch_normalization_6/batchnorm/mul_2Mul@model_2/batch_normalization_6/batchnorm/ReadVariableOp_1:value:0/model_2/batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-model_2/batch_normalization_6/batchnorm/mul_2ò
8model_2/batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_2_batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8model_2/batch_normalization_6/batchnorm/ReadVariableOp_2û
+model_2/batch_normalization_6/batchnorm/subSub@model_2/batch_normalization_6/batchnorm/ReadVariableOp_2:value:01model_2/batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+model_2/batch_normalization_6/batchnorm/sub
-model_2/batch_normalization_6/batchnorm/add_1AddV21model_2/batch_normalization_6/batchnorm/mul_1:z:0/model_2/batch_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2/
-model_2/batch_normalization_6/batchnorm/add_1ì
6model_2/batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp?model_2_batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype028
6model_2/batch_normalization_7/batchnorm/ReadVariableOp£
-model_2/batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-model_2/batch_normalization_7/batchnorm/add/y
+model_2/batch_normalization_7/batchnorm/addAddV2>model_2/batch_normalization_7/batchnorm/ReadVariableOp:value:06model_2/batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2-
+model_2/batch_normalization_7/batchnorm/add½
-model_2/batch_normalization_7/batchnorm/RsqrtRsqrt/model_2/batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-model_2/batch_normalization_7/batchnorm/Rsqrtø
:model_2/batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_2_batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:model_2/batch_normalization_7/batchnorm/mul/ReadVariableOpý
+model_2/batch_normalization_7/batchnorm/mulMul1model_2/batch_normalization_7/batchnorm/Rsqrt:y:0Bmodel_2/batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+model_2/batch_normalization_7/batchnorm/mulú
-model_2/batch_normalization_7/batchnorm/mul_1Mul,model_2/average_pooling1d_8/Squeeze:output:0/model_2/batch_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2/
-model_2/batch_normalization_7/batchnorm/mul_1ò
8model_2/batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_2_batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model_2/batch_normalization_7/batchnorm/ReadVariableOp_1ý
-model_2/batch_normalization_7/batchnorm/mul_2Mul@model_2/batch_normalization_7/batchnorm/ReadVariableOp_1:value:0/model_2/batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-model_2/batch_normalization_7/batchnorm/mul_2ò
8model_2/batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_2_batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8model_2/batch_normalization_7/batchnorm/ReadVariableOp_2û
+model_2/batch_normalization_7/batchnorm/subSub@model_2/batch_normalization_7/batchnorm/ReadVariableOp_2:value:01model_2/batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+model_2/batch_normalization_7/batchnorm/sub
-model_2/batch_normalization_7/batchnorm/add_1AddV21model_2/batch_normalization_7/batchnorm/mul_1:z:0/model_2/batch_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2/
-model_2/batch_normalization_7/batchnorm/add_1Ë
model_2/add_2/addAddV21model_2/batch_normalization_6/batchnorm/add_1:z:01model_2/batch_normalization_7/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
model_2/add_2/addÑ
Umodel_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOp^model_2_transformer_block_5_multi_head_attention_5_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpð
Fmodel_2/transformer_block_5/multi_head_attention_5/query/einsum/EinsumEinsummodel_2/add_2/add:z:0]model_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2H
Fmodel_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum¯
Kmodel_2/transformer_block_5/multi_head_attention_5/query/add/ReadVariableOpReadVariableOpTmodel_2_transformer_block_5_multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_2/transformer_block_5/multi_head_attention_5/query/add/ReadVariableOpå
<model_2/transformer_block_5/multi_head_attention_5/query/addAddV2Omodel_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum:output:0Smodel_2/transformer_block_5/multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2>
<model_2/transformer_block_5/multi_head_attention_5/query/addË
Smodel_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOp\model_2_transformer_block_5_multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02U
Smodel_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpê
Dmodel_2/transformer_block_5/multi_head_attention_5/key/einsum/EinsumEinsummodel_2/add_2/add:z:0[model_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2F
Dmodel_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum©
Imodel_2/transformer_block_5/multi_head_attention_5/key/add/ReadVariableOpReadVariableOpRmodel_2_transformer_block_5_multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02K
Imodel_2/transformer_block_5/multi_head_attention_5/key/add/ReadVariableOpÝ
:model_2/transformer_block_5/multi_head_attention_5/key/addAddV2Mmodel_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum:output:0Qmodel_2/transformer_block_5/multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:model_2/transformer_block_5/multi_head_attention_5/key/addÑ
Umodel_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOp^model_2_transformer_block_5_multi_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpð
Fmodel_2/transformer_block_5/multi_head_attention_5/value/einsum/EinsumEinsummodel_2/add_2/add:z:0]model_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2H
Fmodel_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum¯
Kmodel_2/transformer_block_5/multi_head_attention_5/value/add/ReadVariableOpReadVariableOpTmodel_2_transformer_block_5_multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_2/transformer_block_5/multi_head_attention_5/value/add/ReadVariableOpå
<model_2/transformer_block_5/multi_head_attention_5/value/addAddV2Omodel_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum:output:0Smodel_2/transformer_block_5/multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2>
<model_2/transformer_block_5/multi_head_attention_5/value/add¹
8model_2/transformer_block_5/multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2:
8model_2/transformer_block_5/multi_head_attention_5/Mul/y¶
6model_2/transformer_block_5/multi_head_attention_5/MulMul@model_2/transformer_block_5/multi_head_attention_5/query/add:z:0Amodel_2/transformer_block_5/multi_head_attention_5/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 28
6model_2/transformer_block_5/multi_head_attention_5/Mulì
@model_2/transformer_block_5/multi_head_attention_5/einsum/EinsumEinsum>model_2/transformer_block_5/multi_head_attention_5/key/add:z:0:model_2/transformer_block_5/multi_head_attention_5/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2B
@model_2/transformer_block_5/multi_head_attention_5/einsum/Einsum
Bmodel_2/transformer_block_5/multi_head_attention_5/softmax/SoftmaxSoftmaxImodel_2/transformer_block_5/multi_head_attention_5/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2D
Bmodel_2/transformer_block_5/multi_head_attention_5/softmax/Softmax
Cmodel_2/transformer_block_5/multi_head_attention_5/dropout/IdentityIdentityLmodel_2/transformer_block_5/multi_head_attention_5/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2E
Cmodel_2/transformer_block_5/multi_head_attention_5/dropout/Identity
Bmodel_2/transformer_block_5/multi_head_attention_5/einsum_1/EinsumEinsumLmodel_2/transformer_block_5/multi_head_attention_5/dropout/Identity:output:0@model_2/transformer_block_5/multi_head_attention_5/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2D
Bmodel_2/transformer_block_5/multi_head_attention_5/einsum_1/Einsumò
`model_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpimodel_2_transformer_block_5_multi_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02b
`model_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpÃ
Qmodel_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/EinsumEinsumKmodel_2/transformer_block_5/multi_head_attention_5/einsum_1/Einsum:output:0hmodel_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe2S
Qmodel_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/EinsumÌ
Vmodel_2/transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOp_model_2_transformer_block_5_multi_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02X
Vmodel_2/transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp
Gmodel_2/transformer_block_5/multi_head_attention_5/attention_output/addAddV2Zmodel_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum:output:0^model_2/transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2I
Gmodel_2/transformer_block_5/multi_head_attention_5/attention_output/addñ
/model_2/transformer_block_5/dropout_14/IdentityIdentityKmodel_2/transformer_block_5/multi_head_attention_5/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/model_2/transformer_block_5/dropout_14/IdentityÒ
model_2/transformer_block_5/addAddV2model_2/add_2/add:z:08model_2/transformer_block_5/dropout_14/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
model_2/transformer_block_5/addð
Qmodel_2/transformer_block_5/layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_2/transformer_block_5/layer_normalization_10/moments/mean/reduction_indicesÒ
?model_2/transformer_block_5/layer_normalization_10/moments/meanMean#model_2/transformer_block_5/add:z:0Zmodel_2/transformer_block_5/layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2A
?model_2/transformer_block_5/layer_normalization_10/moments/mean¢
Gmodel_2/transformer_block_5/layer_normalization_10/moments/StopGradientStopGradientHmodel_2/transformer_block_5/layer_normalization_10/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2I
Gmodel_2/transformer_block_5/layer_normalization_10/moments/StopGradientÞ
Lmodel_2/transformer_block_5/layer_normalization_10/moments/SquaredDifferenceSquaredDifference#model_2/transformer_block_5/add:z:0Pmodel_2/transformer_block_5/layer_normalization_10/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2N
Lmodel_2/transformer_block_5/layer_normalization_10/moments/SquaredDifferenceø
Umodel_2/transformer_block_5/layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_2/transformer_block_5/layer_normalization_10/moments/variance/reduction_indices
Cmodel_2/transformer_block_5/layer_normalization_10/moments/varianceMeanPmodel_2/transformer_block_5/layer_normalization_10/moments/SquaredDifference:z:0^model_2/transformer_block_5/layer_normalization_10/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2E
Cmodel_2/transformer_block_5/layer_normalization_10/moments/varianceÍ
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752D
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add/yÞ
@model_2/transformer_block_5/layer_normalization_10/batchnorm/addAddV2Lmodel_2/transformer_block_5/layer_normalization_10/moments/variance:output:0Kmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2B
@model_2/transformer_block_5/layer_normalization_10/batchnorm/add
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/RsqrtRsqrtDmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2D
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/Rsqrt·
Omodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_2_transformer_block_5_layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02Q
Omodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpâ
@model_2/transformer_block_5/layer_normalization_10/batchnorm/mulMulFmodel_2/transformer_block_5/layer_normalization_10/batchnorm/Rsqrt:y:0Wmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2B
@model_2/transformer_block_5/layer_normalization_10/batchnorm/mul°
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul_1Mul#model_2/transformer_block_5/add:z:0Dmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2D
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul_1Õ
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul_2MulHmodel_2/transformer_block_5/layer_normalization_10/moments/mean:output:0Dmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2D
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul_2«
Kmodel_2/transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOpTmodel_2_transformer_block_5_layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02M
Kmodel_2/transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpÞ
@model_2/transformer_block_5/layer_normalization_10/batchnorm/subSubSmodel_2/transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp:value:0Fmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2B
@model_2/transformer_block_5/layer_normalization_10/batchnorm/subÕ
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add_1AddV2Fmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul_1:z:0Dmodel_2/transformer_block_5/layer_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2D
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add_1¬
Jmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOpSmodel_2_transformer_block_5_sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02L
Jmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpÎ
@model_2/transformer_block_5/sequential_5/dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_2/transformer_block_5/sequential_5/dense_16/Tensordot/axesÕ
@model_2/transformer_block_5/sequential_5/dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_2/transformer_block_5/sequential_5/dense_16/Tensordot/freeü
Amodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ShapeShapeFmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add_1:z:0*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ShapeØ
Imodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2/axisË
Dmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2GatherV2Jmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Shape:output:0Imodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/free:output:0Rmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2Ü
Kmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1/axisÑ
Fmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1GatherV2Jmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Shape:output:0Imodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/axes:output:0Tmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1Ð
Amodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ConstÈ
@model_2/transformer_block_5/sequential_5/dense_16/Tensordot/ProdProdMmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2:output:0Jmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_2/transformer_block_5/sequential_5/dense_16/Tensordot/ProdÔ
Cmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Const_1Ð
Bmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Prod_1ProdOmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1:output:0Lmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Prod_1Ô
Gmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat/axisª
Bmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concatConcatV2Imodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/free:output:0Imodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/axes:output:0Pmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concatÔ
Amodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/stackPackImodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Prod:output:0Kmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/stackæ
Emodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/transpose	TransposeFmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add_1:z:0Kmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2G
Emodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/transposeç
Cmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReshapeReshapeImodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/transpose:y:0Jmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2E
Cmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Reshapeæ
Bmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/MatMulMatMulLmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Reshape:output:0Rmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2D
Bmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/MatMulÔ
Cmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2E
Cmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Const_2Ø
Imodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat_1/axis·
Dmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat_1ConcatV2Mmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2:output:0Lmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Const_2:output:0Rmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat_1Ø
;model_2/transformer_block_5/sequential_5/dense_16/TensordotReshapeLmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/MatMul:product:0Mmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2=
;model_2/transformer_block_5/sequential_5/dense_16/Tensordot¢
Hmodel_2/transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOpQmodel_2_transformer_block_5_sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02J
Hmodel_2/transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpÏ
9model_2/transformer_block_5/sequential_5/dense_16/BiasAddBiasAddDmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot:output:0Pmodel_2/transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2;
9model_2/transformer_block_5/sequential_5/dense_16/BiasAddò
6model_2/transformer_block_5/sequential_5/dense_16/ReluReluBmodel_2/transformer_block_5/sequential_5/dense_16/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@28
6model_2/transformer_block_5/sequential_5/dense_16/Relu¬
Jmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOpSmodel_2_transformer_block_5_sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02L
Jmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOpÎ
@model_2/transformer_block_5/sequential_5/dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_2/transformer_block_5/sequential_5/dense_17/Tensordot/axesÕ
@model_2/transformer_block_5/sequential_5/dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_2/transformer_block_5/sequential_5/dense_17/Tensordot/freeú
Amodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ShapeShapeDmodel_2/transformer_block_5/sequential_5/dense_16/Relu:activations:0*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ShapeØ
Imodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2/axisË
Dmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2GatherV2Jmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Shape:output:0Imodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/free:output:0Rmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2Ü
Kmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1/axisÑ
Fmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1GatherV2Jmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Shape:output:0Imodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/axes:output:0Tmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1Ð
Amodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ConstÈ
@model_2/transformer_block_5/sequential_5/dense_17/Tensordot/ProdProdMmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2:output:0Jmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_2/transformer_block_5/sequential_5/dense_17/Tensordot/ProdÔ
Cmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Const_1Ð
Bmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Prod_1ProdOmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1:output:0Lmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Prod_1Ô
Gmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat/axisª
Bmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concatConcatV2Imodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/free:output:0Imodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/axes:output:0Pmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concatÔ
Amodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/stackPackImodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Prod:output:0Kmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/stackä
Emodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/transpose	TransposeDmodel_2/transformer_block_5/sequential_5/dense_16/Relu:activations:0Kmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2G
Emodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/transposeç
Cmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ReshapeReshapeImodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/transpose:y:0Jmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2E
Cmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Reshapeæ
Bmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/MatMulMatMulLmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Reshape:output:0Rmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2D
Bmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/MatMulÔ
Cmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Const_2Ø
Imodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat_1/axis·
Dmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat_1ConcatV2Mmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2:output:0Lmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Const_2:output:0Rmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat_1Ø
;model_2/transformer_block_5/sequential_5/dense_17/TensordotReshapeLmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/MatMul:product:0Mmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2=
;model_2/transformer_block_5/sequential_5/dense_17/Tensordot¢
Hmodel_2/transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOpQmodel_2_transformer_block_5_sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel_2/transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpÏ
9model_2/transformer_block_5/sequential_5/dense_17/BiasAddBiasAddDmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot:output:0Pmodel_2/transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9model_2/transformer_block_5/sequential_5/dense_17/BiasAddè
/model_2/transformer_block_5/dropout_15/IdentityIdentityBmodel_2/transformer_block_5/sequential_5/dense_17/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/model_2/transformer_block_5/dropout_15/Identity
!model_2/transformer_block_5/add_1AddV2Fmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add_1:z:08model_2/transformer_block_5/dropout_15/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2#
!model_2/transformer_block_5/add_1ð
Qmodel_2/transformer_block_5/layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_2/transformer_block_5/layer_normalization_11/moments/mean/reduction_indicesÔ
?model_2/transformer_block_5/layer_normalization_11/moments/meanMean%model_2/transformer_block_5/add_1:z:0Zmodel_2/transformer_block_5/layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2A
?model_2/transformer_block_5/layer_normalization_11/moments/mean¢
Gmodel_2/transformer_block_5/layer_normalization_11/moments/StopGradientStopGradientHmodel_2/transformer_block_5/layer_normalization_11/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2I
Gmodel_2/transformer_block_5/layer_normalization_11/moments/StopGradientà
Lmodel_2/transformer_block_5/layer_normalization_11/moments/SquaredDifferenceSquaredDifference%model_2/transformer_block_5/add_1:z:0Pmodel_2/transformer_block_5/layer_normalization_11/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2N
Lmodel_2/transformer_block_5/layer_normalization_11/moments/SquaredDifferenceø
Umodel_2/transformer_block_5/layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_2/transformer_block_5/layer_normalization_11/moments/variance/reduction_indices
Cmodel_2/transformer_block_5/layer_normalization_11/moments/varianceMeanPmodel_2/transformer_block_5/layer_normalization_11/moments/SquaredDifference:z:0^model_2/transformer_block_5/layer_normalization_11/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2E
Cmodel_2/transformer_block_5/layer_normalization_11/moments/varianceÍ
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752D
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add/yÞ
@model_2/transformer_block_5/layer_normalization_11/batchnorm/addAddV2Lmodel_2/transformer_block_5/layer_normalization_11/moments/variance:output:0Kmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2B
@model_2/transformer_block_5/layer_normalization_11/batchnorm/add
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/RsqrtRsqrtDmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2D
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/Rsqrt·
Omodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_2_transformer_block_5_layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02Q
Omodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpâ
@model_2/transformer_block_5/layer_normalization_11/batchnorm/mulMulFmodel_2/transformer_block_5/layer_normalization_11/batchnorm/Rsqrt:y:0Wmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2B
@model_2/transformer_block_5/layer_normalization_11/batchnorm/mul²
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul_1Mul%model_2/transformer_block_5/add_1:z:0Dmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2D
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul_1Õ
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul_2MulHmodel_2/transformer_block_5/layer_normalization_11/moments/mean:output:0Dmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2D
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul_2«
Kmodel_2/transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOpTmodel_2_transformer_block_5_layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02M
Kmodel_2/transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpÞ
@model_2/transformer_block_5/layer_normalization_11/batchnorm/subSubSmodel_2/transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp:value:0Fmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2B
@model_2/transformer_block_5/layer_normalization_11/batchnorm/subÕ
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add_1AddV2Fmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul_1:z:0Dmodel_2/transformer_block_5/layer_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2D
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add_1
model_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`  2
model_2/flatten_2/ConstÞ
model_2/flatten_2/ReshapeReshapeFmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add_1:z:0 model_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
model_2/flatten_2/Reshapeì
6model_2/batch_normalization_8/batchnorm/ReadVariableOpReadVariableOp?model_2_batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype028
6model_2/batch_normalization_8/batchnorm/ReadVariableOp£
-model_2/batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-model_2/batch_normalization_8/batchnorm/add/y
+model_2/batch_normalization_8/batchnorm/addAddV2>model_2/batch_normalization_8/batchnorm/ReadVariableOp:value:06model_2/batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:2-
+model_2/batch_normalization_8/batchnorm/add½
-model_2/batch_normalization_8/batchnorm/RsqrtRsqrt/model_2/batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:2/
-model_2/batch_normalization_8/batchnorm/Rsqrtø
:model_2/batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_2_batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02<
:model_2/batch_normalization_8/batchnorm/mul/ReadVariableOpý
+model_2/batch_normalization_8/batchnorm/mulMul1model_2/batch_normalization_8/batchnorm/Rsqrt:y:0Bmodel_2/batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+model_2/batch_normalization_8/batchnorm/mulÑ
-model_2/batch_normalization_8/batchnorm/mul_1Mulinput_6/model_2/batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-model_2/batch_normalization_8/batchnorm/mul_1ò
8model_2/batch_normalization_8/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_2_batch_normalization_8_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8model_2/batch_normalization_8/batchnorm/ReadVariableOp_1ý
-model_2/batch_normalization_8/batchnorm/mul_2Mul@model_2/batch_normalization_8/batchnorm/ReadVariableOp_1:value:0/model_2/batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:2/
-model_2/batch_normalization_8/batchnorm/mul_2ò
8model_2/batch_normalization_8/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_2_batch_normalization_8_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02:
8model_2/batch_normalization_8/batchnorm/ReadVariableOp_2û
+model_2/batch_normalization_8/batchnorm/subSub@model_2/batch_normalization_8/batchnorm/ReadVariableOp_2:value:01model_2/batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2-
+model_2/batch_normalization_8/batchnorm/subý
-model_2/batch_normalization_8/batchnorm/add_1AddV21model_2/batch_normalization_8/batchnorm/mul_1:z:0/model_2/batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-model_2/batch_normalization_8/batchnorm/add_1
!model_2/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_2/concatenate_2/concat/axis
model_2/concatenate_2/concatConcatV2"model_2/flatten_2/Reshape:output:01model_2/batch_normalization_8/batchnorm/add_1:z:0*model_2/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
model_2/concatenate_2/concatÁ
&model_2/dense_18/MatMul/ReadVariableOpReadVariableOp/model_2_dense_18_matmul_readvariableop_resource*
_output_shapes
:	è@*
dtype02(
&model_2/dense_18/MatMul/ReadVariableOpÅ
model_2/dense_18/MatMulMatMul%model_2/concatenate_2/concat:output:0.model_2/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_2/dense_18/MatMul¿
'model_2/dense_18/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_2/dense_18/BiasAdd/ReadVariableOpÅ
model_2/dense_18/BiasAddBiasAdd!model_2/dense_18/MatMul:product:0/model_2/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_2/dense_18/BiasAdd
model_2/dense_18/ReluRelu!model_2/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_2/dense_18/Relu
model_2/dropout_16/IdentityIdentity#model_2/dense_18/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_2/dropout_16/IdentityÀ
&model_2/dense_19/MatMul/ReadVariableOpReadVariableOp/model_2_dense_19_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02(
&model_2/dense_19/MatMul/ReadVariableOpÄ
model_2/dense_19/MatMulMatMul$model_2/dropout_16/Identity:output:0.model_2/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_2/dense_19/MatMul¿
'model_2/dense_19/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_2/dense_19/BiasAdd/ReadVariableOpÅ
model_2/dense_19/BiasAddBiasAdd!model_2/dense_19/MatMul:product:0/model_2/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_2/dense_19/BiasAdd
model_2/dense_19/ReluRelu!model_2/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_2/dense_19/Relu
model_2/dropout_17/IdentityIdentity#model_2/dense_19/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_2/dropout_17/IdentityÀ
&model_2/dense_20/MatMul/ReadVariableOpReadVariableOp/model_2_dense_20_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&model_2/dense_20/MatMul/ReadVariableOpÄ
model_2/dense_20/MatMulMatMul$model_2/dropout_17/Identity:output:0.model_2/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/dense_20/MatMul¿
'model_2/dense_20/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_2/dense_20/BiasAdd/ReadVariableOpÅ
model_2/dense_20/BiasAddBiasAdd!model_2/dense_20/MatMul:product:0/model_2/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/dense_20/BiasAdd
IdentityIdentity!model_2/dense_20/BiasAdd:output:07^model_2/batch_normalization_6/batchnorm/ReadVariableOp9^model_2/batch_normalization_6/batchnorm/ReadVariableOp_19^model_2/batch_normalization_6/batchnorm/ReadVariableOp_2;^model_2/batch_normalization_6/batchnorm/mul/ReadVariableOp7^model_2/batch_normalization_7/batchnorm/ReadVariableOp9^model_2/batch_normalization_7/batchnorm/ReadVariableOp_19^model_2/batch_normalization_7/batchnorm/ReadVariableOp_2;^model_2/batch_normalization_7/batchnorm/mul/ReadVariableOp7^model_2/batch_normalization_8/batchnorm/ReadVariableOp9^model_2/batch_normalization_8/batchnorm/ReadVariableOp_19^model_2/batch_normalization_8/batchnorm/ReadVariableOp_2;^model_2/batch_normalization_8/batchnorm/mul/ReadVariableOp(^model_2/conv1d_4/BiasAdd/ReadVariableOp4^model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp(^model_2/conv1d_5/BiasAdd/ReadVariableOp4^model_2/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp(^model_2/dense_18/BiasAdd/ReadVariableOp'^model_2/dense_18/MatMul/ReadVariableOp(^model_2/dense_19/BiasAdd/ReadVariableOp'^model_2/dense_19/MatMul/ReadVariableOp(^model_2/dense_20/BiasAdd/ReadVariableOp'^model_2/dense_20/MatMul/ReadVariableOpD^model_2/token_and_position_embedding_2/embedding_4/embedding_lookupD^model_2/token_and_position_embedding_2/embedding_5/embedding_lookupL^model_2/transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpP^model_2/transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpL^model_2/transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpP^model_2/transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpW^model_2/transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpa^model_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpJ^model_2/transformer_block_5/multi_head_attention_5/key/add/ReadVariableOpT^model_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpL^model_2/transformer_block_5/multi_head_attention_5/query/add/ReadVariableOpV^model_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpL^model_2/transformer_block_5/multi_head_attention_5/value/add/ReadVariableOpV^model_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpI^model_2/transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpK^model_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpI^model_2/transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpK^model_2/transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::2p
6model_2/batch_normalization_6/batchnorm/ReadVariableOp6model_2/batch_normalization_6/batchnorm/ReadVariableOp2t
8model_2/batch_normalization_6/batchnorm/ReadVariableOp_18model_2/batch_normalization_6/batchnorm/ReadVariableOp_12t
8model_2/batch_normalization_6/batchnorm/ReadVariableOp_28model_2/batch_normalization_6/batchnorm/ReadVariableOp_22x
:model_2/batch_normalization_6/batchnorm/mul/ReadVariableOp:model_2/batch_normalization_6/batchnorm/mul/ReadVariableOp2p
6model_2/batch_normalization_7/batchnorm/ReadVariableOp6model_2/batch_normalization_7/batchnorm/ReadVariableOp2t
8model_2/batch_normalization_7/batchnorm/ReadVariableOp_18model_2/batch_normalization_7/batchnorm/ReadVariableOp_12t
8model_2/batch_normalization_7/batchnorm/ReadVariableOp_28model_2/batch_normalization_7/batchnorm/ReadVariableOp_22x
:model_2/batch_normalization_7/batchnorm/mul/ReadVariableOp:model_2/batch_normalization_7/batchnorm/mul/ReadVariableOp2p
6model_2/batch_normalization_8/batchnorm/ReadVariableOp6model_2/batch_normalization_8/batchnorm/ReadVariableOp2t
8model_2/batch_normalization_8/batchnorm/ReadVariableOp_18model_2/batch_normalization_8/batchnorm/ReadVariableOp_12t
8model_2/batch_normalization_8/batchnorm/ReadVariableOp_28model_2/batch_normalization_8/batchnorm/ReadVariableOp_22x
:model_2/batch_normalization_8/batchnorm/mul/ReadVariableOp:model_2/batch_normalization_8/batchnorm/mul/ReadVariableOp2R
'model_2/conv1d_4/BiasAdd/ReadVariableOp'model_2/conv1d_4/BiasAdd/ReadVariableOp2j
3model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp3model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp2R
'model_2/conv1d_5/BiasAdd/ReadVariableOp'model_2/conv1d_5/BiasAdd/ReadVariableOp2j
3model_2/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp3model_2/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp2R
'model_2/dense_18/BiasAdd/ReadVariableOp'model_2/dense_18/BiasAdd/ReadVariableOp2P
&model_2/dense_18/MatMul/ReadVariableOp&model_2/dense_18/MatMul/ReadVariableOp2R
'model_2/dense_19/BiasAdd/ReadVariableOp'model_2/dense_19/BiasAdd/ReadVariableOp2P
&model_2/dense_19/MatMul/ReadVariableOp&model_2/dense_19/MatMul/ReadVariableOp2R
'model_2/dense_20/BiasAdd/ReadVariableOp'model_2/dense_20/BiasAdd/ReadVariableOp2P
&model_2/dense_20/MatMul/ReadVariableOp&model_2/dense_20/MatMul/ReadVariableOp2
Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookupCmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup2
Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookupCmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup2
Kmodel_2/transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpKmodel_2/transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp2¢
Omodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpOmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp2
Kmodel_2/transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpKmodel_2/transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp2¢
Omodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpOmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp2°
Vmodel_2/transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpVmodel_2/transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp2Ä
`model_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp`model_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp2
Imodel_2/transformer_block_5/multi_head_attention_5/key/add/ReadVariableOpImodel_2/transformer_block_5/multi_head_attention_5/key/add/ReadVariableOp2ª
Smodel_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpSmodel_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp2
Kmodel_2/transformer_block_5/multi_head_attention_5/query/add/ReadVariableOpKmodel_2/transformer_block_5/multi_head_attention_5/query/add/ReadVariableOp2®
Umodel_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpUmodel_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp2
Kmodel_2/transformer_block_5/multi_head_attention_5/value/add/ReadVariableOpKmodel_2/transformer_block_5/multi_head_attention_5/value/add/ReadVariableOp2®
Umodel_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpUmodel_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp2
Hmodel_2/transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpHmodel_2/transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp2
Jmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpJmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp2
Hmodel_2/transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpHmodel_2/transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp2
Jmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOpJmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_5:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6

÷
D__inference_conv1d_5_layer_call_and_return_conditional_losses_423930

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
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
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
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÞ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 
 
_user_specified_nameinputs
º
©
6__inference_batch_normalization_8_layer_call_fn_424721

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4215972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
c
 
C__inference_model_2_layer_call_and_return_conditional_losses_422507
input_5
input_6)
%token_and_position_embedding_2_421648)
%token_and_position_embedding_2_421650
conv1d_4_421680
conv1d_4_421682
conv1d_5_421713
conv1d_5_421715 
batch_normalization_6_421802 
batch_normalization_6_421804 
batch_normalization_6_421806 
batch_normalization_6_421808 
batch_normalization_7_421893 
batch_normalization_7_421895 
batch_normalization_7_421897 
batch_normalization_7_421899
transformer_block_5_422268
transformer_block_5_422270
transformer_block_5_422272
transformer_block_5_422274
transformer_block_5_422276
transformer_block_5_422278
transformer_block_5_422280
transformer_block_5_422282
transformer_block_5_422284
transformer_block_5_422286
transformer_block_5_422288
transformer_block_5_422290
transformer_block_5_422292
transformer_block_5_422294
transformer_block_5_422296
transformer_block_5_422298 
batch_normalization_8_422341 
batch_normalization_8_422343 
batch_normalization_8_422345 
batch_normalization_8_422347
dense_18_422388
dense_18_422390
dense_19_422445
dense_19_422447
dense_20_422501
dense_20_422503
identity¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢ conv1d_4/StatefulPartitionedCall¢ conv1d_5/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢ dense_20/StatefulPartitionedCall¢"dropout_16/StatefulPartitionedCall¢"dropout_17/StatefulPartitionedCall¢6token_and_position_embedding_2/StatefulPartitionedCall¢+transformer_block_5/StatefulPartitionedCall
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_5%token_and_position_embedding_2_421648%token_and_position_embedding_2_421650*
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
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_42163728
6token_and_position_embedding_2/StatefulPartitionedCallÕ
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0conv1d_4_421680conv1d_4_421682*
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
GPU2*0J 8 *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_4216692"
 conv1d_4/StatefulPartitionedCall 
#average_pooling1d_6/PartitionedCallPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_6_layer_call_and_return_conditional_losses_4209852%
#average_pooling1d_6/PartitionedCallÂ
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_6/PartitionedCall:output:0conv1d_5_421713conv1d_5_421715*
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
D__inference_conv1d_5_layer_call_and_return_conditional_losses_4217022"
 conv1d_5/StatefulPartitionedCallµ
#average_pooling1d_8/PartitionedCallPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0*
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
O__inference_average_pooling1d_8_layer_call_and_return_conditional_losses_4210152%
#average_pooling1d_8/PartitionedCall
#average_pooling1d_7/PartitionedCallPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0*
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
O__inference_average_pooling1d_7_layer_call_and_return_conditional_losses_4210002%
#average_pooling1d_7/PartitionedCallÀ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_7/PartitionedCall:output:0batch_normalization_6_421802batch_normalization_6_421804batch_normalization_6_421806batch_normalization_6_421808*
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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4217552/
-batch_normalization_6/StatefulPartitionedCallÀ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_8/PartitionedCall:output:0batch_normalization_7_421893batch_normalization_7_421895batch_normalization_7_421897batch_normalization_7_421899*
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4218462/
-batch_normalization_7/StatefulPartitionedCall»
add_2/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
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
A__inference_add_2_layer_call_and_return_conditional_losses_4219082
add_2/PartitionedCall
+transformer_block_5/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0transformer_block_5_422268transformer_block_5_422270transformer_block_5_422272transformer_block_5_422274transformer_block_5_422276transformer_block_5_422278transformer_block_5_422280transformer_block_5_422282transformer_block_5_422284transformer_block_5_422286transformer_block_5_422288transformer_block_5_422290transformer_block_5_422292transformer_block_5_422294transformer_block_5_422296transformer_block_5_422298*
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
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_4220652-
+transformer_block_5/StatefulPartitionedCall
flatten_2/PartitionedCallPartitionedCall4transformer_block_5/StatefulPartitionedCall:output:0*
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
E__inference_flatten_2_layer_call_and_return_conditional_losses_4223072
flatten_2/PartitionedCall
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCallinput_6batch_normalization_8_422341batch_normalization_8_422343batch_normalization_8_422345batch_normalization_8_422347*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4215642/
-batch_normalization_8/StatefulPartitionedCall¼
concatenate_2/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:06batch_normalization_8/StatefulPartitionedCall:output:0*
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
I__inference_concatenate_2_layer_call_and_return_conditional_losses_4223572
concatenate_2/PartitionedCall·
 dense_18/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_18_422388dense_18_422390*
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
D__inference_dense_18_layer_call_and_return_conditional_losses_4223772"
 dense_18/StatefulPartitionedCall
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
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
F__inference_dropout_16_layer_call_and_return_conditional_losses_4224052$
"dropout_16/StatefulPartitionedCall¼
 dense_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0dense_19_422445dense_19_422447*
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
D__inference_dense_19_layer_call_and_return_conditional_losses_4224342"
 dense_19/StatefulPartitionedCall½
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
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
F__inference_dropout_17_layer_call_and_return_conditional_losses_4224622$
"dropout_17/StatefulPartitionedCall¼
 dense_20/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_20_422501dense_20_422503*
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
D__inference_dense_20_layer_call_and_return_conditional_losses_4224902"
 dense_20/StatefulPartitionedCallí
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_5/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_5/StatefulPartitionedCall+transformer_block_5/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_5:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6
â
ä
(__inference_model_2_layer_call_fn_423856
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

unknown_36

unknown_37

unknown_38
identity¢StatefulPartitionedCall
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
unknown_36
unknown_37
unknown_38*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_4229062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::22
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
¬
R
&__inference_add_2_layer_call_fn_424279
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
A__inference_add_2_layer_call_and_return_conditional_losses_4219082
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
¡
F
*__inference_flatten_2_layer_call_fn_424639

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
E__inference_flatten_2_layer_call_and_return_conditional_losses_4223072
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

G
+__inference_dropout_17_layer_call_fn_424828

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
F__inference_dropout_17_layer_call_and_return_conditional_losses_4224672
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


Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_421150

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
Ð
¨
-__inference_sequential_5_layer_call_fn_421441
dense_16_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCalldense_16_inputunknown	unknown_0	unknown_1	unknown_2*
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
H__inference_sequential_5_layer_call_and_return_conditional_losses_4214302
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
_user_specified_namedense_16_input
Ð

à
4__inference_transformer_block_5_layer_call_fn_424628

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
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_4221922
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


?__inference_token_and_position_embedding_2_layer_call_fn_423889
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
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_4216372
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
¼0
È
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_421846

inputs
assignmovingavg_421821
assignmovingavg_1_421827)
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
loc:@AssignMovingAvg/421821*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_421821*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/421821*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/421821*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_421821AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/421821*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/421827*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_421827*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/421827*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/421827*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_421827AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/421827*
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
¨
Z
.__inference_concatenate_2_layer_call_fn_424734
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
I__inference_concatenate_2_layer_call_and_return_conditional_losses_4223572
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
E__inference_flatten_2_layer_call_and_return_conditional_losses_424634

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
ñ	
Ý
D__inference_dense_18_layer_call_and_return_conditional_losses_424745

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
é

H__inference_sequential_5_layer_call_and_return_conditional_losses_421430

inputs
dense_16_421419
dense_16_421421
dense_17_421424
dense_17_421426
identity¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall
 dense_16/StatefulPartitionedCallStatefulPartitionedCallinputsdense_16_421419dense_16_421421*
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
D__inference_dense_16_layer_call_and_return_conditional_losses_4213362"
 dense_16/StatefulPartitionedCall¾
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_421424dense_17_421426*
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
D__inference_dense_17_layer_call_and_return_conditional_losses_4213822"
 dense_17/StatefulPartitionedCallÇ
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs

P
4__inference_average_pooling1d_7_layer_call_fn_421006

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
O__inference_average_pooling1d_7_layer_call_and_return_conditional_losses_4210002
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
¸
©
6__inference_batch_normalization_8_layer_call_fn_424708

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4215642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
¨
-__inference_sequential_5_layer_call_fn_421468
dense_16_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCalldense_16_inputunknown	unknown_0	unknown_1	unknown_2*
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
H__inference_sequential_5_layer_call_and_return_conditional_losses_4214572
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
_user_specified_namedense_16_input
ó0
È
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_421257

inputs
assignmovingavg_421232
assignmovingavg_1_421238)
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
loc:@AssignMovingAvg/421232*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_421232*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/421232*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/421232*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_421232AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/421232*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/421238*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_421238*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/421238*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/421238*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_421238AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/421238*
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
ñ	
Ý
D__inference_dense_18_layer_call_and_return_conditional_losses_422377

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
è

Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_421866

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
± 
ã
D__inference_dense_16_layer_call_and_return_conditional_losses_421336

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
Ñ
ã
D__inference_dense_17_layer_call_and_return_conditional_losses_421382

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
·
k
A__inference_add_2_layer_call_and_return_conditional_losses_421908

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
Ê
©
6__inference_batch_normalization_6_layer_call_fn_424103

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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4217752
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
ô

Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_423880
x'
#embedding_5_embedding_lookup_423867'
#embedding_4_embedding_lookup_423873
identity¢embedding_4/embedding_lookup¢embedding_5/embedding_lookup?
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
embedding_5/embedding_lookupResourceGather#embedding_5_embedding_lookup_423867range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_5/embedding_lookup/423867*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02
embedding_5/embedding_lookup
%embedding_5/embedding_lookup/IdentityIdentity%embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_5/embedding_lookup/423867*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%embedding_5/embedding_lookup/IdentityÀ
'embedding_5/embedding_lookup/Identity_1Identity.embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'embedding_5/embedding_lookup/Identity_1q
embedding_4/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2
embedding_4/Castº
embedding_4/embedding_lookupResourceGather#embedding_4_embedding_lookup_423873embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_4/embedding_lookup/423873*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02
embedding_4/embedding_lookup
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_4/embedding_lookup/423873*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2'
%embedding_4/embedding_lookup/IdentityÅ
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2)
'embedding_4/embedding_lookup/Identity_1®
addAddV20embedding_4/embedding_lookup/Identity_1:output:00embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
add
IdentityIdentityadd:z:0^embedding_4/embedding_lookup^embedding_5/embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿR::2<
embedding_4/embedding_lookupembedding_4/embedding_lookup2<
embedding_5/embedding_lookupembedding_5/embedding_lookup:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR

_user_specified_namex
ó
~
)__inference_conv1d_5_layer_call_fn_423939

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
D__inference_conv1d_5_layer_call_and_return_conditional_losses_4217022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÞ ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 
 
_user_specified_nameinputs

G
+__inference_dropout_16_layer_call_fn_424781

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
F__inference_dropout_16_layer_call_and_return_conditional_losses_4224102
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
Øà
5
"__inference__traced_restore_425580
file_prefix$
 assignvariableop_conv1d_4_kernel$
 assignvariableop_1_conv1d_4_bias&
"assignvariableop_2_conv1d_5_kernel$
 assignvariableop_3_conv1d_5_bias2
.assignvariableop_4_batch_normalization_6_gamma1
-assignvariableop_5_batch_normalization_6_beta8
4assignvariableop_6_batch_normalization_6_moving_mean<
8assignvariableop_7_batch_normalization_6_moving_variance2
.assignvariableop_8_batch_normalization_7_gamma1
-assignvariableop_9_batch_normalization_7_beta9
5assignvariableop_10_batch_normalization_7_moving_mean=
9assignvariableop_11_batch_normalization_7_moving_variance3
/assignvariableop_12_batch_normalization_8_gamma2
.assignvariableop_13_batch_normalization_8_beta9
5assignvariableop_14_batch_normalization_8_moving_mean=
9assignvariableop_15_batch_normalization_8_moving_variance'
#assignvariableop_16_dense_18_kernel%
!assignvariableop_17_dense_18_bias'
#assignvariableop_18_dense_19_kernel%
!assignvariableop_19_dense_19_bias'
#assignvariableop_20_dense_20_kernel%
!assignvariableop_21_dense_20_bias
assignvariableop_22_decay%
!assignvariableop_23_learning_rate 
assignvariableop_24_momentum 
assignvariableop_25_sgd_iterM
Iassignvariableop_26_token_and_position_embedding_2_embedding_4_embeddingsM
Iassignvariableop_27_token_and_position_embedding_2_embedding_5_embeddingsO
Kassignvariableop_28_transformer_block_5_multi_head_attention_5_query_kernelM
Iassignvariableop_29_transformer_block_5_multi_head_attention_5_query_biasM
Iassignvariableop_30_transformer_block_5_multi_head_attention_5_key_kernelK
Gassignvariableop_31_transformer_block_5_multi_head_attention_5_key_biasO
Kassignvariableop_32_transformer_block_5_multi_head_attention_5_value_kernelM
Iassignvariableop_33_transformer_block_5_multi_head_attention_5_value_biasZ
Vassignvariableop_34_transformer_block_5_multi_head_attention_5_attention_output_kernelX
Tassignvariableop_35_transformer_block_5_multi_head_attention_5_attention_output_bias'
#assignvariableop_36_dense_16_kernel%
!assignvariableop_37_dense_16_bias'
#assignvariableop_38_dense_17_kernel%
!assignvariableop_39_dense_17_biasH
Dassignvariableop_40_transformer_block_5_layer_normalization_10_gammaG
Cassignvariableop_41_transformer_block_5_layer_normalization_10_betaH
Dassignvariableop_42_transformer_block_5_layer_normalization_11_gammaG
Cassignvariableop_43_transformer_block_5_layer_normalization_11_beta
assignvariableop_44_total
assignvariableop_45_count4
0assignvariableop_46_sgd_conv1d_4_kernel_momentum2
.assignvariableop_47_sgd_conv1d_4_bias_momentum4
0assignvariableop_48_sgd_conv1d_5_kernel_momentum2
.assignvariableop_49_sgd_conv1d_5_bias_momentum@
<assignvariableop_50_sgd_batch_normalization_6_gamma_momentum?
;assignvariableop_51_sgd_batch_normalization_6_beta_momentum@
<assignvariableop_52_sgd_batch_normalization_7_gamma_momentum?
;assignvariableop_53_sgd_batch_normalization_7_beta_momentum@
<assignvariableop_54_sgd_batch_normalization_8_gamma_momentum?
;assignvariableop_55_sgd_batch_normalization_8_beta_momentum4
0assignvariableop_56_sgd_dense_18_kernel_momentum2
.assignvariableop_57_sgd_dense_18_bias_momentum4
0assignvariableop_58_sgd_dense_19_kernel_momentum2
.assignvariableop_59_sgd_dense_19_bias_momentum4
0assignvariableop_60_sgd_dense_20_kernel_momentum2
.assignvariableop_61_sgd_dense_20_bias_momentumZ
Vassignvariableop_62_sgd_token_and_position_embedding_2_embedding_4_embeddings_momentumZ
Vassignvariableop_63_sgd_token_and_position_embedding_2_embedding_5_embeddings_momentum\
Xassignvariableop_64_sgd_transformer_block_5_multi_head_attention_5_query_kernel_momentumZ
Vassignvariableop_65_sgd_transformer_block_5_multi_head_attention_5_query_bias_momentumZ
Vassignvariableop_66_sgd_transformer_block_5_multi_head_attention_5_key_kernel_momentumX
Tassignvariableop_67_sgd_transformer_block_5_multi_head_attention_5_key_bias_momentum\
Xassignvariableop_68_sgd_transformer_block_5_multi_head_attention_5_value_kernel_momentumZ
Vassignvariableop_69_sgd_transformer_block_5_multi_head_attention_5_value_bias_momentumg
cassignvariableop_70_sgd_transformer_block_5_multi_head_attention_5_attention_output_kernel_momentume
aassignvariableop_71_sgd_transformer_block_5_multi_head_attention_5_attention_output_bias_momentum4
0assignvariableop_72_sgd_dense_16_kernel_momentum2
.assignvariableop_73_sgd_dense_16_bias_momentum4
0assignvariableop_74_sgd_dense_17_kernel_momentum2
.assignvariableop_75_sgd_dense_17_bias_momentumU
Qassignvariableop_76_sgd_transformer_block_5_layer_normalization_10_gamma_momentumT
Passignvariableop_77_sgd_transformer_block_5_layer_normalization_10_beta_momentumU
Qassignvariableop_78_sgd_transformer_block_5_layer_normalization_11_gamma_momentumT
Passignvariableop_79_sgd_transformer_block_5_layer_normalization_11_beta_momentum
identity_81¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_9)
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*(
value(B(QB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names³
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*·
value­BªQB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÃ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ú
_output_shapesÇ
Ä:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*_
dtypesU
S2Q	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_conv1d_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4³
AssignVariableOp_4AssignVariableOp.assignvariableop_4_batch_normalization_6_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5²
AssignVariableOp_5AssignVariableOp-assignvariableop_5_batch_normalization_6_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¹
AssignVariableOp_6AssignVariableOp4assignvariableop_6_batch_normalization_6_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7½
AssignVariableOp_7AssignVariableOp8assignvariableop_7_batch_normalization_6_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8³
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_7_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9²
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_7_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10½
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_7_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Á
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_7_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12·
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_8_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¶
AssignVariableOp_13AssignVariableOp.assignvariableop_13_batch_normalization_8_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14½
AssignVariableOp_14AssignVariableOp5assignvariableop_14_batch_normalization_8_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Á
AssignVariableOp_15AssignVariableOp9assignvariableop_15_batch_normalization_8_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16«
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_18_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17©
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_18_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18«
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_19_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19©
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_19_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20«
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_20_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21©
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_20_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¡
AssignVariableOp_22AssignVariableOpassignvariableop_22_decayIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23©
AssignVariableOp_23AssignVariableOp!assignvariableop_23_learning_rateIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¤
AssignVariableOp_24AssignVariableOpassignvariableop_24_momentumIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_25¤
AssignVariableOp_25AssignVariableOpassignvariableop_25_sgd_iterIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ñ
AssignVariableOp_26AssignVariableOpIassignvariableop_26_token_and_position_embedding_2_embedding_4_embeddingsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ñ
AssignVariableOp_27AssignVariableOpIassignvariableop_27_token_and_position_embedding_2_embedding_5_embeddingsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ó
AssignVariableOp_28AssignVariableOpKassignvariableop_28_transformer_block_5_multi_head_attention_5_query_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ñ
AssignVariableOp_29AssignVariableOpIassignvariableop_29_transformer_block_5_multi_head_attention_5_query_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ñ
AssignVariableOp_30AssignVariableOpIassignvariableop_30_transformer_block_5_multi_head_attention_5_key_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ï
AssignVariableOp_31AssignVariableOpGassignvariableop_31_transformer_block_5_multi_head_attention_5_key_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ó
AssignVariableOp_32AssignVariableOpKassignvariableop_32_transformer_block_5_multi_head_attention_5_value_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ñ
AssignVariableOp_33AssignVariableOpIassignvariableop_33_transformer_block_5_multi_head_attention_5_value_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Þ
AssignVariableOp_34AssignVariableOpVassignvariableop_34_transformer_block_5_multi_head_attention_5_attention_output_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Ü
AssignVariableOp_35AssignVariableOpTassignvariableop_35_transformer_block_5_multi_head_attention_5_attention_output_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36«
AssignVariableOp_36AssignVariableOp#assignvariableop_36_dense_16_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37©
AssignVariableOp_37AssignVariableOp!assignvariableop_37_dense_16_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38«
AssignVariableOp_38AssignVariableOp#assignvariableop_38_dense_17_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39©
AssignVariableOp_39AssignVariableOp!assignvariableop_39_dense_17_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Ì
AssignVariableOp_40AssignVariableOpDassignvariableop_40_transformer_block_5_layer_normalization_10_gammaIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Ë
AssignVariableOp_41AssignVariableOpCassignvariableop_41_transformer_block_5_layer_normalization_10_betaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Ì
AssignVariableOp_42AssignVariableOpDassignvariableop_42_transformer_block_5_layer_normalization_11_gammaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Ë
AssignVariableOp_43AssignVariableOpCassignvariableop_43_transformer_block_5_layer_normalization_11_betaIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¡
AssignVariableOp_44AssignVariableOpassignvariableop_44_totalIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45¡
AssignVariableOp_45AssignVariableOpassignvariableop_45_countIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46¸
AssignVariableOp_46AssignVariableOp0assignvariableop_46_sgd_conv1d_4_kernel_momentumIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47¶
AssignVariableOp_47AssignVariableOp.assignvariableop_47_sgd_conv1d_4_bias_momentumIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48¸
AssignVariableOp_48AssignVariableOp0assignvariableop_48_sgd_conv1d_5_kernel_momentumIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49¶
AssignVariableOp_49AssignVariableOp.assignvariableop_49_sgd_conv1d_5_bias_momentumIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Ä
AssignVariableOp_50AssignVariableOp<assignvariableop_50_sgd_batch_normalization_6_gamma_momentumIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Ã
AssignVariableOp_51AssignVariableOp;assignvariableop_51_sgd_batch_normalization_6_beta_momentumIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Ä
AssignVariableOp_52AssignVariableOp<assignvariableop_52_sgd_batch_normalization_7_gamma_momentumIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Ã
AssignVariableOp_53AssignVariableOp;assignvariableop_53_sgd_batch_normalization_7_beta_momentumIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54Ä
AssignVariableOp_54AssignVariableOp<assignvariableop_54_sgd_batch_normalization_8_gamma_momentumIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55Ã
AssignVariableOp_55AssignVariableOp;assignvariableop_55_sgd_batch_normalization_8_beta_momentumIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56¸
AssignVariableOp_56AssignVariableOp0assignvariableop_56_sgd_dense_18_kernel_momentumIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57¶
AssignVariableOp_57AssignVariableOp.assignvariableop_57_sgd_dense_18_bias_momentumIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58¸
AssignVariableOp_58AssignVariableOp0assignvariableop_58_sgd_dense_19_kernel_momentumIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59¶
AssignVariableOp_59AssignVariableOp.assignvariableop_59_sgd_dense_19_bias_momentumIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60¸
AssignVariableOp_60AssignVariableOp0assignvariableop_60_sgd_dense_20_kernel_momentumIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61¶
AssignVariableOp_61AssignVariableOp.assignvariableop_61_sgd_dense_20_bias_momentumIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62Þ
AssignVariableOp_62AssignVariableOpVassignvariableop_62_sgd_token_and_position_embedding_2_embedding_4_embeddings_momentumIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63Þ
AssignVariableOp_63AssignVariableOpVassignvariableop_63_sgd_token_and_position_embedding_2_embedding_5_embeddings_momentumIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64à
AssignVariableOp_64AssignVariableOpXassignvariableop_64_sgd_transformer_block_5_multi_head_attention_5_query_kernel_momentumIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65Þ
AssignVariableOp_65AssignVariableOpVassignvariableop_65_sgd_transformer_block_5_multi_head_attention_5_query_bias_momentumIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66Þ
AssignVariableOp_66AssignVariableOpVassignvariableop_66_sgd_transformer_block_5_multi_head_attention_5_key_kernel_momentumIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67Ü
AssignVariableOp_67AssignVariableOpTassignvariableop_67_sgd_transformer_block_5_multi_head_attention_5_key_bias_momentumIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68à
AssignVariableOp_68AssignVariableOpXassignvariableop_68_sgd_transformer_block_5_multi_head_attention_5_value_kernel_momentumIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69Þ
AssignVariableOp_69AssignVariableOpVassignvariableop_69_sgd_transformer_block_5_multi_head_attention_5_value_bias_momentumIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70ë
AssignVariableOp_70AssignVariableOpcassignvariableop_70_sgd_transformer_block_5_multi_head_attention_5_attention_output_kernel_momentumIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71é
AssignVariableOp_71AssignVariableOpaassignvariableop_71_sgd_transformer_block_5_multi_head_attention_5_attention_output_bias_momentumIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72¸
AssignVariableOp_72AssignVariableOp0assignvariableop_72_sgd_dense_16_kernel_momentumIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73¶
AssignVariableOp_73AssignVariableOp.assignvariableop_73_sgd_dense_16_bias_momentumIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74¸
AssignVariableOp_74AssignVariableOp0assignvariableop_74_sgd_dense_17_kernel_momentumIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75¶
AssignVariableOp_75AssignVariableOp.assignvariableop_75_sgd_dense_17_bias_momentumIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76Ù
AssignVariableOp_76AssignVariableOpQassignvariableop_76_sgd_transformer_block_5_layer_normalization_10_gamma_momentumIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77Ø
AssignVariableOp_77AssignVariableOpPassignvariableop_77_sgd_transformer_block_5_layer_normalization_10_beta_momentumIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78Ù
AssignVariableOp_78AssignVariableOpQassignvariableop_78_sgd_transformer_block_5_layer_normalization_11_gamma_momentumIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79Ø
AssignVariableOp_79AssignVariableOpPassignvariableop_79_sgd_transformer_block_5_layer_normalization_11_beta_momentumIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_799
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¾
Identity_80Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_80±
Identity_81IdentityIdentity_80:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_81"#
identity_81Identity_81:output:0*×
_input_shapesÅ
Â: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ÿ
â
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_422065

inputsF
Bmulti_head_attention_5_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_5_query_add_readvariableop_resourceD
@multi_head_attention_5_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_5_key_add_readvariableop_resourceF
Bmulti_head_attention_5_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_5_value_add_readvariableop_resourceQ
Mmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_5_attention_output_add_readvariableop_resource@
<layer_normalization_10_batchnorm_mul_readvariableop_resource<
8layer_normalization_10_batchnorm_readvariableop_resource;
7sequential_5_dense_16_tensordot_readvariableop_resource9
5sequential_5_dense_16_biasadd_readvariableop_resource;
7sequential_5_dense_17_tensordot_readvariableop_resource9
5sequential_5_dense_17_biasadd_readvariableop_resource@
<layer_normalization_11_batchnorm_mul_readvariableop_resource<
8layer_normalization_11_batchnorm_readvariableop_resource
identity¢/layer_normalization_10/batchnorm/ReadVariableOp¢3layer_normalization_10/batchnorm/mul/ReadVariableOp¢/layer_normalization_11/batchnorm/ReadVariableOp¢3layer_normalization_11/batchnorm/mul/ReadVariableOp¢:multi_head_attention_5/attention_output/add/ReadVariableOp¢Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_5/key/add/ReadVariableOp¢7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_5/query/add/ReadVariableOp¢9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_5/value/add/ReadVariableOp¢9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp¢,sequential_5/dense_16/BiasAdd/ReadVariableOp¢.sequential_5/dense_16/Tensordot/ReadVariableOp¢,sequential_5/dense_17/BiasAdd/ReadVariableOp¢.sequential_5/dense_17/Tensordot/ReadVariableOpý
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_5/query/einsum/EinsumEinsuminputsAmulti_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_5/query/einsum/EinsumÛ
/multi_head_attention_5/query/add/ReadVariableOpReadVariableOp8multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/query/add/ReadVariableOpõ
 multi_head_attention_5/query/addAddV23multi_head_attention_5/query/einsum/Einsum:output:07multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_5/query/add÷
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_5/key/einsum/EinsumEinsuminputs?multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2*
(multi_head_attention_5/key/einsum/EinsumÕ
-multi_head_attention_5/key/add/ReadVariableOpReadVariableOp6multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_5/key/add/ReadVariableOpí
multi_head_attention_5/key/addAddV21multi_head_attention_5/key/einsum/Einsum:output:05multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
multi_head_attention_5/key/addý
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_5/value/einsum/EinsumEinsuminputsAmulti_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_5/value/einsum/EinsumÛ
/multi_head_attention_5/value/add/ReadVariableOpReadVariableOp8multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/value/add/ReadVariableOpõ
 multi_head_attention_5/value/addAddV23multi_head_attention_5/value/einsum/Einsum:output:07multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_5/value/add
multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_5/Mul/yÆ
multi_head_attention_5/MulMul$multi_head_attention_5/query/add:z:0%multi_head_attention_5/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_5/Mulü
$multi_head_attention_5/einsum/EinsumEinsum"multi_head_attention_5/key/add:z:0multi_head_attention_5/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2&
$multi_head_attention_5/einsum/EinsumÄ
&multi_head_attention_5/softmax/SoftmaxSoftmax-multi_head_attention_5/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2(
&multi_head_attention_5/softmax/Softmax¡
,multi_head_attention_5/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,multi_head_attention_5/dropout/dropout/Const
*multi_head_attention_5/dropout/dropout/MulMul0multi_head_attention_5/softmax/Softmax:softmax:05multi_head_attention_5/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2,
*multi_head_attention_5/dropout/dropout/Mul¼
,multi_head_attention_5/dropout/dropout/ShapeShape0multi_head_attention_5/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_5/dropout/dropout/Shape
Cmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_5/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
dtype02E
Cmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniform³
5multi_head_attention_5/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_5/dropout/dropout/GreaterEqual/yÂ
3multi_head_attention_5/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_5/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##25
3multi_head_attention_5/dropout/dropout/GreaterEqualä
+multi_head_attention_5/dropout/dropout/CastCast7multi_head_attention_5/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2-
+multi_head_attention_5/dropout/dropout/Castþ
,multi_head_attention_5/dropout/dropout/Mul_1Mul.multi_head_attention_5/dropout/dropout/Mul:z:0/multi_head_attention_5/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2.
,multi_head_attention_5/dropout/dropout/Mul_1
&multi_head_attention_5/einsum_1/EinsumEinsum0multi_head_attention_5/dropout/dropout/Mul_1:z:0$multi_head_attention_5/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2(
&multi_head_attention_5/einsum_1/Einsum
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_5/attention_output/einsum/EinsumEinsum/multi_head_attention_5/einsum_1/Einsum:output:0Lmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe27
5multi_head_attention_5/attention_output/einsum/Einsumø
:multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_5/attention_output/add/ReadVariableOp
+multi_head_attention_5/attention_output/addAddV2>multi_head_attention_5/attention_output/einsum/Einsum:output:0Bmulti_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+multi_head_attention_5/attention_output/addy
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_14/dropout/ConstÁ
dropout_14/dropout/MulMul/multi_head_attention_5/attention_output/add:z:0!dropout_14/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_14/dropout/Mul
dropout_14/dropout/ShapeShape/multi_head_attention_5/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_14/dropout/ShapeÙ
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype021
/dropout_14/dropout/random_uniform/RandomUniform
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_14/dropout/GreaterEqual/yî
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
dropout_14/dropout/GreaterEqual¤
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_14/dropout/Castª
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_14/dropout/Mul_1o
addAddV2inputsdropout_14/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¸
5layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_10/moments/mean/reduction_indicesâ
#layer_normalization_10/moments/meanMeanadd:z:0>layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_10/moments/meanÎ
+layer_normalization_10/moments/StopGradientStopGradient,layer_normalization_10/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_10/moments/StopGradientî
0layer_normalization_10/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_10/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_10/moments/SquaredDifferenceÀ
9layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_10/moments/variance/reduction_indices
'layer_normalization_10/moments/varianceMean4layer_normalization_10/moments/SquaredDifference:z:0Blayer_normalization_10/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_10/moments/variance
&layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_10/batchnorm/add/yî
$layer_normalization_10/batchnorm/addAddV20layer_normalization_10/moments/variance:output:0/layer_normalization_10/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_10/batchnorm/add¹
&layer_normalization_10/batchnorm/RsqrtRsqrt(layer_normalization_10/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_10/batchnorm/Rsqrtã
3layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_10/batchnorm/mul/ReadVariableOpò
$layer_normalization_10/batchnorm/mulMul*layer_normalization_10/batchnorm/Rsqrt:y:0;layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_10/batchnorm/mulÀ
&layer_normalization_10/batchnorm/mul_1Muladd:z:0(layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_10/batchnorm/mul_1å
&layer_normalization_10/batchnorm/mul_2Mul,layer_normalization_10/moments/mean:output:0(layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_10/batchnorm/mul_2×
/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_10/batchnorm/ReadVariableOpî
$layer_normalization_10/batchnorm/subSub7layer_normalization_10/batchnorm/ReadVariableOp:value:0*layer_normalization_10/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_10/batchnorm/subå
&layer_normalization_10/batchnorm/add_1AddV2*layer_normalization_10/batchnorm/mul_1:z:0(layer_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_10/batchnorm/add_1Ø
.sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_5/dense_16/Tensordot/ReadVariableOp
$sequential_5/dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_5/dense_16/Tensordot/axes
$sequential_5/dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_5/dense_16/Tensordot/free¨
%sequential_5/dense_16/Tensordot/ShapeShape*layer_normalization_10/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_5/dense_16/Tensordot/Shape 
-sequential_5/dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_16/Tensordot/GatherV2/axis¿
(sequential_5/dense_16/Tensordot/GatherV2GatherV2.sequential_5/dense_16/Tensordot/Shape:output:0-sequential_5/dense_16/Tensordot/free:output:06sequential_5/dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_5/dense_16/Tensordot/GatherV2¤
/sequential_5/dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_5/dense_16/Tensordot/GatherV2_1/axisÅ
*sequential_5/dense_16/Tensordot/GatherV2_1GatherV2.sequential_5/dense_16/Tensordot/Shape:output:0-sequential_5/dense_16/Tensordot/axes:output:08sequential_5/dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_5/dense_16/Tensordot/GatherV2_1
%sequential_5/dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_5/dense_16/Tensordot/ConstØ
$sequential_5/dense_16/Tensordot/ProdProd1sequential_5/dense_16/Tensordot/GatherV2:output:0.sequential_5/dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_5/dense_16/Tensordot/Prod
'sequential_5/dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_16/Tensordot/Const_1à
&sequential_5/dense_16/Tensordot/Prod_1Prod3sequential_5/dense_16/Tensordot/GatherV2_1:output:00sequential_5/dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_5/dense_16/Tensordot/Prod_1
+sequential_5/dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_5/dense_16/Tensordot/concat/axis
&sequential_5/dense_16/Tensordot/concatConcatV2-sequential_5/dense_16/Tensordot/free:output:0-sequential_5/dense_16/Tensordot/axes:output:04sequential_5/dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/dense_16/Tensordot/concatä
%sequential_5/dense_16/Tensordot/stackPack-sequential_5/dense_16/Tensordot/Prod:output:0/sequential_5/dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/dense_16/Tensordot/stackö
)sequential_5/dense_16/Tensordot/transpose	Transpose*layer_normalization_10/batchnorm/add_1:z:0/sequential_5/dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2+
)sequential_5/dense_16/Tensordot/transpose÷
'sequential_5/dense_16/Tensordot/ReshapeReshape-sequential_5/dense_16/Tensordot/transpose:y:0.sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_5/dense_16/Tensordot/Reshapeö
&sequential_5/dense_16/Tensordot/MatMulMatMul0sequential_5/dense_16/Tensordot/Reshape:output:06sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential_5/dense_16/Tensordot/MatMul
'sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_5/dense_16/Tensordot/Const_2 
-sequential_5/dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_16/Tensordot/concat_1/axis«
(sequential_5/dense_16/Tensordot/concat_1ConcatV21sequential_5/dense_16/Tensordot/GatherV2:output:00sequential_5/dense_16/Tensordot/Const_2:output:06sequential_5/dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_5/dense_16/Tensordot/concat_1è
sequential_5/dense_16/TensordotReshape0sequential_5/dense_16/Tensordot/MatMul:product:01sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2!
sequential_5/dense_16/TensordotÎ
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOpß
sequential_5/dense_16/BiasAddBiasAdd(sequential_5/dense_16/Tensordot:output:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_5/dense_16/BiasAdd
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_5/dense_16/ReluØ
.sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_5/dense_17/Tensordot/ReadVariableOp
$sequential_5/dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_5/dense_17/Tensordot/axes
$sequential_5/dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_5/dense_17/Tensordot/free¦
%sequential_5/dense_17/Tensordot/ShapeShape(sequential_5/dense_16/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_5/dense_17/Tensordot/Shape 
-sequential_5/dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_17/Tensordot/GatherV2/axis¿
(sequential_5/dense_17/Tensordot/GatherV2GatherV2.sequential_5/dense_17/Tensordot/Shape:output:0-sequential_5/dense_17/Tensordot/free:output:06sequential_5/dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_5/dense_17/Tensordot/GatherV2¤
/sequential_5/dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_5/dense_17/Tensordot/GatherV2_1/axisÅ
*sequential_5/dense_17/Tensordot/GatherV2_1GatherV2.sequential_5/dense_17/Tensordot/Shape:output:0-sequential_5/dense_17/Tensordot/axes:output:08sequential_5/dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_5/dense_17/Tensordot/GatherV2_1
%sequential_5/dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_5/dense_17/Tensordot/ConstØ
$sequential_5/dense_17/Tensordot/ProdProd1sequential_5/dense_17/Tensordot/GatherV2:output:0.sequential_5/dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_5/dense_17/Tensordot/Prod
'sequential_5/dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_17/Tensordot/Const_1à
&sequential_5/dense_17/Tensordot/Prod_1Prod3sequential_5/dense_17/Tensordot/GatherV2_1:output:00sequential_5/dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_5/dense_17/Tensordot/Prod_1
+sequential_5/dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_5/dense_17/Tensordot/concat/axis
&sequential_5/dense_17/Tensordot/concatConcatV2-sequential_5/dense_17/Tensordot/free:output:0-sequential_5/dense_17/Tensordot/axes:output:04sequential_5/dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/dense_17/Tensordot/concatä
%sequential_5/dense_17/Tensordot/stackPack-sequential_5/dense_17/Tensordot/Prod:output:0/sequential_5/dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/dense_17/Tensordot/stackô
)sequential_5/dense_17/Tensordot/transpose	Transpose(sequential_5/dense_16/Relu:activations:0/sequential_5/dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2+
)sequential_5/dense_17/Tensordot/transpose÷
'sequential_5/dense_17/Tensordot/ReshapeReshape-sequential_5/dense_17/Tensordot/transpose:y:0.sequential_5/dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_5/dense_17/Tensordot/Reshapeö
&sequential_5/dense_17/Tensordot/MatMulMatMul0sequential_5/dense_17/Tensordot/Reshape:output:06sequential_5/dense_17/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_5/dense_17/Tensordot/MatMul
'sequential_5/dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_17/Tensordot/Const_2 
-sequential_5/dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_17/Tensordot/concat_1/axis«
(sequential_5/dense_17/Tensordot/concat_1ConcatV21sequential_5/dense_17/Tensordot/GatherV2:output:00sequential_5/dense_17/Tensordot/Const_2:output:06sequential_5/dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_5/dense_17/Tensordot/concat_1è
sequential_5/dense_17/TensordotReshape0sequential_5/dense_17/Tensordot/MatMul:product:01sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
sequential_5/dense_17/TensordotÎ
,sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_17/BiasAdd/ReadVariableOpß
sequential_5/dense_17/BiasAddBiasAdd(sequential_5/dense_17/Tensordot:output:04sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
sequential_5/dense_17/BiasAddy
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_15/dropout/Const¸
dropout_15/dropout/MulMul&sequential_5/dense_17/BiasAdd:output:0!dropout_15/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_15/dropout/Mul
dropout_15/dropout/ShapeShape&sequential_5/dense_17/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_15/dropout/ShapeÙ
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype021
/dropout_15/dropout/random_uniform/RandomUniform
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_15/dropout/GreaterEqual/yî
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
dropout_15/dropout/GreaterEqual¤
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_15/dropout/Castª
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_15/dropout/Mul_1
add_1AddV2*layer_normalization_10/batchnorm/add_1:z:0dropout_15/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¸
5layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_11/moments/mean/reduction_indicesä
#layer_normalization_11/moments/meanMean	add_1:z:0>layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_11/moments/meanÎ
+layer_normalization_11/moments/StopGradientStopGradient,layer_normalization_11/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_11/moments/StopGradientð
0layer_normalization_11/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_11/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_11/moments/SquaredDifferenceÀ
9layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_11/moments/variance/reduction_indices
'layer_normalization_11/moments/varianceMean4layer_normalization_11/moments/SquaredDifference:z:0Blayer_normalization_11/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_11/moments/variance
&layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_11/batchnorm/add/yî
$layer_normalization_11/batchnorm/addAddV20layer_normalization_11/moments/variance:output:0/layer_normalization_11/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_11/batchnorm/add¹
&layer_normalization_11/batchnorm/RsqrtRsqrt(layer_normalization_11/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_11/batchnorm/Rsqrtã
3layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_11/batchnorm/mul/ReadVariableOpò
$layer_normalization_11/batchnorm/mulMul*layer_normalization_11/batchnorm/Rsqrt:y:0;layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_11/batchnorm/mulÂ
&layer_normalization_11/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_11/batchnorm/mul_1å
&layer_normalization_11/batchnorm/mul_2Mul,layer_normalization_11/moments/mean:output:0(layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_11/batchnorm/mul_2×
/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_11/batchnorm/ReadVariableOpî
$layer_normalization_11/batchnorm/subSub7layer_normalization_11/batchnorm/ReadVariableOp:value:0*layer_normalization_11/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_11/batchnorm/subå
&layer_normalization_11/batchnorm/add_1AddV2*layer_normalization_11/batchnorm/mul_1:z:0(layer_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_11/batchnorm/add_1Ü
IdentityIdentity*layer_normalization_11/batchnorm/add_1:z:00^layer_normalization_10/batchnorm/ReadVariableOp4^layer_normalization_10/batchnorm/mul/ReadVariableOp0^layer_normalization_11/batchnorm/ReadVariableOp4^layer_normalization_11/batchnorm/mul/ReadVariableOp;^multi_head_attention_5/attention_output/add/ReadVariableOpE^multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_5/key/add/ReadVariableOp8^multi_head_attention_5/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/query/add/ReadVariableOp:^multi_head_attention_5/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/value/add/ReadVariableOp:^multi_head_attention_5/value/einsum/Einsum/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp/^sequential_5/dense_16/Tensordot/ReadVariableOp-^sequential_5/dense_17/BiasAdd/ReadVariableOp/^sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2b
/layer_normalization_10/batchnorm/ReadVariableOp/layer_normalization_10/batchnorm/ReadVariableOp2j
3layer_normalization_10/batchnorm/mul/ReadVariableOp3layer_normalization_10/batchnorm/mul/ReadVariableOp2b
/layer_normalization_11/batchnorm/ReadVariableOp/layer_normalization_11/batchnorm/ReadVariableOp2j
3layer_normalization_11/batchnorm/mul/ReadVariableOp3layer_normalization_11/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_5/attention_output/add/ReadVariableOp:multi_head_attention_5/attention_output/add/ReadVariableOp2
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_5/key/add/ReadVariableOp-multi_head_attention_5/key/add/ReadVariableOp2r
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/query/add/ReadVariableOp/multi_head_attention_5/query/add/ReadVariableOp2v
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/value/add/ReadVariableOp/multi_head_attention_5/value/add/ReadVariableOp2v
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2`
.sequential_5/dense_16/Tensordot/ReadVariableOp.sequential_5/dense_16/Tensordot/ReadVariableOp2\
,sequential_5/dense_17/BiasAdd/ReadVariableOp,sequential_5/dense_17/BiasAdd/ReadVariableOp2`
.sequential_5/dense_17/Tensordot/ReadVariableOp.sequential_5/dense_17/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs

e
F__inference_dropout_17_layer_call_and_return_conditional_losses_424813

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
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
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
J
¯
H__inference_sequential_5_layer_call_and_return_conditional_losses_424904

inputs.
*dense_16_tensordot_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource.
*dense_17_tensordot_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
identity¢dense_16/BiasAdd/ReadVariableOp¢!dense_16/Tensordot/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢!dense_17/Tensordot/ReadVariableOp±
!dense_16/Tensordot/ReadVariableOpReadVariableOp*dense_16_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02#
!dense_16/Tensordot/ReadVariableOp|
dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_16/Tensordot/axes
dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_16/Tensordot/freej
dense_16/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_16/Tensordot/Shape
 dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_16/Tensordot/GatherV2/axisþ
dense_16/Tensordot/GatherV2GatherV2!dense_16/Tensordot/Shape:output:0 dense_16/Tensordot/free:output:0)dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_16/Tensordot/GatherV2
"dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_16/Tensordot/GatherV2_1/axis
dense_16/Tensordot/GatherV2_1GatherV2!dense_16/Tensordot/Shape:output:0 dense_16/Tensordot/axes:output:0+dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_16/Tensordot/GatherV2_1~
dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_16/Tensordot/Const¤
dense_16/Tensordot/ProdProd$dense_16/Tensordot/GatherV2:output:0!dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_16/Tensordot/Prod
dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_16/Tensordot/Const_1¬
dense_16/Tensordot/Prod_1Prod&dense_16/Tensordot/GatherV2_1:output:0#dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_16/Tensordot/Prod_1
dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_16/Tensordot/concat/axisÝ
dense_16/Tensordot/concatConcatV2 dense_16/Tensordot/free:output:0 dense_16/Tensordot/axes:output:0'dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_16/Tensordot/concat°
dense_16/Tensordot/stackPack dense_16/Tensordot/Prod:output:0"dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_16/Tensordot/stack«
dense_16/Tensordot/transpose	Transposeinputs"dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_16/Tensordot/transposeÃ
dense_16/Tensordot/ReshapeReshape dense_16/Tensordot/transpose:y:0!dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_16/Tensordot/ReshapeÂ
dense_16/Tensordot/MatMulMatMul#dense_16/Tensordot/Reshape:output:0)dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_16/Tensordot/MatMul
dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_16/Tensordot/Const_2
 dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_16/Tensordot/concat_1/axisê
dense_16/Tensordot/concat_1ConcatV2$dense_16/Tensordot/GatherV2:output:0#dense_16/Tensordot/Const_2:output:0)dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_16/Tensordot/concat_1´
dense_16/TensordotReshape#dense_16/Tensordot/MatMul:product:0$dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_16/Tensordot§
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_16/BiasAdd/ReadVariableOp«
dense_16/BiasAddBiasAdddense_16/Tensordot:output:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_16/BiasAddw
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_16/Relu±
!dense_17/Tensordot/ReadVariableOpReadVariableOp*dense_17_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02#
!dense_17/Tensordot/ReadVariableOp|
dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_17/Tensordot/axes
dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_17/Tensordot/free
dense_17/Tensordot/ShapeShapedense_16/Relu:activations:0*
T0*
_output_shapes
:2
dense_17/Tensordot/Shape
 dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_17/Tensordot/GatherV2/axisþ
dense_17/Tensordot/GatherV2GatherV2!dense_17/Tensordot/Shape:output:0 dense_17/Tensordot/free:output:0)dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_17/Tensordot/GatherV2
"dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_17/Tensordot/GatherV2_1/axis
dense_17/Tensordot/GatherV2_1GatherV2!dense_17/Tensordot/Shape:output:0 dense_17/Tensordot/axes:output:0+dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_17/Tensordot/GatherV2_1~
dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_17/Tensordot/Const¤
dense_17/Tensordot/ProdProd$dense_17/Tensordot/GatherV2:output:0!dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_17/Tensordot/Prod
dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_17/Tensordot/Const_1¬
dense_17/Tensordot/Prod_1Prod&dense_17/Tensordot/GatherV2_1:output:0#dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_17/Tensordot/Prod_1
dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_17/Tensordot/concat/axisÝ
dense_17/Tensordot/concatConcatV2 dense_17/Tensordot/free:output:0 dense_17/Tensordot/axes:output:0'dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_17/Tensordot/concat°
dense_17/Tensordot/stackPack dense_17/Tensordot/Prod:output:0"dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_17/Tensordot/stackÀ
dense_17/Tensordot/transpose	Transposedense_16/Relu:activations:0"dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_17/Tensordot/transposeÃ
dense_17/Tensordot/ReshapeReshape dense_17/Tensordot/transpose:y:0!dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_17/Tensordot/ReshapeÂ
dense_17/Tensordot/MatMulMatMul#dense_17/Tensordot/Reshape:output:0)dense_17/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_17/Tensordot/MatMul
dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_17/Tensordot/Const_2
 dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_17/Tensordot/concat_1/axisê
dense_17/Tensordot/concat_1ConcatV2$dense_17/Tensordot/GatherV2:output:0#dense_17/Tensordot/Const_2:output:0)dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_17/Tensordot/concat_1´
dense_17/TensordotReshape#dense_17/Tensordot/MatMul:product:0$dense_17/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_17/Tensordot§
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_17/BiasAdd/ReadVariableOp«
dense_17/BiasAddBiasAdddense_17/Tensordot:output:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_17/BiasAddý
IdentityIdentitydense_17/BiasAdd:output:0 ^dense_16/BiasAdd/ReadVariableOp"^dense_16/Tensordot/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp"^dense_17/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2F
!dense_16/Tensordot/ReadVariableOp!dense_16/Tensordot/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2F
!dense_17/Tensordot/ReadVariableOp!dense_17/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
ÿ
â
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_424427

inputsF
Bmulti_head_attention_5_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_5_query_add_readvariableop_resourceD
@multi_head_attention_5_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_5_key_add_readvariableop_resourceF
Bmulti_head_attention_5_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_5_value_add_readvariableop_resourceQ
Mmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_5_attention_output_add_readvariableop_resource@
<layer_normalization_10_batchnorm_mul_readvariableop_resource<
8layer_normalization_10_batchnorm_readvariableop_resource;
7sequential_5_dense_16_tensordot_readvariableop_resource9
5sequential_5_dense_16_biasadd_readvariableop_resource;
7sequential_5_dense_17_tensordot_readvariableop_resource9
5sequential_5_dense_17_biasadd_readvariableop_resource@
<layer_normalization_11_batchnorm_mul_readvariableop_resource<
8layer_normalization_11_batchnorm_readvariableop_resource
identity¢/layer_normalization_10/batchnorm/ReadVariableOp¢3layer_normalization_10/batchnorm/mul/ReadVariableOp¢/layer_normalization_11/batchnorm/ReadVariableOp¢3layer_normalization_11/batchnorm/mul/ReadVariableOp¢:multi_head_attention_5/attention_output/add/ReadVariableOp¢Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_5/key/add/ReadVariableOp¢7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_5/query/add/ReadVariableOp¢9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_5/value/add/ReadVariableOp¢9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp¢,sequential_5/dense_16/BiasAdd/ReadVariableOp¢.sequential_5/dense_16/Tensordot/ReadVariableOp¢,sequential_5/dense_17/BiasAdd/ReadVariableOp¢.sequential_5/dense_17/Tensordot/ReadVariableOpý
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_5/query/einsum/EinsumEinsuminputsAmulti_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_5/query/einsum/EinsumÛ
/multi_head_attention_5/query/add/ReadVariableOpReadVariableOp8multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/query/add/ReadVariableOpõ
 multi_head_attention_5/query/addAddV23multi_head_attention_5/query/einsum/Einsum:output:07multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_5/query/add÷
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_5/key/einsum/EinsumEinsuminputs?multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2*
(multi_head_attention_5/key/einsum/EinsumÕ
-multi_head_attention_5/key/add/ReadVariableOpReadVariableOp6multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_5/key/add/ReadVariableOpí
multi_head_attention_5/key/addAddV21multi_head_attention_5/key/einsum/Einsum:output:05multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
multi_head_attention_5/key/addý
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_5/value/einsum/EinsumEinsuminputsAmulti_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_5/value/einsum/EinsumÛ
/multi_head_attention_5/value/add/ReadVariableOpReadVariableOp8multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/value/add/ReadVariableOpõ
 multi_head_attention_5/value/addAddV23multi_head_attention_5/value/einsum/Einsum:output:07multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_5/value/add
multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_5/Mul/yÆ
multi_head_attention_5/MulMul$multi_head_attention_5/query/add:z:0%multi_head_attention_5/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_5/Mulü
$multi_head_attention_5/einsum/EinsumEinsum"multi_head_attention_5/key/add:z:0multi_head_attention_5/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2&
$multi_head_attention_5/einsum/EinsumÄ
&multi_head_attention_5/softmax/SoftmaxSoftmax-multi_head_attention_5/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2(
&multi_head_attention_5/softmax/Softmax¡
,multi_head_attention_5/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,multi_head_attention_5/dropout/dropout/Const
*multi_head_attention_5/dropout/dropout/MulMul0multi_head_attention_5/softmax/Softmax:softmax:05multi_head_attention_5/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2,
*multi_head_attention_5/dropout/dropout/Mul¼
,multi_head_attention_5/dropout/dropout/ShapeShape0multi_head_attention_5/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_5/dropout/dropout/Shape
Cmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_5/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
dtype02E
Cmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniform³
5multi_head_attention_5/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_5/dropout/dropout/GreaterEqual/yÂ
3multi_head_attention_5/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_5/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##25
3multi_head_attention_5/dropout/dropout/GreaterEqualä
+multi_head_attention_5/dropout/dropout/CastCast7multi_head_attention_5/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2-
+multi_head_attention_5/dropout/dropout/Castþ
,multi_head_attention_5/dropout/dropout/Mul_1Mul.multi_head_attention_5/dropout/dropout/Mul:z:0/multi_head_attention_5/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2.
,multi_head_attention_5/dropout/dropout/Mul_1
&multi_head_attention_5/einsum_1/EinsumEinsum0multi_head_attention_5/dropout/dropout/Mul_1:z:0$multi_head_attention_5/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2(
&multi_head_attention_5/einsum_1/Einsum
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_5/attention_output/einsum/EinsumEinsum/multi_head_attention_5/einsum_1/Einsum:output:0Lmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe27
5multi_head_attention_5/attention_output/einsum/Einsumø
:multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_5/attention_output/add/ReadVariableOp
+multi_head_attention_5/attention_output/addAddV2>multi_head_attention_5/attention_output/einsum/Einsum:output:0Bmulti_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+multi_head_attention_5/attention_output/addy
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_14/dropout/ConstÁ
dropout_14/dropout/MulMul/multi_head_attention_5/attention_output/add:z:0!dropout_14/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_14/dropout/Mul
dropout_14/dropout/ShapeShape/multi_head_attention_5/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_14/dropout/ShapeÙ
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype021
/dropout_14/dropout/random_uniform/RandomUniform
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_14/dropout/GreaterEqual/yî
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
dropout_14/dropout/GreaterEqual¤
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_14/dropout/Castª
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_14/dropout/Mul_1o
addAddV2inputsdropout_14/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¸
5layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_10/moments/mean/reduction_indicesâ
#layer_normalization_10/moments/meanMeanadd:z:0>layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_10/moments/meanÎ
+layer_normalization_10/moments/StopGradientStopGradient,layer_normalization_10/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_10/moments/StopGradientî
0layer_normalization_10/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_10/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_10/moments/SquaredDifferenceÀ
9layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_10/moments/variance/reduction_indices
'layer_normalization_10/moments/varianceMean4layer_normalization_10/moments/SquaredDifference:z:0Blayer_normalization_10/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_10/moments/variance
&layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_10/batchnorm/add/yî
$layer_normalization_10/batchnorm/addAddV20layer_normalization_10/moments/variance:output:0/layer_normalization_10/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_10/batchnorm/add¹
&layer_normalization_10/batchnorm/RsqrtRsqrt(layer_normalization_10/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_10/batchnorm/Rsqrtã
3layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_10/batchnorm/mul/ReadVariableOpò
$layer_normalization_10/batchnorm/mulMul*layer_normalization_10/batchnorm/Rsqrt:y:0;layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_10/batchnorm/mulÀ
&layer_normalization_10/batchnorm/mul_1Muladd:z:0(layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_10/batchnorm/mul_1å
&layer_normalization_10/batchnorm/mul_2Mul,layer_normalization_10/moments/mean:output:0(layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_10/batchnorm/mul_2×
/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_10/batchnorm/ReadVariableOpî
$layer_normalization_10/batchnorm/subSub7layer_normalization_10/batchnorm/ReadVariableOp:value:0*layer_normalization_10/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_10/batchnorm/subå
&layer_normalization_10/batchnorm/add_1AddV2*layer_normalization_10/batchnorm/mul_1:z:0(layer_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_10/batchnorm/add_1Ø
.sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_5/dense_16/Tensordot/ReadVariableOp
$sequential_5/dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_5/dense_16/Tensordot/axes
$sequential_5/dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_5/dense_16/Tensordot/free¨
%sequential_5/dense_16/Tensordot/ShapeShape*layer_normalization_10/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_5/dense_16/Tensordot/Shape 
-sequential_5/dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_16/Tensordot/GatherV2/axis¿
(sequential_5/dense_16/Tensordot/GatherV2GatherV2.sequential_5/dense_16/Tensordot/Shape:output:0-sequential_5/dense_16/Tensordot/free:output:06sequential_5/dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_5/dense_16/Tensordot/GatherV2¤
/sequential_5/dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_5/dense_16/Tensordot/GatherV2_1/axisÅ
*sequential_5/dense_16/Tensordot/GatherV2_1GatherV2.sequential_5/dense_16/Tensordot/Shape:output:0-sequential_5/dense_16/Tensordot/axes:output:08sequential_5/dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_5/dense_16/Tensordot/GatherV2_1
%sequential_5/dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_5/dense_16/Tensordot/ConstØ
$sequential_5/dense_16/Tensordot/ProdProd1sequential_5/dense_16/Tensordot/GatherV2:output:0.sequential_5/dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_5/dense_16/Tensordot/Prod
'sequential_5/dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_16/Tensordot/Const_1à
&sequential_5/dense_16/Tensordot/Prod_1Prod3sequential_5/dense_16/Tensordot/GatherV2_1:output:00sequential_5/dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_5/dense_16/Tensordot/Prod_1
+sequential_5/dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_5/dense_16/Tensordot/concat/axis
&sequential_5/dense_16/Tensordot/concatConcatV2-sequential_5/dense_16/Tensordot/free:output:0-sequential_5/dense_16/Tensordot/axes:output:04sequential_5/dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/dense_16/Tensordot/concatä
%sequential_5/dense_16/Tensordot/stackPack-sequential_5/dense_16/Tensordot/Prod:output:0/sequential_5/dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/dense_16/Tensordot/stackö
)sequential_5/dense_16/Tensordot/transpose	Transpose*layer_normalization_10/batchnorm/add_1:z:0/sequential_5/dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2+
)sequential_5/dense_16/Tensordot/transpose÷
'sequential_5/dense_16/Tensordot/ReshapeReshape-sequential_5/dense_16/Tensordot/transpose:y:0.sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_5/dense_16/Tensordot/Reshapeö
&sequential_5/dense_16/Tensordot/MatMulMatMul0sequential_5/dense_16/Tensordot/Reshape:output:06sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential_5/dense_16/Tensordot/MatMul
'sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_5/dense_16/Tensordot/Const_2 
-sequential_5/dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_16/Tensordot/concat_1/axis«
(sequential_5/dense_16/Tensordot/concat_1ConcatV21sequential_5/dense_16/Tensordot/GatherV2:output:00sequential_5/dense_16/Tensordot/Const_2:output:06sequential_5/dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_5/dense_16/Tensordot/concat_1è
sequential_5/dense_16/TensordotReshape0sequential_5/dense_16/Tensordot/MatMul:product:01sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2!
sequential_5/dense_16/TensordotÎ
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOpß
sequential_5/dense_16/BiasAddBiasAdd(sequential_5/dense_16/Tensordot:output:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_5/dense_16/BiasAdd
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_5/dense_16/ReluØ
.sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_5/dense_17/Tensordot/ReadVariableOp
$sequential_5/dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_5/dense_17/Tensordot/axes
$sequential_5/dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_5/dense_17/Tensordot/free¦
%sequential_5/dense_17/Tensordot/ShapeShape(sequential_5/dense_16/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_5/dense_17/Tensordot/Shape 
-sequential_5/dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_17/Tensordot/GatherV2/axis¿
(sequential_5/dense_17/Tensordot/GatherV2GatherV2.sequential_5/dense_17/Tensordot/Shape:output:0-sequential_5/dense_17/Tensordot/free:output:06sequential_5/dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_5/dense_17/Tensordot/GatherV2¤
/sequential_5/dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_5/dense_17/Tensordot/GatherV2_1/axisÅ
*sequential_5/dense_17/Tensordot/GatherV2_1GatherV2.sequential_5/dense_17/Tensordot/Shape:output:0-sequential_5/dense_17/Tensordot/axes:output:08sequential_5/dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_5/dense_17/Tensordot/GatherV2_1
%sequential_5/dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_5/dense_17/Tensordot/ConstØ
$sequential_5/dense_17/Tensordot/ProdProd1sequential_5/dense_17/Tensordot/GatherV2:output:0.sequential_5/dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_5/dense_17/Tensordot/Prod
'sequential_5/dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_17/Tensordot/Const_1à
&sequential_5/dense_17/Tensordot/Prod_1Prod3sequential_5/dense_17/Tensordot/GatherV2_1:output:00sequential_5/dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_5/dense_17/Tensordot/Prod_1
+sequential_5/dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_5/dense_17/Tensordot/concat/axis
&sequential_5/dense_17/Tensordot/concatConcatV2-sequential_5/dense_17/Tensordot/free:output:0-sequential_5/dense_17/Tensordot/axes:output:04sequential_5/dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/dense_17/Tensordot/concatä
%sequential_5/dense_17/Tensordot/stackPack-sequential_5/dense_17/Tensordot/Prod:output:0/sequential_5/dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/dense_17/Tensordot/stackô
)sequential_5/dense_17/Tensordot/transpose	Transpose(sequential_5/dense_16/Relu:activations:0/sequential_5/dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2+
)sequential_5/dense_17/Tensordot/transpose÷
'sequential_5/dense_17/Tensordot/ReshapeReshape-sequential_5/dense_17/Tensordot/transpose:y:0.sequential_5/dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_5/dense_17/Tensordot/Reshapeö
&sequential_5/dense_17/Tensordot/MatMulMatMul0sequential_5/dense_17/Tensordot/Reshape:output:06sequential_5/dense_17/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_5/dense_17/Tensordot/MatMul
'sequential_5/dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_17/Tensordot/Const_2 
-sequential_5/dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_17/Tensordot/concat_1/axis«
(sequential_5/dense_17/Tensordot/concat_1ConcatV21sequential_5/dense_17/Tensordot/GatherV2:output:00sequential_5/dense_17/Tensordot/Const_2:output:06sequential_5/dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_5/dense_17/Tensordot/concat_1è
sequential_5/dense_17/TensordotReshape0sequential_5/dense_17/Tensordot/MatMul:product:01sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
sequential_5/dense_17/TensordotÎ
,sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_17/BiasAdd/ReadVariableOpß
sequential_5/dense_17/BiasAddBiasAdd(sequential_5/dense_17/Tensordot:output:04sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
sequential_5/dense_17/BiasAddy
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_15/dropout/Const¸
dropout_15/dropout/MulMul&sequential_5/dense_17/BiasAdd:output:0!dropout_15/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_15/dropout/Mul
dropout_15/dropout/ShapeShape&sequential_5/dense_17/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_15/dropout/ShapeÙ
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype021
/dropout_15/dropout/random_uniform/RandomUniform
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_15/dropout/GreaterEqual/yî
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
dropout_15/dropout/GreaterEqual¤
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_15/dropout/Castª
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_15/dropout/Mul_1
add_1AddV2*layer_normalization_10/batchnorm/add_1:z:0dropout_15/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¸
5layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_11/moments/mean/reduction_indicesä
#layer_normalization_11/moments/meanMean	add_1:z:0>layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_11/moments/meanÎ
+layer_normalization_11/moments/StopGradientStopGradient,layer_normalization_11/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_11/moments/StopGradientð
0layer_normalization_11/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_11/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_11/moments/SquaredDifferenceÀ
9layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_11/moments/variance/reduction_indices
'layer_normalization_11/moments/varianceMean4layer_normalization_11/moments/SquaredDifference:z:0Blayer_normalization_11/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_11/moments/variance
&layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_11/batchnorm/add/yî
$layer_normalization_11/batchnorm/addAddV20layer_normalization_11/moments/variance:output:0/layer_normalization_11/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_11/batchnorm/add¹
&layer_normalization_11/batchnorm/RsqrtRsqrt(layer_normalization_11/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_11/batchnorm/Rsqrtã
3layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_11/batchnorm/mul/ReadVariableOpò
$layer_normalization_11/batchnorm/mulMul*layer_normalization_11/batchnorm/Rsqrt:y:0;layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_11/batchnorm/mulÂ
&layer_normalization_11/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_11/batchnorm/mul_1å
&layer_normalization_11/batchnorm/mul_2Mul,layer_normalization_11/moments/mean:output:0(layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_11/batchnorm/mul_2×
/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_11/batchnorm/ReadVariableOpî
$layer_normalization_11/batchnorm/subSub7layer_normalization_11/batchnorm/ReadVariableOp:value:0*layer_normalization_11/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_11/batchnorm/subå
&layer_normalization_11/batchnorm/add_1AddV2*layer_normalization_11/batchnorm/mul_1:z:0(layer_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_11/batchnorm/add_1Ü
IdentityIdentity*layer_normalization_11/batchnorm/add_1:z:00^layer_normalization_10/batchnorm/ReadVariableOp4^layer_normalization_10/batchnorm/mul/ReadVariableOp0^layer_normalization_11/batchnorm/ReadVariableOp4^layer_normalization_11/batchnorm/mul/ReadVariableOp;^multi_head_attention_5/attention_output/add/ReadVariableOpE^multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_5/key/add/ReadVariableOp8^multi_head_attention_5/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/query/add/ReadVariableOp:^multi_head_attention_5/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/value/add/ReadVariableOp:^multi_head_attention_5/value/einsum/Einsum/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp/^sequential_5/dense_16/Tensordot/ReadVariableOp-^sequential_5/dense_17/BiasAdd/ReadVariableOp/^sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2b
/layer_normalization_10/batchnorm/ReadVariableOp/layer_normalization_10/batchnorm/ReadVariableOp2j
3layer_normalization_10/batchnorm/mul/ReadVariableOp3layer_normalization_10/batchnorm/mul/ReadVariableOp2b
/layer_normalization_11/batchnorm/ReadVariableOp/layer_normalization_11/batchnorm/ReadVariableOp2j
3layer_normalization_11/batchnorm/mul/ReadVariableOp3layer_normalization_11/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_5/attention_output/add/ReadVariableOp:multi_head_attention_5/attention_output/add/ReadVariableOp2
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_5/key/add/ReadVariableOp-multi_head_attention_5/key/add/ReadVariableOp2r
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/query/add/ReadVariableOp/multi_head_attention_5/query/add/ReadVariableOp2v
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/value/add/ReadVariableOp/multi_head_attention_5/value/add/ReadVariableOp2v
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2`
.sequential_5/dense_16/Tensordot/ReadVariableOp.sequential_5/dense_16/Tensordot/ReadVariableOp2\
,sequential_5/dense_17/BiasAdd/ReadVariableOp,sequential_5/dense_17/BiasAdd/ReadVariableOp2`
.sequential_5/dense_17/Tensordot/ReadVariableOp.sequential_5/dense_17/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
Ô

Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_421597

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼0
È
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_424221

inputs
assignmovingavg_424196
assignmovingavg_1_424202)
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
loc:@AssignMovingAvg/424196*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_424196*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/424196*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/424196*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_424196AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/424196*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/424202*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_424202*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/424202*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/424202*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_424202AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/424202*
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
Ö
â
(__inference_model_2_layer_call_fn_422800
input_5
input_6
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

unknown_36

unknown_37

unknown_38
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_36
unknown_37
unknown_38*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*D
_read_only_resource_inputs&
$"
"#$%&'()*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_4227172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_5:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6
È
©
6__inference_batch_normalization_7_layer_call_fn_424254

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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4218462
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
¥
d
+__inference_dropout_17_layer_call_fn_424823

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
F__inference_dropout_17_layer_call_and_return_conditional_losses_4224622
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
Ü
ä
(__inference_model_2_layer_call_fn_423770
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

unknown_36

unknown_37

unknown_38
identity¢StatefulPartitionedCall
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
unknown_36
unknown_37
unknown_38*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*D
_read_only_resource_inputs&
$"
"#$%&'()*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_4227172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::22
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


Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_423995

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
Ô

Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_424695

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î
©
6__inference_batch_normalization_6_layer_call_fn_424021

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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4211502
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
¥
d
+__inference_dropout_16_layer_call_fn_424776

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
F__inference_dropout_16_layer_call_and_return_conditional_losses_4224052
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
á
~
)__inference_dense_18_layer_call_fn_424754

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
D__inference_dense_18_layer_call_and_return_conditional_losses_4223772
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
 
_user_specified_nameinputs
è

Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_424241

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
ó0
È
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_423975

inputs
assignmovingavg_423950
assignmovingavg_1_423956)
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
loc:@AssignMovingAvg/423950*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_423950*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/423950*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/423950*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_423950AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/423950*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/423956*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_423956*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/423956*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/423956*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_423956AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/423956*
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_421290

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
ß
~
)__inference_dense_20_layer_call_fn_424847

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
D__inference_dense_20_layer_call_and_return_conditional_losses_4224902
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
ì
©
6__inference_batch_normalization_7_layer_call_fn_424172

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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4212572
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
input_51
serving_default_input_5:0ÿÿÿÿÿÿÿÿÿR
;
input_60
serving_default_input_6:0ÿÿÿÿÿÿÿÿÿ<
dense_200
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ì
ïL
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
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer-16
layer_with_weights-8
layer-17
layer-18
layer_with_weights-9
layer-19
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
Â__call__
+Ã&call_and_return_all_conditional_losses
Ä_default_save_signature"G
_tf_keras_networkóF{"class_name": "Functional", "name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}, "name": "token_and_position_embedding_2", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_4", "inbound_nodes": [[["token_and_position_embedding_2", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_6", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_6", "inbound_nodes": [[["conv1d_4", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_5", "inbound_nodes": [[["average_pooling1d_6", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_7", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_7", "inbound_nodes": [[["conv1d_5", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_8", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_8", "inbound_nodes": [[["token_and_position_embedding_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["average_pooling1d_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["average_pooling1d_8", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}], ["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block_5", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["transformer_block_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_2", "inbound_nodes": [[["flatten_2", 0, 0, {}], ["batch_normalization_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["concatenate_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_16", "inbound_nodes": [[["dense_18", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["dropout_16", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_17", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_17", "inbound_nodes": [[["dense_19", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_20", "inbound_nodes": [[["dropout_17", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0], ["input_6", 0, 0]], "output_layers": [["dense_20", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 10500]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 10500]}, {"class_name": "TensorShape", "items": [null, 8]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.0010000000474974513, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
ñ"î
_tf_keras_input_layerÎ{"class_name": "InputLayer", "name": "input_5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}
ç
	token_emb
pos_emb
trainable_variables
	variables
regularization_losses
 	keras_api
Å__call__
+Æ&call_and_return_all_conditional_losses"º
_tf_keras_layer {"class_name": "TokenAndPositionEmbedding", "name": "token_and_position_embedding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
é	

!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses"Â
_tf_keras_layer¨{"class_name": "Conv1D", "name": "conv1d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500, 32]}}

'trainable_variables
(	variables
)regularization_losses
*	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses"ø
_tf_keras_layerÞ{"class_name": "AveragePooling1D", "name": "average_pooling1d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_6", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ç	

+kernel
,bias
-trainable_variables
.	variables
/regularization_losses
0	keras_api
Ë__call__
+Ì&call_and_return_all_conditional_losses"À
_tf_keras_layer¦{"class_name": "Conv1D", "name": "conv1d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 350, 32]}}

1trainable_variables
2	variables
3regularization_losses
4	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses"ø
_tf_keras_layerÞ{"class_name": "AveragePooling1D", "name": "average_pooling1d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_7", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}

5trainable_variables
6	variables
7regularization_losses
8	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "AveragePooling1D", "name": "average_pooling1d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_8", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
¸	
9axis
	:gamma
;beta
<moving_mean
=moving_variance
>trainable_variables
?	variables
@regularization_losses
A	keras_api
Ñ__call__
+Ò&call_and_return_all_conditional_losses"â
_tf_keras_layerÈ{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
¸	
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses"â
_tf_keras_layerÈ{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
³
Ktrainable_variables
L	variables
Mregularization_losses
N	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses"¢
_tf_keras_layer{"class_name": "Add", "name": "add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 35, 32]}, {"class_name": "TensorShape", "items": [null, 35, 32]}]}

Oatt
Pffn
Q
layernorm1
R
layernorm2
Sdropout1
Tdropout2
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses"¥
_tf_keras_layer{"class_name": "TransformerBlock", "name": "transformer_block_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
é"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_6", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}}
è
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api
Ù__call__
+Ú&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
²	
]axis
	^gamma
_beta
`moving_mean
amoving_variance
btrainable_variables
c	variables
dregularization_losses
e	keras_api
Û__call__
+Ü&call_and_return_all_conditional_losses"Ü
_tf_keras_layerÂ{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
Ð
ftrainable_variables
g	variables
hregularization_losses
i	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses"¿
_tf_keras_layer¥{"class_name": "Concatenate", "name": "concatenate_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1120]}, {"class_name": "TensorShape", "items": [null, 8]}]}
ø

jkernel
kbias
ltrainable_variables
m	variables
nregularization_losses
o	keras_api
ß__call__
+à&call_and_return_all_conditional_losses"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "dense_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1128]}}
é
ptrainable_variables
q	variables
rregularization_losses
s	keras_api
á__call__
+â&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
ô

tkernel
ubias
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
ã__call__
+ä&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
é
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
å__call__
+æ&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_17", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
ù

~kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
ç__call__
+è&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
¡

decay
learning_rate
momentum
	iter!momentum "momentum¡+momentum¢,momentum£:momentum¤;momentum¥Cmomentum¦Dmomentum§^momentum¨_momentum©jmomentumªkmomentum«tmomentum¬umomentum­~momentum®momentum¯momentum°momentum±momentum²momentum³momentum´momentumµmomentum¶momentum·momentum¸momentum¹momentumºmomentum»momentum¼momentum½momentum¾momentum¿momentumÀmomentumÁ"
	optimizer
è
0
1
!2
"3
+4
,5
:6
;7
<8
=9
C10
D11
E12
F13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
^30
_31
`32
a33
j34
k35
t36
u37
~38
39"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
0
1
!2
"3
+4
,5
:6
;7
C8
D9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
^26
_27
j28
k29
t30
u31
~32
33"
trackable_list_wrapper
Ó
	variables
metrics
non_trainable_variables
 layer_regularization_losses
regularization_losses
trainable_variables
layer_metrics
layers
Â__call__
Ä_default_save_signature
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
-
éserving_default"
signature_map
µ

embeddings
trainable_variables
 	variables
¡regularization_losses
¢	keras_api
ê__call__
+ë&call_and_return_all_conditional_losses"
_tf_keras_layerõ{"class_name": "Embedding", "name": "embedding_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 5, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500]}}
²

embeddings
£trainable_variables
¤	variables
¥regularization_losses
¦	keras_api
ì__call__
+í&call_and_return_all_conditional_losses"
_tf_keras_layerò{"class_name": "Embedding", "name": "embedding_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 10500, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
§layers
trainable_variables
	variables
¨metrics
regularization_losses
©non_trainable_variables
ªlayer_metrics
 «layer_regularization_losses
Å__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses"
_generic_user_object
%:#  2conv1d_4/kernel
: 2conv1d_4/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¬layers
#trainable_variables
$	variables
­metrics
%regularization_losses
®non_trainable_variables
¯layer_metrics
 °layer_regularization_losses
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
±layers
'trainable_variables
(	variables
²metrics
)regularization_losses
³non_trainable_variables
´layer_metrics
 µlayer_regularization_losses
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
%:#	  2conv1d_5/kernel
: 2conv1d_5/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¶layers
-trainable_variables
.	variables
·metrics
/regularization_losses
¸non_trainable_variables
¹layer_metrics
 ºlayer_regularization_losses
Ë__call__
+Ì&call_and_return_all_conditional_losses
'Ì"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
»layers
1trainable_variables
2	variables
¼metrics
3regularization_losses
½non_trainable_variables
¾layer_metrics
 ¿layer_regularization_losses
Í__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Àlayers
5trainable_variables
6	variables
Ámetrics
7regularization_losses
Ânon_trainable_variables
Ãlayer_metrics
 Älayer_regularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_6/gamma
(:& 2batch_normalization_6/beta
1:/  (2!batch_normalization_6/moving_mean
5:3  (2%batch_normalization_6/moving_variance
.
:0
;1"
trackable_list_wrapper
<
:0
;1
<2
=3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ålayers
>trainable_variables
?	variables
Æmetrics
@regularization_losses
Çnon_trainable_variables
Èlayer_metrics
 Élayer_regularization_losses
Ñ__call__
+Ò&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_7/gamma
(:& 2batch_normalization_7/beta
1:/  (2!batch_normalization_7/moving_mean
5:3  (2%batch_normalization_7/moving_variance
.
C0
D1"
trackable_list_wrapper
<
C0
D1
E2
F3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Êlayers
Gtrainable_variables
H	variables
Ëmetrics
Iregularization_losses
Ìnon_trainable_variables
Ílayer_metrics
 Îlayer_regularization_losses
Ó__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ïlayers
Ktrainable_variables
L	variables
Ðmetrics
Mregularization_losses
Ñnon_trainable_variables
Òlayer_metrics
 Ólayer_regularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object

Ô_query_dense
Õ
_key_dense
Ö_value_dense
×_softmax
Ø_dropout_layer
Ù_output_dense
Útrainable_variables
Û	variables
Üregularization_losses
Ý	keras_api
î__call__
+ï&call_and_return_all_conditional_losses"
_tf_keras_layerê{"class_name": "MultiHeadAttention", "name": "multi_head_attention_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multi_head_attention_5", "trainable": true, "dtype": "float32", "num_heads": 1, "key_dim": 32, "value_dim": 32, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}
¯
Þlayer_with_weights-0
Þlayer-0
ßlayer_with_weights-1
ßlayer-1
à	variables
áregularization_losses
âtrainable_variables
ã	keras_api
ð__call__
+ñ&call_and_return_all_conditional_losses"È
_tf_keras_sequential©{"class_name": "Sequential", "name": "sequential_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_16_input"}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_16_input"}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
ì
	äaxis

gamma
	beta
åtrainable_variables
æ	variables
çregularization_losses
è	keras_api
ò__call__
+ó&call_and_return_all_conditional_losses"µ
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_10", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
ì
	éaxis

gamma
	beta
êtrainable_variables
ë	variables
ìregularization_losses
í	keras_api
ô__call__
+õ&call_and_return_all_conditional_losses"µ
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_11", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
í
îtrainable_variables
ï	variables
ðregularization_losses
ñ	keras_api
ö__call__
+÷&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_14", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
í
òtrainable_variables
ó	variables
ôregularization_losses
õ	keras_api
ø__call__
+ù&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_15", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
¦
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
¦
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ölayers
Utrainable_variables
V	variables
÷metrics
Wregularization_losses
ønon_trainable_variables
ùlayer_metrics
 úlayer_regularization_losses
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ûlayers
Ytrainable_variables
Z	variables
ümetrics
[regularization_losses
ýnon_trainable_variables
þlayer_metrics
 ÿlayer_regularization_losses
Ù__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_8/gamma
(:&2batch_normalization_8/beta
1:/ (2!batch_normalization_8/moving_mean
5:3 (2%batch_normalization_8/moving_variance
.
^0
_1"
trackable_list_wrapper
<
^0
_1
`2
a3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
btrainable_variables
c	variables
metrics
dregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
ftrainable_variables
g	variables
metrics
hregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
": 	è@2dense_18/kernel
:@2dense_18/bias
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
ltrainable_variables
m	variables
metrics
nregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
ptrainable_variables
q	variables
metrics
rregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_19/kernel
:@2dense_19/bias
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
vtrainable_variables
w	variables
metrics
xregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
ã__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
ztrainable_variables
{	variables
metrics
|regularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_20/kernel
:2dense_20/bias
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
trainable_variables
	variables
metrics
regularization_losses
 non_trainable_variables
¡layer_metrics
 ¢layer_regularization_losses
ç__call__
+è&call_and_return_all_conditional_losses
'è"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
G:E 25token_and_position_embedding_2/embedding_4/embeddings
H:F	R 25token_and_position_embedding_2/embedding_5/embeddings
M:K  27transformer_block_5/multi_head_attention_5/query/kernel
G:E 25transformer_block_5/multi_head_attention_5/query/bias
K:I  25transformer_block_5/multi_head_attention_5/key/kernel
E:C 23transformer_block_5/multi_head_attention_5/key/bias
M:K  27transformer_block_5/multi_head_attention_5/value/kernel
G:E 25transformer_block_5/multi_head_attention_5/value/bias
X:V  2Btransformer_block_5/multi_head_attention_5/attention_output/kernel
N:L 2@transformer_block_5/multi_head_attention_5/attention_output/bias
!: @2dense_16/kernel
:@2dense_16/bias
!:@ 2dense_17/kernel
: 2dense_17/bias
>:< 20transformer_block_5/layer_normalization_10/gamma
=:; 2/transformer_block_5/layer_normalization_10/beta
>:< 20transformer_block_5/layer_normalization_11/gamma
=:; 2/transformer_block_5/layer_normalization_11/beta
(
£0"
trackable_list_wrapper
J
<0
=1
E2
F3
`4
a5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
¶
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
19"
trackable_list_wrapper
(
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¤layers
trainable_variables
 	variables
¥metrics
¡regularization_losses
¦non_trainable_variables
§layer_metrics
 ¨layer_regularization_losses
ê__call__
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses"
_generic_user_object
(
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
©layers
£trainable_variables
¤	variables
ªmetrics
¥regularization_losses
«non_trainable_variables
¬layer_metrics
 ­layer_regularization_losses
ì__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
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
.
<0
=1"
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
E0
F1"
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
Ë
®partial_output_shape
¯full_output_shape
kernel
	bias
°trainable_variables
±	variables
²regularization_losses
³	keras_api
ú__call__
+û&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "EinsumDense", "name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
Ç
´partial_output_shape
µfull_output_shape
kernel
	bias
¶trainable_variables
·	variables
¸regularization_losses
¹	keras_api
ü__call__
+ý&call_and_return_all_conditional_losses"ç
_tf_keras_layerÍ{"class_name": "EinsumDense", "name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
Ë
ºpartial_output_shape
»full_output_shape
kernel
	bias
¼trainable_variables
½	variables
¾regularization_losses
¿	keras_api
þ__call__
+ÿ&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "EinsumDense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
ë
Àtrainable_variables
Á	variables
Âregularization_losses
Ã	keras_api
__call__
+&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}}
ç
Ätrainable_variables
Å	variables
Æregularization_losses
Ç	keras_api
__call__
+&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
à
Èpartial_output_shape
Éfull_output_shape
kernel
	bias
Êtrainable_variables
Ë	variables
Ìregularization_losses
Í	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layeræ{"class_name": "EinsumDense", "name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 32], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 1, 32]}}
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Îlayers
Útrainable_variables
Û	variables
Ïmetrics
Üregularization_losses
Ðnon_trainable_variables
Ñlayer_metrics
 Òlayer_regularization_losses
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
_generic_user_object
þ
kernel
	bias
Ótrainable_variables
Ô	variables
Õregularization_losses
Ö	keras_api
__call__
+&call_and_return_all_conditional_losses"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}

kernel
	bias
×trainable_variables
Ø	variables
Ùregularization_losses
Ú	keras_api
__call__
+&call_and_return_all_conditional_losses"Ó
_tf_keras_layer¹{"class_name": "Dense", "name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 64]}}
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
¸
à	variables
Ûmetrics
Ünon_trainable_variables
 Ýlayer_regularization_losses
áregularization_losses
âtrainable_variables
Þlayer_metrics
ßlayers
ð__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
àlayers
åtrainable_variables
æ	variables
ámetrics
çregularization_losses
ânon_trainable_variables
ãlayer_metrics
 älayer_regularization_losses
ò__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ålayers
êtrainable_variables
ë	variables
æmetrics
ìregularization_losses
çnon_trainable_variables
èlayer_metrics
 élayer_regularization_losses
ô__call__
+õ&call_and_return_all_conditional_losses
'õ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
êlayers
îtrainable_variables
ï	variables
ëmetrics
ðregularization_losses
ìnon_trainable_variables
ílayer_metrics
 îlayer_regularization_losses
ö__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ïlayers
òtrainable_variables
ó	variables
ðmetrics
ôregularization_losses
ñnon_trainable_variables
òlayer_metrics
 ólayer_regularization_losses
ø__call__
+ù&call_and_return_all_conditional_losses
'ù"call_and_return_conditional_losses"
_generic_user_object
J
O0
P1
Q2
R3
S4
T5"
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
`0
a1"
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
¿

ôtotal

õcount
ö	variables
÷	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
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
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ølayers
°trainable_variables
±	variables
ùmetrics
²regularization_losses
únon_trainable_variables
ûlayer_metrics
 ülayer_regularization_losses
ú__call__
+û&call_and_return_all_conditional_losses
'û"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ýlayers
¶trainable_variables
·	variables
þmetrics
¸regularization_losses
ÿnon_trainable_variables
layer_metrics
 layer_regularization_losses
ü__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
¼trainable_variables
½	variables
metrics
¾regularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
þ__call__
+ÿ&call_and_return_all_conditional_losses
'ÿ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
Àtrainable_variables
Á	variables
metrics
Âregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
Ätrainable_variables
Å	variables
metrics
Æregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
Êtrainable_variables
Ë	variables
metrics
Ìregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
P
Ô0
Õ1
Ö2
×3
Ø4
Ù5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
Ótrainable_variables
Ô	variables
metrics
Õregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
×trainable_variables
Ø	variables
metrics
Ùregularization_losses
non_trainable_variables
layer_metrics
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Þ0
ß1"
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
:  (2total
:  (2count
0
ô0
õ1"
trackable_list_wrapper
.
ö	variables"
_generic_user_object
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
0:.  2SGD/conv1d_4/kernel/momentum
&:$ 2SGD/conv1d_4/bias/momentum
0:.	  2SGD/conv1d_5/kernel/momentum
&:$ 2SGD/conv1d_5/bias/momentum
4:2 2(SGD/batch_normalization_6/gamma/momentum
3:1 2'SGD/batch_normalization_6/beta/momentum
4:2 2(SGD/batch_normalization_7/gamma/momentum
3:1 2'SGD/batch_normalization_7/beta/momentum
4:22(SGD/batch_normalization_8/gamma/momentum
3:12'SGD/batch_normalization_8/beta/momentum
-:+	è@2SGD/dense_18/kernel/momentum
&:$@2SGD/dense_18/bias/momentum
,:*@@2SGD/dense_19/kernel/momentum
&:$@2SGD/dense_19/bias/momentum
,:*@2SGD/dense_20/kernel/momentum
&:$2SGD/dense_20/bias/momentum
R:P 2BSGD/token_and_position_embedding_2/embedding_4/embeddings/momentum
S:Q	R 2BSGD/token_and_position_embedding_2/embedding_5/embeddings/momentum
X:V  2DSGD/transformer_block_5/multi_head_attention_5/query/kernel/momentum
R:P 2BSGD/transformer_block_5/multi_head_attention_5/query/bias/momentum
V:T  2BSGD/transformer_block_5/multi_head_attention_5/key/kernel/momentum
P:N 2@SGD/transformer_block_5/multi_head_attention_5/key/bias/momentum
X:V  2DSGD/transformer_block_5/multi_head_attention_5/value/kernel/momentum
R:P 2BSGD/transformer_block_5/multi_head_attention_5/value/bias/momentum
c:a  2OSGD/transformer_block_5/multi_head_attention_5/attention_output/kernel/momentum
Y:W 2MSGD/transformer_block_5/multi_head_attention_5/attention_output/bias/momentum
,:* @2SGD/dense_16/kernel/momentum
&:$@2SGD/dense_16/bias/momentum
,:*@ 2SGD/dense_17/kernel/momentum
&:$ 2SGD/dense_17/bias/momentum
I:G 2=SGD/transformer_block_5/layer_normalization_10/gamma/momentum
H:F 2<SGD/transformer_block_5/layer_normalization_10/beta/momentum
I:G 2=SGD/transformer_block_5/layer_normalization_11/gamma/momentum
H:F 2<SGD/transformer_block_5/layer_normalization_11/beta/momentum
î2ë
(__inference_model_2_layer_call_fn_423856
(__inference_model_2_layer_call_fn_423770
(__inference_model_2_layer_call_fn_422800
(__inference_model_2_layer_call_fn_422989À
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
C__inference_model_2_layer_call_and_return_conditional_losses_423425
C__inference_model_2_layer_call_and_return_conditional_losses_423684
C__inference_model_2_layer_call_and_return_conditional_losses_422507
C__inference_model_2_layer_call_and_return_conditional_losses_422610À
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
!__inference__wrapped_model_420976ß
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
input_5ÿÿÿÿÿÿÿÿÿR
!
input_6ÿÿÿÿÿÿÿÿÿ
ä2á
?__inference_token_and_position_embedding_2_layer_call_fn_423889
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
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_423880
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
)__inference_conv1d_4_layer_call_fn_423914¢
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
D__inference_conv1d_4_layer_call_and_return_conditional_losses_423905¢
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
4__inference_average_pooling1d_6_layer_call_fn_420991Ó
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
O__inference_average_pooling1d_6_layer_call_and_return_conditional_losses_420985Ó
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
)__inference_conv1d_5_layer_call_fn_423939¢
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
D__inference_conv1d_5_layer_call_and_return_conditional_losses_423930¢
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
4__inference_average_pooling1d_7_layer_call_fn_421006Ó
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
O__inference_average_pooling1d_7_layer_call_and_return_conditional_losses_421000Ó
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
4__inference_average_pooling1d_8_layer_call_fn_421021Ó
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
O__inference_average_pooling1d_8_layer_call_and_return_conditional_losses_421015Ó
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
6__inference_batch_normalization_6_layer_call_fn_424090
6__inference_batch_normalization_6_layer_call_fn_424021
6__inference_batch_normalization_6_layer_call_fn_424008
6__inference_batch_normalization_6_layer_call_fn_424103´
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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_423975
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_424057
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_423995
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_424077´
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
6__inference_batch_normalization_7_layer_call_fn_424254
6__inference_batch_normalization_7_layer_call_fn_424267
6__inference_batch_normalization_7_layer_call_fn_424185
6__inference_batch_normalization_7_layer_call_fn_424172´
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_424241
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_424139
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_424221
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_424159´
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
&__inference_add_2_layer_call_fn_424279¢
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
A__inference_add_2_layer_call_and_return_conditional_losses_424273¢
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
4__inference_transformer_block_5_layer_call_fn_424628
4__inference_transformer_block_5_layer_call_fn_424591°
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
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_424427
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_424554°
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
*__inference_flatten_2_layer_call_fn_424639¢
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
E__inference_flatten_2_layer_call_and_return_conditional_losses_424634¢
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
ª2§
6__inference_batch_normalization_8_layer_call_fn_424721
6__inference_batch_normalization_8_layer_call_fn_424708´
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
à2Ý
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_424695
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_424675´
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
Ø2Õ
.__inference_concatenate_2_layer_call_fn_424734¢
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
I__inference_concatenate_2_layer_call_and_return_conditional_losses_424728¢
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
)__inference_dense_18_layer_call_fn_424754¢
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
D__inference_dense_18_layer_call_and_return_conditional_losses_424745¢
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
+__inference_dropout_16_layer_call_fn_424781
+__inference_dropout_16_layer_call_fn_424776´
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
F__inference_dropout_16_layer_call_and_return_conditional_losses_424771
F__inference_dropout_16_layer_call_and_return_conditional_losses_424766´
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
)__inference_dense_19_layer_call_fn_424801¢
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
D__inference_dense_19_layer_call_and_return_conditional_losses_424792¢
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
+__inference_dropout_17_layer_call_fn_424828
+__inference_dropout_17_layer_call_fn_424823´
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
F__inference_dropout_17_layer_call_and_return_conditional_losses_424813
F__inference_dropout_17_layer_call_and_return_conditional_losses_424818´
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
)__inference_dense_20_layer_call_fn_424847¢
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
D__inference_dense_20_layer_call_and_return_conditional_losses_424838¢
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
$__inference_signature_wrapper_423083input_5input_6"
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
-__inference_sequential_5_layer_call_fn_421441
-__inference_sequential_5_layer_call_fn_424987
-__inference_sequential_5_layer_call_fn_421468
-__inference_sequential_5_layer_call_fn_424974À
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
H__inference_sequential_5_layer_call_and_return_conditional_losses_424961
H__inference_sequential_5_layer_call_and_return_conditional_losses_424904
H__inference_sequential_5_layer_call_and_return_conditional_losses_421399
H__inference_sequential_5_layer_call_and_return_conditional_losses_421413À
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
)__inference_dense_16_layer_call_fn_425027¢
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
D__inference_dense_16_layer_call_and_return_conditional_losses_425018¢
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
)__inference_dense_17_layer_call_fn_425066¢
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
D__inference_dense_17_layer_call_and_return_conditional_losses_425057¢
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
!__inference__wrapped_model_420976Ì:!"+,=:<;FCEDa^`_jktu~Y¢V
O¢L
JG
"
input_5ÿÿÿÿÿÿÿÿÿR
!
input_6ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_20"
dense_20ÿÿÿÿÿÿÿÿÿÕ
A__inference_add_2_layer_call_and_return_conditional_losses_424273b¢_
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
&__inference_add_2_layer_call_fn_424279b¢_
X¢U
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿ# 
&#
inputs/1ÿÿÿÿÿÿÿÿÿ# 
ª "ÿÿÿÿÿÿÿÿÿ# Ø
O__inference_average_pooling1d_6_layer_call_and_return_conditional_losses_420985E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
4__inference_average_pooling1d_6_layer_call_fn_420991wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØ
O__inference_average_pooling1d_7_layer_call_and_return_conditional_losses_421000E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
4__inference_average_pooling1d_7_layer_call_fn_421006wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØ
O__inference_average_pooling1d_8_layer_call_and_return_conditional_losses_421015E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
4__inference_average_pooling1d_8_layer_call_fn_421021wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÑ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_423975|<=:;@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ñ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_423995|=:<;@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ¿
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_424057j<=:;7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ¿
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_424077j=:<;7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ©
6__inference_batch_normalization_6_layer_call_fn_424008o<=:;@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ©
6__inference_batch_normalization_6_layer_call_fn_424021o=:<;@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
6__inference_batch_normalization_6_layer_call_fn_424090]<=:;7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª "ÿÿÿÿÿÿÿÿÿ# 
6__inference_batch_normalization_6_layer_call_fn_424103]=:<;7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª "ÿÿÿÿÿÿÿÿÿ# Ñ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_424139|EFCD@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ñ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_424159|FCED@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ¿
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_424221jEFCD7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ¿
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_424241jFCED7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ©
6__inference_batch_normalization_7_layer_call_fn_424172oEFCD@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ©
6__inference_batch_normalization_7_layer_call_fn_424185oFCED@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
6__inference_batch_normalization_7_layer_call_fn_424254]EFCD7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª "ÿÿÿÿÿÿÿÿÿ# 
6__inference_batch_normalization_7_layer_call_fn_424267]FCED7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª "ÿÿÿÿÿÿÿÿÿ# ·
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_424675b`a^_3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_424695ba^`_3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
6__inference_batch_normalization_8_layer_call_fn_424708U`a^_3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
6__inference_batch_normalization_8_layer_call_fn_424721Ua^`_3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿÓ
I__inference_concatenate_2_layer_call_and_return_conditional_losses_424728[¢X
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
.__inference_concatenate_2_layer_call_fn_424734x[¢X
Q¢N
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿà
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿè®
D__inference_conv1d_4_layer_call_and_return_conditional_losses_423905f!"4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿR 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿR 
 
)__inference_conv1d_4_layer_call_fn_423914Y!"4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿR 
ª "ÿÿÿÿÿÿÿÿÿR ®
D__inference_conv1d_5_layer_call_and_return_conditional_losses_423930f+,4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÞ 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÞ 
 
)__inference_conv1d_5_layer_call_fn_423939Y+,4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÞ 
ª "ÿÿÿÿÿÿÿÿÿÞ ®
D__inference_dense_16_layer_call_and_return_conditional_losses_425018f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ#@
 
)__inference_dense_16_layer_call_fn_425027Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª "ÿÿÿÿÿÿÿÿÿ#@®
D__inference_dense_17_layer_call_and_return_conditional_losses_425057f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ#@
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 
)__inference_dense_17_layer_call_fn_425066Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ#@
ª "ÿÿÿÿÿÿÿÿÿ# ¥
D__inference_dense_18_layer_call_and_return_conditional_losses_424745]jk0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿè
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 }
)__inference_dense_18_layer_call_fn_424754Pjk0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿè
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dense_19_layer_call_and_return_conditional_losses_424792\tu/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
)__inference_dense_19_layer_call_fn_424801Otu/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dense_20_layer_call_and_return_conditional_losses_424838\~/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_20_layer_call_fn_424847O~/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dropout_16_layer_call_and_return_conditional_losses_424766\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¦
F__inference_dropout_16_layer_call_and_return_conditional_losses_424771\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ~
+__inference_dropout_16_layer_call_fn_424776O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@~
+__inference_dropout_16_layer_call_fn_424781O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@¦
F__inference_dropout_17_layer_call_and_return_conditional_losses_424813\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¦
F__inference_dropout_17_layer_call_and_return_conditional_losses_424818\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ~
+__inference_dropout_17_layer_call_fn_424823O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@~
+__inference_dropout_17_layer_call_fn_424828O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@¦
E__inference_flatten_2_layer_call_and_return_conditional_losses_424634]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿà
 ~
*__inference_flatten_2_layer_call_fn_424639P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª "ÿÿÿÿÿÿÿÿÿà
C__inference_model_2_layer_call_and_return_conditional_losses_422507Æ:!"+,<=:;EFCD`a^_jktu~a¢^
W¢T
JG
"
input_5ÿÿÿÿÿÿÿÿÿR
!
input_6ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
C__inference_model_2_layer_call_and_return_conditional_losses_422610Æ:!"+,=:<;FCEDa^`_jktu~a¢^
W¢T
JG
"
input_5ÿÿÿÿÿÿÿÿÿR
!
input_6ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
C__inference_model_2_layer_call_and_return_conditional_losses_423425È:!"+,<=:;EFCD`a^_jktu~c¢`
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
C__inference_model_2_layer_call_and_return_conditional_losses_423684È:!"+,=:<;FCEDa^`_jktu~c¢`
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
(__inference_model_2_layer_call_fn_422800¹:!"+,<=:;EFCD`a^_jktu~a¢^
W¢T
JG
"
input_5ÿÿÿÿÿÿÿÿÿR
!
input_6ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿæ
(__inference_model_2_layer_call_fn_422989¹:!"+,=:<;FCEDa^`_jktu~a¢^
W¢T
JG
"
input_5ÿÿÿÿÿÿÿÿÿR
!
input_6ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿè
(__inference_model_2_layer_call_fn_423770»:!"+,<=:;EFCD`a^_jktu~c¢`
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
(__inference_model_2_layer_call_fn_423856»:!"+,=:<;FCEDa^`_jktu~c¢`
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
H__inference_sequential_5_layer_call_and_return_conditional_losses_421399zC¢@
9¢6
,)
dense_16_inputÿÿÿÿÿÿÿÿÿ# 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 Æ
H__inference_sequential_5_layer_call_and_return_conditional_losses_421413zC¢@
9¢6
,)
dense_16_inputÿÿÿÿÿÿÿÿÿ# 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ¾
H__inference_sequential_5_layer_call_and_return_conditional_losses_424904r;¢8
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
H__inference_sequential_5_layer_call_and_return_conditional_losses_424961r;¢8
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
-__inference_sequential_5_layer_call_fn_421441mC¢@
9¢6
,)
dense_16_inputÿÿÿÿÿÿÿÿÿ# 
p

 
ª "ÿÿÿÿÿÿÿÿÿ# 
-__inference_sequential_5_layer_call_fn_421468mC¢@
9¢6
,)
dense_16_inputÿÿÿÿÿÿÿÿÿ# 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ# 
-__inference_sequential_5_layer_call_fn_424974e;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p

 
ª "ÿÿÿÿÿÿÿÿÿ# 
-__inference_sequential_5_layer_call_fn_424987e;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ# 
$__inference_signature_wrapper_423083Ý:!"+,=:<;FCEDa^`_jktu~j¢g
¢ 
`ª]
-
input_5"
input_5ÿÿÿÿÿÿÿÿÿR
,
input_6!
input_6ÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_20"
dense_20ÿÿÿÿÿÿÿÿÿ½
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_423880_+¢(
!¢

xÿÿÿÿÿÿÿÿÿR
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿR 
 
?__inference_token_and_position_embedding_2_layer_call_fn_423889R+¢(
!¢

xÿÿÿÿÿÿÿÿÿR
ª "ÿÿÿÿÿÿÿÿÿR Ú
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_424427 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 Ú
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_424554 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ±
4__inference_transformer_block_5_layer_call_fn_424591y 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª "ÿÿÿÿÿÿÿÿÿ# ±
4__inference_transformer_block_5_layer_call_fn_424628y 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª "ÿÿÿÿÿÿÿÿÿ# 