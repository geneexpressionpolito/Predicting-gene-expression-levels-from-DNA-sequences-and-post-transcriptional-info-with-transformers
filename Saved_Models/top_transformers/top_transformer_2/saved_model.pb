Ñ0
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
 "serve*2.4.12v2.4.1-0-g85c8b2a817f8ô)
~
conv1d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv1d_8/kernel
w
#conv1d_8/kernel/Read/ReadVariableOpReadVariableOpconv1d_8/kernel*"
_output_shapes
:  *
dtype0
r
conv1d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_8/bias
k
!conv1d_8/bias/Read/ReadVariableOpReadVariableOpconv1d_8/bias*
_output_shapes
: *
dtype0
~
conv1d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  * 
shared_nameconv1d_9/kernel
w
#conv1d_9/kernel/Read/ReadVariableOpReadVariableOpconv1d_9/kernel*"
_output_shapes
:	  *
dtype0
r
conv1d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_9/bias
k
!conv1d_9/bias/Read/ReadVariableOpReadVariableOpconv1d_9/bias*
_output_shapes
: *
dtype0

batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_8/gamma

/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
: *
dtype0

batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_8/beta

.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
: *
dtype0

!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_8/moving_mean

5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
: *
dtype0
¢
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_8/moving_variance

9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
: *
dtype0

batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_9/gamma

/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes
: *
dtype0

batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_9/beta

.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes
: *
dtype0

!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_9/moving_mean

5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes
: *
dtype0
¢
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_9/moving_variance

9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes
: *
dtype0
{
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	è@* 
shared_namedense_32/kernel
t
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel*
_output_shapes
:	è@*
dtype0
r
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_32/bias
k
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
_output_shapes
:@*
dtype0
z
dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_33/kernel
s
#dense_33/kernel/Read/ReadVariableOpReadVariableOpdense_33/kernel*
_output_shapes

:@@*
dtype0
r
dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_33/bias
k
!dense_33/bias/Read/ReadVariableOpReadVariableOpdense_33/bias*
_output_shapes
:@*
dtype0
z
dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_34/kernel
s
#dense_34/kernel/Read/ReadVariableOpReadVariableOpdense_34/kernel*
_output_shapes

:@*
dtype0
r
dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_34/bias
k
!dense_34/bias/Read/ReadVariableOpReadVariableOpdense_34/bias*
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
5token_and_position_embedding_4/embedding_8/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75token_and_position_embedding_4/embedding_8/embeddings
¿
Itoken_and_position_embedding_4/embedding_8/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_4/embedding_8/embeddings*
_output_shapes

: *
dtype0
Ç
5token_and_position_embedding_4/embedding_9/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	R *F
shared_name75token_and_position_embedding_4/embedding_9/embeddings
À
Itoken_and_position_embedding_4/embedding_9/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_4/embedding_9/embeddings*
_output_shapes
:	R *
dtype0
Î
7transformer_block_9/multi_head_attention_9/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_9/multi_head_attention_9/query/kernel
Ç
Ktransformer_block_9/multi_head_attention_9/query/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_9/multi_head_attention_9/query/kernel*"
_output_shapes
:  *
dtype0
Æ
5transformer_block_9/multi_head_attention_9/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_9/multi_head_attention_9/query/bias
¿
Itransformer_block_9/multi_head_attention_9/query/bias/Read/ReadVariableOpReadVariableOp5transformer_block_9/multi_head_attention_9/query/bias*
_output_shapes

: *
dtype0
Ê
5transformer_block_9/multi_head_attention_9/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *F
shared_name75transformer_block_9/multi_head_attention_9/key/kernel
Ã
Itransformer_block_9/multi_head_attention_9/key/kernel/Read/ReadVariableOpReadVariableOp5transformer_block_9/multi_head_attention_9/key/kernel*"
_output_shapes
:  *
dtype0
Â
3transformer_block_9/multi_head_attention_9/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *D
shared_name53transformer_block_9/multi_head_attention_9/key/bias
»
Gtransformer_block_9/multi_head_attention_9/key/bias/Read/ReadVariableOpReadVariableOp3transformer_block_9/multi_head_attention_9/key/bias*
_output_shapes

: *
dtype0
Î
7transformer_block_9/multi_head_attention_9/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_9/multi_head_attention_9/value/kernel
Ç
Ktransformer_block_9/multi_head_attention_9/value/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_9/multi_head_attention_9/value/kernel*"
_output_shapes
:  *
dtype0
Æ
5transformer_block_9/multi_head_attention_9/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_9/multi_head_attention_9/value/bias
¿
Itransformer_block_9/multi_head_attention_9/value/bias/Read/ReadVariableOpReadVariableOp5transformer_block_9/multi_head_attention_9/value/bias*
_output_shapes

: *
dtype0
ä
Btransformer_block_9/multi_head_attention_9/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBtransformer_block_9/multi_head_attention_9/attention_output/kernel
Ý
Vtransformer_block_9/multi_head_attention_9/attention_output/kernel/Read/ReadVariableOpReadVariableOpBtransformer_block_9/multi_head_attention_9/attention_output/kernel*"
_output_shapes
:  *
dtype0
Ø
@transformer_block_9/multi_head_attention_9/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@transformer_block_9/multi_head_attention_9/attention_output/bias
Ñ
Ttransformer_block_9/multi_head_attention_9/attention_output/bias/Read/ReadVariableOpReadVariableOp@transformer_block_9/multi_head_attention_9/attention_output/bias*
_output_shapes
: *
dtype0
z
dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @* 
shared_namedense_30/kernel
s
#dense_30/kernel/Read/ReadVariableOpReadVariableOpdense_30/kernel*
_output_shapes

: @*
dtype0
r
dense_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_30/bias
k
!dense_30/bias/Read/ReadVariableOpReadVariableOpdense_30/bias*
_output_shapes
:@*
dtype0
z
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_31/kernel
s
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel*
_output_shapes

:@ *
dtype0
r
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_31/bias
k
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes
: *
dtype0
¸
0transformer_block_9/layer_normalization_18/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20transformer_block_9/layer_normalization_18/gamma
±
Dtransformer_block_9/layer_normalization_18/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_9/layer_normalization_18/gamma*
_output_shapes
: *
dtype0
¶
/transformer_block_9/layer_normalization_18/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_9/layer_normalization_18/beta
¯
Ctransformer_block_9/layer_normalization_18/beta/Read/ReadVariableOpReadVariableOp/transformer_block_9/layer_normalization_18/beta*
_output_shapes
: *
dtype0
¸
0transformer_block_9/layer_normalization_19/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20transformer_block_9/layer_normalization_19/gamma
±
Dtransformer_block_9/layer_normalization_19/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_9/layer_normalization_19/gamma*
_output_shapes
: *
dtype0
¶
/transformer_block_9/layer_normalization_19/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_9/layer_normalization_19/beta
¯
Ctransformer_block_9/layer_normalization_19/beta/Read/ReadVariableOpReadVariableOp/transformer_block_9/layer_normalization_19/beta*
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
SGD/conv1d_8/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *-
shared_nameSGD/conv1d_8/kernel/momentum

0SGD/conv1d_8/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_8/kernel/momentum*"
_output_shapes
:  *
dtype0

SGD/conv1d_8/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/conv1d_8/bias/momentum

.SGD/conv1d_8/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_8/bias/momentum*
_output_shapes
: *
dtype0

SGD/conv1d_9/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *-
shared_nameSGD/conv1d_9/kernel/momentum

0SGD/conv1d_9/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_9/kernel/momentum*"
_output_shapes
:	  *
dtype0

SGD/conv1d_9/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/conv1d_9/bias/momentum

.SGD/conv1d_9/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_9/bias/momentum*
_output_shapes
: *
dtype0
¨
(SGD/batch_normalization_8/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(SGD/batch_normalization_8/gamma/momentum
¡
<SGD/batch_normalization_8/gamma/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_8/gamma/momentum*
_output_shapes
: *
dtype0
¦
'SGD/batch_normalization_8/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'SGD/batch_normalization_8/beta/momentum

;SGD/batch_normalization_8/beta/momentum/Read/ReadVariableOpReadVariableOp'SGD/batch_normalization_8/beta/momentum*
_output_shapes
: *
dtype0
¨
(SGD/batch_normalization_9/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(SGD/batch_normalization_9/gamma/momentum
¡
<SGD/batch_normalization_9/gamma/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_9/gamma/momentum*
_output_shapes
: *
dtype0
¦
'SGD/batch_normalization_9/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'SGD/batch_normalization_9/beta/momentum

;SGD/batch_normalization_9/beta/momentum/Read/ReadVariableOpReadVariableOp'SGD/batch_normalization_9/beta/momentum*
_output_shapes
: *
dtype0

SGD/dense_32/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	è@*-
shared_nameSGD/dense_32/kernel/momentum

0SGD/dense_32/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_32/kernel/momentum*
_output_shapes
:	è@*
dtype0

SGD/dense_32/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_32/bias/momentum

.SGD/dense_32/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_32/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_33/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*-
shared_nameSGD/dense_33/kernel/momentum

0SGD/dense_33/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_33/kernel/momentum*
_output_shapes

:@@*
dtype0

SGD/dense_33/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_33/bias/momentum

.SGD/dense_33/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_33/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_34/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*-
shared_nameSGD/dense_34/kernel/momentum

0SGD/dense_34/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_34/kernel/momentum*
_output_shapes

:@*
dtype0

SGD/dense_34/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/dense_34/bias/momentum

.SGD/dense_34/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_34/bias/momentum*
_output_shapes
:*
dtype0
à
BSGD/token_and_position_embedding_4/embedding_8/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/token_and_position_embedding_4/embedding_8/embeddings/momentum
Ù
VSGD/token_and_position_embedding_4/embedding_8/embeddings/momentum/Read/ReadVariableOpReadVariableOpBSGD/token_and_position_embedding_4/embedding_8/embeddings/momentum*
_output_shapes

: *
dtype0
á
BSGD/token_and_position_embedding_4/embedding_9/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	R *S
shared_nameDBSGD/token_and_position_embedding_4/embedding_9/embeddings/momentum
Ú
VSGD/token_and_position_embedding_4/embedding_9/embeddings/momentum/Read/ReadVariableOpReadVariableOpBSGD/token_and_position_embedding_4/embedding_9/embeddings/momentum*
_output_shapes
:	R *
dtype0
è
DSGD/transformer_block_9/multi_head_attention_9/query/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_9/multi_head_attention_9/query/kernel/momentum
á
XSGD/transformer_block_9/multi_head_attention_9/query/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_9/multi_head_attention_9/query/kernel/momentum*"
_output_shapes
:  *
dtype0
à
BSGD/transformer_block_9/multi_head_attention_9/query/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_9/multi_head_attention_9/query/bias/momentum
Ù
VSGD/transformer_block_9/multi_head_attention_9/query/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_9/multi_head_attention_9/query/bias/momentum*
_output_shapes

: *
dtype0
ä
BSGD/transformer_block_9/multi_head_attention_9/key/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBSGD/transformer_block_9/multi_head_attention_9/key/kernel/momentum
Ý
VSGD/transformer_block_9/multi_head_attention_9/key/kernel/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_9/multi_head_attention_9/key/kernel/momentum*"
_output_shapes
:  *
dtype0
Ü
@SGD/transformer_block_9/multi_head_attention_9/key/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *Q
shared_nameB@SGD/transformer_block_9/multi_head_attention_9/key/bias/momentum
Õ
TSGD/transformer_block_9/multi_head_attention_9/key/bias/momentum/Read/ReadVariableOpReadVariableOp@SGD/transformer_block_9/multi_head_attention_9/key/bias/momentum*
_output_shapes

: *
dtype0
è
DSGD/transformer_block_9/multi_head_attention_9/value/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_9/multi_head_attention_9/value/kernel/momentum
á
XSGD/transformer_block_9/multi_head_attention_9/value/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_9/multi_head_attention_9/value/kernel/momentum*"
_output_shapes
:  *
dtype0
à
BSGD/transformer_block_9/multi_head_attention_9/value/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_9/multi_head_attention_9/value/bias/momentum
Ù
VSGD/transformer_block_9/multi_head_attention_9/value/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_9/multi_head_attention_9/value/bias/momentum*
_output_shapes

: *
dtype0
þ
OSGD/transformer_block_9/multi_head_attention_9/attention_output/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *`
shared_nameQOSGD/transformer_block_9/multi_head_attention_9/attention_output/kernel/momentum
÷
cSGD/transformer_block_9/multi_head_attention_9/attention_output/kernel/momentum/Read/ReadVariableOpReadVariableOpOSGD/transformer_block_9/multi_head_attention_9/attention_output/kernel/momentum*"
_output_shapes
:  *
dtype0
ò
MSGD/transformer_block_9/multi_head_attention_9/attention_output/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *^
shared_nameOMSGD/transformer_block_9/multi_head_attention_9/attention_output/bias/momentum
ë
aSGD/transformer_block_9/multi_head_attention_9/attention_output/bias/momentum/Read/ReadVariableOpReadVariableOpMSGD/transformer_block_9/multi_head_attention_9/attention_output/bias/momentum*
_output_shapes
: *
dtype0

SGD/dense_30/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*-
shared_nameSGD/dense_30/kernel/momentum

0SGD/dense_30/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_30/kernel/momentum*
_output_shapes

: @*
dtype0

SGD/dense_30/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_30/bias/momentum

.SGD/dense_30/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_30/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_31/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *-
shared_nameSGD/dense_31/kernel/momentum

0SGD/dense_31/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_31/kernel/momentum*
_output_shapes

:@ *
dtype0

SGD/dense_31/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/dense_31/bias/momentum

.SGD/dense_31/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_31/bias/momentum*
_output_shapes
: *
dtype0
Ò
=SGD/transformer_block_9/layer_normalization_18/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=SGD/transformer_block_9/layer_normalization_18/gamma/momentum
Ë
QSGD/transformer_block_9/layer_normalization_18/gamma/momentum/Read/ReadVariableOpReadVariableOp=SGD/transformer_block_9/layer_normalization_18/gamma/momentum*
_output_shapes
: *
dtype0
Ð
<SGD/transformer_block_9/layer_normalization_18/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_9/layer_normalization_18/beta/momentum
É
PSGD/transformer_block_9/layer_normalization_18/beta/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_9/layer_normalization_18/beta/momentum*
_output_shapes
: *
dtype0
Ò
=SGD/transformer_block_9/layer_normalization_19/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=SGD/transformer_block_9/layer_normalization_19/gamma/momentum
Ë
QSGD/transformer_block_9/layer_normalization_19/gamma/momentum/Read/ReadVariableOpReadVariableOp=SGD/transformer_block_9/layer_normalization_19/gamma/momentum*
_output_shapes
: *
dtype0
Ð
<SGD/transformer_block_9/layer_normalization_19/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_9/layer_normalization_19/beta/momentum
É
PSGD/transformer_block_9/layer_normalization_19/beta/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_9/layer_normalization_19/beta/momentum*
_output_shapes
: *
dtype0

NoOpNoOp
³³
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*í²
valueâ²BÞ² BÖ²
Û
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
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
n
	token_emb
pos_emb
	variables
trainable_variables
regularization_losses
	keras_api
h

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
R
&	variables
'trainable_variables
(regularization_losses
)	keras_api
h

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
R
0	variables
1trainable_variables
2regularization_losses
3	keras_api
R
4	variables
5trainable_variables
6regularization_losses
7	keras_api

8axis
	9gamma
:beta
;moving_mean
<moving_variance
=	variables
>trainable_variables
?regularization_losses
@	keras_api

Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
R
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
 
Natt
Offn
P
layernorm1
Q
layernorm2
Rdropout1
Sdropout2
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
R
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
 
R
\	variables
]trainable_variables
^regularization_losses
_	keras_api
h

`kernel
abias
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
R
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
h

jkernel
kbias
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
R
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
h

tkernel
ubias
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
æ
	zdecay
{learning_rate
|momentum
}iter momentum!momentum*momentum+momentum9momentum:momentumBmomentumCmomentum`momentumamomentumjmomentumkmomentumtmomentumumomentum~momentummomentum momentum¡momentum¢momentum£momentum¤momentum¥momentum¦momentum§momentum¨momentum©momentumªmomentum«momentum¬momentum­momentum®momentum¯momentum°
¦
~0
1
 2
!3
*4
+5
96
:7
;8
<9
B10
C11
D12
E13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
`30
a31
j32
k33
t34
u35

~0
1
 2
!3
*4
+5
96
:7
B8
C9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
`26
a27
j28
k29
t30
u31
 
²
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
	variables
trainable_variables
metrics
regularization_losses
 
f
~
embeddings
	variables
trainable_variables
regularization_losses
	keras_api
f

embeddings
	variables
trainable_variables
regularization_losses
	keras_api

~0
1

~0
1
 
²
layer_metrics
non_trainable_variables
layers
  layer_regularization_losses
	variables
trainable_variables
¡metrics
regularization_losses
[Y
VARIABLE_VALUEconv1d_8/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_8/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
²
¢layer_metrics
£non_trainable_variables
¤layers
 ¥layer_regularization_losses
"	variables
#trainable_variables
¦metrics
$regularization_losses
 
 
 
²
§layer_metrics
¨non_trainable_variables
©layers
 ªlayer_regularization_losses
&	variables
'trainable_variables
«metrics
(regularization_losses
[Y
VARIABLE_VALUEconv1d_9/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_9/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
²
¬layer_metrics
­non_trainable_variables
®layers
 ¯layer_regularization_losses
,	variables
-trainable_variables
°metrics
.regularization_losses
 
 
 
²
±layer_metrics
²non_trainable_variables
³layers
 ´layer_regularization_losses
0	variables
1trainable_variables
µmetrics
2regularization_losses
 
 
 
²
¶layer_metrics
·non_trainable_variables
¸layers
 ¹layer_regularization_losses
4	variables
5trainable_variables
ºmetrics
6regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_8/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_8/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_8/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_8/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

90
:1
;2
<3

90
:1
 
²
»layer_metrics
¼non_trainable_variables
½layers
 ¾layer_regularization_losses
=	variables
>trainable_variables
¿metrics
?regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_9/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_9/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_9/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_9/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
D2
E3

B0
C1
 
²
Àlayer_metrics
Ánon_trainable_variables
Âlayers
 Ãlayer_regularization_losses
F	variables
Gtrainable_variables
Ämetrics
Hregularization_losses
 
 
 
²
Ålayer_metrics
Ænon_trainable_variables
Çlayers
 Èlayer_regularization_losses
J	variables
Ktrainable_variables
Émetrics
Lregularization_losses
Å
Ê_query_dense
Ë
_key_dense
Ì_value_dense
Í_softmax
Î_dropout_layer
Ï_output_dense
Ð	variables
Ñtrainable_variables
Òregularization_losses
Ó	keras_api
¨
Ôlayer_with_weights-0
Ôlayer-0
Õlayer_with_weights-1
Õlayer-1
Ö	variables
×trainable_variables
Øregularization_losses
Ù	keras_api
x
	Úaxis

gamma
	beta
Û	variables
Ütrainable_variables
Ýregularization_losses
Þ	keras_api
x
	ßaxis

gamma
	beta
à	variables
átrainable_variables
âregularization_losses
ã	keras_api
V
ä	variables
åtrainable_variables
æregularization_losses
ç	keras_api
V
è	variables
étrainable_variables
êregularization_losses
ë	keras_api

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
 
²
ìlayer_metrics
ínon_trainable_variables
îlayers
 ïlayer_regularization_losses
T	variables
Utrainable_variables
ðmetrics
Vregularization_losses
 
 
 
²
ñlayer_metrics
ònon_trainable_variables
ólayers
 ôlayer_regularization_losses
X	variables
Ytrainable_variables
õmetrics
Zregularization_losses
 
 
 
²
ölayer_metrics
÷non_trainable_variables
ølayers
 ùlayer_regularization_losses
\	variables
]trainable_variables
úmetrics
^regularization_losses
[Y
VARIABLE_VALUEdense_32/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_32/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

`0
a1

`0
a1
 
²
ûlayer_metrics
ünon_trainable_variables
ýlayers
 þlayer_regularization_losses
b	variables
ctrainable_variables
ÿmetrics
dregularization_losses
 
 
 
²
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
f	variables
gtrainable_variables
metrics
hregularization_losses
[Y
VARIABLE_VALUEdense_33/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_33/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

j0
k1

j0
k1
 
²
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
l	variables
mtrainable_variables
metrics
nregularization_losses
 
 
 
²
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
p	variables
qtrainable_variables
metrics
rregularization_losses
[Y
VARIABLE_VALUEdense_34/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_34/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

t0
u1

t0
u1
 
²
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
v	variables
wtrainable_variables
metrics
xregularization_losses
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5token_and_position_embedding_4/embedding_8/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5token_and_position_embedding_4/embedding_9/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7transformer_block_9/multi_head_attention_9/query/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_9/multi_head_attention_9/query/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_9/multi_head_attention_9/key/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE3transformer_block_9/multi_head_attention_9/key/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7transformer_block_9/multi_head_attention_9/value/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_9/multi_head_attention_9/value/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEBtransformer_block_9/multi_head_attention_9/attention_output/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE@transformer_block_9/multi_head_attention_9/attention_output/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_30/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_30/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_31/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_31/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0transformer_block_9/layer_normalization_18/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_9/layer_normalization_18/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0transformer_block_9/layer_normalization_19/gamma'variables/28/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_9/layer_normalization_19/beta'variables/29/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1
D2
E3

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
 

0

~0

~0
 
µ
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
	variables
trainable_variables
metrics
regularization_losses

0

0
 
µ
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
	variables
trainable_variables
metrics
regularization_losses
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
;0
<1
 
 
 
 

D0
E1
 
 
 
 
 
 
 
 
¡
partial_output_shape
 full_output_shape
kernel
	bias
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¡
¥partial_output_shape
¦full_output_shape
kernel
	bias
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
¡
«partial_output_shape
¬full_output_shape
kernel
	bias
­	variables
®trainable_variables
¯regularization_losses
°	keras_api
V
±	variables
²trainable_variables
³regularization_losses
´	keras_api
V
µ	variables
¶trainable_variables
·regularization_losses
¸	keras_api
¡
¹partial_output_shape
ºfull_output_shape
kernel
	bias
»	variables
¼trainable_variables
½regularization_losses
¾	keras_api
@
0
1
2
3
4
5
6
7
@
0
1
2
3
4
5
6
7
 
µ
¿layer_metrics
Ànon_trainable_variables
Álayers
 Âlayer_regularization_losses
Ð	variables
Ñtrainable_variables
Ãmetrics
Òregularization_losses
n
kernel
	bias
Ä	variables
Åtrainable_variables
Æregularization_losses
Ç	keras_api
n
kernel
	bias
È	variables
Étrainable_variables
Êregularization_losses
Ë	keras_api
 
0
1
2
3
 
0
1
2
3
 
µ
Ìlayer_metrics
Ínon_trainable_variables
Îlayers
 Ïlayer_regularization_losses
Ö	variables
×trainable_variables
Ðmetrics
Øregularization_losses
 

0
1

0
1
 
µ
Ñlayer_metrics
Ònon_trainable_variables
Ólayers
 Ôlayer_regularization_losses
Û	variables
Ütrainable_variables
Õmetrics
Ýregularization_losses
 

0
1

0
1
 
µ
Ölayer_metrics
×non_trainable_variables
Ølayers
 Ùlayer_regularization_losses
à	variables
átrainable_variables
Úmetrics
âregularization_losses
 
 
 
µ
Ûlayer_metrics
Ünon_trainable_variables
Ýlayers
 Þlayer_regularization_losses
ä	variables
åtrainable_variables
ßmetrics
æregularization_losses
 
 
 
µ
àlayer_metrics
ánon_trainable_variables
âlayers
 ãlayer_regularization_losses
è	variables
étrainable_variables
ämetrics
êregularization_losses
 
 
*
N0
O1
P2
Q3
R4
S5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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

åtotal

æcount
ç	variables
è	keras_api
 
 
 
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
0
1

0
1
 
µ
élayer_metrics
ênon_trainable_variables
ëlayers
 ìlayer_regularization_losses
¡	variables
¢trainable_variables
ímetrics
£regularization_losses
 
 

0
1

0
1
 
µ
îlayer_metrics
ïnon_trainable_variables
ðlayers
 ñlayer_regularization_losses
§	variables
¨trainable_variables
òmetrics
©regularization_losses
 
 

0
1

0
1
 
µ
ólayer_metrics
ônon_trainable_variables
õlayers
 ölayer_regularization_losses
­	variables
®trainable_variables
÷metrics
¯regularization_losses
 
 
 
µ
ølayer_metrics
ùnon_trainable_variables
úlayers
 ûlayer_regularization_losses
±	variables
²trainable_variables
ümetrics
³regularization_losses
 
 
 
µ
ýlayer_metrics
þnon_trainable_variables
ÿlayers
 layer_regularization_losses
µ	variables
¶trainable_variables
metrics
·regularization_losses
 
 

0
1

0
1
 
µ
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
»	variables
¼trainable_variables
metrics
½regularization_losses
 
 
0
Ê0
Ë1
Ì2
Í3
Î4
Ï5
 
 

0
1

0
1
 
µ
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
Ä	variables
Åtrainable_variables
metrics
Æregularization_losses

0
1

0
1
 
µ
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
È	variables
Étrainable_variables
metrics
Êregularization_losses
 
 

Ô0
Õ1
 
 
 
 
 
 
 
 
 
 
 
 
 
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
å0
æ1

ç	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
VARIABLE_VALUESGD/conv1d_8/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv1d_8/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv1d_9/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv1d_9/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/batch_normalization_8/gamma/momentumXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'SGD/batch_normalization_8/beta/momentumWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/batch_normalization_9/gamma/momentumXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'SGD/batch_normalization_9/beta/momentumWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_32/kernel/momentumYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_32/bias/momentumWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_33/kernel/momentumYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_33/bias/momentumWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_34/kernel/momentumYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_34/bias/momentumWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¢
VARIABLE_VALUEBSGD/token_and_position_embedding_4/embedding_8/embeddings/momentumIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¢
VARIABLE_VALUEBSGD/token_and_position_embedding_4/embedding_9/embeddings/momentumIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¥¢
VARIABLE_VALUEDSGD/transformer_block_9/multi_head_attention_9/query/kernel/momentumJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEBSGD/transformer_block_9/multi_head_attention_9/query/bias/momentumJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEBSGD/transformer_block_9/multi_head_attention_9/key/kernel/momentumJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¡
VARIABLE_VALUE@SGD/transformer_block_9/multi_head_attention_9/key/bias/momentumJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¥¢
VARIABLE_VALUEDSGD/transformer_block_9/multi_head_attention_9/value/kernel/momentumJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEBSGD/transformer_block_9/multi_head_attention_9/value/bias/momentumJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
°­
VARIABLE_VALUEOSGD/transformer_block_9/multi_head_attention_9/attention_output/kernel/momentumJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
®«
VARIABLE_VALUEMSGD/transformer_block_9/multi_head_attention_9/attention_output/bias/momentumJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUESGD/dense_30/kernel/momentumJvariables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUESGD/dense_30/bias/momentumJvariables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUESGD/dense_31/kernel/momentumJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUESGD/dense_31/bias/momentumJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=SGD/transformer_block_9/layer_normalization_18/gamma/momentumJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<SGD/transformer_block_9/layer_normalization_18/beta/momentumJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=SGD/transformer_block_9/layer_normalization_19/gamma/momentumJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<SGD/transformer_block_9/layer_normalization_19/beta/momentumJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
{
serving_default_input_10Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
|
serving_default_input_9Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿR

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_10serving_default_input_95token_and_position_embedding_4/embedding_9/embeddings5token_and_position_embedding_4/embedding_8/embeddingsconv1d_8/kernelconv1d_8/biasconv1d_9/kernelconv1d_9/bias%batch_normalization_8/moving_variancebatch_normalization_8/gamma!batch_normalization_8/moving_meanbatch_normalization_8/beta%batch_normalization_9/moving_variancebatch_normalization_9/gamma!batch_normalization_9/moving_meanbatch_normalization_9/beta7transformer_block_9/multi_head_attention_9/query/kernel5transformer_block_9/multi_head_attention_9/query/bias5transformer_block_9/multi_head_attention_9/key/kernel3transformer_block_9/multi_head_attention_9/key/bias7transformer_block_9/multi_head_attention_9/value/kernel5transformer_block_9/multi_head_attention_9/value/biasBtransformer_block_9/multi_head_attention_9/attention_output/kernel@transformer_block_9/multi_head_attention_9/attention_output/bias0transformer_block_9/layer_normalization_18/gamma/transformer_block_9/layer_normalization_18/betadense_30/kerneldense_30/biasdense_31/kerneldense_31/bias0transformer_block_9/layer_normalization_19/gamma/transformer_block_9/layer_normalization_19/betadense_32/kerneldense_32/biasdense_33/kerneldense_33/biasdense_34/kerneldense_34/bias*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_514910
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ì$
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1d_8/kernel/Read/ReadVariableOp!conv1d_8/bias/Read/ReadVariableOp#conv1d_9/kernel/Read/ReadVariableOp!conv1d_9/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOp#dense_33/kernel/Read/ReadVariableOp!dense_33/bias/Read/ReadVariableOp#dense_34/kernel/Read/ReadVariableOp!dense_34/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpItoken_and_position_embedding_4/embedding_8/embeddings/Read/ReadVariableOpItoken_and_position_embedding_4/embedding_9/embeddings/Read/ReadVariableOpKtransformer_block_9/multi_head_attention_9/query/kernel/Read/ReadVariableOpItransformer_block_9/multi_head_attention_9/query/bias/Read/ReadVariableOpItransformer_block_9/multi_head_attention_9/key/kernel/Read/ReadVariableOpGtransformer_block_9/multi_head_attention_9/key/bias/Read/ReadVariableOpKtransformer_block_9/multi_head_attention_9/value/kernel/Read/ReadVariableOpItransformer_block_9/multi_head_attention_9/value/bias/Read/ReadVariableOpVtransformer_block_9/multi_head_attention_9/attention_output/kernel/Read/ReadVariableOpTtransformer_block_9/multi_head_attention_9/attention_output/bias/Read/ReadVariableOp#dense_30/kernel/Read/ReadVariableOp!dense_30/bias/Read/ReadVariableOp#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOpDtransformer_block_9/layer_normalization_18/gamma/Read/ReadVariableOpCtransformer_block_9/layer_normalization_18/beta/Read/ReadVariableOpDtransformer_block_9/layer_normalization_19/gamma/Read/ReadVariableOpCtransformer_block_9/layer_normalization_19/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp0SGD/conv1d_8/kernel/momentum/Read/ReadVariableOp.SGD/conv1d_8/bias/momentum/Read/ReadVariableOp0SGD/conv1d_9/kernel/momentum/Read/ReadVariableOp.SGD/conv1d_9/bias/momentum/Read/ReadVariableOp<SGD/batch_normalization_8/gamma/momentum/Read/ReadVariableOp;SGD/batch_normalization_8/beta/momentum/Read/ReadVariableOp<SGD/batch_normalization_9/gamma/momentum/Read/ReadVariableOp;SGD/batch_normalization_9/beta/momentum/Read/ReadVariableOp0SGD/dense_32/kernel/momentum/Read/ReadVariableOp.SGD/dense_32/bias/momentum/Read/ReadVariableOp0SGD/dense_33/kernel/momentum/Read/ReadVariableOp.SGD/dense_33/bias/momentum/Read/ReadVariableOp0SGD/dense_34/kernel/momentum/Read/ReadVariableOp.SGD/dense_34/bias/momentum/Read/ReadVariableOpVSGD/token_and_position_embedding_4/embedding_8/embeddings/momentum/Read/ReadVariableOpVSGD/token_and_position_embedding_4/embedding_9/embeddings/momentum/Read/ReadVariableOpXSGD/transformer_block_9/multi_head_attention_9/query/kernel/momentum/Read/ReadVariableOpVSGD/transformer_block_9/multi_head_attention_9/query/bias/momentum/Read/ReadVariableOpVSGD/transformer_block_9/multi_head_attention_9/key/kernel/momentum/Read/ReadVariableOpTSGD/transformer_block_9/multi_head_attention_9/key/bias/momentum/Read/ReadVariableOpXSGD/transformer_block_9/multi_head_attention_9/value/kernel/momentum/Read/ReadVariableOpVSGD/transformer_block_9/multi_head_attention_9/value/bias/momentum/Read/ReadVariableOpcSGD/transformer_block_9/multi_head_attention_9/attention_output/kernel/momentum/Read/ReadVariableOpaSGD/transformer_block_9/multi_head_attention_9/attention_output/bias/momentum/Read/ReadVariableOp0SGD/dense_30/kernel/momentum/Read/ReadVariableOp.SGD/dense_30/bias/momentum/Read/ReadVariableOp0SGD/dense_31/kernel/momentum/Read/ReadVariableOp.SGD/dense_31/bias/momentum/Read/ReadVariableOpQSGD/transformer_block_9/layer_normalization_18/gamma/momentum/Read/ReadVariableOpPSGD/transformer_block_9/layer_normalization_18/beta/momentum/Read/ReadVariableOpQSGD/transformer_block_9/layer_normalization_19/gamma/momentum/Read/ReadVariableOpPSGD/transformer_block_9/layer_normalization_19/beta/momentum/Read/ReadVariableOpConst*W
TinP
N2L	*
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
__inference__traced_save_516993
ÿ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_8/kernelconv1d_8/biasconv1d_9/kernelconv1d_9/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variancebatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_variancedense_32/kerneldense_32/biasdense_33/kerneldense_33/biasdense_34/kerneldense_34/biasdecaylearning_ratemomentumSGD/iter5token_and_position_embedding_4/embedding_8/embeddings5token_and_position_embedding_4/embedding_9/embeddings7transformer_block_9/multi_head_attention_9/query/kernel5transformer_block_9/multi_head_attention_9/query/bias5transformer_block_9/multi_head_attention_9/key/kernel3transformer_block_9/multi_head_attention_9/key/bias7transformer_block_9/multi_head_attention_9/value/kernel5transformer_block_9/multi_head_attention_9/value/biasBtransformer_block_9/multi_head_attention_9/attention_output/kernel@transformer_block_9/multi_head_attention_9/attention_output/biasdense_30/kerneldense_30/biasdense_31/kerneldense_31/bias0transformer_block_9/layer_normalization_18/gamma/transformer_block_9/layer_normalization_18/beta0transformer_block_9/layer_normalization_19/gamma/transformer_block_9/layer_normalization_19/betatotalcountSGD/conv1d_8/kernel/momentumSGD/conv1d_8/bias/momentumSGD/conv1d_9/kernel/momentumSGD/conv1d_9/bias/momentum(SGD/batch_normalization_8/gamma/momentum'SGD/batch_normalization_8/beta/momentum(SGD/batch_normalization_9/gamma/momentum'SGD/batch_normalization_9/beta/momentumSGD/dense_32/kernel/momentumSGD/dense_32/bias/momentumSGD/dense_33/kernel/momentumSGD/dense_33/bias/momentumSGD/dense_34/kernel/momentumSGD/dense_34/bias/momentumBSGD/token_and_position_embedding_4/embedding_8/embeddings/momentumBSGD/token_and_position_embedding_4/embedding_9/embeddings/momentumDSGD/transformer_block_9/multi_head_attention_9/query/kernel/momentumBSGD/transformer_block_9/multi_head_attention_9/query/bias/momentumBSGD/transformer_block_9/multi_head_attention_9/key/kernel/momentum@SGD/transformer_block_9/multi_head_attention_9/key/bias/momentumDSGD/transformer_block_9/multi_head_attention_9/value/kernel/momentumBSGD/transformer_block_9/multi_head_attention_9/value/bias/momentumOSGD/transformer_block_9/multi_head_attention_9/attention_output/kernel/momentumMSGD/transformer_block_9/multi_head_attention_9/attention_output/bias/momentumSGD/dense_30/kernel/momentumSGD/dense_30/bias/momentumSGD/dense_31/kernel/momentumSGD/dense_31/bias/momentum=SGD/transformer_block_9/layer_normalization_18/gamma/momentum<SGD/transformer_block_9/layer_normalization_18/beta/momentum=SGD/transformer_block_9/layer_normalization_19/gamma/momentum<SGD/transformer_block_9/layer_normalization_19/beta/momentum*V
TinO
M2K*
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
"__inference__traced_restore_517225È·&
è

Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_515840

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
ó
~
)__inference_conv1d_9_layer_call_fn_515702

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
D__inference_conv1d_9_layer_call_and_return_conditional_losses_5136152
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
È
©
6__inference_batch_normalization_9_layer_call_fn_516017

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
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_5137592
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
ö
l
P__inference_average_pooling1d_12_layer_call_and_return_conditional_losses_513038

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
ö
l
P__inference_average_pooling1d_13_layer_call_and_return_conditional_losses_513053

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
	
Ý
D__inference_dense_34_layer_call_and_return_conditional_losses_516519

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
J
¯
H__inference_sequential_9_layer_call_and_return_conditional_losses_516585

inputs.
*dense_30_tensordot_readvariableop_resource,
(dense_30_biasadd_readvariableop_resource.
*dense_31_tensordot_readvariableop_resource,
(dense_31_biasadd_readvariableop_resource
identity¢dense_30/BiasAdd/ReadVariableOp¢!dense_30/Tensordot/ReadVariableOp¢dense_31/BiasAdd/ReadVariableOp¢!dense_31/Tensordot/ReadVariableOp±
!dense_30/Tensordot/ReadVariableOpReadVariableOp*dense_30_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02#
!dense_30/Tensordot/ReadVariableOp|
dense_30/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_30/Tensordot/axes
dense_30/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_30/Tensordot/freej
dense_30/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_30/Tensordot/Shape
 dense_30/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_30/Tensordot/GatherV2/axisþ
dense_30/Tensordot/GatherV2GatherV2!dense_30/Tensordot/Shape:output:0 dense_30/Tensordot/free:output:0)dense_30/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_30/Tensordot/GatherV2
"dense_30/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_30/Tensordot/GatherV2_1/axis
dense_30/Tensordot/GatherV2_1GatherV2!dense_30/Tensordot/Shape:output:0 dense_30/Tensordot/axes:output:0+dense_30/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_30/Tensordot/GatherV2_1~
dense_30/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_30/Tensordot/Const¤
dense_30/Tensordot/ProdProd$dense_30/Tensordot/GatherV2:output:0!dense_30/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_30/Tensordot/Prod
dense_30/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_30/Tensordot/Const_1¬
dense_30/Tensordot/Prod_1Prod&dense_30/Tensordot/GatherV2_1:output:0#dense_30/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_30/Tensordot/Prod_1
dense_30/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_30/Tensordot/concat/axisÝ
dense_30/Tensordot/concatConcatV2 dense_30/Tensordot/free:output:0 dense_30/Tensordot/axes:output:0'dense_30/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_30/Tensordot/concat°
dense_30/Tensordot/stackPack dense_30/Tensordot/Prod:output:0"dense_30/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_30/Tensordot/stack«
dense_30/Tensordot/transpose	Transposeinputs"dense_30/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_30/Tensordot/transposeÃ
dense_30/Tensordot/ReshapeReshape dense_30/Tensordot/transpose:y:0!dense_30/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_30/Tensordot/ReshapeÂ
dense_30/Tensordot/MatMulMatMul#dense_30/Tensordot/Reshape:output:0)dense_30/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_30/Tensordot/MatMul
dense_30/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_30/Tensordot/Const_2
 dense_30/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_30/Tensordot/concat_1/axisê
dense_30/Tensordot/concat_1ConcatV2$dense_30/Tensordot/GatherV2:output:0#dense_30/Tensordot/Const_2:output:0)dense_30/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_30/Tensordot/concat_1´
dense_30/TensordotReshape#dense_30/Tensordot/MatMul:product:0$dense_30/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_30/Tensordot§
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_30/BiasAdd/ReadVariableOp«
dense_30/BiasAddBiasAdddense_30/Tensordot:output:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_30/BiasAddw
dense_30/ReluReludense_30/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_30/Relu±
!dense_31/Tensordot/ReadVariableOpReadVariableOp*dense_31_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02#
!dense_31/Tensordot/ReadVariableOp|
dense_31/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_31/Tensordot/axes
dense_31/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_31/Tensordot/free
dense_31/Tensordot/ShapeShapedense_30/Relu:activations:0*
T0*
_output_shapes
:2
dense_31/Tensordot/Shape
 dense_31/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_31/Tensordot/GatherV2/axisþ
dense_31/Tensordot/GatherV2GatherV2!dense_31/Tensordot/Shape:output:0 dense_31/Tensordot/free:output:0)dense_31/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_31/Tensordot/GatherV2
"dense_31/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_31/Tensordot/GatherV2_1/axis
dense_31/Tensordot/GatherV2_1GatherV2!dense_31/Tensordot/Shape:output:0 dense_31/Tensordot/axes:output:0+dense_31/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_31/Tensordot/GatherV2_1~
dense_31/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_31/Tensordot/Const¤
dense_31/Tensordot/ProdProd$dense_31/Tensordot/GatherV2:output:0!dense_31/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_31/Tensordot/Prod
dense_31/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_31/Tensordot/Const_1¬
dense_31/Tensordot/Prod_1Prod&dense_31/Tensordot/GatherV2_1:output:0#dense_31/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_31/Tensordot/Prod_1
dense_31/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_31/Tensordot/concat/axisÝ
dense_31/Tensordot/concatConcatV2 dense_31/Tensordot/free:output:0 dense_31/Tensordot/axes:output:0'dense_31/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_31/Tensordot/concat°
dense_31/Tensordot/stackPack dense_31/Tensordot/Prod:output:0"dense_31/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_31/Tensordot/stackÀ
dense_31/Tensordot/transpose	Transposedense_30/Relu:activations:0"dense_31/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_31/Tensordot/transposeÃ
dense_31/Tensordot/ReshapeReshape dense_31/Tensordot/transpose:y:0!dense_31/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_31/Tensordot/ReshapeÂ
dense_31/Tensordot/MatMulMatMul#dense_31/Tensordot/Reshape:output:0)dense_31/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_31/Tensordot/MatMul
dense_31/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_31/Tensordot/Const_2
 dense_31/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_31/Tensordot/concat_1/axisê
dense_31/Tensordot/concat_1ConcatV2$dense_31/Tensordot/GatherV2:output:0#dense_31/Tensordot/Const_2:output:0)dense_31/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_31/Tensordot/concat_1´
dense_31/TensordotReshape#dense_31/Tensordot/MatMul:product:0$dense_31/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_31/Tensordot§
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_31/BiasAdd/ReadVariableOp«
dense_31/BiasAddBiasAdddense_31/Tensordot:output:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_31/BiasAddý
IdentityIdentitydense_31/BiasAdd:output:0 ^dense_30/BiasAdd/ReadVariableOp"^dense_30/Tensordot/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp"^dense_31/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2F
!dense_30/Tensordot/ReadVariableOp!dense_30/Tensordot/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2F
!dense_31/Tensordot/ReadVariableOp!dense_31/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
É
d
F__inference_dropout_28_layer_call_and_return_conditional_losses_516452

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
D__inference_dense_33_layer_call_and_return_conditional_losses_516473

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
¸
 
-__inference_sequential_9_layer_call_fn_516668

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
H__inference_sequential_9_layer_call_and_return_conditional_losses_5135102
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
ì
©
6__inference_batch_normalization_8_layer_call_fn_515771

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
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_5131702
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
ÚÆ
1
"__inference__traced_restore_517225
file_prefix$
 assignvariableop_conv1d_8_kernel$
 assignvariableop_1_conv1d_8_bias&
"assignvariableop_2_conv1d_9_kernel$
 assignvariableop_3_conv1d_9_bias2
.assignvariableop_4_batch_normalization_8_gamma1
-assignvariableop_5_batch_normalization_8_beta8
4assignvariableop_6_batch_normalization_8_moving_mean<
8assignvariableop_7_batch_normalization_8_moving_variance2
.assignvariableop_8_batch_normalization_9_gamma1
-assignvariableop_9_batch_normalization_9_beta9
5assignvariableop_10_batch_normalization_9_moving_mean=
9assignvariableop_11_batch_normalization_9_moving_variance'
#assignvariableop_12_dense_32_kernel%
!assignvariableop_13_dense_32_bias'
#assignvariableop_14_dense_33_kernel%
!assignvariableop_15_dense_33_bias'
#assignvariableop_16_dense_34_kernel%
!assignvariableop_17_dense_34_bias
assignvariableop_18_decay%
!assignvariableop_19_learning_rate 
assignvariableop_20_momentum 
assignvariableop_21_sgd_iterM
Iassignvariableop_22_token_and_position_embedding_4_embedding_8_embeddingsM
Iassignvariableop_23_token_and_position_embedding_4_embedding_9_embeddingsO
Kassignvariableop_24_transformer_block_9_multi_head_attention_9_query_kernelM
Iassignvariableop_25_transformer_block_9_multi_head_attention_9_query_biasM
Iassignvariableop_26_transformer_block_9_multi_head_attention_9_key_kernelK
Gassignvariableop_27_transformer_block_9_multi_head_attention_9_key_biasO
Kassignvariableop_28_transformer_block_9_multi_head_attention_9_value_kernelM
Iassignvariableop_29_transformer_block_9_multi_head_attention_9_value_biasZ
Vassignvariableop_30_transformer_block_9_multi_head_attention_9_attention_output_kernelX
Tassignvariableop_31_transformer_block_9_multi_head_attention_9_attention_output_bias'
#assignvariableop_32_dense_30_kernel%
!assignvariableop_33_dense_30_bias'
#assignvariableop_34_dense_31_kernel%
!assignvariableop_35_dense_31_biasH
Dassignvariableop_36_transformer_block_9_layer_normalization_18_gammaG
Cassignvariableop_37_transformer_block_9_layer_normalization_18_betaH
Dassignvariableop_38_transformer_block_9_layer_normalization_19_gammaG
Cassignvariableop_39_transformer_block_9_layer_normalization_19_beta
assignvariableop_40_total
assignvariableop_41_count4
0assignvariableop_42_sgd_conv1d_8_kernel_momentum2
.assignvariableop_43_sgd_conv1d_8_bias_momentum4
0assignvariableop_44_sgd_conv1d_9_kernel_momentum2
.assignvariableop_45_sgd_conv1d_9_bias_momentum@
<assignvariableop_46_sgd_batch_normalization_8_gamma_momentum?
;assignvariableop_47_sgd_batch_normalization_8_beta_momentum@
<assignvariableop_48_sgd_batch_normalization_9_gamma_momentum?
;assignvariableop_49_sgd_batch_normalization_9_beta_momentum4
0assignvariableop_50_sgd_dense_32_kernel_momentum2
.assignvariableop_51_sgd_dense_32_bias_momentum4
0assignvariableop_52_sgd_dense_33_kernel_momentum2
.assignvariableop_53_sgd_dense_33_bias_momentum4
0assignvariableop_54_sgd_dense_34_kernel_momentum2
.assignvariableop_55_sgd_dense_34_bias_momentumZ
Vassignvariableop_56_sgd_token_and_position_embedding_4_embedding_8_embeddings_momentumZ
Vassignvariableop_57_sgd_token_and_position_embedding_4_embedding_9_embeddings_momentum\
Xassignvariableop_58_sgd_transformer_block_9_multi_head_attention_9_query_kernel_momentumZ
Vassignvariableop_59_sgd_transformer_block_9_multi_head_attention_9_query_bias_momentumZ
Vassignvariableop_60_sgd_transformer_block_9_multi_head_attention_9_key_kernel_momentumX
Tassignvariableop_61_sgd_transformer_block_9_multi_head_attention_9_key_bias_momentum\
Xassignvariableop_62_sgd_transformer_block_9_multi_head_attention_9_value_kernel_momentumZ
Vassignvariableop_63_sgd_transformer_block_9_multi_head_attention_9_value_bias_momentumg
cassignvariableop_64_sgd_transformer_block_9_multi_head_attention_9_attention_output_kernel_momentume
aassignvariableop_65_sgd_transformer_block_9_multi_head_attention_9_attention_output_bias_momentum4
0assignvariableop_66_sgd_dense_30_kernel_momentum2
.assignvariableop_67_sgd_dense_30_bias_momentum4
0assignvariableop_68_sgd_dense_31_kernel_momentum2
.assignvariableop_69_sgd_dense_31_bias_momentumU
Qassignvariableop_70_sgd_transformer_block_9_layer_normalization_18_gamma_momentumT
Passignvariableop_71_sgd_transformer_block_9_layer_normalization_18_beta_momentumU
Qassignvariableop_72_sgd_transformer_block_9_layer_normalization_19_gamma_momentumT
Passignvariableop_73_sgd_transformer_block_9_layer_normalization_19_beta_momentum
identity_75¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_8¢AssignVariableOp_9é%
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*õ$
valueë$Bè$KB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names§
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*«
value¡BKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices¥
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Â
_output_shapes¯
¬:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Y
dtypesO
M2K	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_conv1d_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_9_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_9_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4³
AssignVariableOp_4AssignVariableOp.assignvariableop_4_batch_normalization_8_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5²
AssignVariableOp_5AssignVariableOp-assignvariableop_5_batch_normalization_8_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¹
AssignVariableOp_6AssignVariableOp4assignvariableop_6_batch_normalization_8_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7½
AssignVariableOp_7AssignVariableOp8assignvariableop_7_batch_normalization_8_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8³
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_9_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9²
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_9_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10½
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_9_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Á
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_9_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_32_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_32_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_33_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_33_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16«
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_34_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17©
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_34_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¡
AssignVariableOp_18AssignVariableOpassignvariableop_18_decayIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19©
AssignVariableOp_19AssignVariableOp!assignvariableop_19_learning_rateIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¤
AssignVariableOp_20AssignVariableOpassignvariableop_20_momentumIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_21¤
AssignVariableOp_21AssignVariableOpassignvariableop_21_sgd_iterIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ñ
AssignVariableOp_22AssignVariableOpIassignvariableop_22_token_and_position_embedding_4_embedding_8_embeddingsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ñ
AssignVariableOp_23AssignVariableOpIassignvariableop_23_token_and_position_embedding_4_embedding_9_embeddingsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ó
AssignVariableOp_24AssignVariableOpKassignvariableop_24_transformer_block_9_multi_head_attention_9_query_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ñ
AssignVariableOp_25AssignVariableOpIassignvariableop_25_transformer_block_9_multi_head_attention_9_query_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ñ
AssignVariableOp_26AssignVariableOpIassignvariableop_26_transformer_block_9_multi_head_attention_9_key_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ï
AssignVariableOp_27AssignVariableOpGassignvariableop_27_transformer_block_9_multi_head_attention_9_key_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ó
AssignVariableOp_28AssignVariableOpKassignvariableop_28_transformer_block_9_multi_head_attention_9_value_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ñ
AssignVariableOp_29AssignVariableOpIassignvariableop_29_transformer_block_9_multi_head_attention_9_value_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Þ
AssignVariableOp_30AssignVariableOpVassignvariableop_30_transformer_block_9_multi_head_attention_9_attention_output_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ü
AssignVariableOp_31AssignVariableOpTassignvariableop_31_transformer_block_9_multi_head_attention_9_attention_output_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32«
AssignVariableOp_32AssignVariableOp#assignvariableop_32_dense_30_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33©
AssignVariableOp_33AssignVariableOp!assignvariableop_33_dense_30_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34«
AssignVariableOp_34AssignVariableOp#assignvariableop_34_dense_31_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35©
AssignVariableOp_35AssignVariableOp!assignvariableop_35_dense_31_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ì
AssignVariableOp_36AssignVariableOpDassignvariableop_36_transformer_block_9_layer_normalization_18_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ë
AssignVariableOp_37AssignVariableOpCassignvariableop_37_transformer_block_9_layer_normalization_18_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ì
AssignVariableOp_38AssignVariableOpDassignvariableop_38_transformer_block_9_layer_normalization_19_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Ë
AssignVariableOp_39AssignVariableOpCassignvariableop_39_transformer_block_9_layer_normalization_19_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40¡
AssignVariableOp_40AssignVariableOpassignvariableop_40_totalIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41¡
AssignVariableOp_41AssignVariableOpassignvariableop_41_countIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42¸
AssignVariableOp_42AssignVariableOp0assignvariableop_42_sgd_conv1d_8_kernel_momentumIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43¶
AssignVariableOp_43AssignVariableOp.assignvariableop_43_sgd_conv1d_8_bias_momentumIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¸
AssignVariableOp_44AssignVariableOp0assignvariableop_44_sgd_conv1d_9_kernel_momentumIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45¶
AssignVariableOp_45AssignVariableOp.assignvariableop_45_sgd_conv1d_9_bias_momentumIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Ä
AssignVariableOp_46AssignVariableOp<assignvariableop_46_sgd_batch_normalization_8_gamma_momentumIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Ã
AssignVariableOp_47AssignVariableOp;assignvariableop_47_sgd_batch_normalization_8_beta_momentumIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Ä
AssignVariableOp_48AssignVariableOp<assignvariableop_48_sgd_batch_normalization_9_gamma_momentumIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Ã
AssignVariableOp_49AssignVariableOp;assignvariableop_49_sgd_batch_normalization_9_beta_momentumIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50¸
AssignVariableOp_50AssignVariableOp0assignvariableop_50_sgd_dense_32_kernel_momentumIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51¶
AssignVariableOp_51AssignVariableOp.assignvariableop_51_sgd_dense_32_bias_momentumIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52¸
AssignVariableOp_52AssignVariableOp0assignvariableop_52_sgd_dense_33_kernel_momentumIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53¶
AssignVariableOp_53AssignVariableOp.assignvariableop_53_sgd_dense_33_bias_momentumIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54¸
AssignVariableOp_54AssignVariableOp0assignvariableop_54_sgd_dense_34_kernel_momentumIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55¶
AssignVariableOp_55AssignVariableOp.assignvariableop_55_sgd_dense_34_bias_momentumIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Þ
AssignVariableOp_56AssignVariableOpVassignvariableop_56_sgd_token_and_position_embedding_4_embedding_8_embeddings_momentumIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57Þ
AssignVariableOp_57AssignVariableOpVassignvariableop_57_sgd_token_and_position_embedding_4_embedding_9_embeddings_momentumIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58à
AssignVariableOp_58AssignVariableOpXassignvariableop_58_sgd_transformer_block_9_multi_head_attention_9_query_kernel_momentumIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Þ
AssignVariableOp_59AssignVariableOpVassignvariableop_59_sgd_transformer_block_9_multi_head_attention_9_query_bias_momentumIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Þ
AssignVariableOp_60AssignVariableOpVassignvariableop_60_sgd_transformer_block_9_multi_head_attention_9_key_kernel_momentumIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Ü
AssignVariableOp_61AssignVariableOpTassignvariableop_61_sgd_transformer_block_9_multi_head_attention_9_key_bias_momentumIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62à
AssignVariableOp_62AssignVariableOpXassignvariableop_62_sgd_transformer_block_9_multi_head_attention_9_value_kernel_momentumIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63Þ
AssignVariableOp_63AssignVariableOpVassignvariableop_63_sgd_transformer_block_9_multi_head_attention_9_value_bias_momentumIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64ë
AssignVariableOp_64AssignVariableOpcassignvariableop_64_sgd_transformer_block_9_multi_head_attention_9_attention_output_kernel_momentumIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65é
AssignVariableOp_65AssignVariableOpaassignvariableop_65_sgd_transformer_block_9_multi_head_attention_9_attention_output_bias_momentumIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66¸
AssignVariableOp_66AssignVariableOp0assignvariableop_66_sgd_dense_30_kernel_momentumIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67¶
AssignVariableOp_67AssignVariableOp.assignvariableop_67_sgd_dense_30_bias_momentumIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68¸
AssignVariableOp_68AssignVariableOp0assignvariableop_68_sgd_dense_31_kernel_momentumIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69¶
AssignVariableOp_69AssignVariableOp.assignvariableop_69_sgd_dense_31_bias_momentumIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Ù
AssignVariableOp_70AssignVariableOpQassignvariableop_70_sgd_transformer_block_9_layer_normalization_18_gamma_momentumIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71Ø
AssignVariableOp_71AssignVariableOpPassignvariableop_71_sgd_transformer_block_9_layer_normalization_18_beta_momentumIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72Ù
AssignVariableOp_72AssignVariableOpQassignvariableop_72_sgd_transformer_block_9_layer_normalization_19_gamma_momentumIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73Ø
AssignVariableOp_73AssignVariableOpPassignvariableop_73_sgd_transformer_block_9_layer_normalization_19_beta_momentumIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_739
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpº
Identity_74Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_74­
Identity_75IdentityIdentity_74:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_75"#
identity_75Identity_75:output:0*¿
_input_shapes­
ª: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_73AssignVariableOp_732(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ã 
þ(
__inference__traced_save_516993
file_prefix.
*savev2_conv1d_8_kernel_read_readvariableop,
(savev2_conv1d_8_bias_read_readvariableop.
*savev2_conv1d_9_kernel_read_readvariableop,
(savev2_conv1d_9_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableop.
*savev2_dense_33_kernel_read_readvariableop,
(savev2_dense_33_bias_read_readvariableop.
*savev2_dense_34_kernel_read_readvariableop,
(savev2_dense_34_bias_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop'
#savev2_momentum_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	T
Psavev2_token_and_position_embedding_4_embedding_8_embeddings_read_readvariableopT
Psavev2_token_and_position_embedding_4_embedding_9_embeddings_read_readvariableopV
Rsavev2_transformer_block_9_multi_head_attention_9_query_kernel_read_readvariableopT
Psavev2_transformer_block_9_multi_head_attention_9_query_bias_read_readvariableopT
Psavev2_transformer_block_9_multi_head_attention_9_key_kernel_read_readvariableopR
Nsavev2_transformer_block_9_multi_head_attention_9_key_bias_read_readvariableopV
Rsavev2_transformer_block_9_multi_head_attention_9_value_kernel_read_readvariableopT
Psavev2_transformer_block_9_multi_head_attention_9_value_bias_read_readvariableopa
]savev2_transformer_block_9_multi_head_attention_9_attention_output_kernel_read_readvariableop_
[savev2_transformer_block_9_multi_head_attention_9_attention_output_bias_read_readvariableop.
*savev2_dense_30_kernel_read_readvariableop,
(savev2_dense_30_bias_read_readvariableop.
*savev2_dense_31_kernel_read_readvariableop,
(savev2_dense_31_bias_read_readvariableopO
Ksavev2_transformer_block_9_layer_normalization_18_gamma_read_readvariableopN
Jsavev2_transformer_block_9_layer_normalization_18_beta_read_readvariableopO
Ksavev2_transformer_block_9_layer_normalization_19_gamma_read_readvariableopN
Jsavev2_transformer_block_9_layer_normalization_19_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop;
7savev2_sgd_conv1d_8_kernel_momentum_read_readvariableop9
5savev2_sgd_conv1d_8_bias_momentum_read_readvariableop;
7savev2_sgd_conv1d_9_kernel_momentum_read_readvariableop9
5savev2_sgd_conv1d_9_bias_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_8_gamma_momentum_read_readvariableopF
Bsavev2_sgd_batch_normalization_8_beta_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_9_gamma_momentum_read_readvariableopF
Bsavev2_sgd_batch_normalization_9_beta_momentum_read_readvariableop;
7savev2_sgd_dense_32_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_32_bias_momentum_read_readvariableop;
7savev2_sgd_dense_33_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_33_bias_momentum_read_readvariableop;
7savev2_sgd_dense_34_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_34_bias_momentum_read_readvariableopa
]savev2_sgd_token_and_position_embedding_4_embedding_8_embeddings_momentum_read_readvariableopa
]savev2_sgd_token_and_position_embedding_4_embedding_9_embeddings_momentum_read_readvariableopc
_savev2_sgd_transformer_block_9_multi_head_attention_9_query_kernel_momentum_read_readvariableopa
]savev2_sgd_transformer_block_9_multi_head_attention_9_query_bias_momentum_read_readvariableopa
]savev2_sgd_transformer_block_9_multi_head_attention_9_key_kernel_momentum_read_readvariableop_
[savev2_sgd_transformer_block_9_multi_head_attention_9_key_bias_momentum_read_readvariableopc
_savev2_sgd_transformer_block_9_multi_head_attention_9_value_kernel_momentum_read_readvariableopa
]savev2_sgd_transformer_block_9_multi_head_attention_9_value_bias_momentum_read_readvariableopn
jsavev2_sgd_transformer_block_9_multi_head_attention_9_attention_output_kernel_momentum_read_readvariableopl
hsavev2_sgd_transformer_block_9_multi_head_attention_9_attention_output_bias_momentum_read_readvariableop;
7savev2_sgd_dense_30_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_30_bias_momentum_read_readvariableop;
7savev2_sgd_dense_31_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_31_bias_momentum_read_readvariableop\
Xsavev2_sgd_transformer_block_9_layer_normalization_18_gamma_momentum_read_readvariableop[
Wsavev2_sgd_transformer_block_9_layer_normalization_18_beta_momentum_read_readvariableop\
Xsavev2_sgd_transformer_block_9_layer_normalization_19_gamma_momentum_read_readvariableop[
Wsavev2_sgd_transformer_block_9_layer_normalization_19_beta_momentum_read_readvariableop
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
ShardedFilenameã%
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*õ$
valueë$Bè$KB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¡
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*«
value¡BKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesî'
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_8_kernel_read_readvariableop(savev2_conv1d_8_bias_read_readvariableop*savev2_conv1d_9_kernel_read_readvariableop(savev2_conv1d_9_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop*savev2_dense_33_kernel_read_readvariableop(savev2_dense_33_bias_read_readvariableop*savev2_dense_34_kernel_read_readvariableop(savev2_dense_34_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableopPsavev2_token_and_position_embedding_4_embedding_8_embeddings_read_readvariableopPsavev2_token_and_position_embedding_4_embedding_9_embeddings_read_readvariableopRsavev2_transformer_block_9_multi_head_attention_9_query_kernel_read_readvariableopPsavev2_transformer_block_9_multi_head_attention_9_query_bias_read_readvariableopPsavev2_transformer_block_9_multi_head_attention_9_key_kernel_read_readvariableopNsavev2_transformer_block_9_multi_head_attention_9_key_bias_read_readvariableopRsavev2_transformer_block_9_multi_head_attention_9_value_kernel_read_readvariableopPsavev2_transformer_block_9_multi_head_attention_9_value_bias_read_readvariableop]savev2_transformer_block_9_multi_head_attention_9_attention_output_kernel_read_readvariableop[savev2_transformer_block_9_multi_head_attention_9_attention_output_bias_read_readvariableop*savev2_dense_30_kernel_read_readvariableop(savev2_dense_30_bias_read_readvariableop*savev2_dense_31_kernel_read_readvariableop(savev2_dense_31_bias_read_readvariableopKsavev2_transformer_block_9_layer_normalization_18_gamma_read_readvariableopJsavev2_transformer_block_9_layer_normalization_18_beta_read_readvariableopKsavev2_transformer_block_9_layer_normalization_19_gamma_read_readvariableopJsavev2_transformer_block_9_layer_normalization_19_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop7savev2_sgd_conv1d_8_kernel_momentum_read_readvariableop5savev2_sgd_conv1d_8_bias_momentum_read_readvariableop7savev2_sgd_conv1d_9_kernel_momentum_read_readvariableop5savev2_sgd_conv1d_9_bias_momentum_read_readvariableopCsavev2_sgd_batch_normalization_8_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_8_beta_momentum_read_readvariableopCsavev2_sgd_batch_normalization_9_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_9_beta_momentum_read_readvariableop7savev2_sgd_dense_32_kernel_momentum_read_readvariableop5savev2_sgd_dense_32_bias_momentum_read_readvariableop7savev2_sgd_dense_33_kernel_momentum_read_readvariableop5savev2_sgd_dense_33_bias_momentum_read_readvariableop7savev2_sgd_dense_34_kernel_momentum_read_readvariableop5savev2_sgd_dense_34_bias_momentum_read_readvariableop]savev2_sgd_token_and_position_embedding_4_embedding_8_embeddings_momentum_read_readvariableop]savev2_sgd_token_and_position_embedding_4_embedding_9_embeddings_momentum_read_readvariableop_savev2_sgd_transformer_block_9_multi_head_attention_9_query_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_9_multi_head_attention_9_query_bias_momentum_read_readvariableop]savev2_sgd_transformer_block_9_multi_head_attention_9_key_kernel_momentum_read_readvariableop[savev2_sgd_transformer_block_9_multi_head_attention_9_key_bias_momentum_read_readvariableop_savev2_sgd_transformer_block_9_multi_head_attention_9_value_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_9_multi_head_attention_9_value_bias_momentum_read_readvariableopjsavev2_sgd_transformer_block_9_multi_head_attention_9_attention_output_kernel_momentum_read_readvariableophsavev2_sgd_transformer_block_9_multi_head_attention_9_attention_output_bias_momentum_read_readvariableop7savev2_sgd_dense_30_kernel_momentum_read_readvariableop5savev2_sgd_dense_30_bias_momentum_read_readvariableop7savev2_sgd_dense_31_kernel_momentum_read_readvariableop5savev2_sgd_dense_31_bias_momentum_read_readvariableopXsavev2_sgd_transformer_block_9_layer_normalization_18_gamma_momentum_read_readvariableopWsavev2_sgd_transformer_block_9_layer_normalization_18_beta_momentum_read_readvariableopXsavev2_sgd_transformer_block_9_layer_normalization_19_gamma_momentum_read_readvariableopWsavev2_sgd_transformer_block_9_layer_normalization_19_beta_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Y
dtypesO
M2K	2
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

identity_1Identity_1:output:0*ñ
_input_shapesß
Ü: :  : :	  : : : : : : : : : :	è@:@:@@:@:@:: : : : : :	R :  : :  : :  : :  : : @:@:@ : : : : : : : :  : :	  : : : : : :	è@:@:@@:@:@:: :	R :  : :  : :  : :  : : @:@:@ : : : : : : 2(
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

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: :%!

_output_shapes
:	R :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
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
:  :  

_output_shapes
: :$! 

_output_shapes

: @: "

_output_shapes
:@:$# 

_output_shapes

:@ : $

_output_shapes
: : %

_output_shapes
: : &

_output_shapes
: : '

_output_shapes
: : (

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :(+$
"
_output_shapes
:  : ,

_output_shapes
: :(-$
"
_output_shapes
:	  : .

_output_shapes
: : /

_output_shapes
: : 0

_output_shapes
: : 1

_output_shapes
: : 2

_output_shapes
: :%3!

_output_shapes
:	è@: 4

_output_shapes
:@:$5 

_output_shapes

:@@: 6

_output_shapes
:@:$7 

_output_shapes

:@: 8

_output_shapes
::$9 

_output_shapes

: :%:!

_output_shapes
:	R :(;$
"
_output_shapes
:  :$< 

_output_shapes

: :(=$
"
_output_shapes
:  :$> 

_output_shapes

: :(?$
"
_output_shapes
:  :$@ 

_output_shapes

: :(A$
"
_output_shapes
:  : B

_output_shapes
: :$C 

_output_shapes

: @: D

_output_shapes
:@:$E 

_output_shapes

:@ : F

_output_shapes
: : G

_output_shapes
: : H

_output_shapes
: : I

_output_shapes
: : J

_output_shapes
: :K

_output_shapes
: 

÷
D__inference_conv1d_9_layer_call_and_return_conditional_losses_513615

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

e
F__inference_dropout_28_layer_call_and_return_conditional_losses_514283

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
î
©
6__inference_batch_normalization_8_layer_call_fn_515784

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
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_5132032
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
¹Þ
â
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_514105

inputsF
Bmulti_head_attention_9_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_9_query_add_readvariableop_resourceD
@multi_head_attention_9_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_9_key_add_readvariableop_resourceF
Bmulti_head_attention_9_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_9_value_add_readvariableop_resourceQ
Mmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_9_attention_output_add_readvariableop_resource@
<layer_normalization_18_batchnorm_mul_readvariableop_resource<
8layer_normalization_18_batchnorm_readvariableop_resource;
7sequential_9_dense_30_tensordot_readvariableop_resource9
5sequential_9_dense_30_biasadd_readvariableop_resource;
7sequential_9_dense_31_tensordot_readvariableop_resource9
5sequential_9_dense_31_biasadd_readvariableop_resource@
<layer_normalization_19_batchnorm_mul_readvariableop_resource<
8layer_normalization_19_batchnorm_readvariableop_resource
identity¢/layer_normalization_18/batchnorm/ReadVariableOp¢3layer_normalization_18/batchnorm/mul/ReadVariableOp¢/layer_normalization_19/batchnorm/ReadVariableOp¢3layer_normalization_19/batchnorm/mul/ReadVariableOp¢:multi_head_attention_9/attention_output/add/ReadVariableOp¢Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_9/key/add/ReadVariableOp¢7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_9/query/add/ReadVariableOp¢9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_9/value/add/ReadVariableOp¢9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp¢,sequential_9/dense_30/BiasAdd/ReadVariableOp¢.sequential_9/dense_30/Tensordot/ReadVariableOp¢,sequential_9/dense_31/BiasAdd/ReadVariableOp¢.sequential_9/dense_31/Tensordot/ReadVariableOpý
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_9/query/einsum/EinsumEinsuminputsAmulti_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_9/query/einsum/EinsumÛ
/multi_head_attention_9/query/add/ReadVariableOpReadVariableOp8multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_9/query/add/ReadVariableOpõ
 multi_head_attention_9/query/addAddV23multi_head_attention_9/query/einsum/Einsum:output:07multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_9/query/add÷
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_9_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_9/key/einsum/EinsumEinsuminputs?multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2*
(multi_head_attention_9/key/einsum/EinsumÕ
-multi_head_attention_9/key/add/ReadVariableOpReadVariableOp6multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_9/key/add/ReadVariableOpí
multi_head_attention_9/key/addAddV21multi_head_attention_9/key/einsum/Einsum:output:05multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
multi_head_attention_9/key/addý
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_9/value/einsum/EinsumEinsuminputsAmulti_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_9/value/einsum/EinsumÛ
/multi_head_attention_9/value/add/ReadVariableOpReadVariableOp8multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_9/value/add/ReadVariableOpõ
 multi_head_attention_9/value/addAddV23multi_head_attention_9/value/einsum/Einsum:output:07multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_9/value/add
multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_9/Mul/yÆ
multi_head_attention_9/MulMul$multi_head_attention_9/query/add:z:0%multi_head_attention_9/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_9/Mulü
$multi_head_attention_9/einsum/EinsumEinsum"multi_head_attention_9/key/add:z:0multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2&
$multi_head_attention_9/einsum/EinsumÄ
&multi_head_attention_9/softmax/SoftmaxSoftmax-multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2(
&multi_head_attention_9/softmax/SoftmaxÊ
'multi_head_attention_9/dropout/IdentityIdentity0multi_head_attention_9/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2)
'multi_head_attention_9/dropout/Identity
&multi_head_attention_9/einsum_1/EinsumEinsum0multi_head_attention_9/dropout/Identity:output:0$multi_head_attention_9/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2(
&multi_head_attention_9/einsum_1/Einsum
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_9/attention_output/einsum/EinsumEinsum/multi_head_attention_9/einsum_1/Einsum:output:0Lmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe27
5multi_head_attention_9/attention_output/einsum/Einsumø
:multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_9/attention_output/add/ReadVariableOp
+multi_head_attention_9/attention_output/addAddV2>multi_head_attention_9/attention_output/einsum/Einsum:output:0Bmulti_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+multi_head_attention_9/attention_output/add
dropout_26/IdentityIdentity/multi_head_attention_9/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_26/Identityo
addAddV2inputsdropout_26/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¸
5layer_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_18/moments/mean/reduction_indicesâ
#layer_normalization_18/moments/meanMeanadd:z:0>layer_normalization_18/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_18/moments/meanÎ
+layer_normalization_18/moments/StopGradientStopGradient,layer_normalization_18/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_18/moments/StopGradientî
0layer_normalization_18/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_18/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_18/moments/SquaredDifferenceÀ
9layer_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_18/moments/variance/reduction_indices
'layer_normalization_18/moments/varianceMean4layer_normalization_18/moments/SquaredDifference:z:0Blayer_normalization_18/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_18/moments/variance
&layer_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_18/batchnorm/add/yî
$layer_normalization_18/batchnorm/addAddV20layer_normalization_18/moments/variance:output:0/layer_normalization_18/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_18/batchnorm/add¹
&layer_normalization_18/batchnorm/RsqrtRsqrt(layer_normalization_18/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_18/batchnorm/Rsqrtã
3layer_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_18/batchnorm/mul/ReadVariableOpò
$layer_normalization_18/batchnorm/mulMul*layer_normalization_18/batchnorm/Rsqrt:y:0;layer_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_18/batchnorm/mulÀ
&layer_normalization_18/batchnorm/mul_1Muladd:z:0(layer_normalization_18/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_18/batchnorm/mul_1å
&layer_normalization_18/batchnorm/mul_2Mul,layer_normalization_18/moments/mean:output:0(layer_normalization_18/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_18/batchnorm/mul_2×
/layer_normalization_18/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_18_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_18/batchnorm/ReadVariableOpî
$layer_normalization_18/batchnorm/subSub7layer_normalization_18/batchnorm/ReadVariableOp:value:0*layer_normalization_18/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_18/batchnorm/subå
&layer_normalization_18/batchnorm/add_1AddV2*layer_normalization_18/batchnorm/mul_1:z:0(layer_normalization_18/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_18/batchnorm/add_1Ø
.sequential_9/dense_30/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_30_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_9/dense_30/Tensordot/ReadVariableOp
$sequential_9/dense_30/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_9/dense_30/Tensordot/axes
$sequential_9/dense_30/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_9/dense_30/Tensordot/free¨
%sequential_9/dense_30/Tensordot/ShapeShape*layer_normalization_18/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_9/dense_30/Tensordot/Shape 
-sequential_9/dense_30/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_30/Tensordot/GatherV2/axis¿
(sequential_9/dense_30/Tensordot/GatherV2GatherV2.sequential_9/dense_30/Tensordot/Shape:output:0-sequential_9/dense_30/Tensordot/free:output:06sequential_9/dense_30/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_9/dense_30/Tensordot/GatherV2¤
/sequential_9/dense_30/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_9/dense_30/Tensordot/GatherV2_1/axisÅ
*sequential_9/dense_30/Tensordot/GatherV2_1GatherV2.sequential_9/dense_30/Tensordot/Shape:output:0-sequential_9/dense_30/Tensordot/axes:output:08sequential_9/dense_30/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_9/dense_30/Tensordot/GatherV2_1
%sequential_9/dense_30/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_9/dense_30/Tensordot/ConstØ
$sequential_9/dense_30/Tensordot/ProdProd1sequential_9/dense_30/Tensordot/GatherV2:output:0.sequential_9/dense_30/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_9/dense_30/Tensordot/Prod
'sequential_9/dense_30/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_30/Tensordot/Const_1à
&sequential_9/dense_30/Tensordot/Prod_1Prod3sequential_9/dense_30/Tensordot/GatherV2_1:output:00sequential_9/dense_30/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_9/dense_30/Tensordot/Prod_1
+sequential_9/dense_30/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_9/dense_30/Tensordot/concat/axis
&sequential_9/dense_30/Tensordot/concatConcatV2-sequential_9/dense_30/Tensordot/free:output:0-sequential_9/dense_30/Tensordot/axes:output:04sequential_9/dense_30/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/dense_30/Tensordot/concatä
%sequential_9/dense_30/Tensordot/stackPack-sequential_9/dense_30/Tensordot/Prod:output:0/sequential_9/dense_30/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/dense_30/Tensordot/stackö
)sequential_9/dense_30/Tensordot/transpose	Transpose*layer_normalization_18/batchnorm/add_1:z:0/sequential_9/dense_30/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2+
)sequential_9/dense_30/Tensordot/transpose÷
'sequential_9/dense_30/Tensordot/ReshapeReshape-sequential_9/dense_30/Tensordot/transpose:y:0.sequential_9/dense_30/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_9/dense_30/Tensordot/Reshapeö
&sequential_9/dense_30/Tensordot/MatMulMatMul0sequential_9/dense_30/Tensordot/Reshape:output:06sequential_9/dense_30/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential_9/dense_30/Tensordot/MatMul
'sequential_9/dense_30/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_9/dense_30/Tensordot/Const_2 
-sequential_9/dense_30/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_30/Tensordot/concat_1/axis«
(sequential_9/dense_30/Tensordot/concat_1ConcatV21sequential_9/dense_30/Tensordot/GatherV2:output:00sequential_9/dense_30/Tensordot/Const_2:output:06sequential_9/dense_30/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_9/dense_30/Tensordot/concat_1è
sequential_9/dense_30/TensordotReshape0sequential_9/dense_30/Tensordot/MatMul:product:01sequential_9/dense_30/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2!
sequential_9/dense_30/TensordotÎ
,sequential_9/dense_30/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_9/dense_30/BiasAdd/ReadVariableOpß
sequential_9/dense_30/BiasAddBiasAdd(sequential_9/dense_30/Tensordot:output:04sequential_9/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_9/dense_30/BiasAdd
sequential_9/dense_30/ReluRelu&sequential_9/dense_30/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_9/dense_30/ReluØ
.sequential_9/dense_31/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_31_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_9/dense_31/Tensordot/ReadVariableOp
$sequential_9/dense_31/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_9/dense_31/Tensordot/axes
$sequential_9/dense_31/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_9/dense_31/Tensordot/free¦
%sequential_9/dense_31/Tensordot/ShapeShape(sequential_9/dense_30/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dense_31/Tensordot/Shape 
-sequential_9/dense_31/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_31/Tensordot/GatherV2/axis¿
(sequential_9/dense_31/Tensordot/GatherV2GatherV2.sequential_9/dense_31/Tensordot/Shape:output:0-sequential_9/dense_31/Tensordot/free:output:06sequential_9/dense_31/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_9/dense_31/Tensordot/GatherV2¤
/sequential_9/dense_31/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_9/dense_31/Tensordot/GatherV2_1/axisÅ
*sequential_9/dense_31/Tensordot/GatherV2_1GatherV2.sequential_9/dense_31/Tensordot/Shape:output:0-sequential_9/dense_31/Tensordot/axes:output:08sequential_9/dense_31/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_9/dense_31/Tensordot/GatherV2_1
%sequential_9/dense_31/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_9/dense_31/Tensordot/ConstØ
$sequential_9/dense_31/Tensordot/ProdProd1sequential_9/dense_31/Tensordot/GatherV2:output:0.sequential_9/dense_31/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_9/dense_31/Tensordot/Prod
'sequential_9/dense_31/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_31/Tensordot/Const_1à
&sequential_9/dense_31/Tensordot/Prod_1Prod3sequential_9/dense_31/Tensordot/GatherV2_1:output:00sequential_9/dense_31/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_9/dense_31/Tensordot/Prod_1
+sequential_9/dense_31/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_9/dense_31/Tensordot/concat/axis
&sequential_9/dense_31/Tensordot/concatConcatV2-sequential_9/dense_31/Tensordot/free:output:0-sequential_9/dense_31/Tensordot/axes:output:04sequential_9/dense_31/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/dense_31/Tensordot/concatä
%sequential_9/dense_31/Tensordot/stackPack-sequential_9/dense_31/Tensordot/Prod:output:0/sequential_9/dense_31/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/dense_31/Tensordot/stackô
)sequential_9/dense_31/Tensordot/transpose	Transpose(sequential_9/dense_30/Relu:activations:0/sequential_9/dense_31/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2+
)sequential_9/dense_31/Tensordot/transpose÷
'sequential_9/dense_31/Tensordot/ReshapeReshape-sequential_9/dense_31/Tensordot/transpose:y:0.sequential_9/dense_31/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_9/dense_31/Tensordot/Reshapeö
&sequential_9/dense_31/Tensordot/MatMulMatMul0sequential_9/dense_31/Tensordot/Reshape:output:06sequential_9/dense_31/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_9/dense_31/Tensordot/MatMul
'sequential_9/dense_31/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_31/Tensordot/Const_2 
-sequential_9/dense_31/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_31/Tensordot/concat_1/axis«
(sequential_9/dense_31/Tensordot/concat_1ConcatV21sequential_9/dense_31/Tensordot/GatherV2:output:00sequential_9/dense_31/Tensordot/Const_2:output:06sequential_9/dense_31/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_9/dense_31/Tensordot/concat_1è
sequential_9/dense_31/TensordotReshape0sequential_9/dense_31/Tensordot/MatMul:product:01sequential_9/dense_31/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
sequential_9/dense_31/TensordotÎ
,sequential_9/dense_31/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_9/dense_31/BiasAdd/ReadVariableOpß
sequential_9/dense_31/BiasAddBiasAdd(sequential_9/dense_31/Tensordot:output:04sequential_9/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
sequential_9/dense_31/BiasAdd
dropout_27/IdentityIdentity&sequential_9/dense_31/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_27/Identity
add_1AddV2*layer_normalization_18/batchnorm/add_1:z:0dropout_27/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¸
5layer_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_19/moments/mean/reduction_indicesä
#layer_normalization_19/moments/meanMean	add_1:z:0>layer_normalization_19/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_19/moments/meanÎ
+layer_normalization_19/moments/StopGradientStopGradient,layer_normalization_19/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_19/moments/StopGradientð
0layer_normalization_19/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_19/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_19/moments/SquaredDifferenceÀ
9layer_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_19/moments/variance/reduction_indices
'layer_normalization_19/moments/varianceMean4layer_normalization_19/moments/SquaredDifference:z:0Blayer_normalization_19/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_19/moments/variance
&layer_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_19/batchnorm/add/yî
$layer_normalization_19/batchnorm/addAddV20layer_normalization_19/moments/variance:output:0/layer_normalization_19/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_19/batchnorm/add¹
&layer_normalization_19/batchnorm/RsqrtRsqrt(layer_normalization_19/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_19/batchnorm/Rsqrtã
3layer_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_19/batchnorm/mul/ReadVariableOpò
$layer_normalization_19/batchnorm/mulMul*layer_normalization_19/batchnorm/Rsqrt:y:0;layer_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_19/batchnorm/mulÂ
&layer_normalization_19/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_19/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_19/batchnorm/mul_1å
&layer_normalization_19/batchnorm/mul_2Mul,layer_normalization_19/moments/mean:output:0(layer_normalization_19/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_19/batchnorm/mul_2×
/layer_normalization_19/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_19_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_19/batchnorm/ReadVariableOpî
$layer_normalization_19/batchnorm/subSub7layer_normalization_19/batchnorm/ReadVariableOp:value:0*layer_normalization_19/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_19/batchnorm/subå
&layer_normalization_19/batchnorm/add_1AddV2*layer_normalization_19/batchnorm/mul_1:z:0(layer_normalization_19/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_19/batchnorm/add_1Ü
IdentityIdentity*layer_normalization_19/batchnorm/add_1:z:00^layer_normalization_18/batchnorm/ReadVariableOp4^layer_normalization_18/batchnorm/mul/ReadVariableOp0^layer_normalization_19/batchnorm/ReadVariableOp4^layer_normalization_19/batchnorm/mul/ReadVariableOp;^multi_head_attention_9/attention_output/add/ReadVariableOpE^multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_9/key/add/ReadVariableOp8^multi_head_attention_9/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/query/add/ReadVariableOp:^multi_head_attention_9/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/value/add/ReadVariableOp:^multi_head_attention_9/value/einsum/Einsum/ReadVariableOp-^sequential_9/dense_30/BiasAdd/ReadVariableOp/^sequential_9/dense_30/Tensordot/ReadVariableOp-^sequential_9/dense_31/BiasAdd/ReadVariableOp/^sequential_9/dense_31/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2b
/layer_normalization_18/batchnorm/ReadVariableOp/layer_normalization_18/batchnorm/ReadVariableOp2j
3layer_normalization_18/batchnorm/mul/ReadVariableOp3layer_normalization_18/batchnorm/mul/ReadVariableOp2b
/layer_normalization_19/batchnorm/ReadVariableOp/layer_normalization_19/batchnorm/ReadVariableOp2j
3layer_normalization_19/batchnorm/mul/ReadVariableOp3layer_normalization_19/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_9/attention_output/add/ReadVariableOp:multi_head_attention_9/attention_output/add/ReadVariableOp2
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_9/key/add/ReadVariableOp-multi_head_attention_9/key/add/ReadVariableOp2r
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/query/add/ReadVariableOp/multi_head_attention_9/query/add/ReadVariableOp2v
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/value/add/ReadVariableOp/multi_head_attention_9/value/add/ReadVariableOp2v
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp2\
,sequential_9/dense_30/BiasAdd/ReadVariableOp,sequential_9/dense_30/BiasAdd/ReadVariableOp2`
.sequential_9/dense_30/Tensordot/ReadVariableOp.sequential_9/dense_30/Tensordot/ReadVariableOp2\
,sequential_9/dense_31/BiasAdd/ReadVariableOp,sequential_9/dense_31/BiasAdd/ReadVariableOp2`
.sequential_9/dense_31/Tensordot/ReadVariableOp.sequential_9/dense_31/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs


Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_515922

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
ï
~
)__inference_dense_31_layer_call_fn_516747

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
D__inference_dense_31_layer_call_and_return_conditional_losses_5134352
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
¼0
È
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_513759

inputs
assignmovingavg_513734
assignmovingavg_1_513740)
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
loc:@AssignMovingAvg/513734*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_513734*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/513734*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/513734*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_513734AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/513734*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/513740*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_513740*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/513740*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/513740*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_513740AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/513740*
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
¥
d
+__inference_dropout_28_layer_call_fn_516457

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
F__inference_dropout_28_layer_call_and_return_conditional_losses_5142832
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
ó0
È
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_515738

inputs
assignmovingavg_515713
assignmovingavg_1_515719)
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
loc:@AssignMovingAvg/515713*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_515713*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/515713*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/515713*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_515713AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/515713*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/515719*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_515719*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/515719*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/515719*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_515719AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/515719*
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
Ð
¨
-__inference_sequential_9_layer_call_fn_513494
dense_30_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCalldense_30_inputunknown	unknown_0	unknown_1	unknown_2*
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
H__inference_sequential_9_layer_call_and_return_conditional_losses_5134832
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
_user_specified_namedense_30_input
ø
l
P__inference_average_pooling1d_14_layer_call_and_return_conditional_losses_513068

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
¹Þ
â
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_516317

inputsF
Bmulti_head_attention_9_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_9_query_add_readvariableop_resourceD
@multi_head_attention_9_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_9_key_add_readvariableop_resourceF
Bmulti_head_attention_9_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_9_value_add_readvariableop_resourceQ
Mmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_9_attention_output_add_readvariableop_resource@
<layer_normalization_18_batchnorm_mul_readvariableop_resource<
8layer_normalization_18_batchnorm_readvariableop_resource;
7sequential_9_dense_30_tensordot_readvariableop_resource9
5sequential_9_dense_30_biasadd_readvariableop_resource;
7sequential_9_dense_31_tensordot_readvariableop_resource9
5sequential_9_dense_31_biasadd_readvariableop_resource@
<layer_normalization_19_batchnorm_mul_readvariableop_resource<
8layer_normalization_19_batchnorm_readvariableop_resource
identity¢/layer_normalization_18/batchnorm/ReadVariableOp¢3layer_normalization_18/batchnorm/mul/ReadVariableOp¢/layer_normalization_19/batchnorm/ReadVariableOp¢3layer_normalization_19/batchnorm/mul/ReadVariableOp¢:multi_head_attention_9/attention_output/add/ReadVariableOp¢Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_9/key/add/ReadVariableOp¢7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_9/query/add/ReadVariableOp¢9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_9/value/add/ReadVariableOp¢9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp¢,sequential_9/dense_30/BiasAdd/ReadVariableOp¢.sequential_9/dense_30/Tensordot/ReadVariableOp¢,sequential_9/dense_31/BiasAdd/ReadVariableOp¢.sequential_9/dense_31/Tensordot/ReadVariableOpý
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_9/query/einsum/EinsumEinsuminputsAmulti_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_9/query/einsum/EinsumÛ
/multi_head_attention_9/query/add/ReadVariableOpReadVariableOp8multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_9/query/add/ReadVariableOpõ
 multi_head_attention_9/query/addAddV23multi_head_attention_9/query/einsum/Einsum:output:07multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_9/query/add÷
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_9_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_9/key/einsum/EinsumEinsuminputs?multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2*
(multi_head_attention_9/key/einsum/EinsumÕ
-multi_head_attention_9/key/add/ReadVariableOpReadVariableOp6multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_9/key/add/ReadVariableOpí
multi_head_attention_9/key/addAddV21multi_head_attention_9/key/einsum/Einsum:output:05multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
multi_head_attention_9/key/addý
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_9/value/einsum/EinsumEinsuminputsAmulti_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_9/value/einsum/EinsumÛ
/multi_head_attention_9/value/add/ReadVariableOpReadVariableOp8multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_9/value/add/ReadVariableOpõ
 multi_head_attention_9/value/addAddV23multi_head_attention_9/value/einsum/Einsum:output:07multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_9/value/add
multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_9/Mul/yÆ
multi_head_attention_9/MulMul$multi_head_attention_9/query/add:z:0%multi_head_attention_9/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_9/Mulü
$multi_head_attention_9/einsum/EinsumEinsum"multi_head_attention_9/key/add:z:0multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2&
$multi_head_attention_9/einsum/EinsumÄ
&multi_head_attention_9/softmax/SoftmaxSoftmax-multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2(
&multi_head_attention_9/softmax/SoftmaxÊ
'multi_head_attention_9/dropout/IdentityIdentity0multi_head_attention_9/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2)
'multi_head_attention_9/dropout/Identity
&multi_head_attention_9/einsum_1/EinsumEinsum0multi_head_attention_9/dropout/Identity:output:0$multi_head_attention_9/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2(
&multi_head_attention_9/einsum_1/Einsum
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_9/attention_output/einsum/EinsumEinsum/multi_head_attention_9/einsum_1/Einsum:output:0Lmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe27
5multi_head_attention_9/attention_output/einsum/Einsumø
:multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_9/attention_output/add/ReadVariableOp
+multi_head_attention_9/attention_output/addAddV2>multi_head_attention_9/attention_output/einsum/Einsum:output:0Bmulti_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+multi_head_attention_9/attention_output/add
dropout_26/IdentityIdentity/multi_head_attention_9/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_26/Identityo
addAddV2inputsdropout_26/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¸
5layer_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_18/moments/mean/reduction_indicesâ
#layer_normalization_18/moments/meanMeanadd:z:0>layer_normalization_18/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_18/moments/meanÎ
+layer_normalization_18/moments/StopGradientStopGradient,layer_normalization_18/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_18/moments/StopGradientî
0layer_normalization_18/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_18/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_18/moments/SquaredDifferenceÀ
9layer_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_18/moments/variance/reduction_indices
'layer_normalization_18/moments/varianceMean4layer_normalization_18/moments/SquaredDifference:z:0Blayer_normalization_18/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_18/moments/variance
&layer_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_18/batchnorm/add/yî
$layer_normalization_18/batchnorm/addAddV20layer_normalization_18/moments/variance:output:0/layer_normalization_18/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_18/batchnorm/add¹
&layer_normalization_18/batchnorm/RsqrtRsqrt(layer_normalization_18/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_18/batchnorm/Rsqrtã
3layer_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_18/batchnorm/mul/ReadVariableOpò
$layer_normalization_18/batchnorm/mulMul*layer_normalization_18/batchnorm/Rsqrt:y:0;layer_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_18/batchnorm/mulÀ
&layer_normalization_18/batchnorm/mul_1Muladd:z:0(layer_normalization_18/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_18/batchnorm/mul_1å
&layer_normalization_18/batchnorm/mul_2Mul,layer_normalization_18/moments/mean:output:0(layer_normalization_18/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_18/batchnorm/mul_2×
/layer_normalization_18/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_18_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_18/batchnorm/ReadVariableOpî
$layer_normalization_18/batchnorm/subSub7layer_normalization_18/batchnorm/ReadVariableOp:value:0*layer_normalization_18/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_18/batchnorm/subå
&layer_normalization_18/batchnorm/add_1AddV2*layer_normalization_18/batchnorm/mul_1:z:0(layer_normalization_18/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_18/batchnorm/add_1Ø
.sequential_9/dense_30/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_30_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_9/dense_30/Tensordot/ReadVariableOp
$sequential_9/dense_30/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_9/dense_30/Tensordot/axes
$sequential_9/dense_30/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_9/dense_30/Tensordot/free¨
%sequential_9/dense_30/Tensordot/ShapeShape*layer_normalization_18/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_9/dense_30/Tensordot/Shape 
-sequential_9/dense_30/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_30/Tensordot/GatherV2/axis¿
(sequential_9/dense_30/Tensordot/GatherV2GatherV2.sequential_9/dense_30/Tensordot/Shape:output:0-sequential_9/dense_30/Tensordot/free:output:06sequential_9/dense_30/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_9/dense_30/Tensordot/GatherV2¤
/sequential_9/dense_30/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_9/dense_30/Tensordot/GatherV2_1/axisÅ
*sequential_9/dense_30/Tensordot/GatherV2_1GatherV2.sequential_9/dense_30/Tensordot/Shape:output:0-sequential_9/dense_30/Tensordot/axes:output:08sequential_9/dense_30/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_9/dense_30/Tensordot/GatherV2_1
%sequential_9/dense_30/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_9/dense_30/Tensordot/ConstØ
$sequential_9/dense_30/Tensordot/ProdProd1sequential_9/dense_30/Tensordot/GatherV2:output:0.sequential_9/dense_30/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_9/dense_30/Tensordot/Prod
'sequential_9/dense_30/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_30/Tensordot/Const_1à
&sequential_9/dense_30/Tensordot/Prod_1Prod3sequential_9/dense_30/Tensordot/GatherV2_1:output:00sequential_9/dense_30/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_9/dense_30/Tensordot/Prod_1
+sequential_9/dense_30/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_9/dense_30/Tensordot/concat/axis
&sequential_9/dense_30/Tensordot/concatConcatV2-sequential_9/dense_30/Tensordot/free:output:0-sequential_9/dense_30/Tensordot/axes:output:04sequential_9/dense_30/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/dense_30/Tensordot/concatä
%sequential_9/dense_30/Tensordot/stackPack-sequential_9/dense_30/Tensordot/Prod:output:0/sequential_9/dense_30/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/dense_30/Tensordot/stackö
)sequential_9/dense_30/Tensordot/transpose	Transpose*layer_normalization_18/batchnorm/add_1:z:0/sequential_9/dense_30/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2+
)sequential_9/dense_30/Tensordot/transpose÷
'sequential_9/dense_30/Tensordot/ReshapeReshape-sequential_9/dense_30/Tensordot/transpose:y:0.sequential_9/dense_30/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_9/dense_30/Tensordot/Reshapeö
&sequential_9/dense_30/Tensordot/MatMulMatMul0sequential_9/dense_30/Tensordot/Reshape:output:06sequential_9/dense_30/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential_9/dense_30/Tensordot/MatMul
'sequential_9/dense_30/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_9/dense_30/Tensordot/Const_2 
-sequential_9/dense_30/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_30/Tensordot/concat_1/axis«
(sequential_9/dense_30/Tensordot/concat_1ConcatV21sequential_9/dense_30/Tensordot/GatherV2:output:00sequential_9/dense_30/Tensordot/Const_2:output:06sequential_9/dense_30/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_9/dense_30/Tensordot/concat_1è
sequential_9/dense_30/TensordotReshape0sequential_9/dense_30/Tensordot/MatMul:product:01sequential_9/dense_30/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2!
sequential_9/dense_30/TensordotÎ
,sequential_9/dense_30/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_9/dense_30/BiasAdd/ReadVariableOpß
sequential_9/dense_30/BiasAddBiasAdd(sequential_9/dense_30/Tensordot:output:04sequential_9/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_9/dense_30/BiasAdd
sequential_9/dense_30/ReluRelu&sequential_9/dense_30/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_9/dense_30/ReluØ
.sequential_9/dense_31/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_31_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_9/dense_31/Tensordot/ReadVariableOp
$sequential_9/dense_31/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_9/dense_31/Tensordot/axes
$sequential_9/dense_31/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_9/dense_31/Tensordot/free¦
%sequential_9/dense_31/Tensordot/ShapeShape(sequential_9/dense_30/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dense_31/Tensordot/Shape 
-sequential_9/dense_31/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_31/Tensordot/GatherV2/axis¿
(sequential_9/dense_31/Tensordot/GatherV2GatherV2.sequential_9/dense_31/Tensordot/Shape:output:0-sequential_9/dense_31/Tensordot/free:output:06sequential_9/dense_31/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_9/dense_31/Tensordot/GatherV2¤
/sequential_9/dense_31/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_9/dense_31/Tensordot/GatherV2_1/axisÅ
*sequential_9/dense_31/Tensordot/GatherV2_1GatherV2.sequential_9/dense_31/Tensordot/Shape:output:0-sequential_9/dense_31/Tensordot/axes:output:08sequential_9/dense_31/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_9/dense_31/Tensordot/GatherV2_1
%sequential_9/dense_31/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_9/dense_31/Tensordot/ConstØ
$sequential_9/dense_31/Tensordot/ProdProd1sequential_9/dense_31/Tensordot/GatherV2:output:0.sequential_9/dense_31/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_9/dense_31/Tensordot/Prod
'sequential_9/dense_31/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_31/Tensordot/Const_1à
&sequential_9/dense_31/Tensordot/Prod_1Prod3sequential_9/dense_31/Tensordot/GatherV2_1:output:00sequential_9/dense_31/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_9/dense_31/Tensordot/Prod_1
+sequential_9/dense_31/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_9/dense_31/Tensordot/concat/axis
&sequential_9/dense_31/Tensordot/concatConcatV2-sequential_9/dense_31/Tensordot/free:output:0-sequential_9/dense_31/Tensordot/axes:output:04sequential_9/dense_31/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/dense_31/Tensordot/concatä
%sequential_9/dense_31/Tensordot/stackPack-sequential_9/dense_31/Tensordot/Prod:output:0/sequential_9/dense_31/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/dense_31/Tensordot/stackô
)sequential_9/dense_31/Tensordot/transpose	Transpose(sequential_9/dense_30/Relu:activations:0/sequential_9/dense_31/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2+
)sequential_9/dense_31/Tensordot/transpose÷
'sequential_9/dense_31/Tensordot/ReshapeReshape-sequential_9/dense_31/Tensordot/transpose:y:0.sequential_9/dense_31/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_9/dense_31/Tensordot/Reshapeö
&sequential_9/dense_31/Tensordot/MatMulMatMul0sequential_9/dense_31/Tensordot/Reshape:output:06sequential_9/dense_31/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_9/dense_31/Tensordot/MatMul
'sequential_9/dense_31/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_31/Tensordot/Const_2 
-sequential_9/dense_31/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_31/Tensordot/concat_1/axis«
(sequential_9/dense_31/Tensordot/concat_1ConcatV21sequential_9/dense_31/Tensordot/GatherV2:output:00sequential_9/dense_31/Tensordot/Const_2:output:06sequential_9/dense_31/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_9/dense_31/Tensordot/concat_1è
sequential_9/dense_31/TensordotReshape0sequential_9/dense_31/Tensordot/MatMul:product:01sequential_9/dense_31/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
sequential_9/dense_31/TensordotÎ
,sequential_9/dense_31/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_9/dense_31/BiasAdd/ReadVariableOpß
sequential_9/dense_31/BiasAddBiasAdd(sequential_9/dense_31/Tensordot:output:04sequential_9/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
sequential_9/dense_31/BiasAdd
dropout_27/IdentityIdentity&sequential_9/dense_31/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_27/Identity
add_1AddV2*layer_normalization_18/batchnorm/add_1:z:0dropout_27/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¸
5layer_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_19/moments/mean/reduction_indicesä
#layer_normalization_19/moments/meanMean	add_1:z:0>layer_normalization_19/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_19/moments/meanÎ
+layer_normalization_19/moments/StopGradientStopGradient,layer_normalization_19/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_19/moments/StopGradientð
0layer_normalization_19/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_19/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_19/moments/SquaredDifferenceÀ
9layer_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_19/moments/variance/reduction_indices
'layer_normalization_19/moments/varianceMean4layer_normalization_19/moments/SquaredDifference:z:0Blayer_normalization_19/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_19/moments/variance
&layer_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_19/batchnorm/add/yî
$layer_normalization_19/batchnorm/addAddV20layer_normalization_19/moments/variance:output:0/layer_normalization_19/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_19/batchnorm/add¹
&layer_normalization_19/batchnorm/RsqrtRsqrt(layer_normalization_19/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_19/batchnorm/Rsqrtã
3layer_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_19/batchnorm/mul/ReadVariableOpò
$layer_normalization_19/batchnorm/mulMul*layer_normalization_19/batchnorm/Rsqrt:y:0;layer_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_19/batchnorm/mulÂ
&layer_normalization_19/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_19/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_19/batchnorm/mul_1å
&layer_normalization_19/batchnorm/mul_2Mul,layer_normalization_19/moments/mean:output:0(layer_normalization_19/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_19/batchnorm/mul_2×
/layer_normalization_19/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_19_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_19/batchnorm/ReadVariableOpî
$layer_normalization_19/batchnorm/subSub7layer_normalization_19/batchnorm/ReadVariableOp:value:0*layer_normalization_19/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_19/batchnorm/subå
&layer_normalization_19/batchnorm/add_1AddV2*layer_normalization_19/batchnorm/mul_1:z:0(layer_normalization_19/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_19/batchnorm/add_1Ü
IdentityIdentity*layer_normalization_19/batchnorm/add_1:z:00^layer_normalization_18/batchnorm/ReadVariableOp4^layer_normalization_18/batchnorm/mul/ReadVariableOp0^layer_normalization_19/batchnorm/ReadVariableOp4^layer_normalization_19/batchnorm/mul/ReadVariableOp;^multi_head_attention_9/attention_output/add/ReadVariableOpE^multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_9/key/add/ReadVariableOp8^multi_head_attention_9/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/query/add/ReadVariableOp:^multi_head_attention_9/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/value/add/ReadVariableOp:^multi_head_attention_9/value/einsum/Einsum/ReadVariableOp-^sequential_9/dense_30/BiasAdd/ReadVariableOp/^sequential_9/dense_30/Tensordot/ReadVariableOp-^sequential_9/dense_31/BiasAdd/ReadVariableOp/^sequential_9/dense_31/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2b
/layer_normalization_18/batchnorm/ReadVariableOp/layer_normalization_18/batchnorm/ReadVariableOp2j
3layer_normalization_18/batchnorm/mul/ReadVariableOp3layer_normalization_18/batchnorm/mul/ReadVariableOp2b
/layer_normalization_19/batchnorm/ReadVariableOp/layer_normalization_19/batchnorm/ReadVariableOp2j
3layer_normalization_19/batchnorm/mul/ReadVariableOp3layer_normalization_19/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_9/attention_output/add/ReadVariableOp:multi_head_attention_9/attention_output/add/ReadVariableOp2
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_9/key/add/ReadVariableOp-multi_head_attention_9/key/add/ReadVariableOp2r
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/query/add/ReadVariableOp/multi_head_attention_9/query/add/ReadVariableOp2v
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/value/add/ReadVariableOp/multi_head_attention_9/value/add/ReadVariableOp2v
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp2\
,sequential_9/dense_30/BiasAdd/ReadVariableOp,sequential_9/dense_30/BiasAdd/ReadVariableOp2`
.sequential_9/dense_30/Tensordot/ReadVariableOp.sequential_9/dense_30/Tensordot/ReadVariableOp2\
,sequential_9/dense_31/BiasAdd/ReadVariableOp,sequential_9/dense_31/BiasAdd/ReadVariableOp2`
.sequential_9/dense_31/Tensordot/ReadVariableOp.sequential_9/dense_31/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
É
d
F__inference_dropout_29_layer_call_and_return_conditional_losses_516499

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


?__inference_token_and_position_embedding_4_layer_call_fn_515652
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
Z__inference_token_and_position_embedding_4_layer_call_and_return_conditional_losses_5135502
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
ô[
è
C__inference_model_4_layer_call_and_return_conditional_losses_514577

inputs
inputs_1)
%token_and_position_embedding_4_514487)
%token_and_position_embedding_4_514489
conv1d_8_514492
conv1d_8_514494
conv1d_9_514498
conv1d_9_514500 
batch_normalization_8_514505 
batch_normalization_8_514507 
batch_normalization_8_514509 
batch_normalization_8_514511 
batch_normalization_9_514514 
batch_normalization_9_514516 
batch_normalization_9_514518 
batch_normalization_9_514520
transformer_block_9_514524
transformer_block_9_514526
transformer_block_9_514528
transformer_block_9_514530
transformer_block_9_514532
transformer_block_9_514534
transformer_block_9_514536
transformer_block_9_514538
transformer_block_9_514540
transformer_block_9_514542
transformer_block_9_514544
transformer_block_9_514546
transformer_block_9_514548
transformer_block_9_514550
transformer_block_9_514552
transformer_block_9_514554
dense_32_514559
dense_32_514561
dense_33_514565
dense_33_514567
dense_34_514571
dense_34_514573
identity¢-batch_normalization_8/StatefulPartitionedCall¢-batch_normalization_9/StatefulPartitionedCall¢ conv1d_8/StatefulPartitionedCall¢ conv1d_9/StatefulPartitionedCall¢ dense_32/StatefulPartitionedCall¢ dense_33/StatefulPartitionedCall¢ dense_34/StatefulPartitionedCall¢"dropout_28/StatefulPartitionedCall¢"dropout_29/StatefulPartitionedCall¢6token_and_position_embedding_4/StatefulPartitionedCall¢+transformer_block_9/StatefulPartitionedCall
6token_and_position_embedding_4/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_4_514487%token_and_position_embedding_4_514489*
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
Z__inference_token_and_position_embedding_4_layer_call_and_return_conditional_losses_51355028
6token_and_position_embedding_4/StatefulPartitionedCallÕ
 conv1d_8/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_4/StatefulPartitionedCall:output:0conv1d_8_514492conv1d_8_514494*
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
D__inference_conv1d_8_layer_call_and_return_conditional_losses_5135822"
 conv1d_8/StatefulPartitionedCall£
$average_pooling1d_12/PartitionedCallPartitionedCall)conv1d_8/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_12_layer_call_and_return_conditional_losses_5130382&
$average_pooling1d_12/PartitionedCallÃ
 conv1d_9/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_12/PartitionedCall:output:0conv1d_9_514498conv1d_9_514500*
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
D__inference_conv1d_9_layer_call_and_return_conditional_losses_5136152"
 conv1d_9/StatefulPartitionedCall¸
$average_pooling1d_14/PartitionedCallPartitionedCall?token_and_position_embedding_4/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_14_layer_call_and_return_conditional_losses_5130682&
$average_pooling1d_14/PartitionedCall¢
$average_pooling1d_13/PartitionedCallPartitionedCall)conv1d_9/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_13_layer_call_and_return_conditional_losses_5130532&
$average_pooling1d_13/PartitionedCallÁ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_13/PartitionedCall:output:0batch_normalization_8_514505batch_normalization_8_514507batch_normalization_8_514509batch_normalization_8_514511*
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
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_5136682/
-batch_normalization_8/StatefulPartitionedCallÁ
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_14/PartitionedCall:output:0batch_normalization_9_514514batch_normalization_9_514516batch_normalization_9_514518batch_normalization_9_514520*
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
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_5137592/
-batch_normalization_9/StatefulPartitionedCall»
add_4/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:06batch_normalization_9/StatefulPartitionedCall:output:0*
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
A__inference_add_4_layer_call_and_return_conditional_losses_5138212
add_4/PartitionedCall
+transformer_block_9/StatefulPartitionedCallStatefulPartitionedCalladd_4/PartitionedCall:output:0transformer_block_9_514524transformer_block_9_514526transformer_block_9_514528transformer_block_9_514530transformer_block_9_514532transformer_block_9_514534transformer_block_9_514536transformer_block_9_514538transformer_block_9_514540transformer_block_9_514542transformer_block_9_514544transformer_block_9_514546transformer_block_9_514548transformer_block_9_514550transformer_block_9_514552transformer_block_9_514554*
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
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_5139782-
+transformer_block_9/StatefulPartitionedCall
flatten_4/PartitionedCallPartitionedCall4transformer_block_9/StatefulPartitionedCall:output:0*
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
E__inference_flatten_4_layer_call_and_return_conditional_losses_5142202
flatten_4/PartitionedCall
concatenate_4/PartitionedCallPartitionedCall"flatten_4/PartitionedCall:output:0inputs_1*
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
I__inference_concatenate_4_layer_call_and_return_conditional_losses_5142352
concatenate_4/PartitionedCall·
 dense_32/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_32_514559dense_32_514561*
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
D__inference_dense_32_layer_call_and_return_conditional_losses_5142552"
 dense_32/StatefulPartitionedCall
"dropout_28/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0*
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
F__inference_dropout_28_layer_call_and_return_conditional_losses_5142832$
"dropout_28/StatefulPartitionedCall¼
 dense_33/StatefulPartitionedCallStatefulPartitionedCall+dropout_28/StatefulPartitionedCall:output:0dense_33_514565dense_33_514567*
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
D__inference_dense_33_layer_call_and_return_conditional_losses_5143122"
 dense_33/StatefulPartitionedCall½
"dropout_29/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0#^dropout_28/StatefulPartitionedCall*
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
F__inference_dropout_29_layer_call_and_return_conditional_losses_5143402$
"dropout_29/StatefulPartitionedCall¼
 dense_34/StatefulPartitionedCallStatefulPartitionedCall+dropout_29/StatefulPartitionedCall:output:0dense_34_514571dense_34_514573*
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
D__inference_dense_34_layer_call_and_return_conditional_losses_5143682"
 dense_34/StatefulPartitionedCall½
IdentityIdentity)dense_34/StatefulPartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv1d_8/StatefulPartitionedCall!^conv1d_9/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall#^dropout_28/StatefulPartitionedCall#^dropout_29/StatefulPartitionedCall7^token_and_position_embedding_4/StatefulPartitionedCall,^transformer_block_9/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv1d_8/StatefulPartitionedCall conv1d_8/StatefulPartitionedCall2D
 conv1d_9/StatefulPartitionedCall conv1d_9/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2H
"dropout_28/StatefulPartitionedCall"dropout_28/StatefulPartitionedCall2H
"dropout_29/StatefulPartitionedCall"dropout_29/StatefulPartitionedCall2p
6token_and_position_embedding_4/StatefulPartitionedCall6token_and_position_embedding_4/StatefulPartitionedCall2Z
+transformer_block_9/StatefulPartitionedCall+transformer_block_9/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±

$__inference_signature_wrapper_514910
input_10
input_9
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

unknown_34
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinput_9input_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_5130292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿR::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_10:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_9
×
£
(__inference_model_4_layer_call_fn_514824
input_9
input_10
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

unknown_34
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinput_9input_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_5147492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_9:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_10
ñ	
Ý
D__inference_dense_32_layer_call_and_return_conditional_losses_514255

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
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_513688

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
D__inference_dense_30_layer_call_and_return_conditional_losses_513389

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
µ
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_516397

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
± 
ã
D__inference_dense_30_layer_call_and_return_conditional_losses_516699

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
é

H__inference_sequential_9_layer_call_and_return_conditional_losses_513483

inputs
dense_30_513472
dense_30_513474
dense_31_513477
dense_31_513479
identity¢ dense_30/StatefulPartitionedCall¢ dense_31/StatefulPartitionedCall
 dense_30/StatefulPartitionedCallStatefulPartitionedCallinputsdense_30_513472dense_30_513474*
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
D__inference_dense_30_layer_call_and_return_conditional_losses_5133892"
 dense_30/StatefulPartitionedCall¾
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_513477dense_31_513479*
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
D__inference_dense_31_layer_call_and_return_conditional_losses_5134352"
 dense_31/StatefulPartitionedCallÇ
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
Ð
¨
-__inference_sequential_9_layer_call_fn_513521
dense_30_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCalldense_30_inputunknown	unknown_0	unknown_1	unknown_2*
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
H__inference_sequential_9_layer_call_and_return_conditional_losses_5135102
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
_user_specified_namedense_30_input
¼0
È
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_515820

inputs
assignmovingavg_515795
assignmovingavg_1_515801)
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
loc:@AssignMovingAvg/515795*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_515795*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/515795*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/515795*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_515795AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/515795*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/515801*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_515801*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/515801*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/515801*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_515801AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/515801*
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
J
¯
H__inference_sequential_9_layer_call_and_return_conditional_losses_516642

inputs.
*dense_30_tensordot_readvariableop_resource,
(dense_30_biasadd_readvariableop_resource.
*dense_31_tensordot_readvariableop_resource,
(dense_31_biasadd_readvariableop_resource
identity¢dense_30/BiasAdd/ReadVariableOp¢!dense_30/Tensordot/ReadVariableOp¢dense_31/BiasAdd/ReadVariableOp¢!dense_31/Tensordot/ReadVariableOp±
!dense_30/Tensordot/ReadVariableOpReadVariableOp*dense_30_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02#
!dense_30/Tensordot/ReadVariableOp|
dense_30/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_30/Tensordot/axes
dense_30/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_30/Tensordot/freej
dense_30/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_30/Tensordot/Shape
 dense_30/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_30/Tensordot/GatherV2/axisþ
dense_30/Tensordot/GatherV2GatherV2!dense_30/Tensordot/Shape:output:0 dense_30/Tensordot/free:output:0)dense_30/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_30/Tensordot/GatherV2
"dense_30/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_30/Tensordot/GatherV2_1/axis
dense_30/Tensordot/GatherV2_1GatherV2!dense_30/Tensordot/Shape:output:0 dense_30/Tensordot/axes:output:0+dense_30/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_30/Tensordot/GatherV2_1~
dense_30/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_30/Tensordot/Const¤
dense_30/Tensordot/ProdProd$dense_30/Tensordot/GatherV2:output:0!dense_30/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_30/Tensordot/Prod
dense_30/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_30/Tensordot/Const_1¬
dense_30/Tensordot/Prod_1Prod&dense_30/Tensordot/GatherV2_1:output:0#dense_30/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_30/Tensordot/Prod_1
dense_30/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_30/Tensordot/concat/axisÝ
dense_30/Tensordot/concatConcatV2 dense_30/Tensordot/free:output:0 dense_30/Tensordot/axes:output:0'dense_30/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_30/Tensordot/concat°
dense_30/Tensordot/stackPack dense_30/Tensordot/Prod:output:0"dense_30/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_30/Tensordot/stack«
dense_30/Tensordot/transpose	Transposeinputs"dense_30/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_30/Tensordot/transposeÃ
dense_30/Tensordot/ReshapeReshape dense_30/Tensordot/transpose:y:0!dense_30/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_30/Tensordot/ReshapeÂ
dense_30/Tensordot/MatMulMatMul#dense_30/Tensordot/Reshape:output:0)dense_30/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_30/Tensordot/MatMul
dense_30/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_30/Tensordot/Const_2
 dense_30/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_30/Tensordot/concat_1/axisê
dense_30/Tensordot/concat_1ConcatV2$dense_30/Tensordot/GatherV2:output:0#dense_30/Tensordot/Const_2:output:0)dense_30/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_30/Tensordot/concat_1´
dense_30/TensordotReshape#dense_30/Tensordot/MatMul:product:0$dense_30/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_30/Tensordot§
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_30/BiasAdd/ReadVariableOp«
dense_30/BiasAddBiasAdddense_30/Tensordot:output:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_30/BiasAddw
dense_30/ReluReludense_30/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_30/Relu±
!dense_31/Tensordot/ReadVariableOpReadVariableOp*dense_31_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02#
!dense_31/Tensordot/ReadVariableOp|
dense_31/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_31/Tensordot/axes
dense_31/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_31/Tensordot/free
dense_31/Tensordot/ShapeShapedense_30/Relu:activations:0*
T0*
_output_shapes
:2
dense_31/Tensordot/Shape
 dense_31/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_31/Tensordot/GatherV2/axisþ
dense_31/Tensordot/GatherV2GatherV2!dense_31/Tensordot/Shape:output:0 dense_31/Tensordot/free:output:0)dense_31/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_31/Tensordot/GatherV2
"dense_31/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_31/Tensordot/GatherV2_1/axis
dense_31/Tensordot/GatherV2_1GatherV2!dense_31/Tensordot/Shape:output:0 dense_31/Tensordot/axes:output:0+dense_31/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_31/Tensordot/GatherV2_1~
dense_31/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_31/Tensordot/Const¤
dense_31/Tensordot/ProdProd$dense_31/Tensordot/GatherV2:output:0!dense_31/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_31/Tensordot/Prod
dense_31/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_31/Tensordot/Const_1¬
dense_31/Tensordot/Prod_1Prod&dense_31/Tensordot/GatherV2_1:output:0#dense_31/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_31/Tensordot/Prod_1
dense_31/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_31/Tensordot/concat/axisÝ
dense_31/Tensordot/concatConcatV2 dense_31/Tensordot/free:output:0 dense_31/Tensordot/axes:output:0'dense_31/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_31/Tensordot/concat°
dense_31/Tensordot/stackPack dense_31/Tensordot/Prod:output:0"dense_31/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_31/Tensordot/stackÀ
dense_31/Tensordot/transpose	Transposedense_30/Relu:activations:0"dense_31/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_31/Tensordot/transposeÃ
dense_31/Tensordot/ReshapeReshape dense_31/Tensordot/transpose:y:0!dense_31/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_31/Tensordot/ReshapeÂ
dense_31/Tensordot/MatMulMatMul#dense_31/Tensordot/Reshape:output:0)dense_31/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_31/Tensordot/MatMul
dense_31/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_31/Tensordot/Const_2
 dense_31/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_31/Tensordot/concat_1/axisê
dense_31/Tensordot/concat_1ConcatV2$dense_31/Tensordot/GatherV2:output:0#dense_31/Tensordot/Const_2:output:0)dense_31/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_31/Tensordot/concat_1´
dense_31/TensordotReshape#dense_31/Tensordot/MatMul:product:0$dense_31/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_31/Tensordot§
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_31/BiasAdd/ReadVariableOp«
dense_31/BiasAddBiasAdddense_31/Tensordot:output:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_31/BiasAddý
IdentityIdentitydense_31/BiasAdd:output:0 ^dense_30/BiasAdd/ReadVariableOp"^dense_30/Tensordot/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp"^dense_31/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2F
!dense_30/Tensordot/ReadVariableOp!dense_30/Tensordot/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2F
!dense_31/Tensordot/ReadVariableOp!dense_31/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
È
©
6__inference_batch_normalization_8_layer_call_fn_515853

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
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_5136682
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
ðX

C__inference_model_4_layer_call_and_return_conditional_losses_514479
input_9
input_10)
%token_and_position_embedding_4_514389)
%token_and_position_embedding_4_514391
conv1d_8_514394
conv1d_8_514396
conv1d_9_514400
conv1d_9_514402 
batch_normalization_8_514407 
batch_normalization_8_514409 
batch_normalization_8_514411 
batch_normalization_8_514413 
batch_normalization_9_514416 
batch_normalization_9_514418 
batch_normalization_9_514420 
batch_normalization_9_514422
transformer_block_9_514426
transformer_block_9_514428
transformer_block_9_514430
transformer_block_9_514432
transformer_block_9_514434
transformer_block_9_514436
transformer_block_9_514438
transformer_block_9_514440
transformer_block_9_514442
transformer_block_9_514444
transformer_block_9_514446
transformer_block_9_514448
transformer_block_9_514450
transformer_block_9_514452
transformer_block_9_514454
transformer_block_9_514456
dense_32_514461
dense_32_514463
dense_33_514467
dense_33_514469
dense_34_514473
dense_34_514475
identity¢-batch_normalization_8/StatefulPartitionedCall¢-batch_normalization_9/StatefulPartitionedCall¢ conv1d_8/StatefulPartitionedCall¢ conv1d_9/StatefulPartitionedCall¢ dense_32/StatefulPartitionedCall¢ dense_33/StatefulPartitionedCall¢ dense_34/StatefulPartitionedCall¢6token_and_position_embedding_4/StatefulPartitionedCall¢+transformer_block_9/StatefulPartitionedCall
6token_and_position_embedding_4/StatefulPartitionedCallStatefulPartitionedCallinput_9%token_and_position_embedding_4_514389%token_and_position_embedding_4_514391*
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
Z__inference_token_and_position_embedding_4_layer_call_and_return_conditional_losses_51355028
6token_and_position_embedding_4/StatefulPartitionedCallÕ
 conv1d_8/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_4/StatefulPartitionedCall:output:0conv1d_8_514394conv1d_8_514396*
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
D__inference_conv1d_8_layer_call_and_return_conditional_losses_5135822"
 conv1d_8/StatefulPartitionedCall£
$average_pooling1d_12/PartitionedCallPartitionedCall)conv1d_8/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_12_layer_call_and_return_conditional_losses_5130382&
$average_pooling1d_12/PartitionedCallÃ
 conv1d_9/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_12/PartitionedCall:output:0conv1d_9_514400conv1d_9_514402*
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
D__inference_conv1d_9_layer_call_and_return_conditional_losses_5136152"
 conv1d_9/StatefulPartitionedCall¸
$average_pooling1d_14/PartitionedCallPartitionedCall?token_and_position_embedding_4/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_14_layer_call_and_return_conditional_losses_5130682&
$average_pooling1d_14/PartitionedCall¢
$average_pooling1d_13/PartitionedCallPartitionedCall)conv1d_9/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_13_layer_call_and_return_conditional_losses_5130532&
$average_pooling1d_13/PartitionedCallÃ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_13/PartitionedCall:output:0batch_normalization_8_514407batch_normalization_8_514409batch_normalization_8_514411batch_normalization_8_514413*
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
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_5136882/
-batch_normalization_8/StatefulPartitionedCallÃ
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_14/PartitionedCall:output:0batch_normalization_9_514416batch_normalization_9_514418batch_normalization_9_514420batch_normalization_9_514422*
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
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_5137792/
-batch_normalization_9/StatefulPartitionedCall»
add_4/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:06batch_normalization_9/StatefulPartitionedCall:output:0*
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
A__inference_add_4_layer_call_and_return_conditional_losses_5138212
add_4/PartitionedCall
+transformer_block_9/StatefulPartitionedCallStatefulPartitionedCalladd_4/PartitionedCall:output:0transformer_block_9_514426transformer_block_9_514428transformer_block_9_514430transformer_block_9_514432transformer_block_9_514434transformer_block_9_514436transformer_block_9_514438transformer_block_9_514440transformer_block_9_514442transformer_block_9_514444transformer_block_9_514446transformer_block_9_514448transformer_block_9_514450transformer_block_9_514452transformer_block_9_514454transformer_block_9_514456*
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
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_5141052-
+transformer_block_9/StatefulPartitionedCall
flatten_4/PartitionedCallPartitionedCall4transformer_block_9/StatefulPartitionedCall:output:0*
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
E__inference_flatten_4_layer_call_and_return_conditional_losses_5142202
flatten_4/PartitionedCall
concatenate_4/PartitionedCallPartitionedCall"flatten_4/PartitionedCall:output:0input_10*
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
I__inference_concatenate_4_layer_call_and_return_conditional_losses_5142352
concatenate_4/PartitionedCall·
 dense_32/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_32_514461dense_32_514463*
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
D__inference_dense_32_layer_call_and_return_conditional_losses_5142552"
 dense_32/StatefulPartitionedCall
dropout_28/PartitionedCallPartitionedCall)dense_32/StatefulPartitionedCall:output:0*
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
F__inference_dropout_28_layer_call_and_return_conditional_losses_5142882
dropout_28/PartitionedCall´
 dense_33/StatefulPartitionedCallStatefulPartitionedCall#dropout_28/PartitionedCall:output:0dense_33_514467dense_33_514469*
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
D__inference_dense_33_layer_call_and_return_conditional_losses_5143122"
 dense_33/StatefulPartitionedCall
dropout_29/PartitionedCallPartitionedCall)dense_33/StatefulPartitionedCall:output:0*
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
F__inference_dropout_29_layer_call_and_return_conditional_losses_5143452
dropout_29/PartitionedCall´
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#dropout_29/PartitionedCall:output:0dense_34_514473dense_34_514475*
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
D__inference_dense_34_layer_call_and_return_conditional_losses_5143682"
 dense_34/StatefulPartitionedCalló
IdentityIdentity)dense_34/StatefulPartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv1d_8/StatefulPartitionedCall!^conv1d_9/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall7^token_and_position_embedding_4/StatefulPartitionedCall,^transformer_block_9/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv1d_8/StatefulPartitionedCall conv1d_8/StatefulPartitionedCall2D
 conv1d_9/StatefulPartitionedCall conv1d_9/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2p
6token_and_position_embedding_4/StatefulPartitionedCall6token_and_position_embedding_4/StatefulPartitionedCall2Z
+transformer_block_9/StatefulPartitionedCall+transformer_block_9/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_9:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_10

÷
D__inference_conv1d_8_layer_call_and_return_conditional_losses_513582

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
É
d
F__inference_dropout_29_layer_call_and_return_conditional_losses_514345

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
ñ	
Ý
D__inference_dense_32_layer_call_and_return_conditional_losses_516426

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
î	
Ý
D__inference_dense_33_layer_call_and_return_conditional_losses_514312

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


Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_513203

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
D__inference_dense_31_layer_call_and_return_conditional_losses_513435

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
ô

Z__inference_token_and_position_embedding_4_layer_call_and_return_conditional_losses_513550
x'
#embedding_9_embedding_lookup_513537'
#embedding_8_embedding_lookup_513543
identity¢embedding_8/embedding_lookup¢embedding_9/embedding_lookup?
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
embedding_9/embedding_lookupResourceGather#embedding_9_embedding_lookup_513537range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_9/embedding_lookup/513537*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02
embedding_9/embedding_lookup
%embedding_9/embedding_lookup/IdentityIdentity%embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_9/embedding_lookup/513537*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%embedding_9/embedding_lookup/IdentityÀ
'embedding_9/embedding_lookup/Identity_1Identity.embedding_9/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'embedding_9/embedding_lookup/Identity_1q
embedding_8/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2
embedding_8/Castº
embedding_8/embedding_lookupResourceGather#embedding_8_embedding_lookup_513543embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_8/embedding_lookup/513543*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02
embedding_8/embedding_lookup
%embedding_8/embedding_lookup/IdentityIdentity%embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_8/embedding_lookup/513543*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2'
%embedding_8/embedding_lookup/IdentityÅ
'embedding_8/embedding_lookup/Identity_1Identity.embedding_8/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2)
'embedding_8/embedding_lookup/Identity_1®
addAddV20embedding_8/embedding_lookup/Identity_1:output:00embedding_9/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
add
IdentityIdentityadd:z:0^embedding_8/embedding_lookup^embedding_9/embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿR::2<
embedding_8/embedding_lookupembedding_8/embedding_lookup2<
embedding_9/embedding_lookupembedding_9/embedding_lookup:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR

_user_specified_namex
ÿ
â
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_516190

inputsF
Bmulti_head_attention_9_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_9_query_add_readvariableop_resourceD
@multi_head_attention_9_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_9_key_add_readvariableop_resourceF
Bmulti_head_attention_9_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_9_value_add_readvariableop_resourceQ
Mmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_9_attention_output_add_readvariableop_resource@
<layer_normalization_18_batchnorm_mul_readvariableop_resource<
8layer_normalization_18_batchnorm_readvariableop_resource;
7sequential_9_dense_30_tensordot_readvariableop_resource9
5sequential_9_dense_30_biasadd_readvariableop_resource;
7sequential_9_dense_31_tensordot_readvariableop_resource9
5sequential_9_dense_31_biasadd_readvariableop_resource@
<layer_normalization_19_batchnorm_mul_readvariableop_resource<
8layer_normalization_19_batchnorm_readvariableop_resource
identity¢/layer_normalization_18/batchnorm/ReadVariableOp¢3layer_normalization_18/batchnorm/mul/ReadVariableOp¢/layer_normalization_19/batchnorm/ReadVariableOp¢3layer_normalization_19/batchnorm/mul/ReadVariableOp¢:multi_head_attention_9/attention_output/add/ReadVariableOp¢Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_9/key/add/ReadVariableOp¢7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_9/query/add/ReadVariableOp¢9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_9/value/add/ReadVariableOp¢9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp¢,sequential_9/dense_30/BiasAdd/ReadVariableOp¢.sequential_9/dense_30/Tensordot/ReadVariableOp¢,sequential_9/dense_31/BiasAdd/ReadVariableOp¢.sequential_9/dense_31/Tensordot/ReadVariableOpý
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_9/query/einsum/EinsumEinsuminputsAmulti_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_9/query/einsum/EinsumÛ
/multi_head_attention_9/query/add/ReadVariableOpReadVariableOp8multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_9/query/add/ReadVariableOpõ
 multi_head_attention_9/query/addAddV23multi_head_attention_9/query/einsum/Einsum:output:07multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_9/query/add÷
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_9_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_9/key/einsum/EinsumEinsuminputs?multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2*
(multi_head_attention_9/key/einsum/EinsumÕ
-multi_head_attention_9/key/add/ReadVariableOpReadVariableOp6multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_9/key/add/ReadVariableOpí
multi_head_attention_9/key/addAddV21multi_head_attention_9/key/einsum/Einsum:output:05multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
multi_head_attention_9/key/addý
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_9/value/einsum/EinsumEinsuminputsAmulti_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_9/value/einsum/EinsumÛ
/multi_head_attention_9/value/add/ReadVariableOpReadVariableOp8multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_9/value/add/ReadVariableOpõ
 multi_head_attention_9/value/addAddV23multi_head_attention_9/value/einsum/Einsum:output:07multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_9/value/add
multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_9/Mul/yÆ
multi_head_attention_9/MulMul$multi_head_attention_9/query/add:z:0%multi_head_attention_9/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_9/Mulü
$multi_head_attention_9/einsum/EinsumEinsum"multi_head_attention_9/key/add:z:0multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2&
$multi_head_attention_9/einsum/EinsumÄ
&multi_head_attention_9/softmax/SoftmaxSoftmax-multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2(
&multi_head_attention_9/softmax/Softmax¡
,multi_head_attention_9/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,multi_head_attention_9/dropout/dropout/Const
*multi_head_attention_9/dropout/dropout/MulMul0multi_head_attention_9/softmax/Softmax:softmax:05multi_head_attention_9/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2,
*multi_head_attention_9/dropout/dropout/Mul¼
,multi_head_attention_9/dropout/dropout/ShapeShape0multi_head_attention_9/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_9/dropout/dropout/Shape
Cmulti_head_attention_9/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_9/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
dtype02E
Cmulti_head_attention_9/dropout/dropout/random_uniform/RandomUniform³
5multi_head_attention_9/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_9/dropout/dropout/GreaterEqual/yÂ
3multi_head_attention_9/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_9/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_9/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##25
3multi_head_attention_9/dropout/dropout/GreaterEqualä
+multi_head_attention_9/dropout/dropout/CastCast7multi_head_attention_9/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2-
+multi_head_attention_9/dropout/dropout/Castþ
,multi_head_attention_9/dropout/dropout/Mul_1Mul.multi_head_attention_9/dropout/dropout/Mul:z:0/multi_head_attention_9/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2.
,multi_head_attention_9/dropout/dropout/Mul_1
&multi_head_attention_9/einsum_1/EinsumEinsum0multi_head_attention_9/dropout/dropout/Mul_1:z:0$multi_head_attention_9/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2(
&multi_head_attention_9/einsum_1/Einsum
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_9/attention_output/einsum/EinsumEinsum/multi_head_attention_9/einsum_1/Einsum:output:0Lmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe27
5multi_head_attention_9/attention_output/einsum/Einsumø
:multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_9/attention_output/add/ReadVariableOp
+multi_head_attention_9/attention_output/addAddV2>multi_head_attention_9/attention_output/einsum/Einsum:output:0Bmulti_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+multi_head_attention_9/attention_output/addy
dropout_26/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_26/dropout/ConstÁ
dropout_26/dropout/MulMul/multi_head_attention_9/attention_output/add:z:0!dropout_26/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_26/dropout/Mul
dropout_26/dropout/ShapeShape/multi_head_attention_9/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_26/dropout/ShapeÙ
/dropout_26/dropout/random_uniform/RandomUniformRandomUniform!dropout_26/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype021
/dropout_26/dropout/random_uniform/RandomUniform
!dropout_26/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_26/dropout/GreaterEqual/yî
dropout_26/dropout/GreaterEqualGreaterEqual8dropout_26/dropout/random_uniform/RandomUniform:output:0*dropout_26/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
dropout_26/dropout/GreaterEqual¤
dropout_26/dropout/CastCast#dropout_26/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_26/dropout/Castª
dropout_26/dropout/Mul_1Muldropout_26/dropout/Mul:z:0dropout_26/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_26/dropout/Mul_1o
addAddV2inputsdropout_26/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¸
5layer_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_18/moments/mean/reduction_indicesâ
#layer_normalization_18/moments/meanMeanadd:z:0>layer_normalization_18/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_18/moments/meanÎ
+layer_normalization_18/moments/StopGradientStopGradient,layer_normalization_18/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_18/moments/StopGradientî
0layer_normalization_18/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_18/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_18/moments/SquaredDifferenceÀ
9layer_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_18/moments/variance/reduction_indices
'layer_normalization_18/moments/varianceMean4layer_normalization_18/moments/SquaredDifference:z:0Blayer_normalization_18/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_18/moments/variance
&layer_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_18/batchnorm/add/yî
$layer_normalization_18/batchnorm/addAddV20layer_normalization_18/moments/variance:output:0/layer_normalization_18/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_18/batchnorm/add¹
&layer_normalization_18/batchnorm/RsqrtRsqrt(layer_normalization_18/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_18/batchnorm/Rsqrtã
3layer_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_18/batchnorm/mul/ReadVariableOpò
$layer_normalization_18/batchnorm/mulMul*layer_normalization_18/batchnorm/Rsqrt:y:0;layer_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_18/batchnorm/mulÀ
&layer_normalization_18/batchnorm/mul_1Muladd:z:0(layer_normalization_18/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_18/batchnorm/mul_1å
&layer_normalization_18/batchnorm/mul_2Mul,layer_normalization_18/moments/mean:output:0(layer_normalization_18/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_18/batchnorm/mul_2×
/layer_normalization_18/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_18_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_18/batchnorm/ReadVariableOpî
$layer_normalization_18/batchnorm/subSub7layer_normalization_18/batchnorm/ReadVariableOp:value:0*layer_normalization_18/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_18/batchnorm/subå
&layer_normalization_18/batchnorm/add_1AddV2*layer_normalization_18/batchnorm/mul_1:z:0(layer_normalization_18/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_18/batchnorm/add_1Ø
.sequential_9/dense_30/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_30_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_9/dense_30/Tensordot/ReadVariableOp
$sequential_9/dense_30/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_9/dense_30/Tensordot/axes
$sequential_9/dense_30/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_9/dense_30/Tensordot/free¨
%sequential_9/dense_30/Tensordot/ShapeShape*layer_normalization_18/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_9/dense_30/Tensordot/Shape 
-sequential_9/dense_30/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_30/Tensordot/GatherV2/axis¿
(sequential_9/dense_30/Tensordot/GatherV2GatherV2.sequential_9/dense_30/Tensordot/Shape:output:0-sequential_9/dense_30/Tensordot/free:output:06sequential_9/dense_30/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_9/dense_30/Tensordot/GatherV2¤
/sequential_9/dense_30/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_9/dense_30/Tensordot/GatherV2_1/axisÅ
*sequential_9/dense_30/Tensordot/GatherV2_1GatherV2.sequential_9/dense_30/Tensordot/Shape:output:0-sequential_9/dense_30/Tensordot/axes:output:08sequential_9/dense_30/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_9/dense_30/Tensordot/GatherV2_1
%sequential_9/dense_30/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_9/dense_30/Tensordot/ConstØ
$sequential_9/dense_30/Tensordot/ProdProd1sequential_9/dense_30/Tensordot/GatherV2:output:0.sequential_9/dense_30/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_9/dense_30/Tensordot/Prod
'sequential_9/dense_30/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_30/Tensordot/Const_1à
&sequential_9/dense_30/Tensordot/Prod_1Prod3sequential_9/dense_30/Tensordot/GatherV2_1:output:00sequential_9/dense_30/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_9/dense_30/Tensordot/Prod_1
+sequential_9/dense_30/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_9/dense_30/Tensordot/concat/axis
&sequential_9/dense_30/Tensordot/concatConcatV2-sequential_9/dense_30/Tensordot/free:output:0-sequential_9/dense_30/Tensordot/axes:output:04sequential_9/dense_30/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/dense_30/Tensordot/concatä
%sequential_9/dense_30/Tensordot/stackPack-sequential_9/dense_30/Tensordot/Prod:output:0/sequential_9/dense_30/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/dense_30/Tensordot/stackö
)sequential_9/dense_30/Tensordot/transpose	Transpose*layer_normalization_18/batchnorm/add_1:z:0/sequential_9/dense_30/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2+
)sequential_9/dense_30/Tensordot/transpose÷
'sequential_9/dense_30/Tensordot/ReshapeReshape-sequential_9/dense_30/Tensordot/transpose:y:0.sequential_9/dense_30/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_9/dense_30/Tensordot/Reshapeö
&sequential_9/dense_30/Tensordot/MatMulMatMul0sequential_9/dense_30/Tensordot/Reshape:output:06sequential_9/dense_30/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential_9/dense_30/Tensordot/MatMul
'sequential_9/dense_30/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_9/dense_30/Tensordot/Const_2 
-sequential_9/dense_30/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_30/Tensordot/concat_1/axis«
(sequential_9/dense_30/Tensordot/concat_1ConcatV21sequential_9/dense_30/Tensordot/GatherV2:output:00sequential_9/dense_30/Tensordot/Const_2:output:06sequential_9/dense_30/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_9/dense_30/Tensordot/concat_1è
sequential_9/dense_30/TensordotReshape0sequential_9/dense_30/Tensordot/MatMul:product:01sequential_9/dense_30/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2!
sequential_9/dense_30/TensordotÎ
,sequential_9/dense_30/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_9/dense_30/BiasAdd/ReadVariableOpß
sequential_9/dense_30/BiasAddBiasAdd(sequential_9/dense_30/Tensordot:output:04sequential_9/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_9/dense_30/BiasAdd
sequential_9/dense_30/ReluRelu&sequential_9/dense_30/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_9/dense_30/ReluØ
.sequential_9/dense_31/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_31_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_9/dense_31/Tensordot/ReadVariableOp
$sequential_9/dense_31/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_9/dense_31/Tensordot/axes
$sequential_9/dense_31/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_9/dense_31/Tensordot/free¦
%sequential_9/dense_31/Tensordot/ShapeShape(sequential_9/dense_30/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dense_31/Tensordot/Shape 
-sequential_9/dense_31/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_31/Tensordot/GatherV2/axis¿
(sequential_9/dense_31/Tensordot/GatherV2GatherV2.sequential_9/dense_31/Tensordot/Shape:output:0-sequential_9/dense_31/Tensordot/free:output:06sequential_9/dense_31/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_9/dense_31/Tensordot/GatherV2¤
/sequential_9/dense_31/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_9/dense_31/Tensordot/GatherV2_1/axisÅ
*sequential_9/dense_31/Tensordot/GatherV2_1GatherV2.sequential_9/dense_31/Tensordot/Shape:output:0-sequential_9/dense_31/Tensordot/axes:output:08sequential_9/dense_31/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_9/dense_31/Tensordot/GatherV2_1
%sequential_9/dense_31/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_9/dense_31/Tensordot/ConstØ
$sequential_9/dense_31/Tensordot/ProdProd1sequential_9/dense_31/Tensordot/GatherV2:output:0.sequential_9/dense_31/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_9/dense_31/Tensordot/Prod
'sequential_9/dense_31/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_31/Tensordot/Const_1à
&sequential_9/dense_31/Tensordot/Prod_1Prod3sequential_9/dense_31/Tensordot/GatherV2_1:output:00sequential_9/dense_31/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_9/dense_31/Tensordot/Prod_1
+sequential_9/dense_31/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_9/dense_31/Tensordot/concat/axis
&sequential_9/dense_31/Tensordot/concatConcatV2-sequential_9/dense_31/Tensordot/free:output:0-sequential_9/dense_31/Tensordot/axes:output:04sequential_9/dense_31/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/dense_31/Tensordot/concatä
%sequential_9/dense_31/Tensordot/stackPack-sequential_9/dense_31/Tensordot/Prod:output:0/sequential_9/dense_31/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/dense_31/Tensordot/stackô
)sequential_9/dense_31/Tensordot/transpose	Transpose(sequential_9/dense_30/Relu:activations:0/sequential_9/dense_31/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2+
)sequential_9/dense_31/Tensordot/transpose÷
'sequential_9/dense_31/Tensordot/ReshapeReshape-sequential_9/dense_31/Tensordot/transpose:y:0.sequential_9/dense_31/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_9/dense_31/Tensordot/Reshapeö
&sequential_9/dense_31/Tensordot/MatMulMatMul0sequential_9/dense_31/Tensordot/Reshape:output:06sequential_9/dense_31/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_9/dense_31/Tensordot/MatMul
'sequential_9/dense_31/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_31/Tensordot/Const_2 
-sequential_9/dense_31/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_31/Tensordot/concat_1/axis«
(sequential_9/dense_31/Tensordot/concat_1ConcatV21sequential_9/dense_31/Tensordot/GatherV2:output:00sequential_9/dense_31/Tensordot/Const_2:output:06sequential_9/dense_31/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_9/dense_31/Tensordot/concat_1è
sequential_9/dense_31/TensordotReshape0sequential_9/dense_31/Tensordot/MatMul:product:01sequential_9/dense_31/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
sequential_9/dense_31/TensordotÎ
,sequential_9/dense_31/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_9/dense_31/BiasAdd/ReadVariableOpß
sequential_9/dense_31/BiasAddBiasAdd(sequential_9/dense_31/Tensordot:output:04sequential_9/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
sequential_9/dense_31/BiasAddy
dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_27/dropout/Const¸
dropout_27/dropout/MulMul&sequential_9/dense_31/BiasAdd:output:0!dropout_27/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_27/dropout/Mul
dropout_27/dropout/ShapeShape&sequential_9/dense_31/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_27/dropout/ShapeÙ
/dropout_27/dropout/random_uniform/RandomUniformRandomUniform!dropout_27/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype021
/dropout_27/dropout/random_uniform/RandomUniform
!dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_27/dropout/GreaterEqual/yî
dropout_27/dropout/GreaterEqualGreaterEqual8dropout_27/dropout/random_uniform/RandomUniform:output:0*dropout_27/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
dropout_27/dropout/GreaterEqual¤
dropout_27/dropout/CastCast#dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_27/dropout/Castª
dropout_27/dropout/Mul_1Muldropout_27/dropout/Mul:z:0dropout_27/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_27/dropout/Mul_1
add_1AddV2*layer_normalization_18/batchnorm/add_1:z:0dropout_27/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¸
5layer_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_19/moments/mean/reduction_indicesä
#layer_normalization_19/moments/meanMean	add_1:z:0>layer_normalization_19/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_19/moments/meanÎ
+layer_normalization_19/moments/StopGradientStopGradient,layer_normalization_19/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_19/moments/StopGradientð
0layer_normalization_19/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_19/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_19/moments/SquaredDifferenceÀ
9layer_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_19/moments/variance/reduction_indices
'layer_normalization_19/moments/varianceMean4layer_normalization_19/moments/SquaredDifference:z:0Blayer_normalization_19/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_19/moments/variance
&layer_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_19/batchnorm/add/yî
$layer_normalization_19/batchnorm/addAddV20layer_normalization_19/moments/variance:output:0/layer_normalization_19/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_19/batchnorm/add¹
&layer_normalization_19/batchnorm/RsqrtRsqrt(layer_normalization_19/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_19/batchnorm/Rsqrtã
3layer_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_19/batchnorm/mul/ReadVariableOpò
$layer_normalization_19/batchnorm/mulMul*layer_normalization_19/batchnorm/Rsqrt:y:0;layer_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_19/batchnorm/mulÂ
&layer_normalization_19/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_19/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_19/batchnorm/mul_1å
&layer_normalization_19/batchnorm/mul_2Mul,layer_normalization_19/moments/mean:output:0(layer_normalization_19/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_19/batchnorm/mul_2×
/layer_normalization_19/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_19_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_19/batchnorm/ReadVariableOpî
$layer_normalization_19/batchnorm/subSub7layer_normalization_19/batchnorm/ReadVariableOp:value:0*layer_normalization_19/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_19/batchnorm/subå
&layer_normalization_19/batchnorm/add_1AddV2*layer_normalization_19/batchnorm/mul_1:z:0(layer_normalization_19/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_19/batchnorm/add_1Ü
IdentityIdentity*layer_normalization_19/batchnorm/add_1:z:00^layer_normalization_18/batchnorm/ReadVariableOp4^layer_normalization_18/batchnorm/mul/ReadVariableOp0^layer_normalization_19/batchnorm/ReadVariableOp4^layer_normalization_19/batchnorm/mul/ReadVariableOp;^multi_head_attention_9/attention_output/add/ReadVariableOpE^multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_9/key/add/ReadVariableOp8^multi_head_attention_9/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/query/add/ReadVariableOp:^multi_head_attention_9/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/value/add/ReadVariableOp:^multi_head_attention_9/value/einsum/Einsum/ReadVariableOp-^sequential_9/dense_30/BiasAdd/ReadVariableOp/^sequential_9/dense_30/Tensordot/ReadVariableOp-^sequential_9/dense_31/BiasAdd/ReadVariableOp/^sequential_9/dense_31/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2b
/layer_normalization_18/batchnorm/ReadVariableOp/layer_normalization_18/batchnorm/ReadVariableOp2j
3layer_normalization_18/batchnorm/mul/ReadVariableOp3layer_normalization_18/batchnorm/mul/ReadVariableOp2b
/layer_normalization_19/batchnorm/ReadVariableOp/layer_normalization_19/batchnorm/ReadVariableOp2j
3layer_normalization_19/batchnorm/mul/ReadVariableOp3layer_normalization_19/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_9/attention_output/add/ReadVariableOp:multi_head_attention_9/attention_output/add/ReadVariableOp2
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_9/key/add/ReadVariableOp-multi_head_attention_9/key/add/ReadVariableOp2r
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/query/add/ReadVariableOp/multi_head_attention_9/query/add/ReadVariableOp2v
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/value/add/ReadVariableOp/multi_head_attention_9/value/add/ReadVariableOp2v
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp2\
,sequential_9/dense_30/BiasAdd/ReadVariableOp,sequential_9/dense_30/BiasAdd/ReadVariableOp2`
.sequential_9/dense_30/Tensordot/ReadVariableOp.sequential_9/dense_30/Tensordot/ReadVariableOp2\
,sequential_9/dense_31/BiasAdd/ReadVariableOp,sequential_9/dense_31/BiasAdd/ReadVariableOp2`
.sequential_9/dense_31/Tensordot/ReadVariableOp.sequential_9/dense_31/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
ô

Z__inference_token_and_position_embedding_4_layer_call_and_return_conditional_losses_515643
x'
#embedding_9_embedding_lookup_515630'
#embedding_8_embedding_lookup_515636
identity¢embedding_8/embedding_lookup¢embedding_9/embedding_lookup?
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
embedding_9/embedding_lookupResourceGather#embedding_9_embedding_lookup_515630range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_9/embedding_lookup/515630*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02
embedding_9/embedding_lookup
%embedding_9/embedding_lookup/IdentityIdentity%embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_9/embedding_lookup/515630*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%embedding_9/embedding_lookup/IdentityÀ
'embedding_9/embedding_lookup/Identity_1Identity.embedding_9/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'embedding_9/embedding_lookup/Identity_1q
embedding_8/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2
embedding_8/Castº
embedding_8/embedding_lookupResourceGather#embedding_8_embedding_lookup_515636embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_8/embedding_lookup/515636*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02
embedding_8/embedding_lookup
%embedding_8/embedding_lookup/IdentityIdentity%embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_8/embedding_lookup/515636*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2'
%embedding_8/embedding_lookup/IdentityÅ
'embedding_8/embedding_lookup/Identity_1Identity.embedding_8/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2)
'embedding_8/embedding_lookup/Identity_1®
addAddV20embedding_8/embedding_lookup/Identity_1:output:00embedding_9/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
add
IdentityIdentityadd:z:0^embedding_8/embedding_lookup^embedding_9/embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿR::2<
embedding_8/embedding_lookupembedding_8/embedding_lookup2<
embedding_9/embedding_lookupembedding_9/embedding_lookup:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR

_user_specified_namex
·
k
A__inference_add_4_layer_call_and_return_conditional_losses_513821

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
Ð

à
4__inference_transformer_block_9_layer_call_fn_516354

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
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_5139782
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
ó0
È
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_513310

inputs
assignmovingavg_513285
assignmovingavg_1_513291)
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
loc:@AssignMovingAvg/513285*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_513285*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/513285*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/513285*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_513285AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/513285*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/513291*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_513291*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/513291*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/513291*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_513291AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/513291*
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
ëX

C__inference_model_4_layer_call_and_return_conditional_losses_514749

inputs
inputs_1)
%token_and_position_embedding_4_514659)
%token_and_position_embedding_4_514661
conv1d_8_514664
conv1d_8_514666
conv1d_9_514670
conv1d_9_514672 
batch_normalization_8_514677 
batch_normalization_8_514679 
batch_normalization_8_514681 
batch_normalization_8_514683 
batch_normalization_9_514686 
batch_normalization_9_514688 
batch_normalization_9_514690 
batch_normalization_9_514692
transformer_block_9_514696
transformer_block_9_514698
transformer_block_9_514700
transformer_block_9_514702
transformer_block_9_514704
transformer_block_9_514706
transformer_block_9_514708
transformer_block_9_514710
transformer_block_9_514712
transformer_block_9_514714
transformer_block_9_514716
transformer_block_9_514718
transformer_block_9_514720
transformer_block_9_514722
transformer_block_9_514724
transformer_block_9_514726
dense_32_514731
dense_32_514733
dense_33_514737
dense_33_514739
dense_34_514743
dense_34_514745
identity¢-batch_normalization_8/StatefulPartitionedCall¢-batch_normalization_9/StatefulPartitionedCall¢ conv1d_8/StatefulPartitionedCall¢ conv1d_9/StatefulPartitionedCall¢ dense_32/StatefulPartitionedCall¢ dense_33/StatefulPartitionedCall¢ dense_34/StatefulPartitionedCall¢6token_and_position_embedding_4/StatefulPartitionedCall¢+transformer_block_9/StatefulPartitionedCall
6token_and_position_embedding_4/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_4_514659%token_and_position_embedding_4_514661*
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
Z__inference_token_and_position_embedding_4_layer_call_and_return_conditional_losses_51355028
6token_and_position_embedding_4/StatefulPartitionedCallÕ
 conv1d_8/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_4/StatefulPartitionedCall:output:0conv1d_8_514664conv1d_8_514666*
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
D__inference_conv1d_8_layer_call_and_return_conditional_losses_5135822"
 conv1d_8/StatefulPartitionedCall£
$average_pooling1d_12/PartitionedCallPartitionedCall)conv1d_8/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_12_layer_call_and_return_conditional_losses_5130382&
$average_pooling1d_12/PartitionedCallÃ
 conv1d_9/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_12/PartitionedCall:output:0conv1d_9_514670conv1d_9_514672*
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
D__inference_conv1d_9_layer_call_and_return_conditional_losses_5136152"
 conv1d_9/StatefulPartitionedCall¸
$average_pooling1d_14/PartitionedCallPartitionedCall?token_and_position_embedding_4/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_14_layer_call_and_return_conditional_losses_5130682&
$average_pooling1d_14/PartitionedCall¢
$average_pooling1d_13/PartitionedCallPartitionedCall)conv1d_9/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_13_layer_call_and_return_conditional_losses_5130532&
$average_pooling1d_13/PartitionedCallÃ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_13/PartitionedCall:output:0batch_normalization_8_514677batch_normalization_8_514679batch_normalization_8_514681batch_normalization_8_514683*
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
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_5136882/
-batch_normalization_8/StatefulPartitionedCallÃ
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_14/PartitionedCall:output:0batch_normalization_9_514686batch_normalization_9_514688batch_normalization_9_514690batch_normalization_9_514692*
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
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_5137792/
-batch_normalization_9/StatefulPartitionedCall»
add_4/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:06batch_normalization_9/StatefulPartitionedCall:output:0*
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
A__inference_add_4_layer_call_and_return_conditional_losses_5138212
add_4/PartitionedCall
+transformer_block_9/StatefulPartitionedCallStatefulPartitionedCalladd_4/PartitionedCall:output:0transformer_block_9_514696transformer_block_9_514698transformer_block_9_514700transformer_block_9_514702transformer_block_9_514704transformer_block_9_514706transformer_block_9_514708transformer_block_9_514710transformer_block_9_514712transformer_block_9_514714transformer_block_9_514716transformer_block_9_514718transformer_block_9_514720transformer_block_9_514722transformer_block_9_514724transformer_block_9_514726*
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
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_5141052-
+transformer_block_9/StatefulPartitionedCall
flatten_4/PartitionedCallPartitionedCall4transformer_block_9/StatefulPartitionedCall:output:0*
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
E__inference_flatten_4_layer_call_and_return_conditional_losses_5142202
flatten_4/PartitionedCall
concatenate_4/PartitionedCallPartitionedCall"flatten_4/PartitionedCall:output:0inputs_1*
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
I__inference_concatenate_4_layer_call_and_return_conditional_losses_5142352
concatenate_4/PartitionedCall·
 dense_32/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_32_514731dense_32_514733*
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
D__inference_dense_32_layer_call_and_return_conditional_losses_5142552"
 dense_32/StatefulPartitionedCall
dropout_28/PartitionedCallPartitionedCall)dense_32/StatefulPartitionedCall:output:0*
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
F__inference_dropout_28_layer_call_and_return_conditional_losses_5142882
dropout_28/PartitionedCall´
 dense_33/StatefulPartitionedCallStatefulPartitionedCall#dropout_28/PartitionedCall:output:0dense_33_514737dense_33_514739*
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
D__inference_dense_33_layer_call_and_return_conditional_losses_5143122"
 dense_33/StatefulPartitionedCall
dropout_29/PartitionedCallPartitionedCall)dense_33/StatefulPartitionedCall:output:0*
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
F__inference_dropout_29_layer_call_and_return_conditional_losses_5143452
dropout_29/PartitionedCall´
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#dropout_29/PartitionedCall:output:0dense_34_514743dense_34_514745*
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
D__inference_dense_34_layer_call_and_return_conditional_losses_5143682"
 dense_34/StatefulPartitionedCalló
IdentityIdentity)dense_34/StatefulPartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv1d_8/StatefulPartitionedCall!^conv1d_9/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall7^token_and_position_embedding_4/StatefulPartitionedCall,^transformer_block_9/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv1d_8/StatefulPartitionedCall conv1d_8/StatefulPartitionedCall2D
 conv1d_9/StatefulPartitionedCall conv1d_9/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2p
6token_and_position_embedding_4/StatefulPartitionedCall6token_and_position_embedding_4/StatefulPartitionedCall2Z
+transformer_block_9/StatefulPartitionedCall+transformer_block_9/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼0
È
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_515984

inputs
assignmovingavg_515959
assignmovingavg_1_515965)
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
loc:@AssignMovingAvg/515959*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_515959*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/515959*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/515959*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_515959AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/515959*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/515965*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_515965*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/515965*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/515965*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_515965AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/515965*
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

é(
!__inference__wrapped_model_513029
input_9
input_10N
Jmodel_4_token_and_position_embedding_4_embedding_9_embedding_lookup_512798N
Jmodel_4_token_and_position_embedding_4_embedding_8_embedding_lookup_512804@
<model_4_conv1d_8_conv1d_expanddims_1_readvariableop_resource4
0model_4_conv1d_8_biasadd_readvariableop_resource@
<model_4_conv1d_9_conv1d_expanddims_1_readvariableop_resource4
0model_4_conv1d_9_biasadd_readvariableop_resourceC
?model_4_batch_normalization_8_batchnorm_readvariableop_resourceG
Cmodel_4_batch_normalization_8_batchnorm_mul_readvariableop_resourceE
Amodel_4_batch_normalization_8_batchnorm_readvariableop_1_resourceE
Amodel_4_batch_normalization_8_batchnorm_readvariableop_2_resourceC
?model_4_batch_normalization_9_batchnorm_readvariableop_resourceG
Cmodel_4_batch_normalization_9_batchnorm_mul_readvariableop_resourceE
Amodel_4_batch_normalization_9_batchnorm_readvariableop_1_resourceE
Amodel_4_batch_normalization_9_batchnorm_readvariableop_2_resourceb
^model_4_transformer_block_9_multi_head_attention_9_query_einsum_einsum_readvariableop_resourceX
Tmodel_4_transformer_block_9_multi_head_attention_9_query_add_readvariableop_resource`
\model_4_transformer_block_9_multi_head_attention_9_key_einsum_einsum_readvariableop_resourceV
Rmodel_4_transformer_block_9_multi_head_attention_9_key_add_readvariableop_resourceb
^model_4_transformer_block_9_multi_head_attention_9_value_einsum_einsum_readvariableop_resourceX
Tmodel_4_transformer_block_9_multi_head_attention_9_value_add_readvariableop_resourcem
imodel_4_transformer_block_9_multi_head_attention_9_attention_output_einsum_einsum_readvariableop_resourcec
_model_4_transformer_block_9_multi_head_attention_9_attention_output_add_readvariableop_resource\
Xmodel_4_transformer_block_9_layer_normalization_18_batchnorm_mul_readvariableop_resourceX
Tmodel_4_transformer_block_9_layer_normalization_18_batchnorm_readvariableop_resourceW
Smodel_4_transformer_block_9_sequential_9_dense_30_tensordot_readvariableop_resourceU
Qmodel_4_transformer_block_9_sequential_9_dense_30_biasadd_readvariableop_resourceW
Smodel_4_transformer_block_9_sequential_9_dense_31_tensordot_readvariableop_resourceU
Qmodel_4_transformer_block_9_sequential_9_dense_31_biasadd_readvariableop_resource\
Xmodel_4_transformer_block_9_layer_normalization_19_batchnorm_mul_readvariableop_resourceX
Tmodel_4_transformer_block_9_layer_normalization_19_batchnorm_readvariableop_resource3
/model_4_dense_32_matmul_readvariableop_resource4
0model_4_dense_32_biasadd_readvariableop_resource3
/model_4_dense_33_matmul_readvariableop_resource4
0model_4_dense_33_biasadd_readvariableop_resource3
/model_4_dense_34_matmul_readvariableop_resource4
0model_4_dense_34_biasadd_readvariableop_resource
identity¢6model_4/batch_normalization_8/batchnorm/ReadVariableOp¢8model_4/batch_normalization_8/batchnorm/ReadVariableOp_1¢8model_4/batch_normalization_8/batchnorm/ReadVariableOp_2¢:model_4/batch_normalization_8/batchnorm/mul/ReadVariableOp¢6model_4/batch_normalization_9/batchnorm/ReadVariableOp¢8model_4/batch_normalization_9/batchnorm/ReadVariableOp_1¢8model_4/batch_normalization_9/batchnorm/ReadVariableOp_2¢:model_4/batch_normalization_9/batchnorm/mul/ReadVariableOp¢'model_4/conv1d_8/BiasAdd/ReadVariableOp¢3model_4/conv1d_8/conv1d/ExpandDims_1/ReadVariableOp¢'model_4/conv1d_9/BiasAdd/ReadVariableOp¢3model_4/conv1d_9/conv1d/ExpandDims_1/ReadVariableOp¢'model_4/dense_32/BiasAdd/ReadVariableOp¢&model_4/dense_32/MatMul/ReadVariableOp¢'model_4/dense_33/BiasAdd/ReadVariableOp¢&model_4/dense_33/MatMul/ReadVariableOp¢'model_4/dense_34/BiasAdd/ReadVariableOp¢&model_4/dense_34/MatMul/ReadVariableOp¢Cmodel_4/token_and_position_embedding_4/embedding_8/embedding_lookup¢Cmodel_4/token_and_position_embedding_4/embedding_9/embedding_lookup¢Kmodel_4/transformer_block_9/layer_normalization_18/batchnorm/ReadVariableOp¢Omodel_4/transformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOp¢Kmodel_4/transformer_block_9/layer_normalization_19/batchnorm/ReadVariableOp¢Omodel_4/transformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOp¢Vmodel_4/transformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOp¢`model_4/transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp¢Imodel_4/transformer_block_9/multi_head_attention_9/key/add/ReadVariableOp¢Smodel_4/transformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp¢Kmodel_4/transformer_block_9/multi_head_attention_9/query/add/ReadVariableOp¢Umodel_4/transformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp¢Kmodel_4/transformer_block_9/multi_head_attention_9/value/add/ReadVariableOp¢Umodel_4/transformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp¢Hmodel_4/transformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOp¢Jmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOp¢Hmodel_4/transformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOp¢Jmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOp
,model_4/token_and_position_embedding_4/ShapeShapeinput_9*
T0*
_output_shapes
:2.
,model_4/token_and_position_embedding_4/ShapeË
:model_4/token_and_position_embedding_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2<
:model_4/token_and_position_embedding_4/strided_slice/stackÆ
<model_4/token_and_position_embedding_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<model_4/token_and_position_embedding_4/strided_slice/stack_1Æ
<model_4/token_and_position_embedding_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model_4/token_and_position_embedding_4/strided_slice/stack_2Ì
4model_4/token_and_position_embedding_4/strided_sliceStridedSlice5model_4/token_and_position_embedding_4/Shape:output:0Cmodel_4/token_and_position_embedding_4/strided_slice/stack:output:0Emodel_4/token_and_position_embedding_4/strided_slice/stack_1:output:0Emodel_4/token_and_position_embedding_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4model_4/token_and_position_embedding_4/strided_sliceª
2model_4/token_and_position_embedding_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : 24
2model_4/token_and_position_embedding_4/range/startª
2model_4/token_and_position_embedding_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :24
2model_4/token_and_position_embedding_4/range/deltaÃ
,model_4/token_and_position_embedding_4/rangeRange;model_4/token_and_position_embedding_4/range/start:output:0=model_4/token_and_position_embedding_4/strided_slice:output:0;model_4/token_and_position_embedding_4/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,model_4/token_and_position_embedding_4/rangeò
Cmodel_4/token_and_position_embedding_4/embedding_9/embedding_lookupResourceGatherJmodel_4_token_and_position_embedding_4_embedding_9_embedding_lookup_5127985model_4/token_and_position_embedding_4/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*]
_classS
QOloc:@model_4/token_and_position_embedding_4/embedding_9/embedding_lookup/512798*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02E
Cmodel_4/token_and_position_embedding_4/embedding_9/embedding_lookupµ
Lmodel_4/token_and_position_embedding_4/embedding_9/embedding_lookup/IdentityIdentityLmodel_4/token_and_position_embedding_4/embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*]
_classS
QOloc:@model_4/token_and_position_embedding_4/embedding_9/embedding_lookup/512798*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2N
Lmodel_4/token_and_position_embedding_4/embedding_9/embedding_lookup/Identityµ
Nmodel_4/token_and_position_embedding_4/embedding_9/embedding_lookup/Identity_1IdentityUmodel_4/token_and_position_embedding_4/embedding_9/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2P
Nmodel_4/token_and_position_embedding_4/embedding_9/embedding_lookup/Identity_1Å
7model_4/token_and_position_embedding_4/embedding_8/CastCastinput_9*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR29
7model_4/token_and_position_embedding_4/embedding_8/Castý
Cmodel_4/token_and_position_embedding_4/embedding_8/embedding_lookupResourceGatherJmodel_4_token_and_position_embedding_4_embedding_8_embedding_lookup_512804;model_4/token_and_position_embedding_4/embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*]
_classS
QOloc:@model_4/token_and_position_embedding_4/embedding_8/embedding_lookup/512804*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02E
Cmodel_4/token_and_position_embedding_4/embedding_8/embedding_lookupº
Lmodel_4/token_and_position_embedding_4/embedding_8/embedding_lookup/IdentityIdentityLmodel_4/token_and_position_embedding_4/embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*]
_classS
QOloc:@model_4/token_and_position_embedding_4/embedding_8/embedding_lookup/512804*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2N
Lmodel_4/token_and_position_embedding_4/embedding_8/embedding_lookup/Identityº
Nmodel_4/token_and_position_embedding_4/embedding_8/embedding_lookup/Identity_1IdentityUmodel_4/token_and_position_embedding_4/embedding_8/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2P
Nmodel_4/token_and_position_embedding_4/embedding_8/embedding_lookup/Identity_1Ê
*model_4/token_and_position_embedding_4/addAddV2Wmodel_4/token_and_position_embedding_4/embedding_8/embedding_lookup/Identity_1:output:0Wmodel_4/token_and_position_embedding_4/embedding_9/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2,
*model_4/token_and_position_embedding_4/add
&model_4/conv1d_8/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2(
&model_4/conv1d_8/conv1d/ExpandDims/dimò
"model_4/conv1d_8/conv1d/ExpandDims
ExpandDims.model_4/token_and_position_embedding_4/add:z:0/model_4/conv1d_8/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2$
"model_4/conv1d_8/conv1d/ExpandDimsë
3model_4/conv1d_8/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<model_4_conv1d_8_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype025
3model_4/conv1d_8/conv1d/ExpandDims_1/ReadVariableOp
(model_4/conv1d_8/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_4/conv1d_8/conv1d/ExpandDims_1/dimû
$model_4/conv1d_8/conv1d/ExpandDims_1
ExpandDims;model_4/conv1d_8/conv1d/ExpandDims_1/ReadVariableOp:value:01model_4/conv1d_8/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2&
$model_4/conv1d_8/conv1d/ExpandDims_1û
model_4/conv1d_8/conv1dConv2D+model_4/conv1d_8/conv1d/ExpandDims:output:0-model_4/conv1d_8/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
paddingSAME*
strides
2
model_4/conv1d_8/conv1dÆ
model_4/conv1d_8/conv1d/SqueezeSqueeze model_4/conv1d_8/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2!
model_4/conv1d_8/conv1d/Squeeze¿
'model_4/conv1d_8/BiasAdd/ReadVariableOpReadVariableOp0model_4_conv1d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_4/conv1d_8/BiasAdd/ReadVariableOpÑ
model_4/conv1d_8/BiasAddBiasAdd(model_4/conv1d_8/conv1d/Squeeze:output:0/model_4/conv1d_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
model_4/conv1d_8/BiasAdd
model_4/conv1d_8/ReluRelu!model_4/conv1d_8/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
model_4/conv1d_8/Relu
+model_4/average_pooling1d_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+model_4/average_pooling1d_12/ExpandDims/dimö
'model_4/average_pooling1d_12/ExpandDims
ExpandDims#model_4/conv1d_8/Relu:activations:04model_4/average_pooling1d_12/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2)
'model_4/average_pooling1d_12/ExpandDims
$model_4/average_pooling1d_12/AvgPoolAvgPool0model_4/average_pooling1d_12/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
ksize
*
paddingVALID*
strides
2&
$model_4/average_pooling1d_12/AvgPoolÔ
$model_4/average_pooling1d_12/SqueezeSqueeze-model_4/average_pooling1d_12/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims
2&
$model_4/average_pooling1d_12/Squeeze
&model_4/conv1d_9/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2(
&model_4/conv1d_9/conv1d/ExpandDims/dimñ
"model_4/conv1d_9/conv1d/ExpandDims
ExpandDims-model_4/average_pooling1d_12/Squeeze:output:0/model_4/conv1d_9/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2$
"model_4/conv1d_9/conv1d/ExpandDimsë
3model_4/conv1d_9/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<model_4_conv1d_9_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype025
3model_4/conv1d_9/conv1d/ExpandDims_1/ReadVariableOp
(model_4/conv1d_9/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_4/conv1d_9/conv1d/ExpandDims_1/dimû
$model_4/conv1d_9/conv1d/ExpandDims_1
ExpandDims;model_4/conv1d_9/conv1d/ExpandDims_1/ReadVariableOp:value:01model_4/conv1d_9/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2&
$model_4/conv1d_9/conv1d/ExpandDims_1û
model_4/conv1d_9/conv1dConv2D+model_4/conv1d_9/conv1d/ExpandDims:output:0-model_4/conv1d_9/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
paddingSAME*
strides
2
model_4/conv1d_9/conv1dÆ
model_4/conv1d_9/conv1d/SqueezeSqueeze model_4/conv1d_9/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2!
model_4/conv1d_9/conv1d/Squeeze¿
'model_4/conv1d_9/BiasAdd/ReadVariableOpReadVariableOp0model_4_conv1d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_4/conv1d_9/BiasAdd/ReadVariableOpÑ
model_4/conv1d_9/BiasAddBiasAdd(model_4/conv1d_9/conv1d/Squeeze:output:0/model_4/conv1d_9/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
model_4/conv1d_9/BiasAdd
model_4/conv1d_9/ReluRelu!model_4/conv1d_9/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
model_4/conv1d_9/Relu
+model_4/average_pooling1d_14/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+model_4/average_pooling1d_14/ExpandDims/dim
'model_4/average_pooling1d_14/ExpandDims
ExpandDims.model_4/token_and_position_embedding_4/add:z:04model_4/average_pooling1d_14/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2)
'model_4/average_pooling1d_14/ExpandDims
$model_4/average_pooling1d_14/AvgPoolAvgPool0model_4/average_pooling1d_14/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize	
¬*
paddingVALID*
strides	
¬2&
$model_4/average_pooling1d_14/AvgPoolÓ
$model_4/average_pooling1d_14/SqueezeSqueeze-model_4/average_pooling1d_14/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2&
$model_4/average_pooling1d_14/Squeeze
+model_4/average_pooling1d_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+model_4/average_pooling1d_13/ExpandDims/dimö
'model_4/average_pooling1d_13/ExpandDims
ExpandDims#model_4/conv1d_9/Relu:activations:04model_4/average_pooling1d_13/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2)
'model_4/average_pooling1d_13/ExpandDimsÿ
$model_4/average_pooling1d_13/AvgPoolAvgPool0model_4/average_pooling1d_13/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize

*
paddingVALID*
strides

2&
$model_4/average_pooling1d_13/AvgPoolÓ
$model_4/average_pooling1d_13/SqueezeSqueeze-model_4/average_pooling1d_13/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2&
$model_4/average_pooling1d_13/Squeezeì
6model_4/batch_normalization_8/batchnorm/ReadVariableOpReadVariableOp?model_4_batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype028
6model_4/batch_normalization_8/batchnorm/ReadVariableOp£
-model_4/batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-model_4/batch_normalization_8/batchnorm/add/y
+model_4/batch_normalization_8/batchnorm/addAddV2>model_4/batch_normalization_8/batchnorm/ReadVariableOp:value:06model_4/batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2-
+model_4/batch_normalization_8/batchnorm/add½
-model_4/batch_normalization_8/batchnorm/RsqrtRsqrt/model_4/batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-model_4/batch_normalization_8/batchnorm/Rsqrtø
:model_4/batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_4_batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:model_4/batch_normalization_8/batchnorm/mul/ReadVariableOpý
+model_4/batch_normalization_8/batchnorm/mulMul1model_4/batch_normalization_8/batchnorm/Rsqrt:y:0Bmodel_4/batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+model_4/batch_normalization_8/batchnorm/mulû
-model_4/batch_normalization_8/batchnorm/mul_1Mul-model_4/average_pooling1d_13/Squeeze:output:0/model_4/batch_normalization_8/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2/
-model_4/batch_normalization_8/batchnorm/mul_1ò
8model_4/batch_normalization_8/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_4_batch_normalization_8_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model_4/batch_normalization_8/batchnorm/ReadVariableOp_1ý
-model_4/batch_normalization_8/batchnorm/mul_2Mul@model_4/batch_normalization_8/batchnorm/ReadVariableOp_1:value:0/model_4/batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-model_4/batch_normalization_8/batchnorm/mul_2ò
8model_4/batch_normalization_8/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_4_batch_normalization_8_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8model_4/batch_normalization_8/batchnorm/ReadVariableOp_2û
+model_4/batch_normalization_8/batchnorm/subSub@model_4/batch_normalization_8/batchnorm/ReadVariableOp_2:value:01model_4/batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+model_4/batch_normalization_8/batchnorm/sub
-model_4/batch_normalization_8/batchnorm/add_1AddV21model_4/batch_normalization_8/batchnorm/mul_1:z:0/model_4/batch_normalization_8/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2/
-model_4/batch_normalization_8/batchnorm/add_1ì
6model_4/batch_normalization_9/batchnorm/ReadVariableOpReadVariableOp?model_4_batch_normalization_9_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype028
6model_4/batch_normalization_9/batchnorm/ReadVariableOp£
-model_4/batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-model_4/batch_normalization_9/batchnorm/add/y
+model_4/batch_normalization_9/batchnorm/addAddV2>model_4/batch_normalization_9/batchnorm/ReadVariableOp:value:06model_4/batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2-
+model_4/batch_normalization_9/batchnorm/add½
-model_4/batch_normalization_9/batchnorm/RsqrtRsqrt/model_4/batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-model_4/batch_normalization_9/batchnorm/Rsqrtø
:model_4/batch_normalization_9/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_4_batch_normalization_9_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:model_4/batch_normalization_9/batchnorm/mul/ReadVariableOpý
+model_4/batch_normalization_9/batchnorm/mulMul1model_4/batch_normalization_9/batchnorm/Rsqrt:y:0Bmodel_4/batch_normalization_9/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+model_4/batch_normalization_9/batchnorm/mulû
-model_4/batch_normalization_9/batchnorm/mul_1Mul-model_4/average_pooling1d_14/Squeeze:output:0/model_4/batch_normalization_9/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2/
-model_4/batch_normalization_9/batchnorm/mul_1ò
8model_4/batch_normalization_9/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_4_batch_normalization_9_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model_4/batch_normalization_9/batchnorm/ReadVariableOp_1ý
-model_4/batch_normalization_9/batchnorm/mul_2Mul@model_4/batch_normalization_9/batchnorm/ReadVariableOp_1:value:0/model_4/batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-model_4/batch_normalization_9/batchnorm/mul_2ò
8model_4/batch_normalization_9/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_4_batch_normalization_9_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8model_4/batch_normalization_9/batchnorm/ReadVariableOp_2û
+model_4/batch_normalization_9/batchnorm/subSub@model_4/batch_normalization_9/batchnorm/ReadVariableOp_2:value:01model_4/batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+model_4/batch_normalization_9/batchnorm/sub
-model_4/batch_normalization_9/batchnorm/add_1AddV21model_4/batch_normalization_9/batchnorm/mul_1:z:0/model_4/batch_normalization_9/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2/
-model_4/batch_normalization_9/batchnorm/add_1Ë
model_4/add_4/addAddV21model_4/batch_normalization_8/batchnorm/add_1:z:01model_4/batch_normalization_9/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
model_4/add_4/addÑ
Umodel_4/transformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOp^model_4_transformer_block_9_multi_head_attention_9_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_4/transformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpð
Fmodel_4/transformer_block_9/multi_head_attention_9/query/einsum/EinsumEinsummodel_4/add_4/add:z:0]model_4/transformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2H
Fmodel_4/transformer_block_9/multi_head_attention_9/query/einsum/Einsum¯
Kmodel_4/transformer_block_9/multi_head_attention_9/query/add/ReadVariableOpReadVariableOpTmodel_4_transformer_block_9_multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_4/transformer_block_9/multi_head_attention_9/query/add/ReadVariableOpå
<model_4/transformer_block_9/multi_head_attention_9/query/addAddV2Omodel_4/transformer_block_9/multi_head_attention_9/query/einsum/Einsum:output:0Smodel_4/transformer_block_9/multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2>
<model_4/transformer_block_9/multi_head_attention_9/query/addË
Smodel_4/transformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOp\model_4_transformer_block_9_multi_head_attention_9_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02U
Smodel_4/transformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpê
Dmodel_4/transformer_block_9/multi_head_attention_9/key/einsum/EinsumEinsummodel_4/add_4/add:z:0[model_4/transformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2F
Dmodel_4/transformer_block_9/multi_head_attention_9/key/einsum/Einsum©
Imodel_4/transformer_block_9/multi_head_attention_9/key/add/ReadVariableOpReadVariableOpRmodel_4_transformer_block_9_multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes

: *
dtype02K
Imodel_4/transformer_block_9/multi_head_attention_9/key/add/ReadVariableOpÝ
:model_4/transformer_block_9/multi_head_attention_9/key/addAddV2Mmodel_4/transformer_block_9/multi_head_attention_9/key/einsum/Einsum:output:0Qmodel_4/transformer_block_9/multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:model_4/transformer_block_9/multi_head_attention_9/key/addÑ
Umodel_4/transformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOp^model_4_transformer_block_9_multi_head_attention_9_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_4/transformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpð
Fmodel_4/transformer_block_9/multi_head_attention_9/value/einsum/EinsumEinsummodel_4/add_4/add:z:0]model_4/transformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2H
Fmodel_4/transformer_block_9/multi_head_attention_9/value/einsum/Einsum¯
Kmodel_4/transformer_block_9/multi_head_attention_9/value/add/ReadVariableOpReadVariableOpTmodel_4_transformer_block_9_multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_4/transformer_block_9/multi_head_attention_9/value/add/ReadVariableOpå
<model_4/transformer_block_9/multi_head_attention_9/value/addAddV2Omodel_4/transformer_block_9/multi_head_attention_9/value/einsum/Einsum:output:0Smodel_4/transformer_block_9/multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2>
<model_4/transformer_block_9/multi_head_attention_9/value/add¹
8model_4/transformer_block_9/multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2:
8model_4/transformer_block_9/multi_head_attention_9/Mul/y¶
6model_4/transformer_block_9/multi_head_attention_9/MulMul@model_4/transformer_block_9/multi_head_attention_9/query/add:z:0Amodel_4/transformer_block_9/multi_head_attention_9/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 28
6model_4/transformer_block_9/multi_head_attention_9/Mulì
@model_4/transformer_block_9/multi_head_attention_9/einsum/EinsumEinsum>model_4/transformer_block_9/multi_head_attention_9/key/add:z:0:model_4/transformer_block_9/multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2B
@model_4/transformer_block_9/multi_head_attention_9/einsum/Einsum
Bmodel_4/transformer_block_9/multi_head_attention_9/softmax/SoftmaxSoftmaxImodel_4/transformer_block_9/multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2D
Bmodel_4/transformer_block_9/multi_head_attention_9/softmax/Softmax
Cmodel_4/transformer_block_9/multi_head_attention_9/dropout/IdentityIdentityLmodel_4/transformer_block_9/multi_head_attention_9/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2E
Cmodel_4/transformer_block_9/multi_head_attention_9/dropout/Identity
Bmodel_4/transformer_block_9/multi_head_attention_9/einsum_1/EinsumEinsumLmodel_4/transformer_block_9/multi_head_attention_9/dropout/Identity:output:0@model_4/transformer_block_9/multi_head_attention_9/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2D
Bmodel_4/transformer_block_9/multi_head_attention_9/einsum_1/Einsumò
`model_4/transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpimodel_4_transformer_block_9_multi_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02b
`model_4/transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpÃ
Qmodel_4/transformer_block_9/multi_head_attention_9/attention_output/einsum/EinsumEinsumKmodel_4/transformer_block_9/multi_head_attention_9/einsum_1/Einsum:output:0hmodel_4/transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe2S
Qmodel_4/transformer_block_9/multi_head_attention_9/attention_output/einsum/EinsumÌ
Vmodel_4/transformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOp_model_4_transformer_block_9_multi_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02X
Vmodel_4/transformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOp
Gmodel_4/transformer_block_9/multi_head_attention_9/attention_output/addAddV2Zmodel_4/transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum:output:0^model_4/transformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2I
Gmodel_4/transformer_block_9/multi_head_attention_9/attention_output/addñ
/model_4/transformer_block_9/dropout_26/IdentityIdentityKmodel_4/transformer_block_9/multi_head_attention_9/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/model_4/transformer_block_9/dropout_26/IdentityÒ
model_4/transformer_block_9/addAddV2model_4/add_4/add:z:08model_4/transformer_block_9/dropout_26/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
model_4/transformer_block_9/addð
Qmodel_4/transformer_block_9/layer_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_4/transformer_block_9/layer_normalization_18/moments/mean/reduction_indicesÒ
?model_4/transformer_block_9/layer_normalization_18/moments/meanMean#model_4/transformer_block_9/add:z:0Zmodel_4/transformer_block_9/layer_normalization_18/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2A
?model_4/transformer_block_9/layer_normalization_18/moments/mean¢
Gmodel_4/transformer_block_9/layer_normalization_18/moments/StopGradientStopGradientHmodel_4/transformer_block_9/layer_normalization_18/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2I
Gmodel_4/transformer_block_9/layer_normalization_18/moments/StopGradientÞ
Lmodel_4/transformer_block_9/layer_normalization_18/moments/SquaredDifferenceSquaredDifference#model_4/transformer_block_9/add:z:0Pmodel_4/transformer_block_9/layer_normalization_18/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2N
Lmodel_4/transformer_block_9/layer_normalization_18/moments/SquaredDifferenceø
Umodel_4/transformer_block_9/layer_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_4/transformer_block_9/layer_normalization_18/moments/variance/reduction_indices
Cmodel_4/transformer_block_9/layer_normalization_18/moments/varianceMeanPmodel_4/transformer_block_9/layer_normalization_18/moments/SquaredDifference:z:0^model_4/transformer_block_9/layer_normalization_18/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2E
Cmodel_4/transformer_block_9/layer_normalization_18/moments/varianceÍ
Bmodel_4/transformer_block_9/layer_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752D
Bmodel_4/transformer_block_9/layer_normalization_18/batchnorm/add/yÞ
@model_4/transformer_block_9/layer_normalization_18/batchnorm/addAddV2Lmodel_4/transformer_block_9/layer_normalization_18/moments/variance:output:0Kmodel_4/transformer_block_9/layer_normalization_18/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2B
@model_4/transformer_block_9/layer_normalization_18/batchnorm/add
Bmodel_4/transformer_block_9/layer_normalization_18/batchnorm/RsqrtRsqrtDmodel_4/transformer_block_9/layer_normalization_18/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2D
Bmodel_4/transformer_block_9/layer_normalization_18/batchnorm/Rsqrt·
Omodel_4/transformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_4_transformer_block_9_layer_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02Q
Omodel_4/transformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpâ
@model_4/transformer_block_9/layer_normalization_18/batchnorm/mulMulFmodel_4/transformer_block_9/layer_normalization_18/batchnorm/Rsqrt:y:0Wmodel_4/transformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2B
@model_4/transformer_block_9/layer_normalization_18/batchnorm/mul°
Bmodel_4/transformer_block_9/layer_normalization_18/batchnorm/mul_1Mul#model_4/transformer_block_9/add:z:0Dmodel_4/transformer_block_9/layer_normalization_18/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2D
Bmodel_4/transformer_block_9/layer_normalization_18/batchnorm/mul_1Õ
Bmodel_4/transformer_block_9/layer_normalization_18/batchnorm/mul_2MulHmodel_4/transformer_block_9/layer_normalization_18/moments/mean:output:0Dmodel_4/transformer_block_9/layer_normalization_18/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2D
Bmodel_4/transformer_block_9/layer_normalization_18/batchnorm/mul_2«
Kmodel_4/transformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpReadVariableOpTmodel_4_transformer_block_9_layer_normalization_18_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02M
Kmodel_4/transformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpÞ
@model_4/transformer_block_9/layer_normalization_18/batchnorm/subSubSmodel_4/transformer_block_9/layer_normalization_18/batchnorm/ReadVariableOp:value:0Fmodel_4/transformer_block_9/layer_normalization_18/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2B
@model_4/transformer_block_9/layer_normalization_18/batchnorm/subÕ
Bmodel_4/transformer_block_9/layer_normalization_18/batchnorm/add_1AddV2Fmodel_4/transformer_block_9/layer_normalization_18/batchnorm/mul_1:z:0Dmodel_4/transformer_block_9/layer_normalization_18/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2D
Bmodel_4/transformer_block_9/layer_normalization_18/batchnorm/add_1¬
Jmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOpReadVariableOpSmodel_4_transformer_block_9_sequential_9_dense_30_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02L
Jmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOpÎ
@model_4/transformer_block_9/sequential_9/dense_30/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_4/transformer_block_9/sequential_9/dense_30/Tensordot/axesÕ
@model_4/transformer_block_9/sequential_9/dense_30/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_4/transformer_block_9/sequential_9/dense_30/Tensordot/freeü
Amodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/ShapeShapeFmodel_4/transformer_block_9/layer_normalization_18/batchnorm/add_1:z:0*
T0*
_output_shapes
:2C
Amodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/ShapeØ
Imodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2/axisË
Dmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2GatherV2Jmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/Shape:output:0Imodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/free:output:0Rmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2Ü
Kmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2_1/axisÑ
Fmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2_1GatherV2Jmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/Shape:output:0Imodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/axes:output:0Tmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2_1Ð
Amodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/ConstÈ
@model_4/transformer_block_9/sequential_9/dense_30/Tensordot/ProdProdMmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2:output:0Jmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_4/transformer_block_9/sequential_9/dense_30/Tensordot/ProdÔ
Cmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/Const_1Ð
Bmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/Prod_1ProdOmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2_1:output:0Lmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/Prod_1Ô
Gmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/concat/axisª
Bmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/concatConcatV2Imodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/free:output:0Imodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/axes:output:0Pmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/concatÔ
Amodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/stackPackImodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/Prod:output:0Kmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/stackæ
Emodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/transpose	TransposeFmodel_4/transformer_block_9/layer_normalization_18/batchnorm/add_1:z:0Kmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2G
Emodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/transposeç
Cmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/ReshapeReshapeImodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/transpose:y:0Jmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2E
Cmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/Reshapeæ
Bmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/MatMulMatMulLmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/Reshape:output:0Rmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2D
Bmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/MatMulÔ
Cmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2E
Cmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/Const_2Ø
Imodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/concat_1/axis·
Dmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/concat_1ConcatV2Mmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2:output:0Lmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/Const_2:output:0Rmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/concat_1Ø
;model_4/transformer_block_9/sequential_9/dense_30/TensordotReshapeLmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/MatMul:product:0Mmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2=
;model_4/transformer_block_9/sequential_9/dense_30/Tensordot¢
Hmodel_4/transformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOpReadVariableOpQmodel_4_transformer_block_9_sequential_9_dense_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02J
Hmodel_4/transformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOpÏ
9model_4/transformer_block_9/sequential_9/dense_30/BiasAddBiasAddDmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot:output:0Pmodel_4/transformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2;
9model_4/transformer_block_9/sequential_9/dense_30/BiasAddò
6model_4/transformer_block_9/sequential_9/dense_30/ReluReluBmodel_4/transformer_block_9/sequential_9/dense_30/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@28
6model_4/transformer_block_9/sequential_9/dense_30/Relu¬
Jmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOpReadVariableOpSmodel_4_transformer_block_9_sequential_9_dense_31_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02L
Jmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOpÎ
@model_4/transformer_block_9/sequential_9/dense_31/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_4/transformer_block_9/sequential_9/dense_31/Tensordot/axesÕ
@model_4/transformer_block_9/sequential_9/dense_31/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_4/transformer_block_9/sequential_9/dense_31/Tensordot/freeú
Amodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/ShapeShapeDmodel_4/transformer_block_9/sequential_9/dense_30/Relu:activations:0*
T0*
_output_shapes
:2C
Amodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/ShapeØ
Imodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2/axisË
Dmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2GatherV2Jmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/Shape:output:0Imodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/free:output:0Rmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2Ü
Kmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2_1/axisÑ
Fmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2_1GatherV2Jmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/Shape:output:0Imodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/axes:output:0Tmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2_1Ð
Amodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/ConstÈ
@model_4/transformer_block_9/sequential_9/dense_31/Tensordot/ProdProdMmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2:output:0Jmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_4/transformer_block_9/sequential_9/dense_31/Tensordot/ProdÔ
Cmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/Const_1Ð
Bmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/Prod_1ProdOmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2_1:output:0Lmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/Prod_1Ô
Gmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/concat/axisª
Bmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/concatConcatV2Imodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/free:output:0Imodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/axes:output:0Pmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/concatÔ
Amodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/stackPackImodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/Prod:output:0Kmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/stackä
Emodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/transpose	TransposeDmodel_4/transformer_block_9/sequential_9/dense_30/Relu:activations:0Kmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2G
Emodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/transposeç
Cmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/ReshapeReshapeImodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/transpose:y:0Jmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2E
Cmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/Reshapeæ
Bmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/MatMulMatMulLmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/Reshape:output:0Rmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2D
Bmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/MatMulÔ
Cmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/Const_2Ø
Imodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/concat_1/axis·
Dmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/concat_1ConcatV2Mmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2:output:0Lmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/Const_2:output:0Rmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/concat_1Ø
;model_4/transformer_block_9/sequential_9/dense_31/TensordotReshapeLmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/MatMul:product:0Mmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2=
;model_4/transformer_block_9/sequential_9/dense_31/Tensordot¢
Hmodel_4/transformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOpReadVariableOpQmodel_4_transformer_block_9_sequential_9_dense_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel_4/transformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOpÏ
9model_4/transformer_block_9/sequential_9/dense_31/BiasAddBiasAddDmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot:output:0Pmodel_4/transformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9model_4/transformer_block_9/sequential_9/dense_31/BiasAddè
/model_4/transformer_block_9/dropout_27/IdentityIdentityBmodel_4/transformer_block_9/sequential_9/dense_31/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/model_4/transformer_block_9/dropout_27/Identity
!model_4/transformer_block_9/add_1AddV2Fmodel_4/transformer_block_9/layer_normalization_18/batchnorm/add_1:z:08model_4/transformer_block_9/dropout_27/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2#
!model_4/transformer_block_9/add_1ð
Qmodel_4/transformer_block_9/layer_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_4/transformer_block_9/layer_normalization_19/moments/mean/reduction_indicesÔ
?model_4/transformer_block_9/layer_normalization_19/moments/meanMean%model_4/transformer_block_9/add_1:z:0Zmodel_4/transformer_block_9/layer_normalization_19/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2A
?model_4/transformer_block_9/layer_normalization_19/moments/mean¢
Gmodel_4/transformer_block_9/layer_normalization_19/moments/StopGradientStopGradientHmodel_4/transformer_block_9/layer_normalization_19/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2I
Gmodel_4/transformer_block_9/layer_normalization_19/moments/StopGradientà
Lmodel_4/transformer_block_9/layer_normalization_19/moments/SquaredDifferenceSquaredDifference%model_4/transformer_block_9/add_1:z:0Pmodel_4/transformer_block_9/layer_normalization_19/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2N
Lmodel_4/transformer_block_9/layer_normalization_19/moments/SquaredDifferenceø
Umodel_4/transformer_block_9/layer_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_4/transformer_block_9/layer_normalization_19/moments/variance/reduction_indices
Cmodel_4/transformer_block_9/layer_normalization_19/moments/varianceMeanPmodel_4/transformer_block_9/layer_normalization_19/moments/SquaredDifference:z:0^model_4/transformer_block_9/layer_normalization_19/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2E
Cmodel_4/transformer_block_9/layer_normalization_19/moments/varianceÍ
Bmodel_4/transformer_block_9/layer_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752D
Bmodel_4/transformer_block_9/layer_normalization_19/batchnorm/add/yÞ
@model_4/transformer_block_9/layer_normalization_19/batchnorm/addAddV2Lmodel_4/transformer_block_9/layer_normalization_19/moments/variance:output:0Kmodel_4/transformer_block_9/layer_normalization_19/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2B
@model_4/transformer_block_9/layer_normalization_19/batchnorm/add
Bmodel_4/transformer_block_9/layer_normalization_19/batchnorm/RsqrtRsqrtDmodel_4/transformer_block_9/layer_normalization_19/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2D
Bmodel_4/transformer_block_9/layer_normalization_19/batchnorm/Rsqrt·
Omodel_4/transformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_4_transformer_block_9_layer_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02Q
Omodel_4/transformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpâ
@model_4/transformer_block_9/layer_normalization_19/batchnorm/mulMulFmodel_4/transformer_block_9/layer_normalization_19/batchnorm/Rsqrt:y:0Wmodel_4/transformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2B
@model_4/transformer_block_9/layer_normalization_19/batchnorm/mul²
Bmodel_4/transformer_block_9/layer_normalization_19/batchnorm/mul_1Mul%model_4/transformer_block_9/add_1:z:0Dmodel_4/transformer_block_9/layer_normalization_19/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2D
Bmodel_4/transformer_block_9/layer_normalization_19/batchnorm/mul_1Õ
Bmodel_4/transformer_block_9/layer_normalization_19/batchnorm/mul_2MulHmodel_4/transformer_block_9/layer_normalization_19/moments/mean:output:0Dmodel_4/transformer_block_9/layer_normalization_19/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2D
Bmodel_4/transformer_block_9/layer_normalization_19/batchnorm/mul_2«
Kmodel_4/transformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpReadVariableOpTmodel_4_transformer_block_9_layer_normalization_19_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02M
Kmodel_4/transformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpÞ
@model_4/transformer_block_9/layer_normalization_19/batchnorm/subSubSmodel_4/transformer_block_9/layer_normalization_19/batchnorm/ReadVariableOp:value:0Fmodel_4/transformer_block_9/layer_normalization_19/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2B
@model_4/transformer_block_9/layer_normalization_19/batchnorm/subÕ
Bmodel_4/transformer_block_9/layer_normalization_19/batchnorm/add_1AddV2Fmodel_4/transformer_block_9/layer_normalization_19/batchnorm/mul_1:z:0Dmodel_4/transformer_block_9/layer_normalization_19/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2D
Bmodel_4/transformer_block_9/layer_normalization_19/batchnorm/add_1
model_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`  2
model_4/flatten_4/ConstÞ
model_4/flatten_4/ReshapeReshapeFmodel_4/transformer_block_9/layer_normalization_19/batchnorm/add_1:z:0 model_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
model_4/flatten_4/Reshape
!model_4/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_4/concatenate_4/concat/axisÞ
model_4/concatenate_4/concatConcatV2"model_4/flatten_4/Reshape:output:0input_10*model_4/concatenate_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
model_4/concatenate_4/concatÁ
&model_4/dense_32/MatMul/ReadVariableOpReadVariableOp/model_4_dense_32_matmul_readvariableop_resource*
_output_shapes
:	è@*
dtype02(
&model_4/dense_32/MatMul/ReadVariableOpÅ
model_4/dense_32/MatMulMatMul%model_4/concatenate_4/concat:output:0.model_4/dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_4/dense_32/MatMul¿
'model_4/dense_32/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_4/dense_32/BiasAdd/ReadVariableOpÅ
model_4/dense_32/BiasAddBiasAdd!model_4/dense_32/MatMul:product:0/model_4/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_4/dense_32/BiasAdd
model_4/dense_32/ReluRelu!model_4/dense_32/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_4/dense_32/Relu
model_4/dropout_28/IdentityIdentity#model_4/dense_32/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_4/dropout_28/IdentityÀ
&model_4/dense_33/MatMul/ReadVariableOpReadVariableOp/model_4_dense_33_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02(
&model_4/dense_33/MatMul/ReadVariableOpÄ
model_4/dense_33/MatMulMatMul$model_4/dropout_28/Identity:output:0.model_4/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_4/dense_33/MatMul¿
'model_4/dense_33/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_4/dense_33/BiasAdd/ReadVariableOpÅ
model_4/dense_33/BiasAddBiasAdd!model_4/dense_33/MatMul:product:0/model_4/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_4/dense_33/BiasAdd
model_4/dense_33/ReluRelu!model_4/dense_33/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_4/dense_33/Relu
model_4/dropout_29/IdentityIdentity#model_4/dense_33/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_4/dropout_29/IdentityÀ
&model_4/dense_34/MatMul/ReadVariableOpReadVariableOp/model_4_dense_34_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&model_4/dense_34/MatMul/ReadVariableOpÄ
model_4/dense_34/MatMulMatMul$model_4/dropout_29/Identity:output:0.model_4/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_4/dense_34/MatMul¿
'model_4/dense_34/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_4/dense_34/BiasAdd/ReadVariableOpÅ
model_4/dense_34/BiasAddBiasAdd!model_4/dense_34/MatMul:product:0/model_4/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_4/dense_34/BiasAdd¬
IdentityIdentity!model_4/dense_34/BiasAdd:output:07^model_4/batch_normalization_8/batchnorm/ReadVariableOp9^model_4/batch_normalization_8/batchnorm/ReadVariableOp_19^model_4/batch_normalization_8/batchnorm/ReadVariableOp_2;^model_4/batch_normalization_8/batchnorm/mul/ReadVariableOp7^model_4/batch_normalization_9/batchnorm/ReadVariableOp9^model_4/batch_normalization_9/batchnorm/ReadVariableOp_19^model_4/batch_normalization_9/batchnorm/ReadVariableOp_2;^model_4/batch_normalization_9/batchnorm/mul/ReadVariableOp(^model_4/conv1d_8/BiasAdd/ReadVariableOp4^model_4/conv1d_8/conv1d/ExpandDims_1/ReadVariableOp(^model_4/conv1d_9/BiasAdd/ReadVariableOp4^model_4/conv1d_9/conv1d/ExpandDims_1/ReadVariableOp(^model_4/dense_32/BiasAdd/ReadVariableOp'^model_4/dense_32/MatMul/ReadVariableOp(^model_4/dense_33/BiasAdd/ReadVariableOp'^model_4/dense_33/MatMul/ReadVariableOp(^model_4/dense_34/BiasAdd/ReadVariableOp'^model_4/dense_34/MatMul/ReadVariableOpD^model_4/token_and_position_embedding_4/embedding_8/embedding_lookupD^model_4/token_and_position_embedding_4/embedding_9/embedding_lookupL^model_4/transformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpP^model_4/transformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpL^model_4/transformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpP^model_4/transformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpW^model_4/transformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpa^model_4/transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpJ^model_4/transformer_block_9/multi_head_attention_9/key/add/ReadVariableOpT^model_4/transformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpL^model_4/transformer_block_9/multi_head_attention_9/query/add/ReadVariableOpV^model_4/transformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpL^model_4/transformer_block_9/multi_head_attention_9/value/add/ReadVariableOpV^model_4/transformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpI^model_4/transformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOpK^model_4/transformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOpI^model_4/transformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOpK^model_4/transformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2p
6model_4/batch_normalization_8/batchnorm/ReadVariableOp6model_4/batch_normalization_8/batchnorm/ReadVariableOp2t
8model_4/batch_normalization_8/batchnorm/ReadVariableOp_18model_4/batch_normalization_8/batchnorm/ReadVariableOp_12t
8model_4/batch_normalization_8/batchnorm/ReadVariableOp_28model_4/batch_normalization_8/batchnorm/ReadVariableOp_22x
:model_4/batch_normalization_8/batchnorm/mul/ReadVariableOp:model_4/batch_normalization_8/batchnorm/mul/ReadVariableOp2p
6model_4/batch_normalization_9/batchnorm/ReadVariableOp6model_4/batch_normalization_9/batchnorm/ReadVariableOp2t
8model_4/batch_normalization_9/batchnorm/ReadVariableOp_18model_4/batch_normalization_9/batchnorm/ReadVariableOp_12t
8model_4/batch_normalization_9/batchnorm/ReadVariableOp_28model_4/batch_normalization_9/batchnorm/ReadVariableOp_22x
:model_4/batch_normalization_9/batchnorm/mul/ReadVariableOp:model_4/batch_normalization_9/batchnorm/mul/ReadVariableOp2R
'model_4/conv1d_8/BiasAdd/ReadVariableOp'model_4/conv1d_8/BiasAdd/ReadVariableOp2j
3model_4/conv1d_8/conv1d/ExpandDims_1/ReadVariableOp3model_4/conv1d_8/conv1d/ExpandDims_1/ReadVariableOp2R
'model_4/conv1d_9/BiasAdd/ReadVariableOp'model_4/conv1d_9/BiasAdd/ReadVariableOp2j
3model_4/conv1d_9/conv1d/ExpandDims_1/ReadVariableOp3model_4/conv1d_9/conv1d/ExpandDims_1/ReadVariableOp2R
'model_4/dense_32/BiasAdd/ReadVariableOp'model_4/dense_32/BiasAdd/ReadVariableOp2P
&model_4/dense_32/MatMul/ReadVariableOp&model_4/dense_32/MatMul/ReadVariableOp2R
'model_4/dense_33/BiasAdd/ReadVariableOp'model_4/dense_33/BiasAdd/ReadVariableOp2P
&model_4/dense_33/MatMul/ReadVariableOp&model_4/dense_33/MatMul/ReadVariableOp2R
'model_4/dense_34/BiasAdd/ReadVariableOp'model_4/dense_34/BiasAdd/ReadVariableOp2P
&model_4/dense_34/MatMul/ReadVariableOp&model_4/dense_34/MatMul/ReadVariableOp2
Cmodel_4/token_and_position_embedding_4/embedding_8/embedding_lookupCmodel_4/token_and_position_embedding_4/embedding_8/embedding_lookup2
Cmodel_4/token_and_position_embedding_4/embedding_9/embedding_lookupCmodel_4/token_and_position_embedding_4/embedding_9/embedding_lookup2
Kmodel_4/transformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpKmodel_4/transformer_block_9/layer_normalization_18/batchnorm/ReadVariableOp2¢
Omodel_4/transformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpOmodel_4/transformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOp2
Kmodel_4/transformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpKmodel_4/transformer_block_9/layer_normalization_19/batchnorm/ReadVariableOp2¢
Omodel_4/transformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpOmodel_4/transformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOp2°
Vmodel_4/transformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpVmodel_4/transformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOp2Ä
`model_4/transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp`model_4/transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2
Imodel_4/transformer_block_9/multi_head_attention_9/key/add/ReadVariableOpImodel_4/transformer_block_9/multi_head_attention_9/key/add/ReadVariableOp2ª
Smodel_4/transformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpSmodel_4/transformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2
Kmodel_4/transformer_block_9/multi_head_attention_9/query/add/ReadVariableOpKmodel_4/transformer_block_9/multi_head_attention_9/query/add/ReadVariableOp2®
Umodel_4/transformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpUmodel_4/transformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2
Kmodel_4/transformer_block_9/multi_head_attention_9/value/add/ReadVariableOpKmodel_4/transformer_block_9/multi_head_attention_9/value/add/ReadVariableOp2®
Umodel_4/transformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpUmodel_4/transformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp2
Hmodel_4/transformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOpHmodel_4/transformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOp2
Jmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOpJmodel_4/transformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOp2
Hmodel_4/transformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOpHmodel_4/transformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOp2
Jmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOpJmodel_4/transformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOp:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_9:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_10
Ê
©
6__inference_batch_normalization_9_layer_call_fn_516030

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
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_5137792
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
ß
~
)__inference_dense_34_layer_call_fn_516528

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
D__inference_dense_34_layer_call_and_return_conditional_losses_5143682
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
Ú
¤
(__inference_model_4_layer_call_fn_515619
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

unknown_34
identity¢StatefulPartitionedCallÖ
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
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_5147492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::22
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
¥
d
+__inference_dropout_29_layer_call_fn_516504

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
F__inference_dropout_29_layer_call_and_return_conditional_losses_5143402
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

Q
5__inference_average_pooling1d_14_layer_call_fn_513074

inputs
identityç
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_14_layer_call_and_return_conditional_losses_5130682
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
è

Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_513779

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
D__inference_conv1d_8_layer_call_and_return_conditional_losses_515668

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
ß
~
)__inference_dense_33_layer_call_fn_516482

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
D__inference_dense_33_layer_call_and_return_conditional_losses_5143122
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
ù[
é
C__inference_model_4_layer_call_and_return_conditional_losses_514385
input_9
input_10)
%token_and_position_embedding_4_513561)
%token_and_position_embedding_4_513563
conv1d_8_513593
conv1d_8_513595
conv1d_9_513626
conv1d_9_513628 
batch_normalization_8_513715 
batch_normalization_8_513717 
batch_normalization_8_513719 
batch_normalization_8_513721 
batch_normalization_9_513806 
batch_normalization_9_513808 
batch_normalization_9_513810 
batch_normalization_9_513812
transformer_block_9_514181
transformer_block_9_514183
transformer_block_9_514185
transformer_block_9_514187
transformer_block_9_514189
transformer_block_9_514191
transformer_block_9_514193
transformer_block_9_514195
transformer_block_9_514197
transformer_block_9_514199
transformer_block_9_514201
transformer_block_9_514203
transformer_block_9_514205
transformer_block_9_514207
transformer_block_9_514209
transformer_block_9_514211
dense_32_514266
dense_32_514268
dense_33_514323
dense_33_514325
dense_34_514379
dense_34_514381
identity¢-batch_normalization_8/StatefulPartitionedCall¢-batch_normalization_9/StatefulPartitionedCall¢ conv1d_8/StatefulPartitionedCall¢ conv1d_9/StatefulPartitionedCall¢ dense_32/StatefulPartitionedCall¢ dense_33/StatefulPartitionedCall¢ dense_34/StatefulPartitionedCall¢"dropout_28/StatefulPartitionedCall¢"dropout_29/StatefulPartitionedCall¢6token_and_position_embedding_4/StatefulPartitionedCall¢+transformer_block_9/StatefulPartitionedCall
6token_and_position_embedding_4/StatefulPartitionedCallStatefulPartitionedCallinput_9%token_and_position_embedding_4_513561%token_and_position_embedding_4_513563*
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
Z__inference_token_and_position_embedding_4_layer_call_and_return_conditional_losses_51355028
6token_and_position_embedding_4/StatefulPartitionedCallÕ
 conv1d_8/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_4/StatefulPartitionedCall:output:0conv1d_8_513593conv1d_8_513595*
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
D__inference_conv1d_8_layer_call_and_return_conditional_losses_5135822"
 conv1d_8/StatefulPartitionedCall£
$average_pooling1d_12/PartitionedCallPartitionedCall)conv1d_8/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_12_layer_call_and_return_conditional_losses_5130382&
$average_pooling1d_12/PartitionedCallÃ
 conv1d_9/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_12/PartitionedCall:output:0conv1d_9_513626conv1d_9_513628*
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
D__inference_conv1d_9_layer_call_and_return_conditional_losses_5136152"
 conv1d_9/StatefulPartitionedCall¸
$average_pooling1d_14/PartitionedCallPartitionedCall?token_and_position_embedding_4/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_14_layer_call_and_return_conditional_losses_5130682&
$average_pooling1d_14/PartitionedCall¢
$average_pooling1d_13/PartitionedCallPartitionedCall)conv1d_9/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_13_layer_call_and_return_conditional_losses_5130532&
$average_pooling1d_13/PartitionedCallÁ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_13/PartitionedCall:output:0batch_normalization_8_513715batch_normalization_8_513717batch_normalization_8_513719batch_normalization_8_513721*
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
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_5136682/
-batch_normalization_8/StatefulPartitionedCallÁ
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_14/PartitionedCall:output:0batch_normalization_9_513806batch_normalization_9_513808batch_normalization_9_513810batch_normalization_9_513812*
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
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_5137592/
-batch_normalization_9/StatefulPartitionedCall»
add_4/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:06batch_normalization_9/StatefulPartitionedCall:output:0*
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
A__inference_add_4_layer_call_and_return_conditional_losses_5138212
add_4/PartitionedCall
+transformer_block_9/StatefulPartitionedCallStatefulPartitionedCalladd_4/PartitionedCall:output:0transformer_block_9_514181transformer_block_9_514183transformer_block_9_514185transformer_block_9_514187transformer_block_9_514189transformer_block_9_514191transformer_block_9_514193transformer_block_9_514195transformer_block_9_514197transformer_block_9_514199transformer_block_9_514201transformer_block_9_514203transformer_block_9_514205transformer_block_9_514207transformer_block_9_514209transformer_block_9_514211*
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
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_5139782-
+transformer_block_9/StatefulPartitionedCall
flatten_4/PartitionedCallPartitionedCall4transformer_block_9/StatefulPartitionedCall:output:0*
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
E__inference_flatten_4_layer_call_and_return_conditional_losses_5142202
flatten_4/PartitionedCall
concatenate_4/PartitionedCallPartitionedCall"flatten_4/PartitionedCall:output:0input_10*
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
I__inference_concatenate_4_layer_call_and_return_conditional_losses_5142352
concatenate_4/PartitionedCall·
 dense_32/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_32_514266dense_32_514268*
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
D__inference_dense_32_layer_call_and_return_conditional_losses_5142552"
 dense_32/StatefulPartitionedCall
"dropout_28/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0*
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
F__inference_dropout_28_layer_call_and_return_conditional_losses_5142832$
"dropout_28/StatefulPartitionedCall¼
 dense_33/StatefulPartitionedCallStatefulPartitionedCall+dropout_28/StatefulPartitionedCall:output:0dense_33_514323dense_33_514325*
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
D__inference_dense_33_layer_call_and_return_conditional_losses_5143122"
 dense_33/StatefulPartitionedCall½
"dropout_29/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0#^dropout_28/StatefulPartitionedCall*
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
F__inference_dropout_29_layer_call_and_return_conditional_losses_5143402$
"dropout_29/StatefulPartitionedCall¼
 dense_34/StatefulPartitionedCallStatefulPartitionedCall+dropout_29/StatefulPartitionedCall:output:0dense_34_514379dense_34_514381*
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
D__inference_dense_34_layer_call_and_return_conditional_losses_5143682"
 dense_34/StatefulPartitionedCall½
IdentityIdentity)dense_34/StatefulPartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv1d_8/StatefulPartitionedCall!^conv1d_9/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall#^dropout_28/StatefulPartitionedCall#^dropout_29/StatefulPartitionedCall7^token_and_position_embedding_4/StatefulPartitionedCall,^transformer_block_9/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv1d_8/StatefulPartitionedCall conv1d_8/StatefulPartitionedCall2D
 conv1d_9/StatefulPartitionedCall conv1d_9/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2H
"dropout_28/StatefulPartitionedCall"dropout_28/StatefulPartitionedCall2H
"dropout_29/StatefulPartitionedCall"dropout_29/StatefulPartitionedCall2p
6token_and_position_embedding_4/StatefulPartitionedCall6token_and_position_embedding_4/StatefulPartitionedCall2Z
+transformer_block_9/StatefulPartitionedCall+transformer_block_9/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_9:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_10
ó0
È
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_513170

inputs
assignmovingavg_513145
assignmovingavg_1_513151)
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
loc:@AssignMovingAvg/513145*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_513145*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/513145*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/513145*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_513145AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/513145*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/513151*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_513151*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/513151*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/513151*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_513151AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/513151*
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

G
+__inference_dropout_28_layer_call_fn_516462

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
F__inference_dropout_28_layer_call_and_return_conditional_losses_5142882
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

Q
5__inference_average_pooling1d_13_layer_call_fn_513059

inputs
identityç
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_13_layer_call_and_return_conditional_losses_5130532
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


Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_515758

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
¼0
È
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_513668

inputs
assignmovingavg_513643
assignmovingavg_1_513649)
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
loc:@AssignMovingAvg/513643*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_513643*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/513643*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/513643*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_513643AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/513643*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/513649*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_513649*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/513649*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/513649*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_513649AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/513649*
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
¿
m
A__inference_add_4_layer_call_and_return_conditional_losses_516036
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
ï
~
)__inference_dense_30_layer_call_fn_516708

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
D__inference_dense_30_layer_call_and_return_conditional_losses_5133892
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

÷
D__inference_conv1d_9_layer_call_and_return_conditional_losses_515693

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
ó
~
)__inference_conv1d_8_layer_call_fn_515677

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
D__inference_conv1d_8_layer_call_and_return_conditional_losses_5135822
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
ì
©
6__inference_batch_normalization_9_layer_call_fn_515935

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
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_5133102
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
É
d
F__inference_dropout_28_layer_call_and_return_conditional_losses_514288

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
¸
 
-__inference_sequential_9_layer_call_fn_516655

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
H__inference_sequential_9_layer_call_and_return_conditional_losses_5134832
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
î
©
6__inference_batch_normalization_9_layer_call_fn_515948

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
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_5133432
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
Â
u
I__inference_concatenate_4_layer_call_and_return_conditional_losses_516409
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
Ö
¤
(__inference_model_4_layer_call_fn_515541
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

unknown_34
identity¢StatefulPartitionedCallÒ
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
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_5145772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::22
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
&__inference_add_4_layer_call_fn_516042
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
A__inference_add_4_layer_call_and_return_conditional_losses_5138212
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
Ó
£
(__inference_model_4_layer_call_fn_514652
input_9
input_10
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

unknown_34
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinput_9input_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_5145772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_9:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_10
é

H__inference_sequential_9_layer_call_and_return_conditional_losses_513510

inputs
dense_30_513499
dense_30_513501
dense_31_513504
dense_31_513506
identity¢ dense_30/StatefulPartitionedCall¢ dense_31/StatefulPartitionedCall
 dense_30/StatefulPartitionedCallStatefulPartitionedCallinputsdense_30_513499dense_30_513501*
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
D__inference_dense_30_layer_call_and_return_conditional_losses_5133892"
 dense_30/StatefulPartitionedCall¾
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_513504dense_31_513506*
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
D__inference_dense_31_layer_call_and_return_conditional_losses_5134352"
 dense_31/StatefulPartitionedCallÇ
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs

e
F__inference_dropout_28_layer_call_and_return_conditional_losses_516447

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
ÿ
â
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_513978

inputsF
Bmulti_head_attention_9_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_9_query_add_readvariableop_resourceD
@multi_head_attention_9_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_9_key_add_readvariableop_resourceF
Bmulti_head_attention_9_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_9_value_add_readvariableop_resourceQ
Mmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_9_attention_output_add_readvariableop_resource@
<layer_normalization_18_batchnorm_mul_readvariableop_resource<
8layer_normalization_18_batchnorm_readvariableop_resource;
7sequential_9_dense_30_tensordot_readvariableop_resource9
5sequential_9_dense_30_biasadd_readvariableop_resource;
7sequential_9_dense_31_tensordot_readvariableop_resource9
5sequential_9_dense_31_biasadd_readvariableop_resource@
<layer_normalization_19_batchnorm_mul_readvariableop_resource<
8layer_normalization_19_batchnorm_readvariableop_resource
identity¢/layer_normalization_18/batchnorm/ReadVariableOp¢3layer_normalization_18/batchnorm/mul/ReadVariableOp¢/layer_normalization_19/batchnorm/ReadVariableOp¢3layer_normalization_19/batchnorm/mul/ReadVariableOp¢:multi_head_attention_9/attention_output/add/ReadVariableOp¢Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_9/key/add/ReadVariableOp¢7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_9/query/add/ReadVariableOp¢9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_9/value/add/ReadVariableOp¢9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp¢,sequential_9/dense_30/BiasAdd/ReadVariableOp¢.sequential_9/dense_30/Tensordot/ReadVariableOp¢,sequential_9/dense_31/BiasAdd/ReadVariableOp¢.sequential_9/dense_31/Tensordot/ReadVariableOpý
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_9/query/einsum/EinsumEinsuminputsAmulti_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_9/query/einsum/EinsumÛ
/multi_head_attention_9/query/add/ReadVariableOpReadVariableOp8multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_9/query/add/ReadVariableOpõ
 multi_head_attention_9/query/addAddV23multi_head_attention_9/query/einsum/Einsum:output:07multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_9/query/add÷
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_9_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_9/key/einsum/EinsumEinsuminputs?multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2*
(multi_head_attention_9/key/einsum/EinsumÕ
-multi_head_attention_9/key/add/ReadVariableOpReadVariableOp6multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_9/key/add/ReadVariableOpí
multi_head_attention_9/key/addAddV21multi_head_attention_9/key/einsum/Einsum:output:05multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
multi_head_attention_9/key/addý
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_9/value/einsum/EinsumEinsuminputsAmulti_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_9/value/einsum/EinsumÛ
/multi_head_attention_9/value/add/ReadVariableOpReadVariableOp8multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_9/value/add/ReadVariableOpõ
 multi_head_attention_9/value/addAddV23multi_head_attention_9/value/einsum/Einsum:output:07multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_9/value/add
multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_9/Mul/yÆ
multi_head_attention_9/MulMul$multi_head_attention_9/query/add:z:0%multi_head_attention_9/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_9/Mulü
$multi_head_attention_9/einsum/EinsumEinsum"multi_head_attention_9/key/add:z:0multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2&
$multi_head_attention_9/einsum/EinsumÄ
&multi_head_attention_9/softmax/SoftmaxSoftmax-multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2(
&multi_head_attention_9/softmax/Softmax¡
,multi_head_attention_9/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,multi_head_attention_9/dropout/dropout/Const
*multi_head_attention_9/dropout/dropout/MulMul0multi_head_attention_9/softmax/Softmax:softmax:05multi_head_attention_9/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2,
*multi_head_attention_9/dropout/dropout/Mul¼
,multi_head_attention_9/dropout/dropout/ShapeShape0multi_head_attention_9/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_9/dropout/dropout/Shape
Cmulti_head_attention_9/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_9/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
dtype02E
Cmulti_head_attention_9/dropout/dropout/random_uniform/RandomUniform³
5multi_head_attention_9/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_9/dropout/dropout/GreaterEqual/yÂ
3multi_head_attention_9/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_9/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_9/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##25
3multi_head_attention_9/dropout/dropout/GreaterEqualä
+multi_head_attention_9/dropout/dropout/CastCast7multi_head_attention_9/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2-
+multi_head_attention_9/dropout/dropout/Castþ
,multi_head_attention_9/dropout/dropout/Mul_1Mul.multi_head_attention_9/dropout/dropout/Mul:z:0/multi_head_attention_9/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2.
,multi_head_attention_9/dropout/dropout/Mul_1
&multi_head_attention_9/einsum_1/EinsumEinsum0multi_head_attention_9/dropout/dropout/Mul_1:z:0$multi_head_attention_9/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2(
&multi_head_attention_9/einsum_1/Einsum
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_9/attention_output/einsum/EinsumEinsum/multi_head_attention_9/einsum_1/Einsum:output:0Lmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe27
5multi_head_attention_9/attention_output/einsum/Einsumø
:multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_9/attention_output/add/ReadVariableOp
+multi_head_attention_9/attention_output/addAddV2>multi_head_attention_9/attention_output/einsum/Einsum:output:0Bmulti_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+multi_head_attention_9/attention_output/addy
dropout_26/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_26/dropout/ConstÁ
dropout_26/dropout/MulMul/multi_head_attention_9/attention_output/add:z:0!dropout_26/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_26/dropout/Mul
dropout_26/dropout/ShapeShape/multi_head_attention_9/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_26/dropout/ShapeÙ
/dropout_26/dropout/random_uniform/RandomUniformRandomUniform!dropout_26/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype021
/dropout_26/dropout/random_uniform/RandomUniform
!dropout_26/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_26/dropout/GreaterEqual/yî
dropout_26/dropout/GreaterEqualGreaterEqual8dropout_26/dropout/random_uniform/RandomUniform:output:0*dropout_26/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
dropout_26/dropout/GreaterEqual¤
dropout_26/dropout/CastCast#dropout_26/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_26/dropout/Castª
dropout_26/dropout/Mul_1Muldropout_26/dropout/Mul:z:0dropout_26/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_26/dropout/Mul_1o
addAddV2inputsdropout_26/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¸
5layer_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_18/moments/mean/reduction_indicesâ
#layer_normalization_18/moments/meanMeanadd:z:0>layer_normalization_18/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_18/moments/meanÎ
+layer_normalization_18/moments/StopGradientStopGradient,layer_normalization_18/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_18/moments/StopGradientî
0layer_normalization_18/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_18/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_18/moments/SquaredDifferenceÀ
9layer_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_18/moments/variance/reduction_indices
'layer_normalization_18/moments/varianceMean4layer_normalization_18/moments/SquaredDifference:z:0Blayer_normalization_18/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_18/moments/variance
&layer_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_18/batchnorm/add/yî
$layer_normalization_18/batchnorm/addAddV20layer_normalization_18/moments/variance:output:0/layer_normalization_18/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_18/batchnorm/add¹
&layer_normalization_18/batchnorm/RsqrtRsqrt(layer_normalization_18/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_18/batchnorm/Rsqrtã
3layer_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_18/batchnorm/mul/ReadVariableOpò
$layer_normalization_18/batchnorm/mulMul*layer_normalization_18/batchnorm/Rsqrt:y:0;layer_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_18/batchnorm/mulÀ
&layer_normalization_18/batchnorm/mul_1Muladd:z:0(layer_normalization_18/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_18/batchnorm/mul_1å
&layer_normalization_18/batchnorm/mul_2Mul,layer_normalization_18/moments/mean:output:0(layer_normalization_18/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_18/batchnorm/mul_2×
/layer_normalization_18/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_18_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_18/batchnorm/ReadVariableOpî
$layer_normalization_18/batchnorm/subSub7layer_normalization_18/batchnorm/ReadVariableOp:value:0*layer_normalization_18/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_18/batchnorm/subå
&layer_normalization_18/batchnorm/add_1AddV2*layer_normalization_18/batchnorm/mul_1:z:0(layer_normalization_18/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_18/batchnorm/add_1Ø
.sequential_9/dense_30/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_30_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_9/dense_30/Tensordot/ReadVariableOp
$sequential_9/dense_30/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_9/dense_30/Tensordot/axes
$sequential_9/dense_30/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_9/dense_30/Tensordot/free¨
%sequential_9/dense_30/Tensordot/ShapeShape*layer_normalization_18/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_9/dense_30/Tensordot/Shape 
-sequential_9/dense_30/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_30/Tensordot/GatherV2/axis¿
(sequential_9/dense_30/Tensordot/GatherV2GatherV2.sequential_9/dense_30/Tensordot/Shape:output:0-sequential_9/dense_30/Tensordot/free:output:06sequential_9/dense_30/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_9/dense_30/Tensordot/GatherV2¤
/sequential_9/dense_30/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_9/dense_30/Tensordot/GatherV2_1/axisÅ
*sequential_9/dense_30/Tensordot/GatherV2_1GatherV2.sequential_9/dense_30/Tensordot/Shape:output:0-sequential_9/dense_30/Tensordot/axes:output:08sequential_9/dense_30/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_9/dense_30/Tensordot/GatherV2_1
%sequential_9/dense_30/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_9/dense_30/Tensordot/ConstØ
$sequential_9/dense_30/Tensordot/ProdProd1sequential_9/dense_30/Tensordot/GatherV2:output:0.sequential_9/dense_30/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_9/dense_30/Tensordot/Prod
'sequential_9/dense_30/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_30/Tensordot/Const_1à
&sequential_9/dense_30/Tensordot/Prod_1Prod3sequential_9/dense_30/Tensordot/GatherV2_1:output:00sequential_9/dense_30/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_9/dense_30/Tensordot/Prod_1
+sequential_9/dense_30/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_9/dense_30/Tensordot/concat/axis
&sequential_9/dense_30/Tensordot/concatConcatV2-sequential_9/dense_30/Tensordot/free:output:0-sequential_9/dense_30/Tensordot/axes:output:04sequential_9/dense_30/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/dense_30/Tensordot/concatä
%sequential_9/dense_30/Tensordot/stackPack-sequential_9/dense_30/Tensordot/Prod:output:0/sequential_9/dense_30/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/dense_30/Tensordot/stackö
)sequential_9/dense_30/Tensordot/transpose	Transpose*layer_normalization_18/batchnorm/add_1:z:0/sequential_9/dense_30/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2+
)sequential_9/dense_30/Tensordot/transpose÷
'sequential_9/dense_30/Tensordot/ReshapeReshape-sequential_9/dense_30/Tensordot/transpose:y:0.sequential_9/dense_30/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_9/dense_30/Tensordot/Reshapeö
&sequential_9/dense_30/Tensordot/MatMulMatMul0sequential_9/dense_30/Tensordot/Reshape:output:06sequential_9/dense_30/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential_9/dense_30/Tensordot/MatMul
'sequential_9/dense_30/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_9/dense_30/Tensordot/Const_2 
-sequential_9/dense_30/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_30/Tensordot/concat_1/axis«
(sequential_9/dense_30/Tensordot/concat_1ConcatV21sequential_9/dense_30/Tensordot/GatherV2:output:00sequential_9/dense_30/Tensordot/Const_2:output:06sequential_9/dense_30/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_9/dense_30/Tensordot/concat_1è
sequential_9/dense_30/TensordotReshape0sequential_9/dense_30/Tensordot/MatMul:product:01sequential_9/dense_30/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2!
sequential_9/dense_30/TensordotÎ
,sequential_9/dense_30/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_9/dense_30/BiasAdd/ReadVariableOpß
sequential_9/dense_30/BiasAddBiasAdd(sequential_9/dense_30/Tensordot:output:04sequential_9/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_9/dense_30/BiasAdd
sequential_9/dense_30/ReluRelu&sequential_9/dense_30/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_9/dense_30/ReluØ
.sequential_9/dense_31/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_31_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_9/dense_31/Tensordot/ReadVariableOp
$sequential_9/dense_31/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_9/dense_31/Tensordot/axes
$sequential_9/dense_31/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_9/dense_31/Tensordot/free¦
%sequential_9/dense_31/Tensordot/ShapeShape(sequential_9/dense_30/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dense_31/Tensordot/Shape 
-sequential_9/dense_31/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_31/Tensordot/GatherV2/axis¿
(sequential_9/dense_31/Tensordot/GatherV2GatherV2.sequential_9/dense_31/Tensordot/Shape:output:0-sequential_9/dense_31/Tensordot/free:output:06sequential_9/dense_31/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_9/dense_31/Tensordot/GatherV2¤
/sequential_9/dense_31/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_9/dense_31/Tensordot/GatherV2_1/axisÅ
*sequential_9/dense_31/Tensordot/GatherV2_1GatherV2.sequential_9/dense_31/Tensordot/Shape:output:0-sequential_9/dense_31/Tensordot/axes:output:08sequential_9/dense_31/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_9/dense_31/Tensordot/GatherV2_1
%sequential_9/dense_31/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_9/dense_31/Tensordot/ConstØ
$sequential_9/dense_31/Tensordot/ProdProd1sequential_9/dense_31/Tensordot/GatherV2:output:0.sequential_9/dense_31/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_9/dense_31/Tensordot/Prod
'sequential_9/dense_31/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_31/Tensordot/Const_1à
&sequential_9/dense_31/Tensordot/Prod_1Prod3sequential_9/dense_31/Tensordot/GatherV2_1:output:00sequential_9/dense_31/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_9/dense_31/Tensordot/Prod_1
+sequential_9/dense_31/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_9/dense_31/Tensordot/concat/axis
&sequential_9/dense_31/Tensordot/concatConcatV2-sequential_9/dense_31/Tensordot/free:output:0-sequential_9/dense_31/Tensordot/axes:output:04sequential_9/dense_31/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/dense_31/Tensordot/concatä
%sequential_9/dense_31/Tensordot/stackPack-sequential_9/dense_31/Tensordot/Prod:output:0/sequential_9/dense_31/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/dense_31/Tensordot/stackô
)sequential_9/dense_31/Tensordot/transpose	Transpose(sequential_9/dense_30/Relu:activations:0/sequential_9/dense_31/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2+
)sequential_9/dense_31/Tensordot/transpose÷
'sequential_9/dense_31/Tensordot/ReshapeReshape-sequential_9/dense_31/Tensordot/transpose:y:0.sequential_9/dense_31/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_9/dense_31/Tensordot/Reshapeö
&sequential_9/dense_31/Tensordot/MatMulMatMul0sequential_9/dense_31/Tensordot/Reshape:output:06sequential_9/dense_31/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_9/dense_31/Tensordot/MatMul
'sequential_9/dense_31/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_31/Tensordot/Const_2 
-sequential_9/dense_31/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_31/Tensordot/concat_1/axis«
(sequential_9/dense_31/Tensordot/concat_1ConcatV21sequential_9/dense_31/Tensordot/GatherV2:output:00sequential_9/dense_31/Tensordot/Const_2:output:06sequential_9/dense_31/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_9/dense_31/Tensordot/concat_1è
sequential_9/dense_31/TensordotReshape0sequential_9/dense_31/Tensordot/MatMul:product:01sequential_9/dense_31/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
sequential_9/dense_31/TensordotÎ
,sequential_9/dense_31/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_9/dense_31/BiasAdd/ReadVariableOpß
sequential_9/dense_31/BiasAddBiasAdd(sequential_9/dense_31/Tensordot:output:04sequential_9/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
sequential_9/dense_31/BiasAddy
dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_27/dropout/Const¸
dropout_27/dropout/MulMul&sequential_9/dense_31/BiasAdd:output:0!dropout_27/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_27/dropout/Mul
dropout_27/dropout/ShapeShape&sequential_9/dense_31/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_27/dropout/ShapeÙ
/dropout_27/dropout/random_uniform/RandomUniformRandomUniform!dropout_27/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype021
/dropout_27/dropout/random_uniform/RandomUniform
!dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_27/dropout/GreaterEqual/yî
dropout_27/dropout/GreaterEqualGreaterEqual8dropout_27/dropout/random_uniform/RandomUniform:output:0*dropout_27/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
dropout_27/dropout/GreaterEqual¤
dropout_27/dropout/CastCast#dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_27/dropout/Castª
dropout_27/dropout/Mul_1Muldropout_27/dropout/Mul:z:0dropout_27/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_27/dropout/Mul_1
add_1AddV2*layer_normalization_18/batchnorm/add_1:z:0dropout_27/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¸
5layer_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_19/moments/mean/reduction_indicesä
#layer_normalization_19/moments/meanMean	add_1:z:0>layer_normalization_19/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_19/moments/meanÎ
+layer_normalization_19/moments/StopGradientStopGradient,layer_normalization_19/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_19/moments/StopGradientð
0layer_normalization_19/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_19/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_19/moments/SquaredDifferenceÀ
9layer_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_19/moments/variance/reduction_indices
'layer_normalization_19/moments/varianceMean4layer_normalization_19/moments/SquaredDifference:z:0Blayer_normalization_19/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_19/moments/variance
&layer_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_19/batchnorm/add/yî
$layer_normalization_19/batchnorm/addAddV20layer_normalization_19/moments/variance:output:0/layer_normalization_19/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_19/batchnorm/add¹
&layer_normalization_19/batchnorm/RsqrtRsqrt(layer_normalization_19/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_19/batchnorm/Rsqrtã
3layer_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_19/batchnorm/mul/ReadVariableOpò
$layer_normalization_19/batchnorm/mulMul*layer_normalization_19/batchnorm/Rsqrt:y:0;layer_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_19/batchnorm/mulÂ
&layer_normalization_19/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_19/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_19/batchnorm/mul_1å
&layer_normalization_19/batchnorm/mul_2Mul,layer_normalization_19/moments/mean:output:0(layer_normalization_19/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_19/batchnorm/mul_2×
/layer_normalization_19/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_19_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_19/batchnorm/ReadVariableOpî
$layer_normalization_19/batchnorm/subSub7layer_normalization_19/batchnorm/ReadVariableOp:value:0*layer_normalization_19/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_19/batchnorm/subå
&layer_normalization_19/batchnorm/add_1AddV2*layer_normalization_19/batchnorm/mul_1:z:0(layer_normalization_19/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_19/batchnorm/add_1Ü
IdentityIdentity*layer_normalization_19/batchnorm/add_1:z:00^layer_normalization_18/batchnorm/ReadVariableOp4^layer_normalization_18/batchnorm/mul/ReadVariableOp0^layer_normalization_19/batchnorm/ReadVariableOp4^layer_normalization_19/batchnorm/mul/ReadVariableOp;^multi_head_attention_9/attention_output/add/ReadVariableOpE^multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_9/key/add/ReadVariableOp8^multi_head_attention_9/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/query/add/ReadVariableOp:^multi_head_attention_9/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/value/add/ReadVariableOp:^multi_head_attention_9/value/einsum/Einsum/ReadVariableOp-^sequential_9/dense_30/BiasAdd/ReadVariableOp/^sequential_9/dense_30/Tensordot/ReadVariableOp-^sequential_9/dense_31/BiasAdd/ReadVariableOp/^sequential_9/dense_31/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2b
/layer_normalization_18/batchnorm/ReadVariableOp/layer_normalization_18/batchnorm/ReadVariableOp2j
3layer_normalization_18/batchnorm/mul/ReadVariableOp3layer_normalization_18/batchnorm/mul/ReadVariableOp2b
/layer_normalization_19/batchnorm/ReadVariableOp/layer_normalization_19/batchnorm/ReadVariableOp2j
3layer_normalization_19/batchnorm/mul/ReadVariableOp3layer_normalization_19/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_9/attention_output/add/ReadVariableOp:multi_head_attention_9/attention_output/add/ReadVariableOp2
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_9/key/add/ReadVariableOp-multi_head_attention_9/key/add/ReadVariableOp2r
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/query/add/ReadVariableOp/multi_head_attention_9/query/add/ReadVariableOp2v
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/value/add/ReadVariableOp/multi_head_attention_9/value/add/ReadVariableOp2v
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp2\
,sequential_9/dense_30/BiasAdd/ReadVariableOp,sequential_9/dense_30/BiasAdd/ReadVariableOp2`
.sequential_9/dense_30/Tensordot/ReadVariableOp.sequential_9/dense_30/Tensordot/ReadVariableOp2\
,sequential_9/dense_31/BiasAdd/ReadVariableOp,sequential_9/dense_31/BiasAdd/ReadVariableOp2`
.sequential_9/dense_31/Tensordot/ReadVariableOp.sequential_9/dense_31/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
è

Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_516004

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
á
~
)__inference_dense_32_layer_call_fn_516435

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
D__inference_dense_32_layer_call_and_return_conditional_losses_5142552
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
µ
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_514220

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
	
Ý
D__inference_dense_34_layer_call_and_return_conditional_losses_514368

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

e
F__inference_dropout_29_layer_call_and_return_conditional_losses_514340

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

e
F__inference_dropout_29_layer_call_and_return_conditional_losses_516494

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

G
+__inference_dropout_29_layer_call_fn_516509

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
F__inference_dropout_29_layer_call_and_return_conditional_losses_5143452
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


H__inference_sequential_9_layer_call_and_return_conditional_losses_513466
dense_30_input
dense_30_513455
dense_30_513457
dense_31_513460
dense_31_513462
identity¢ dense_30/StatefulPartitionedCall¢ dense_31/StatefulPartitionedCall£
 dense_30/StatefulPartitionedCallStatefulPartitionedCalldense_30_inputdense_30_513455dense_30_513457*
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
D__inference_dense_30_layer_call_and_return_conditional_losses_5133892"
 dense_30/StatefulPartitionedCall¾
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_513460dense_31_513462*
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
D__inference_dense_31_layer_call_and_return_conditional_losses_5134352"
 dense_31/StatefulPartitionedCallÇ
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
(
_user_specified_namedense_30_input
Ð

à
4__inference_transformer_block_9_layer_call_fn_516391

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
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_5141052
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
ó0
È
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_515902

inputs
assignmovingavg_515877
assignmovingavg_1_515883)
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
loc:@AssignMovingAvg/515877*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_515877*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/515877*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/515877*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_515877AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/515877*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/515883*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_515883*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/515883*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/515883*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_515883AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/515883*
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


H__inference_sequential_9_layer_call_and_return_conditional_losses_513452
dense_30_input
dense_30_513400
dense_30_513402
dense_31_513446
dense_31_513448
identity¢ dense_30/StatefulPartitionedCall¢ dense_31/StatefulPartitionedCall£
 dense_30/StatefulPartitionedCallStatefulPartitionedCalldense_30_inputdense_30_513400dense_30_513402*
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
D__inference_dense_30_layer_call_and_return_conditional_losses_5133892"
 dense_30/StatefulPartitionedCall¾
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_513446dense_31_513448*
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
D__inference_dense_31_layer_call_and_return_conditional_losses_5134352"
 dense_31/StatefulPartitionedCallÇ
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
(
_user_specified_namedense_30_input
¡
F
*__inference_flatten_4_layer_call_fn_516402

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
E__inference_flatten_4_layer_call_and_return_conditional_losses_5142202
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
ÒÜ
Ì$
C__inference_model_4_layer_call_and_return_conditional_losses_515463
inputs_0
inputs_1F
Btoken_and_position_embedding_4_embedding_9_embedding_lookup_515232F
Btoken_and_position_embedding_4_embedding_8_embedding_lookup_5152388
4conv1d_8_conv1d_expanddims_1_readvariableop_resource,
(conv1d_8_biasadd_readvariableop_resource8
4conv1d_9_conv1d_expanddims_1_readvariableop_resource,
(conv1d_9_biasadd_readvariableop_resource;
7batch_normalization_8_batchnorm_readvariableop_resource?
;batch_normalization_8_batchnorm_mul_readvariableop_resource=
9batch_normalization_8_batchnorm_readvariableop_1_resource=
9batch_normalization_8_batchnorm_readvariableop_2_resource;
7batch_normalization_9_batchnorm_readvariableop_resource?
;batch_normalization_9_batchnorm_mul_readvariableop_resource=
9batch_normalization_9_batchnorm_readvariableop_1_resource=
9batch_normalization_9_batchnorm_readvariableop_2_resourceZ
Vtransformer_block_9_multi_head_attention_9_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_9_multi_head_attention_9_query_add_readvariableop_resourceX
Ttransformer_block_9_multi_head_attention_9_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_9_multi_head_attention_9_key_add_readvariableop_resourceZ
Vtransformer_block_9_multi_head_attention_9_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_9_multi_head_attention_9_value_add_readvariableop_resourcee
atransformer_block_9_multi_head_attention_9_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_9_multi_head_attention_9_attention_output_add_readvariableop_resourceT
Ptransformer_block_9_layer_normalization_18_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_9_layer_normalization_18_batchnorm_readvariableop_resourceO
Ktransformer_block_9_sequential_9_dense_30_tensordot_readvariableop_resourceM
Itransformer_block_9_sequential_9_dense_30_biasadd_readvariableop_resourceO
Ktransformer_block_9_sequential_9_dense_31_tensordot_readvariableop_resourceM
Itransformer_block_9_sequential_9_dense_31_biasadd_readvariableop_resourceT
Ptransformer_block_9_layer_normalization_19_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_9_layer_normalization_19_batchnorm_readvariableop_resource+
'dense_32_matmul_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource
identity¢.batch_normalization_8/batchnorm/ReadVariableOp¢0batch_normalization_8/batchnorm/ReadVariableOp_1¢0batch_normalization_8/batchnorm/ReadVariableOp_2¢2batch_normalization_8/batchnorm/mul/ReadVariableOp¢.batch_normalization_9/batchnorm/ReadVariableOp¢0batch_normalization_9/batchnorm/ReadVariableOp_1¢0batch_normalization_9/batchnorm/ReadVariableOp_2¢2batch_normalization_9/batchnorm/mul/ReadVariableOp¢conv1d_8/BiasAdd/ReadVariableOp¢+conv1d_8/conv1d/ExpandDims_1/ReadVariableOp¢conv1d_9/BiasAdd/ReadVariableOp¢+conv1d_9/conv1d/ExpandDims_1/ReadVariableOp¢dense_32/BiasAdd/ReadVariableOp¢dense_32/MatMul/ReadVariableOp¢dense_33/BiasAdd/ReadVariableOp¢dense_33/MatMul/ReadVariableOp¢dense_34/BiasAdd/ReadVariableOp¢dense_34/MatMul/ReadVariableOp¢;token_and_position_embedding_4/embedding_8/embedding_lookup¢;token_and_position_embedding_4/embedding_9/embedding_lookup¢Ctransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOp¢Gtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOp¢Ctransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOp¢Gtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOp¢Ntransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOp¢Xtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp¢Atransformer_block_9/multi_head_attention_9/key/add/ReadVariableOp¢Ktransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp¢Ctransformer_block_9/multi_head_attention_9/query/add/ReadVariableOp¢Mtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp¢Ctransformer_block_9/multi_head_attention_9/value/add/ReadVariableOp¢Mtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp¢@transformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOp¢Btransformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOp¢@transformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOp¢Btransformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOp
$token_and_position_embedding_4/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_4/Shape»
2token_and_position_embedding_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ24
2token_and_position_embedding_4/strided_slice/stack¶
4token_and_position_embedding_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_4/strided_slice/stack_1¶
4token_and_position_embedding_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_4/strided_slice/stack_2
,token_and_position_embedding_4/strided_sliceStridedSlice-token_and_position_embedding_4/Shape:output:0;token_and_position_embedding_4/strided_slice/stack:output:0=token_and_position_embedding_4/strided_slice/stack_1:output:0=token_and_position_embedding_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_4/strided_slice
*token_and_position_embedding_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_4/range/start
*token_and_position_embedding_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_4/range/delta
$token_and_position_embedding_4/rangeRange3token_and_position_embedding_4/range/start:output:05token_and_position_embedding_4/strided_slice:output:03token_and_position_embedding_4/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$token_and_position_embedding_4/rangeÊ
;token_and_position_embedding_4/embedding_9/embedding_lookupResourceGatherBtoken_and_position_embedding_4_embedding_9_embedding_lookup_515232-token_and_position_embedding_4/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_4/embedding_9/embedding_lookup/515232*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02=
;token_and_position_embedding_4/embedding_9/embedding_lookup
Dtoken_and_position_embedding_4/embedding_9/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_4/embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_4/embedding_9/embedding_lookup/515232*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2F
Dtoken_and_position_embedding_4/embedding_9/embedding_lookup/Identity
Ftoken_and_position_embedding_4/embedding_9/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_4/embedding_9/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2H
Ftoken_and_position_embedding_4/embedding_9/embedding_lookup/Identity_1¶
/token_and_position_embedding_4/embedding_8/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR21
/token_and_position_embedding_4/embedding_8/CastÕ
;token_and_position_embedding_4/embedding_8/embedding_lookupResourceGatherBtoken_and_position_embedding_4_embedding_8_embedding_lookup_5152383token_and_position_embedding_4/embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_4/embedding_8/embedding_lookup/515238*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02=
;token_and_position_embedding_4/embedding_8/embedding_lookup
Dtoken_and_position_embedding_4/embedding_8/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_4/embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_4/embedding_8/embedding_lookup/515238*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2F
Dtoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity¢
Ftoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2H
Ftoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity_1ª
"token_and_position_embedding_4/addAddV2Otoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_4/embedding_9/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2$
"token_and_position_embedding_4/add
conv1d_8/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_8/conv1d/ExpandDims/dimÒ
conv1d_8/conv1d/ExpandDims
ExpandDims&token_and_position_embedding_4/add:z:0'conv1d_8/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_8/conv1d/ExpandDimsÓ
+conv1d_8/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_8_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02-
+conv1d_8/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_8/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_8/conv1d/ExpandDims_1/dimÛ
conv1d_8/conv1d/ExpandDims_1
ExpandDims3conv1d_8/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_8/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_8/conv1d/ExpandDims_1Û
conv1d_8/conv1dConv2D#conv1d_8/conv1d/ExpandDims:output:0%conv1d_8/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
paddingSAME*
strides
2
conv1d_8/conv1d®
conv1d_8/conv1d/SqueezeSqueezeconv1d_8/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_8/conv1d/Squeeze§
conv1d_8/BiasAdd/ReadVariableOpReadVariableOp(conv1d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_8/BiasAdd/ReadVariableOp±
conv1d_8/BiasAddBiasAdd conv1d_8/conv1d/Squeeze:output:0'conv1d_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_8/BiasAddx
conv1d_8/ReluReluconv1d_8/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_8/Relu
#average_pooling1d_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_12/ExpandDims/dimÖ
average_pooling1d_12/ExpandDims
ExpandDimsconv1d_8/Relu:activations:0,average_pooling1d_12/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2!
average_pooling1d_12/ExpandDimsè
average_pooling1d_12/AvgPoolAvgPool(average_pooling1d_12/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_12/AvgPool¼
average_pooling1d_12/SqueezeSqueeze%average_pooling1d_12/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims
2
average_pooling1d_12/Squeeze
conv1d_9/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_9/conv1d/ExpandDims/dimÑ
conv1d_9/conv1d/ExpandDims
ExpandDims%average_pooling1d_12/Squeeze:output:0'conv1d_9/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_9/conv1d/ExpandDimsÓ
+conv1d_9/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_9_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02-
+conv1d_9/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_9/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_9/conv1d/ExpandDims_1/dimÛ
conv1d_9/conv1d/ExpandDims_1
ExpandDims3conv1d_9/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_9/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d_9/conv1d/ExpandDims_1Û
conv1d_9/conv1dConv2D#conv1d_9/conv1d/ExpandDims:output:0%conv1d_9/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
paddingSAME*
strides
2
conv1d_9/conv1d®
conv1d_9/conv1d/SqueezeSqueezeconv1d_9/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_9/conv1d/Squeeze§
conv1d_9/BiasAdd/ReadVariableOpReadVariableOp(conv1d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_9/BiasAdd/ReadVariableOp±
conv1d_9/BiasAddBiasAdd conv1d_9/conv1d/Squeeze:output:0'conv1d_9/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_9/BiasAddx
conv1d_9/ReluReluconv1d_9/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_9/Relu
#average_pooling1d_14/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_14/ExpandDims/dimá
average_pooling1d_14/ExpandDims
ExpandDims&token_and_position_embedding_4/add:z:0,average_pooling1d_14/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2!
average_pooling1d_14/ExpandDimsé
average_pooling1d_14/AvgPoolAvgPool(average_pooling1d_14/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize	
¬*
paddingVALID*
strides	
¬2
average_pooling1d_14/AvgPool»
average_pooling1d_14/SqueezeSqueeze%average_pooling1d_14/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_14/Squeeze
#average_pooling1d_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_13/ExpandDims/dimÖ
average_pooling1d_13/ExpandDims
ExpandDimsconv1d_9/Relu:activations:0,average_pooling1d_13/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2!
average_pooling1d_13/ExpandDimsç
average_pooling1d_13/AvgPoolAvgPool(average_pooling1d_13/ExpandDims:output:0*
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
average_pooling1d_13/AvgPool»
average_pooling1d_13/SqueezeSqueeze%average_pooling1d_13/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_13/SqueezeÔ
.batch_normalization_8/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes
: *
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
: 2%
#batch_normalization_8/batchnorm/add¥
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_8/batchnorm/Rsqrtà
2batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_8/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:0:batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_8/batchnorm/mulÛ
%batch_normalization_8/batchnorm/mul_1Mul%average_pooling1d_13/Squeeze:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_8/batchnorm/mul_1Ú
0batch_normalization_8/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_8_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_8/batchnorm/ReadVariableOp_1Ý
%batch_normalization_8/batchnorm/mul_2Mul8batch_normalization_8/batchnorm/ReadVariableOp_1:value:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_8/batchnorm/mul_2Ú
0batch_normalization_8/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_8_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_8/batchnorm/ReadVariableOp_2Û
#batch_normalization_8/batchnorm/subSub8batch_normalization_8/batchnorm/ReadVariableOp_2:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_8/batchnorm/subá
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_8/batchnorm/add_1Ô
.batch_normalization_9/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_9_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_9/batchnorm/ReadVariableOp
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_9/batchnorm/add/yà
#batch_normalization_9/batchnorm/addAddV26batch_normalization_9/batchnorm/ReadVariableOp:value:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_9/batchnorm/add¥
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_9/batchnorm/Rsqrtà
2batch_normalization_9/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_9_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_9/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:0:batch_normalization_9/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_9/batchnorm/mulÛ
%batch_normalization_9/batchnorm/mul_1Mul%average_pooling1d_14/Squeeze:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_9/batchnorm/mul_1Ú
0batch_normalization_9/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_9_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_9/batchnorm/ReadVariableOp_1Ý
%batch_normalization_9/batchnorm/mul_2Mul8batch_normalization_9/batchnorm/ReadVariableOp_1:value:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_9/batchnorm/mul_2Ú
0batch_normalization_9/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_9_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_9/batchnorm/ReadVariableOp_2Û
#batch_normalization_9/batchnorm/subSub8batch_normalization_9/batchnorm/ReadVariableOp_2:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_9/batchnorm/subá
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_9/batchnorm/add_1«
	add_4/addAddV2)batch_normalization_8/batchnorm/add_1:z:0)batch_normalization_9/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
	add_4/add¹
Mtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_9_multi_head_attention_9_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpÐ
>transformer_block_9/multi_head_attention_9/query/einsum/EinsumEinsumadd_4/add:z:0Utransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2@
>transformer_block_9/multi_head_attention_9/query/einsum/Einsum
Ctransformer_block_9/multi_head_attention_9/query/add/ReadVariableOpReadVariableOpLtransformer_block_9_multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_9/multi_head_attention_9/query/add/ReadVariableOpÅ
4transformer_block_9/multi_head_attention_9/query/addAddV2Gtransformer_block_9/multi_head_attention_9/query/einsum/Einsum:output:0Ktransformer_block_9/multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_9/multi_head_attention_9/query/add³
Ktransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_9_multi_head_attention_9_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpÊ
<transformer_block_9/multi_head_attention_9/key/einsum/EinsumEinsumadd_4/add:z:0Stransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2>
<transformer_block_9/multi_head_attention_9/key/einsum/Einsum
Atransformer_block_9/multi_head_attention_9/key/add/ReadVariableOpReadVariableOpJtransformer_block_9_multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_9/multi_head_attention_9/key/add/ReadVariableOp½
2transformer_block_9/multi_head_attention_9/key/addAddV2Etransformer_block_9/multi_head_attention_9/key/einsum/Einsum:output:0Itransformer_block_9/multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 24
2transformer_block_9/multi_head_attention_9/key/add¹
Mtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_9_multi_head_attention_9_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpÐ
>transformer_block_9/multi_head_attention_9/value/einsum/EinsumEinsumadd_4/add:z:0Utransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2@
>transformer_block_9/multi_head_attention_9/value/einsum/Einsum
Ctransformer_block_9/multi_head_attention_9/value/add/ReadVariableOpReadVariableOpLtransformer_block_9_multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_9/multi_head_attention_9/value/add/ReadVariableOpÅ
4transformer_block_9/multi_head_attention_9/value/addAddV2Gtransformer_block_9/multi_head_attention_9/value/einsum/Einsum:output:0Ktransformer_block_9/multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_9/multi_head_attention_9/value/add©
0transformer_block_9/multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>22
0transformer_block_9/multi_head_attention_9/Mul/y
.transformer_block_9/multi_head_attention_9/MulMul8transformer_block_9/multi_head_attention_9/query/add:z:09transformer_block_9/multi_head_attention_9/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 20
.transformer_block_9/multi_head_attention_9/MulÌ
8transformer_block_9/multi_head_attention_9/einsum/EinsumEinsum6transformer_block_9/multi_head_attention_9/key/add:z:02transformer_block_9/multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2:
8transformer_block_9/multi_head_attention_9/einsum/Einsum
:transformer_block_9/multi_head_attention_9/softmax/SoftmaxSoftmaxAtransformer_block_9/multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2<
:transformer_block_9/multi_head_attention_9/softmax/Softmax
;transformer_block_9/multi_head_attention_9/dropout/IdentityIdentityDtransformer_block_9/multi_head_attention_9/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2=
;transformer_block_9/multi_head_attention_9/dropout/Identityä
:transformer_block_9/multi_head_attention_9/einsum_1/EinsumEinsumDtransformer_block_9/multi_head_attention_9/dropout/Identity:output:08transformer_block_9/multi_head_attention_9/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2<
:transformer_block_9/multi_head_attention_9/einsum_1/EinsumÚ
Xtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_9_multi_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp£
Itransformer_block_9/multi_head_attention_9/attention_output/einsum/EinsumEinsumCtransformer_block_9/multi_head_attention_9/einsum_1/Einsum:output:0`transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe2K
Itransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum´
Ntransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_9_multi_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpí
?transformer_block_9/multi_head_attention_9/attention_output/addAddV2Rtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum:output:0Vtransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?transformer_block_9/multi_head_attention_9/attention_output/addÙ
'transformer_block_9/dropout_26/IdentityIdentityCtransformer_block_9/multi_head_attention_9/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2)
'transformer_block_9/dropout_26/Identity²
transformer_block_9/addAddV2add_4/add:z:00transformer_block_9/dropout_26/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_9/addà
Itransformer_block_9/layer_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_9/layer_normalization_18/moments/mean/reduction_indices²
7transformer_block_9/layer_normalization_18/moments/meanMeantransformer_block_9/add:z:0Rtransformer_block_9/layer_normalization_18/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(29
7transformer_block_9/layer_normalization_18/moments/mean
?transformer_block_9/layer_normalization_18/moments/StopGradientStopGradient@transformer_block_9/layer_normalization_18/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2A
?transformer_block_9/layer_normalization_18/moments/StopGradient¾
Dtransformer_block_9/layer_normalization_18/moments/SquaredDifferenceSquaredDifferencetransformer_block_9/add:z:0Htransformer_block_9/layer_normalization_18/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2F
Dtransformer_block_9/layer_normalization_18/moments/SquaredDifferenceè
Mtransformer_block_9/layer_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_9/layer_normalization_18/moments/variance/reduction_indicesë
;transformer_block_9/layer_normalization_18/moments/varianceMeanHtransformer_block_9/layer_normalization_18/moments/SquaredDifference:z:0Vtransformer_block_9/layer_normalization_18/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2=
;transformer_block_9/layer_normalization_18/moments/variance½
:transformer_block_9/layer_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_9/layer_normalization_18/batchnorm/add/y¾
8transformer_block_9/layer_normalization_18/batchnorm/addAddV2Dtransformer_block_9/layer_normalization_18/moments/variance:output:0Ctransformer_block_9/layer_normalization_18/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2:
8transformer_block_9/layer_normalization_18/batchnorm/addõ
:transformer_block_9/layer_normalization_18/batchnorm/RsqrtRsqrt<transformer_block_9/layer_normalization_18/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2<
:transformer_block_9/layer_normalization_18/batchnorm/Rsqrt
Gtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_9_layer_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpÂ
8transformer_block_9/layer_normalization_18/batchnorm/mulMul>transformer_block_9/layer_normalization_18/batchnorm/Rsqrt:y:0Otransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2:
8transformer_block_9/layer_normalization_18/batchnorm/mul
:transformer_block_9/layer_normalization_18/batchnorm/mul_1Multransformer_block_9/add:z:0<transformer_block_9/layer_normalization_18/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_9/layer_normalization_18/batchnorm/mul_1µ
:transformer_block_9/layer_normalization_18/batchnorm/mul_2Mul@transformer_block_9/layer_normalization_18/moments/mean:output:0<transformer_block_9/layer_normalization_18/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_9/layer_normalization_18/batchnorm/mul_2
Ctransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_9_layer_normalization_18_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOp¾
8transformer_block_9/layer_normalization_18/batchnorm/subSubKtransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOp:value:0>transformer_block_9/layer_normalization_18/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2:
8transformer_block_9/layer_normalization_18/batchnorm/subµ
:transformer_block_9/layer_normalization_18/batchnorm/add_1AddV2>transformer_block_9/layer_normalization_18/batchnorm/mul_1:z:0<transformer_block_9/layer_normalization_18/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_9/layer_normalization_18/batchnorm/add_1
Btransformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_9_sequential_9_dense_30_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02D
Btransformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOp¾
8transformer_block_9/sequential_9/dense_30/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_9/sequential_9/dense_30/Tensordot/axesÅ
8transformer_block_9/sequential_9/dense_30/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_9/sequential_9/dense_30/Tensordot/freeä
9transformer_block_9/sequential_9/dense_30/Tensordot/ShapeShape>transformer_block_9/layer_normalization_18/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_9/sequential_9/dense_30/Tensordot/ShapeÈ
Atransformer_block_9/sequential_9/dense_30/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_9/sequential_9/dense_30/Tensordot/GatherV2/axis£
<transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2GatherV2Btransformer_block_9/sequential_9/dense_30/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_30/Tensordot/free:output:0Jtransformer_block_9/sequential_9/dense_30/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2Ì
Ctransformer_block_9/sequential_9/dense_30/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_9/sequential_9/dense_30/Tensordot/GatherV2_1/axis©
>transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2_1GatherV2Btransformer_block_9/sequential_9/dense_30/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_30/Tensordot/axes:output:0Ltransformer_block_9/sequential_9/dense_30/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2_1À
9transformer_block_9/sequential_9/dense_30/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_9/sequential_9/dense_30/Tensordot/Const¨
8transformer_block_9/sequential_9/dense_30/Tensordot/ProdProdEtransformer_block_9/sequential_9/dense_30/Tensordot/GatherV2:output:0Btransformer_block_9/sequential_9/dense_30/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_9/sequential_9/dense_30/Tensordot/ProdÄ
;transformer_block_9/sequential_9/dense_30/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_9/sequential_9/dense_30/Tensordot/Const_1°
:transformer_block_9/sequential_9/dense_30/Tensordot/Prod_1ProdGtransformer_block_9/sequential_9/dense_30/Tensordot/GatherV2_1:output:0Dtransformer_block_9/sequential_9/dense_30/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_9/sequential_9/dense_30/Tensordot/Prod_1Ä
?transformer_block_9/sequential_9/dense_30/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_9/sequential_9/dense_30/Tensordot/concat/axis
:transformer_block_9/sequential_9/dense_30/Tensordot/concatConcatV2Atransformer_block_9/sequential_9/dense_30/Tensordot/free:output:0Atransformer_block_9/sequential_9/dense_30/Tensordot/axes:output:0Htransformer_block_9/sequential_9/dense_30/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_9/sequential_9/dense_30/Tensordot/concat´
9transformer_block_9/sequential_9/dense_30/Tensordot/stackPackAtransformer_block_9/sequential_9/dense_30/Tensordot/Prod:output:0Ctransformer_block_9/sequential_9/dense_30/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_9/sequential_9/dense_30/Tensordot/stackÆ
=transformer_block_9/sequential_9/dense_30/Tensordot/transpose	Transpose>transformer_block_9/layer_normalization_18/batchnorm/add_1:z:0Ctransformer_block_9/sequential_9/dense_30/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2?
=transformer_block_9/sequential_9/dense_30/Tensordot/transposeÇ
;transformer_block_9/sequential_9/dense_30/Tensordot/ReshapeReshapeAtransformer_block_9/sequential_9/dense_30/Tensordot/transpose:y:0Btransformer_block_9/sequential_9/dense_30/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2=
;transformer_block_9/sequential_9/dense_30/Tensordot/ReshapeÆ
:transformer_block_9/sequential_9/dense_30/Tensordot/MatMulMatMulDtransformer_block_9/sequential_9/dense_30/Tensordot/Reshape:output:0Jtransformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2<
:transformer_block_9/sequential_9/dense_30/Tensordot/MatMulÄ
;transformer_block_9/sequential_9/dense_30/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2=
;transformer_block_9/sequential_9/dense_30/Tensordot/Const_2È
Atransformer_block_9/sequential_9/dense_30/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_9/sequential_9/dense_30/Tensordot/concat_1/axis
<transformer_block_9/sequential_9/dense_30/Tensordot/concat_1ConcatV2Etransformer_block_9/sequential_9/dense_30/Tensordot/GatherV2:output:0Dtransformer_block_9/sequential_9/dense_30/Tensordot/Const_2:output:0Jtransformer_block_9/sequential_9/dense_30/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_9/sequential_9/dense_30/Tensordot/concat_1¸
3transformer_block_9/sequential_9/dense_30/TensordotReshapeDtransformer_block_9/sequential_9/dense_30/Tensordot/MatMul:product:0Etransformer_block_9/sequential_9/dense_30/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@25
3transformer_block_9/sequential_9/dense_30/Tensordot
@transformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_9_sequential_9_dense_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@transformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOp¯
1transformer_block_9/sequential_9/dense_30/BiasAddBiasAdd<transformer_block_9/sequential_9/dense_30/Tensordot:output:0Htransformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@23
1transformer_block_9/sequential_9/dense_30/BiasAddÚ
.transformer_block_9/sequential_9/dense_30/ReluRelu:transformer_block_9/sequential_9/dense_30/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@20
.transformer_block_9/sequential_9/dense_30/Relu
Btransformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_9_sequential_9_dense_31_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02D
Btransformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOp¾
8transformer_block_9/sequential_9/dense_31/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_9/sequential_9/dense_31/Tensordot/axesÅ
8transformer_block_9/sequential_9/dense_31/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_9/sequential_9/dense_31/Tensordot/freeâ
9transformer_block_9/sequential_9/dense_31/Tensordot/ShapeShape<transformer_block_9/sequential_9/dense_30/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_9/sequential_9/dense_31/Tensordot/ShapeÈ
Atransformer_block_9/sequential_9/dense_31/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_9/sequential_9/dense_31/Tensordot/GatherV2/axis£
<transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2GatherV2Btransformer_block_9/sequential_9/dense_31/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_31/Tensordot/free:output:0Jtransformer_block_9/sequential_9/dense_31/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2Ì
Ctransformer_block_9/sequential_9/dense_31/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_9/sequential_9/dense_31/Tensordot/GatherV2_1/axis©
>transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2_1GatherV2Btransformer_block_9/sequential_9/dense_31/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_31/Tensordot/axes:output:0Ltransformer_block_9/sequential_9/dense_31/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2_1À
9transformer_block_9/sequential_9/dense_31/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_9/sequential_9/dense_31/Tensordot/Const¨
8transformer_block_9/sequential_9/dense_31/Tensordot/ProdProdEtransformer_block_9/sequential_9/dense_31/Tensordot/GatherV2:output:0Btransformer_block_9/sequential_9/dense_31/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_9/sequential_9/dense_31/Tensordot/ProdÄ
;transformer_block_9/sequential_9/dense_31/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_9/sequential_9/dense_31/Tensordot/Const_1°
:transformer_block_9/sequential_9/dense_31/Tensordot/Prod_1ProdGtransformer_block_9/sequential_9/dense_31/Tensordot/GatherV2_1:output:0Dtransformer_block_9/sequential_9/dense_31/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_9/sequential_9/dense_31/Tensordot/Prod_1Ä
?transformer_block_9/sequential_9/dense_31/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_9/sequential_9/dense_31/Tensordot/concat/axis
:transformer_block_9/sequential_9/dense_31/Tensordot/concatConcatV2Atransformer_block_9/sequential_9/dense_31/Tensordot/free:output:0Atransformer_block_9/sequential_9/dense_31/Tensordot/axes:output:0Htransformer_block_9/sequential_9/dense_31/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_9/sequential_9/dense_31/Tensordot/concat´
9transformer_block_9/sequential_9/dense_31/Tensordot/stackPackAtransformer_block_9/sequential_9/dense_31/Tensordot/Prod:output:0Ctransformer_block_9/sequential_9/dense_31/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_9/sequential_9/dense_31/Tensordot/stackÄ
=transformer_block_9/sequential_9/dense_31/Tensordot/transpose	Transpose<transformer_block_9/sequential_9/dense_30/Relu:activations:0Ctransformer_block_9/sequential_9/dense_31/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2?
=transformer_block_9/sequential_9/dense_31/Tensordot/transposeÇ
;transformer_block_9/sequential_9/dense_31/Tensordot/ReshapeReshapeAtransformer_block_9/sequential_9/dense_31/Tensordot/transpose:y:0Btransformer_block_9/sequential_9/dense_31/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2=
;transformer_block_9/sequential_9/dense_31/Tensordot/ReshapeÆ
:transformer_block_9/sequential_9/dense_31/Tensordot/MatMulMatMulDtransformer_block_9/sequential_9/dense_31/Tensordot/Reshape:output:0Jtransformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2<
:transformer_block_9/sequential_9/dense_31/Tensordot/MatMulÄ
;transformer_block_9/sequential_9/dense_31/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_9/sequential_9/dense_31/Tensordot/Const_2È
Atransformer_block_9/sequential_9/dense_31/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_9/sequential_9/dense_31/Tensordot/concat_1/axis
<transformer_block_9/sequential_9/dense_31/Tensordot/concat_1ConcatV2Etransformer_block_9/sequential_9/dense_31/Tensordot/GatherV2:output:0Dtransformer_block_9/sequential_9/dense_31/Tensordot/Const_2:output:0Jtransformer_block_9/sequential_9/dense_31/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_9/sequential_9/dense_31/Tensordot/concat_1¸
3transformer_block_9/sequential_9/dense_31/TensordotReshapeDtransformer_block_9/sequential_9/dense_31/Tensordot/MatMul:product:0Etransformer_block_9/sequential_9/dense_31/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 25
3transformer_block_9/sequential_9/dense_31/Tensordot
@transformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_9_sequential_9_dense_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOp¯
1transformer_block_9/sequential_9/dense_31/BiasAddBiasAdd<transformer_block_9/sequential_9/dense_31/Tensordot:output:0Htransformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 23
1transformer_block_9/sequential_9/dense_31/BiasAddÐ
'transformer_block_9/dropout_27/IdentityIdentity:transformer_block_9/sequential_9/dense_31/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2)
'transformer_block_9/dropout_27/Identityç
transformer_block_9/add_1AddV2>transformer_block_9/layer_normalization_18/batchnorm/add_1:z:00transformer_block_9/dropout_27/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_9/add_1à
Itransformer_block_9/layer_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_9/layer_normalization_19/moments/mean/reduction_indices´
7transformer_block_9/layer_normalization_19/moments/meanMeantransformer_block_9/add_1:z:0Rtransformer_block_9/layer_normalization_19/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(29
7transformer_block_9/layer_normalization_19/moments/mean
?transformer_block_9/layer_normalization_19/moments/StopGradientStopGradient@transformer_block_9/layer_normalization_19/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2A
?transformer_block_9/layer_normalization_19/moments/StopGradientÀ
Dtransformer_block_9/layer_normalization_19/moments/SquaredDifferenceSquaredDifferencetransformer_block_9/add_1:z:0Htransformer_block_9/layer_normalization_19/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2F
Dtransformer_block_9/layer_normalization_19/moments/SquaredDifferenceè
Mtransformer_block_9/layer_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_9/layer_normalization_19/moments/variance/reduction_indicesë
;transformer_block_9/layer_normalization_19/moments/varianceMeanHtransformer_block_9/layer_normalization_19/moments/SquaredDifference:z:0Vtransformer_block_9/layer_normalization_19/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2=
;transformer_block_9/layer_normalization_19/moments/variance½
:transformer_block_9/layer_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_9/layer_normalization_19/batchnorm/add/y¾
8transformer_block_9/layer_normalization_19/batchnorm/addAddV2Dtransformer_block_9/layer_normalization_19/moments/variance:output:0Ctransformer_block_9/layer_normalization_19/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2:
8transformer_block_9/layer_normalization_19/batchnorm/addõ
:transformer_block_9/layer_normalization_19/batchnorm/RsqrtRsqrt<transformer_block_9/layer_normalization_19/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2<
:transformer_block_9/layer_normalization_19/batchnorm/Rsqrt
Gtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_9_layer_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpÂ
8transformer_block_9/layer_normalization_19/batchnorm/mulMul>transformer_block_9/layer_normalization_19/batchnorm/Rsqrt:y:0Otransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2:
8transformer_block_9/layer_normalization_19/batchnorm/mul
:transformer_block_9/layer_normalization_19/batchnorm/mul_1Multransformer_block_9/add_1:z:0<transformer_block_9/layer_normalization_19/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_9/layer_normalization_19/batchnorm/mul_1µ
:transformer_block_9/layer_normalization_19/batchnorm/mul_2Mul@transformer_block_9/layer_normalization_19/moments/mean:output:0<transformer_block_9/layer_normalization_19/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_9/layer_normalization_19/batchnorm/mul_2
Ctransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_9_layer_normalization_19_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOp¾
8transformer_block_9/layer_normalization_19/batchnorm/subSubKtransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOp:value:0>transformer_block_9/layer_normalization_19/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2:
8transformer_block_9/layer_normalization_19/batchnorm/subµ
:transformer_block_9/layer_normalization_19/batchnorm/add_1AddV2>transformer_block_9/layer_normalization_19/batchnorm/mul_1:z:0<transformer_block_9/layer_normalization_19/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_9/layer_normalization_19/batchnorm/add_1s
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`  2
flatten_4/Const¾
flatten_4/ReshapeReshape>transformer_block_9/layer_normalization_19/batchnorm/add_1:z:0flatten_4/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
flatten_4/Reshapex
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_4/concat/axis¾
concatenate_4/concatConcatV2flatten_4/Reshape:output:0inputs_1"concatenate_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
concatenate_4/concat©
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	è@*
dtype02 
dense_32/MatMul/ReadVariableOp¥
dense_32/MatMulMatMulconcatenate_4/concat:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_32/MatMul§
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_32/BiasAdd/ReadVariableOp¥
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_32/BiasAdds
dense_32/ReluReludense_32/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_32/Relu
dropout_28/IdentityIdentitydense_32/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_28/Identity¨
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_33/MatMul/ReadVariableOp¤
dense_33/MatMulMatMuldropout_28/Identity:output:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_33/MatMul§
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_33/BiasAdd/ReadVariableOp¥
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_33/BiasAdds
dense_33/ReluReludense_33/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_33/Relu
dropout_29/IdentityIdentitydense_33/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_29/Identity¨
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_34/MatMul/ReadVariableOp¤
dense_34/MatMulMatMuldropout_29/Identity:output:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_34/MatMul§
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_34/BiasAdd/ReadVariableOp¥
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_34/BiasAdd
IdentityIdentitydense_34/BiasAdd:output:0/^batch_normalization_8/batchnorm/ReadVariableOp1^batch_normalization_8/batchnorm/ReadVariableOp_11^batch_normalization_8/batchnorm/ReadVariableOp_23^batch_normalization_8/batchnorm/mul/ReadVariableOp/^batch_normalization_9/batchnorm/ReadVariableOp1^batch_normalization_9/batchnorm/ReadVariableOp_11^batch_normalization_9/batchnorm/ReadVariableOp_23^batch_normalization_9/batchnorm/mul/ReadVariableOp ^conv1d_8/BiasAdd/ReadVariableOp,^conv1d_8/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_9/BiasAdd/ReadVariableOp,^conv1d_9/conv1d/ExpandDims_1/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp<^token_and_position_embedding_4/embedding_8/embedding_lookup<^token_and_position_embedding_4/embedding_9/embedding_lookupD^transformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpH^transformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpD^transformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpH^transformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpO^transformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpY^transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_9/multi_head_attention_9/key/add/ReadVariableOpL^transformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpD^transformer_block_9/multi_head_attention_9/query/add/ReadVariableOpN^transformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpD^transformer_block_9/multi_head_attention_9/value/add/ReadVariableOpN^transformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpA^transformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOpC^transformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOpA^transformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOpC^transformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2`
.batch_normalization_8/batchnorm/ReadVariableOp.batch_normalization_8/batchnorm/ReadVariableOp2d
0batch_normalization_8/batchnorm/ReadVariableOp_10batch_normalization_8/batchnorm/ReadVariableOp_12d
0batch_normalization_8/batchnorm/ReadVariableOp_20batch_normalization_8/batchnorm/ReadVariableOp_22h
2batch_normalization_8/batchnorm/mul/ReadVariableOp2batch_normalization_8/batchnorm/mul/ReadVariableOp2`
.batch_normalization_9/batchnorm/ReadVariableOp.batch_normalization_9/batchnorm/ReadVariableOp2d
0batch_normalization_9/batchnorm/ReadVariableOp_10batch_normalization_9/batchnorm/ReadVariableOp_12d
0batch_normalization_9/batchnorm/ReadVariableOp_20batch_normalization_9/batchnorm/ReadVariableOp_22h
2batch_normalization_9/batchnorm/mul/ReadVariableOp2batch_normalization_9/batchnorm/mul/ReadVariableOp2B
conv1d_8/BiasAdd/ReadVariableOpconv1d_8/BiasAdd/ReadVariableOp2Z
+conv1d_8/conv1d/ExpandDims_1/ReadVariableOp+conv1d_8/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_9/BiasAdd/ReadVariableOpconv1d_9/BiasAdd/ReadVariableOp2Z
+conv1d_9/conv1d/ExpandDims_1/ReadVariableOp+conv1d_9/conv1d/ExpandDims_1/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2z
;token_and_position_embedding_4/embedding_8/embedding_lookup;token_and_position_embedding_4/embedding_8/embedding_lookup2z
;token_and_position_embedding_4/embedding_9/embedding_lookup;token_and_position_embedding_4/embedding_9/embedding_lookup2
Ctransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpCtransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOp2
Gtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpGtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOp2
Ctransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpCtransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOp2
Gtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpGtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOp2 
Ntransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpNtransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOp2´
Xtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2
Atransformer_block_9/multi_head_attention_9/key/add/ReadVariableOpAtransformer_block_9/multi_head_attention_9/key/add/ReadVariableOp2
Ktransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpKtransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2
Ctransformer_block_9/multi_head_attention_9/query/add/ReadVariableOpCtransformer_block_9/multi_head_attention_9/query/add/ReadVariableOp2
Mtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpMtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2
Ctransformer_block_9/multi_head_attention_9/value/add/ReadVariableOpCtransformer_block_9/multi_head_attention_9/value/add/ReadVariableOp2
Mtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpMtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp2
@transformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOp@transformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOp2
Btransformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOpBtransformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOp2
@transformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOp@transformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOp2
Btransformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOpBtransformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOp:R N
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
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_513343

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
º
s
I__inference_concatenate_4_layer_call_and_return_conditional_losses_514235

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
¨
Z
.__inference_concatenate_4_layer_call_fn_516415
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
I__inference_concatenate_4_layer_call_and_return_conditional_losses_5142352
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

Q
5__inference_average_pooling1d_12_layer_call_fn_513044

inputs
identityç
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_12_layer_call_and_return_conditional_losses_5130382
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
Õê
¤&
C__inference_model_4_layer_call_and_return_conditional_losses_515220
inputs_0
inputs_1F
Btoken_and_position_embedding_4_embedding_9_embedding_lookup_514922F
Btoken_and_position_embedding_4_embedding_8_embedding_lookup_5149288
4conv1d_8_conv1d_expanddims_1_readvariableop_resource,
(conv1d_8_biasadd_readvariableop_resource8
4conv1d_9_conv1d_expanddims_1_readvariableop_resource,
(conv1d_9_biasadd_readvariableop_resource0
,batch_normalization_8_assignmovingavg_5149782
.batch_normalization_8_assignmovingavg_1_514984?
;batch_normalization_8_batchnorm_mul_readvariableop_resource;
7batch_normalization_8_batchnorm_readvariableop_resource0
,batch_normalization_9_assignmovingavg_5150102
.batch_normalization_9_assignmovingavg_1_515016?
;batch_normalization_9_batchnorm_mul_readvariableop_resource;
7batch_normalization_9_batchnorm_readvariableop_resourceZ
Vtransformer_block_9_multi_head_attention_9_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_9_multi_head_attention_9_query_add_readvariableop_resourceX
Ttransformer_block_9_multi_head_attention_9_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_9_multi_head_attention_9_key_add_readvariableop_resourceZ
Vtransformer_block_9_multi_head_attention_9_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_9_multi_head_attention_9_value_add_readvariableop_resourcee
atransformer_block_9_multi_head_attention_9_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_9_multi_head_attention_9_attention_output_add_readvariableop_resourceT
Ptransformer_block_9_layer_normalization_18_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_9_layer_normalization_18_batchnorm_readvariableop_resourceO
Ktransformer_block_9_sequential_9_dense_30_tensordot_readvariableop_resourceM
Itransformer_block_9_sequential_9_dense_30_biasadd_readvariableop_resourceO
Ktransformer_block_9_sequential_9_dense_31_tensordot_readvariableop_resourceM
Itransformer_block_9_sequential_9_dense_31_biasadd_readvariableop_resourceT
Ptransformer_block_9_layer_normalization_19_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_9_layer_normalization_19_batchnorm_readvariableop_resource+
'dense_32_matmul_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource
identity¢9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_8/AssignMovingAvg/ReadVariableOp¢;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_8/batchnorm/ReadVariableOp¢2batch_normalization_8/batchnorm/mul/ReadVariableOp¢9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_9/AssignMovingAvg/ReadVariableOp¢;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_9/batchnorm/ReadVariableOp¢2batch_normalization_9/batchnorm/mul/ReadVariableOp¢conv1d_8/BiasAdd/ReadVariableOp¢+conv1d_8/conv1d/ExpandDims_1/ReadVariableOp¢conv1d_9/BiasAdd/ReadVariableOp¢+conv1d_9/conv1d/ExpandDims_1/ReadVariableOp¢dense_32/BiasAdd/ReadVariableOp¢dense_32/MatMul/ReadVariableOp¢dense_33/BiasAdd/ReadVariableOp¢dense_33/MatMul/ReadVariableOp¢dense_34/BiasAdd/ReadVariableOp¢dense_34/MatMul/ReadVariableOp¢;token_and_position_embedding_4/embedding_8/embedding_lookup¢;token_and_position_embedding_4/embedding_9/embedding_lookup¢Ctransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOp¢Gtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOp¢Ctransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOp¢Gtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOp¢Ntransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOp¢Xtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp¢Atransformer_block_9/multi_head_attention_9/key/add/ReadVariableOp¢Ktransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp¢Ctransformer_block_9/multi_head_attention_9/query/add/ReadVariableOp¢Mtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp¢Ctransformer_block_9/multi_head_attention_9/value/add/ReadVariableOp¢Mtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp¢@transformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOp¢Btransformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOp¢@transformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOp¢Btransformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOp
$token_and_position_embedding_4/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_4/Shape»
2token_and_position_embedding_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ24
2token_and_position_embedding_4/strided_slice/stack¶
4token_and_position_embedding_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_4/strided_slice/stack_1¶
4token_and_position_embedding_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_4/strided_slice/stack_2
,token_and_position_embedding_4/strided_sliceStridedSlice-token_and_position_embedding_4/Shape:output:0;token_and_position_embedding_4/strided_slice/stack:output:0=token_and_position_embedding_4/strided_slice/stack_1:output:0=token_and_position_embedding_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_4/strided_slice
*token_and_position_embedding_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_4/range/start
*token_and_position_embedding_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_4/range/delta
$token_and_position_embedding_4/rangeRange3token_and_position_embedding_4/range/start:output:05token_and_position_embedding_4/strided_slice:output:03token_and_position_embedding_4/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$token_and_position_embedding_4/rangeÊ
;token_and_position_embedding_4/embedding_9/embedding_lookupResourceGatherBtoken_and_position_embedding_4_embedding_9_embedding_lookup_514922-token_and_position_embedding_4/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_4/embedding_9/embedding_lookup/514922*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02=
;token_and_position_embedding_4/embedding_9/embedding_lookup
Dtoken_and_position_embedding_4/embedding_9/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_4/embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_4/embedding_9/embedding_lookup/514922*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2F
Dtoken_and_position_embedding_4/embedding_9/embedding_lookup/Identity
Ftoken_and_position_embedding_4/embedding_9/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_4/embedding_9/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2H
Ftoken_and_position_embedding_4/embedding_9/embedding_lookup/Identity_1¶
/token_and_position_embedding_4/embedding_8/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR21
/token_and_position_embedding_4/embedding_8/CastÕ
;token_and_position_embedding_4/embedding_8/embedding_lookupResourceGatherBtoken_and_position_embedding_4_embedding_8_embedding_lookup_5149283token_and_position_embedding_4/embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_4/embedding_8/embedding_lookup/514928*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02=
;token_and_position_embedding_4/embedding_8/embedding_lookup
Dtoken_and_position_embedding_4/embedding_8/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_4/embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_4/embedding_8/embedding_lookup/514928*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2F
Dtoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity¢
Ftoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2H
Ftoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity_1ª
"token_and_position_embedding_4/addAddV2Otoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_4/embedding_9/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2$
"token_and_position_embedding_4/add
conv1d_8/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_8/conv1d/ExpandDims/dimÒ
conv1d_8/conv1d/ExpandDims
ExpandDims&token_and_position_embedding_4/add:z:0'conv1d_8/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_8/conv1d/ExpandDimsÓ
+conv1d_8/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_8_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02-
+conv1d_8/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_8/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_8/conv1d/ExpandDims_1/dimÛ
conv1d_8/conv1d/ExpandDims_1
ExpandDims3conv1d_8/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_8/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_8/conv1d/ExpandDims_1Û
conv1d_8/conv1dConv2D#conv1d_8/conv1d/ExpandDims:output:0%conv1d_8/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
paddingSAME*
strides
2
conv1d_8/conv1d®
conv1d_8/conv1d/SqueezeSqueezeconv1d_8/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_8/conv1d/Squeeze§
conv1d_8/BiasAdd/ReadVariableOpReadVariableOp(conv1d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_8/BiasAdd/ReadVariableOp±
conv1d_8/BiasAddBiasAdd conv1d_8/conv1d/Squeeze:output:0'conv1d_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_8/BiasAddx
conv1d_8/ReluReluconv1d_8/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_8/Relu
#average_pooling1d_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_12/ExpandDims/dimÖ
average_pooling1d_12/ExpandDims
ExpandDimsconv1d_8/Relu:activations:0,average_pooling1d_12/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2!
average_pooling1d_12/ExpandDimsè
average_pooling1d_12/AvgPoolAvgPool(average_pooling1d_12/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_12/AvgPool¼
average_pooling1d_12/SqueezeSqueeze%average_pooling1d_12/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims
2
average_pooling1d_12/Squeeze
conv1d_9/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_9/conv1d/ExpandDims/dimÑ
conv1d_9/conv1d/ExpandDims
ExpandDims%average_pooling1d_12/Squeeze:output:0'conv1d_9/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_9/conv1d/ExpandDimsÓ
+conv1d_9/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_9_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02-
+conv1d_9/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_9/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_9/conv1d/ExpandDims_1/dimÛ
conv1d_9/conv1d/ExpandDims_1
ExpandDims3conv1d_9/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_9/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d_9/conv1d/ExpandDims_1Û
conv1d_9/conv1dConv2D#conv1d_9/conv1d/ExpandDims:output:0%conv1d_9/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
paddingSAME*
strides
2
conv1d_9/conv1d®
conv1d_9/conv1d/SqueezeSqueezeconv1d_9/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_9/conv1d/Squeeze§
conv1d_9/BiasAdd/ReadVariableOpReadVariableOp(conv1d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_9/BiasAdd/ReadVariableOp±
conv1d_9/BiasAddBiasAdd conv1d_9/conv1d/Squeeze:output:0'conv1d_9/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_9/BiasAddx
conv1d_9/ReluReluconv1d_9/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_9/Relu
#average_pooling1d_14/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_14/ExpandDims/dimá
average_pooling1d_14/ExpandDims
ExpandDims&token_and_position_embedding_4/add:z:0,average_pooling1d_14/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2!
average_pooling1d_14/ExpandDimsé
average_pooling1d_14/AvgPoolAvgPool(average_pooling1d_14/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize	
¬*
paddingVALID*
strides	
¬2
average_pooling1d_14/AvgPool»
average_pooling1d_14/SqueezeSqueeze%average_pooling1d_14/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_14/Squeeze
#average_pooling1d_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_13/ExpandDims/dimÖ
average_pooling1d_13/ExpandDims
ExpandDimsconv1d_9/Relu:activations:0,average_pooling1d_13/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2!
average_pooling1d_13/ExpandDimsç
average_pooling1d_13/AvgPoolAvgPool(average_pooling1d_13/ExpandDims:output:0*
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
average_pooling1d_13/AvgPool»
average_pooling1d_13/SqueezeSqueeze%average_pooling1d_13/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_13/Squeeze½
4batch_normalization_8/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_8/moments/mean/reduction_indicesô
"batch_normalization_8/moments/meanMean%average_pooling1d_13/Squeeze:output:0=batch_normalization_8/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_8/moments/meanÂ
*batch_normalization_8/moments/StopGradientStopGradient+batch_normalization_8/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_8/moments/StopGradient
/batch_normalization_8/moments/SquaredDifferenceSquaredDifference%average_pooling1d_13/Squeeze:output:03batch_normalization_8/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/batch_normalization_8/moments/SquaredDifferenceÅ
8batch_normalization_8/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_8/moments/variance/reduction_indices
&batch_normalization_8/moments/varianceMean3batch_normalization_8/moments/SquaredDifference:z:0Abatch_normalization_8/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2(
&batch_normalization_8/moments/varianceÃ
%batch_normalization_8/moments/SqueezeSqueeze+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_8/moments/SqueezeË
'batch_normalization_8/moments/Squeeze_1Squeeze/batch_normalization_8/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_8/moments/Squeeze_1
+batch_normalization_8/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_8/AssignMovingAvg/514978*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_8/AssignMovingAvg/decayÕ
4batch_normalization_8/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_8_assignmovingavg_514978*
_output_shapes
: *
dtype026
4batch_normalization_8/AssignMovingAvg/ReadVariableOpß
)batch_normalization_8/AssignMovingAvg/subSub<batch_normalization_8/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_8/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_8/AssignMovingAvg/514978*
_output_shapes
: 2+
)batch_normalization_8/AssignMovingAvg/subÖ
)batch_normalization_8/AssignMovingAvg/mulMul-batch_normalization_8/AssignMovingAvg/sub:z:04batch_normalization_8/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_8/AssignMovingAvg/514978*
_output_shapes
: 2+
)batch_normalization_8/AssignMovingAvg/mul³
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_8_assignmovingavg_514978-batch_normalization_8/AssignMovingAvg/mul:z:05^batch_normalization_8/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_8/AssignMovingAvg/514978*
_output_shapes
 *
dtype02;
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_8/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg_1/514984*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_8/AssignMovingAvg_1/decayÛ
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_8_assignmovingavg_1_514984*
_output_shapes
: *
dtype028
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpé
+batch_normalization_8/AssignMovingAvg_1/subSub>batch_normalization_8/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_8/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg_1/514984*
_output_shapes
: 2-
+batch_normalization_8/AssignMovingAvg_1/subà
+batch_normalization_8/AssignMovingAvg_1/mulMul/batch_normalization_8/AssignMovingAvg_1/sub:z:06batch_normalization_8/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg_1/514984*
_output_shapes
: 2-
+batch_normalization_8/AssignMovingAvg_1/mul¿
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_8_assignmovingavg_1_514984/batch_normalization_8/AssignMovingAvg_1/mul:z:07^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg_1/514984*
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
: 2%
#batch_normalization_8/batchnorm/add¥
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_8/batchnorm/Rsqrtà
2batch_normalization_8/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_8_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_8/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:0:batch_normalization_8/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_8/batchnorm/mulÛ
%batch_normalization_8/batchnorm/mul_1Mul%average_pooling1d_13/Squeeze:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_8/batchnorm/mul_1Ó
%batch_normalization_8/batchnorm/mul_2Mul.batch_normalization_8/moments/Squeeze:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_8/batchnorm/mul_2Ô
.batch_normalization_8/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_8_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_8/batchnorm/ReadVariableOpÙ
#batch_normalization_8/batchnorm/subSub6batch_normalization_8/batchnorm/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_8/batchnorm/subá
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_8/batchnorm/add_1½
4batch_normalization_9/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_9/moments/mean/reduction_indicesô
"batch_normalization_9/moments/meanMean%average_pooling1d_14/Squeeze:output:0=batch_normalization_9/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_9/moments/meanÂ
*batch_normalization_9/moments/StopGradientStopGradient+batch_normalization_9/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_9/moments/StopGradient
/batch_normalization_9/moments/SquaredDifferenceSquaredDifference%average_pooling1d_14/Squeeze:output:03batch_normalization_9/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/batch_normalization_9/moments/SquaredDifferenceÅ
8batch_normalization_9/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_9/moments/variance/reduction_indices
&batch_normalization_9/moments/varianceMean3batch_normalization_9/moments/SquaredDifference:z:0Abatch_normalization_9/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2(
&batch_normalization_9/moments/varianceÃ
%batch_normalization_9/moments/SqueezeSqueeze+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_9/moments/SqueezeË
'batch_normalization_9/moments/Squeeze_1Squeeze/batch_normalization_9/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_9/moments/Squeeze_1
+batch_normalization_9/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_9/AssignMovingAvg/515010*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_9/AssignMovingAvg/decayÕ
4batch_normalization_9/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_9_assignmovingavg_515010*
_output_shapes
: *
dtype026
4batch_normalization_9/AssignMovingAvg/ReadVariableOpß
)batch_normalization_9/AssignMovingAvg/subSub<batch_normalization_9/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_9/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_9/AssignMovingAvg/515010*
_output_shapes
: 2+
)batch_normalization_9/AssignMovingAvg/subÖ
)batch_normalization_9/AssignMovingAvg/mulMul-batch_normalization_9/AssignMovingAvg/sub:z:04batch_normalization_9/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_9/AssignMovingAvg/515010*
_output_shapes
: 2+
)batch_normalization_9/AssignMovingAvg/mul³
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_9_assignmovingavg_515010-batch_normalization_9/AssignMovingAvg/mul:z:05^batch_normalization_9/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_9/AssignMovingAvg/515010*
_output_shapes
 *
dtype02;
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_9/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg_1/515016*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_9/AssignMovingAvg_1/decayÛ
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_9_assignmovingavg_1_515016*
_output_shapes
: *
dtype028
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpé
+batch_normalization_9/AssignMovingAvg_1/subSub>batch_normalization_9/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_9/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg_1/515016*
_output_shapes
: 2-
+batch_normalization_9/AssignMovingAvg_1/subà
+batch_normalization_9/AssignMovingAvg_1/mulMul/batch_normalization_9/AssignMovingAvg_1/sub:z:06batch_normalization_9/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg_1/515016*
_output_shapes
: 2-
+batch_normalization_9/AssignMovingAvg_1/mul¿
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_9_assignmovingavg_1_515016/batch_normalization_9/AssignMovingAvg_1/mul:z:07^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg_1/515016*
_output_shapes
 *
dtype02=
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_9/batchnorm/add/yÚ
#batch_normalization_9/batchnorm/addAddV20batch_normalization_9/moments/Squeeze_1:output:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_9/batchnorm/add¥
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_9/batchnorm/Rsqrtà
2batch_normalization_9/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_9_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_9/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:0:batch_normalization_9/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_9/batchnorm/mulÛ
%batch_normalization_9/batchnorm/mul_1Mul%average_pooling1d_14/Squeeze:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_9/batchnorm/mul_1Ó
%batch_normalization_9/batchnorm/mul_2Mul.batch_normalization_9/moments/Squeeze:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_9/batchnorm/mul_2Ô
.batch_normalization_9/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_9_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_9/batchnorm/ReadVariableOpÙ
#batch_normalization_9/batchnorm/subSub6batch_normalization_9/batchnorm/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_9/batchnorm/subá
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_9/batchnorm/add_1«
	add_4/addAddV2)batch_normalization_8/batchnorm/add_1:z:0)batch_normalization_9/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
	add_4/add¹
Mtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_9_multi_head_attention_9_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpÐ
>transformer_block_9/multi_head_attention_9/query/einsum/EinsumEinsumadd_4/add:z:0Utransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2@
>transformer_block_9/multi_head_attention_9/query/einsum/Einsum
Ctransformer_block_9/multi_head_attention_9/query/add/ReadVariableOpReadVariableOpLtransformer_block_9_multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_9/multi_head_attention_9/query/add/ReadVariableOpÅ
4transformer_block_9/multi_head_attention_9/query/addAddV2Gtransformer_block_9/multi_head_attention_9/query/einsum/Einsum:output:0Ktransformer_block_9/multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_9/multi_head_attention_9/query/add³
Ktransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_9_multi_head_attention_9_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpÊ
<transformer_block_9/multi_head_attention_9/key/einsum/EinsumEinsumadd_4/add:z:0Stransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2>
<transformer_block_9/multi_head_attention_9/key/einsum/Einsum
Atransformer_block_9/multi_head_attention_9/key/add/ReadVariableOpReadVariableOpJtransformer_block_9_multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_9/multi_head_attention_9/key/add/ReadVariableOp½
2transformer_block_9/multi_head_attention_9/key/addAddV2Etransformer_block_9/multi_head_attention_9/key/einsum/Einsum:output:0Itransformer_block_9/multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 24
2transformer_block_9/multi_head_attention_9/key/add¹
Mtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_9_multi_head_attention_9_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpÐ
>transformer_block_9/multi_head_attention_9/value/einsum/EinsumEinsumadd_4/add:z:0Utransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2@
>transformer_block_9/multi_head_attention_9/value/einsum/Einsum
Ctransformer_block_9/multi_head_attention_9/value/add/ReadVariableOpReadVariableOpLtransformer_block_9_multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_9/multi_head_attention_9/value/add/ReadVariableOpÅ
4transformer_block_9/multi_head_attention_9/value/addAddV2Gtransformer_block_9/multi_head_attention_9/value/einsum/Einsum:output:0Ktransformer_block_9/multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_9/multi_head_attention_9/value/add©
0transformer_block_9/multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>22
0transformer_block_9/multi_head_attention_9/Mul/y
.transformer_block_9/multi_head_attention_9/MulMul8transformer_block_9/multi_head_attention_9/query/add:z:09transformer_block_9/multi_head_attention_9/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 20
.transformer_block_9/multi_head_attention_9/MulÌ
8transformer_block_9/multi_head_attention_9/einsum/EinsumEinsum6transformer_block_9/multi_head_attention_9/key/add:z:02transformer_block_9/multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2:
8transformer_block_9/multi_head_attention_9/einsum/Einsum
:transformer_block_9/multi_head_attention_9/softmax/SoftmaxSoftmaxAtransformer_block_9/multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2<
:transformer_block_9/multi_head_attention_9/softmax/SoftmaxÉ
@transformer_block_9/multi_head_attention_9/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2B
@transformer_block_9/multi_head_attention_9/dropout/dropout/ConstÒ
>transformer_block_9/multi_head_attention_9/dropout/dropout/MulMulDtransformer_block_9/multi_head_attention_9/softmax/Softmax:softmax:0Itransformer_block_9/multi_head_attention_9/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2@
>transformer_block_9/multi_head_attention_9/dropout/dropout/Mulø
@transformer_block_9/multi_head_attention_9/dropout/dropout/ShapeShapeDtransformer_block_9/multi_head_attention_9/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2B
@transformer_block_9/multi_head_attention_9/dropout/dropout/ShapeÕ
Wtransformer_block_9/multi_head_attention_9/dropout/dropout/random_uniform/RandomUniformRandomUniformItransformer_block_9/multi_head_attention_9/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
dtype02Y
Wtransformer_block_9/multi_head_attention_9/dropout/dropout/random_uniform/RandomUniformÛ
Itransformer_block_9/multi_head_attention_9/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2K
Itransformer_block_9/multi_head_attention_9/dropout/dropout/GreaterEqual/y
Gtransformer_block_9/multi_head_attention_9/dropout/dropout/GreaterEqualGreaterEqual`transformer_block_9/multi_head_attention_9/dropout/dropout/random_uniform/RandomUniform:output:0Rtransformer_block_9/multi_head_attention_9/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2I
Gtransformer_block_9/multi_head_attention_9/dropout/dropout/GreaterEqual 
?transformer_block_9/multi_head_attention_9/dropout/dropout/CastCastKtransformer_block_9/multi_head_attention_9/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2A
?transformer_block_9/multi_head_attention_9/dropout/dropout/CastÎ
@transformer_block_9/multi_head_attention_9/dropout/dropout/Mul_1MulBtransformer_block_9/multi_head_attention_9/dropout/dropout/Mul:z:0Ctransformer_block_9/multi_head_attention_9/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2B
@transformer_block_9/multi_head_attention_9/dropout/dropout/Mul_1ä
:transformer_block_9/multi_head_attention_9/einsum_1/EinsumEinsumDtransformer_block_9/multi_head_attention_9/dropout/dropout/Mul_1:z:08transformer_block_9/multi_head_attention_9/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2<
:transformer_block_9/multi_head_attention_9/einsum_1/EinsumÚ
Xtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_9_multi_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp£
Itransformer_block_9/multi_head_attention_9/attention_output/einsum/EinsumEinsumCtransformer_block_9/multi_head_attention_9/einsum_1/Einsum:output:0`transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe2K
Itransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum´
Ntransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_9_multi_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpí
?transformer_block_9/multi_head_attention_9/attention_output/addAddV2Rtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum:output:0Vtransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?transformer_block_9/multi_head_attention_9/attention_output/add¡
,transformer_block_9/dropout_26/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2.
,transformer_block_9/dropout_26/dropout/Const
*transformer_block_9/dropout_26/dropout/MulMulCtransformer_block_9/multi_head_attention_9/attention_output/add:z:05transformer_block_9/dropout_26/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2,
*transformer_block_9/dropout_26/dropout/MulÏ
,transformer_block_9/dropout_26/dropout/ShapeShapeCtransformer_block_9/multi_head_attention_9/attention_output/add:z:0*
T0*
_output_shapes
:2.
,transformer_block_9/dropout_26/dropout/Shape
Ctransformer_block_9/dropout_26/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_9/dropout_26/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype02E
Ctransformer_block_9/dropout_26/dropout/random_uniform/RandomUniform³
5transformer_block_9/dropout_26/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=27
5transformer_block_9/dropout_26/dropout/GreaterEqual/y¾
3transformer_block_9/dropout_26/dropout/GreaterEqualGreaterEqualLtransformer_block_9/dropout_26/dropout/random_uniform/RandomUniform:output:0>transformer_block_9/dropout_26/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 25
3transformer_block_9/dropout_26/dropout/GreaterEqualà
+transformer_block_9/dropout_26/dropout/CastCast7transformer_block_9/dropout_26/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+transformer_block_9/dropout_26/dropout/Castú
,transformer_block_9/dropout_26/dropout/Mul_1Mul.transformer_block_9/dropout_26/dropout/Mul:z:0/transformer_block_9/dropout_26/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2.
,transformer_block_9/dropout_26/dropout/Mul_1²
transformer_block_9/addAddV2add_4/add:z:00transformer_block_9/dropout_26/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_9/addà
Itransformer_block_9/layer_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_9/layer_normalization_18/moments/mean/reduction_indices²
7transformer_block_9/layer_normalization_18/moments/meanMeantransformer_block_9/add:z:0Rtransformer_block_9/layer_normalization_18/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(29
7transformer_block_9/layer_normalization_18/moments/mean
?transformer_block_9/layer_normalization_18/moments/StopGradientStopGradient@transformer_block_9/layer_normalization_18/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2A
?transformer_block_9/layer_normalization_18/moments/StopGradient¾
Dtransformer_block_9/layer_normalization_18/moments/SquaredDifferenceSquaredDifferencetransformer_block_9/add:z:0Htransformer_block_9/layer_normalization_18/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2F
Dtransformer_block_9/layer_normalization_18/moments/SquaredDifferenceè
Mtransformer_block_9/layer_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_9/layer_normalization_18/moments/variance/reduction_indicesë
;transformer_block_9/layer_normalization_18/moments/varianceMeanHtransformer_block_9/layer_normalization_18/moments/SquaredDifference:z:0Vtransformer_block_9/layer_normalization_18/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2=
;transformer_block_9/layer_normalization_18/moments/variance½
:transformer_block_9/layer_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_9/layer_normalization_18/batchnorm/add/y¾
8transformer_block_9/layer_normalization_18/batchnorm/addAddV2Dtransformer_block_9/layer_normalization_18/moments/variance:output:0Ctransformer_block_9/layer_normalization_18/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2:
8transformer_block_9/layer_normalization_18/batchnorm/addõ
:transformer_block_9/layer_normalization_18/batchnorm/RsqrtRsqrt<transformer_block_9/layer_normalization_18/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2<
:transformer_block_9/layer_normalization_18/batchnorm/Rsqrt
Gtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_9_layer_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpÂ
8transformer_block_9/layer_normalization_18/batchnorm/mulMul>transformer_block_9/layer_normalization_18/batchnorm/Rsqrt:y:0Otransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2:
8transformer_block_9/layer_normalization_18/batchnorm/mul
:transformer_block_9/layer_normalization_18/batchnorm/mul_1Multransformer_block_9/add:z:0<transformer_block_9/layer_normalization_18/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_9/layer_normalization_18/batchnorm/mul_1µ
:transformer_block_9/layer_normalization_18/batchnorm/mul_2Mul@transformer_block_9/layer_normalization_18/moments/mean:output:0<transformer_block_9/layer_normalization_18/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_9/layer_normalization_18/batchnorm/mul_2
Ctransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_9_layer_normalization_18_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOp¾
8transformer_block_9/layer_normalization_18/batchnorm/subSubKtransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOp:value:0>transformer_block_9/layer_normalization_18/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2:
8transformer_block_9/layer_normalization_18/batchnorm/subµ
:transformer_block_9/layer_normalization_18/batchnorm/add_1AddV2>transformer_block_9/layer_normalization_18/batchnorm/mul_1:z:0<transformer_block_9/layer_normalization_18/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_9/layer_normalization_18/batchnorm/add_1
Btransformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_9_sequential_9_dense_30_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02D
Btransformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOp¾
8transformer_block_9/sequential_9/dense_30/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_9/sequential_9/dense_30/Tensordot/axesÅ
8transformer_block_9/sequential_9/dense_30/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_9/sequential_9/dense_30/Tensordot/freeä
9transformer_block_9/sequential_9/dense_30/Tensordot/ShapeShape>transformer_block_9/layer_normalization_18/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_9/sequential_9/dense_30/Tensordot/ShapeÈ
Atransformer_block_9/sequential_9/dense_30/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_9/sequential_9/dense_30/Tensordot/GatherV2/axis£
<transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2GatherV2Btransformer_block_9/sequential_9/dense_30/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_30/Tensordot/free:output:0Jtransformer_block_9/sequential_9/dense_30/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2Ì
Ctransformer_block_9/sequential_9/dense_30/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_9/sequential_9/dense_30/Tensordot/GatherV2_1/axis©
>transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2_1GatherV2Btransformer_block_9/sequential_9/dense_30/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_30/Tensordot/axes:output:0Ltransformer_block_9/sequential_9/dense_30/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_9/sequential_9/dense_30/Tensordot/GatherV2_1À
9transformer_block_9/sequential_9/dense_30/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_9/sequential_9/dense_30/Tensordot/Const¨
8transformer_block_9/sequential_9/dense_30/Tensordot/ProdProdEtransformer_block_9/sequential_9/dense_30/Tensordot/GatherV2:output:0Btransformer_block_9/sequential_9/dense_30/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_9/sequential_9/dense_30/Tensordot/ProdÄ
;transformer_block_9/sequential_9/dense_30/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_9/sequential_9/dense_30/Tensordot/Const_1°
:transformer_block_9/sequential_9/dense_30/Tensordot/Prod_1ProdGtransformer_block_9/sequential_9/dense_30/Tensordot/GatherV2_1:output:0Dtransformer_block_9/sequential_9/dense_30/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_9/sequential_9/dense_30/Tensordot/Prod_1Ä
?transformer_block_9/sequential_9/dense_30/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_9/sequential_9/dense_30/Tensordot/concat/axis
:transformer_block_9/sequential_9/dense_30/Tensordot/concatConcatV2Atransformer_block_9/sequential_9/dense_30/Tensordot/free:output:0Atransformer_block_9/sequential_9/dense_30/Tensordot/axes:output:0Htransformer_block_9/sequential_9/dense_30/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_9/sequential_9/dense_30/Tensordot/concat´
9transformer_block_9/sequential_9/dense_30/Tensordot/stackPackAtransformer_block_9/sequential_9/dense_30/Tensordot/Prod:output:0Ctransformer_block_9/sequential_9/dense_30/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_9/sequential_9/dense_30/Tensordot/stackÆ
=transformer_block_9/sequential_9/dense_30/Tensordot/transpose	Transpose>transformer_block_9/layer_normalization_18/batchnorm/add_1:z:0Ctransformer_block_9/sequential_9/dense_30/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2?
=transformer_block_9/sequential_9/dense_30/Tensordot/transposeÇ
;transformer_block_9/sequential_9/dense_30/Tensordot/ReshapeReshapeAtransformer_block_9/sequential_9/dense_30/Tensordot/transpose:y:0Btransformer_block_9/sequential_9/dense_30/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2=
;transformer_block_9/sequential_9/dense_30/Tensordot/ReshapeÆ
:transformer_block_9/sequential_9/dense_30/Tensordot/MatMulMatMulDtransformer_block_9/sequential_9/dense_30/Tensordot/Reshape:output:0Jtransformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2<
:transformer_block_9/sequential_9/dense_30/Tensordot/MatMulÄ
;transformer_block_9/sequential_9/dense_30/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2=
;transformer_block_9/sequential_9/dense_30/Tensordot/Const_2È
Atransformer_block_9/sequential_9/dense_30/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_9/sequential_9/dense_30/Tensordot/concat_1/axis
<transformer_block_9/sequential_9/dense_30/Tensordot/concat_1ConcatV2Etransformer_block_9/sequential_9/dense_30/Tensordot/GatherV2:output:0Dtransformer_block_9/sequential_9/dense_30/Tensordot/Const_2:output:0Jtransformer_block_9/sequential_9/dense_30/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_9/sequential_9/dense_30/Tensordot/concat_1¸
3transformer_block_9/sequential_9/dense_30/TensordotReshapeDtransformer_block_9/sequential_9/dense_30/Tensordot/MatMul:product:0Etransformer_block_9/sequential_9/dense_30/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@25
3transformer_block_9/sequential_9/dense_30/Tensordot
@transformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_9_sequential_9_dense_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@transformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOp¯
1transformer_block_9/sequential_9/dense_30/BiasAddBiasAdd<transformer_block_9/sequential_9/dense_30/Tensordot:output:0Htransformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@23
1transformer_block_9/sequential_9/dense_30/BiasAddÚ
.transformer_block_9/sequential_9/dense_30/ReluRelu:transformer_block_9/sequential_9/dense_30/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@20
.transformer_block_9/sequential_9/dense_30/Relu
Btransformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_9_sequential_9_dense_31_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02D
Btransformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOp¾
8transformer_block_9/sequential_9/dense_31/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_9/sequential_9/dense_31/Tensordot/axesÅ
8transformer_block_9/sequential_9/dense_31/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_9/sequential_9/dense_31/Tensordot/freeâ
9transformer_block_9/sequential_9/dense_31/Tensordot/ShapeShape<transformer_block_9/sequential_9/dense_30/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_9/sequential_9/dense_31/Tensordot/ShapeÈ
Atransformer_block_9/sequential_9/dense_31/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_9/sequential_9/dense_31/Tensordot/GatherV2/axis£
<transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2GatherV2Btransformer_block_9/sequential_9/dense_31/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_31/Tensordot/free:output:0Jtransformer_block_9/sequential_9/dense_31/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2Ì
Ctransformer_block_9/sequential_9/dense_31/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_9/sequential_9/dense_31/Tensordot/GatherV2_1/axis©
>transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2_1GatherV2Btransformer_block_9/sequential_9/dense_31/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_31/Tensordot/axes:output:0Ltransformer_block_9/sequential_9/dense_31/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_9/sequential_9/dense_31/Tensordot/GatherV2_1À
9transformer_block_9/sequential_9/dense_31/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_9/sequential_9/dense_31/Tensordot/Const¨
8transformer_block_9/sequential_9/dense_31/Tensordot/ProdProdEtransformer_block_9/sequential_9/dense_31/Tensordot/GatherV2:output:0Btransformer_block_9/sequential_9/dense_31/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_9/sequential_9/dense_31/Tensordot/ProdÄ
;transformer_block_9/sequential_9/dense_31/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_9/sequential_9/dense_31/Tensordot/Const_1°
:transformer_block_9/sequential_9/dense_31/Tensordot/Prod_1ProdGtransformer_block_9/sequential_9/dense_31/Tensordot/GatherV2_1:output:0Dtransformer_block_9/sequential_9/dense_31/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_9/sequential_9/dense_31/Tensordot/Prod_1Ä
?transformer_block_9/sequential_9/dense_31/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_9/sequential_9/dense_31/Tensordot/concat/axis
:transformer_block_9/sequential_9/dense_31/Tensordot/concatConcatV2Atransformer_block_9/sequential_9/dense_31/Tensordot/free:output:0Atransformer_block_9/sequential_9/dense_31/Tensordot/axes:output:0Htransformer_block_9/sequential_9/dense_31/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_9/sequential_9/dense_31/Tensordot/concat´
9transformer_block_9/sequential_9/dense_31/Tensordot/stackPackAtransformer_block_9/sequential_9/dense_31/Tensordot/Prod:output:0Ctransformer_block_9/sequential_9/dense_31/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_9/sequential_9/dense_31/Tensordot/stackÄ
=transformer_block_9/sequential_9/dense_31/Tensordot/transpose	Transpose<transformer_block_9/sequential_9/dense_30/Relu:activations:0Ctransformer_block_9/sequential_9/dense_31/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2?
=transformer_block_9/sequential_9/dense_31/Tensordot/transposeÇ
;transformer_block_9/sequential_9/dense_31/Tensordot/ReshapeReshapeAtransformer_block_9/sequential_9/dense_31/Tensordot/transpose:y:0Btransformer_block_9/sequential_9/dense_31/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2=
;transformer_block_9/sequential_9/dense_31/Tensordot/ReshapeÆ
:transformer_block_9/sequential_9/dense_31/Tensordot/MatMulMatMulDtransformer_block_9/sequential_9/dense_31/Tensordot/Reshape:output:0Jtransformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2<
:transformer_block_9/sequential_9/dense_31/Tensordot/MatMulÄ
;transformer_block_9/sequential_9/dense_31/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_9/sequential_9/dense_31/Tensordot/Const_2È
Atransformer_block_9/sequential_9/dense_31/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_9/sequential_9/dense_31/Tensordot/concat_1/axis
<transformer_block_9/sequential_9/dense_31/Tensordot/concat_1ConcatV2Etransformer_block_9/sequential_9/dense_31/Tensordot/GatherV2:output:0Dtransformer_block_9/sequential_9/dense_31/Tensordot/Const_2:output:0Jtransformer_block_9/sequential_9/dense_31/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_9/sequential_9/dense_31/Tensordot/concat_1¸
3transformer_block_9/sequential_9/dense_31/TensordotReshapeDtransformer_block_9/sequential_9/dense_31/Tensordot/MatMul:product:0Etransformer_block_9/sequential_9/dense_31/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 25
3transformer_block_9/sequential_9/dense_31/Tensordot
@transformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_9_sequential_9_dense_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOp¯
1transformer_block_9/sequential_9/dense_31/BiasAddBiasAdd<transformer_block_9/sequential_9/dense_31/Tensordot:output:0Htransformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 23
1transformer_block_9/sequential_9/dense_31/BiasAdd¡
,transformer_block_9/dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2.
,transformer_block_9/dropout_27/dropout/Const
*transformer_block_9/dropout_27/dropout/MulMul:transformer_block_9/sequential_9/dense_31/BiasAdd:output:05transformer_block_9/dropout_27/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2,
*transformer_block_9/dropout_27/dropout/MulÆ
,transformer_block_9/dropout_27/dropout/ShapeShape:transformer_block_9/sequential_9/dense_31/BiasAdd:output:0*
T0*
_output_shapes
:2.
,transformer_block_9/dropout_27/dropout/Shape
Ctransformer_block_9/dropout_27/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_9/dropout_27/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype02E
Ctransformer_block_9/dropout_27/dropout/random_uniform/RandomUniform³
5transformer_block_9/dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=27
5transformer_block_9/dropout_27/dropout/GreaterEqual/y¾
3transformer_block_9/dropout_27/dropout/GreaterEqualGreaterEqualLtransformer_block_9/dropout_27/dropout/random_uniform/RandomUniform:output:0>transformer_block_9/dropout_27/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 25
3transformer_block_9/dropout_27/dropout/GreaterEqualà
+transformer_block_9/dropout_27/dropout/CastCast7transformer_block_9/dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+transformer_block_9/dropout_27/dropout/Castú
,transformer_block_9/dropout_27/dropout/Mul_1Mul.transformer_block_9/dropout_27/dropout/Mul:z:0/transformer_block_9/dropout_27/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2.
,transformer_block_9/dropout_27/dropout/Mul_1ç
transformer_block_9/add_1AddV2>transformer_block_9/layer_normalization_18/batchnorm/add_1:z:00transformer_block_9/dropout_27/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_9/add_1à
Itransformer_block_9/layer_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_9/layer_normalization_19/moments/mean/reduction_indices´
7transformer_block_9/layer_normalization_19/moments/meanMeantransformer_block_9/add_1:z:0Rtransformer_block_9/layer_normalization_19/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(29
7transformer_block_9/layer_normalization_19/moments/mean
?transformer_block_9/layer_normalization_19/moments/StopGradientStopGradient@transformer_block_9/layer_normalization_19/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2A
?transformer_block_9/layer_normalization_19/moments/StopGradientÀ
Dtransformer_block_9/layer_normalization_19/moments/SquaredDifferenceSquaredDifferencetransformer_block_9/add_1:z:0Htransformer_block_9/layer_normalization_19/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2F
Dtransformer_block_9/layer_normalization_19/moments/SquaredDifferenceè
Mtransformer_block_9/layer_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_9/layer_normalization_19/moments/variance/reduction_indicesë
;transformer_block_9/layer_normalization_19/moments/varianceMeanHtransformer_block_9/layer_normalization_19/moments/SquaredDifference:z:0Vtransformer_block_9/layer_normalization_19/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2=
;transformer_block_9/layer_normalization_19/moments/variance½
:transformer_block_9/layer_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_9/layer_normalization_19/batchnorm/add/y¾
8transformer_block_9/layer_normalization_19/batchnorm/addAddV2Dtransformer_block_9/layer_normalization_19/moments/variance:output:0Ctransformer_block_9/layer_normalization_19/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2:
8transformer_block_9/layer_normalization_19/batchnorm/addõ
:transformer_block_9/layer_normalization_19/batchnorm/RsqrtRsqrt<transformer_block_9/layer_normalization_19/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2<
:transformer_block_9/layer_normalization_19/batchnorm/Rsqrt
Gtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_9_layer_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpÂ
8transformer_block_9/layer_normalization_19/batchnorm/mulMul>transformer_block_9/layer_normalization_19/batchnorm/Rsqrt:y:0Otransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2:
8transformer_block_9/layer_normalization_19/batchnorm/mul
:transformer_block_9/layer_normalization_19/batchnorm/mul_1Multransformer_block_9/add_1:z:0<transformer_block_9/layer_normalization_19/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_9/layer_normalization_19/batchnorm/mul_1µ
:transformer_block_9/layer_normalization_19/batchnorm/mul_2Mul@transformer_block_9/layer_normalization_19/moments/mean:output:0<transformer_block_9/layer_normalization_19/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_9/layer_normalization_19/batchnorm/mul_2
Ctransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_9_layer_normalization_19_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOp¾
8transformer_block_9/layer_normalization_19/batchnorm/subSubKtransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOp:value:0>transformer_block_9/layer_normalization_19/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2:
8transformer_block_9/layer_normalization_19/batchnorm/subµ
:transformer_block_9/layer_normalization_19/batchnorm/add_1AddV2>transformer_block_9/layer_normalization_19/batchnorm/mul_1:z:0<transformer_block_9/layer_normalization_19/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:transformer_block_9/layer_normalization_19/batchnorm/add_1s
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`  2
flatten_4/Const¾
flatten_4/ReshapeReshape>transformer_block_9/layer_normalization_19/batchnorm/add_1:z:0flatten_4/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
flatten_4/Reshapex
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_4/concat/axis¾
concatenate_4/concatConcatV2flatten_4/Reshape:output:0inputs_1"concatenate_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
concatenate_4/concat©
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	è@*
dtype02 
dense_32/MatMul/ReadVariableOp¥
dense_32/MatMulMatMulconcatenate_4/concat:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_32/MatMul§
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_32/BiasAdd/ReadVariableOp¥
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_32/BiasAdds
dense_32/ReluReludense_32/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_32/Reluy
dropout_28/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_28/dropout/Const©
dropout_28/dropout/MulMuldense_32/Relu:activations:0!dropout_28/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_28/dropout/Mul
dropout_28/dropout/ShapeShapedense_32/Relu:activations:0*
T0*
_output_shapes
:2
dropout_28/dropout/ShapeÕ
/dropout_28/dropout/random_uniform/RandomUniformRandomUniform!dropout_28/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype021
/dropout_28/dropout/random_uniform/RandomUniform
!dropout_28/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_28/dropout/GreaterEqual/yê
dropout_28/dropout/GreaterEqualGreaterEqual8dropout_28/dropout/random_uniform/RandomUniform:output:0*dropout_28/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
dropout_28/dropout/GreaterEqual 
dropout_28/dropout/CastCast#dropout_28/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_28/dropout/Cast¦
dropout_28/dropout/Mul_1Muldropout_28/dropout/Mul:z:0dropout_28/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_28/dropout/Mul_1¨
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_33/MatMul/ReadVariableOp¤
dense_33/MatMulMatMuldropout_28/dropout/Mul_1:z:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_33/MatMul§
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_33/BiasAdd/ReadVariableOp¥
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_33/BiasAdds
dense_33/ReluReludense_33/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_33/Reluy
dropout_29/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_29/dropout/Const©
dropout_29/dropout/MulMuldense_33/Relu:activations:0!dropout_29/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_29/dropout/Mul
dropout_29/dropout/ShapeShapedense_33/Relu:activations:0*
T0*
_output_shapes
:2
dropout_29/dropout/ShapeÕ
/dropout_29/dropout/random_uniform/RandomUniformRandomUniform!dropout_29/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype021
/dropout_29/dropout/random_uniform/RandomUniform
!dropout_29/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_29/dropout/GreaterEqual/yê
dropout_29/dropout/GreaterEqualGreaterEqual8dropout_29/dropout/random_uniform/RandomUniform:output:0*dropout_29/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
dropout_29/dropout/GreaterEqual 
dropout_29/dropout/CastCast#dropout_29/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_29/dropout/Cast¦
dropout_29/dropout/Mul_1Muldropout_29/dropout/Mul:z:0dropout_29/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_29/dropout/Mul_1¨
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_34/MatMul/ReadVariableOp¤
dense_34/MatMulMatMuldropout_29/dropout/Mul_1:z:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_34/MatMul§
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_34/BiasAdd/ReadVariableOp¥
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_34/BiasAdd
IdentityIdentitydense_34/BiasAdd:output:0:^batch_normalization_8/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_8/AssignMovingAvg/ReadVariableOp<^batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_8/batchnorm/ReadVariableOp3^batch_normalization_8/batchnorm/mul/ReadVariableOp:^batch_normalization_9/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_9/AssignMovingAvg/ReadVariableOp<^batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_9/batchnorm/ReadVariableOp3^batch_normalization_9/batchnorm/mul/ReadVariableOp ^conv1d_8/BiasAdd/ReadVariableOp,^conv1d_8/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_9/BiasAdd/ReadVariableOp,^conv1d_9/conv1d/ExpandDims_1/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp<^token_and_position_embedding_4/embedding_8/embedding_lookup<^token_and_position_embedding_4/embedding_9/embedding_lookupD^transformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpH^transformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpD^transformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpH^transformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpO^transformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpY^transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_9/multi_head_attention_9/key/add/ReadVariableOpL^transformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpD^transformer_block_9/multi_head_attention_9/query/add/ReadVariableOpN^transformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpD^transformer_block_9/multi_head_attention_9/value/add/ReadVariableOpN^transformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpA^transformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOpC^transformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOpA^transformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOpC^transformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2v
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_8/AssignMovingAvg/ReadVariableOp4batch_normalization_8/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_8/batchnorm/ReadVariableOp.batch_normalization_8/batchnorm/ReadVariableOp2h
2batch_normalization_8/batchnorm/mul/ReadVariableOp2batch_normalization_8/batchnorm/mul/ReadVariableOp2v
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_9/AssignMovingAvg/ReadVariableOp4batch_normalization_9/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_9/batchnorm/ReadVariableOp.batch_normalization_9/batchnorm/ReadVariableOp2h
2batch_normalization_9/batchnorm/mul/ReadVariableOp2batch_normalization_9/batchnorm/mul/ReadVariableOp2B
conv1d_8/BiasAdd/ReadVariableOpconv1d_8/BiasAdd/ReadVariableOp2Z
+conv1d_8/conv1d/ExpandDims_1/ReadVariableOp+conv1d_8/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_9/BiasAdd/ReadVariableOpconv1d_9/BiasAdd/ReadVariableOp2Z
+conv1d_9/conv1d/ExpandDims_1/ReadVariableOp+conv1d_9/conv1d/ExpandDims_1/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2z
;token_and_position_embedding_4/embedding_8/embedding_lookup;token_and_position_embedding_4/embedding_8/embedding_lookup2z
;token_and_position_embedding_4/embedding_9/embedding_lookup;token_and_position_embedding_4/embedding_9/embedding_lookup2
Ctransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpCtransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOp2
Gtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpGtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOp2
Ctransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpCtransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOp2
Gtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpGtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOp2 
Ntransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpNtransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOp2´
Xtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2
Atransformer_block_9/multi_head_attention_9/key/add/ReadVariableOpAtransformer_block_9/multi_head_attention_9/key/add/ReadVariableOp2
Ktransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpKtransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2
Ctransformer_block_9/multi_head_attention_9/query/add/ReadVariableOpCtransformer_block_9/multi_head_attention_9/query/add/ReadVariableOp2
Mtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpMtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2
Ctransformer_block_9/multi_head_attention_9/value/add/ReadVariableOpCtransformer_block_9/multi_head_attention_9/value/add/ReadVariableOp2
Mtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpMtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp2
@transformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOp@transformer_block_9/sequential_9/dense_30/BiasAdd/ReadVariableOp2
Btransformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOpBtransformer_block_9/sequential_9/dense_30/Tensordot/ReadVariableOp2
@transformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOp@transformer_block_9/sequential_9/dense_31/BiasAdd/ReadVariableOp2
Btransformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOpBtransformer_block_9/sequential_9/dense_31/Tensordot/ReadVariableOp:R N
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
Ñ
ã
D__inference_dense_31_layer_call_and_return_conditional_losses_516738

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
Ê
©
6__inference_batch_normalization_8_layer_call_fn_515866

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
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_5136882
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
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ë
serving_default×
=
input_101
serving_default_input_10:0ÿÿÿÿÿÿÿÿÿ
<
input_91
serving_default_input_9:0ÿÿÿÿÿÿÿÿÿR<
dense_340
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:èü
ÚG
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
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
+±&call_and_return_all_conditional_losses
²_default_save_signature
³__call__"¢B
_tf_keras_networkB{"class_name": "Functional", "name": "model_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}, "name": "input_9", "inbound_nodes": []}, {"class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}, "name": "token_and_position_embedding_4", "inbound_nodes": [[["input_9", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_8", "inbound_nodes": [[["token_and_position_embedding_4", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_12", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_12", "inbound_nodes": [[["conv1d_8", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_9", "inbound_nodes": [[["average_pooling1d_12", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_13", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_13", "inbound_nodes": [[["conv1d_9", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_14", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_14", "inbound_nodes": [[["token_and_position_embedding_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["average_pooling1d_13", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_9", "inbound_nodes": [[["average_pooling1d_14", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "name": "add_4", "inbound_nodes": [[["batch_normalization_8", 0, 0, {}], ["batch_normalization_9", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block_9", "inbound_nodes": [[["add_4", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["transformer_block_9", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}, "name": "input_10", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_4", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_4", "inbound_nodes": [[["flatten_4", 0, 0, {}], ["input_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_32", "inbound_nodes": [[["concatenate_4", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_28", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_28", "inbound_nodes": [[["dense_32", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_33", "inbound_nodes": [[["dropout_28", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_29", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_29", "inbound_nodes": [[["dense_33", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_34", "inbound_nodes": [[["dropout_29", 0, 0, {}]]]}], "input_layers": [["input_9", 0, 0], ["input_10", 0, 0]], "output_layers": [["dense_34", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 10500]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 10500]}, {"class_name": "TensorShape", "items": [null, 8]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.00020000000949949026, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
ñ"î
_tf_keras_input_layerÎ{"class_name": "InputLayer", "name": "input_9", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}}
ç
	token_emb
pos_emb
	variables
trainable_variables
regularization_losses
	keras_api
+´&call_and_return_all_conditional_losses
µ__call__"º
_tf_keras_layer {"class_name": "TokenAndPositionEmbedding", "name": "token_and_position_embedding_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
é	

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
+¶&call_and_return_all_conditional_losses
·__call__"Â
_tf_keras_layer¨{"class_name": "Conv1D", "name": "conv1d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500, 32]}}

&	variables
'trainable_variables
(regularization_losses
)	keras_api
+¸&call_and_return_all_conditional_losses
¹__call__"ú
_tf_keras_layerà{"class_name": "AveragePooling1D", "name": "average_pooling1d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_12", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ç	

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
+º&call_and_return_all_conditional_losses
»__call__"À
_tf_keras_layer¦{"class_name": "Conv1D", "name": "conv1d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 350, 32]}}

0	variables
1trainable_variables
2regularization_losses
3	keras_api
+¼&call_and_return_all_conditional_losses
½__call__"ú
_tf_keras_layerà{"class_name": "AveragePooling1D", "name": "average_pooling1d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_13", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}

4	variables
5trainable_variables
6regularization_losses
7	keras_api
+¾&call_and_return_all_conditional_losses
¿__call__"ü
_tf_keras_layerâ{"class_name": "AveragePooling1D", "name": "average_pooling1d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_14", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
¸	
8axis
	9gamma
:beta
;moving_mean
<moving_variance
=	variables
>trainable_variables
?regularization_losses
@	keras_api
+À&call_and_return_all_conditional_losses
Á__call__"â
_tf_keras_layerÈ{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
¸	
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
+Â&call_and_return_all_conditional_losses
Ã__call__"â
_tf_keras_layerÈ{"class_name": "BatchNormalization", "name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
³
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
+Ä&call_and_return_all_conditional_losses
Å__call__"¢
_tf_keras_layer{"class_name": "Add", "name": "add_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 35, 32]}, {"class_name": "TensorShape", "items": [null, 35, 32]}]}

Natt
Offn
P
layernorm1
Q
layernorm2
Rdropout1
Sdropout2
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
+Æ&call_and_return_all_conditional_losses
Ç__call__"¥
_tf_keras_layer{"class_name": "TransformerBlock", "name": "transformer_block_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
è
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
+È&call_and_return_all_conditional_losses
É__call__"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ë"è
_tf_keras_input_layerÈ{"class_name": "InputLayer", "name": "input_10", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}}
Ð
\	variables
]trainable_variables
^regularization_losses
_	keras_api
+Ê&call_and_return_all_conditional_losses
Ë__call__"¿
_tf_keras_layer¥{"class_name": "Concatenate", "name": "concatenate_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_4", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1120]}, {"class_name": "TensorShape", "items": [null, 8]}]}
ø

`kernel
abias
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
+Ì&call_and_return_all_conditional_losses
Í__call__"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "dense_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1128]}}
é
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
+Î&call_and_return_all_conditional_losses
Ï__call__"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_28", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_28", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
ô

jkernel
kbias
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
+Ð&call_and_return_all_conditional_losses
Ñ__call__"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
é
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
+Ò&call_and_return_all_conditional_losses
Ó__call__"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_29", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_29", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
õ

tkernel
ubias
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
+Ô&call_and_return_all_conditional_losses
Õ__call__"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ù
	zdecay
{learning_rate
|momentum
}iter momentum!momentum*momentum+momentum9momentum:momentumBmomentumCmomentum`momentumamomentumjmomentumkmomentumtmomentumumomentum~momentummomentum momentum¡momentum¢momentum£momentum¤momentum¥momentum¦momentum§momentum¨momentum©momentumªmomentum«momentum¬momentum­momentum®momentum¯momentum°"
	optimizer
Æ
~0
1
 2
!3
*4
+5
96
:7
;8
<9
B10
C11
D12
E13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
`30
a31
j32
k33
t34
u35"
trackable_list_wrapper
¦
~0
1
 2
!3
*4
+5
96
:7
B8
C9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
`26
a27
j28
k29
t30
u31"
trackable_list_wrapper
 "
trackable_list_wrapper
Ó
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
	variables
trainable_variables
metrics
regularization_losses
³__call__
²_default_save_signature
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
-
Öserving_default"
signature_map
´
~
embeddings
	variables
trainable_variables
regularization_losses
	keras_api
+×&call_and_return_all_conditional_losses
Ø__call__"
_tf_keras_layerõ{"class_name": "Embedding", "name": "embedding_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500]}}
±

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
+Ù&call_and_return_all_conditional_losses
Ú__call__"
_tf_keras_layerò{"class_name": "Embedding", "name": "embedding_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 10500, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
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
µ
layer_metrics
non_trainable_variables
layers
  layer_regularization_losses
	variables
trainable_variables
¡metrics
regularization_losses
µ__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
%:#  2conv1d_8/kernel
: 2conv1d_8/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¢layer_metrics
£non_trainable_variables
¤layers
 ¥layer_regularization_losses
"	variables
#trainable_variables
¦metrics
$regularization_losses
·__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
§layer_metrics
¨non_trainable_variables
©layers
 ªlayer_regularization_losses
&	variables
'trainable_variables
«metrics
(regularization_losses
¹__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
%:#	  2conv1d_9/kernel
: 2conv1d_9/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¬layer_metrics
­non_trainable_variables
®layers
 ¯layer_regularization_losses
,	variables
-trainable_variables
°metrics
.regularization_losses
»__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
±layer_metrics
²non_trainable_variables
³layers
 ´layer_regularization_losses
0	variables
1trainable_variables
µmetrics
2regularization_losses
½__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¶layer_metrics
·non_trainable_variables
¸layers
 ¹layer_regularization_losses
4	variables
5trainable_variables
ºmetrics
6regularization_losses
¿__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_8/gamma
(:& 2batch_normalization_8/beta
1:/  (2!batch_normalization_8/moving_mean
5:3  (2%batch_normalization_8/moving_variance
<
90
:1
;2
<3"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
»layer_metrics
¼non_trainable_variables
½layers
 ¾layer_regularization_losses
=	variables
>trainable_variables
¿metrics
?regularization_losses
Á__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_9/gamma
(:& 2batch_normalization_9/beta
1:/  (2!batch_normalization_9/moving_mean
5:3  (2%batch_normalization_9/moving_variance
<
B0
C1
D2
E3"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Àlayer_metrics
Ánon_trainable_variables
Âlayers
 Ãlayer_regularization_losses
F	variables
Gtrainable_variables
Ämetrics
Hregularization_losses
Ã__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ålayer_metrics
Ænon_trainable_variables
Çlayers
 Èlayer_regularization_losses
J	variables
Ktrainable_variables
Émetrics
Lregularization_losses
Å__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object

Ê_query_dense
Ë
_key_dense
Ì_value_dense
Í_softmax
Î_dropout_layer
Ï_output_dense
Ð	variables
Ñtrainable_variables
Òregularization_losses
Ó	keras_api
+Û&call_and_return_all_conditional_losses
Ü__call__"
_tf_keras_layerê{"class_name": "MultiHeadAttention", "name": "multi_head_attention_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multi_head_attention_9", "trainable": true, "dtype": "float32", "num_heads": 1, "key_dim": 32, "value_dim": 32, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}
¯
Ôlayer_with_weights-0
Ôlayer-0
Õlayer_with_weights-1
Õlayer-1
Ö	variables
×trainable_variables
Øregularization_losses
Ù	keras_api
+Ý&call_and_return_all_conditional_losses
Þ__call__"È
_tf_keras_sequential©{"class_name": "Sequential", "name": "sequential_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_30_input"}}, {"class_name": "Dense", "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_30_input"}}, {"class_name": "Dense", "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
ì
	Úaxis

gamma
	beta
Û	variables
Ütrainable_variables
Ýregularization_losses
Þ	keras_api
+ß&call_and_return_all_conditional_losses
à__call__"µ
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_18", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
ì
	ßaxis

gamma
	beta
à	variables
átrainable_variables
âregularization_losses
ã	keras_api
+á&call_and_return_all_conditional_losses
â__call__"µ
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_19", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
í
ä	variables
åtrainable_variables
æregularization_losses
ç	keras_api
+ã&call_and_return_all_conditional_losses
ä__call__"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_26", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_26", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
í
è	variables
étrainable_variables
êregularization_losses
ë	keras_api
+å&call_and_return_all_conditional_losses
æ__call__"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_27", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_27", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
¦
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
¦
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ìlayer_metrics
ínon_trainable_variables
îlayers
 ïlayer_regularization_losses
T	variables
Utrainable_variables
ðmetrics
Vregularization_losses
Ç__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ñlayer_metrics
ònon_trainable_variables
ólayers
 ôlayer_regularization_losses
X	variables
Ytrainable_variables
õmetrics
Zregularization_losses
É__call__
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
ölayer_metrics
÷non_trainable_variables
ølayers
 ùlayer_regularization_losses
\	variables
]trainable_variables
úmetrics
^regularization_losses
Ë__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
": 	è@2dense_32/kernel
:@2dense_32/bias
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ûlayer_metrics
ünon_trainable_variables
ýlayers
 þlayer_regularization_losses
b	variables
ctrainable_variables
ÿmetrics
dregularization_losses
Í__call__
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
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
f	variables
gtrainable_variables
metrics
hregularization_losses
Ï__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_33/kernel
:@2dense_33/bias
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
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
l	variables
mtrainable_variables
metrics
nregularization_losses
Ñ__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
p	variables
qtrainable_variables
metrics
rregularization_losses
Ó__call__
+Ò&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_34/kernel
:2dense_34/bias
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
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
v	variables
wtrainable_variables
metrics
xregularization_losses
Õ__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
G:E 25token_and_position_embedding_4/embedding_8/embeddings
H:F	R 25token_and_position_embedding_4/embedding_9/embeddings
M:K  27transformer_block_9/multi_head_attention_9/query/kernel
G:E 25transformer_block_9/multi_head_attention_9/query/bias
K:I  25transformer_block_9/multi_head_attention_9/key/kernel
E:C 23transformer_block_9/multi_head_attention_9/key/bias
M:K  27transformer_block_9/multi_head_attention_9/value/kernel
G:E 25transformer_block_9/multi_head_attention_9/value/bias
X:V  2Btransformer_block_9/multi_head_attention_9/attention_output/kernel
N:L 2@transformer_block_9/multi_head_attention_9/attention_output/bias
!: @2dense_30/kernel
:@2dense_30/bias
!:@ 2dense_31/kernel
: 2dense_31/bias
>:< 20transformer_block_9/layer_normalization_18/gamma
=:; 2/transformer_block_9/layer_normalization_18/beta
>:< 20transformer_block_9/layer_normalization_19/gamma
=:; 2/transformer_block_9/layer_normalization_19/beta
 "
trackable_dict_wrapper
<
;0
<1
D2
E3"
trackable_list_wrapper
®
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
18"
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
'
~0"
trackable_list_wrapper
'
~0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
	variables
trainable_variables
metrics
regularization_losses
Ø__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
	variables
trainable_variables
metrics
regularization_losses
Ú__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
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
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
D0
E1"
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
Ë
partial_output_shape
 full_output_shape
kernel
	bias
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
+ç&call_and_return_all_conditional_losses
è__call__"ë
_tf_keras_layerÑ{"class_name": "EinsumDense", "name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
Ç
¥partial_output_shape
¦full_output_shape
kernel
	bias
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
+é&call_and_return_all_conditional_losses
ê__call__"ç
_tf_keras_layerÍ{"class_name": "EinsumDense", "name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
Ë
«partial_output_shape
¬full_output_shape
kernel
	bias
­	variables
®trainable_variables
¯regularization_losses
°	keras_api
+ë&call_and_return_all_conditional_losses
ì__call__"ë
_tf_keras_layerÑ{"class_name": "EinsumDense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
ë
±	variables
²trainable_variables
³regularization_losses
´	keras_api
+í&call_and_return_all_conditional_losses
î__call__"Ö
_tf_keras_layer¼{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}}
ç
µ	variables
¶trainable_variables
·regularization_losses
¸	keras_api
+ï&call_and_return_all_conditional_losses
ð__call__"Ò
_tf_keras_layer¸{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
à
¹partial_output_shape
ºfull_output_shape
kernel
	bias
»	variables
¼trainable_variables
½regularization_losses
¾	keras_api
+ñ&call_and_return_all_conditional_losses
ò__call__"
_tf_keras_layeræ{"class_name": "EinsumDense", "name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 32], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 1, 32]}}
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¿layer_metrics
Ànon_trainable_variables
Álayers
 Âlayer_regularization_losses
Ð	variables
Ñtrainable_variables
Ãmetrics
Òregularization_losses
Ü__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
þ
kernel
	bias
Ä	variables
Åtrainable_variables
Æregularization_losses
Ç	keras_api
+ó&call_and_return_all_conditional_losses
ô__call__"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "dense_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}

kernel
	bias
È	variables
Étrainable_variables
Êregularization_losses
Ë	keras_api
+õ&call_and_return_all_conditional_losses
ö__call__"Ó
_tf_keras_layer¹{"class_name": "Dense", "name": "dense_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 64]}}
@
0
1
2
3"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ìlayer_metrics
Ínon_trainable_variables
Îlayers
 Ïlayer_regularization_losses
Ö	variables
×trainable_variables
Ðmetrics
Øregularization_losses
Þ__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses"
_generic_user_object
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
Ñlayer_metrics
Ònon_trainable_variables
Ólayers
 Ôlayer_regularization_losses
Û	variables
Ütrainable_variables
Õmetrics
Ýregularization_losses
à__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses"
_generic_user_object
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
Ölayer_metrics
×non_trainable_variables
Ølayers
 Ùlayer_regularization_losses
à	variables
átrainable_variables
Úmetrics
âregularization_losses
â__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ûlayer_metrics
Ünon_trainable_variables
Ýlayers
 Þlayer_regularization_losses
ä	variables
åtrainable_variables
ßmetrics
æregularization_losses
ä__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
àlayer_metrics
ánon_trainable_variables
âlayers
 ãlayer_regularization_losses
è	variables
étrainable_variables
ämetrics
êregularization_losses
æ__call__
+å&call_and_return_all_conditional_losses
'å"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
N0
O1
P2
Q3
R4
S5"
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
¿

åtotal

æcount
ç	variables
è	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
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
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
élayer_metrics
ênon_trainable_variables
ëlayers
 ìlayer_regularization_losses
¡	variables
¢trainable_variables
ímetrics
£regularization_losses
è__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
îlayer_metrics
ïnon_trainable_variables
ðlayers
 ñlayer_regularization_losses
§	variables
¨trainable_variables
òmetrics
©regularization_losses
ê__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ólayer_metrics
ônon_trainable_variables
õlayers
 ölayer_regularization_losses
­	variables
®trainable_variables
÷metrics
¯regularization_losses
ì__call__
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ølayer_metrics
ùnon_trainable_variables
úlayers
 ûlayer_regularization_losses
±	variables
²trainable_variables
ümetrics
³regularization_losses
î__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ýlayer_metrics
þnon_trainable_variables
ÿlayers
 layer_regularization_losses
µ	variables
¶trainable_variables
metrics
·regularization_losses
ð__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
»	variables
¼trainable_variables
metrics
½regularization_losses
ò__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
P
Ê0
Ë1
Ì2
Í3
Î4
Ï5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
¸
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
Ä	variables
Åtrainable_variables
metrics
Æregularization_losses
ô__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
_generic_user_object
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
layer_metrics
non_trainable_variables
layers
 layer_regularization_losses
È	variables
Étrainable_variables
metrics
Êregularization_losses
ö__call__
+õ&call_and_return_all_conditional_losses
'õ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Ô0
Õ1"
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
:  (2total
:  (2count
0
å0
æ1"
trackable_list_wrapper
.
ç	variables"
_generic_user_object
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
0:.  2SGD/conv1d_8/kernel/momentum
&:$ 2SGD/conv1d_8/bias/momentum
0:.	  2SGD/conv1d_9/kernel/momentum
&:$ 2SGD/conv1d_9/bias/momentum
4:2 2(SGD/batch_normalization_8/gamma/momentum
3:1 2'SGD/batch_normalization_8/beta/momentum
4:2 2(SGD/batch_normalization_9/gamma/momentum
3:1 2'SGD/batch_normalization_9/beta/momentum
-:+	è@2SGD/dense_32/kernel/momentum
&:$@2SGD/dense_32/bias/momentum
,:*@@2SGD/dense_33/kernel/momentum
&:$@2SGD/dense_33/bias/momentum
,:*@2SGD/dense_34/kernel/momentum
&:$2SGD/dense_34/bias/momentum
R:P 2BSGD/token_and_position_embedding_4/embedding_8/embeddings/momentum
S:Q	R 2BSGD/token_and_position_embedding_4/embedding_9/embeddings/momentum
X:V  2DSGD/transformer_block_9/multi_head_attention_9/query/kernel/momentum
R:P 2BSGD/transformer_block_9/multi_head_attention_9/query/bias/momentum
V:T  2BSGD/transformer_block_9/multi_head_attention_9/key/kernel/momentum
P:N 2@SGD/transformer_block_9/multi_head_attention_9/key/bias/momentum
X:V  2DSGD/transformer_block_9/multi_head_attention_9/value/kernel/momentum
R:P 2BSGD/transformer_block_9/multi_head_attention_9/value/bias/momentum
c:a  2OSGD/transformer_block_9/multi_head_attention_9/attention_output/kernel/momentum
Y:W 2MSGD/transformer_block_9/multi_head_attention_9/attention_output/bias/momentum
,:* @2SGD/dense_30/kernel/momentum
&:$@2SGD/dense_30/bias/momentum
,:*@ 2SGD/dense_31/kernel/momentum
&:$ 2SGD/dense_31/bias/momentum
I:G 2=SGD/transformer_block_9/layer_normalization_18/gamma/momentum
H:F 2<SGD/transformer_block_9/layer_normalization_18/beta/momentum
I:G 2=SGD/transformer_block_9/layer_normalization_19/gamma/momentum
H:F 2<SGD/transformer_block_9/layer_normalization_19/beta/momentum
Ú2×
C__inference_model_4_layer_call_and_return_conditional_losses_514385
C__inference_model_4_layer_call_and_return_conditional_losses_515220
C__inference_model_4_layer_call_and_return_conditional_losses_515463
C__inference_model_4_layer_call_and_return_conditional_losses_514479À
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
2
!__inference__wrapped_model_513029à
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
annotationsª *P¢M
KH
"
input_9ÿÿÿÿÿÿÿÿÿR
"
input_10ÿÿÿÿÿÿÿÿÿ
î2ë
(__inference_model_4_layer_call_fn_514652
(__inference_model_4_layer_call_fn_515541
(__inference_model_4_layer_call_fn_515619
(__inference_model_4_layer_call_fn_514824À
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
ÿ2ü
Z__inference_token_and_position_embedding_4_layer_call_and_return_conditional_losses_515643
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
ä2á
?__inference_token_and_position_embedding_4_layer_call_fn_515652
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
î2ë
D__inference_conv1d_8_layer_call_and_return_conditional_losses_515668¢
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
)__inference_conv1d_8_layer_call_fn_515677¢
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
«2¨
P__inference_average_pooling1d_12_layer_call_and_return_conditional_losses_513038Ó
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
2
5__inference_average_pooling1d_12_layer_call_fn_513044Ó
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
î2ë
D__inference_conv1d_9_layer_call_and_return_conditional_losses_515693¢
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
)__inference_conv1d_9_layer_call_fn_515702¢
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
«2¨
P__inference_average_pooling1d_13_layer_call_and_return_conditional_losses_513053Ó
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
2
5__inference_average_pooling1d_13_layer_call_fn_513059Ó
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
«2¨
P__inference_average_pooling1d_14_layer_call_and_return_conditional_losses_513068Ó
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
2
5__inference_average_pooling1d_14_layer_call_fn_513074Ó
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
2
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_515738
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_515758
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_515820
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_515840´
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
6__inference_batch_normalization_8_layer_call_fn_515853
6__inference_batch_normalization_8_layer_call_fn_515771
6__inference_batch_normalization_8_layer_call_fn_515866
6__inference_batch_normalization_8_layer_call_fn_515784´
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
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_515922
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_516004
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_515984
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_515902´
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
6__inference_batch_normalization_9_layer_call_fn_515935
6__inference_batch_normalization_9_layer_call_fn_516030
6__inference_batch_normalization_9_layer_call_fn_516017
6__inference_batch_normalization_9_layer_call_fn_515948´
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
ë2è
A__inference_add_4_layer_call_and_return_conditional_losses_516036¢
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
Ð2Í
&__inference_add_4_layer_call_fn_516042¢
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
Ø2Õ
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_516190
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_516317°
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
¢2
4__inference_transformer_block_9_layer_call_fn_516354
4__inference_transformer_block_9_layer_call_fn_516391°
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
ï2ì
E__inference_flatten_4_layer_call_and_return_conditional_losses_516397¢
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
Ô2Ñ
*__inference_flatten_4_layer_call_fn_516402¢
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
I__inference_concatenate_4_layer_call_and_return_conditional_losses_516409¢
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
.__inference_concatenate_4_layer_call_fn_516415¢
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
D__inference_dense_32_layer_call_and_return_conditional_losses_516426¢
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
)__inference_dense_32_layer_call_fn_516435¢
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
Ê2Ç
F__inference_dropout_28_layer_call_and_return_conditional_losses_516452
F__inference_dropout_28_layer_call_and_return_conditional_losses_516447´
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
2
+__inference_dropout_28_layer_call_fn_516462
+__inference_dropout_28_layer_call_fn_516457´
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
î2ë
D__inference_dense_33_layer_call_and_return_conditional_losses_516473¢
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
)__inference_dense_33_layer_call_fn_516482¢
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
Ê2Ç
F__inference_dropout_29_layer_call_and_return_conditional_losses_516499
F__inference_dropout_29_layer_call_and_return_conditional_losses_516494´
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
2
+__inference_dropout_29_layer_call_fn_516504
+__inference_dropout_29_layer_call_fn_516509´
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
î2ë
D__inference_dense_34_layer_call_and_return_conditional_losses_516519¢
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
)__inference_dense_34_layer_call_fn_516528¢
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
ÓBÐ
$__inference_signature_wrapper_514910input_10input_9"
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
î2ë
H__inference_sequential_9_layer_call_and_return_conditional_losses_516585
H__inference_sequential_9_layer_call_and_return_conditional_losses_513466
H__inference_sequential_9_layer_call_and_return_conditional_losses_516642
H__inference_sequential_9_layer_call_and_return_conditional_losses_513452À
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
2ÿ
-__inference_sequential_9_layer_call_fn_513494
-__inference_sequential_9_layer_call_fn_513521
-__inference_sequential_9_layer_call_fn_516655
-__inference_sequential_9_layer_call_fn_516668À
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
î2ë
D__inference_dense_30_layer_call_and_return_conditional_losses_516699¢
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
)__inference_dense_30_layer_call_fn_516708¢
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
D__inference_dense_31_layer_call_and_return_conditional_losses_516738¢
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
)__inference_dense_31_layer_call_fn_516747¢
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
 í
!__inference__wrapped_model_513029Ç4~ !*+<9;:EBDC`ajktuZ¢W
P¢M
KH
"
input_9ÿÿÿÿÿÿÿÿÿR
"
input_10ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_34"
dense_34ÿÿÿÿÿÿÿÿÿÕ
A__inference_add_4_layer_call_and_return_conditional_losses_516036b¢_
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
&__inference_add_4_layer_call_fn_516042b¢_
X¢U
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿ# 
&#
inputs/1ÿÿÿÿÿÿÿÿÿ# 
ª "ÿÿÿÿÿÿÿÿÿ# Ù
P__inference_average_pooling1d_12_layer_call_and_return_conditional_losses_513038E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 °
5__inference_average_pooling1d_12_layer_call_fn_513044wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÙ
P__inference_average_pooling1d_13_layer_call_and_return_conditional_losses_513053E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 °
5__inference_average_pooling1d_13_layer_call_fn_513059wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÙ
P__inference_average_pooling1d_14_layer_call_and_return_conditional_losses_513068E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 °
5__inference_average_pooling1d_14_layer_call_fn_513074wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÑ
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_515738|;<9:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ñ
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_515758|<9;:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ¿
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_515820j;<9:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ¿
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_515840j<9;:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ©
6__inference_batch_normalization_8_layer_call_fn_515771o;<9:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ©
6__inference_batch_normalization_8_layer_call_fn_515784o<9;:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
6__inference_batch_normalization_8_layer_call_fn_515853];<9:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª "ÿÿÿÿÿÿÿÿÿ# 
6__inference_batch_normalization_8_layer_call_fn_515866]<9;:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª "ÿÿÿÿÿÿÿÿÿ# Ñ
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_515902|DEBC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ñ
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_515922|EBDC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ¿
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_515984jDEBC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ¿
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_516004jEBDC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ©
6__inference_batch_normalization_9_layer_call_fn_515935oDEBC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ©
6__inference_batch_normalization_9_layer_call_fn_515948oEBDC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
6__inference_batch_normalization_9_layer_call_fn_516017]DEBC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª "ÿÿÿÿÿÿÿÿÿ# 
6__inference_batch_normalization_9_layer_call_fn_516030]EBDC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª "ÿÿÿÿÿÿÿÿÿ# Ó
I__inference_concatenate_4_layer_call_and_return_conditional_losses_516409[¢X
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
.__inference_concatenate_4_layer_call_fn_516415x[¢X
Q¢N
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿà
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿè®
D__inference_conv1d_8_layer_call_and_return_conditional_losses_515668f !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿR 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿR 
 
)__inference_conv1d_8_layer_call_fn_515677Y !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿR 
ª "ÿÿÿÿÿÿÿÿÿR ®
D__inference_conv1d_9_layer_call_and_return_conditional_losses_515693f*+4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÞ 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÞ 
 
)__inference_conv1d_9_layer_call_fn_515702Y*+4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÞ 
ª "ÿÿÿÿÿÿÿÿÿÞ ®
D__inference_dense_30_layer_call_and_return_conditional_losses_516699f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ#@
 
)__inference_dense_30_layer_call_fn_516708Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª "ÿÿÿÿÿÿÿÿÿ#@®
D__inference_dense_31_layer_call_and_return_conditional_losses_516738f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ#@
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 
)__inference_dense_31_layer_call_fn_516747Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ#@
ª "ÿÿÿÿÿÿÿÿÿ# ¥
D__inference_dense_32_layer_call_and_return_conditional_losses_516426]`a0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿè
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 }
)__inference_dense_32_layer_call_fn_516435P`a0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿè
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dense_33_layer_call_and_return_conditional_losses_516473\jk/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
)__inference_dense_33_layer_call_fn_516482Ojk/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dense_34_layer_call_and_return_conditional_losses_516519\tu/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_34_layer_call_fn_516528Otu/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dropout_28_layer_call_and_return_conditional_losses_516447\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¦
F__inference_dropout_28_layer_call_and_return_conditional_losses_516452\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ~
+__inference_dropout_28_layer_call_fn_516457O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@~
+__inference_dropout_28_layer_call_fn_516462O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@¦
F__inference_dropout_29_layer_call_and_return_conditional_losses_516494\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¦
F__inference_dropout_29_layer_call_and_return_conditional_losses_516499\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ~
+__inference_dropout_29_layer_call_fn_516504O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@~
+__inference_dropout_29_layer_call_fn_516509O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@¦
E__inference_flatten_4_layer_call_and_return_conditional_losses_516397]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿà
 ~
*__inference_flatten_4_layer_call_fn_516402P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª "ÿÿÿÿÿÿÿÿÿà
C__inference_model_4_layer_call_and_return_conditional_losses_514385Á4~ !*+;<9:DEBC`ajktub¢_
X¢U
KH
"
input_9ÿÿÿÿÿÿÿÿÿR
"
input_10ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
C__inference_model_4_layer_call_and_return_conditional_losses_514479Á4~ !*+<9;:EBDC`ajktub¢_
X¢U
KH
"
input_9ÿÿÿÿÿÿÿÿÿR
"
input_10ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
C__inference_model_4_layer_call_and_return_conditional_losses_515220Â4~ !*+;<9:DEBC`ajktuc¢`
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
 
C__inference_model_4_layer_call_and_return_conditional_losses_515463Â4~ !*+<9;:EBDC`ajktuc¢`
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
 á
(__inference_model_4_layer_call_fn_514652´4~ !*+;<9:DEBC`ajktub¢_
X¢U
KH
"
input_9ÿÿÿÿÿÿÿÿÿR
"
input_10ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿá
(__inference_model_4_layer_call_fn_514824´4~ !*+<9;:EBDC`ajktub¢_
X¢U
KH
"
input_9ÿÿÿÿÿÿÿÿÿR
"
input_10ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿâ
(__inference_model_4_layer_call_fn_515541µ4~ !*+;<9:DEBC`ajktuc¢`
Y¢V
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿR
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿâ
(__inference_model_4_layer_call_fn_515619µ4~ !*+<9;:EBDC`ajktuc¢`
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
H__inference_sequential_9_layer_call_and_return_conditional_losses_513452zC¢@
9¢6
,)
dense_30_inputÿÿÿÿÿÿÿÿÿ# 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 Æ
H__inference_sequential_9_layer_call_and_return_conditional_losses_513466zC¢@
9¢6
,)
dense_30_inputÿÿÿÿÿÿÿÿÿ# 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ¾
H__inference_sequential_9_layer_call_and_return_conditional_losses_516585r;¢8
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
H__inference_sequential_9_layer_call_and_return_conditional_losses_516642r;¢8
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
-__inference_sequential_9_layer_call_fn_513494mC¢@
9¢6
,)
dense_30_inputÿÿÿÿÿÿÿÿÿ# 
p

 
ª "ÿÿÿÿÿÿÿÿÿ# 
-__inference_sequential_9_layer_call_fn_513521mC¢@
9¢6
,)
dense_30_inputÿÿÿÿÿÿÿÿÿ# 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ# 
-__inference_sequential_9_layer_call_fn_516655e;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p

 
ª "ÿÿÿÿÿÿÿÿÿ# 
-__inference_sequential_9_layer_call_fn_516668e;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ# 
$__inference_signature_wrapper_514910Ù4~ !*+<9;:EBDC`ajktul¢i
¢ 
bª_
.
input_10"
input_10ÿÿÿÿÿÿÿÿÿ
-
input_9"
input_9ÿÿÿÿÿÿÿÿÿR"3ª0
.
dense_34"
dense_34ÿÿÿÿÿÿÿÿÿ»
Z__inference_token_and_position_embedding_4_layer_call_and_return_conditional_losses_515643]~+¢(
!¢

xÿÿÿÿÿÿÿÿÿR
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿR 
 
?__inference_token_and_position_embedding_4_layer_call_fn_515652P~+¢(
!¢

xÿÿÿÿÿÿÿÿÿR
ª "ÿÿÿÿÿÿÿÿÿR Ú
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_516190 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 Ú
O__inference_transformer_block_9_layer_call_and_return_conditional_losses_516317 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ±
4__inference_transformer_block_9_layer_call_fn_516354y 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª "ÿÿÿÿÿÿÿÿÿ# ±
4__inference_transformer_block_9_layer_call_fn_516391y 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª "ÿÿÿÿÿÿÿÿÿ# 