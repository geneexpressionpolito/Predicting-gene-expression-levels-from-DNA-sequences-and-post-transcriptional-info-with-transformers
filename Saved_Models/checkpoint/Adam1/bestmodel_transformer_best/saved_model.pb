Áü2
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
 "serve*2.4.12v2.4.1-0-g85c8b2a817f8ÕÌ,
~
conv1d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv1d_6/kernel
w
#conv1d_6/kernel/Read/ReadVariableOpReadVariableOpconv1d_6/kernel*"
_output_shapes
:  *
dtype0
r
conv1d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_6/bias
k
!conv1d_6/bias/Read/ReadVariableOpReadVariableOpconv1d_6/bias*
_output_shapes
: *
dtype0
~
conv1d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  * 
shared_nameconv1d_7/kernel
w
#conv1d_7/kernel/Read/ReadVariableOpReadVariableOpconv1d_7/kernel*"
_output_shapes
:	  *
dtype0
r
conv1d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_7/bias
k
!conv1d_7/bias/Read/ReadVariableOpReadVariableOpconv1d_7/bias*
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
{
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	È@* 
shared_namedense_25/kernel
t
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel*
_output_shapes
:	È@*
dtype0
r
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_25/bias
k
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes
:@*
dtype0
z
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_26/kernel
s
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes

:@@*
dtype0
r
dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_26/bias
k
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes
:@*
dtype0
z
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_27/kernel
s
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel*
_output_shapes

:@*
dtype0
r
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_27/bias
k
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes
:*
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
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
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
Æ
5token_and_position_embedding_3/embedding_6/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75token_and_position_embedding_3/embedding_6/embeddings
¿
Itoken_and_position_embedding_3/embedding_6/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_3/embedding_6/embeddings*
_output_shapes

: *
dtype0
È
5token_and_position_embedding_3/embedding_7/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *F
shared_name75token_and_position_embedding_3/embedding_7/embeddings
Á
Itoken_and_position_embedding_3/embedding_7/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_3/embedding_7/embeddings* 
_output_shapes
:
  *
dtype0
Î
7transformer_block_7/multi_head_attention_7/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_7/multi_head_attention_7/query/kernel
Ç
Ktransformer_block_7/multi_head_attention_7/query/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_7/multi_head_attention_7/query/kernel*"
_output_shapes
:  *
dtype0
Æ
5transformer_block_7/multi_head_attention_7/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_7/multi_head_attention_7/query/bias
¿
Itransformer_block_7/multi_head_attention_7/query/bias/Read/ReadVariableOpReadVariableOp5transformer_block_7/multi_head_attention_7/query/bias*
_output_shapes

: *
dtype0
Ê
5transformer_block_7/multi_head_attention_7/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *F
shared_name75transformer_block_7/multi_head_attention_7/key/kernel
Ã
Itransformer_block_7/multi_head_attention_7/key/kernel/Read/ReadVariableOpReadVariableOp5transformer_block_7/multi_head_attention_7/key/kernel*"
_output_shapes
:  *
dtype0
Â
3transformer_block_7/multi_head_attention_7/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *D
shared_name53transformer_block_7/multi_head_attention_7/key/bias
»
Gtransformer_block_7/multi_head_attention_7/key/bias/Read/ReadVariableOpReadVariableOp3transformer_block_7/multi_head_attention_7/key/bias*
_output_shapes

: *
dtype0
Î
7transformer_block_7/multi_head_attention_7/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_7/multi_head_attention_7/value/kernel
Ç
Ktransformer_block_7/multi_head_attention_7/value/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_7/multi_head_attention_7/value/kernel*"
_output_shapes
:  *
dtype0
Æ
5transformer_block_7/multi_head_attention_7/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_7/multi_head_attention_7/value/bias
¿
Itransformer_block_7/multi_head_attention_7/value/bias/Read/ReadVariableOpReadVariableOp5transformer_block_7/multi_head_attention_7/value/bias*
_output_shapes

: *
dtype0
ä
Btransformer_block_7/multi_head_attention_7/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBtransformer_block_7/multi_head_attention_7/attention_output/kernel
Ý
Vtransformer_block_7/multi_head_attention_7/attention_output/kernel/Read/ReadVariableOpReadVariableOpBtransformer_block_7/multi_head_attention_7/attention_output/kernel*"
_output_shapes
:  *
dtype0
Ø
@transformer_block_7/multi_head_attention_7/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@transformer_block_7/multi_head_attention_7/attention_output/bias
Ñ
Ttransformer_block_7/multi_head_attention_7/attention_output/bias/Read/ReadVariableOpReadVariableOp@transformer_block_7/multi_head_attention_7/attention_output/bias*
_output_shapes
: *
dtype0
z
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @* 
shared_namedense_23/kernel
s
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes

: @*
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:@*
dtype0
z
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_24/kernel
s
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes

:@ *
dtype0
r
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
: *
dtype0
¸
0transformer_block_7/layer_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20transformer_block_7/layer_normalization_14/gamma
±
Dtransformer_block_7/layer_normalization_14/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_7/layer_normalization_14/gamma*
_output_shapes
: *
dtype0
¶
/transformer_block_7/layer_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_7/layer_normalization_14/beta
¯
Ctransformer_block_7/layer_normalization_14/beta/Read/ReadVariableOpReadVariableOp/transformer_block_7/layer_normalization_14/beta*
_output_shapes
: *
dtype0
¸
0transformer_block_7/layer_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20transformer_block_7/layer_normalization_15/gamma
±
Dtransformer_block_7/layer_normalization_15/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_7/layer_normalization_15/gamma*
_output_shapes
: *
dtype0
¶
/transformer_block_7/layer_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_7/layer_normalization_15/beta
¯
Ctransformer_block_7/layer_normalization_15/beta/Read/ReadVariableOpReadVariableOp/transformer_block_7/layer_normalization_15/beta*
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

Adam/conv1d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/conv1d_6/kernel/m

*Adam/conv1d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_6/kernel/m*"
_output_shapes
:  *
dtype0

Adam/conv1d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_6/bias/m
y
(Adam/conv1d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_6/bias/m*
_output_shapes
: *
dtype0

Adam/conv1d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *'
shared_nameAdam/conv1d_7/kernel/m

*Adam/conv1d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_7/kernel/m*"
_output_shapes
:	  *
dtype0

Adam/conv1d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_7/bias/m
y
(Adam/conv1d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_7/bias/m*
_output_shapes
: *
dtype0

"Adam/batch_normalization_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_6/gamma/m

6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/m*
_output_shapes
: *
dtype0

!Adam/batch_normalization_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_6/beta/m

5Adam/batch_normalization_6/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/m*
_output_shapes
: *
dtype0

"Adam/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_7/gamma/m

6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/m*
_output_shapes
: *
dtype0

!Adam/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_7/beta/m

5Adam/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/m*
_output_shapes
: *
dtype0

Adam/dense_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	È@*'
shared_nameAdam/dense_25/kernel/m

*Adam/dense_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/m*
_output_shapes
:	È@*
dtype0

Adam/dense_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_25/bias/m
y
(Adam/dense_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_26/kernel/m

*Adam/dense_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/m*
_output_shapes

:@@*
dtype0

Adam/dense_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_26/bias/m
y
(Adam/dense_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_27/kernel/m

*Adam/dense_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/m*
_output_shapes

:@*
dtype0

Adam/dense_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_27/bias/m
y
(Adam/dense_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/m*
_output_shapes
:*
dtype0
Ô
<Adam/token_and_position_embedding_3/embedding_6/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *M
shared_name><Adam/token_and_position_embedding_3/embedding_6/embeddings/m
Í
PAdam/token_and_position_embedding_3/embedding_6/embeddings/m/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_3/embedding_6/embeddings/m*
_output_shapes

: *
dtype0
Ö
<Adam/token_and_position_embedding_3/embedding_7/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *M
shared_name><Adam/token_and_position_embedding_3/embedding_7/embeddings/m
Ï
PAdam/token_and_position_embedding_3/embedding_7/embeddings/m/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_3/embedding_7/embeddings/m* 
_output_shapes
:
  *
dtype0
Ü
>Adam/transformer_block_7/multi_head_attention_7/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *O
shared_name@>Adam/transformer_block_7/multi_head_attention_7/query/kernel/m
Õ
RAdam/transformer_block_7/multi_head_attention_7/query/kernel/m/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_7/multi_head_attention_7/query/kernel/m*"
_output_shapes
:  *
dtype0
Ô
<Adam/transformer_block_7/multi_head_attention_7/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *M
shared_name><Adam/transformer_block_7/multi_head_attention_7/query/bias/m
Í
PAdam/transformer_block_7/multi_head_attention_7/query/bias/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_7/multi_head_attention_7/query/bias/m*
_output_shapes

: *
dtype0
Ø
<Adam/transformer_block_7/multi_head_attention_7/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *M
shared_name><Adam/transformer_block_7/multi_head_attention_7/key/kernel/m
Ñ
PAdam/transformer_block_7/multi_head_attention_7/key/kernel/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_7/multi_head_attention_7/key/kernel/m*"
_output_shapes
:  *
dtype0
Ð
:Adam/transformer_block_7/multi_head_attention_7/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *K
shared_name<:Adam/transformer_block_7/multi_head_attention_7/key/bias/m
É
NAdam/transformer_block_7/multi_head_attention_7/key/bias/m/Read/ReadVariableOpReadVariableOp:Adam/transformer_block_7/multi_head_attention_7/key/bias/m*
_output_shapes

: *
dtype0
Ü
>Adam/transformer_block_7/multi_head_attention_7/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *O
shared_name@>Adam/transformer_block_7/multi_head_attention_7/value/kernel/m
Õ
RAdam/transformer_block_7/multi_head_attention_7/value/kernel/m/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_7/multi_head_attention_7/value/kernel/m*"
_output_shapes
:  *
dtype0
Ô
<Adam/transformer_block_7/multi_head_attention_7/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *M
shared_name><Adam/transformer_block_7/multi_head_attention_7/value/bias/m
Í
PAdam/transformer_block_7/multi_head_attention_7/value/bias/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_7/multi_head_attention_7/value/bias/m*
_output_shapes

: *
dtype0
ò
IAdam/transformer_block_7/multi_head_attention_7/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *Z
shared_nameKIAdam/transformer_block_7/multi_head_attention_7/attention_output/kernel/m
ë
]Adam/transformer_block_7/multi_head_attention_7/attention_output/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/transformer_block_7/multi_head_attention_7/attention_output/kernel/m*"
_output_shapes
:  *
dtype0
æ
GAdam/transformer_block_7/multi_head_attention_7/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/transformer_block_7/multi_head_attention_7/attention_output/bias/m
ß
[Adam/transformer_block_7/multi_head_attention_7/attention_output/bias/m/Read/ReadVariableOpReadVariableOpGAdam/transformer_block_7/multi_head_attention_7/attention_output/bias/m*
_output_shapes
: *
dtype0

Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*'
shared_nameAdam/dense_23/kernel/m

*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m*
_output_shapes

: @*
dtype0

Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_23/bias/m
y
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_24/kernel/m

*Adam/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/m*
_output_shapes

:@ *
dtype0

Adam/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_24/bias/m
y
(Adam/dense_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/m*
_output_shapes
: *
dtype0
Æ
7Adam/transformer_block_7/layer_normalization_14/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/transformer_block_7/layer_normalization_14/gamma/m
¿
KAdam/transformer_block_7/layer_normalization_14/gamma/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_7/layer_normalization_14/gamma/m*
_output_shapes
: *
dtype0
Ä
6Adam/transformer_block_7/layer_normalization_14/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_7/layer_normalization_14/beta/m
½
JAdam/transformer_block_7/layer_normalization_14/beta/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_7/layer_normalization_14/beta/m*
_output_shapes
: *
dtype0
Æ
7Adam/transformer_block_7/layer_normalization_15/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/transformer_block_7/layer_normalization_15/gamma/m
¿
KAdam/transformer_block_7/layer_normalization_15/gamma/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_7/layer_normalization_15/gamma/m*
_output_shapes
: *
dtype0
Ä
6Adam/transformer_block_7/layer_normalization_15/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_7/layer_normalization_15/beta/m
½
JAdam/transformer_block_7/layer_normalization_15/beta/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_7/layer_normalization_15/beta/m*
_output_shapes
: *
dtype0

Adam/conv1d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/conv1d_6/kernel/v

*Adam/conv1d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_6/kernel/v*"
_output_shapes
:  *
dtype0

Adam/conv1d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_6/bias/v
y
(Adam/conv1d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_6/bias/v*
_output_shapes
: *
dtype0

Adam/conv1d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *'
shared_nameAdam/conv1d_7/kernel/v

*Adam/conv1d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_7/kernel/v*"
_output_shapes
:	  *
dtype0

Adam/conv1d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_7/bias/v
y
(Adam/conv1d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_7/bias/v*
_output_shapes
: *
dtype0

"Adam/batch_normalization_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_6/gamma/v

6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/v*
_output_shapes
: *
dtype0

!Adam/batch_normalization_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_6/beta/v

5Adam/batch_normalization_6/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/v*
_output_shapes
: *
dtype0

"Adam/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_7/gamma/v

6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/v*
_output_shapes
: *
dtype0

!Adam/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_7/beta/v

5Adam/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/v*
_output_shapes
: *
dtype0

Adam/dense_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	È@*'
shared_nameAdam/dense_25/kernel/v

*Adam/dense_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/v*
_output_shapes
:	È@*
dtype0

Adam/dense_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_25/bias/v
y
(Adam/dense_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_26/kernel/v

*Adam/dense_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/v*
_output_shapes

:@@*
dtype0

Adam/dense_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_26/bias/v
y
(Adam/dense_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_27/kernel/v

*Adam/dense_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/v*
_output_shapes

:@*
dtype0

Adam/dense_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_27/bias/v
y
(Adam/dense_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/v*
_output_shapes
:*
dtype0
Ô
<Adam/token_and_position_embedding_3/embedding_6/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *M
shared_name><Adam/token_and_position_embedding_3/embedding_6/embeddings/v
Í
PAdam/token_and_position_embedding_3/embedding_6/embeddings/v/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_3/embedding_6/embeddings/v*
_output_shapes

: *
dtype0
Ö
<Adam/token_and_position_embedding_3/embedding_7/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *M
shared_name><Adam/token_and_position_embedding_3/embedding_7/embeddings/v
Ï
PAdam/token_and_position_embedding_3/embedding_7/embeddings/v/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_3/embedding_7/embeddings/v* 
_output_shapes
:
  *
dtype0
Ü
>Adam/transformer_block_7/multi_head_attention_7/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *O
shared_name@>Adam/transformer_block_7/multi_head_attention_7/query/kernel/v
Õ
RAdam/transformer_block_7/multi_head_attention_7/query/kernel/v/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_7/multi_head_attention_7/query/kernel/v*"
_output_shapes
:  *
dtype0
Ô
<Adam/transformer_block_7/multi_head_attention_7/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *M
shared_name><Adam/transformer_block_7/multi_head_attention_7/query/bias/v
Í
PAdam/transformer_block_7/multi_head_attention_7/query/bias/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_7/multi_head_attention_7/query/bias/v*
_output_shapes

: *
dtype0
Ø
<Adam/transformer_block_7/multi_head_attention_7/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *M
shared_name><Adam/transformer_block_7/multi_head_attention_7/key/kernel/v
Ñ
PAdam/transformer_block_7/multi_head_attention_7/key/kernel/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_7/multi_head_attention_7/key/kernel/v*"
_output_shapes
:  *
dtype0
Ð
:Adam/transformer_block_7/multi_head_attention_7/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *K
shared_name<:Adam/transformer_block_7/multi_head_attention_7/key/bias/v
É
NAdam/transformer_block_7/multi_head_attention_7/key/bias/v/Read/ReadVariableOpReadVariableOp:Adam/transformer_block_7/multi_head_attention_7/key/bias/v*
_output_shapes

: *
dtype0
Ü
>Adam/transformer_block_7/multi_head_attention_7/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *O
shared_name@>Adam/transformer_block_7/multi_head_attention_7/value/kernel/v
Õ
RAdam/transformer_block_7/multi_head_attention_7/value/kernel/v/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_7/multi_head_attention_7/value/kernel/v*"
_output_shapes
:  *
dtype0
Ô
<Adam/transformer_block_7/multi_head_attention_7/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *M
shared_name><Adam/transformer_block_7/multi_head_attention_7/value/bias/v
Í
PAdam/transformer_block_7/multi_head_attention_7/value/bias/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_7/multi_head_attention_7/value/bias/v*
_output_shapes

: *
dtype0
ò
IAdam/transformer_block_7/multi_head_attention_7/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *Z
shared_nameKIAdam/transformer_block_7/multi_head_attention_7/attention_output/kernel/v
ë
]Adam/transformer_block_7/multi_head_attention_7/attention_output/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/transformer_block_7/multi_head_attention_7/attention_output/kernel/v*"
_output_shapes
:  *
dtype0
æ
GAdam/transformer_block_7/multi_head_attention_7/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/transformer_block_7/multi_head_attention_7/attention_output/bias/v
ß
[Adam/transformer_block_7/multi_head_attention_7/attention_output/bias/v/Read/ReadVariableOpReadVariableOpGAdam/transformer_block_7/multi_head_attention_7/attention_output/bias/v*
_output_shapes
: *
dtype0

Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*'
shared_nameAdam/dense_23/kernel/v

*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v*
_output_shapes

: @*
dtype0

Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_23/bias/v
y
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_24/kernel/v

*Adam/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/v*
_output_shapes

:@ *
dtype0

Adam/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_24/bias/v
y
(Adam/dense_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/v*
_output_shapes
: *
dtype0
Æ
7Adam/transformer_block_7/layer_normalization_14/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/transformer_block_7/layer_normalization_14/gamma/v
¿
KAdam/transformer_block_7/layer_normalization_14/gamma/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_7/layer_normalization_14/gamma/v*
_output_shapes
: *
dtype0
Ä
6Adam/transformer_block_7/layer_normalization_14/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_7/layer_normalization_14/beta/v
½
JAdam/transformer_block_7/layer_normalization_14/beta/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_7/layer_normalization_14/beta/v*
_output_shapes
: *
dtype0
Æ
7Adam/transformer_block_7/layer_normalization_15/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/transformer_block_7/layer_normalization_15/gamma/v
¿
KAdam/transformer_block_7/layer_normalization_15/gamma/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_7/layer_normalization_15/gamma/v*
_output_shapes
: *
dtype0
Ä
6Adam/transformer_block_7/layer_normalization_15/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_7/layer_normalization_15/beta/v
½
JAdam/transformer_block_7/layer_normalization_15/beta/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_7/layer_normalization_15/beta/v*
_output_shapes
: *
dtype0

NoOpNoOp
Ô
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¾Ó
value³ÓB¯Ó B§Ó
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
â

zbeta_1

{beta_2
	|decay
}learning_rate
~iter m!m*m+m9m:mBmCm`mamjmkmtmumm 	m¡	m¢	m£	m¤	m¥	m¦	m§	m¨	m©	mª	m«	m¬	m­	m®	m¯	m°	m± v²!v³*v´+vµ9v¶:v·Bv¸Cv¹`vºav»jv¼kv½tv¾uv¿vÀ	vÁ	vÂ	vÃ	vÄ	vÅ	vÆ	vÇ	vÈ	vÉ	vÊ	vË	vÌ	vÍ	vÎ	vÏ	vÐ	vÑ
§
0
1
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
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
`30
a31
j32
k33
t34
u35

0
1
 2
!3
*4
+5
96
:7
B8
C9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
`26
a27
j28
k29
t30
u31
 
²
layer_metrics
metrics
 layer_regularization_losses
non_trainable_variables
	variables
layers
trainable_variables
regularization_losses
 
f

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
g

embeddings
	variables
trainable_variables
regularization_losses
	keras_api

0
1

0
1
 
²
layer_metrics
metrics
  layer_regularization_losses
	variables
trainable_variables
¡layers
¢non_trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv1d_6/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_6/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
²
£layer_metrics
¤metrics
 ¥layer_regularization_losses
"	variables
#trainable_variables
¦layers
§non_trainable_variables
$regularization_losses
 
 
 
²
¨layer_metrics
©metrics
 ªlayer_regularization_losses
&	variables
'trainable_variables
«layers
¬non_trainable_variables
(regularization_losses
[Y
VARIABLE_VALUEconv1d_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
²
­layer_metrics
®metrics
 ¯layer_regularization_losses
,	variables
-trainable_variables
°layers
±non_trainable_variables
.regularization_losses
 
 
 
²
²layer_metrics
³metrics
 ´layer_regularization_losses
0	variables
1trainable_variables
µlayers
¶non_trainable_variables
2regularization_losses
 
 
 
²
·layer_metrics
¸metrics
 ¹layer_regularization_losses
4	variables
5trainable_variables
ºlayers
»non_trainable_variables
6regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_6/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_6/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_6/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_6/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

90
:1
;2
<3

90
:1
 
²
¼layer_metrics
½metrics
 ¾layer_regularization_losses
=	variables
>trainable_variables
¿layers
Ànon_trainable_variables
?regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_7/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_7/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_7/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_7/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
D2
E3

B0
C1
 
²
Álayer_metrics
Âmetrics
 Ãlayer_regularization_losses
F	variables
Gtrainable_variables
Älayers
Ånon_trainable_variables
Hregularization_losses
 
 
 
²
Ælayer_metrics
Çmetrics
 Èlayer_regularization_losses
J	variables
Ktrainable_variables
Élayers
Ênon_trainable_variables
Lregularization_losses
Å
Ë_query_dense
Ì
_key_dense
Í_value_dense
Î_softmax
Ï_dropout_layer
Ð_output_dense
Ñ	variables
Òtrainable_variables
Óregularization_losses
Ô	keras_api
¨
Õlayer_with_weights-0
Õlayer-0
Ölayer_with_weights-1
Ölayer-1
×	variables
Øtrainable_variables
Ùregularization_losses
Ú	keras_api
x
	Ûaxis

gamma
	beta
Ü	variables
Ýtrainable_variables
Þregularization_losses
ß	keras_api
x
	àaxis

gamma
	beta
á	variables
âtrainable_variables
ãregularization_losses
ä	keras_api
V
å	variables
ætrainable_variables
çregularization_losses
è	keras_api
V
é	variables
êtrainable_variables
ëregularization_losses
ì	keras_api

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
 
²
ílayer_metrics
îmetrics
 ïlayer_regularization_losses
T	variables
Utrainable_variables
ðlayers
ñnon_trainable_variables
Vregularization_losses
 
 
 
²
òlayer_metrics
ómetrics
 ôlayer_regularization_losses
X	variables
Ytrainable_variables
õlayers
önon_trainable_variables
Zregularization_losses
 
 
 
²
÷layer_metrics
ømetrics
 ùlayer_regularization_losses
\	variables
]trainable_variables
úlayers
ûnon_trainable_variables
^regularization_losses
[Y
VARIABLE_VALUEdense_25/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_25/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

`0
a1

`0
a1
 
²
ülayer_metrics
ýmetrics
 þlayer_regularization_losses
b	variables
ctrainable_variables
ÿlayers
non_trainable_variables
dregularization_losses
 
 
 
²
layer_metrics
metrics
 layer_regularization_losses
f	variables
gtrainable_variables
layers
non_trainable_variables
hregularization_losses
[Y
VARIABLE_VALUEdense_26/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_26/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

j0
k1

j0
k1
 
²
layer_metrics
metrics
 layer_regularization_losses
l	variables
mtrainable_variables
layers
non_trainable_variables
nregularization_losses
 
 
 
²
layer_metrics
metrics
 layer_regularization_losses
p	variables
qtrainable_variables
layers
non_trainable_variables
rregularization_losses
[Y
VARIABLE_VALUEdense_27/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_27/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

t0
u1

t0
u1
 
²
layer_metrics
metrics
 layer_regularization_losses
v	variables
wtrainable_variables
layers
non_trainable_variables
xregularization_losses
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5token_and_position_embedding_3/embedding_6/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5token_and_position_embedding_3/embedding_7/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7transformer_block_7/multi_head_attention_7/query/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_7/multi_head_attention_7/query/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_7/multi_head_attention_7/key/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE3transformer_block_7/multi_head_attention_7/key/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7transformer_block_7/multi_head_attention_7/value/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_7/multi_head_attention_7/value/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEBtransformer_block_7/multi_head_attention_7/attention_output/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE@transformer_block_7/multi_head_attention_7/attention_output/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_23/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_23/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_24/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_24/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0transformer_block_7/layer_normalization_14/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_7/layer_normalization_14/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0transformer_block_7/layer_normalization_15/gamma'variables/28/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_7/layer_normalization_15/beta'variables/29/.ATTRIBUTES/VARIABLE_VALUE
 

0
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

0

0
 
µ
layer_metrics
metrics
 layer_regularization_losses
	variables
trainable_variables
layers
non_trainable_variables
regularization_losses

0

0
 
µ
layer_metrics
metrics
 layer_regularization_losses
	variables
trainable_variables
layers
non_trainable_variables
regularization_losses
 
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
¡
 partial_output_shape
¡full_output_shape
kernel
	bias
¢	variables
£trainable_variables
¤regularization_losses
¥	keras_api
¡
¦partial_output_shape
§full_output_shape
kernel
	bias
¨	variables
©trainable_variables
ªregularization_losses
«	keras_api
¡
¬partial_output_shape
­full_output_shape
kernel
	bias
®	variables
¯trainable_variables
°regularization_losses
±	keras_api
V
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
V
¶	variables
·trainable_variables
¸regularization_losses
¹	keras_api
¡
ºpartial_output_shape
»full_output_shape
kernel
	bias
¼	variables
½trainable_variables
¾regularization_losses
¿	keras_api
@
0
1
2
3
4
5
6
7
@
0
1
2
3
4
5
6
7
 
µ
Àlayer_metrics
Ámetrics
 Âlayer_regularization_losses
Ñ	variables
Òtrainable_variables
Ãlayers
Änon_trainable_variables
Óregularization_losses
n
kernel
	bias
Å	variables
Ætrainable_variables
Çregularization_losses
È	keras_api
n
kernel
	bias
É	variables
Êtrainable_variables
Ëregularization_losses
Ì	keras_api
 
0
1
2
3
 
0
1
2
3
 
µ
Ílayer_metrics
Îmetrics
 Ïlayer_regularization_losses
Ðnon_trainable_variables
×	variables
Ñlayers
Øtrainable_variables
Ùregularization_losses
 

0
1

0
1
 
µ
Òlayer_metrics
Ómetrics
 Ôlayer_regularization_losses
Ü	variables
Ýtrainable_variables
Õlayers
Önon_trainable_variables
Þregularization_losses
 

0
1

0
1
 
µ
×layer_metrics
Ømetrics
 Ùlayer_regularization_losses
á	variables
âtrainable_variables
Úlayers
Ûnon_trainable_variables
ãregularization_losses
 
 
 
µ
Ülayer_metrics
Ýmetrics
 Þlayer_regularization_losses
å	variables
ætrainable_variables
ßlayers
ànon_trainable_variables
çregularization_losses
 
 
 
µ
álayer_metrics
âmetrics
 ãlayer_regularization_losses
é	variables
êtrainable_variables
älayers
ånon_trainable_variables
ëregularization_losses
 
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
8

ætotal

çcount
è	variables
é	keras_api
 
 
 
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
0
1

0
1
 
µ
êlayer_metrics
ëmetrics
 ìlayer_regularization_losses
¢	variables
£trainable_variables
ílayers
înon_trainable_variables
¤regularization_losses
 
 

0
1

0
1
 
µ
ïlayer_metrics
ðmetrics
 ñlayer_regularization_losses
¨	variables
©trainable_variables
òlayers
ónon_trainable_variables
ªregularization_losses
 
 

0
1

0
1
 
µ
ôlayer_metrics
õmetrics
 ölayer_regularization_losses
®	variables
¯trainable_variables
÷layers
ønon_trainable_variables
°regularization_losses
 
 
 
µ
ùlayer_metrics
úmetrics
 ûlayer_regularization_losses
²	variables
³trainable_variables
ülayers
ýnon_trainable_variables
´regularization_losses
 
 
 
µ
þlayer_metrics
ÿmetrics
 layer_regularization_losses
¶	variables
·trainable_variables
layers
non_trainable_variables
¸regularization_losses
 
 

0
1

0
1
 
µ
layer_metrics
metrics
 layer_regularization_losses
¼	variables
½trainable_variables
layers
non_trainable_variables
¾regularization_losses
 
 
 
0
Ë0
Ì1
Í2
Î3
Ï4
Ð5
 

0
1

0
1
 
µ
layer_metrics
metrics
 layer_regularization_losses
Å	variables
Ætrainable_variables
layers
non_trainable_variables
Çregularization_losses

0
1

0
1
 
µ
layer_metrics
metrics
 layer_regularization_losses
É	variables
Êtrainable_variables
layers
non_trainable_variables
Ëregularization_losses
 
 
 
 

Õ0
Ö1
 
 
 
 
 
 
 
 
 
 
 
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
æ0
ç1

è	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
~|
VARIABLE_VALUEAdam/conv1d_6/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_6/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_7/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_7/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_6/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_6/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_7/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_7/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_25/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_25/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_26/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_26/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_27/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_27/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/token_and_position_embedding_3/embedding_6/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/token_and_position_embedding_3/embedding_7/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adam/transformer_block_7/multi_head_attention_7/query/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/transformer_block_7/multi_head_attention_7/query/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/transformer_block_7/multi_head_attention_7/key/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE:Adam/transformer_block_7/multi_head_attention_7/key/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adam/transformer_block_7/multi_head_attention_7/value/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/transformer_block_7/multi_head_attention_7/value/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEIAdam/transformer_block_7/multi_head_attention_7/attention_output/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
¡
VARIABLE_VALUEGAdam/transformer_block_7/multi_head_attention_7/attention_output/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_23/kernel/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_23/bias/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_24/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_24/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE7Adam/transformer_block_7/layer_normalization_14/gamma/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE6Adam/transformer_block_7/layer_normalization_14/beta/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE7Adam/transformer_block_7/layer_normalization_15/gamma/mCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE6Adam/transformer_block_7/layer_normalization_15/beta/mCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_6/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_6/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_7/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_7/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_6/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_6/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_7/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_7/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_25/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_25/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_26/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_26/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_27/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_27/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/token_and_position_embedding_3/embedding_6/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/token_and_position_embedding_3/embedding_7/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adam/transformer_block_7/multi_head_attention_7/query/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/transformer_block_7/multi_head_attention_7/query/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/transformer_block_7/multi_head_attention_7/key/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE:Adam/transformer_block_7/multi_head_attention_7/key/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adam/transformer_block_7/multi_head_attention_7/value/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adam/transformer_block_7/multi_head_attention_7/value/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEIAdam/transformer_block_7/multi_head_attention_7/attention_output/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
¡
VARIABLE_VALUEGAdam/transformer_block_7/multi_head_attention_7/attention_output/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_23/kernel/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_23/bias/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_24/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_24/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE7Adam/transformer_block_7/layer_normalization_14/gamma/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE6Adam/transformer_block_7/layer_normalization_14/beta/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE7Adam/transformer_block_7/layer_normalization_15/gamma/vCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE6Adam/transformer_block_7/layer_normalization_15/beta/vCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
serving_default_input_7Placeholder*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ 
z
serving_default_input_8Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_7serving_default_input_85token_and_position_embedding_3/embedding_7/embeddings5token_and_position_embedding_3/embedding_6/embeddingsconv1d_6/kernelconv1d_6/biasconv1d_7/kernelconv1d_7/bias%batch_normalization_6/moving_variancebatch_normalization_6/gamma!batch_normalization_6/moving_meanbatch_normalization_6/beta%batch_normalization_7/moving_variancebatch_normalization_7/gamma!batch_normalization_7/moving_meanbatch_normalization_7/beta7transformer_block_7/multi_head_attention_7/query/kernel5transformer_block_7/multi_head_attention_7/query/bias5transformer_block_7/multi_head_attention_7/key/kernel3transformer_block_7/multi_head_attention_7/key/bias7transformer_block_7/multi_head_attention_7/value/kernel5transformer_block_7/multi_head_attention_7/value/biasBtransformer_block_7/multi_head_attention_7/attention_output/kernel@transformer_block_7/multi_head_attention_7/attention_output/bias0transformer_block_7/layer_normalization_14/gamma/transformer_block_7/layer_normalization_14/betadense_23/kerneldense_23/biasdense_24/kerneldense_24/bias0transformer_block_7/layer_normalization_15/gamma/transformer_block_7/layer_normalization_15/betadense_25/kerneldense_25/biasdense_26/kerneldense_26/biasdense_27/kerneldense_27/bias*1
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
$__inference_signature_wrapper_196213
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
3
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1d_6/kernel/Read/ReadVariableOp!conv1d_6/bias/Read/ReadVariableOp#conv1d_7/kernel/Read/ReadVariableOp!conv1d_7/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOp#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOp#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpItoken_and_position_embedding_3/embedding_6/embeddings/Read/ReadVariableOpItoken_and_position_embedding_3/embedding_7/embeddings/Read/ReadVariableOpKtransformer_block_7/multi_head_attention_7/query/kernel/Read/ReadVariableOpItransformer_block_7/multi_head_attention_7/query/bias/Read/ReadVariableOpItransformer_block_7/multi_head_attention_7/key/kernel/Read/ReadVariableOpGtransformer_block_7/multi_head_attention_7/key/bias/Read/ReadVariableOpKtransformer_block_7/multi_head_attention_7/value/kernel/Read/ReadVariableOpItransformer_block_7/multi_head_attention_7/value/bias/Read/ReadVariableOpVtransformer_block_7/multi_head_attention_7/attention_output/kernel/Read/ReadVariableOpTtransformer_block_7/multi_head_attention_7/attention_output/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOpDtransformer_block_7/layer_normalization_14/gamma/Read/ReadVariableOpCtransformer_block_7/layer_normalization_14/beta/Read/ReadVariableOpDtransformer_block_7/layer_normalization_15/gamma/Read/ReadVariableOpCtransformer_block_7/layer_normalization_15/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/conv1d_6/kernel/m/Read/ReadVariableOp(Adam/conv1d_6/bias/m/Read/ReadVariableOp*Adam/conv1d_7/kernel/m/Read/ReadVariableOp(Adam/conv1d_7/bias/m/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_6/beta/m/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_7/beta/m/Read/ReadVariableOp*Adam/dense_25/kernel/m/Read/ReadVariableOp(Adam/dense_25/bias/m/Read/ReadVariableOp*Adam/dense_26/kernel/m/Read/ReadVariableOp(Adam/dense_26/bias/m/Read/ReadVariableOp*Adam/dense_27/kernel/m/Read/ReadVariableOp(Adam/dense_27/bias/m/Read/ReadVariableOpPAdam/token_and_position_embedding_3/embedding_6/embeddings/m/Read/ReadVariableOpPAdam/token_and_position_embedding_3/embedding_7/embeddings/m/Read/ReadVariableOpRAdam/transformer_block_7/multi_head_attention_7/query/kernel/m/Read/ReadVariableOpPAdam/transformer_block_7/multi_head_attention_7/query/bias/m/Read/ReadVariableOpPAdam/transformer_block_7/multi_head_attention_7/key/kernel/m/Read/ReadVariableOpNAdam/transformer_block_7/multi_head_attention_7/key/bias/m/Read/ReadVariableOpRAdam/transformer_block_7/multi_head_attention_7/value/kernel/m/Read/ReadVariableOpPAdam/transformer_block_7/multi_head_attention_7/value/bias/m/Read/ReadVariableOp]Adam/transformer_block_7/multi_head_attention_7/attention_output/kernel/m/Read/ReadVariableOp[Adam/transformer_block_7/multi_head_attention_7/attention_output/bias/m/Read/ReadVariableOp*Adam/dense_23/kernel/m/Read/ReadVariableOp(Adam/dense_23/bias/m/Read/ReadVariableOp*Adam/dense_24/kernel/m/Read/ReadVariableOp(Adam/dense_24/bias/m/Read/ReadVariableOpKAdam/transformer_block_7/layer_normalization_14/gamma/m/Read/ReadVariableOpJAdam/transformer_block_7/layer_normalization_14/beta/m/Read/ReadVariableOpKAdam/transformer_block_7/layer_normalization_15/gamma/m/Read/ReadVariableOpJAdam/transformer_block_7/layer_normalization_15/beta/m/Read/ReadVariableOp*Adam/conv1d_6/kernel/v/Read/ReadVariableOp(Adam/conv1d_6/bias/v/Read/ReadVariableOp*Adam/conv1d_7/kernel/v/Read/ReadVariableOp(Adam/conv1d_7/bias/v/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_6/beta/v/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_7/beta/v/Read/ReadVariableOp*Adam/dense_25/kernel/v/Read/ReadVariableOp(Adam/dense_25/bias/v/Read/ReadVariableOp*Adam/dense_26/kernel/v/Read/ReadVariableOp(Adam/dense_26/bias/v/Read/ReadVariableOp*Adam/dense_27/kernel/v/Read/ReadVariableOp(Adam/dense_27/bias/v/Read/ReadVariableOpPAdam/token_and_position_embedding_3/embedding_6/embeddings/v/Read/ReadVariableOpPAdam/token_and_position_embedding_3/embedding_7/embeddings/v/Read/ReadVariableOpRAdam/transformer_block_7/multi_head_attention_7/query/kernel/v/Read/ReadVariableOpPAdam/transformer_block_7/multi_head_attention_7/query/bias/v/Read/ReadVariableOpPAdam/transformer_block_7/multi_head_attention_7/key/kernel/v/Read/ReadVariableOpNAdam/transformer_block_7/multi_head_attention_7/key/bias/v/Read/ReadVariableOpRAdam/transformer_block_7/multi_head_attention_7/value/kernel/v/Read/ReadVariableOpPAdam/transformer_block_7/multi_head_attention_7/value/bias/v/Read/ReadVariableOp]Adam/transformer_block_7/multi_head_attention_7/attention_output/kernel/v/Read/ReadVariableOp[Adam/transformer_block_7/multi_head_attention_7/attention_output/bias/v/Read/ReadVariableOp*Adam/dense_23/kernel/v/Read/ReadVariableOp(Adam/dense_23/bias/v/Read/ReadVariableOp*Adam/dense_24/kernel/v/Read/ReadVariableOp(Adam/dense_24/bias/v/Read/ReadVariableOpKAdam/transformer_block_7/layer_normalization_14/gamma/v/Read/ReadVariableOpJAdam/transformer_block_7/layer_normalization_14/beta/v/Read/ReadVariableOpKAdam/transformer_block_7/layer_normalization_15/gamma/v/Read/ReadVariableOpJAdam/transformer_block_7/layer_normalization_15/beta/v/Read/ReadVariableOpConst*x
Tinq
o2m	*
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
__inference__traced_save_198395
"
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_6/kernelconv1d_6/biasconv1d_7/kernelconv1d_7/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancebatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancedense_25/kerneldense_25/biasdense_26/kerneldense_26/biasdense_27/kerneldense_27/biasbeta_1beta_2decaylearning_rate	Adam/iter5token_and_position_embedding_3/embedding_6/embeddings5token_and_position_embedding_3/embedding_7/embeddings7transformer_block_7/multi_head_attention_7/query/kernel5transformer_block_7/multi_head_attention_7/query/bias5transformer_block_7/multi_head_attention_7/key/kernel3transformer_block_7/multi_head_attention_7/key/bias7transformer_block_7/multi_head_attention_7/value/kernel5transformer_block_7/multi_head_attention_7/value/biasBtransformer_block_7/multi_head_attention_7/attention_output/kernel@transformer_block_7/multi_head_attention_7/attention_output/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/bias0transformer_block_7/layer_normalization_14/gamma/transformer_block_7/layer_normalization_14/beta0transformer_block_7/layer_normalization_15/gamma/transformer_block_7/layer_normalization_15/betatotalcountAdam/conv1d_6/kernel/mAdam/conv1d_6/bias/mAdam/conv1d_7/kernel/mAdam/conv1d_7/bias/m"Adam/batch_normalization_6/gamma/m!Adam/batch_normalization_6/beta/m"Adam/batch_normalization_7/gamma/m!Adam/batch_normalization_7/beta/mAdam/dense_25/kernel/mAdam/dense_25/bias/mAdam/dense_26/kernel/mAdam/dense_26/bias/mAdam/dense_27/kernel/mAdam/dense_27/bias/m<Adam/token_and_position_embedding_3/embedding_6/embeddings/m<Adam/token_and_position_embedding_3/embedding_7/embeddings/m>Adam/transformer_block_7/multi_head_attention_7/query/kernel/m<Adam/transformer_block_7/multi_head_attention_7/query/bias/m<Adam/transformer_block_7/multi_head_attention_7/key/kernel/m:Adam/transformer_block_7/multi_head_attention_7/key/bias/m>Adam/transformer_block_7/multi_head_attention_7/value/kernel/m<Adam/transformer_block_7/multi_head_attention_7/value/bias/mIAdam/transformer_block_7/multi_head_attention_7/attention_output/kernel/mGAdam/transformer_block_7/multi_head_attention_7/attention_output/bias/mAdam/dense_23/kernel/mAdam/dense_23/bias/mAdam/dense_24/kernel/mAdam/dense_24/bias/m7Adam/transformer_block_7/layer_normalization_14/gamma/m6Adam/transformer_block_7/layer_normalization_14/beta/m7Adam/transformer_block_7/layer_normalization_15/gamma/m6Adam/transformer_block_7/layer_normalization_15/beta/mAdam/conv1d_6/kernel/vAdam/conv1d_6/bias/vAdam/conv1d_7/kernel/vAdam/conv1d_7/bias/v"Adam/batch_normalization_6/gamma/v!Adam/batch_normalization_6/beta/v"Adam/batch_normalization_7/gamma/v!Adam/batch_normalization_7/beta/vAdam/dense_25/kernel/vAdam/dense_25/bias/vAdam/dense_26/kernel/vAdam/dense_26/bias/vAdam/dense_27/kernel/vAdam/dense_27/bias/v<Adam/token_and_position_embedding_3/embedding_6/embeddings/v<Adam/token_and_position_embedding_3/embedding_7/embeddings/v>Adam/transformer_block_7/multi_head_attention_7/query/kernel/v<Adam/transformer_block_7/multi_head_attention_7/query/bias/v<Adam/transformer_block_7/multi_head_attention_7/key/kernel/v:Adam/transformer_block_7/multi_head_attention_7/key/bias/v>Adam/transformer_block_7/multi_head_attention_7/value/kernel/v<Adam/transformer_block_7/multi_head_attention_7/value/bias/vIAdam/transformer_block_7/multi_head_attention_7/attention_output/kernel/vGAdam/transformer_block_7/multi_head_attention_7/attention_output/bias/vAdam/dense_23/kernel/vAdam/dense_23/bias/vAdam/dense_24/kernel/vAdam/dense_24/bias/v7Adam/transformer_block_7/layer_normalization_14/gamma/v6Adam/transformer_block_7/layer_normalization_14/beta/v7Adam/transformer_block_7/layer_normalization_15/gamma/v6Adam/transformer_block_7/layer_normalization_15/beta/v*w
Tinp
n2l*
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
"__inference__traced_restore_198726Ñ(
ÔÜ
Ì$
C__inference_model_3_layer_call_and_return_conditional_losses_196766
inputs_0
inputs_1F
Btoken_and_position_embedding_3_embedding_7_embedding_lookup_196535F
Btoken_and_position_embedding_3_embedding_6_embedding_lookup_1965418
4conv1d_6_conv1d_expanddims_1_readvariableop_resource,
(conv1d_6_biasadd_readvariableop_resource8
4conv1d_7_conv1d_expanddims_1_readvariableop_resource,
(conv1d_7_biasadd_readvariableop_resource;
7batch_normalization_6_batchnorm_readvariableop_resource?
;batch_normalization_6_batchnorm_mul_readvariableop_resource=
9batch_normalization_6_batchnorm_readvariableop_1_resource=
9batch_normalization_6_batchnorm_readvariableop_2_resource;
7batch_normalization_7_batchnorm_readvariableop_resource?
;batch_normalization_7_batchnorm_mul_readvariableop_resource=
9batch_normalization_7_batchnorm_readvariableop_1_resource=
9batch_normalization_7_batchnorm_readvariableop_2_resourceZ
Vtransformer_block_7_multi_head_attention_7_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_7_multi_head_attention_7_query_add_readvariableop_resourceX
Ttransformer_block_7_multi_head_attention_7_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_7_multi_head_attention_7_key_add_readvariableop_resourceZ
Vtransformer_block_7_multi_head_attention_7_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_7_multi_head_attention_7_value_add_readvariableop_resourcee
atransformer_block_7_multi_head_attention_7_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_7_multi_head_attention_7_attention_output_add_readvariableop_resourceT
Ptransformer_block_7_layer_normalization_14_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_7_layer_normalization_14_batchnorm_readvariableop_resourceO
Ktransformer_block_7_sequential_7_dense_23_tensordot_readvariableop_resourceM
Itransformer_block_7_sequential_7_dense_23_biasadd_readvariableop_resourceO
Ktransformer_block_7_sequential_7_dense_24_tensordot_readvariableop_resourceM
Itransformer_block_7_sequential_7_dense_24_biasadd_readvariableop_resourceT
Ptransformer_block_7_layer_normalization_15_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_7_layer_normalization_15_batchnorm_readvariableop_resource+
'dense_25_matmul_readvariableop_resource,
(dense_25_biasadd_readvariableop_resource+
'dense_26_matmul_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource
identity¢.batch_normalization_6/batchnorm/ReadVariableOp¢0batch_normalization_6/batchnorm/ReadVariableOp_1¢0batch_normalization_6/batchnorm/ReadVariableOp_2¢2batch_normalization_6/batchnorm/mul/ReadVariableOp¢.batch_normalization_7/batchnorm/ReadVariableOp¢0batch_normalization_7/batchnorm/ReadVariableOp_1¢0batch_normalization_7/batchnorm/ReadVariableOp_2¢2batch_normalization_7/batchnorm/mul/ReadVariableOp¢conv1d_6/BiasAdd/ReadVariableOp¢+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp¢conv1d_7/BiasAdd/ReadVariableOp¢+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp¢dense_25/BiasAdd/ReadVariableOp¢dense_25/MatMul/ReadVariableOp¢dense_26/BiasAdd/ReadVariableOp¢dense_26/MatMul/ReadVariableOp¢dense_27/BiasAdd/ReadVariableOp¢dense_27/MatMul/ReadVariableOp¢;token_and_position_embedding_3/embedding_6/embedding_lookup¢;token_and_position_embedding_3/embedding_7/embedding_lookup¢Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp¢Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp¢Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp¢Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp¢Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp¢Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp¢Atransformer_block_7/multi_head_attention_7/key/add/ReadVariableOp¢Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp¢Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOp¢Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp¢Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOp¢Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp¢@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp¢Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp¢@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp¢Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp
$token_and_position_embedding_3/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_3/Shape»
2token_and_position_embedding_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ24
2token_and_position_embedding_3/strided_slice/stack¶
4token_and_position_embedding_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_3/strided_slice/stack_1¶
4token_and_position_embedding_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_3/strided_slice/stack_2
,token_and_position_embedding_3/strided_sliceStridedSlice-token_and_position_embedding_3/Shape:output:0;token_and_position_embedding_3/strided_slice/stack:output:0=token_and_position_embedding_3/strided_slice/stack_1:output:0=token_and_position_embedding_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_3/strided_slice
*token_and_position_embedding_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_3/range/start
*token_and_position_embedding_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_3/range/delta
$token_and_position_embedding_3/rangeRange3token_and_position_embedding_3/range/start:output:05token_and_position_embedding_3/strided_slice:output:03token_and_position_embedding_3/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$token_and_position_embedding_3/rangeÊ
;token_and_position_embedding_3/embedding_7/embedding_lookupResourceGatherBtoken_and_position_embedding_3_embedding_7_embedding_lookup_196535-token_and_position_embedding_3/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_7/embedding_lookup/196535*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02=
;token_and_position_embedding_3/embedding_7/embedding_lookup
Dtoken_and_position_embedding_3/embedding_7/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_3/embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_7/embedding_lookup/196535*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2F
Dtoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity
Ftoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2H
Ftoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1·
/token_and_position_embedding_3/embedding_6/CastCastinputs_0*

DstT0*

SrcT0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 21
/token_and_position_embedding_3/embedding_6/CastÖ
;token_and_position_embedding_3/embedding_6/embedding_lookupResourceGatherBtoken_and_position_embedding_3_embedding_6_embedding_lookup_1965413token_and_position_embedding_3/embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_6/embedding_lookup/196541*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype02=
;token_and_position_embedding_3/embedding_6/embedding_lookup
Dtoken_and_position_embedding_3/embedding_6/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_3/embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_6/embedding_lookup/196541*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2F
Dtoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity£
Ftoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2H
Ftoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1«
"token_and_position_embedding_3/addAddV2Otoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2$
"token_and_position_embedding_3/add
conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_6/conv1d/ExpandDims/dimÓ
conv1d_6/conv1d/ExpandDims
ExpandDims&token_and_position_embedding_3/add:z:0'conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv1d_6/conv1d/ExpandDimsÓ
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02-
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_6/conv1d/ExpandDims_1/dimÛ
conv1d_6/conv1d/ExpandDims_1
ExpandDims3conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_6/conv1d/ExpandDims_1Ü
conv1d_6/conv1dConv2D#conv1d_6/conv1d/ExpandDims:output:0%conv1d_6/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
conv1d_6/conv1d¯
conv1d_6/conv1d/SqueezeSqueezeconv1d_6/conv1d:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_6/conv1d/Squeeze§
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_6/BiasAdd/ReadVariableOp²
conv1d_6/BiasAddBiasAdd conv1d_6/conv1d/Squeeze:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv1d_6/BiasAddy
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv1d_6/Relu
"average_pooling1d_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_9/ExpandDims/dimÔ
average_pooling1d_9/ExpandDims
ExpandDimsconv1d_6/Relu:activations:0+average_pooling1d_9/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2 
average_pooling1d_9/ExpandDimså
average_pooling1d_9/AvgPoolAvgPool'average_pooling1d_9/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_9/AvgPool¹
average_pooling1d_9/SqueezeSqueeze$average_pooling1d_9/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims
2
average_pooling1d_9/Squeeze
conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_7/conv1d/ExpandDims/dimÐ
conv1d_7/conv1d/ExpandDims
ExpandDims$average_pooling1d_9/Squeeze:output:0'conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv1d_7/conv1d/ExpandDimsÓ
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02-
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_7/conv1d/ExpandDims_1/dimÛ
conv1d_7/conv1d/ExpandDims_1
ExpandDims3conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d_7/conv1d/ExpandDims_1Û
conv1d_7/conv1dConv2D#conv1d_7/conv1d/ExpandDims:output:0%conv1d_7/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv1d_7/conv1d®
conv1d_7/conv1d/SqueezeSqueezeconv1d_7/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_7/conv1d/Squeeze§
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_7/BiasAdd/ReadVariableOp±
conv1d_7/BiasAddBiasAdd conv1d_7/conv1d/Squeeze:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv1d_7/BiasAddx
conv1d_7/ReluReluconv1d_7/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv1d_7/Relu
#average_pooling1d_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_11/ExpandDims/dimâ
average_pooling1d_11/ExpandDims
ExpandDims&token_and_position_embedding_3/add:z:0,average_pooling1d_11/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2!
average_pooling1d_11/ExpandDimsé
average_pooling1d_11/AvgPoolAvgPool(average_pooling1d_11/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
ksize	
¬*
paddingVALID*
strides	
¬2
average_pooling1d_11/AvgPool»
average_pooling1d_11/SqueezeSqueeze%average_pooling1d_11/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
squeeze_dims
2
average_pooling1d_11/Squeeze
#average_pooling1d_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_10/ExpandDims/dimÖ
average_pooling1d_10/ExpandDims
ExpandDimsconv1d_7/Relu:activations:0,average_pooling1d_10/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
average_pooling1d_10/ExpandDimsç
average_pooling1d_10/AvgPoolAvgPool(average_pooling1d_10/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
ksize

*
paddingVALID*
strides

2
average_pooling1d_10/AvgPool»
average_pooling1d_10/SqueezeSqueeze%average_pooling1d_10/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
squeeze_dims
2
average_pooling1d_10/SqueezeÔ
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
#batch_normalization_6/batchnorm/mulÛ
%batch_normalization_6/batchnorm/mul_1Mul%average_pooling1d_10/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2'
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
:ÿÿÿÿÿÿÿÿÿB 2'
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
#batch_normalization_7/batchnorm/mulÛ
%batch_normalization_7/batchnorm/mul_1Mul%average_pooling1d_11/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2'
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
:ÿÿÿÿÿÿÿÿÿB 2'
%batch_normalization_7/batchnorm/add_1«
	add_3/addAddV2)batch_normalization_6/batchnorm/add_1:z:0)batch_normalization_7/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
	add_3/add¹
Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_7_multi_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpÐ
>transformer_block_7/multi_head_attention_7/query/einsum/EinsumEinsumadd_3/add:z:0Utransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2@
>transformer_block_7/multi_head_attention_7/query/einsum/Einsum
Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOpReadVariableOpLtransformer_block_7_multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOpÅ
4transformer_block_7/multi_head_attention_7/query/addAddV2Gtransformer_block_7/multi_head_attention_7/query/einsum/Einsum:output:0Ktransformer_block_7/multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 26
4transformer_block_7/multi_head_attention_7/query/add³
Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_7_multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpÊ
<transformer_block_7/multi_head_attention_7/key/einsum/EinsumEinsumadd_3/add:z:0Stransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2>
<transformer_block_7/multi_head_attention_7/key/einsum/Einsum
Atransformer_block_7/multi_head_attention_7/key/add/ReadVariableOpReadVariableOpJtransformer_block_7_multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_7/multi_head_attention_7/key/add/ReadVariableOp½
2transformer_block_7/multi_head_attention_7/key/addAddV2Etransformer_block_7/multi_head_attention_7/key/einsum/Einsum:output:0Itransformer_block_7/multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 24
2transformer_block_7/multi_head_attention_7/key/add¹
Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_7_multi_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpÐ
>transformer_block_7/multi_head_attention_7/value/einsum/EinsumEinsumadd_3/add:z:0Utransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2@
>transformer_block_7/multi_head_attention_7/value/einsum/Einsum
Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOpReadVariableOpLtransformer_block_7_multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOpÅ
4transformer_block_7/multi_head_attention_7/value/addAddV2Gtransformer_block_7/multi_head_attention_7/value/einsum/Einsum:output:0Ktransformer_block_7/multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 26
4transformer_block_7/multi_head_attention_7/value/add©
0transformer_block_7/multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>22
0transformer_block_7/multi_head_attention_7/Mul/y
.transformer_block_7/multi_head_attention_7/MulMul8transformer_block_7/multi_head_attention_7/query/add:z:09transformer_block_7/multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 20
.transformer_block_7/multi_head_attention_7/MulÌ
8transformer_block_7/multi_head_attention_7/einsum/EinsumEinsum6transformer_block_7/multi_head_attention_7/key/add:z:02transformer_block_7/multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
equationaecd,abcd->acbe2:
8transformer_block_7/multi_head_attention_7/einsum/Einsum
:transformer_block_7/multi_head_attention_7/softmax/SoftmaxSoftmaxAtransformer_block_7/multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2<
:transformer_block_7/multi_head_attention_7/softmax/Softmax
;transformer_block_7/multi_head_attention_7/dropout/IdentityIdentityDtransformer_block_7/multi_head_attention_7/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2=
;transformer_block_7/multi_head_attention_7/dropout/Identityä
:transformer_block_7/multi_head_attention_7/einsum_1/EinsumEinsumDtransformer_block_7/multi_head_attention_7/dropout/Identity:output:08transformer_block_7/multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationacbe,aecd->abcd2<
:transformer_block_7/multi_head_attention_7/einsum_1/EinsumÚ
Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_7_multi_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp£
Itransformer_block_7/multi_head_attention_7/attention_output/einsum/EinsumEinsumCtransformer_block_7/multi_head_attention_7/einsum_1/Einsum:output:0`transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabcd,cde->abe2K
Itransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum´
Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_7_multi_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpí
?transformer_block_7/multi_head_attention_7/attention_output/addAddV2Rtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum:output:0Vtransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2A
?transformer_block_7/multi_head_attention_7/attention_output/addÙ
'transformer_block_7/dropout_20/IdentityIdentityCtransformer_block_7/multi_head_attention_7/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2)
'transformer_block_7/dropout_20/Identity²
transformer_block_7/addAddV2add_3/add:z:00transformer_block_7/dropout_20/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
transformer_block_7/addà
Itransformer_block_7/layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_7/layer_normalization_14/moments/mean/reduction_indices²
7transformer_block_7/layer_normalization_14/moments/meanMeantransformer_block_7/add:z:0Rtransformer_block_7/layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(29
7transformer_block_7/layer_normalization_14/moments/mean
?transformer_block_7/layer_normalization_14/moments/StopGradientStopGradient@transformer_block_7/layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2A
?transformer_block_7/layer_normalization_14/moments/StopGradient¾
Dtransformer_block_7/layer_normalization_14/moments/SquaredDifferenceSquaredDifferencetransformer_block_7/add:z:0Htransformer_block_7/layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2F
Dtransformer_block_7/layer_normalization_14/moments/SquaredDifferenceè
Mtransformer_block_7/layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_7/layer_normalization_14/moments/variance/reduction_indicesë
;transformer_block_7/layer_normalization_14/moments/varianceMeanHtransformer_block_7/layer_normalization_14/moments/SquaredDifference:z:0Vtransformer_block_7/layer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2=
;transformer_block_7/layer_normalization_14/moments/variance½
:transformer_block_7/layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_7/layer_normalization_14/batchnorm/add/y¾
8transformer_block_7/layer_normalization_14/batchnorm/addAddV2Dtransformer_block_7/layer_normalization_14/moments/variance:output:0Ctransformer_block_7/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2:
8transformer_block_7/layer_normalization_14/batchnorm/addõ
:transformer_block_7/layer_normalization_14/batchnorm/RsqrtRsqrt<transformer_block_7/layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2<
:transformer_block_7/layer_normalization_14/batchnorm/Rsqrt
Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_7_layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpÂ
8transformer_block_7/layer_normalization_14/batchnorm/mulMul>transformer_block_7/layer_normalization_14/batchnorm/Rsqrt:y:0Otransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2:
8transformer_block_7/layer_normalization_14/batchnorm/mul
:transformer_block_7/layer_normalization_14/batchnorm/mul_1Multransformer_block_7/add:z:0<transformer_block_7/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2<
:transformer_block_7/layer_normalization_14/batchnorm/mul_1µ
:transformer_block_7/layer_normalization_14/batchnorm/mul_2Mul@transformer_block_7/layer_normalization_14/moments/mean:output:0<transformer_block_7/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2<
:transformer_block_7/layer_normalization_14/batchnorm/mul_2
Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_7_layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp¾
8transformer_block_7/layer_normalization_14/batchnorm/subSubKtransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp:value:0>transformer_block_7/layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2:
8transformer_block_7/layer_normalization_14/batchnorm/subµ
:transformer_block_7/layer_normalization_14/batchnorm/add_1AddV2>transformer_block_7/layer_normalization_14/batchnorm/mul_1:z:0<transformer_block_7/layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2<
:transformer_block_7/layer_normalization_14/batchnorm/add_1
Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_7_sequential_7_dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02D
Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp¾
8transformer_block_7/sequential_7/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_7/sequential_7/dense_23/Tensordot/axesÅ
8transformer_block_7/sequential_7/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_7/sequential_7/dense_23/Tensordot/freeä
9transformer_block_7/sequential_7/dense_23/Tensordot/ShapeShape>transformer_block_7/layer_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_23/Tensordot/ShapeÈ
Atransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axis£
<transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2GatherV2Btransformer_block_7/sequential_7/dense_23/Tensordot/Shape:output:0Atransformer_block_7/sequential_7/dense_23/Tensordot/free:output:0Jtransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2Ì
Ctransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axis©
>transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1GatherV2Btransformer_block_7/sequential_7/dense_23/Tensordot/Shape:output:0Atransformer_block_7/sequential_7/dense_23/Tensordot/axes:output:0Ltransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1À
9transformer_block_7/sequential_7/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_7/sequential_7/dense_23/Tensordot/Const¨
8transformer_block_7/sequential_7/dense_23/Tensordot/ProdProdEtransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2:output:0Btransformer_block_7/sequential_7/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_7/sequential_7/dense_23/Tensordot/ProdÄ
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_1°
:transformer_block_7/sequential_7/dense_23/Tensordot/Prod_1ProdGtransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1:output:0Dtransformer_block_7/sequential_7/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_7/sequential_7/dense_23/Tensordot/Prod_1Ä
?transformer_block_7/sequential_7/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_7/sequential_7/dense_23/Tensordot/concat/axis
:transformer_block_7/sequential_7/dense_23/Tensordot/concatConcatV2Atransformer_block_7/sequential_7/dense_23/Tensordot/free:output:0Atransformer_block_7/sequential_7/dense_23/Tensordot/axes:output:0Htransformer_block_7/sequential_7/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_7/sequential_7/dense_23/Tensordot/concat´
9transformer_block_7/sequential_7/dense_23/Tensordot/stackPackAtransformer_block_7/sequential_7/dense_23/Tensordot/Prod:output:0Ctransformer_block_7/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_23/Tensordot/stackÆ
=transformer_block_7/sequential_7/dense_23/Tensordot/transpose	Transpose>transformer_block_7/layer_normalization_14/batchnorm/add_1:z:0Ctransformer_block_7/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2?
=transformer_block_7/sequential_7/dense_23/Tensordot/transposeÇ
;transformer_block_7/sequential_7/dense_23/Tensordot/ReshapeReshapeAtransformer_block_7/sequential_7/dense_23/Tensordot/transpose:y:0Btransformer_block_7/sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2=
;transformer_block_7/sequential_7/dense_23/Tensordot/ReshapeÆ
:transformer_block_7/sequential_7/dense_23/Tensordot/MatMulMatMulDtransformer_block_7/sequential_7/dense_23/Tensordot/Reshape:output:0Jtransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2<
:transformer_block_7/sequential_7/dense_23/Tensordot/MatMulÄ
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2=
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_2È
Atransformer_block_7/sequential_7/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_7/sequential_7/dense_23/Tensordot/concat_1/axis
<transformer_block_7/sequential_7/dense_23/Tensordot/concat_1ConcatV2Etransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2:output:0Dtransformer_block_7/sequential_7/dense_23/Tensordot/Const_2:output:0Jtransformer_block_7/sequential_7/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_7/sequential_7/dense_23/Tensordot/concat_1¸
3transformer_block_7/sequential_7/dense_23/TensordotReshapeDtransformer_block_7/sequential_7/dense_23/Tensordot/MatMul:product:0Etransformer_block_7/sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@25
3transformer_block_7/sequential_7/dense_23/Tensordot
@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_7_sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp¯
1transformer_block_7/sequential_7/dense_23/BiasAddBiasAdd<transformer_block_7/sequential_7/dense_23/Tensordot:output:0Htransformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@23
1transformer_block_7/sequential_7/dense_23/BiasAddÚ
.transformer_block_7/sequential_7/dense_23/ReluRelu:transformer_block_7/sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@20
.transformer_block_7/sequential_7/dense_23/Relu
Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_7_sequential_7_dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02D
Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp¾
8transformer_block_7/sequential_7/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_7/sequential_7/dense_24/Tensordot/axesÅ
8transformer_block_7/sequential_7/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_7/sequential_7/dense_24/Tensordot/freeâ
9transformer_block_7/sequential_7/dense_24/Tensordot/ShapeShape<transformer_block_7/sequential_7/dense_23/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_24/Tensordot/ShapeÈ
Atransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axis£
<transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2GatherV2Btransformer_block_7/sequential_7/dense_24/Tensordot/Shape:output:0Atransformer_block_7/sequential_7/dense_24/Tensordot/free:output:0Jtransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2Ì
Ctransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axis©
>transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1GatherV2Btransformer_block_7/sequential_7/dense_24/Tensordot/Shape:output:0Atransformer_block_7/sequential_7/dense_24/Tensordot/axes:output:0Ltransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1À
9transformer_block_7/sequential_7/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_7/sequential_7/dense_24/Tensordot/Const¨
8transformer_block_7/sequential_7/dense_24/Tensordot/ProdProdEtransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2:output:0Btransformer_block_7/sequential_7/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_7/sequential_7/dense_24/Tensordot/ProdÄ
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_1°
:transformer_block_7/sequential_7/dense_24/Tensordot/Prod_1ProdGtransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1:output:0Dtransformer_block_7/sequential_7/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_7/sequential_7/dense_24/Tensordot/Prod_1Ä
?transformer_block_7/sequential_7/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_7/sequential_7/dense_24/Tensordot/concat/axis
:transformer_block_7/sequential_7/dense_24/Tensordot/concatConcatV2Atransformer_block_7/sequential_7/dense_24/Tensordot/free:output:0Atransformer_block_7/sequential_7/dense_24/Tensordot/axes:output:0Htransformer_block_7/sequential_7/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_7/sequential_7/dense_24/Tensordot/concat´
9transformer_block_7/sequential_7/dense_24/Tensordot/stackPackAtransformer_block_7/sequential_7/dense_24/Tensordot/Prod:output:0Ctransformer_block_7/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_24/Tensordot/stackÄ
=transformer_block_7/sequential_7/dense_24/Tensordot/transpose	Transpose<transformer_block_7/sequential_7/dense_23/Relu:activations:0Ctransformer_block_7/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2?
=transformer_block_7/sequential_7/dense_24/Tensordot/transposeÇ
;transformer_block_7/sequential_7/dense_24/Tensordot/ReshapeReshapeAtransformer_block_7/sequential_7/dense_24/Tensordot/transpose:y:0Btransformer_block_7/sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2=
;transformer_block_7/sequential_7/dense_24/Tensordot/ReshapeÆ
:transformer_block_7/sequential_7/dense_24/Tensordot/MatMulMatMulDtransformer_block_7/sequential_7/dense_24/Tensordot/Reshape:output:0Jtransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2<
:transformer_block_7/sequential_7/dense_24/Tensordot/MatMulÄ
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_2È
Atransformer_block_7/sequential_7/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_7/sequential_7/dense_24/Tensordot/concat_1/axis
<transformer_block_7/sequential_7/dense_24/Tensordot/concat_1ConcatV2Etransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2:output:0Dtransformer_block_7/sequential_7/dense_24/Tensordot/Const_2:output:0Jtransformer_block_7/sequential_7/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_7/sequential_7/dense_24/Tensordot/concat_1¸
3transformer_block_7/sequential_7/dense_24/TensordotReshapeDtransformer_block_7/sequential_7/dense_24/Tensordot/MatMul:product:0Etransformer_block_7/sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 25
3transformer_block_7/sequential_7/dense_24/Tensordot
@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_7_sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp¯
1transformer_block_7/sequential_7/dense_24/BiasAddBiasAdd<transformer_block_7/sequential_7/dense_24/Tensordot:output:0Htransformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 23
1transformer_block_7/sequential_7/dense_24/BiasAddÐ
'transformer_block_7/dropout_21/IdentityIdentity:transformer_block_7/sequential_7/dense_24/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2)
'transformer_block_7/dropout_21/Identityç
transformer_block_7/add_1AddV2>transformer_block_7/layer_normalization_14/batchnorm/add_1:z:00transformer_block_7/dropout_21/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
transformer_block_7/add_1à
Itransformer_block_7/layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_7/layer_normalization_15/moments/mean/reduction_indices´
7transformer_block_7/layer_normalization_15/moments/meanMeantransformer_block_7/add_1:z:0Rtransformer_block_7/layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(29
7transformer_block_7/layer_normalization_15/moments/mean
?transformer_block_7/layer_normalization_15/moments/StopGradientStopGradient@transformer_block_7/layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2A
?transformer_block_7/layer_normalization_15/moments/StopGradientÀ
Dtransformer_block_7/layer_normalization_15/moments/SquaredDifferenceSquaredDifferencetransformer_block_7/add_1:z:0Htransformer_block_7/layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2F
Dtransformer_block_7/layer_normalization_15/moments/SquaredDifferenceè
Mtransformer_block_7/layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_7/layer_normalization_15/moments/variance/reduction_indicesë
;transformer_block_7/layer_normalization_15/moments/varianceMeanHtransformer_block_7/layer_normalization_15/moments/SquaredDifference:z:0Vtransformer_block_7/layer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2=
;transformer_block_7/layer_normalization_15/moments/variance½
:transformer_block_7/layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_7/layer_normalization_15/batchnorm/add/y¾
8transformer_block_7/layer_normalization_15/batchnorm/addAddV2Dtransformer_block_7/layer_normalization_15/moments/variance:output:0Ctransformer_block_7/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2:
8transformer_block_7/layer_normalization_15/batchnorm/addõ
:transformer_block_7/layer_normalization_15/batchnorm/RsqrtRsqrt<transformer_block_7/layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2<
:transformer_block_7/layer_normalization_15/batchnorm/Rsqrt
Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_7_layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpÂ
8transformer_block_7/layer_normalization_15/batchnorm/mulMul>transformer_block_7/layer_normalization_15/batchnorm/Rsqrt:y:0Otransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2:
8transformer_block_7/layer_normalization_15/batchnorm/mul
:transformer_block_7/layer_normalization_15/batchnorm/mul_1Multransformer_block_7/add_1:z:0<transformer_block_7/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2<
:transformer_block_7/layer_normalization_15/batchnorm/mul_1µ
:transformer_block_7/layer_normalization_15/batchnorm/mul_2Mul@transformer_block_7/layer_normalization_15/moments/mean:output:0<transformer_block_7/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2<
:transformer_block_7/layer_normalization_15/batchnorm/mul_2
Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_7_layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp¾
8transformer_block_7/layer_normalization_15/batchnorm/subSubKtransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp:value:0>transformer_block_7/layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2:
8transformer_block_7/layer_normalization_15/batchnorm/subµ
:transformer_block_7/layer_normalization_15/batchnorm/add_1AddV2>transformer_block_7/layer_normalization_15/batchnorm/mul_1:z:0<transformer_block_7/layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2<
:transformer_block_7/layer_normalization_15/batchnorm/add_1s
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
flatten_3/Const¾
flatten_3/ReshapeReshape>transformer_block_7/layer_normalization_15/batchnorm/add_1:z:0flatten_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
flatten_3/Reshapex
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axis¾
concatenate_3/concatConcatV2flatten_3/Reshape:output:0inputs_1"concatenate_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
concatenate_3/concat©
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	È@*
dtype02 
dense_25/MatMul/ReadVariableOp¥
dense_25/MatMulMatMulconcatenate_3/concat:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_25/MatMul§
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_25/BiasAdd/ReadVariableOp¥
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_25/BiasAdds
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_25/Relu
dropout_22/IdentityIdentitydense_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_22/Identity¨
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_26/MatMul/ReadVariableOp¤
dense_26/MatMulMatMuldropout_22/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_26/MatMul§
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_26/BiasAdd/ReadVariableOp¥
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_26/BiasAdds
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_26/Relu
dropout_23/IdentityIdentitydense_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_23/Identity¨
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_27/MatMul/ReadVariableOp¤
dense_27/MatMulMatMuldropout_23/Identity:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_27/MatMul§
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_27/BiasAdd/ReadVariableOp¥
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_27/BiasAdd
IdentityIdentitydense_27/BiasAdd:output:0/^batch_normalization_6/batchnorm/ReadVariableOp1^batch_normalization_6/batchnorm/ReadVariableOp_11^batch_normalization_6/batchnorm/ReadVariableOp_23^batch_normalization_6/batchnorm/mul/ReadVariableOp/^batch_normalization_7/batchnorm/ReadVariableOp1^batch_normalization_7/batchnorm/ReadVariableOp_11^batch_normalization_7/batchnorm/ReadVariableOp_23^batch_normalization_7/batchnorm/mul/ReadVariableOp ^conv1d_6/BiasAdd/ReadVariableOp,^conv1d_6/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_7/BiasAdd/ReadVariableOp,^conv1d_7/conv1d/ExpandDims_1/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp<^token_and_position_embedding_3/embedding_6/embedding_lookup<^token_and_position_embedding_3/embedding_7/embedding_lookupD^transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpH^transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpD^transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpH^transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpO^transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpY^transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_7/multi_head_attention_7/key/add/ReadVariableOpL^transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpD^transformer_block_7/multi_head_attention_7/query/add/ReadVariableOpN^transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpD^transformer_block_7/multi_head_attention_7/value/add/ReadVariableOpN^transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpA^transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpC^transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpA^transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpC^transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Í
_input_shapes»
¸:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2`
.batch_normalization_6/batchnorm/ReadVariableOp.batch_normalization_6/batchnorm/ReadVariableOp2d
0batch_normalization_6/batchnorm/ReadVariableOp_10batch_normalization_6/batchnorm/ReadVariableOp_12d
0batch_normalization_6/batchnorm/ReadVariableOp_20batch_normalization_6/batchnorm/ReadVariableOp_22h
2batch_normalization_6/batchnorm/mul/ReadVariableOp2batch_normalization_6/batchnorm/mul/ReadVariableOp2`
.batch_normalization_7/batchnorm/ReadVariableOp.batch_normalization_7/batchnorm/ReadVariableOp2d
0batch_normalization_7/batchnorm/ReadVariableOp_10batch_normalization_7/batchnorm/ReadVariableOp_12d
0batch_normalization_7/batchnorm/ReadVariableOp_20batch_normalization_7/batchnorm/ReadVariableOp_22h
2batch_normalization_7/batchnorm/mul/ReadVariableOp2batch_normalization_7/batchnorm/mul/ReadVariableOp2B
conv1d_6/BiasAdd/ReadVariableOpconv1d_6/BiasAdd/ReadVariableOp2Z
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_7/BiasAdd/ReadVariableOpconv1d_7/BiasAdd/ReadVariableOp2Z
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2z
;token_and_position_embedding_3/embedding_6/embedding_lookup;token_and_position_embedding_3/embedding_6/embedding_lookup2z
;token_and_position_embedding_3/embedding_7/embedding_lookup;token_and_position_embedding_3/embedding_7/embedding_lookup2
Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpCtransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp2
Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpGtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp2
Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpCtransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp2
Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpGtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp2 
Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpNtransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp2´
Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp2
Atransformer_block_7/multi_head_attention_7/key/add/ReadVariableOpAtransformer_block_7/multi_head_attention_7/key/add/ReadVariableOp2
Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpKtransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp2
Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOpCtransformer_block_7/multi_head_attention_7/query/add/ReadVariableOp2
Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpMtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp2
Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOpCtransformer_block_7/multi_head_attention_7/value/add/ReadVariableOp2
Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpMtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp2
@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp2
Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpBtransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp2
@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp2
Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpBtransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp:S O
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
É
d
F__inference_dropout_22_layer_call_and_return_conditional_losses_197755

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
D__inference_dense_25_layer_call_and_return_conditional_losses_195556

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	È@*
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
:ÿÿÿÿÿÿÿÿÿÈ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
¡
F
*__inference_flatten_3_layer_call_fn_197705

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
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1955212
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿB :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
ì
©
6__inference_batch_normalization_7_layer_call_fn_197238

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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1946112
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
F__inference_dropout_23_layer_call_and_return_conditional_losses_197802

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

e
F__inference_dropout_23_layer_call_and_return_conditional_losses_195641

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

seed*2&
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

P
4__inference_average_pooling1d_9_layer_call_fn_194345

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
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_1943392
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
¼0
È
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_195060

inputs
assignmovingavg_195035
assignmovingavg_1_195041)
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
:ÿÿÿÿÿÿÿÿÿB 2
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
loc:@AssignMovingAvg/195035*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_195035*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/195035*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/195035*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_195035AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/195035*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/195041*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_195041*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/195041*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/195041*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_195041AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/195041*
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
:ÿÿÿÿÿÿÿÿÿB 2
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
:ÿÿÿÿÿÿÿÿÿB 2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
ó
~
)__inference_conv1d_7_layer_call_fn_197005

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
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_1949162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
J
¯
H__inference_sequential_7_layer_call_and_return_conditional_losses_197945

inputs.
*dense_23_tensordot_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource.
*dense_24_tensordot_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource
identity¢dense_23/BiasAdd/ReadVariableOp¢!dense_23/Tensordot/ReadVariableOp¢dense_24/BiasAdd/ReadVariableOp¢!dense_24/Tensordot/ReadVariableOp±
!dense_23/Tensordot/ReadVariableOpReadVariableOp*dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02#
!dense_23/Tensordot/ReadVariableOp|
dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_23/Tensordot/axes
dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_23/Tensordot/freej
dense_23/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_23/Tensordot/Shape
 dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_23/Tensordot/GatherV2/axisþ
dense_23/Tensordot/GatherV2GatherV2!dense_23/Tensordot/Shape:output:0 dense_23/Tensordot/free:output:0)dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_23/Tensordot/GatherV2
"dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_23/Tensordot/GatherV2_1/axis
dense_23/Tensordot/GatherV2_1GatherV2!dense_23/Tensordot/Shape:output:0 dense_23/Tensordot/axes:output:0+dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_23/Tensordot/GatherV2_1~
dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_23/Tensordot/Const¤
dense_23/Tensordot/ProdProd$dense_23/Tensordot/GatherV2:output:0!dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_23/Tensordot/Prod
dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_23/Tensordot/Const_1¬
dense_23/Tensordot/Prod_1Prod&dense_23/Tensordot/GatherV2_1:output:0#dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_23/Tensordot/Prod_1
dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_23/Tensordot/concat/axisÝ
dense_23/Tensordot/concatConcatV2 dense_23/Tensordot/free:output:0 dense_23/Tensordot/axes:output:0'dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/concat°
dense_23/Tensordot/stackPack dense_23/Tensordot/Prod:output:0"dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/stack«
dense_23/Tensordot/transpose	Transposeinputs"dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dense_23/Tensordot/transposeÃ
dense_23/Tensordot/ReshapeReshape dense_23/Tensordot/transpose:y:0!dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_23/Tensordot/ReshapeÂ
dense_23/Tensordot/MatMulMatMul#dense_23/Tensordot/Reshape:output:0)dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_23/Tensordot/MatMul
dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_23/Tensordot/Const_2
 dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_23/Tensordot/concat_1/axisê
dense_23/Tensordot/concat_1ConcatV2$dense_23/Tensordot/GatherV2:output:0#dense_23/Tensordot/Const_2:output:0)dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/concat_1´
dense_23/TensordotReshape#dense_23/Tensordot/MatMul:product:0$dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2
dense_23/Tensordot§
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_23/BiasAdd/ReadVariableOp«
dense_23/BiasAddBiasAdddense_23/Tensordot:output:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2
dense_23/BiasAddw
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2
dense_23/Relu±
!dense_24/Tensordot/ReadVariableOpReadVariableOp*dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02#
!dense_24/Tensordot/ReadVariableOp|
dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_24/Tensordot/axes
dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_24/Tensordot/free
dense_24/Tensordot/ShapeShapedense_23/Relu:activations:0*
T0*
_output_shapes
:2
dense_24/Tensordot/Shape
 dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_24/Tensordot/GatherV2/axisþ
dense_24/Tensordot/GatherV2GatherV2!dense_24/Tensordot/Shape:output:0 dense_24/Tensordot/free:output:0)dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_24/Tensordot/GatherV2
"dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_24/Tensordot/GatherV2_1/axis
dense_24/Tensordot/GatherV2_1GatherV2!dense_24/Tensordot/Shape:output:0 dense_24/Tensordot/axes:output:0+dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_24/Tensordot/GatherV2_1~
dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const¤
dense_24/Tensordot/ProdProd$dense_24/Tensordot/GatherV2:output:0!dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_24/Tensordot/Prod
dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const_1¬
dense_24/Tensordot/Prod_1Prod&dense_24/Tensordot/GatherV2_1:output:0#dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_24/Tensordot/Prod_1
dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_24/Tensordot/concat/axisÝ
dense_24/Tensordot/concatConcatV2 dense_24/Tensordot/free:output:0 dense_24/Tensordot/axes:output:0'dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/concat°
dense_24/Tensordot/stackPack dense_24/Tensordot/Prod:output:0"dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/stackÀ
dense_24/Tensordot/transpose	Transposedense_23/Relu:activations:0"dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2
dense_24/Tensordot/transposeÃ
dense_24/Tensordot/ReshapeReshape dense_24/Tensordot/transpose:y:0!dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_24/Tensordot/ReshapeÂ
dense_24/Tensordot/MatMulMatMul#dense_24/Tensordot/Reshape:output:0)dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_24/Tensordot/MatMul
dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const_2
 dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_24/Tensordot/concat_1/axisê
dense_24/Tensordot/concat_1ConcatV2$dense_24/Tensordot/GatherV2:output:0#dense_24/Tensordot/Const_2:output:0)dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/concat_1´
dense_24/TensordotReshape#dense_24/Tensordot/MatMul:product:0$dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dense_24/Tensordot§
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_24/BiasAdd/ReadVariableOp«
dense_24/BiasAddBiasAdddense_24/Tensordot:output:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dense_24/BiasAddý
IdentityIdentitydense_24/BiasAdd:output:0 ^dense_23/BiasAdd/ReadVariableOp"^dense_23/Tensordot/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp"^dense_24/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2F
!dense_23/Tensordot/ReadVariableOp!dense_23/Tensordot/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2F
!dense_24/Tensordot/ReadVariableOp!dense_24/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs

Q
5__inference_average_pooling1d_10_layer_call_fn_194360

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
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_1943542
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
Çë
¤&
C__inference_model_3_layer_call_and_return_conditional_losses_196523
inputs_0
inputs_1F
Btoken_and_position_embedding_3_embedding_7_embedding_lookup_196225F
Btoken_and_position_embedding_3_embedding_6_embedding_lookup_1962318
4conv1d_6_conv1d_expanddims_1_readvariableop_resource,
(conv1d_6_biasadd_readvariableop_resource8
4conv1d_7_conv1d_expanddims_1_readvariableop_resource,
(conv1d_7_biasadd_readvariableop_resource0
,batch_normalization_6_assignmovingavg_1962812
.batch_normalization_6_assignmovingavg_1_196287?
;batch_normalization_6_batchnorm_mul_readvariableop_resource;
7batch_normalization_6_batchnorm_readvariableop_resource0
,batch_normalization_7_assignmovingavg_1963132
.batch_normalization_7_assignmovingavg_1_196319?
;batch_normalization_7_batchnorm_mul_readvariableop_resource;
7batch_normalization_7_batchnorm_readvariableop_resourceZ
Vtransformer_block_7_multi_head_attention_7_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_7_multi_head_attention_7_query_add_readvariableop_resourceX
Ttransformer_block_7_multi_head_attention_7_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_7_multi_head_attention_7_key_add_readvariableop_resourceZ
Vtransformer_block_7_multi_head_attention_7_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_7_multi_head_attention_7_value_add_readvariableop_resourcee
atransformer_block_7_multi_head_attention_7_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_7_multi_head_attention_7_attention_output_add_readvariableop_resourceT
Ptransformer_block_7_layer_normalization_14_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_7_layer_normalization_14_batchnorm_readvariableop_resourceO
Ktransformer_block_7_sequential_7_dense_23_tensordot_readvariableop_resourceM
Itransformer_block_7_sequential_7_dense_23_biasadd_readvariableop_resourceO
Ktransformer_block_7_sequential_7_dense_24_tensordot_readvariableop_resourceM
Itransformer_block_7_sequential_7_dense_24_biasadd_readvariableop_resourceT
Ptransformer_block_7_layer_normalization_15_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_7_layer_normalization_15_batchnorm_readvariableop_resource+
'dense_25_matmul_readvariableop_resource,
(dense_25_biasadd_readvariableop_resource+
'dense_26_matmul_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource
identity¢9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_6/AssignMovingAvg/ReadVariableOp¢;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_6/batchnorm/ReadVariableOp¢2batch_normalization_6/batchnorm/mul/ReadVariableOp¢9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_7/AssignMovingAvg/ReadVariableOp¢;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_7/batchnorm/ReadVariableOp¢2batch_normalization_7/batchnorm/mul/ReadVariableOp¢conv1d_6/BiasAdd/ReadVariableOp¢+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp¢conv1d_7/BiasAdd/ReadVariableOp¢+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp¢dense_25/BiasAdd/ReadVariableOp¢dense_25/MatMul/ReadVariableOp¢dense_26/BiasAdd/ReadVariableOp¢dense_26/MatMul/ReadVariableOp¢dense_27/BiasAdd/ReadVariableOp¢dense_27/MatMul/ReadVariableOp¢;token_and_position_embedding_3/embedding_6/embedding_lookup¢;token_and_position_embedding_3/embedding_7/embedding_lookup¢Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp¢Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp¢Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp¢Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp¢Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp¢Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp¢Atransformer_block_7/multi_head_attention_7/key/add/ReadVariableOp¢Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp¢Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOp¢Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp¢Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOp¢Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp¢@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp¢Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp¢@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp¢Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp
$token_and_position_embedding_3/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_3/Shape»
2token_and_position_embedding_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ24
2token_and_position_embedding_3/strided_slice/stack¶
4token_and_position_embedding_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_3/strided_slice/stack_1¶
4token_and_position_embedding_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_3/strided_slice/stack_2
,token_and_position_embedding_3/strided_sliceStridedSlice-token_and_position_embedding_3/Shape:output:0;token_and_position_embedding_3/strided_slice/stack:output:0=token_and_position_embedding_3/strided_slice/stack_1:output:0=token_and_position_embedding_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_3/strided_slice
*token_and_position_embedding_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_3/range/start
*token_and_position_embedding_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_3/range/delta
$token_and_position_embedding_3/rangeRange3token_and_position_embedding_3/range/start:output:05token_and_position_embedding_3/strided_slice:output:03token_and_position_embedding_3/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$token_and_position_embedding_3/rangeÊ
;token_and_position_embedding_3/embedding_7/embedding_lookupResourceGatherBtoken_and_position_embedding_3_embedding_7_embedding_lookup_196225-token_and_position_embedding_3/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_7/embedding_lookup/196225*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02=
;token_and_position_embedding_3/embedding_7/embedding_lookup
Dtoken_and_position_embedding_3/embedding_7/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_3/embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_7/embedding_lookup/196225*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2F
Dtoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity
Ftoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2H
Ftoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1·
/token_and_position_embedding_3/embedding_6/CastCastinputs_0*

DstT0*

SrcT0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 21
/token_and_position_embedding_3/embedding_6/CastÖ
;token_and_position_embedding_3/embedding_6/embedding_lookupResourceGatherBtoken_and_position_embedding_3_embedding_6_embedding_lookup_1962313token_and_position_embedding_3/embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_6/embedding_lookup/196231*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype02=
;token_and_position_embedding_3/embedding_6/embedding_lookup
Dtoken_and_position_embedding_3/embedding_6/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_3/embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_6/embedding_lookup/196231*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2F
Dtoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity£
Ftoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2H
Ftoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1«
"token_and_position_embedding_3/addAddV2Otoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2$
"token_and_position_embedding_3/add
conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_6/conv1d/ExpandDims/dimÓ
conv1d_6/conv1d/ExpandDims
ExpandDims&token_and_position_embedding_3/add:z:0'conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv1d_6/conv1d/ExpandDimsÓ
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02-
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_6/conv1d/ExpandDims_1/dimÛ
conv1d_6/conv1d/ExpandDims_1
ExpandDims3conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_6/conv1d/ExpandDims_1Ü
conv1d_6/conv1dConv2D#conv1d_6/conv1d/ExpandDims:output:0%conv1d_6/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
conv1d_6/conv1d¯
conv1d_6/conv1d/SqueezeSqueezeconv1d_6/conv1d:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_6/conv1d/Squeeze§
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_6/BiasAdd/ReadVariableOp²
conv1d_6/BiasAddBiasAdd conv1d_6/conv1d/Squeeze:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv1d_6/BiasAddy
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv1d_6/Relu
"average_pooling1d_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_9/ExpandDims/dimÔ
average_pooling1d_9/ExpandDims
ExpandDimsconv1d_6/Relu:activations:0+average_pooling1d_9/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2 
average_pooling1d_9/ExpandDimså
average_pooling1d_9/AvgPoolAvgPool'average_pooling1d_9/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_9/AvgPool¹
average_pooling1d_9/SqueezeSqueeze$average_pooling1d_9/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims
2
average_pooling1d_9/Squeeze
conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_7/conv1d/ExpandDims/dimÐ
conv1d_7/conv1d/ExpandDims
ExpandDims$average_pooling1d_9/Squeeze:output:0'conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv1d_7/conv1d/ExpandDimsÓ
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02-
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_7/conv1d/ExpandDims_1/dimÛ
conv1d_7/conv1d/ExpandDims_1
ExpandDims3conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d_7/conv1d/ExpandDims_1Û
conv1d_7/conv1dConv2D#conv1d_7/conv1d/ExpandDims:output:0%conv1d_7/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv1d_7/conv1d®
conv1d_7/conv1d/SqueezeSqueezeconv1d_7/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_7/conv1d/Squeeze§
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_7/BiasAdd/ReadVariableOp±
conv1d_7/BiasAddBiasAdd conv1d_7/conv1d/Squeeze:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv1d_7/BiasAddx
conv1d_7/ReluReluconv1d_7/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv1d_7/Relu
#average_pooling1d_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_11/ExpandDims/dimâ
average_pooling1d_11/ExpandDims
ExpandDims&token_and_position_embedding_3/add:z:0,average_pooling1d_11/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2!
average_pooling1d_11/ExpandDimsé
average_pooling1d_11/AvgPoolAvgPool(average_pooling1d_11/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
ksize	
¬*
paddingVALID*
strides	
¬2
average_pooling1d_11/AvgPool»
average_pooling1d_11/SqueezeSqueeze%average_pooling1d_11/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
squeeze_dims
2
average_pooling1d_11/Squeeze
#average_pooling1d_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_10/ExpandDims/dimÖ
average_pooling1d_10/ExpandDims
ExpandDimsconv1d_7/Relu:activations:0,average_pooling1d_10/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
average_pooling1d_10/ExpandDimsç
average_pooling1d_10/AvgPoolAvgPool(average_pooling1d_10/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
ksize

*
paddingVALID*
strides

2
average_pooling1d_10/AvgPool»
average_pooling1d_10/SqueezeSqueeze%average_pooling1d_10/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
squeeze_dims
2
average_pooling1d_10/Squeeze½
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_6/moments/mean/reduction_indicesô
"batch_normalization_6/moments/meanMean%average_pooling1d_10/Squeeze:output:0=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_6/moments/meanÂ
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_6/moments/StopGradient
/batch_normalization_6/moments/SquaredDifferenceSquaredDifference%average_pooling1d_10/Squeeze:output:03batch_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 21
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
31loc:@batch_normalization_6/AssignMovingAvg/196281*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_6/AssignMovingAvg/decayÕ
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_6_assignmovingavg_196281*
_output_shapes
: *
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOpß
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg/196281*
_output_shapes
: 2+
)batch_normalization_6/AssignMovingAvg/subÖ
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:04batch_normalization_6/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg/196281*
_output_shapes
: 2+
)batch_normalization_6/AssignMovingAvg/mul³
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_6_assignmovingavg_196281-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg/196281*
_output_shapes
 *
dtype02;
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_6/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/196287*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_6/AssignMovingAvg_1/decayÛ
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_6_assignmovingavg_1_196287*
_output_shapes
: *
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpé
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/196287*
_output_shapes
: 2-
+batch_normalization_6/AssignMovingAvg_1/subà
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:06batch_normalization_6/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/196287*
_output_shapes
: 2-
+batch_normalization_6/AssignMovingAvg_1/mul¿
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_6_assignmovingavg_1_196287/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/196287*
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
#batch_normalization_6/batchnorm/mulÛ
%batch_normalization_6/batchnorm/mul_1Mul%average_pooling1d_10/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2'
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
:ÿÿÿÿÿÿÿÿÿB 2'
%batch_normalization_6/batchnorm/add_1½
4batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_7/moments/mean/reduction_indicesô
"batch_normalization_7/moments/meanMean%average_pooling1d_11/Squeeze:output:0=batch_normalization_7/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_7/moments/meanÂ
*batch_normalization_7/moments/StopGradientStopGradient+batch_normalization_7/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_7/moments/StopGradient
/batch_normalization_7/moments/SquaredDifferenceSquaredDifference%average_pooling1d_11/Squeeze:output:03batch_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 21
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
31loc:@batch_normalization_7/AssignMovingAvg/196313*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_7/AssignMovingAvg/decayÕ
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_7_assignmovingavg_196313*
_output_shapes
: *
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOpß
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg/196313*
_output_shapes
: 2+
)batch_normalization_7/AssignMovingAvg/subÖ
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:04batch_normalization_7/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg/196313*
_output_shapes
: 2+
)batch_normalization_7/AssignMovingAvg/mul³
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_7_assignmovingavg_196313-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg/196313*
_output_shapes
 *
dtype02;
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_7/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/196319*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_7/AssignMovingAvg_1/decayÛ
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_7_assignmovingavg_1_196319*
_output_shapes
: *
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpé
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/196319*
_output_shapes
: 2-
+batch_normalization_7/AssignMovingAvg_1/subà
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:06batch_normalization_7/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/196319*
_output_shapes
: 2-
+batch_normalization_7/AssignMovingAvg_1/mul¿
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_7_assignmovingavg_1_196319/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/196319*
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
#batch_normalization_7/batchnorm/mulÛ
%batch_normalization_7/batchnorm/mul_1Mul%average_pooling1d_11/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2'
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
:ÿÿÿÿÿÿÿÿÿB 2'
%batch_normalization_7/batchnorm/add_1«
	add_3/addAddV2)batch_normalization_6/batchnorm/add_1:z:0)batch_normalization_7/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
	add_3/add¹
Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_7_multi_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpÐ
>transformer_block_7/multi_head_attention_7/query/einsum/EinsumEinsumadd_3/add:z:0Utransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2@
>transformer_block_7/multi_head_attention_7/query/einsum/Einsum
Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOpReadVariableOpLtransformer_block_7_multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOpÅ
4transformer_block_7/multi_head_attention_7/query/addAddV2Gtransformer_block_7/multi_head_attention_7/query/einsum/Einsum:output:0Ktransformer_block_7/multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 26
4transformer_block_7/multi_head_attention_7/query/add³
Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_7_multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpÊ
<transformer_block_7/multi_head_attention_7/key/einsum/EinsumEinsumadd_3/add:z:0Stransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2>
<transformer_block_7/multi_head_attention_7/key/einsum/Einsum
Atransformer_block_7/multi_head_attention_7/key/add/ReadVariableOpReadVariableOpJtransformer_block_7_multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_7/multi_head_attention_7/key/add/ReadVariableOp½
2transformer_block_7/multi_head_attention_7/key/addAddV2Etransformer_block_7/multi_head_attention_7/key/einsum/Einsum:output:0Itransformer_block_7/multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 24
2transformer_block_7/multi_head_attention_7/key/add¹
Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_7_multi_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpÐ
>transformer_block_7/multi_head_attention_7/value/einsum/EinsumEinsumadd_3/add:z:0Utransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2@
>transformer_block_7/multi_head_attention_7/value/einsum/Einsum
Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOpReadVariableOpLtransformer_block_7_multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOpÅ
4transformer_block_7/multi_head_attention_7/value/addAddV2Gtransformer_block_7/multi_head_attention_7/value/einsum/Einsum:output:0Ktransformer_block_7/multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 26
4transformer_block_7/multi_head_attention_7/value/add©
0transformer_block_7/multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>22
0transformer_block_7/multi_head_attention_7/Mul/y
.transformer_block_7/multi_head_attention_7/MulMul8transformer_block_7/multi_head_attention_7/query/add:z:09transformer_block_7/multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 20
.transformer_block_7/multi_head_attention_7/MulÌ
8transformer_block_7/multi_head_attention_7/einsum/EinsumEinsum6transformer_block_7/multi_head_attention_7/key/add:z:02transformer_block_7/multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
equationaecd,abcd->acbe2:
8transformer_block_7/multi_head_attention_7/einsum/Einsum
:transformer_block_7/multi_head_attention_7/softmax/SoftmaxSoftmaxAtransformer_block_7/multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2<
:transformer_block_7/multi_head_attention_7/softmax/SoftmaxÉ
@transformer_block_7/multi_head_attention_7/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2B
@transformer_block_7/multi_head_attention_7/dropout/dropout/ConstÒ
>transformer_block_7/multi_head_attention_7/dropout/dropout/MulMulDtransformer_block_7/multi_head_attention_7/softmax/Softmax:softmax:0Itransformer_block_7/multi_head_attention_7/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2@
>transformer_block_7/multi_head_attention_7/dropout/dropout/Mulø
@transformer_block_7/multi_head_attention_7/dropout/dropout/ShapeShapeDtransformer_block_7/multi_head_attention_7/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2B
@transformer_block_7/multi_head_attention_7/dropout/dropout/Shapeá
Wtransformer_block_7/multi_head_attention_7/dropout/dropout/random_uniform/RandomUniformRandomUniformItransformer_block_7/multi_head_attention_7/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
dtype0*

seed*2Y
Wtransformer_block_7/multi_head_attention_7/dropout/dropout/random_uniform/RandomUniformÛ
Itransformer_block_7/multi_head_attention_7/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2K
Itransformer_block_7/multi_head_attention_7/dropout/dropout/GreaterEqual/y
Gtransformer_block_7/multi_head_attention_7/dropout/dropout/GreaterEqualGreaterEqual`transformer_block_7/multi_head_attention_7/dropout/dropout/random_uniform/RandomUniform:output:0Rtransformer_block_7/multi_head_attention_7/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2I
Gtransformer_block_7/multi_head_attention_7/dropout/dropout/GreaterEqual 
?transformer_block_7/multi_head_attention_7/dropout/dropout/CastCastKtransformer_block_7/multi_head_attention_7/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2A
?transformer_block_7/multi_head_attention_7/dropout/dropout/CastÎ
@transformer_block_7/multi_head_attention_7/dropout/dropout/Mul_1MulBtransformer_block_7/multi_head_attention_7/dropout/dropout/Mul:z:0Ctransformer_block_7/multi_head_attention_7/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2B
@transformer_block_7/multi_head_attention_7/dropout/dropout/Mul_1ä
:transformer_block_7/multi_head_attention_7/einsum_1/EinsumEinsumDtransformer_block_7/multi_head_attention_7/dropout/dropout/Mul_1:z:08transformer_block_7/multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationacbe,aecd->abcd2<
:transformer_block_7/multi_head_attention_7/einsum_1/EinsumÚ
Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_7_multi_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp£
Itransformer_block_7/multi_head_attention_7/attention_output/einsum/EinsumEinsumCtransformer_block_7/multi_head_attention_7/einsum_1/Einsum:output:0`transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabcd,cde->abe2K
Itransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum´
Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_7_multi_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpí
?transformer_block_7/multi_head_attention_7/attention_output/addAddV2Rtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum:output:0Vtransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2A
?transformer_block_7/multi_head_attention_7/attention_output/add¡
,transformer_block_7/dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2.
,transformer_block_7/dropout_20/dropout/Const
*transformer_block_7/dropout_20/dropout/MulMulCtransformer_block_7/multi_head_attention_7/attention_output/add:z:05transformer_block_7/dropout_20/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2,
*transformer_block_7/dropout_20/dropout/MulÏ
,transformer_block_7/dropout_20/dropout/ShapeShapeCtransformer_block_7/multi_head_attention_7/attention_output/add:z:0*
T0*
_output_shapes
:2.
,transformer_block_7/dropout_20/dropout/Shape®
Ctransformer_block_7/dropout_20/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_7/dropout_20/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
dtype0*

seed**
seed22E
Ctransformer_block_7/dropout_20/dropout/random_uniform/RandomUniform³
5transformer_block_7/dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=27
5transformer_block_7/dropout_20/dropout/GreaterEqual/y¾
3transformer_block_7/dropout_20/dropout/GreaterEqualGreaterEqualLtransformer_block_7/dropout_20/dropout/random_uniform/RandomUniform:output:0>transformer_block_7/dropout_20/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 25
3transformer_block_7/dropout_20/dropout/GreaterEqualà
+transformer_block_7/dropout_20/dropout/CastCast7transformer_block_7/dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2-
+transformer_block_7/dropout_20/dropout/Castú
,transformer_block_7/dropout_20/dropout/Mul_1Mul.transformer_block_7/dropout_20/dropout/Mul:z:0/transformer_block_7/dropout_20/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2.
,transformer_block_7/dropout_20/dropout/Mul_1²
transformer_block_7/addAddV2add_3/add:z:00transformer_block_7/dropout_20/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
transformer_block_7/addà
Itransformer_block_7/layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_7/layer_normalization_14/moments/mean/reduction_indices²
7transformer_block_7/layer_normalization_14/moments/meanMeantransformer_block_7/add:z:0Rtransformer_block_7/layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(29
7transformer_block_7/layer_normalization_14/moments/mean
?transformer_block_7/layer_normalization_14/moments/StopGradientStopGradient@transformer_block_7/layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2A
?transformer_block_7/layer_normalization_14/moments/StopGradient¾
Dtransformer_block_7/layer_normalization_14/moments/SquaredDifferenceSquaredDifferencetransformer_block_7/add:z:0Htransformer_block_7/layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2F
Dtransformer_block_7/layer_normalization_14/moments/SquaredDifferenceè
Mtransformer_block_7/layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_7/layer_normalization_14/moments/variance/reduction_indicesë
;transformer_block_7/layer_normalization_14/moments/varianceMeanHtransformer_block_7/layer_normalization_14/moments/SquaredDifference:z:0Vtransformer_block_7/layer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2=
;transformer_block_7/layer_normalization_14/moments/variance½
:transformer_block_7/layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_7/layer_normalization_14/batchnorm/add/y¾
8transformer_block_7/layer_normalization_14/batchnorm/addAddV2Dtransformer_block_7/layer_normalization_14/moments/variance:output:0Ctransformer_block_7/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2:
8transformer_block_7/layer_normalization_14/batchnorm/addõ
:transformer_block_7/layer_normalization_14/batchnorm/RsqrtRsqrt<transformer_block_7/layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2<
:transformer_block_7/layer_normalization_14/batchnorm/Rsqrt
Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_7_layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpÂ
8transformer_block_7/layer_normalization_14/batchnorm/mulMul>transformer_block_7/layer_normalization_14/batchnorm/Rsqrt:y:0Otransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2:
8transformer_block_7/layer_normalization_14/batchnorm/mul
:transformer_block_7/layer_normalization_14/batchnorm/mul_1Multransformer_block_7/add:z:0<transformer_block_7/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2<
:transformer_block_7/layer_normalization_14/batchnorm/mul_1µ
:transformer_block_7/layer_normalization_14/batchnorm/mul_2Mul@transformer_block_7/layer_normalization_14/moments/mean:output:0<transformer_block_7/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2<
:transformer_block_7/layer_normalization_14/batchnorm/mul_2
Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_7_layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp¾
8transformer_block_7/layer_normalization_14/batchnorm/subSubKtransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp:value:0>transformer_block_7/layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2:
8transformer_block_7/layer_normalization_14/batchnorm/subµ
:transformer_block_7/layer_normalization_14/batchnorm/add_1AddV2>transformer_block_7/layer_normalization_14/batchnorm/mul_1:z:0<transformer_block_7/layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2<
:transformer_block_7/layer_normalization_14/batchnorm/add_1
Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_7_sequential_7_dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02D
Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp¾
8transformer_block_7/sequential_7/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_7/sequential_7/dense_23/Tensordot/axesÅ
8transformer_block_7/sequential_7/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_7/sequential_7/dense_23/Tensordot/freeä
9transformer_block_7/sequential_7/dense_23/Tensordot/ShapeShape>transformer_block_7/layer_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_23/Tensordot/ShapeÈ
Atransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axis£
<transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2GatherV2Btransformer_block_7/sequential_7/dense_23/Tensordot/Shape:output:0Atransformer_block_7/sequential_7/dense_23/Tensordot/free:output:0Jtransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2Ì
Ctransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axis©
>transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1GatherV2Btransformer_block_7/sequential_7/dense_23/Tensordot/Shape:output:0Atransformer_block_7/sequential_7/dense_23/Tensordot/axes:output:0Ltransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1À
9transformer_block_7/sequential_7/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_7/sequential_7/dense_23/Tensordot/Const¨
8transformer_block_7/sequential_7/dense_23/Tensordot/ProdProdEtransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2:output:0Btransformer_block_7/sequential_7/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_7/sequential_7/dense_23/Tensordot/ProdÄ
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_1°
:transformer_block_7/sequential_7/dense_23/Tensordot/Prod_1ProdGtransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1:output:0Dtransformer_block_7/sequential_7/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_7/sequential_7/dense_23/Tensordot/Prod_1Ä
?transformer_block_7/sequential_7/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_7/sequential_7/dense_23/Tensordot/concat/axis
:transformer_block_7/sequential_7/dense_23/Tensordot/concatConcatV2Atransformer_block_7/sequential_7/dense_23/Tensordot/free:output:0Atransformer_block_7/sequential_7/dense_23/Tensordot/axes:output:0Htransformer_block_7/sequential_7/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_7/sequential_7/dense_23/Tensordot/concat´
9transformer_block_7/sequential_7/dense_23/Tensordot/stackPackAtransformer_block_7/sequential_7/dense_23/Tensordot/Prod:output:0Ctransformer_block_7/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_23/Tensordot/stackÆ
=transformer_block_7/sequential_7/dense_23/Tensordot/transpose	Transpose>transformer_block_7/layer_normalization_14/batchnorm/add_1:z:0Ctransformer_block_7/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2?
=transformer_block_7/sequential_7/dense_23/Tensordot/transposeÇ
;transformer_block_7/sequential_7/dense_23/Tensordot/ReshapeReshapeAtransformer_block_7/sequential_7/dense_23/Tensordot/transpose:y:0Btransformer_block_7/sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2=
;transformer_block_7/sequential_7/dense_23/Tensordot/ReshapeÆ
:transformer_block_7/sequential_7/dense_23/Tensordot/MatMulMatMulDtransformer_block_7/sequential_7/dense_23/Tensordot/Reshape:output:0Jtransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2<
:transformer_block_7/sequential_7/dense_23/Tensordot/MatMulÄ
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2=
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_2È
Atransformer_block_7/sequential_7/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_7/sequential_7/dense_23/Tensordot/concat_1/axis
<transformer_block_7/sequential_7/dense_23/Tensordot/concat_1ConcatV2Etransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2:output:0Dtransformer_block_7/sequential_7/dense_23/Tensordot/Const_2:output:0Jtransformer_block_7/sequential_7/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_7/sequential_7/dense_23/Tensordot/concat_1¸
3transformer_block_7/sequential_7/dense_23/TensordotReshapeDtransformer_block_7/sequential_7/dense_23/Tensordot/MatMul:product:0Etransformer_block_7/sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@25
3transformer_block_7/sequential_7/dense_23/Tensordot
@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_7_sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp¯
1transformer_block_7/sequential_7/dense_23/BiasAddBiasAdd<transformer_block_7/sequential_7/dense_23/Tensordot:output:0Htransformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@23
1transformer_block_7/sequential_7/dense_23/BiasAddÚ
.transformer_block_7/sequential_7/dense_23/ReluRelu:transformer_block_7/sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@20
.transformer_block_7/sequential_7/dense_23/Relu
Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_7_sequential_7_dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02D
Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp¾
8transformer_block_7/sequential_7/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_7/sequential_7/dense_24/Tensordot/axesÅ
8transformer_block_7/sequential_7/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_7/sequential_7/dense_24/Tensordot/freeâ
9transformer_block_7/sequential_7/dense_24/Tensordot/ShapeShape<transformer_block_7/sequential_7/dense_23/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_24/Tensordot/ShapeÈ
Atransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axis£
<transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2GatherV2Btransformer_block_7/sequential_7/dense_24/Tensordot/Shape:output:0Atransformer_block_7/sequential_7/dense_24/Tensordot/free:output:0Jtransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2Ì
Ctransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axis©
>transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1GatherV2Btransformer_block_7/sequential_7/dense_24/Tensordot/Shape:output:0Atransformer_block_7/sequential_7/dense_24/Tensordot/axes:output:0Ltransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1À
9transformer_block_7/sequential_7/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_7/sequential_7/dense_24/Tensordot/Const¨
8transformer_block_7/sequential_7/dense_24/Tensordot/ProdProdEtransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2:output:0Btransformer_block_7/sequential_7/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_7/sequential_7/dense_24/Tensordot/ProdÄ
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_1°
:transformer_block_7/sequential_7/dense_24/Tensordot/Prod_1ProdGtransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1:output:0Dtransformer_block_7/sequential_7/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_7/sequential_7/dense_24/Tensordot/Prod_1Ä
?transformer_block_7/sequential_7/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_7/sequential_7/dense_24/Tensordot/concat/axis
:transformer_block_7/sequential_7/dense_24/Tensordot/concatConcatV2Atransformer_block_7/sequential_7/dense_24/Tensordot/free:output:0Atransformer_block_7/sequential_7/dense_24/Tensordot/axes:output:0Htransformer_block_7/sequential_7/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_7/sequential_7/dense_24/Tensordot/concat´
9transformer_block_7/sequential_7/dense_24/Tensordot/stackPackAtransformer_block_7/sequential_7/dense_24/Tensordot/Prod:output:0Ctransformer_block_7/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_24/Tensordot/stackÄ
=transformer_block_7/sequential_7/dense_24/Tensordot/transpose	Transpose<transformer_block_7/sequential_7/dense_23/Relu:activations:0Ctransformer_block_7/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2?
=transformer_block_7/sequential_7/dense_24/Tensordot/transposeÇ
;transformer_block_7/sequential_7/dense_24/Tensordot/ReshapeReshapeAtransformer_block_7/sequential_7/dense_24/Tensordot/transpose:y:0Btransformer_block_7/sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2=
;transformer_block_7/sequential_7/dense_24/Tensordot/ReshapeÆ
:transformer_block_7/sequential_7/dense_24/Tensordot/MatMulMatMulDtransformer_block_7/sequential_7/dense_24/Tensordot/Reshape:output:0Jtransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2<
:transformer_block_7/sequential_7/dense_24/Tensordot/MatMulÄ
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_2È
Atransformer_block_7/sequential_7/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_7/sequential_7/dense_24/Tensordot/concat_1/axis
<transformer_block_7/sequential_7/dense_24/Tensordot/concat_1ConcatV2Etransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2:output:0Dtransformer_block_7/sequential_7/dense_24/Tensordot/Const_2:output:0Jtransformer_block_7/sequential_7/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_7/sequential_7/dense_24/Tensordot/concat_1¸
3transformer_block_7/sequential_7/dense_24/TensordotReshapeDtransformer_block_7/sequential_7/dense_24/Tensordot/MatMul:product:0Etransformer_block_7/sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 25
3transformer_block_7/sequential_7/dense_24/Tensordot
@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_7_sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp¯
1transformer_block_7/sequential_7/dense_24/BiasAddBiasAdd<transformer_block_7/sequential_7/dense_24/Tensordot:output:0Htransformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 23
1transformer_block_7/sequential_7/dense_24/BiasAdd¡
,transformer_block_7/dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2.
,transformer_block_7/dropout_21/dropout/Const
*transformer_block_7/dropout_21/dropout/MulMul:transformer_block_7/sequential_7/dense_24/BiasAdd:output:05transformer_block_7/dropout_21/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2,
*transformer_block_7/dropout_21/dropout/MulÆ
,transformer_block_7/dropout_21/dropout/ShapeShape:transformer_block_7/sequential_7/dense_24/BiasAdd:output:0*
T0*
_output_shapes
:2.
,transformer_block_7/dropout_21/dropout/Shape®
Ctransformer_block_7/dropout_21/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_7/dropout_21/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
dtype0*

seed**
seed22E
Ctransformer_block_7/dropout_21/dropout/random_uniform/RandomUniform³
5transformer_block_7/dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=27
5transformer_block_7/dropout_21/dropout/GreaterEqual/y¾
3transformer_block_7/dropout_21/dropout/GreaterEqualGreaterEqualLtransformer_block_7/dropout_21/dropout/random_uniform/RandomUniform:output:0>transformer_block_7/dropout_21/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 25
3transformer_block_7/dropout_21/dropout/GreaterEqualà
+transformer_block_7/dropout_21/dropout/CastCast7transformer_block_7/dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2-
+transformer_block_7/dropout_21/dropout/Castú
,transformer_block_7/dropout_21/dropout/Mul_1Mul.transformer_block_7/dropout_21/dropout/Mul:z:0/transformer_block_7/dropout_21/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2.
,transformer_block_7/dropout_21/dropout/Mul_1ç
transformer_block_7/add_1AddV2>transformer_block_7/layer_normalization_14/batchnorm/add_1:z:00transformer_block_7/dropout_21/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
transformer_block_7/add_1à
Itransformer_block_7/layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_7/layer_normalization_15/moments/mean/reduction_indices´
7transformer_block_7/layer_normalization_15/moments/meanMeantransformer_block_7/add_1:z:0Rtransformer_block_7/layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(29
7transformer_block_7/layer_normalization_15/moments/mean
?transformer_block_7/layer_normalization_15/moments/StopGradientStopGradient@transformer_block_7/layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2A
?transformer_block_7/layer_normalization_15/moments/StopGradientÀ
Dtransformer_block_7/layer_normalization_15/moments/SquaredDifferenceSquaredDifferencetransformer_block_7/add_1:z:0Htransformer_block_7/layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2F
Dtransformer_block_7/layer_normalization_15/moments/SquaredDifferenceè
Mtransformer_block_7/layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_7/layer_normalization_15/moments/variance/reduction_indicesë
;transformer_block_7/layer_normalization_15/moments/varianceMeanHtransformer_block_7/layer_normalization_15/moments/SquaredDifference:z:0Vtransformer_block_7/layer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2=
;transformer_block_7/layer_normalization_15/moments/variance½
:transformer_block_7/layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_7/layer_normalization_15/batchnorm/add/y¾
8transformer_block_7/layer_normalization_15/batchnorm/addAddV2Dtransformer_block_7/layer_normalization_15/moments/variance:output:0Ctransformer_block_7/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2:
8transformer_block_7/layer_normalization_15/batchnorm/addõ
:transformer_block_7/layer_normalization_15/batchnorm/RsqrtRsqrt<transformer_block_7/layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2<
:transformer_block_7/layer_normalization_15/batchnorm/Rsqrt
Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_7_layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpÂ
8transformer_block_7/layer_normalization_15/batchnorm/mulMul>transformer_block_7/layer_normalization_15/batchnorm/Rsqrt:y:0Otransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2:
8transformer_block_7/layer_normalization_15/batchnorm/mul
:transformer_block_7/layer_normalization_15/batchnorm/mul_1Multransformer_block_7/add_1:z:0<transformer_block_7/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2<
:transformer_block_7/layer_normalization_15/batchnorm/mul_1µ
:transformer_block_7/layer_normalization_15/batchnorm/mul_2Mul@transformer_block_7/layer_normalization_15/moments/mean:output:0<transformer_block_7/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2<
:transformer_block_7/layer_normalization_15/batchnorm/mul_2
Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_7_layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp¾
8transformer_block_7/layer_normalization_15/batchnorm/subSubKtransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp:value:0>transformer_block_7/layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2:
8transformer_block_7/layer_normalization_15/batchnorm/subµ
:transformer_block_7/layer_normalization_15/batchnorm/add_1AddV2>transformer_block_7/layer_normalization_15/batchnorm/mul_1:z:0<transformer_block_7/layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2<
:transformer_block_7/layer_normalization_15/batchnorm/add_1s
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
flatten_3/Const¾
flatten_3/ReshapeReshape>transformer_block_7/layer_normalization_15/batchnorm/add_1:z:0flatten_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
flatten_3/Reshapex
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axis¾
concatenate_3/concatConcatV2flatten_3/Reshape:output:0inputs_1"concatenate_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
concatenate_3/concat©
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	È@*
dtype02 
dense_25/MatMul/ReadVariableOp¥
dense_25/MatMulMatMulconcatenate_3/concat:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_25/MatMul§
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_25/BiasAdd/ReadVariableOp¥
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_25/BiasAdds
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_25/Reluy
dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_22/dropout/Const©
dropout_22/dropout/MulMuldense_25/Relu:activations:0!dropout_22/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_22/dropout/Mul
dropout_22/dropout/ShapeShapedense_25/Relu:activations:0*
T0*
_output_shapes
:2
dropout_22/dropout/Shapeî
/dropout_22/dropout/random_uniform/RandomUniformRandomUniform!dropout_22/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*

seed**
seed221
/dropout_22/dropout/random_uniform/RandomUniform
!dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_22/dropout/GreaterEqual/yê
dropout_22/dropout/GreaterEqualGreaterEqual8dropout_22/dropout/random_uniform/RandomUniform:output:0*dropout_22/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
dropout_22/dropout/GreaterEqual 
dropout_22/dropout/CastCast#dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_22/dropout/Cast¦
dropout_22/dropout/Mul_1Muldropout_22/dropout/Mul:z:0dropout_22/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_22/dropout/Mul_1¨
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_26/MatMul/ReadVariableOp¤
dense_26/MatMulMatMuldropout_22/dropout/Mul_1:z:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_26/MatMul§
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_26/BiasAdd/ReadVariableOp¥
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_26/BiasAdds
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_26/Reluy
dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_23/dropout/Const©
dropout_23/dropout/MulMuldense_26/Relu:activations:0!dropout_23/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_23/dropout/Mul
dropout_23/dropout/ShapeShapedense_26/Relu:activations:0*
T0*
_output_shapes
:2
dropout_23/dropout/Shapeî
/dropout_23/dropout/random_uniform/RandomUniformRandomUniform!dropout_23/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*

seed**
seed221
/dropout_23/dropout/random_uniform/RandomUniform
!dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_23/dropout/GreaterEqual/yê
dropout_23/dropout/GreaterEqualGreaterEqual8dropout_23/dropout/random_uniform/RandomUniform:output:0*dropout_23/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
dropout_23/dropout/GreaterEqual 
dropout_23/dropout/CastCast#dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_23/dropout/Cast¦
dropout_23/dropout/Mul_1Muldropout_23/dropout/Mul:z:0dropout_23/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_23/dropout/Mul_1¨
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_27/MatMul/ReadVariableOp¤
dense_27/MatMulMatMuldropout_23/dropout/Mul_1:z:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_27/MatMul§
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_27/BiasAdd/ReadVariableOp¥
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_27/BiasAdd
IdentityIdentitydense_27/BiasAdd:output:0:^batch_normalization_6/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_6/AssignMovingAvg/ReadVariableOp<^batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_6/batchnorm/ReadVariableOp3^batch_normalization_6/batchnorm/mul/ReadVariableOp:^batch_normalization_7/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_7/AssignMovingAvg/ReadVariableOp<^batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_7/batchnorm/ReadVariableOp3^batch_normalization_7/batchnorm/mul/ReadVariableOp ^conv1d_6/BiasAdd/ReadVariableOp,^conv1d_6/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_7/BiasAdd/ReadVariableOp,^conv1d_7/conv1d/ExpandDims_1/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp<^token_and_position_embedding_3/embedding_6/embedding_lookup<^token_and_position_embedding_3/embedding_7/embedding_lookupD^transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpH^transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpD^transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpH^transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpO^transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpY^transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_7/multi_head_attention_7/key/add/ReadVariableOpL^transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpD^transformer_block_7/multi_head_attention_7/query/add/ReadVariableOpN^transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpD^transformer_block_7/multi_head_attention_7/value/add/ReadVariableOpN^transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpA^transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpC^transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpA^transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpC^transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Í
_input_shapes»
¸:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2v
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
2batch_normalization_7/batchnorm/mul/ReadVariableOp2batch_normalization_7/batchnorm/mul/ReadVariableOp2B
conv1d_6/BiasAdd/ReadVariableOpconv1d_6/BiasAdd/ReadVariableOp2Z
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_7/BiasAdd/ReadVariableOpconv1d_7/BiasAdd/ReadVariableOp2Z
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2z
;token_and_position_embedding_3/embedding_6/embedding_lookup;token_and_position_embedding_3/embedding_6/embedding_lookup2z
;token_and_position_embedding_3/embedding_7/embedding_lookup;token_and_position_embedding_3/embedding_7/embedding_lookup2
Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpCtransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp2
Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpGtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp2
Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpCtransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp2
Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpGtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp2 
Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpNtransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp2´
Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp2
Atransformer_block_7/multi_head_attention_7/key/add/ReadVariableOpAtransformer_block_7/multi_head_attention_7/key/add/ReadVariableOp2
Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpKtransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp2
Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOpCtransformer_block_7/multi_head_attention_7/query/add/ReadVariableOp2
Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpMtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp2
Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOpCtransformer_block_7/multi_head_attention_7/value/add/ReadVariableOp2
Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpMtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp2
@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp2
Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpBtransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp2
@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp2
Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpBtransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp:S O
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ø
l
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_194369

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
È
©
6__inference_batch_normalization_7_layer_call_fn_197320

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
:ÿÿÿÿÿÿÿÿÿB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1950602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
ÄÕ
F
"__inference__traced_restore_198726
file_prefix$
 assignvariableop_conv1d_6_kernel$
 assignvariableop_1_conv1d_6_bias&
"assignvariableop_2_conv1d_7_kernel$
 assignvariableop_3_conv1d_7_bias2
.assignvariableop_4_batch_normalization_6_gamma1
-assignvariableop_5_batch_normalization_6_beta8
4assignvariableop_6_batch_normalization_6_moving_mean<
8assignvariableop_7_batch_normalization_6_moving_variance2
.assignvariableop_8_batch_normalization_7_gamma1
-assignvariableop_9_batch_normalization_7_beta9
5assignvariableop_10_batch_normalization_7_moving_mean=
9assignvariableop_11_batch_normalization_7_moving_variance'
#assignvariableop_12_dense_25_kernel%
!assignvariableop_13_dense_25_bias'
#assignvariableop_14_dense_26_kernel%
!assignvariableop_15_dense_26_bias'
#assignvariableop_16_dense_27_kernel%
!assignvariableop_17_dense_27_bias
assignvariableop_18_beta_1
assignvariableop_19_beta_2
assignvariableop_20_decay%
!assignvariableop_21_learning_rate!
assignvariableop_22_adam_iterM
Iassignvariableop_23_token_and_position_embedding_3_embedding_6_embeddingsM
Iassignvariableop_24_token_and_position_embedding_3_embedding_7_embeddingsO
Kassignvariableop_25_transformer_block_7_multi_head_attention_7_query_kernelM
Iassignvariableop_26_transformer_block_7_multi_head_attention_7_query_biasM
Iassignvariableop_27_transformer_block_7_multi_head_attention_7_key_kernelK
Gassignvariableop_28_transformer_block_7_multi_head_attention_7_key_biasO
Kassignvariableop_29_transformer_block_7_multi_head_attention_7_value_kernelM
Iassignvariableop_30_transformer_block_7_multi_head_attention_7_value_biasZ
Vassignvariableop_31_transformer_block_7_multi_head_attention_7_attention_output_kernelX
Tassignvariableop_32_transformer_block_7_multi_head_attention_7_attention_output_bias'
#assignvariableop_33_dense_23_kernel%
!assignvariableop_34_dense_23_bias'
#assignvariableop_35_dense_24_kernel%
!assignvariableop_36_dense_24_biasH
Dassignvariableop_37_transformer_block_7_layer_normalization_14_gammaG
Cassignvariableop_38_transformer_block_7_layer_normalization_14_betaH
Dassignvariableop_39_transformer_block_7_layer_normalization_15_gammaG
Cassignvariableop_40_transformer_block_7_layer_normalization_15_beta
assignvariableop_41_total
assignvariableop_42_count.
*assignvariableop_43_adam_conv1d_6_kernel_m,
(assignvariableop_44_adam_conv1d_6_bias_m.
*assignvariableop_45_adam_conv1d_7_kernel_m,
(assignvariableop_46_adam_conv1d_7_bias_m:
6assignvariableop_47_adam_batch_normalization_6_gamma_m9
5assignvariableop_48_adam_batch_normalization_6_beta_m:
6assignvariableop_49_adam_batch_normalization_7_gamma_m9
5assignvariableop_50_adam_batch_normalization_7_beta_m.
*assignvariableop_51_adam_dense_25_kernel_m,
(assignvariableop_52_adam_dense_25_bias_m.
*assignvariableop_53_adam_dense_26_kernel_m,
(assignvariableop_54_adam_dense_26_bias_m.
*assignvariableop_55_adam_dense_27_kernel_m,
(assignvariableop_56_adam_dense_27_bias_mT
Passignvariableop_57_adam_token_and_position_embedding_3_embedding_6_embeddings_mT
Passignvariableop_58_adam_token_and_position_embedding_3_embedding_7_embeddings_mV
Rassignvariableop_59_adam_transformer_block_7_multi_head_attention_7_query_kernel_mT
Passignvariableop_60_adam_transformer_block_7_multi_head_attention_7_query_bias_mT
Passignvariableop_61_adam_transformer_block_7_multi_head_attention_7_key_kernel_mR
Nassignvariableop_62_adam_transformer_block_7_multi_head_attention_7_key_bias_mV
Rassignvariableop_63_adam_transformer_block_7_multi_head_attention_7_value_kernel_mT
Passignvariableop_64_adam_transformer_block_7_multi_head_attention_7_value_bias_ma
]assignvariableop_65_adam_transformer_block_7_multi_head_attention_7_attention_output_kernel_m_
[assignvariableop_66_adam_transformer_block_7_multi_head_attention_7_attention_output_bias_m.
*assignvariableop_67_adam_dense_23_kernel_m,
(assignvariableop_68_adam_dense_23_bias_m.
*assignvariableop_69_adam_dense_24_kernel_m,
(assignvariableop_70_adam_dense_24_bias_mO
Kassignvariableop_71_adam_transformer_block_7_layer_normalization_14_gamma_mN
Jassignvariableop_72_adam_transformer_block_7_layer_normalization_14_beta_mO
Kassignvariableop_73_adam_transformer_block_7_layer_normalization_15_gamma_mN
Jassignvariableop_74_adam_transformer_block_7_layer_normalization_15_beta_m.
*assignvariableop_75_adam_conv1d_6_kernel_v,
(assignvariableop_76_adam_conv1d_6_bias_v.
*assignvariableop_77_adam_conv1d_7_kernel_v,
(assignvariableop_78_adam_conv1d_7_bias_v:
6assignvariableop_79_adam_batch_normalization_6_gamma_v9
5assignvariableop_80_adam_batch_normalization_6_beta_v:
6assignvariableop_81_adam_batch_normalization_7_gamma_v9
5assignvariableop_82_adam_batch_normalization_7_beta_v.
*assignvariableop_83_adam_dense_25_kernel_v,
(assignvariableop_84_adam_dense_25_bias_v.
*assignvariableop_85_adam_dense_26_kernel_v,
(assignvariableop_86_adam_dense_26_bias_v.
*assignvariableop_87_adam_dense_27_kernel_v,
(assignvariableop_88_adam_dense_27_bias_vT
Passignvariableop_89_adam_token_and_position_embedding_3_embedding_6_embeddings_vT
Passignvariableop_90_adam_token_and_position_embedding_3_embedding_7_embeddings_vV
Rassignvariableop_91_adam_transformer_block_7_multi_head_attention_7_query_kernel_vT
Passignvariableop_92_adam_transformer_block_7_multi_head_attention_7_query_bias_vT
Passignvariableop_93_adam_transformer_block_7_multi_head_attention_7_key_kernel_vR
Nassignvariableop_94_adam_transformer_block_7_multi_head_attention_7_key_bias_vV
Rassignvariableop_95_adam_transformer_block_7_multi_head_attention_7_value_kernel_vT
Passignvariableop_96_adam_transformer_block_7_multi_head_attention_7_value_bias_va
]assignvariableop_97_adam_transformer_block_7_multi_head_attention_7_attention_output_kernel_v_
[assignvariableop_98_adam_transformer_block_7_multi_head_attention_7_attention_output_bias_v.
*assignvariableop_99_adam_dense_23_kernel_v-
)assignvariableop_100_adam_dense_23_bias_v/
+assignvariableop_101_adam_dense_24_kernel_v-
)assignvariableop_102_adam_dense_24_bias_vP
Lassignvariableop_103_adam_transformer_block_7_layer_normalization_14_gamma_vO
Kassignvariableop_104_adam_transformer_block_7_layer_normalization_14_beta_vP
Lassignvariableop_105_adam_transformer_block_7_layer_normalization_15_gamma_vO
Kassignvariableop_106_adam_transformer_block_7_layer_normalization_15_beta_v
identity_108¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_100¢AssignVariableOp_101¢AssignVariableOp_102¢AssignVariableOp_103¢AssignVariableOp_104¢AssignVariableOp_105¢AssignVariableOp_106¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98¢AssignVariableOp_997
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:l*
dtype0* 6
value6B6lB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesé
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:l*
dtype0*í
valueãBàlB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÊ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Æ
_output_shapes³
°::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*z
dtypesp
n2l	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_conv1d_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_7_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_7_biasIdentity_3:output:0"/device:CPU:0*
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
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_25_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_25_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_26_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_26_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16«
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_27_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17©
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_27_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¢
AssignVariableOp_18AssignVariableOpassignvariableop_18_beta_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¢
AssignVariableOp_19AssignVariableOpassignvariableop_19_beta_2Identity_19:output:0"/device:CPU:0*
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
T0	*
_output_shapes
:2
Identity_22¥
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_iterIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ñ
AssignVariableOp_23AssignVariableOpIassignvariableop_23_token_and_position_embedding_3_embedding_6_embeddingsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ñ
AssignVariableOp_24AssignVariableOpIassignvariableop_24_token_and_position_embedding_3_embedding_7_embeddingsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ó
AssignVariableOp_25AssignVariableOpKassignvariableop_25_transformer_block_7_multi_head_attention_7_query_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ñ
AssignVariableOp_26AssignVariableOpIassignvariableop_26_transformer_block_7_multi_head_attention_7_query_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ñ
AssignVariableOp_27AssignVariableOpIassignvariableop_27_transformer_block_7_multi_head_attention_7_key_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ï
AssignVariableOp_28AssignVariableOpGassignvariableop_28_transformer_block_7_multi_head_attention_7_key_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ó
AssignVariableOp_29AssignVariableOpKassignvariableop_29_transformer_block_7_multi_head_attention_7_value_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ñ
AssignVariableOp_30AssignVariableOpIassignvariableop_30_transformer_block_7_multi_head_attention_7_value_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Þ
AssignVariableOp_31AssignVariableOpVassignvariableop_31_transformer_block_7_multi_head_attention_7_attention_output_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ü
AssignVariableOp_32AssignVariableOpTassignvariableop_32_transformer_block_7_multi_head_attention_7_attention_output_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33«
AssignVariableOp_33AssignVariableOp#assignvariableop_33_dense_23_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34©
AssignVariableOp_34AssignVariableOp!assignvariableop_34_dense_23_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35«
AssignVariableOp_35AssignVariableOp#assignvariableop_35_dense_24_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36©
AssignVariableOp_36AssignVariableOp!assignvariableop_36_dense_24_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ì
AssignVariableOp_37AssignVariableOpDassignvariableop_37_transformer_block_7_layer_normalization_14_gammaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ë
AssignVariableOp_38AssignVariableOpCassignvariableop_38_transformer_block_7_layer_normalization_14_betaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Ì
AssignVariableOp_39AssignVariableOpDassignvariableop_39_transformer_block_7_layer_normalization_15_gammaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Ë
AssignVariableOp_40AssignVariableOpCassignvariableop_40_transformer_block_7_layer_normalization_15_betaIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41¡
AssignVariableOp_41AssignVariableOpassignvariableop_41_totalIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42¡
AssignVariableOp_42AssignVariableOpassignvariableop_42_countIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43²
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv1d_6_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44°
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv1d_6_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45²
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv1d_7_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46°
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv1d_7_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47¾
AssignVariableOp_47AssignVariableOp6assignvariableop_47_adam_batch_normalization_6_gamma_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48½
AssignVariableOp_48AssignVariableOp5assignvariableop_48_adam_batch_normalization_6_beta_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49¾
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_batch_normalization_7_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50½
AssignVariableOp_50AssignVariableOp5assignvariableop_50_adam_batch_normalization_7_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51²
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_25_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52°
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_25_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53²
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_26_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54°
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_26_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55²
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_27_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56°
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_27_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57Ø
AssignVariableOp_57AssignVariableOpPassignvariableop_57_adam_token_and_position_embedding_3_embedding_6_embeddings_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58Ø
AssignVariableOp_58AssignVariableOpPassignvariableop_58_adam_token_and_position_embedding_3_embedding_7_embeddings_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Ú
AssignVariableOp_59AssignVariableOpRassignvariableop_59_adam_transformer_block_7_multi_head_attention_7_query_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Ø
AssignVariableOp_60AssignVariableOpPassignvariableop_60_adam_transformer_block_7_multi_head_attention_7_query_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Ø
AssignVariableOp_61AssignVariableOpPassignvariableop_61_adam_transformer_block_7_multi_head_attention_7_key_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62Ö
AssignVariableOp_62AssignVariableOpNassignvariableop_62_adam_transformer_block_7_multi_head_attention_7_key_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63Ú
AssignVariableOp_63AssignVariableOpRassignvariableop_63_adam_transformer_block_7_multi_head_attention_7_value_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Ø
AssignVariableOp_64AssignVariableOpPassignvariableop_64_adam_transformer_block_7_multi_head_attention_7_value_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65å
AssignVariableOp_65AssignVariableOp]assignvariableop_65_adam_transformer_block_7_multi_head_attention_7_attention_output_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66ã
AssignVariableOp_66AssignVariableOp[assignvariableop_66_adam_transformer_block_7_multi_head_attention_7_attention_output_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67²
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_dense_23_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68°
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_dense_23_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69²
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_dense_24_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70°
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_dense_24_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71Ó
AssignVariableOp_71AssignVariableOpKassignvariableop_71_adam_transformer_block_7_layer_normalization_14_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72Ò
AssignVariableOp_72AssignVariableOpJassignvariableop_72_adam_transformer_block_7_layer_normalization_14_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73Ó
AssignVariableOp_73AssignVariableOpKassignvariableop_73_adam_transformer_block_7_layer_normalization_15_gamma_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74Ò
AssignVariableOp_74AssignVariableOpJassignvariableop_74_adam_transformer_block_7_layer_normalization_15_beta_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75²
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_conv1d_6_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76°
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_conv1d_6_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77²
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_conv1d_7_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78°
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_conv1d_7_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79¾
AssignVariableOp_79AssignVariableOp6assignvariableop_79_adam_batch_normalization_6_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80½
AssignVariableOp_80AssignVariableOp5assignvariableop_80_adam_batch_normalization_6_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81¾
AssignVariableOp_81AssignVariableOp6assignvariableop_81_adam_batch_normalization_7_gamma_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82½
AssignVariableOp_82AssignVariableOp5assignvariableop_82_adam_batch_normalization_7_beta_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83²
AssignVariableOp_83AssignVariableOp*assignvariableop_83_adam_dense_25_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84°
AssignVariableOp_84AssignVariableOp(assignvariableop_84_adam_dense_25_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85²
AssignVariableOp_85AssignVariableOp*assignvariableop_85_adam_dense_26_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86°
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adam_dense_26_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87²
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_dense_27_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88°
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_dense_27_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89Ø
AssignVariableOp_89AssignVariableOpPassignvariableop_89_adam_token_and_position_embedding_3_embedding_6_embeddings_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90Ø
AssignVariableOp_90AssignVariableOpPassignvariableop_90_adam_token_and_position_embedding_3_embedding_7_embeddings_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91Ú
AssignVariableOp_91AssignVariableOpRassignvariableop_91_adam_transformer_block_7_multi_head_attention_7_query_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92Ø
AssignVariableOp_92AssignVariableOpPassignvariableop_92_adam_transformer_block_7_multi_head_attention_7_query_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93Ø
AssignVariableOp_93AssignVariableOpPassignvariableop_93_adam_transformer_block_7_multi_head_attention_7_key_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94Ö
AssignVariableOp_94AssignVariableOpNassignvariableop_94_adam_transformer_block_7_multi_head_attention_7_key_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95Ú
AssignVariableOp_95AssignVariableOpRassignvariableop_95_adam_transformer_block_7_multi_head_attention_7_value_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96Ø
AssignVariableOp_96AssignVariableOpPassignvariableop_96_adam_transformer_block_7_multi_head_attention_7_value_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97å
AssignVariableOp_97AssignVariableOp]assignvariableop_97_adam_transformer_block_7_multi_head_attention_7_attention_output_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98ã
AssignVariableOp_98AssignVariableOp[assignvariableop_98_adam_transformer_block_7_multi_head_attention_7_attention_output_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99²
AssignVariableOp_99AssignVariableOp*assignvariableop_99_adam_dense_23_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100´
AssignVariableOp_100AssignVariableOp)assignvariableop_100_adam_dense_23_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101¶
AssignVariableOp_101AssignVariableOp+assignvariableop_101_adam_dense_24_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102´
AssignVariableOp_102AssignVariableOp)assignvariableop_102_adam_dense_24_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103×
AssignVariableOp_103AssignVariableOpLassignvariableop_103_adam_transformer_block_7_layer_normalization_14_gamma_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104Ö
AssignVariableOp_104AssignVariableOpKassignvariableop_104_adam_transformer_block_7_layer_normalization_14_beta_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105×
AssignVariableOp_105AssignVariableOpLassignvariableop_105_adam_transformer_block_7_layer_normalization_15_gamma_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106Ö
AssignVariableOp_106AssignVariableOpKassignvariableop_106_adam_transformer_block_7_layer_normalization_15_beta_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1069
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_107Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_107
Identity_108IdentityIdentity_107:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_108"%
identity_108Identity_108:output:0*Ã
_input_shapes±
®: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062*
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
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
è

Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_195080

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
:ÿÿÿÿÿÿÿÿÿB 2
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
:ÿÿÿÿÿÿÿÿÿB 2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs

G
+__inference_dropout_22_layer_call_fn_197765

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
F__inference_dropout_22_layer_call_and_return_conditional_losses_1955892
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
± 
ã
D__inference_dense_23_layer_call_and_return_conditional_losses_194690

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
:ÿÿÿÿÿÿÿÿÿB 2
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
:ÿÿÿÿÿÿÿÿÿB@2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿB ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
ï
~
)__inference_dense_24_layer_call_fn_198050

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
:ÿÿÿÿÿÿÿÿÿB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_1947362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿB@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@
 
_user_specified_nameinputs

G
+__inference_dropout_23_layer_call_fn_197812

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
F__inference_dropout_23_layer_call_and_return_conditional_losses_1956462
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
È
©
6__inference_batch_normalization_6_layer_call_fn_197074

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
:ÿÿÿÿÿÿÿÿÿB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1949692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
é

H__inference_sequential_7_layer_call_and_return_conditional_losses_194811

inputs
dense_23_194800
dense_23_194802
dense_24_194805
dense_24_194807
identity¢ dense_23/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCall
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinputsdense_23_194800dense_23_194802*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_1946902"
 dense_23/StatefulPartitionedCall¾
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_194805dense_24_194807*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_1947362"
 dense_24/StatefulPartitionedCallÇ
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
ü

Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_194851
x'
#embedding_7_embedding_lookup_194838'
#embedding_6_embedding_lookup_194844
identity¢embedding_6/embedding_lookup¢embedding_7/embedding_lookup?
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
embedding_7/embedding_lookupResourceGather#embedding_7_embedding_lookup_194838range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_7/embedding_lookup/194838*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02
embedding_7/embedding_lookup
%embedding_7/embedding_lookup/IdentityIdentity%embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_7/embedding_lookup/194838*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%embedding_7/embedding_lookup/IdentityÀ
'embedding_7/embedding_lookup/Identity_1Identity.embedding_7/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'embedding_7/embedding_lookup/Identity_1r
embedding_6/CastCastx*

DstT0*

SrcT0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
embedding_6/Cast»
embedding_6/embedding_lookupResourceGather#embedding_6_embedding_lookup_194844embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_6/embedding_lookup/194844*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype02
embedding_6/embedding_lookup
%embedding_6/embedding_lookup/IdentityIdentity%embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_6/embedding_lookup/194844*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2'
%embedding_6/embedding_lookup/IdentityÆ
'embedding_6/embedding_lookup/Identity_1Identity.embedding_6/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2)
'embedding_6/embedding_lookup/Identity_1¯
addAddV20embedding_6/embedding_lookup/Identity_1:output:00embedding_7/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
add
IdentityIdentityadd:z:0^embedding_6/embedding_lookup^embedding_7/embedding_lookup*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::2<
embedding_6/embedding_lookupembedding_6/embedding_lookup2<
embedding_7/embedding_lookupembedding_7/embedding_lookup:L H
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 

_user_specified_namex
ó0
È
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_194471

inputs
assignmovingavg_194446
assignmovingavg_1_194452)
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
loc:@AssignMovingAvg/194446*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_194446*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/194446*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/194446*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_194446AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/194446*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/194452*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_194452*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/194452*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/194452*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_194452AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/194452*
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
¥
d
+__inference_dropout_22_layer_call_fn_197760

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
F__inference_dropout_22_layer_call_and_return_conditional_losses_1955842
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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_197123

inputs
assignmovingavg_197098
assignmovingavg_1_197104)
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
loc:@AssignMovingAvg/197098*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_197098*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/197098*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/197098*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_197098AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/197098*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/197104*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_197104*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/197104*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/197104*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_197104AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/197104*
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
Ðÿ
â
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_197493

inputsF
Bmulti_head_attention_7_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_7_query_add_readvariableop_resourceD
@multi_head_attention_7_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_7_key_add_readvariableop_resourceF
Bmulti_head_attention_7_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_7_value_add_readvariableop_resourceQ
Mmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_7_attention_output_add_readvariableop_resource@
<layer_normalization_14_batchnorm_mul_readvariableop_resource<
8layer_normalization_14_batchnorm_readvariableop_resource;
7sequential_7_dense_23_tensordot_readvariableop_resource9
5sequential_7_dense_23_biasadd_readvariableop_resource;
7sequential_7_dense_24_tensordot_readvariableop_resource9
5sequential_7_dense_24_biasadd_readvariableop_resource@
<layer_normalization_15_batchnorm_mul_readvariableop_resource<
8layer_normalization_15_batchnorm_readvariableop_resource
identity¢/layer_normalization_14/batchnorm/ReadVariableOp¢3layer_normalization_14/batchnorm/mul/ReadVariableOp¢/layer_normalization_15/batchnorm/ReadVariableOp¢3layer_normalization_15/batchnorm/mul/ReadVariableOp¢:multi_head_attention_7/attention_output/add/ReadVariableOp¢Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_7/key/add/ReadVariableOp¢7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_7/query/add/ReadVariableOp¢9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_7/value/add/ReadVariableOp¢9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp¢,sequential_7/dense_23/BiasAdd/ReadVariableOp¢.sequential_7/dense_23/Tensordot/ReadVariableOp¢,sequential_7/dense_24/BiasAdd/ReadVariableOp¢.sequential_7/dense_24/Tensordot/ReadVariableOpý
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_7/query/einsum/EinsumEinsuminputsAmulti_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2,
*multi_head_attention_7/query/einsum/EinsumÛ
/multi_head_attention_7/query/add/ReadVariableOpReadVariableOp8multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/query/add/ReadVariableOpõ
 multi_head_attention_7/query/addAddV23multi_head_attention_7/query/einsum/Einsum:output:07multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2"
 multi_head_attention_7/query/add÷
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_7/key/einsum/EinsumEinsuminputs?multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2*
(multi_head_attention_7/key/einsum/EinsumÕ
-multi_head_attention_7/key/add/ReadVariableOpReadVariableOp6multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_7/key/add/ReadVariableOpí
multi_head_attention_7/key/addAddV21multi_head_attention_7/key/einsum/Einsum:output:05multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2 
multi_head_attention_7/key/addý
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_7/value/einsum/EinsumEinsuminputsAmulti_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2,
*multi_head_attention_7/value/einsum/EinsumÛ
/multi_head_attention_7/value/add/ReadVariableOpReadVariableOp8multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/value/add/ReadVariableOpõ
 multi_head_attention_7/value/addAddV23multi_head_attention_7/value/einsum/Einsum:output:07multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2"
 multi_head_attention_7/value/add
multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_7/Mul/yÆ
multi_head_attention_7/MulMul$multi_head_attention_7/query/add:z:0%multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
multi_head_attention_7/Mulü
$multi_head_attention_7/einsum/EinsumEinsum"multi_head_attention_7/key/add:z:0multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
equationaecd,abcd->acbe2&
$multi_head_attention_7/einsum/EinsumÄ
&multi_head_attention_7/softmax/SoftmaxSoftmax-multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2(
&multi_head_attention_7/softmax/Softmax¡
,multi_head_attention_7/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,multi_head_attention_7/dropout/dropout/Const
*multi_head_attention_7/dropout/dropout/MulMul0multi_head_attention_7/softmax/Softmax:softmax:05multi_head_attention_7/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2,
*multi_head_attention_7/dropout/dropout/Mul¼
,multi_head_attention_7/dropout/dropout/ShapeShape0multi_head_attention_7/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_7/dropout/dropout/Shape¥
Cmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_7/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
dtype0*

seed*2E
Cmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniform³
5multi_head_attention_7/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_7/dropout/dropout/GreaterEqual/yÂ
3multi_head_attention_7/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_7/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB25
3multi_head_attention_7/dropout/dropout/GreaterEqualä
+multi_head_attention_7/dropout/dropout/CastCast7multi_head_attention_7/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2-
+multi_head_attention_7/dropout/dropout/Castþ
,multi_head_attention_7/dropout/dropout/Mul_1Mul.multi_head_attention_7/dropout/dropout/Mul:z:0/multi_head_attention_7/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2.
,multi_head_attention_7/dropout/dropout/Mul_1
&multi_head_attention_7/einsum_1/EinsumEinsum0multi_head_attention_7/dropout/dropout/Mul_1:z:0$multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationacbe,aecd->abcd2(
&multi_head_attention_7/einsum_1/Einsum
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_7/attention_output/einsum/EinsumEinsum/multi_head_attention_7/einsum_1/Einsum:output:0Lmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabcd,cde->abe27
5multi_head_attention_7/attention_output/einsum/Einsumø
:multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_7/attention_output/add/ReadVariableOp
+multi_head_attention_7/attention_output/addAddV2>multi_head_attention_7/attention_output/einsum/Einsum:output:0Bmulti_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2-
+multi_head_attention_7/attention_output/addy
dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_20/dropout/ConstÁ
dropout_20/dropout/MulMul/multi_head_attention_7/attention_output/add:z:0!dropout_20/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dropout_20/dropout/Mul
dropout_20/dropout/ShapeShape/multi_head_attention_7/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_20/dropout/Shapeò
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
dtype0*

seed**
seed221
/dropout_20/dropout/random_uniform/RandomUniform
!dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_20/dropout/GreaterEqual/yî
dropout_20/dropout/GreaterEqualGreaterEqual8dropout_20/dropout/random_uniform/RandomUniform:output:0*dropout_20/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2!
dropout_20/dropout/GreaterEqual¤
dropout_20/dropout/CastCast#dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dropout_20/dropout/Castª
dropout_20/dropout/Mul_1Muldropout_20/dropout/Mul:z:0dropout_20/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dropout_20/dropout/Mul_1o
addAddV2inputsdropout_20/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
add¸
5layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_14/moments/mean/reduction_indicesâ
#layer_normalization_14/moments/meanMeanadd:z:0>layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2%
#layer_normalization_14/moments/meanÎ
+layer_normalization_14/moments/StopGradientStopGradient,layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2-
+layer_normalization_14/moments/StopGradientî
0layer_normalization_14/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 22
0layer_normalization_14/moments/SquaredDifferenceÀ
9layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_14/moments/variance/reduction_indices
'layer_normalization_14/moments/varianceMean4layer_normalization_14/moments/SquaredDifference:z:0Blayer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2)
'layer_normalization_14/moments/variance
&layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_14/batchnorm/add/yî
$layer_normalization_14/batchnorm/addAddV20layer_normalization_14/moments/variance:output:0/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2&
$layer_normalization_14/batchnorm/add¹
&layer_normalization_14/batchnorm/RsqrtRsqrt(layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2(
&layer_normalization_14/batchnorm/Rsqrtã
3layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_14/batchnorm/mul/ReadVariableOpò
$layer_normalization_14/batchnorm/mulMul*layer_normalization_14/batchnorm/Rsqrt:y:0;layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2&
$layer_normalization_14/batchnorm/mulÀ
&layer_normalization_14/batchnorm/mul_1Muladd:z:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_14/batchnorm/mul_1å
&layer_normalization_14/batchnorm/mul_2Mul,layer_normalization_14/moments/mean:output:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_14/batchnorm/mul_2×
/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_14/batchnorm/ReadVariableOpî
$layer_normalization_14/batchnorm/subSub7layer_normalization_14/batchnorm/ReadVariableOp:value:0*layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2&
$layer_normalization_14/batchnorm/subå
&layer_normalization_14/batchnorm/add_1AddV2*layer_normalization_14/batchnorm/mul_1:z:0(layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_14/batchnorm/add_1Ø
.sequential_7/dense_23/Tensordot/ReadVariableOpReadVariableOp7sequential_7_dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_7/dense_23/Tensordot/ReadVariableOp
$sequential_7/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_7/dense_23/Tensordot/axes
$sequential_7/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_7/dense_23/Tensordot/free¨
%sequential_7/dense_23/Tensordot/ShapeShape*layer_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/Shape 
-sequential_7/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_23/Tensordot/GatherV2/axis¿
(sequential_7/dense_23/Tensordot/GatherV2GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/free:output:06sequential_7/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/GatherV2¤
/sequential_7/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_23/Tensordot/GatherV2_1/axisÅ
*sequential_7/dense_23/Tensordot/GatherV2_1GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/axes:output:08sequential_7/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_7/dense_23/Tensordot/GatherV2_1
%sequential_7/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_7/dense_23/Tensordot/ConstØ
$sequential_7/dense_23/Tensordot/ProdProd1sequential_7/dense_23/Tensordot/GatherV2:output:0.sequential_7/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_7/dense_23/Tensordot/Prod
'sequential_7/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_23/Tensordot/Const_1à
&sequential_7/dense_23/Tensordot/Prod_1Prod3sequential_7/dense_23/Tensordot/GatherV2_1:output:00sequential_7/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_7/dense_23/Tensordot/Prod_1
+sequential_7/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_7/dense_23/Tensordot/concat/axis
&sequential_7/dense_23/Tensordot/concatConcatV2-sequential_7/dense_23/Tensordot/free:output:0-sequential_7/dense_23/Tensordot/axes:output:04sequential_7/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/dense_23/Tensordot/concatä
%sequential_7/dense_23/Tensordot/stackPack-sequential_7/dense_23/Tensordot/Prod:output:0/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/stackö
)sequential_7/dense_23/Tensordot/transpose	Transpose*layer_normalization_14/batchnorm/add_1:z:0/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2+
)sequential_7/dense_23/Tensordot/transpose÷
'sequential_7/dense_23/Tensordot/ReshapeReshape-sequential_7/dense_23/Tensordot/transpose:y:0.sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_7/dense_23/Tensordot/Reshapeö
&sequential_7/dense_23/Tensordot/MatMulMatMul0sequential_7/dense_23/Tensordot/Reshape:output:06sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential_7/dense_23/Tensordot/MatMul
'sequential_7/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_7/dense_23/Tensordot/Const_2 
-sequential_7/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_23/Tensordot/concat_1/axis«
(sequential_7/dense_23/Tensordot/concat_1ConcatV21sequential_7/dense_23/Tensordot/GatherV2:output:00sequential_7/dense_23/Tensordot/Const_2:output:06sequential_7/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/concat_1è
sequential_7/dense_23/TensordotReshape0sequential_7/dense_23/Tensordot/MatMul:product:01sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2!
sequential_7/dense_23/TensordotÎ
,sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_7/dense_23/BiasAdd/ReadVariableOpß
sequential_7/dense_23/BiasAddBiasAdd(sequential_7/dense_23/Tensordot:output:04sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2
sequential_7/dense_23/BiasAdd
sequential_7/dense_23/ReluRelu&sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2
sequential_7/dense_23/ReluØ
.sequential_7/dense_24/Tensordot/ReadVariableOpReadVariableOp7sequential_7_dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_7/dense_24/Tensordot/ReadVariableOp
$sequential_7/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_7/dense_24/Tensordot/axes
$sequential_7/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_7/dense_24/Tensordot/free¦
%sequential_7/dense_24/Tensordot/ShapeShape(sequential_7/dense_23/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/Shape 
-sequential_7/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_24/Tensordot/GatherV2/axis¿
(sequential_7/dense_24/Tensordot/GatherV2GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/free:output:06sequential_7/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/GatherV2¤
/sequential_7/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_24/Tensordot/GatherV2_1/axisÅ
*sequential_7/dense_24/Tensordot/GatherV2_1GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/axes:output:08sequential_7/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_7/dense_24/Tensordot/GatherV2_1
%sequential_7/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_7/dense_24/Tensordot/ConstØ
$sequential_7/dense_24/Tensordot/ProdProd1sequential_7/dense_24/Tensordot/GatherV2:output:0.sequential_7/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_7/dense_24/Tensordot/Prod
'sequential_7/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_24/Tensordot/Const_1à
&sequential_7/dense_24/Tensordot/Prod_1Prod3sequential_7/dense_24/Tensordot/GatherV2_1:output:00sequential_7/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_7/dense_24/Tensordot/Prod_1
+sequential_7/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_7/dense_24/Tensordot/concat/axis
&sequential_7/dense_24/Tensordot/concatConcatV2-sequential_7/dense_24/Tensordot/free:output:0-sequential_7/dense_24/Tensordot/axes:output:04sequential_7/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/dense_24/Tensordot/concatä
%sequential_7/dense_24/Tensordot/stackPack-sequential_7/dense_24/Tensordot/Prod:output:0/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/stackô
)sequential_7/dense_24/Tensordot/transpose	Transpose(sequential_7/dense_23/Relu:activations:0/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2+
)sequential_7/dense_24/Tensordot/transpose÷
'sequential_7/dense_24/Tensordot/ReshapeReshape-sequential_7/dense_24/Tensordot/transpose:y:0.sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_7/dense_24/Tensordot/Reshapeö
&sequential_7/dense_24/Tensordot/MatMulMatMul0sequential_7/dense_24/Tensordot/Reshape:output:06sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_7/dense_24/Tensordot/MatMul
'sequential_7/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_24/Tensordot/Const_2 
-sequential_7/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_24/Tensordot/concat_1/axis«
(sequential_7/dense_24/Tensordot/concat_1ConcatV21sequential_7/dense_24/Tensordot/GatherV2:output:00sequential_7/dense_24/Tensordot/Const_2:output:06sequential_7/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/concat_1è
sequential_7/dense_24/TensordotReshape0sequential_7/dense_24/Tensordot/MatMul:product:01sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2!
sequential_7/dense_24/TensordotÎ
,sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_7/dense_24/BiasAdd/ReadVariableOpß
sequential_7/dense_24/BiasAddBiasAdd(sequential_7/dense_24/Tensordot:output:04sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
sequential_7/dense_24/BiasAddy
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_21/dropout/Const¸
dropout_21/dropout/MulMul&sequential_7/dense_24/BiasAdd:output:0!dropout_21/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dropout_21/dropout/Mul
dropout_21/dropout/ShapeShape&sequential_7/dense_24/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_21/dropout/Shapeò
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
dtype0*

seed**
seed221
/dropout_21/dropout/random_uniform/RandomUniform
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_21/dropout/GreaterEqual/yî
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2!
dropout_21/dropout/GreaterEqual¤
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dropout_21/dropout/Castª
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dropout_21/dropout/Mul_1
add_1AddV2*layer_normalization_14/batchnorm/add_1:z:0dropout_21/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
add_1¸
5layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_15/moments/mean/reduction_indicesä
#layer_normalization_15/moments/meanMean	add_1:z:0>layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2%
#layer_normalization_15/moments/meanÎ
+layer_normalization_15/moments/StopGradientStopGradient,layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2-
+layer_normalization_15/moments/StopGradientð
0layer_normalization_15/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 22
0layer_normalization_15/moments/SquaredDifferenceÀ
9layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_15/moments/variance/reduction_indices
'layer_normalization_15/moments/varianceMean4layer_normalization_15/moments/SquaredDifference:z:0Blayer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2)
'layer_normalization_15/moments/variance
&layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_15/batchnorm/add/yî
$layer_normalization_15/batchnorm/addAddV20layer_normalization_15/moments/variance:output:0/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2&
$layer_normalization_15/batchnorm/add¹
&layer_normalization_15/batchnorm/RsqrtRsqrt(layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2(
&layer_normalization_15/batchnorm/Rsqrtã
3layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_15/batchnorm/mul/ReadVariableOpò
$layer_normalization_15/batchnorm/mulMul*layer_normalization_15/batchnorm/Rsqrt:y:0;layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2&
$layer_normalization_15/batchnorm/mulÂ
&layer_normalization_15/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_15/batchnorm/mul_1å
&layer_normalization_15/batchnorm/mul_2Mul,layer_normalization_15/moments/mean:output:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_15/batchnorm/mul_2×
/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_15/batchnorm/ReadVariableOpî
$layer_normalization_15/batchnorm/subSub7layer_normalization_15/batchnorm/ReadVariableOp:value:0*layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2&
$layer_normalization_15/batchnorm/subå
&layer_normalization_15/batchnorm/add_1AddV2*layer_normalization_15/batchnorm/mul_1:z:0(layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_15/batchnorm/add_1Ü
IdentityIdentity*layer_normalization_15/batchnorm/add_1:z:00^layer_normalization_14/batchnorm/ReadVariableOp4^layer_normalization_14/batchnorm/mul/ReadVariableOp0^layer_normalization_15/batchnorm/ReadVariableOp4^layer_normalization_15/batchnorm/mul/ReadVariableOp;^multi_head_attention_7/attention_output/add/ReadVariableOpE^multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_7/key/add/ReadVariableOp8^multi_head_attention_7/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/query/add/ReadVariableOp:^multi_head_attention_7/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/value/add/ReadVariableOp:^multi_head_attention_7/value/einsum/Einsum/ReadVariableOp-^sequential_7/dense_23/BiasAdd/ReadVariableOp/^sequential_7/dense_23/Tensordot/ReadVariableOp-^sequential_7/dense_24/BiasAdd/ReadVariableOp/^sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿB ::::::::::::::::2b
/layer_normalization_14/batchnorm/ReadVariableOp/layer_normalization_14/batchnorm/ReadVariableOp2j
3layer_normalization_14/batchnorm/mul/ReadVariableOp3layer_normalization_14/batchnorm/mul/ReadVariableOp2b
/layer_normalization_15/batchnorm/ReadVariableOp/layer_normalization_15/batchnorm/ReadVariableOp2j
3layer_normalization_15/batchnorm/mul/ReadVariableOp3layer_normalization_15/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_7/attention_output/add/ReadVariableOp:multi_head_attention_7/attention_output/add/ReadVariableOp2
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_7/key/add/ReadVariableOp-multi_head_attention_7/key/add/ReadVariableOp2r
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/query/add/ReadVariableOp/multi_head_attention_7/query/add/ReadVariableOp2v
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/value/add/ReadVariableOp/multi_head_attention_7/value/add/ReadVariableOp2v
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp2\
,sequential_7/dense_23/BiasAdd/ReadVariableOp,sequential_7/dense_23/BiasAdd/ReadVariableOp2`
.sequential_7/dense_23/Tensordot/ReadVariableOp.sequential_7/dense_23/Tensordot/ReadVariableOp2\
,sequential_7/dense_24/BiasAdd/ReadVariableOp,sequential_7/dense_24/BiasAdd/ReadVariableOp2`
.sequential_7/dense_24/Tensordot/ReadVariableOp.sequential_7/dense_24/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
ï
~
)__inference_dense_23_layer_call_fn_198011

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
:ÿÿÿÿÿÿÿÿÿB@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_1946902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿB ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
ö
l
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_194354

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
íX

C__inference_model_3_layer_call_and_return_conditional_losses_195780
input_7
input_8)
%token_and_position_embedding_3_195690)
%token_and_position_embedding_3_195692
conv1d_6_195695
conv1d_6_195697
conv1d_7_195701
conv1d_7_195703 
batch_normalization_6_195708 
batch_normalization_6_195710 
batch_normalization_6_195712 
batch_normalization_6_195714 
batch_normalization_7_195717 
batch_normalization_7_195719 
batch_normalization_7_195721 
batch_normalization_7_195723
transformer_block_7_195727
transformer_block_7_195729
transformer_block_7_195731
transformer_block_7_195733
transformer_block_7_195735
transformer_block_7_195737
transformer_block_7_195739
transformer_block_7_195741
transformer_block_7_195743
transformer_block_7_195745
transformer_block_7_195747
transformer_block_7_195749
transformer_block_7_195751
transformer_block_7_195753
transformer_block_7_195755
transformer_block_7_195757
dense_25_195762
dense_25_195764
dense_26_195768
dense_26_195770
dense_27_195774
dense_27_195776
identity¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢ conv1d_6/StatefulPartitionedCall¢ conv1d_7/StatefulPartitionedCall¢ dense_25/StatefulPartitionedCall¢ dense_26/StatefulPartitionedCall¢ dense_27/StatefulPartitionedCall¢6token_and_position_embedding_3/StatefulPartitionedCall¢+transformer_block_7/StatefulPartitionedCall
6token_and_position_embedding_3/StatefulPartitionedCallStatefulPartitionedCallinput_7%token_and_position_embedding_3_195690%token_and_position_embedding_3_195692*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_19485128
6token_and_position_embedding_3/StatefulPartitionedCallÖ
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0conv1d_6_195695conv1d_6_195697*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_1948832"
 conv1d_6/StatefulPartitionedCall 
#average_pooling1d_9/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_1943392%
#average_pooling1d_9/PartitionedCallÂ
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_9/PartitionedCall:output:0conv1d_7_195701conv1d_7_195703*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_1949162"
 conv1d_7/StatefulPartitionedCall¸
$average_pooling1d_11/PartitionedCallPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_1943692&
$average_pooling1d_11/PartitionedCall¢
$average_pooling1d_10/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_1943542&
$average_pooling1d_10/PartitionedCallÃ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_10/PartitionedCall:output:0batch_normalization_6_195708batch_normalization_6_195710batch_normalization_6_195712batch_normalization_6_195714*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1949892/
-batch_normalization_6/StatefulPartitionedCallÃ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_11/PartitionedCall:output:0batch_normalization_7_195717batch_normalization_7_195719batch_normalization_7_195721batch_normalization_7_195723*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1950802/
-batch_normalization_7/StatefulPartitionedCall»
add_3/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_1951222
add_3/PartitionedCall
+transformer_block_7/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0transformer_block_7_195727transformer_block_7_195729transformer_block_7_195731transformer_block_7_195733transformer_block_7_195735transformer_block_7_195737transformer_block_7_195739transformer_block_7_195741transformer_block_7_195743transformer_block_7_195745transformer_block_7_195747transformer_block_7_195749transformer_block_7_195751transformer_block_7_195753transformer_block_7_195755transformer_block_7_195757*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_1954062-
+transformer_block_7/StatefulPartitionedCall
flatten_3/PartitionedCallPartitionedCall4transformer_block_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1955212
flatten_3/PartitionedCall
concatenate_3/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0input_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_1955362
concatenate_3/PartitionedCall·
 dense_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_25_195762dense_25_195764*
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
D__inference_dense_25_layer_call_and_return_conditional_losses_1955562"
 dense_25/StatefulPartitionedCall
dropout_22/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
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
F__inference_dropout_22_layer_call_and_return_conditional_losses_1955892
dropout_22/PartitionedCall´
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_26_195768dense_26_195770*
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
D__inference_dense_26_layer_call_and_return_conditional_losses_1956132"
 dense_26/StatefulPartitionedCall
dropout_23/PartitionedCallPartitionedCall)dense_26/StatefulPartitionedCall:output:0*
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
F__inference_dropout_23_layer_call_and_return_conditional_losses_1956462
dropout_23/PartitionedCall´
 dense_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_27_195774dense_27_195776*
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
D__inference_dense_27_layer_call_and_return_conditional_losses_1956692"
 dense_27/StatefulPartitionedCalló
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall7^token_and_position_embedding_3/StatefulPartitionedCall,^transformer_block_7/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Í
_input_shapes»
¸:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2p
6token_and_position_embedding_3/StatefulPartitionedCall6token_and_position_embedding_3/StatefulPartitionedCall2Z
+transformer_block_7/StatefulPartitionedCall+transformer_block_7/StatefulPartitionedCall:R N
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!
_user_specified_name	input_7:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_8
·
k
A__inference_add_3_layer_call_and_return_conditional_losses_195122

inputs
inputs_1
identity[
addAddV2inputsinputs_1*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿB :ÿÿÿÿÿÿÿÿÿB :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
¥
d
+__inference_dropout_23_layer_call_fn_197807

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
F__inference_dropout_23_layer_call_and_return_conditional_losses_1956412
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
î
©
6__inference_batch_normalization_7_layer_call_fn_197251

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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1946442
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
¢
÷
D__inference_conv1d_6_layer_call_and_return_conditional_losses_194883

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
:ÿÿÿÿÿÿÿÿÿ  2
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
conv1d/ExpandDims_1¸
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2	
BiasAdd^
ReluReluBiasAdd:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
Reluª
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ó0
È
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_197205

inputs
assignmovingavg_197180
assignmovingavg_1_197186)
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
loc:@AssignMovingAvg/197180*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_197180*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/197180*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/197180*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_197180AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/197180*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/197186*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_197186*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/197186*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/197186*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_197186AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/197186*
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
é

H__inference_sequential_7_layer_call_and_return_conditional_losses_194784

inputs
dense_23_194773
dense_23_194775
dense_24_194778
dense_24_194780
identity¢ dense_23/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCall
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinputsdense_23_194773dense_23_194775*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_1946902"
 dense_23/StatefulPartitionedCall¾
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_194778dense_24_194780*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_1947362"
 dense_24/StatefulPartitionedCallÇ
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
ß
~
)__inference_dense_26_layer_call_fn_197785

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
D__inference_dense_26_layer_call_and_return_conditional_losses_1956132
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
µ
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_195521

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿB :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
µ
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_197700

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿB :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs

è(
!__inference__wrapped_model_194330
input_7
input_8N
Jmodel_3_token_and_position_embedding_3_embedding_7_embedding_lookup_194099N
Jmodel_3_token_and_position_embedding_3_embedding_6_embedding_lookup_194105@
<model_3_conv1d_6_conv1d_expanddims_1_readvariableop_resource4
0model_3_conv1d_6_biasadd_readvariableop_resource@
<model_3_conv1d_7_conv1d_expanddims_1_readvariableop_resource4
0model_3_conv1d_7_biasadd_readvariableop_resourceC
?model_3_batch_normalization_6_batchnorm_readvariableop_resourceG
Cmodel_3_batch_normalization_6_batchnorm_mul_readvariableop_resourceE
Amodel_3_batch_normalization_6_batchnorm_readvariableop_1_resourceE
Amodel_3_batch_normalization_6_batchnorm_readvariableop_2_resourceC
?model_3_batch_normalization_7_batchnorm_readvariableop_resourceG
Cmodel_3_batch_normalization_7_batchnorm_mul_readvariableop_resourceE
Amodel_3_batch_normalization_7_batchnorm_readvariableop_1_resourceE
Amodel_3_batch_normalization_7_batchnorm_readvariableop_2_resourceb
^model_3_transformer_block_7_multi_head_attention_7_query_einsum_einsum_readvariableop_resourceX
Tmodel_3_transformer_block_7_multi_head_attention_7_query_add_readvariableop_resource`
\model_3_transformer_block_7_multi_head_attention_7_key_einsum_einsum_readvariableop_resourceV
Rmodel_3_transformer_block_7_multi_head_attention_7_key_add_readvariableop_resourceb
^model_3_transformer_block_7_multi_head_attention_7_value_einsum_einsum_readvariableop_resourceX
Tmodel_3_transformer_block_7_multi_head_attention_7_value_add_readvariableop_resourcem
imodel_3_transformer_block_7_multi_head_attention_7_attention_output_einsum_einsum_readvariableop_resourcec
_model_3_transformer_block_7_multi_head_attention_7_attention_output_add_readvariableop_resource\
Xmodel_3_transformer_block_7_layer_normalization_14_batchnorm_mul_readvariableop_resourceX
Tmodel_3_transformer_block_7_layer_normalization_14_batchnorm_readvariableop_resourceW
Smodel_3_transformer_block_7_sequential_7_dense_23_tensordot_readvariableop_resourceU
Qmodel_3_transformer_block_7_sequential_7_dense_23_biasadd_readvariableop_resourceW
Smodel_3_transformer_block_7_sequential_7_dense_24_tensordot_readvariableop_resourceU
Qmodel_3_transformer_block_7_sequential_7_dense_24_biasadd_readvariableop_resource\
Xmodel_3_transformer_block_7_layer_normalization_15_batchnorm_mul_readvariableop_resourceX
Tmodel_3_transformer_block_7_layer_normalization_15_batchnorm_readvariableop_resource3
/model_3_dense_25_matmul_readvariableop_resource4
0model_3_dense_25_biasadd_readvariableop_resource3
/model_3_dense_26_matmul_readvariableop_resource4
0model_3_dense_26_biasadd_readvariableop_resource3
/model_3_dense_27_matmul_readvariableop_resource4
0model_3_dense_27_biasadd_readvariableop_resource
identity¢6model_3/batch_normalization_6/batchnorm/ReadVariableOp¢8model_3/batch_normalization_6/batchnorm/ReadVariableOp_1¢8model_3/batch_normalization_6/batchnorm/ReadVariableOp_2¢:model_3/batch_normalization_6/batchnorm/mul/ReadVariableOp¢6model_3/batch_normalization_7/batchnorm/ReadVariableOp¢8model_3/batch_normalization_7/batchnorm/ReadVariableOp_1¢8model_3/batch_normalization_7/batchnorm/ReadVariableOp_2¢:model_3/batch_normalization_7/batchnorm/mul/ReadVariableOp¢'model_3/conv1d_6/BiasAdd/ReadVariableOp¢3model_3/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp¢'model_3/conv1d_7/BiasAdd/ReadVariableOp¢3model_3/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp¢'model_3/dense_25/BiasAdd/ReadVariableOp¢&model_3/dense_25/MatMul/ReadVariableOp¢'model_3/dense_26/BiasAdd/ReadVariableOp¢&model_3/dense_26/MatMul/ReadVariableOp¢'model_3/dense_27/BiasAdd/ReadVariableOp¢&model_3/dense_27/MatMul/ReadVariableOp¢Cmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup¢Cmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup¢Kmodel_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp¢Omodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp¢Kmodel_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp¢Omodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp¢Vmodel_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp¢`model_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp¢Imodel_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOp¢Smodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp¢Kmodel_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOp¢Umodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp¢Kmodel_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOp¢Umodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp¢Hmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp¢Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp¢Hmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp¢Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp
,model_3/token_and_position_embedding_3/ShapeShapeinput_7*
T0*
_output_shapes
:2.
,model_3/token_and_position_embedding_3/ShapeË
:model_3/token_and_position_embedding_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2<
:model_3/token_and_position_embedding_3/strided_slice/stackÆ
<model_3/token_and_position_embedding_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<model_3/token_and_position_embedding_3/strided_slice/stack_1Æ
<model_3/token_and_position_embedding_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model_3/token_and_position_embedding_3/strided_slice/stack_2Ì
4model_3/token_and_position_embedding_3/strided_sliceStridedSlice5model_3/token_and_position_embedding_3/Shape:output:0Cmodel_3/token_and_position_embedding_3/strided_slice/stack:output:0Emodel_3/token_and_position_embedding_3/strided_slice/stack_1:output:0Emodel_3/token_and_position_embedding_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4model_3/token_and_position_embedding_3/strided_sliceª
2model_3/token_and_position_embedding_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : 24
2model_3/token_and_position_embedding_3/range/startª
2model_3/token_and_position_embedding_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :24
2model_3/token_and_position_embedding_3/range/deltaÃ
,model_3/token_and_position_embedding_3/rangeRange;model_3/token_and_position_embedding_3/range/start:output:0=model_3/token_and_position_embedding_3/strided_slice:output:0;model_3/token_and_position_embedding_3/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,model_3/token_and_position_embedding_3/rangeò
Cmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookupResourceGatherJmodel_3_token_and_position_embedding_3_embedding_7_embedding_lookup_1940995model_3/token_and_position_embedding_3/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*]
_classS
QOloc:@model_3/token_and_position_embedding_3/embedding_7/embedding_lookup/194099*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02E
Cmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookupµ
Lmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup/IdentityIdentityLmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*]
_classS
QOloc:@model_3/token_and_position_embedding_3/embedding_7/embedding_lookup/194099*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2N
Lmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup/Identityµ
Nmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1IdentityUmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2P
Nmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1Æ
7model_3/token_and_position_embedding_3/embedding_6/CastCastinput_7*

DstT0*

SrcT0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 29
7model_3/token_and_position_embedding_3/embedding_6/Castþ
Cmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookupResourceGatherJmodel_3_token_and_position_embedding_3_embedding_6_embedding_lookup_194105;model_3/token_and_position_embedding_3/embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*]
_classS
QOloc:@model_3/token_and_position_embedding_3/embedding_6/embedding_lookup/194105*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype02E
Cmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup»
Lmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup/IdentityIdentityLmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*]
_classS
QOloc:@model_3/token_and_position_embedding_3/embedding_6/embedding_lookup/194105*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2N
Lmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup/Identity»
Nmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1IdentityUmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2P
Nmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1Ë
*model_3/token_and_position_embedding_3/addAddV2Wmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1:output:0Wmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2,
*model_3/token_and_position_embedding_3/add
&model_3/conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2(
&model_3/conv1d_6/conv1d/ExpandDims/dimó
"model_3/conv1d_6/conv1d/ExpandDims
ExpandDims.model_3/token_and_position_embedding_3/add:z:0/model_3/conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2$
"model_3/conv1d_6/conv1d/ExpandDimsë
3model_3/conv1d_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<model_3_conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype025
3model_3/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp
(model_3/conv1d_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_3/conv1d_6/conv1d/ExpandDims_1/dimû
$model_3/conv1d_6/conv1d/ExpandDims_1
ExpandDims;model_3/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:01model_3/conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2&
$model_3/conv1d_6/conv1d/ExpandDims_1ü
model_3/conv1d_6/conv1dConv2D+model_3/conv1d_6/conv1d/ExpandDims:output:0-model_3/conv1d_6/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
model_3/conv1d_6/conv1dÇ
model_3/conv1d_6/conv1d/SqueezeSqueeze model_3/conv1d_6/conv1d:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2!
model_3/conv1d_6/conv1d/Squeeze¿
'model_3/conv1d_6/BiasAdd/ReadVariableOpReadVariableOp0model_3_conv1d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_3/conv1d_6/BiasAdd/ReadVariableOpÒ
model_3/conv1d_6/BiasAddBiasAdd(model_3/conv1d_6/conv1d/Squeeze:output:0/model_3/conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
model_3/conv1d_6/BiasAdd
model_3/conv1d_6/ReluRelu!model_3/conv1d_6/BiasAdd:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
model_3/conv1d_6/Relu
*model_3/average_pooling1d_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_3/average_pooling1d_9/ExpandDims/dimô
&model_3/average_pooling1d_9/ExpandDims
ExpandDims#model_3/conv1d_6/Relu:activations:03model_3/average_pooling1d_9/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2(
&model_3/average_pooling1d_9/ExpandDimsý
#model_3/average_pooling1d_9/AvgPoolAvgPool/model_3/average_pooling1d_9/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2%
#model_3/average_pooling1d_9/AvgPoolÑ
#model_3/average_pooling1d_9/SqueezeSqueeze,model_3/average_pooling1d_9/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims
2%
#model_3/average_pooling1d_9/Squeeze
&model_3/conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2(
&model_3/conv1d_7/conv1d/ExpandDims/dimð
"model_3/conv1d_7/conv1d/ExpandDims
ExpandDims,model_3/average_pooling1d_9/Squeeze:output:0/model_3/conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"model_3/conv1d_7/conv1d/ExpandDimsë
3model_3/conv1d_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<model_3_conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype025
3model_3/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp
(model_3/conv1d_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_3/conv1d_7/conv1d/ExpandDims_1/dimû
$model_3/conv1d_7/conv1d/ExpandDims_1
ExpandDims;model_3/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:01model_3/conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2&
$model_3/conv1d_7/conv1d/ExpandDims_1û
model_3/conv1d_7/conv1dConv2D+model_3/conv1d_7/conv1d/ExpandDims:output:0-model_3/conv1d_7/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
model_3/conv1d_7/conv1dÆ
model_3/conv1d_7/conv1d/SqueezeSqueeze model_3/conv1d_7/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2!
model_3/conv1d_7/conv1d/Squeeze¿
'model_3/conv1d_7/BiasAdd/ReadVariableOpReadVariableOp0model_3_conv1d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_3/conv1d_7/BiasAdd/ReadVariableOpÑ
model_3/conv1d_7/BiasAddBiasAdd(model_3/conv1d_7/conv1d/Squeeze:output:0/model_3/conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_3/conv1d_7/BiasAdd
model_3/conv1d_7/ReluRelu!model_3/conv1d_7/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_3/conv1d_7/Relu
+model_3/average_pooling1d_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+model_3/average_pooling1d_11/ExpandDims/dim
'model_3/average_pooling1d_11/ExpandDims
ExpandDims.model_3/token_and_position_embedding_3/add:z:04model_3/average_pooling1d_11/ExpandDims/dim:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2)
'model_3/average_pooling1d_11/ExpandDims
$model_3/average_pooling1d_11/AvgPoolAvgPool0model_3/average_pooling1d_11/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
ksize	
¬*
paddingVALID*
strides	
¬2&
$model_3/average_pooling1d_11/AvgPoolÓ
$model_3/average_pooling1d_11/SqueezeSqueeze-model_3/average_pooling1d_11/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
squeeze_dims
2&
$model_3/average_pooling1d_11/Squeeze
+model_3/average_pooling1d_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+model_3/average_pooling1d_10/ExpandDims/dimö
'model_3/average_pooling1d_10/ExpandDims
ExpandDims#model_3/conv1d_7/Relu:activations:04model_3/average_pooling1d_10/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'model_3/average_pooling1d_10/ExpandDimsÿ
$model_3/average_pooling1d_10/AvgPoolAvgPool0model_3/average_pooling1d_10/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
ksize

*
paddingVALID*
strides

2&
$model_3/average_pooling1d_10/AvgPoolÓ
$model_3/average_pooling1d_10/SqueezeSqueeze-model_3/average_pooling1d_10/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
squeeze_dims
2&
$model_3/average_pooling1d_10/Squeezeì
6model_3/batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp?model_3_batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype028
6model_3/batch_normalization_6/batchnorm/ReadVariableOp£
-model_3/batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-model_3/batch_normalization_6/batchnorm/add/y
+model_3/batch_normalization_6/batchnorm/addAddV2>model_3/batch_normalization_6/batchnorm/ReadVariableOp:value:06model_3/batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2-
+model_3/batch_normalization_6/batchnorm/add½
-model_3/batch_normalization_6/batchnorm/RsqrtRsqrt/model_3/batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-model_3/batch_normalization_6/batchnorm/Rsqrtø
:model_3/batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_3_batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:model_3/batch_normalization_6/batchnorm/mul/ReadVariableOpý
+model_3/batch_normalization_6/batchnorm/mulMul1model_3/batch_normalization_6/batchnorm/Rsqrt:y:0Bmodel_3/batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+model_3/batch_normalization_6/batchnorm/mulû
-model_3/batch_normalization_6/batchnorm/mul_1Mul-model_3/average_pooling1d_10/Squeeze:output:0/model_3/batch_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2/
-model_3/batch_normalization_6/batchnorm/mul_1ò
8model_3/batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_3_batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model_3/batch_normalization_6/batchnorm/ReadVariableOp_1ý
-model_3/batch_normalization_6/batchnorm/mul_2Mul@model_3/batch_normalization_6/batchnorm/ReadVariableOp_1:value:0/model_3/batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-model_3/batch_normalization_6/batchnorm/mul_2ò
8model_3/batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_3_batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8model_3/batch_normalization_6/batchnorm/ReadVariableOp_2û
+model_3/batch_normalization_6/batchnorm/subSub@model_3/batch_normalization_6/batchnorm/ReadVariableOp_2:value:01model_3/batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+model_3/batch_normalization_6/batchnorm/sub
-model_3/batch_normalization_6/batchnorm/add_1AddV21model_3/batch_normalization_6/batchnorm/mul_1:z:0/model_3/batch_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2/
-model_3/batch_normalization_6/batchnorm/add_1ì
6model_3/batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp?model_3_batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype028
6model_3/batch_normalization_7/batchnorm/ReadVariableOp£
-model_3/batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-model_3/batch_normalization_7/batchnorm/add/y
+model_3/batch_normalization_7/batchnorm/addAddV2>model_3/batch_normalization_7/batchnorm/ReadVariableOp:value:06model_3/batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2-
+model_3/batch_normalization_7/batchnorm/add½
-model_3/batch_normalization_7/batchnorm/RsqrtRsqrt/model_3/batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-model_3/batch_normalization_7/batchnorm/Rsqrtø
:model_3/batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_3_batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:model_3/batch_normalization_7/batchnorm/mul/ReadVariableOpý
+model_3/batch_normalization_7/batchnorm/mulMul1model_3/batch_normalization_7/batchnorm/Rsqrt:y:0Bmodel_3/batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+model_3/batch_normalization_7/batchnorm/mulû
-model_3/batch_normalization_7/batchnorm/mul_1Mul-model_3/average_pooling1d_11/Squeeze:output:0/model_3/batch_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2/
-model_3/batch_normalization_7/batchnorm/mul_1ò
8model_3/batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_3_batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model_3/batch_normalization_7/batchnorm/ReadVariableOp_1ý
-model_3/batch_normalization_7/batchnorm/mul_2Mul@model_3/batch_normalization_7/batchnorm/ReadVariableOp_1:value:0/model_3/batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-model_3/batch_normalization_7/batchnorm/mul_2ò
8model_3/batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_3_batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8model_3/batch_normalization_7/batchnorm/ReadVariableOp_2û
+model_3/batch_normalization_7/batchnorm/subSub@model_3/batch_normalization_7/batchnorm/ReadVariableOp_2:value:01model_3/batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+model_3/batch_normalization_7/batchnorm/sub
-model_3/batch_normalization_7/batchnorm/add_1AddV21model_3/batch_normalization_7/batchnorm/mul_1:z:0/model_3/batch_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2/
-model_3/batch_normalization_7/batchnorm/add_1Ë
model_3/add_3/addAddV21model_3/batch_normalization_6/batchnorm/add_1:z:01model_3/batch_normalization_7/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
model_3/add_3/addÑ
Umodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOp^model_3_transformer_block_7_multi_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpð
Fmodel_3/transformer_block_7/multi_head_attention_7/query/einsum/EinsumEinsummodel_3/add_3/add:z:0]model_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2H
Fmodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum¯
Kmodel_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOpReadVariableOpTmodel_3_transformer_block_7_multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOpå
<model_3/transformer_block_7/multi_head_attention_7/query/addAddV2Omodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum:output:0Smodel_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2>
<model_3/transformer_block_7/multi_head_attention_7/query/addË
Smodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOp\model_3_transformer_block_7_multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02U
Smodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpê
Dmodel_3/transformer_block_7/multi_head_attention_7/key/einsum/EinsumEinsummodel_3/add_3/add:z:0[model_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2F
Dmodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum©
Imodel_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOpReadVariableOpRmodel_3_transformer_block_7_multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

: *
dtype02K
Imodel_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOpÝ
:model_3/transformer_block_7/multi_head_attention_7/key/addAddV2Mmodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum:output:0Qmodel_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2<
:model_3/transformer_block_7/multi_head_attention_7/key/addÑ
Umodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOp^model_3_transformer_block_7_multi_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpð
Fmodel_3/transformer_block_7/multi_head_attention_7/value/einsum/EinsumEinsummodel_3/add_3/add:z:0]model_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2H
Fmodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum¯
Kmodel_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOpReadVariableOpTmodel_3_transformer_block_7_multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOpå
<model_3/transformer_block_7/multi_head_attention_7/value/addAddV2Omodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum:output:0Smodel_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2>
<model_3/transformer_block_7/multi_head_attention_7/value/add¹
8model_3/transformer_block_7/multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2:
8model_3/transformer_block_7/multi_head_attention_7/Mul/y¶
6model_3/transformer_block_7/multi_head_attention_7/MulMul@model_3/transformer_block_7/multi_head_attention_7/query/add:z:0Amodel_3/transformer_block_7/multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 28
6model_3/transformer_block_7/multi_head_attention_7/Mulì
@model_3/transformer_block_7/multi_head_attention_7/einsum/EinsumEinsum>model_3/transformer_block_7/multi_head_attention_7/key/add:z:0:model_3/transformer_block_7/multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
equationaecd,abcd->acbe2B
@model_3/transformer_block_7/multi_head_attention_7/einsum/Einsum
Bmodel_3/transformer_block_7/multi_head_attention_7/softmax/SoftmaxSoftmaxImodel_3/transformer_block_7/multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2D
Bmodel_3/transformer_block_7/multi_head_attention_7/softmax/Softmax
Cmodel_3/transformer_block_7/multi_head_attention_7/dropout/IdentityIdentityLmodel_3/transformer_block_7/multi_head_attention_7/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2E
Cmodel_3/transformer_block_7/multi_head_attention_7/dropout/Identity
Bmodel_3/transformer_block_7/multi_head_attention_7/einsum_1/EinsumEinsumLmodel_3/transformer_block_7/multi_head_attention_7/dropout/Identity:output:0@model_3/transformer_block_7/multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationacbe,aecd->abcd2D
Bmodel_3/transformer_block_7/multi_head_attention_7/einsum_1/Einsumò
`model_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpimodel_3_transformer_block_7_multi_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02b
`model_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpÃ
Qmodel_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/EinsumEinsumKmodel_3/transformer_block_7/multi_head_attention_7/einsum_1/Einsum:output:0hmodel_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabcd,cde->abe2S
Qmodel_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/EinsumÌ
Vmodel_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOp_model_3_transformer_block_7_multi_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02X
Vmodel_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp
Gmodel_3/transformer_block_7/multi_head_attention_7/attention_output/addAddV2Zmodel_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum:output:0^model_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2I
Gmodel_3/transformer_block_7/multi_head_attention_7/attention_output/addñ
/model_3/transformer_block_7/dropout_20/IdentityIdentityKmodel_3/transformer_block_7/multi_head_attention_7/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 21
/model_3/transformer_block_7/dropout_20/IdentityÒ
model_3/transformer_block_7/addAddV2model_3/add_3/add:z:08model_3/transformer_block_7/dropout_20/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2!
model_3/transformer_block_7/addð
Qmodel_3/transformer_block_7/layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_3/transformer_block_7/layer_normalization_14/moments/mean/reduction_indicesÒ
?model_3/transformer_block_7/layer_normalization_14/moments/meanMean#model_3/transformer_block_7/add:z:0Zmodel_3/transformer_block_7/layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2A
?model_3/transformer_block_7/layer_normalization_14/moments/mean¢
Gmodel_3/transformer_block_7/layer_normalization_14/moments/StopGradientStopGradientHmodel_3/transformer_block_7/layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2I
Gmodel_3/transformer_block_7/layer_normalization_14/moments/StopGradientÞ
Lmodel_3/transformer_block_7/layer_normalization_14/moments/SquaredDifferenceSquaredDifference#model_3/transformer_block_7/add:z:0Pmodel_3/transformer_block_7/layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2N
Lmodel_3/transformer_block_7/layer_normalization_14/moments/SquaredDifferenceø
Umodel_3/transformer_block_7/layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_3/transformer_block_7/layer_normalization_14/moments/variance/reduction_indices
Cmodel_3/transformer_block_7/layer_normalization_14/moments/varianceMeanPmodel_3/transformer_block_7/layer_normalization_14/moments/SquaredDifference:z:0^model_3/transformer_block_7/layer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2E
Cmodel_3/transformer_block_7/layer_normalization_14/moments/varianceÍ
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752D
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add/yÞ
@model_3/transformer_block_7/layer_normalization_14/batchnorm/addAddV2Lmodel_3/transformer_block_7/layer_normalization_14/moments/variance:output:0Kmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2B
@model_3/transformer_block_7/layer_normalization_14/batchnorm/add
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/RsqrtRsqrtDmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2D
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/Rsqrt·
Omodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_3_transformer_block_7_layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02Q
Omodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpâ
@model_3/transformer_block_7/layer_normalization_14/batchnorm/mulMulFmodel_3/transformer_block_7/layer_normalization_14/batchnorm/Rsqrt:y:0Wmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2B
@model_3/transformer_block_7/layer_normalization_14/batchnorm/mul°
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul_1Mul#model_3/transformer_block_7/add:z:0Dmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2D
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul_1Õ
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul_2MulHmodel_3/transformer_block_7/layer_normalization_14/moments/mean:output:0Dmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2D
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul_2«
Kmodel_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOpTmodel_3_transformer_block_7_layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02M
Kmodel_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpÞ
@model_3/transformer_block_7/layer_normalization_14/batchnorm/subSubSmodel_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp:value:0Fmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2B
@model_3/transformer_block_7/layer_normalization_14/batchnorm/subÕ
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add_1AddV2Fmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul_1:z:0Dmodel_3/transformer_block_7/layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2D
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add_1¬
Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpReadVariableOpSmodel_3_transformer_block_7_sequential_7_dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02L
Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpÎ
@model_3/transformer_block_7/sequential_7/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_3/transformer_block_7/sequential_7/dense_23/Tensordot/axesÕ
@model_3/transformer_block_7/sequential_7/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_3/transformer_block_7/sequential_7/dense_23/Tensordot/freeü
Amodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ShapeShapeFmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2C
Amodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ShapeØ
Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axisË
Dmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2GatherV2Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Shape:output:0Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/free:output:0Rmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2Ü
Kmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axisÑ
Fmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1GatherV2Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Shape:output:0Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/axes:output:0Tmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1Ð
Amodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ConstÈ
@model_3/transformer_block_7/sequential_7/dense_23/Tensordot/ProdProdMmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2:output:0Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_3/transformer_block_7/sequential_7/dense_23/Tensordot/ProdÔ
Cmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const_1Ð
Bmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Prod_1ProdOmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1:output:0Lmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Prod_1Ô
Gmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat/axisª
Bmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concatConcatV2Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/free:output:0Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/axes:output:0Pmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concatÔ
Amodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/stackPackImodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Prod:output:0Kmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/stackæ
Emodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/transpose	TransposeFmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add_1:z:0Kmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2G
Emodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/transposeç
Cmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReshapeReshapeImodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/transpose:y:0Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2E
Cmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Reshapeæ
Bmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/MatMulMatMulLmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Reshape:output:0Rmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2D
Bmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/MatMulÔ
Cmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2E
Cmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const_2Ø
Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat_1/axis·
Dmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat_1ConcatV2Mmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2:output:0Lmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const_2:output:0Rmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat_1Ø
;model_3/transformer_block_7/sequential_7/dense_23/TensordotReshapeLmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/MatMul:product:0Mmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2=
;model_3/transformer_block_7/sequential_7/dense_23/Tensordot¢
Hmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOpQmodel_3_transformer_block_7_sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02J
Hmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpÏ
9model_3/transformer_block_7/sequential_7/dense_23/BiasAddBiasAddDmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot:output:0Pmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2;
9model_3/transformer_block_7/sequential_7/dense_23/BiasAddò
6model_3/transformer_block_7/sequential_7/dense_23/ReluReluBmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@28
6model_3/transformer_block_7/sequential_7/dense_23/Relu¬
Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpReadVariableOpSmodel_3_transformer_block_7_sequential_7_dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02L
Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpÎ
@model_3/transformer_block_7/sequential_7/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_3/transformer_block_7/sequential_7/dense_24/Tensordot/axesÕ
@model_3/transformer_block_7/sequential_7/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_3/transformer_block_7/sequential_7/dense_24/Tensordot/freeú
Amodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ShapeShapeDmodel_3/transformer_block_7/sequential_7/dense_23/Relu:activations:0*
T0*
_output_shapes
:2C
Amodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ShapeØ
Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axisË
Dmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2GatherV2Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Shape:output:0Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/free:output:0Rmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2Ü
Kmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axisÑ
Fmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1GatherV2Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Shape:output:0Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/axes:output:0Tmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1Ð
Amodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ConstÈ
@model_3/transformer_block_7/sequential_7/dense_24/Tensordot/ProdProdMmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2:output:0Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_3/transformer_block_7/sequential_7/dense_24/Tensordot/ProdÔ
Cmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const_1Ð
Bmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Prod_1ProdOmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1:output:0Lmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Prod_1Ô
Gmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat/axisª
Bmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concatConcatV2Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/free:output:0Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/axes:output:0Pmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concatÔ
Amodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/stackPackImodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Prod:output:0Kmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/stackä
Emodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/transpose	TransposeDmodel_3/transformer_block_7/sequential_7/dense_23/Relu:activations:0Kmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2G
Emodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/transposeç
Cmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReshapeReshapeImodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/transpose:y:0Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2E
Cmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Reshapeæ
Bmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/MatMulMatMulLmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Reshape:output:0Rmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2D
Bmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/MatMulÔ
Cmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const_2Ø
Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat_1/axis·
Dmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat_1ConcatV2Mmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2:output:0Lmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const_2:output:0Rmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat_1Ø
;model_3/transformer_block_7/sequential_7/dense_24/TensordotReshapeLmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/MatMul:product:0Mmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2=
;model_3/transformer_block_7/sequential_7/dense_24/Tensordot¢
Hmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOpQmodel_3_transformer_block_7_sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpÏ
9model_3/transformer_block_7/sequential_7/dense_24/BiasAddBiasAddDmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot:output:0Pmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2;
9model_3/transformer_block_7/sequential_7/dense_24/BiasAddè
/model_3/transformer_block_7/dropout_21/IdentityIdentityBmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 21
/model_3/transformer_block_7/dropout_21/Identity
!model_3/transformer_block_7/add_1AddV2Fmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add_1:z:08model_3/transformer_block_7/dropout_21/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2#
!model_3/transformer_block_7/add_1ð
Qmodel_3/transformer_block_7/layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_3/transformer_block_7/layer_normalization_15/moments/mean/reduction_indicesÔ
?model_3/transformer_block_7/layer_normalization_15/moments/meanMean%model_3/transformer_block_7/add_1:z:0Zmodel_3/transformer_block_7/layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2A
?model_3/transformer_block_7/layer_normalization_15/moments/mean¢
Gmodel_3/transformer_block_7/layer_normalization_15/moments/StopGradientStopGradientHmodel_3/transformer_block_7/layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2I
Gmodel_3/transformer_block_7/layer_normalization_15/moments/StopGradientà
Lmodel_3/transformer_block_7/layer_normalization_15/moments/SquaredDifferenceSquaredDifference%model_3/transformer_block_7/add_1:z:0Pmodel_3/transformer_block_7/layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2N
Lmodel_3/transformer_block_7/layer_normalization_15/moments/SquaredDifferenceø
Umodel_3/transformer_block_7/layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_3/transformer_block_7/layer_normalization_15/moments/variance/reduction_indices
Cmodel_3/transformer_block_7/layer_normalization_15/moments/varianceMeanPmodel_3/transformer_block_7/layer_normalization_15/moments/SquaredDifference:z:0^model_3/transformer_block_7/layer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2E
Cmodel_3/transformer_block_7/layer_normalization_15/moments/varianceÍ
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752D
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add/yÞ
@model_3/transformer_block_7/layer_normalization_15/batchnorm/addAddV2Lmodel_3/transformer_block_7/layer_normalization_15/moments/variance:output:0Kmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2B
@model_3/transformer_block_7/layer_normalization_15/batchnorm/add
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/RsqrtRsqrtDmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2D
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/Rsqrt·
Omodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_3_transformer_block_7_layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02Q
Omodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpâ
@model_3/transformer_block_7/layer_normalization_15/batchnorm/mulMulFmodel_3/transformer_block_7/layer_normalization_15/batchnorm/Rsqrt:y:0Wmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2B
@model_3/transformer_block_7/layer_normalization_15/batchnorm/mul²
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul_1Mul%model_3/transformer_block_7/add_1:z:0Dmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2D
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul_1Õ
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul_2MulHmodel_3/transformer_block_7/layer_normalization_15/moments/mean:output:0Dmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2D
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul_2«
Kmodel_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOpTmodel_3_transformer_block_7_layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02M
Kmodel_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpÞ
@model_3/transformer_block_7/layer_normalization_15/batchnorm/subSubSmodel_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp:value:0Fmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2B
@model_3/transformer_block_7/layer_normalization_15/batchnorm/subÕ
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add_1AddV2Fmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul_1:z:0Dmodel_3/transformer_block_7/layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2D
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add_1
model_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
model_3/flatten_3/ConstÞ
model_3/flatten_3/ReshapeReshapeFmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add_1:z:0 model_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
model_3/flatten_3/Reshape
!model_3/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_3/concatenate_3/concat/axisÝ
model_3/concatenate_3/concatConcatV2"model_3/flatten_3/Reshape:output:0input_8*model_3/concatenate_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
model_3/concatenate_3/concatÁ
&model_3/dense_25/MatMul/ReadVariableOpReadVariableOp/model_3_dense_25_matmul_readvariableop_resource*
_output_shapes
:	È@*
dtype02(
&model_3/dense_25/MatMul/ReadVariableOpÅ
model_3/dense_25/MatMulMatMul%model_3/concatenate_3/concat:output:0.model_3/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_3/dense_25/MatMul¿
'model_3/dense_25/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_3/dense_25/BiasAdd/ReadVariableOpÅ
model_3/dense_25/BiasAddBiasAdd!model_3/dense_25/MatMul:product:0/model_3/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_3/dense_25/BiasAdd
model_3/dense_25/ReluRelu!model_3/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_3/dense_25/Relu
model_3/dropout_22/IdentityIdentity#model_3/dense_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_3/dropout_22/IdentityÀ
&model_3/dense_26/MatMul/ReadVariableOpReadVariableOp/model_3_dense_26_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02(
&model_3/dense_26/MatMul/ReadVariableOpÄ
model_3/dense_26/MatMulMatMul$model_3/dropout_22/Identity:output:0.model_3/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_3/dense_26/MatMul¿
'model_3/dense_26/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_3/dense_26/BiasAdd/ReadVariableOpÅ
model_3/dense_26/BiasAddBiasAdd!model_3/dense_26/MatMul:product:0/model_3/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_3/dense_26/BiasAdd
model_3/dense_26/ReluRelu!model_3/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_3/dense_26/Relu
model_3/dropout_23/IdentityIdentity#model_3/dense_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_3/dropout_23/IdentityÀ
&model_3/dense_27/MatMul/ReadVariableOpReadVariableOp/model_3_dense_27_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&model_3/dense_27/MatMul/ReadVariableOpÄ
model_3/dense_27/MatMulMatMul$model_3/dropout_23/Identity:output:0.model_3/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/dense_27/MatMul¿
'model_3/dense_27/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_3/dense_27/BiasAdd/ReadVariableOpÅ
model_3/dense_27/BiasAddBiasAdd!model_3/dense_27/MatMul:product:0/model_3/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/dense_27/BiasAdd¬
IdentityIdentity!model_3/dense_27/BiasAdd:output:07^model_3/batch_normalization_6/batchnorm/ReadVariableOp9^model_3/batch_normalization_6/batchnorm/ReadVariableOp_19^model_3/batch_normalization_6/batchnorm/ReadVariableOp_2;^model_3/batch_normalization_6/batchnorm/mul/ReadVariableOp7^model_3/batch_normalization_7/batchnorm/ReadVariableOp9^model_3/batch_normalization_7/batchnorm/ReadVariableOp_19^model_3/batch_normalization_7/batchnorm/ReadVariableOp_2;^model_3/batch_normalization_7/batchnorm/mul/ReadVariableOp(^model_3/conv1d_6/BiasAdd/ReadVariableOp4^model_3/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp(^model_3/conv1d_7/BiasAdd/ReadVariableOp4^model_3/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp(^model_3/dense_25/BiasAdd/ReadVariableOp'^model_3/dense_25/MatMul/ReadVariableOp(^model_3/dense_26/BiasAdd/ReadVariableOp'^model_3/dense_26/MatMul/ReadVariableOp(^model_3/dense_27/BiasAdd/ReadVariableOp'^model_3/dense_27/MatMul/ReadVariableOpD^model_3/token_and_position_embedding_3/embedding_6/embedding_lookupD^model_3/token_and_position_embedding_3/embedding_7/embedding_lookupL^model_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpP^model_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpL^model_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpP^model_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpW^model_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpa^model_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpJ^model_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOpT^model_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpL^model_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOpV^model_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpL^model_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOpV^model_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpI^model_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpK^model_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpI^model_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpK^model_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Í
_input_shapes»
¸:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2p
6model_3/batch_normalization_6/batchnorm/ReadVariableOp6model_3/batch_normalization_6/batchnorm/ReadVariableOp2t
8model_3/batch_normalization_6/batchnorm/ReadVariableOp_18model_3/batch_normalization_6/batchnorm/ReadVariableOp_12t
8model_3/batch_normalization_6/batchnorm/ReadVariableOp_28model_3/batch_normalization_6/batchnorm/ReadVariableOp_22x
:model_3/batch_normalization_6/batchnorm/mul/ReadVariableOp:model_3/batch_normalization_6/batchnorm/mul/ReadVariableOp2p
6model_3/batch_normalization_7/batchnorm/ReadVariableOp6model_3/batch_normalization_7/batchnorm/ReadVariableOp2t
8model_3/batch_normalization_7/batchnorm/ReadVariableOp_18model_3/batch_normalization_7/batchnorm/ReadVariableOp_12t
8model_3/batch_normalization_7/batchnorm/ReadVariableOp_28model_3/batch_normalization_7/batchnorm/ReadVariableOp_22x
:model_3/batch_normalization_7/batchnorm/mul/ReadVariableOp:model_3/batch_normalization_7/batchnorm/mul/ReadVariableOp2R
'model_3/conv1d_6/BiasAdd/ReadVariableOp'model_3/conv1d_6/BiasAdd/ReadVariableOp2j
3model_3/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp3model_3/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp2R
'model_3/conv1d_7/BiasAdd/ReadVariableOp'model_3/conv1d_7/BiasAdd/ReadVariableOp2j
3model_3/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp3model_3/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp2R
'model_3/dense_25/BiasAdd/ReadVariableOp'model_3/dense_25/BiasAdd/ReadVariableOp2P
&model_3/dense_25/MatMul/ReadVariableOp&model_3/dense_25/MatMul/ReadVariableOp2R
'model_3/dense_26/BiasAdd/ReadVariableOp'model_3/dense_26/BiasAdd/ReadVariableOp2P
&model_3/dense_26/MatMul/ReadVariableOp&model_3/dense_26/MatMul/ReadVariableOp2R
'model_3/dense_27/BiasAdd/ReadVariableOp'model_3/dense_27/BiasAdd/ReadVariableOp2P
&model_3/dense_27/MatMul/ReadVariableOp&model_3/dense_27/MatMul/ReadVariableOp2
Cmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookupCmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup2
Cmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookupCmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup2
Kmodel_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpKmodel_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp2¢
Omodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpOmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp2
Kmodel_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpKmodel_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp2¢
Omodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpOmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp2°
Vmodel_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpVmodel_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp2Ä
`model_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp`model_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp2
Imodel_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOpImodel_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOp2ª
Smodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpSmodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp2
Kmodel_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOpKmodel_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOp2®
Umodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpUmodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp2
Kmodel_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOpKmodel_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOp2®
Umodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpUmodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp2
Hmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpHmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp2
Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpJmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp2
Hmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpHmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp2
Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpJmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp:R N
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!
_user_specified_name	input_7:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_8
ñ	
Ý
D__inference_dense_25_layer_call_and_return_conditional_losses_197729

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	È@*
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
:ÿÿÿÿÿÿÿÿÿÈ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ö[
è
C__inference_model_3_layer_call_and_return_conditional_losses_195686
input_7
input_8)
%token_and_position_embedding_3_194862)
%token_and_position_embedding_3_194864
conv1d_6_194894
conv1d_6_194896
conv1d_7_194927
conv1d_7_194929 
batch_normalization_6_195016 
batch_normalization_6_195018 
batch_normalization_6_195020 
batch_normalization_6_195022 
batch_normalization_7_195107 
batch_normalization_7_195109 
batch_normalization_7_195111 
batch_normalization_7_195113
transformer_block_7_195482
transformer_block_7_195484
transformer_block_7_195486
transformer_block_7_195488
transformer_block_7_195490
transformer_block_7_195492
transformer_block_7_195494
transformer_block_7_195496
transformer_block_7_195498
transformer_block_7_195500
transformer_block_7_195502
transformer_block_7_195504
transformer_block_7_195506
transformer_block_7_195508
transformer_block_7_195510
transformer_block_7_195512
dense_25_195567
dense_25_195569
dense_26_195624
dense_26_195626
dense_27_195680
dense_27_195682
identity¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢ conv1d_6/StatefulPartitionedCall¢ conv1d_7/StatefulPartitionedCall¢ dense_25/StatefulPartitionedCall¢ dense_26/StatefulPartitionedCall¢ dense_27/StatefulPartitionedCall¢"dropout_22/StatefulPartitionedCall¢"dropout_23/StatefulPartitionedCall¢6token_and_position_embedding_3/StatefulPartitionedCall¢+transformer_block_7/StatefulPartitionedCall
6token_and_position_embedding_3/StatefulPartitionedCallStatefulPartitionedCallinput_7%token_and_position_embedding_3_194862%token_and_position_embedding_3_194864*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_19485128
6token_and_position_embedding_3/StatefulPartitionedCallÖ
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0conv1d_6_194894conv1d_6_194896*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_1948832"
 conv1d_6/StatefulPartitionedCall 
#average_pooling1d_9/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_1943392%
#average_pooling1d_9/PartitionedCallÂ
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_9/PartitionedCall:output:0conv1d_7_194927conv1d_7_194929*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_1949162"
 conv1d_7/StatefulPartitionedCall¸
$average_pooling1d_11/PartitionedCallPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_1943692&
$average_pooling1d_11/PartitionedCall¢
$average_pooling1d_10/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_1943542&
$average_pooling1d_10/PartitionedCallÁ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_10/PartitionedCall:output:0batch_normalization_6_195016batch_normalization_6_195018batch_normalization_6_195020batch_normalization_6_195022*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1949692/
-batch_normalization_6/StatefulPartitionedCallÁ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_11/PartitionedCall:output:0batch_normalization_7_195107batch_normalization_7_195109batch_normalization_7_195111batch_normalization_7_195113*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1950602/
-batch_normalization_7/StatefulPartitionedCall»
add_3/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_1951222
add_3/PartitionedCall
+transformer_block_7/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0transformer_block_7_195482transformer_block_7_195484transformer_block_7_195486transformer_block_7_195488transformer_block_7_195490transformer_block_7_195492transformer_block_7_195494transformer_block_7_195496transformer_block_7_195498transformer_block_7_195500transformer_block_7_195502transformer_block_7_195504transformer_block_7_195506transformer_block_7_195508transformer_block_7_195510transformer_block_7_195512*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_1952792-
+transformer_block_7/StatefulPartitionedCall
flatten_3/PartitionedCallPartitionedCall4transformer_block_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1955212
flatten_3/PartitionedCall
concatenate_3/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0input_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_1955362
concatenate_3/PartitionedCall·
 dense_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_25_195567dense_25_195569*
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
D__inference_dense_25_layer_call_and_return_conditional_losses_1955562"
 dense_25/StatefulPartitionedCall
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
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
F__inference_dropout_22_layer_call_and_return_conditional_losses_1955842$
"dropout_22/StatefulPartitionedCall¼
 dense_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_26_195624dense_26_195626*
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
D__inference_dense_26_layer_call_and_return_conditional_losses_1956132"
 dense_26/StatefulPartitionedCall½
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
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
F__inference_dropout_23_layer_call_and_return_conditional_losses_1956412$
"dropout_23/StatefulPartitionedCall¼
 dense_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_27_195680dense_27_195682*
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
D__inference_dense_27_layer_call_and_return_conditional_losses_1956692"
 dense_27/StatefulPartitionedCall½
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall7^token_and_position_embedding_3/StatefulPartitionedCall,^transformer_block_7/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Í
_input_shapes»
¸:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall2p
6token_and_position_embedding_3/StatefulPartitionedCall6token_and_position_embedding_3/StatefulPartitionedCall2Z
+transformer_block_7/StatefulPartitionedCall+transformer_block_7/StatefulPartitionedCall:R N
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!
_user_specified_name	input_7:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_8
Ðÿ
â
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_195279

inputsF
Bmulti_head_attention_7_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_7_query_add_readvariableop_resourceD
@multi_head_attention_7_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_7_key_add_readvariableop_resourceF
Bmulti_head_attention_7_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_7_value_add_readvariableop_resourceQ
Mmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_7_attention_output_add_readvariableop_resource@
<layer_normalization_14_batchnorm_mul_readvariableop_resource<
8layer_normalization_14_batchnorm_readvariableop_resource;
7sequential_7_dense_23_tensordot_readvariableop_resource9
5sequential_7_dense_23_biasadd_readvariableop_resource;
7sequential_7_dense_24_tensordot_readvariableop_resource9
5sequential_7_dense_24_biasadd_readvariableop_resource@
<layer_normalization_15_batchnorm_mul_readvariableop_resource<
8layer_normalization_15_batchnorm_readvariableop_resource
identity¢/layer_normalization_14/batchnorm/ReadVariableOp¢3layer_normalization_14/batchnorm/mul/ReadVariableOp¢/layer_normalization_15/batchnorm/ReadVariableOp¢3layer_normalization_15/batchnorm/mul/ReadVariableOp¢:multi_head_attention_7/attention_output/add/ReadVariableOp¢Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_7/key/add/ReadVariableOp¢7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_7/query/add/ReadVariableOp¢9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_7/value/add/ReadVariableOp¢9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp¢,sequential_7/dense_23/BiasAdd/ReadVariableOp¢.sequential_7/dense_23/Tensordot/ReadVariableOp¢,sequential_7/dense_24/BiasAdd/ReadVariableOp¢.sequential_7/dense_24/Tensordot/ReadVariableOpý
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_7/query/einsum/EinsumEinsuminputsAmulti_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2,
*multi_head_attention_7/query/einsum/EinsumÛ
/multi_head_attention_7/query/add/ReadVariableOpReadVariableOp8multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/query/add/ReadVariableOpõ
 multi_head_attention_7/query/addAddV23multi_head_attention_7/query/einsum/Einsum:output:07multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2"
 multi_head_attention_7/query/add÷
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_7/key/einsum/EinsumEinsuminputs?multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2*
(multi_head_attention_7/key/einsum/EinsumÕ
-multi_head_attention_7/key/add/ReadVariableOpReadVariableOp6multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_7/key/add/ReadVariableOpí
multi_head_attention_7/key/addAddV21multi_head_attention_7/key/einsum/Einsum:output:05multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2 
multi_head_attention_7/key/addý
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_7/value/einsum/EinsumEinsuminputsAmulti_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2,
*multi_head_attention_7/value/einsum/EinsumÛ
/multi_head_attention_7/value/add/ReadVariableOpReadVariableOp8multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/value/add/ReadVariableOpõ
 multi_head_attention_7/value/addAddV23multi_head_attention_7/value/einsum/Einsum:output:07multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2"
 multi_head_attention_7/value/add
multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_7/Mul/yÆ
multi_head_attention_7/MulMul$multi_head_attention_7/query/add:z:0%multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
multi_head_attention_7/Mulü
$multi_head_attention_7/einsum/EinsumEinsum"multi_head_attention_7/key/add:z:0multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
equationaecd,abcd->acbe2&
$multi_head_attention_7/einsum/EinsumÄ
&multi_head_attention_7/softmax/SoftmaxSoftmax-multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2(
&multi_head_attention_7/softmax/Softmax¡
,multi_head_attention_7/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,multi_head_attention_7/dropout/dropout/Const
*multi_head_attention_7/dropout/dropout/MulMul0multi_head_attention_7/softmax/Softmax:softmax:05multi_head_attention_7/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2,
*multi_head_attention_7/dropout/dropout/Mul¼
,multi_head_attention_7/dropout/dropout/ShapeShape0multi_head_attention_7/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_7/dropout/dropout/Shape¥
Cmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_7/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
dtype0*

seed*2E
Cmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniform³
5multi_head_attention_7/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_7/dropout/dropout/GreaterEqual/yÂ
3multi_head_attention_7/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_7/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB25
3multi_head_attention_7/dropout/dropout/GreaterEqualä
+multi_head_attention_7/dropout/dropout/CastCast7multi_head_attention_7/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2-
+multi_head_attention_7/dropout/dropout/Castþ
,multi_head_attention_7/dropout/dropout/Mul_1Mul.multi_head_attention_7/dropout/dropout/Mul:z:0/multi_head_attention_7/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2.
,multi_head_attention_7/dropout/dropout/Mul_1
&multi_head_attention_7/einsum_1/EinsumEinsum0multi_head_attention_7/dropout/dropout/Mul_1:z:0$multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationacbe,aecd->abcd2(
&multi_head_attention_7/einsum_1/Einsum
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_7/attention_output/einsum/EinsumEinsum/multi_head_attention_7/einsum_1/Einsum:output:0Lmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabcd,cde->abe27
5multi_head_attention_7/attention_output/einsum/Einsumø
:multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_7/attention_output/add/ReadVariableOp
+multi_head_attention_7/attention_output/addAddV2>multi_head_attention_7/attention_output/einsum/Einsum:output:0Bmulti_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2-
+multi_head_attention_7/attention_output/addy
dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_20/dropout/ConstÁ
dropout_20/dropout/MulMul/multi_head_attention_7/attention_output/add:z:0!dropout_20/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dropout_20/dropout/Mul
dropout_20/dropout/ShapeShape/multi_head_attention_7/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_20/dropout/Shapeò
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
dtype0*

seed**
seed221
/dropout_20/dropout/random_uniform/RandomUniform
!dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_20/dropout/GreaterEqual/yî
dropout_20/dropout/GreaterEqualGreaterEqual8dropout_20/dropout/random_uniform/RandomUniform:output:0*dropout_20/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2!
dropout_20/dropout/GreaterEqual¤
dropout_20/dropout/CastCast#dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dropout_20/dropout/Castª
dropout_20/dropout/Mul_1Muldropout_20/dropout/Mul:z:0dropout_20/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dropout_20/dropout/Mul_1o
addAddV2inputsdropout_20/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
add¸
5layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_14/moments/mean/reduction_indicesâ
#layer_normalization_14/moments/meanMeanadd:z:0>layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2%
#layer_normalization_14/moments/meanÎ
+layer_normalization_14/moments/StopGradientStopGradient,layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2-
+layer_normalization_14/moments/StopGradientî
0layer_normalization_14/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 22
0layer_normalization_14/moments/SquaredDifferenceÀ
9layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_14/moments/variance/reduction_indices
'layer_normalization_14/moments/varianceMean4layer_normalization_14/moments/SquaredDifference:z:0Blayer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2)
'layer_normalization_14/moments/variance
&layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_14/batchnorm/add/yî
$layer_normalization_14/batchnorm/addAddV20layer_normalization_14/moments/variance:output:0/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2&
$layer_normalization_14/batchnorm/add¹
&layer_normalization_14/batchnorm/RsqrtRsqrt(layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2(
&layer_normalization_14/batchnorm/Rsqrtã
3layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_14/batchnorm/mul/ReadVariableOpò
$layer_normalization_14/batchnorm/mulMul*layer_normalization_14/batchnorm/Rsqrt:y:0;layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2&
$layer_normalization_14/batchnorm/mulÀ
&layer_normalization_14/batchnorm/mul_1Muladd:z:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_14/batchnorm/mul_1å
&layer_normalization_14/batchnorm/mul_2Mul,layer_normalization_14/moments/mean:output:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_14/batchnorm/mul_2×
/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_14/batchnorm/ReadVariableOpî
$layer_normalization_14/batchnorm/subSub7layer_normalization_14/batchnorm/ReadVariableOp:value:0*layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2&
$layer_normalization_14/batchnorm/subå
&layer_normalization_14/batchnorm/add_1AddV2*layer_normalization_14/batchnorm/mul_1:z:0(layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_14/batchnorm/add_1Ø
.sequential_7/dense_23/Tensordot/ReadVariableOpReadVariableOp7sequential_7_dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_7/dense_23/Tensordot/ReadVariableOp
$sequential_7/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_7/dense_23/Tensordot/axes
$sequential_7/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_7/dense_23/Tensordot/free¨
%sequential_7/dense_23/Tensordot/ShapeShape*layer_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/Shape 
-sequential_7/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_23/Tensordot/GatherV2/axis¿
(sequential_7/dense_23/Tensordot/GatherV2GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/free:output:06sequential_7/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/GatherV2¤
/sequential_7/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_23/Tensordot/GatherV2_1/axisÅ
*sequential_7/dense_23/Tensordot/GatherV2_1GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/axes:output:08sequential_7/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_7/dense_23/Tensordot/GatherV2_1
%sequential_7/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_7/dense_23/Tensordot/ConstØ
$sequential_7/dense_23/Tensordot/ProdProd1sequential_7/dense_23/Tensordot/GatherV2:output:0.sequential_7/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_7/dense_23/Tensordot/Prod
'sequential_7/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_23/Tensordot/Const_1à
&sequential_7/dense_23/Tensordot/Prod_1Prod3sequential_7/dense_23/Tensordot/GatherV2_1:output:00sequential_7/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_7/dense_23/Tensordot/Prod_1
+sequential_7/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_7/dense_23/Tensordot/concat/axis
&sequential_7/dense_23/Tensordot/concatConcatV2-sequential_7/dense_23/Tensordot/free:output:0-sequential_7/dense_23/Tensordot/axes:output:04sequential_7/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/dense_23/Tensordot/concatä
%sequential_7/dense_23/Tensordot/stackPack-sequential_7/dense_23/Tensordot/Prod:output:0/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/stackö
)sequential_7/dense_23/Tensordot/transpose	Transpose*layer_normalization_14/batchnorm/add_1:z:0/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2+
)sequential_7/dense_23/Tensordot/transpose÷
'sequential_7/dense_23/Tensordot/ReshapeReshape-sequential_7/dense_23/Tensordot/transpose:y:0.sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_7/dense_23/Tensordot/Reshapeö
&sequential_7/dense_23/Tensordot/MatMulMatMul0sequential_7/dense_23/Tensordot/Reshape:output:06sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential_7/dense_23/Tensordot/MatMul
'sequential_7/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_7/dense_23/Tensordot/Const_2 
-sequential_7/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_23/Tensordot/concat_1/axis«
(sequential_7/dense_23/Tensordot/concat_1ConcatV21sequential_7/dense_23/Tensordot/GatherV2:output:00sequential_7/dense_23/Tensordot/Const_2:output:06sequential_7/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/concat_1è
sequential_7/dense_23/TensordotReshape0sequential_7/dense_23/Tensordot/MatMul:product:01sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2!
sequential_7/dense_23/TensordotÎ
,sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_7/dense_23/BiasAdd/ReadVariableOpß
sequential_7/dense_23/BiasAddBiasAdd(sequential_7/dense_23/Tensordot:output:04sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2
sequential_7/dense_23/BiasAdd
sequential_7/dense_23/ReluRelu&sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2
sequential_7/dense_23/ReluØ
.sequential_7/dense_24/Tensordot/ReadVariableOpReadVariableOp7sequential_7_dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_7/dense_24/Tensordot/ReadVariableOp
$sequential_7/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_7/dense_24/Tensordot/axes
$sequential_7/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_7/dense_24/Tensordot/free¦
%sequential_7/dense_24/Tensordot/ShapeShape(sequential_7/dense_23/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/Shape 
-sequential_7/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_24/Tensordot/GatherV2/axis¿
(sequential_7/dense_24/Tensordot/GatherV2GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/free:output:06sequential_7/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/GatherV2¤
/sequential_7/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_24/Tensordot/GatherV2_1/axisÅ
*sequential_7/dense_24/Tensordot/GatherV2_1GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/axes:output:08sequential_7/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_7/dense_24/Tensordot/GatherV2_1
%sequential_7/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_7/dense_24/Tensordot/ConstØ
$sequential_7/dense_24/Tensordot/ProdProd1sequential_7/dense_24/Tensordot/GatherV2:output:0.sequential_7/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_7/dense_24/Tensordot/Prod
'sequential_7/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_24/Tensordot/Const_1à
&sequential_7/dense_24/Tensordot/Prod_1Prod3sequential_7/dense_24/Tensordot/GatherV2_1:output:00sequential_7/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_7/dense_24/Tensordot/Prod_1
+sequential_7/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_7/dense_24/Tensordot/concat/axis
&sequential_7/dense_24/Tensordot/concatConcatV2-sequential_7/dense_24/Tensordot/free:output:0-sequential_7/dense_24/Tensordot/axes:output:04sequential_7/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/dense_24/Tensordot/concatä
%sequential_7/dense_24/Tensordot/stackPack-sequential_7/dense_24/Tensordot/Prod:output:0/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/stackô
)sequential_7/dense_24/Tensordot/transpose	Transpose(sequential_7/dense_23/Relu:activations:0/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2+
)sequential_7/dense_24/Tensordot/transpose÷
'sequential_7/dense_24/Tensordot/ReshapeReshape-sequential_7/dense_24/Tensordot/transpose:y:0.sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_7/dense_24/Tensordot/Reshapeö
&sequential_7/dense_24/Tensordot/MatMulMatMul0sequential_7/dense_24/Tensordot/Reshape:output:06sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_7/dense_24/Tensordot/MatMul
'sequential_7/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_24/Tensordot/Const_2 
-sequential_7/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_24/Tensordot/concat_1/axis«
(sequential_7/dense_24/Tensordot/concat_1ConcatV21sequential_7/dense_24/Tensordot/GatherV2:output:00sequential_7/dense_24/Tensordot/Const_2:output:06sequential_7/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/concat_1è
sequential_7/dense_24/TensordotReshape0sequential_7/dense_24/Tensordot/MatMul:product:01sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2!
sequential_7/dense_24/TensordotÎ
,sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_7/dense_24/BiasAdd/ReadVariableOpß
sequential_7/dense_24/BiasAddBiasAdd(sequential_7/dense_24/Tensordot:output:04sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
sequential_7/dense_24/BiasAddy
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_21/dropout/Const¸
dropout_21/dropout/MulMul&sequential_7/dense_24/BiasAdd:output:0!dropout_21/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dropout_21/dropout/Mul
dropout_21/dropout/ShapeShape&sequential_7/dense_24/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_21/dropout/Shapeò
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
dtype0*

seed**
seed221
/dropout_21/dropout/random_uniform/RandomUniform
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_21/dropout/GreaterEqual/yî
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2!
dropout_21/dropout/GreaterEqual¤
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dropout_21/dropout/Castª
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dropout_21/dropout/Mul_1
add_1AddV2*layer_normalization_14/batchnorm/add_1:z:0dropout_21/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
add_1¸
5layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_15/moments/mean/reduction_indicesä
#layer_normalization_15/moments/meanMean	add_1:z:0>layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2%
#layer_normalization_15/moments/meanÎ
+layer_normalization_15/moments/StopGradientStopGradient,layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2-
+layer_normalization_15/moments/StopGradientð
0layer_normalization_15/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 22
0layer_normalization_15/moments/SquaredDifferenceÀ
9layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_15/moments/variance/reduction_indices
'layer_normalization_15/moments/varianceMean4layer_normalization_15/moments/SquaredDifference:z:0Blayer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2)
'layer_normalization_15/moments/variance
&layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_15/batchnorm/add/yî
$layer_normalization_15/batchnorm/addAddV20layer_normalization_15/moments/variance:output:0/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2&
$layer_normalization_15/batchnorm/add¹
&layer_normalization_15/batchnorm/RsqrtRsqrt(layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2(
&layer_normalization_15/batchnorm/Rsqrtã
3layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_15/batchnorm/mul/ReadVariableOpò
$layer_normalization_15/batchnorm/mulMul*layer_normalization_15/batchnorm/Rsqrt:y:0;layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2&
$layer_normalization_15/batchnorm/mulÂ
&layer_normalization_15/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_15/batchnorm/mul_1å
&layer_normalization_15/batchnorm/mul_2Mul,layer_normalization_15/moments/mean:output:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_15/batchnorm/mul_2×
/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_15/batchnorm/ReadVariableOpî
$layer_normalization_15/batchnorm/subSub7layer_normalization_15/batchnorm/ReadVariableOp:value:0*layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2&
$layer_normalization_15/batchnorm/subå
&layer_normalization_15/batchnorm/add_1AddV2*layer_normalization_15/batchnorm/mul_1:z:0(layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_15/batchnorm/add_1Ü
IdentityIdentity*layer_normalization_15/batchnorm/add_1:z:00^layer_normalization_14/batchnorm/ReadVariableOp4^layer_normalization_14/batchnorm/mul/ReadVariableOp0^layer_normalization_15/batchnorm/ReadVariableOp4^layer_normalization_15/batchnorm/mul/ReadVariableOp;^multi_head_attention_7/attention_output/add/ReadVariableOpE^multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_7/key/add/ReadVariableOp8^multi_head_attention_7/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/query/add/ReadVariableOp:^multi_head_attention_7/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/value/add/ReadVariableOp:^multi_head_attention_7/value/einsum/Einsum/ReadVariableOp-^sequential_7/dense_23/BiasAdd/ReadVariableOp/^sequential_7/dense_23/Tensordot/ReadVariableOp-^sequential_7/dense_24/BiasAdd/ReadVariableOp/^sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿB ::::::::::::::::2b
/layer_normalization_14/batchnorm/ReadVariableOp/layer_normalization_14/batchnorm/ReadVariableOp2j
3layer_normalization_14/batchnorm/mul/ReadVariableOp3layer_normalization_14/batchnorm/mul/ReadVariableOp2b
/layer_normalization_15/batchnorm/ReadVariableOp/layer_normalization_15/batchnorm/ReadVariableOp2j
3layer_normalization_15/batchnorm/mul/ReadVariableOp3layer_normalization_15/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_7/attention_output/add/ReadVariableOp:multi_head_attention_7/attention_output/add/ReadVariableOp2
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_7/key/add/ReadVariableOp-multi_head_attention_7/key/add/ReadVariableOp2r
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/query/add/ReadVariableOp/multi_head_attention_7/query/add/ReadVariableOp2v
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/value/add/ReadVariableOp/multi_head_attention_7/value/add/ReadVariableOp2v
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp2\
,sequential_7/dense_23/BiasAdd/ReadVariableOp,sequential_7/dense_23/BiasAdd/ReadVariableOp2`
.sequential_7/dense_23/Tensordot/ReadVariableOp.sequential_7/dense_23/Tensordot/ReadVariableOp2\
,sequential_7/dense_24/BiasAdd/ReadVariableOp,sequential_7/dense_24/BiasAdd/ReadVariableOp2`
.sequential_7/dense_24/Tensordot/ReadVariableOp.sequential_7/dense_24/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
°

$__inference_signature_wrapper_196213
input_7
input_8
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
identity¢StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallinput_7input_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
!__inference__wrapped_model_1943302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Í
_input_shapes»
¸:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!
_user_specified_name	input_7:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_8

e
F__inference_dropout_22_layer_call_and_return_conditional_losses_195584

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

seed*2&
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
¹Þ
â
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_197620

inputsF
Bmulti_head_attention_7_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_7_query_add_readvariableop_resourceD
@multi_head_attention_7_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_7_key_add_readvariableop_resourceF
Bmulti_head_attention_7_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_7_value_add_readvariableop_resourceQ
Mmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_7_attention_output_add_readvariableop_resource@
<layer_normalization_14_batchnorm_mul_readvariableop_resource<
8layer_normalization_14_batchnorm_readvariableop_resource;
7sequential_7_dense_23_tensordot_readvariableop_resource9
5sequential_7_dense_23_biasadd_readvariableop_resource;
7sequential_7_dense_24_tensordot_readvariableop_resource9
5sequential_7_dense_24_biasadd_readvariableop_resource@
<layer_normalization_15_batchnorm_mul_readvariableop_resource<
8layer_normalization_15_batchnorm_readvariableop_resource
identity¢/layer_normalization_14/batchnorm/ReadVariableOp¢3layer_normalization_14/batchnorm/mul/ReadVariableOp¢/layer_normalization_15/batchnorm/ReadVariableOp¢3layer_normalization_15/batchnorm/mul/ReadVariableOp¢:multi_head_attention_7/attention_output/add/ReadVariableOp¢Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_7/key/add/ReadVariableOp¢7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_7/query/add/ReadVariableOp¢9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_7/value/add/ReadVariableOp¢9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp¢,sequential_7/dense_23/BiasAdd/ReadVariableOp¢.sequential_7/dense_23/Tensordot/ReadVariableOp¢,sequential_7/dense_24/BiasAdd/ReadVariableOp¢.sequential_7/dense_24/Tensordot/ReadVariableOpý
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_7/query/einsum/EinsumEinsuminputsAmulti_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2,
*multi_head_attention_7/query/einsum/EinsumÛ
/multi_head_attention_7/query/add/ReadVariableOpReadVariableOp8multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/query/add/ReadVariableOpõ
 multi_head_attention_7/query/addAddV23multi_head_attention_7/query/einsum/Einsum:output:07multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2"
 multi_head_attention_7/query/add÷
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_7/key/einsum/EinsumEinsuminputs?multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2*
(multi_head_attention_7/key/einsum/EinsumÕ
-multi_head_attention_7/key/add/ReadVariableOpReadVariableOp6multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_7/key/add/ReadVariableOpí
multi_head_attention_7/key/addAddV21multi_head_attention_7/key/einsum/Einsum:output:05multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2 
multi_head_attention_7/key/addý
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_7/value/einsum/EinsumEinsuminputsAmulti_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2,
*multi_head_attention_7/value/einsum/EinsumÛ
/multi_head_attention_7/value/add/ReadVariableOpReadVariableOp8multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/value/add/ReadVariableOpõ
 multi_head_attention_7/value/addAddV23multi_head_attention_7/value/einsum/Einsum:output:07multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2"
 multi_head_attention_7/value/add
multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_7/Mul/yÆ
multi_head_attention_7/MulMul$multi_head_attention_7/query/add:z:0%multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
multi_head_attention_7/Mulü
$multi_head_attention_7/einsum/EinsumEinsum"multi_head_attention_7/key/add:z:0multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
equationaecd,abcd->acbe2&
$multi_head_attention_7/einsum/EinsumÄ
&multi_head_attention_7/softmax/SoftmaxSoftmax-multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2(
&multi_head_attention_7/softmax/SoftmaxÊ
'multi_head_attention_7/dropout/IdentityIdentity0multi_head_attention_7/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2)
'multi_head_attention_7/dropout/Identity
&multi_head_attention_7/einsum_1/EinsumEinsum0multi_head_attention_7/dropout/Identity:output:0$multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationacbe,aecd->abcd2(
&multi_head_attention_7/einsum_1/Einsum
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_7/attention_output/einsum/EinsumEinsum/multi_head_attention_7/einsum_1/Einsum:output:0Lmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabcd,cde->abe27
5multi_head_attention_7/attention_output/einsum/Einsumø
:multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_7/attention_output/add/ReadVariableOp
+multi_head_attention_7/attention_output/addAddV2>multi_head_attention_7/attention_output/einsum/Einsum:output:0Bmulti_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2-
+multi_head_attention_7/attention_output/add
dropout_20/IdentityIdentity/multi_head_attention_7/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dropout_20/Identityo
addAddV2inputsdropout_20/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
add¸
5layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_14/moments/mean/reduction_indicesâ
#layer_normalization_14/moments/meanMeanadd:z:0>layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2%
#layer_normalization_14/moments/meanÎ
+layer_normalization_14/moments/StopGradientStopGradient,layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2-
+layer_normalization_14/moments/StopGradientî
0layer_normalization_14/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 22
0layer_normalization_14/moments/SquaredDifferenceÀ
9layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_14/moments/variance/reduction_indices
'layer_normalization_14/moments/varianceMean4layer_normalization_14/moments/SquaredDifference:z:0Blayer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2)
'layer_normalization_14/moments/variance
&layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_14/batchnorm/add/yî
$layer_normalization_14/batchnorm/addAddV20layer_normalization_14/moments/variance:output:0/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2&
$layer_normalization_14/batchnorm/add¹
&layer_normalization_14/batchnorm/RsqrtRsqrt(layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2(
&layer_normalization_14/batchnorm/Rsqrtã
3layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_14/batchnorm/mul/ReadVariableOpò
$layer_normalization_14/batchnorm/mulMul*layer_normalization_14/batchnorm/Rsqrt:y:0;layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2&
$layer_normalization_14/batchnorm/mulÀ
&layer_normalization_14/batchnorm/mul_1Muladd:z:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_14/batchnorm/mul_1å
&layer_normalization_14/batchnorm/mul_2Mul,layer_normalization_14/moments/mean:output:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_14/batchnorm/mul_2×
/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_14/batchnorm/ReadVariableOpî
$layer_normalization_14/batchnorm/subSub7layer_normalization_14/batchnorm/ReadVariableOp:value:0*layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2&
$layer_normalization_14/batchnorm/subå
&layer_normalization_14/batchnorm/add_1AddV2*layer_normalization_14/batchnorm/mul_1:z:0(layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_14/batchnorm/add_1Ø
.sequential_7/dense_23/Tensordot/ReadVariableOpReadVariableOp7sequential_7_dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_7/dense_23/Tensordot/ReadVariableOp
$sequential_7/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_7/dense_23/Tensordot/axes
$sequential_7/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_7/dense_23/Tensordot/free¨
%sequential_7/dense_23/Tensordot/ShapeShape*layer_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/Shape 
-sequential_7/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_23/Tensordot/GatherV2/axis¿
(sequential_7/dense_23/Tensordot/GatherV2GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/free:output:06sequential_7/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/GatherV2¤
/sequential_7/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_23/Tensordot/GatherV2_1/axisÅ
*sequential_7/dense_23/Tensordot/GatherV2_1GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/axes:output:08sequential_7/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_7/dense_23/Tensordot/GatherV2_1
%sequential_7/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_7/dense_23/Tensordot/ConstØ
$sequential_7/dense_23/Tensordot/ProdProd1sequential_7/dense_23/Tensordot/GatherV2:output:0.sequential_7/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_7/dense_23/Tensordot/Prod
'sequential_7/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_23/Tensordot/Const_1à
&sequential_7/dense_23/Tensordot/Prod_1Prod3sequential_7/dense_23/Tensordot/GatherV2_1:output:00sequential_7/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_7/dense_23/Tensordot/Prod_1
+sequential_7/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_7/dense_23/Tensordot/concat/axis
&sequential_7/dense_23/Tensordot/concatConcatV2-sequential_7/dense_23/Tensordot/free:output:0-sequential_7/dense_23/Tensordot/axes:output:04sequential_7/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/dense_23/Tensordot/concatä
%sequential_7/dense_23/Tensordot/stackPack-sequential_7/dense_23/Tensordot/Prod:output:0/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/stackö
)sequential_7/dense_23/Tensordot/transpose	Transpose*layer_normalization_14/batchnorm/add_1:z:0/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2+
)sequential_7/dense_23/Tensordot/transpose÷
'sequential_7/dense_23/Tensordot/ReshapeReshape-sequential_7/dense_23/Tensordot/transpose:y:0.sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_7/dense_23/Tensordot/Reshapeö
&sequential_7/dense_23/Tensordot/MatMulMatMul0sequential_7/dense_23/Tensordot/Reshape:output:06sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential_7/dense_23/Tensordot/MatMul
'sequential_7/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_7/dense_23/Tensordot/Const_2 
-sequential_7/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_23/Tensordot/concat_1/axis«
(sequential_7/dense_23/Tensordot/concat_1ConcatV21sequential_7/dense_23/Tensordot/GatherV2:output:00sequential_7/dense_23/Tensordot/Const_2:output:06sequential_7/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/concat_1è
sequential_7/dense_23/TensordotReshape0sequential_7/dense_23/Tensordot/MatMul:product:01sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2!
sequential_7/dense_23/TensordotÎ
,sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_7/dense_23/BiasAdd/ReadVariableOpß
sequential_7/dense_23/BiasAddBiasAdd(sequential_7/dense_23/Tensordot:output:04sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2
sequential_7/dense_23/BiasAdd
sequential_7/dense_23/ReluRelu&sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2
sequential_7/dense_23/ReluØ
.sequential_7/dense_24/Tensordot/ReadVariableOpReadVariableOp7sequential_7_dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_7/dense_24/Tensordot/ReadVariableOp
$sequential_7/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_7/dense_24/Tensordot/axes
$sequential_7/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_7/dense_24/Tensordot/free¦
%sequential_7/dense_24/Tensordot/ShapeShape(sequential_7/dense_23/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/Shape 
-sequential_7/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_24/Tensordot/GatherV2/axis¿
(sequential_7/dense_24/Tensordot/GatherV2GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/free:output:06sequential_7/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/GatherV2¤
/sequential_7/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_24/Tensordot/GatherV2_1/axisÅ
*sequential_7/dense_24/Tensordot/GatherV2_1GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/axes:output:08sequential_7/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_7/dense_24/Tensordot/GatherV2_1
%sequential_7/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_7/dense_24/Tensordot/ConstØ
$sequential_7/dense_24/Tensordot/ProdProd1sequential_7/dense_24/Tensordot/GatherV2:output:0.sequential_7/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_7/dense_24/Tensordot/Prod
'sequential_7/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_24/Tensordot/Const_1à
&sequential_7/dense_24/Tensordot/Prod_1Prod3sequential_7/dense_24/Tensordot/GatherV2_1:output:00sequential_7/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_7/dense_24/Tensordot/Prod_1
+sequential_7/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_7/dense_24/Tensordot/concat/axis
&sequential_7/dense_24/Tensordot/concatConcatV2-sequential_7/dense_24/Tensordot/free:output:0-sequential_7/dense_24/Tensordot/axes:output:04sequential_7/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/dense_24/Tensordot/concatä
%sequential_7/dense_24/Tensordot/stackPack-sequential_7/dense_24/Tensordot/Prod:output:0/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/stackô
)sequential_7/dense_24/Tensordot/transpose	Transpose(sequential_7/dense_23/Relu:activations:0/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2+
)sequential_7/dense_24/Tensordot/transpose÷
'sequential_7/dense_24/Tensordot/ReshapeReshape-sequential_7/dense_24/Tensordot/transpose:y:0.sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_7/dense_24/Tensordot/Reshapeö
&sequential_7/dense_24/Tensordot/MatMulMatMul0sequential_7/dense_24/Tensordot/Reshape:output:06sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_7/dense_24/Tensordot/MatMul
'sequential_7/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_24/Tensordot/Const_2 
-sequential_7/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_24/Tensordot/concat_1/axis«
(sequential_7/dense_24/Tensordot/concat_1ConcatV21sequential_7/dense_24/Tensordot/GatherV2:output:00sequential_7/dense_24/Tensordot/Const_2:output:06sequential_7/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/concat_1è
sequential_7/dense_24/TensordotReshape0sequential_7/dense_24/Tensordot/MatMul:product:01sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2!
sequential_7/dense_24/TensordotÎ
,sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_7/dense_24/BiasAdd/ReadVariableOpß
sequential_7/dense_24/BiasAddBiasAdd(sequential_7/dense_24/Tensordot:output:04sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
sequential_7/dense_24/BiasAdd
dropout_21/IdentityIdentity&sequential_7/dense_24/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dropout_21/Identity
add_1AddV2*layer_normalization_14/batchnorm/add_1:z:0dropout_21/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
add_1¸
5layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_15/moments/mean/reduction_indicesä
#layer_normalization_15/moments/meanMean	add_1:z:0>layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2%
#layer_normalization_15/moments/meanÎ
+layer_normalization_15/moments/StopGradientStopGradient,layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2-
+layer_normalization_15/moments/StopGradientð
0layer_normalization_15/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 22
0layer_normalization_15/moments/SquaredDifferenceÀ
9layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_15/moments/variance/reduction_indices
'layer_normalization_15/moments/varianceMean4layer_normalization_15/moments/SquaredDifference:z:0Blayer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2)
'layer_normalization_15/moments/variance
&layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_15/batchnorm/add/yî
$layer_normalization_15/batchnorm/addAddV20layer_normalization_15/moments/variance:output:0/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2&
$layer_normalization_15/batchnorm/add¹
&layer_normalization_15/batchnorm/RsqrtRsqrt(layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2(
&layer_normalization_15/batchnorm/Rsqrtã
3layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_15/batchnorm/mul/ReadVariableOpò
$layer_normalization_15/batchnorm/mulMul*layer_normalization_15/batchnorm/Rsqrt:y:0;layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2&
$layer_normalization_15/batchnorm/mulÂ
&layer_normalization_15/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_15/batchnorm/mul_1å
&layer_normalization_15/batchnorm/mul_2Mul,layer_normalization_15/moments/mean:output:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_15/batchnorm/mul_2×
/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_15/batchnorm/ReadVariableOpî
$layer_normalization_15/batchnorm/subSub7layer_normalization_15/batchnorm/ReadVariableOp:value:0*layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2&
$layer_normalization_15/batchnorm/subå
&layer_normalization_15/batchnorm/add_1AddV2*layer_normalization_15/batchnorm/mul_1:z:0(layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_15/batchnorm/add_1Ü
IdentityIdentity*layer_normalization_15/batchnorm/add_1:z:00^layer_normalization_14/batchnorm/ReadVariableOp4^layer_normalization_14/batchnorm/mul/ReadVariableOp0^layer_normalization_15/batchnorm/ReadVariableOp4^layer_normalization_15/batchnorm/mul/ReadVariableOp;^multi_head_attention_7/attention_output/add/ReadVariableOpE^multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_7/key/add/ReadVariableOp8^multi_head_attention_7/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/query/add/ReadVariableOp:^multi_head_attention_7/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/value/add/ReadVariableOp:^multi_head_attention_7/value/einsum/Einsum/ReadVariableOp-^sequential_7/dense_23/BiasAdd/ReadVariableOp/^sequential_7/dense_23/Tensordot/ReadVariableOp-^sequential_7/dense_24/BiasAdd/ReadVariableOp/^sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿB ::::::::::::::::2b
/layer_normalization_14/batchnorm/ReadVariableOp/layer_normalization_14/batchnorm/ReadVariableOp2j
3layer_normalization_14/batchnorm/mul/ReadVariableOp3layer_normalization_14/batchnorm/mul/ReadVariableOp2b
/layer_normalization_15/batchnorm/ReadVariableOp/layer_normalization_15/batchnorm/ReadVariableOp2j
3layer_normalization_15/batchnorm/mul/ReadVariableOp3layer_normalization_15/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_7/attention_output/add/ReadVariableOp:multi_head_attention_7/attention_output/add/ReadVariableOp2
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_7/key/add/ReadVariableOp-multi_head_attention_7/key/add/ReadVariableOp2r
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/query/add/ReadVariableOp/multi_head_attention_7/query/add/ReadVariableOp2v
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/value/add/ReadVariableOp/multi_head_attention_7/value/add/ReadVariableOp2v
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp2\
,sequential_7/dense_23/BiasAdd/ReadVariableOp,sequential_7/dense_23/BiasAdd/ReadVariableOp2`
.sequential_7/dense_23/Tensordot/ReadVariableOp.sequential_7/dense_23/Tensordot/ReadVariableOp2\
,sequential_7/dense_24/BiasAdd/ReadVariableOp,sequential_7/dense_24/BiasAdd/ReadVariableOp2`
.sequential_7/dense_24/Tensordot/ReadVariableOp.sequential_7/dense_24/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
¹Þ
â
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_195406

inputsF
Bmulti_head_attention_7_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_7_query_add_readvariableop_resourceD
@multi_head_attention_7_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_7_key_add_readvariableop_resourceF
Bmulti_head_attention_7_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_7_value_add_readvariableop_resourceQ
Mmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_7_attention_output_add_readvariableop_resource@
<layer_normalization_14_batchnorm_mul_readvariableop_resource<
8layer_normalization_14_batchnorm_readvariableop_resource;
7sequential_7_dense_23_tensordot_readvariableop_resource9
5sequential_7_dense_23_biasadd_readvariableop_resource;
7sequential_7_dense_24_tensordot_readvariableop_resource9
5sequential_7_dense_24_biasadd_readvariableop_resource@
<layer_normalization_15_batchnorm_mul_readvariableop_resource<
8layer_normalization_15_batchnorm_readvariableop_resource
identity¢/layer_normalization_14/batchnorm/ReadVariableOp¢3layer_normalization_14/batchnorm/mul/ReadVariableOp¢/layer_normalization_15/batchnorm/ReadVariableOp¢3layer_normalization_15/batchnorm/mul/ReadVariableOp¢:multi_head_attention_7/attention_output/add/ReadVariableOp¢Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_7/key/add/ReadVariableOp¢7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_7/query/add/ReadVariableOp¢9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_7/value/add/ReadVariableOp¢9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp¢,sequential_7/dense_23/BiasAdd/ReadVariableOp¢.sequential_7/dense_23/Tensordot/ReadVariableOp¢,sequential_7/dense_24/BiasAdd/ReadVariableOp¢.sequential_7/dense_24/Tensordot/ReadVariableOpý
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_7/query/einsum/EinsumEinsuminputsAmulti_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2,
*multi_head_attention_7/query/einsum/EinsumÛ
/multi_head_attention_7/query/add/ReadVariableOpReadVariableOp8multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/query/add/ReadVariableOpõ
 multi_head_attention_7/query/addAddV23multi_head_attention_7/query/einsum/Einsum:output:07multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2"
 multi_head_attention_7/query/add÷
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_7/key/einsum/EinsumEinsuminputs?multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2*
(multi_head_attention_7/key/einsum/EinsumÕ
-multi_head_attention_7/key/add/ReadVariableOpReadVariableOp6multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_7/key/add/ReadVariableOpí
multi_head_attention_7/key/addAddV21multi_head_attention_7/key/einsum/Einsum:output:05multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2 
multi_head_attention_7/key/addý
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_7/value/einsum/EinsumEinsuminputsAmulti_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabc,cde->abde2,
*multi_head_attention_7/value/einsum/EinsumÛ
/multi_head_attention_7/value/add/ReadVariableOpReadVariableOp8multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/value/add/ReadVariableOpõ
 multi_head_attention_7/value/addAddV23multi_head_attention_7/value/einsum/Einsum:output:07multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2"
 multi_head_attention_7/value/add
multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_7/Mul/yÆ
multi_head_attention_7/MulMul$multi_head_attention_7/query/add:z:0%multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
multi_head_attention_7/Mulü
$multi_head_attention_7/einsum/EinsumEinsum"multi_head_attention_7/key/add:z:0multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
equationaecd,abcd->acbe2&
$multi_head_attention_7/einsum/EinsumÄ
&multi_head_attention_7/softmax/SoftmaxSoftmax-multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2(
&multi_head_attention_7/softmax/SoftmaxÊ
'multi_head_attention_7/dropout/IdentityIdentity0multi_head_attention_7/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB2)
'multi_head_attention_7/dropout/Identity
&multi_head_attention_7/einsum_1/EinsumEinsum0multi_head_attention_7/dropout/Identity:output:0$multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationacbe,aecd->abcd2(
&multi_head_attention_7/einsum_1/Einsum
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_7/attention_output/einsum/EinsumEinsum/multi_head_attention_7/einsum_1/Einsum:output:0Lmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
equationabcd,cde->abe27
5multi_head_attention_7/attention_output/einsum/Einsumø
:multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_7/attention_output/add/ReadVariableOp
+multi_head_attention_7/attention_output/addAddV2>multi_head_attention_7/attention_output/einsum/Einsum:output:0Bmulti_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2-
+multi_head_attention_7/attention_output/add
dropout_20/IdentityIdentity/multi_head_attention_7/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dropout_20/Identityo
addAddV2inputsdropout_20/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
add¸
5layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_14/moments/mean/reduction_indicesâ
#layer_normalization_14/moments/meanMeanadd:z:0>layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2%
#layer_normalization_14/moments/meanÎ
+layer_normalization_14/moments/StopGradientStopGradient,layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2-
+layer_normalization_14/moments/StopGradientî
0layer_normalization_14/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 22
0layer_normalization_14/moments/SquaredDifferenceÀ
9layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_14/moments/variance/reduction_indices
'layer_normalization_14/moments/varianceMean4layer_normalization_14/moments/SquaredDifference:z:0Blayer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2)
'layer_normalization_14/moments/variance
&layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_14/batchnorm/add/yî
$layer_normalization_14/batchnorm/addAddV20layer_normalization_14/moments/variance:output:0/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2&
$layer_normalization_14/batchnorm/add¹
&layer_normalization_14/batchnorm/RsqrtRsqrt(layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2(
&layer_normalization_14/batchnorm/Rsqrtã
3layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_14/batchnorm/mul/ReadVariableOpò
$layer_normalization_14/batchnorm/mulMul*layer_normalization_14/batchnorm/Rsqrt:y:0;layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2&
$layer_normalization_14/batchnorm/mulÀ
&layer_normalization_14/batchnorm/mul_1Muladd:z:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_14/batchnorm/mul_1å
&layer_normalization_14/batchnorm/mul_2Mul,layer_normalization_14/moments/mean:output:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_14/batchnorm/mul_2×
/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_14/batchnorm/ReadVariableOpî
$layer_normalization_14/batchnorm/subSub7layer_normalization_14/batchnorm/ReadVariableOp:value:0*layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2&
$layer_normalization_14/batchnorm/subå
&layer_normalization_14/batchnorm/add_1AddV2*layer_normalization_14/batchnorm/mul_1:z:0(layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_14/batchnorm/add_1Ø
.sequential_7/dense_23/Tensordot/ReadVariableOpReadVariableOp7sequential_7_dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_7/dense_23/Tensordot/ReadVariableOp
$sequential_7/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_7/dense_23/Tensordot/axes
$sequential_7/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_7/dense_23/Tensordot/free¨
%sequential_7/dense_23/Tensordot/ShapeShape*layer_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/Shape 
-sequential_7/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_23/Tensordot/GatherV2/axis¿
(sequential_7/dense_23/Tensordot/GatherV2GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/free:output:06sequential_7/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/GatherV2¤
/sequential_7/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_23/Tensordot/GatherV2_1/axisÅ
*sequential_7/dense_23/Tensordot/GatherV2_1GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/axes:output:08sequential_7/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_7/dense_23/Tensordot/GatherV2_1
%sequential_7/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_7/dense_23/Tensordot/ConstØ
$sequential_7/dense_23/Tensordot/ProdProd1sequential_7/dense_23/Tensordot/GatherV2:output:0.sequential_7/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_7/dense_23/Tensordot/Prod
'sequential_7/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_23/Tensordot/Const_1à
&sequential_7/dense_23/Tensordot/Prod_1Prod3sequential_7/dense_23/Tensordot/GatherV2_1:output:00sequential_7/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_7/dense_23/Tensordot/Prod_1
+sequential_7/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_7/dense_23/Tensordot/concat/axis
&sequential_7/dense_23/Tensordot/concatConcatV2-sequential_7/dense_23/Tensordot/free:output:0-sequential_7/dense_23/Tensordot/axes:output:04sequential_7/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/dense_23/Tensordot/concatä
%sequential_7/dense_23/Tensordot/stackPack-sequential_7/dense_23/Tensordot/Prod:output:0/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/stackö
)sequential_7/dense_23/Tensordot/transpose	Transpose*layer_normalization_14/batchnorm/add_1:z:0/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2+
)sequential_7/dense_23/Tensordot/transpose÷
'sequential_7/dense_23/Tensordot/ReshapeReshape-sequential_7/dense_23/Tensordot/transpose:y:0.sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_7/dense_23/Tensordot/Reshapeö
&sequential_7/dense_23/Tensordot/MatMulMatMul0sequential_7/dense_23/Tensordot/Reshape:output:06sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential_7/dense_23/Tensordot/MatMul
'sequential_7/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_7/dense_23/Tensordot/Const_2 
-sequential_7/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_23/Tensordot/concat_1/axis«
(sequential_7/dense_23/Tensordot/concat_1ConcatV21sequential_7/dense_23/Tensordot/GatherV2:output:00sequential_7/dense_23/Tensordot/Const_2:output:06sequential_7/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/concat_1è
sequential_7/dense_23/TensordotReshape0sequential_7/dense_23/Tensordot/MatMul:product:01sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2!
sequential_7/dense_23/TensordotÎ
,sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_7/dense_23/BiasAdd/ReadVariableOpß
sequential_7/dense_23/BiasAddBiasAdd(sequential_7/dense_23/Tensordot:output:04sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2
sequential_7/dense_23/BiasAdd
sequential_7/dense_23/ReluRelu&sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2
sequential_7/dense_23/ReluØ
.sequential_7/dense_24/Tensordot/ReadVariableOpReadVariableOp7sequential_7_dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_7/dense_24/Tensordot/ReadVariableOp
$sequential_7/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_7/dense_24/Tensordot/axes
$sequential_7/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_7/dense_24/Tensordot/free¦
%sequential_7/dense_24/Tensordot/ShapeShape(sequential_7/dense_23/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/Shape 
-sequential_7/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_24/Tensordot/GatherV2/axis¿
(sequential_7/dense_24/Tensordot/GatherV2GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/free:output:06sequential_7/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/GatherV2¤
/sequential_7/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_24/Tensordot/GatherV2_1/axisÅ
*sequential_7/dense_24/Tensordot/GatherV2_1GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/axes:output:08sequential_7/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_7/dense_24/Tensordot/GatherV2_1
%sequential_7/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_7/dense_24/Tensordot/ConstØ
$sequential_7/dense_24/Tensordot/ProdProd1sequential_7/dense_24/Tensordot/GatherV2:output:0.sequential_7/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_7/dense_24/Tensordot/Prod
'sequential_7/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_24/Tensordot/Const_1à
&sequential_7/dense_24/Tensordot/Prod_1Prod3sequential_7/dense_24/Tensordot/GatherV2_1:output:00sequential_7/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_7/dense_24/Tensordot/Prod_1
+sequential_7/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_7/dense_24/Tensordot/concat/axis
&sequential_7/dense_24/Tensordot/concatConcatV2-sequential_7/dense_24/Tensordot/free:output:0-sequential_7/dense_24/Tensordot/axes:output:04sequential_7/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/dense_24/Tensordot/concatä
%sequential_7/dense_24/Tensordot/stackPack-sequential_7/dense_24/Tensordot/Prod:output:0/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/stackô
)sequential_7/dense_24/Tensordot/transpose	Transpose(sequential_7/dense_23/Relu:activations:0/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2+
)sequential_7/dense_24/Tensordot/transpose÷
'sequential_7/dense_24/Tensordot/ReshapeReshape-sequential_7/dense_24/Tensordot/transpose:y:0.sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_7/dense_24/Tensordot/Reshapeö
&sequential_7/dense_24/Tensordot/MatMulMatMul0sequential_7/dense_24/Tensordot/Reshape:output:06sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_7/dense_24/Tensordot/MatMul
'sequential_7/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_24/Tensordot/Const_2 
-sequential_7/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_24/Tensordot/concat_1/axis«
(sequential_7/dense_24/Tensordot/concat_1ConcatV21sequential_7/dense_24/Tensordot/GatherV2:output:00sequential_7/dense_24/Tensordot/Const_2:output:06sequential_7/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/concat_1è
sequential_7/dense_24/TensordotReshape0sequential_7/dense_24/Tensordot/MatMul:product:01sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2!
sequential_7/dense_24/TensordotÎ
,sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_7/dense_24/BiasAdd/ReadVariableOpß
sequential_7/dense_24/BiasAddBiasAdd(sequential_7/dense_24/Tensordot:output:04sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
sequential_7/dense_24/BiasAdd
dropout_21/IdentityIdentity&sequential_7/dense_24/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dropout_21/Identity
add_1AddV2*layer_normalization_14/batchnorm/add_1:z:0dropout_21/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
add_1¸
5layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_15/moments/mean/reduction_indicesä
#layer_normalization_15/moments/meanMean	add_1:z:0>layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2%
#layer_normalization_15/moments/meanÎ
+layer_normalization_15/moments/StopGradientStopGradient,layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2-
+layer_normalization_15/moments/StopGradientð
0layer_normalization_15/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 22
0layer_normalization_15/moments/SquaredDifferenceÀ
9layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_15/moments/variance/reduction_indices
'layer_normalization_15/moments/varianceMean4layer_normalization_15/moments/SquaredDifference:z:0Blayer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*
	keep_dims(2)
'layer_normalization_15/moments/variance
&layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_15/batchnorm/add/yî
$layer_normalization_15/batchnorm/addAddV20layer_normalization_15/moments/variance:output:0/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2&
$layer_normalization_15/batchnorm/add¹
&layer_normalization_15/batchnorm/RsqrtRsqrt(layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2(
&layer_normalization_15/batchnorm/Rsqrtã
3layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_15/batchnorm/mul/ReadVariableOpò
$layer_normalization_15/batchnorm/mulMul*layer_normalization_15/batchnorm/Rsqrt:y:0;layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2&
$layer_normalization_15/batchnorm/mulÂ
&layer_normalization_15/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_15/batchnorm/mul_1å
&layer_normalization_15/batchnorm/mul_2Mul,layer_normalization_15/moments/mean:output:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_15/batchnorm/mul_2×
/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_15/batchnorm/ReadVariableOpî
$layer_normalization_15/batchnorm/subSub7layer_normalization_15/batchnorm/ReadVariableOp:value:0*layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2&
$layer_normalization_15/batchnorm/subå
&layer_normalization_15/batchnorm/add_1AddV2*layer_normalization_15/batchnorm/mul_1:z:0(layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2(
&layer_normalization_15/batchnorm/add_1Ü
IdentityIdentity*layer_normalization_15/batchnorm/add_1:z:00^layer_normalization_14/batchnorm/ReadVariableOp4^layer_normalization_14/batchnorm/mul/ReadVariableOp0^layer_normalization_15/batchnorm/ReadVariableOp4^layer_normalization_15/batchnorm/mul/ReadVariableOp;^multi_head_attention_7/attention_output/add/ReadVariableOpE^multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_7/key/add/ReadVariableOp8^multi_head_attention_7/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/query/add/ReadVariableOp:^multi_head_attention_7/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/value/add/ReadVariableOp:^multi_head_attention_7/value/einsum/Einsum/ReadVariableOp-^sequential_7/dense_23/BiasAdd/ReadVariableOp/^sequential_7/dense_23/Tensordot/ReadVariableOp-^sequential_7/dense_24/BiasAdd/ReadVariableOp/^sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿB ::::::::::::::::2b
/layer_normalization_14/batchnorm/ReadVariableOp/layer_normalization_14/batchnorm/ReadVariableOp2j
3layer_normalization_14/batchnorm/mul/ReadVariableOp3layer_normalization_14/batchnorm/mul/ReadVariableOp2b
/layer_normalization_15/batchnorm/ReadVariableOp/layer_normalization_15/batchnorm/ReadVariableOp2j
3layer_normalization_15/batchnorm/mul/ReadVariableOp3layer_normalization_15/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_7/attention_output/add/ReadVariableOp:multi_head_attention_7/attention_output/add/ReadVariableOp2
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_7/key/add/ReadVariableOp-multi_head_attention_7/key/add/ReadVariableOp2r
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/query/add/ReadVariableOp/multi_head_attention_7/query/add/ReadVariableOp2v
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/value/add/ReadVariableOp/multi_head_attention_7/value/add/ReadVariableOp2v
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp2\
,sequential_7/dense_23/BiasAdd/ReadVariableOp,sequential_7/dense_23/BiasAdd/ReadVariableOp2`
.sequential_7/dense_23/Tensordot/ReadVariableOp.sequential_7/dense_23/Tensordot/ReadVariableOp2\
,sequential_7/dense_24/BiasAdd/ReadVariableOp,sequential_7/dense_24/BiasAdd/ReadVariableOp2`
.sequential_7/dense_24/Tensordot/ReadVariableOp.sequential_7/dense_24/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
î
©
6__inference_batch_normalization_6_layer_call_fn_197169

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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1945042
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


Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_194644

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
Â
u
I__inference_concatenate_3_layer_call_and_return_conditional_losses_197712
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
:ÿÿÿÿÿÿÿÿÿÈ2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÀ:ÿÿÿÿÿÿÿÿÿ:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ê
©
6__inference_batch_normalization_7_layer_call_fn_197333

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
:ÿÿÿÿÿÿÿÿÿB *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1950802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
¬
R
&__inference_add_3_layer_call_fn_197345
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
:ÿÿÿÿÿÿÿÿÿB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_1951222
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿB :ÿÿÿÿÿÿÿÿÿB :U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
"
_user_specified_name
inputs/1
Ð

à
4__inference_transformer_block_7_layer_call_fn_197694

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
:ÿÿÿÿÿÿÿÿÿB *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_1954062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿB ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
ó0
È
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_194611

inputs
assignmovingavg_194586
assignmovingavg_1_194592)
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
loc:@AssignMovingAvg/194586*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_194586*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/194586*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/194586*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_194586AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/194586*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/194592*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_194592*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/194592*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/194592*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_194592AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/194592*
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

Q
5__inference_average_pooling1d_11_layer_call_fn_194375

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
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_1943692
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
Ð
¨
-__inference_sequential_7_layer_call_fn_194822
dense_23_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCalldense_23_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_1948112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
(
_user_specified_namedense_23_input

÷
D__inference_conv1d_7_layer_call_and_return_conditional_losses_194916

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
:ÿÿÿÿÿÿÿÿÿ 2
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
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
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
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¼0
È
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_197287

inputs
assignmovingavg_197262
assignmovingavg_1_197268)
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
:ÿÿÿÿÿÿÿÿÿB 2
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
loc:@AssignMovingAvg/197262*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_197262*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/197262*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/197262*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_197262AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/197262*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/197268*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_197268*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/197268*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/197268*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_197268AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/197268*
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
:ÿÿÿÿÿÿÿÿÿB 2
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
:ÿÿÿÿÿÿÿÿÿB 2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
è

Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_197061

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
:ÿÿÿÿÿÿÿÿÿB 2
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
:ÿÿÿÿÿÿÿÿÿB 2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs


H__inference_sequential_7_layer_call_and_return_conditional_losses_194767
dense_23_input
dense_23_194756
dense_23_194758
dense_24_194761
dense_24_194763
identity¢ dense_23/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCall£
 dense_23/StatefulPartitionedCallStatefulPartitionedCalldense_23_inputdense_23_194756dense_23_194758*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_1946902"
 dense_23/StatefulPartitionedCall¾
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_194761dense_24_194763*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_1947362"
 dense_24/StatefulPartitionedCallÇ
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
(
_user_specified_namedense_23_input


Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_197143

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
î	
Ý
D__inference_dense_26_layer_call_and_return_conditional_losses_195613

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
÷
~
)__inference_conv1d_6_layer_call_fn_196980

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
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_1948832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ  ::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
º
s
I__inference_concatenate_3_layer_call_and_return_conditional_losses_195536

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
:ÿÿÿÿÿÿÿÿÿÈ2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÀ:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
 
-__inference_sequential_7_layer_call_fn_197958

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
:ÿÿÿÿÿÿÿÿÿB *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_1947842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
Ò
¢
(__inference_model_3_layer_call_fn_195953
input_7
input_8
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
identity¢StatefulPartitionedCallÐ
StatefulPartitionedCallStatefulPartitionedCallinput_7input_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
C__inference_model_3_layer_call_and_return_conditional_losses_1958782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Í
_input_shapes»
¸:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!
_user_specified_name	input_7:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_8
É
d
F__inference_dropout_23_layer_call_and_return_conditional_losses_195646

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
D__inference_conv1d_7_layer_call_and_return_conditional_losses_196996

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
:ÿÿÿÿÿÿÿÿÿ 2
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
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
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
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¼0
È
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_194969

inputs
assignmovingavg_194944
assignmovingavg_1_194950)
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
:ÿÿÿÿÿÿÿÿÿB 2
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
loc:@AssignMovingAvg/194944*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_194944*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/194944*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/194944*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_194944AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/194944*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/194950*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_194950*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/194950*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/194950*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_194950AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/194950*
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
:ÿÿÿÿÿÿÿÿÿB 2
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
:ÿÿÿÿÿÿÿÿÿB 2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs

e
F__inference_dropout_23_layer_call_and_return_conditional_losses_197797

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

seed*2&
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
ëX

C__inference_model_3_layer_call_and_return_conditional_losses_196050

inputs
inputs_1)
%token_and_position_embedding_3_195960)
%token_and_position_embedding_3_195962
conv1d_6_195965
conv1d_6_195967
conv1d_7_195971
conv1d_7_195973 
batch_normalization_6_195978 
batch_normalization_6_195980 
batch_normalization_6_195982 
batch_normalization_6_195984 
batch_normalization_7_195987 
batch_normalization_7_195989 
batch_normalization_7_195991 
batch_normalization_7_195993
transformer_block_7_195997
transformer_block_7_195999
transformer_block_7_196001
transformer_block_7_196003
transformer_block_7_196005
transformer_block_7_196007
transformer_block_7_196009
transformer_block_7_196011
transformer_block_7_196013
transformer_block_7_196015
transformer_block_7_196017
transformer_block_7_196019
transformer_block_7_196021
transformer_block_7_196023
transformer_block_7_196025
transformer_block_7_196027
dense_25_196032
dense_25_196034
dense_26_196038
dense_26_196040
dense_27_196044
dense_27_196046
identity¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢ conv1d_6/StatefulPartitionedCall¢ conv1d_7/StatefulPartitionedCall¢ dense_25/StatefulPartitionedCall¢ dense_26/StatefulPartitionedCall¢ dense_27/StatefulPartitionedCall¢6token_and_position_embedding_3/StatefulPartitionedCall¢+transformer_block_7/StatefulPartitionedCall
6token_and_position_embedding_3/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_3_195960%token_and_position_embedding_3_195962*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_19485128
6token_and_position_embedding_3/StatefulPartitionedCallÖ
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0conv1d_6_195965conv1d_6_195967*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_1948832"
 conv1d_6/StatefulPartitionedCall 
#average_pooling1d_9/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_1943392%
#average_pooling1d_9/PartitionedCallÂ
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_9/PartitionedCall:output:0conv1d_7_195971conv1d_7_195973*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_1949162"
 conv1d_7/StatefulPartitionedCall¸
$average_pooling1d_11/PartitionedCallPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_1943692&
$average_pooling1d_11/PartitionedCall¢
$average_pooling1d_10/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_1943542&
$average_pooling1d_10/PartitionedCallÃ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_10/PartitionedCall:output:0batch_normalization_6_195978batch_normalization_6_195980batch_normalization_6_195982batch_normalization_6_195984*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1949892/
-batch_normalization_6/StatefulPartitionedCallÃ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_11/PartitionedCall:output:0batch_normalization_7_195987batch_normalization_7_195989batch_normalization_7_195991batch_normalization_7_195993*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1950802/
-batch_normalization_7/StatefulPartitionedCall»
add_3/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_1951222
add_3/PartitionedCall
+transformer_block_7/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0transformer_block_7_195997transformer_block_7_195999transformer_block_7_196001transformer_block_7_196003transformer_block_7_196005transformer_block_7_196007transformer_block_7_196009transformer_block_7_196011transformer_block_7_196013transformer_block_7_196015transformer_block_7_196017transformer_block_7_196019transformer_block_7_196021transformer_block_7_196023transformer_block_7_196025transformer_block_7_196027*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_1954062-
+transformer_block_7/StatefulPartitionedCall
flatten_3/PartitionedCallPartitionedCall4transformer_block_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1955212
flatten_3/PartitionedCall
concatenate_3/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_1955362
concatenate_3/PartitionedCall·
 dense_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_25_196032dense_25_196034*
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
D__inference_dense_25_layer_call_and_return_conditional_losses_1955562"
 dense_25/StatefulPartitionedCall
dropout_22/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
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
F__inference_dropout_22_layer_call_and_return_conditional_losses_1955892
dropout_22/PartitionedCall´
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_26_196038dense_26_196040*
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
D__inference_dense_26_layer_call_and_return_conditional_losses_1956132"
 dense_26/StatefulPartitionedCall
dropout_23/PartitionedCallPartitionedCall)dense_26/StatefulPartitionedCall:output:0*
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
F__inference_dropout_23_layer_call_and_return_conditional_losses_1956462
dropout_23/PartitionedCall´
 dense_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_27_196044dense_27_196046*
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
D__inference_dense_27_layer_call_and_return_conditional_losses_1956692"
 dense_27/StatefulPartitionedCalló
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall7^token_and_position_embedding_3/StatefulPartitionedCall,^transformer_block_7/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Í
_input_shapes»
¸:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2p
6token_and_position_embedding_3/StatefulPartitionedCall6token_and_position_embedding_3/StatefulPartitionedCall2Z
+transformer_block_7/StatefulPartitionedCall+transformer_block_7/StatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î	
Ý
D__inference_dense_26_layer_call_and_return_conditional_losses_197776

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
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_194339

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
¸
 
-__inference_sequential_7_layer_call_fn_197971

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
:ÿÿÿÿÿÿÿÿÿB *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_1948112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
¢
÷
D__inference_conv1d_6_layer_call_and_return_conditional_losses_196971

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
:ÿÿÿÿÿÿÿÿÿ  2
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
conv1d/ExpandDims_1¸
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2	
BiasAdd^
ReluReluBiasAdd:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
Reluª
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs


H__inference_sequential_7_layer_call_and_return_conditional_losses_194753
dense_23_input
dense_23_194701
dense_23_194703
dense_24_194747
dense_24_194749
identity¢ dense_23/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCall£
 dense_23/StatefulPartitionedCallStatefulPartitionedCalldense_23_inputdense_23_194701dense_23_194703*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_1946902"
 dense_23/StatefulPartitionedCall¾
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_194747dense_24_194749*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_1947362"
 dense_24/StatefulPartitionedCallÇ
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
(
_user_specified_namedense_23_input
J
¯
H__inference_sequential_7_layer_call_and_return_conditional_losses_197888

inputs.
*dense_23_tensordot_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource.
*dense_24_tensordot_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource
identity¢dense_23/BiasAdd/ReadVariableOp¢!dense_23/Tensordot/ReadVariableOp¢dense_24/BiasAdd/ReadVariableOp¢!dense_24/Tensordot/ReadVariableOp±
!dense_23/Tensordot/ReadVariableOpReadVariableOp*dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02#
!dense_23/Tensordot/ReadVariableOp|
dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_23/Tensordot/axes
dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_23/Tensordot/freej
dense_23/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_23/Tensordot/Shape
 dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_23/Tensordot/GatherV2/axisþ
dense_23/Tensordot/GatherV2GatherV2!dense_23/Tensordot/Shape:output:0 dense_23/Tensordot/free:output:0)dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_23/Tensordot/GatherV2
"dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_23/Tensordot/GatherV2_1/axis
dense_23/Tensordot/GatherV2_1GatherV2!dense_23/Tensordot/Shape:output:0 dense_23/Tensordot/axes:output:0+dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_23/Tensordot/GatherV2_1~
dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_23/Tensordot/Const¤
dense_23/Tensordot/ProdProd$dense_23/Tensordot/GatherV2:output:0!dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_23/Tensordot/Prod
dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_23/Tensordot/Const_1¬
dense_23/Tensordot/Prod_1Prod&dense_23/Tensordot/GatherV2_1:output:0#dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_23/Tensordot/Prod_1
dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_23/Tensordot/concat/axisÝ
dense_23/Tensordot/concatConcatV2 dense_23/Tensordot/free:output:0 dense_23/Tensordot/axes:output:0'dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/concat°
dense_23/Tensordot/stackPack dense_23/Tensordot/Prod:output:0"dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/stack«
dense_23/Tensordot/transpose	Transposeinputs"dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dense_23/Tensordot/transposeÃ
dense_23/Tensordot/ReshapeReshape dense_23/Tensordot/transpose:y:0!dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_23/Tensordot/ReshapeÂ
dense_23/Tensordot/MatMulMatMul#dense_23/Tensordot/Reshape:output:0)dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_23/Tensordot/MatMul
dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_23/Tensordot/Const_2
 dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_23/Tensordot/concat_1/axisê
dense_23/Tensordot/concat_1ConcatV2$dense_23/Tensordot/GatherV2:output:0#dense_23/Tensordot/Const_2:output:0)dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/concat_1´
dense_23/TensordotReshape#dense_23/Tensordot/MatMul:product:0$dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2
dense_23/Tensordot§
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_23/BiasAdd/ReadVariableOp«
dense_23/BiasAddBiasAdddense_23/Tensordot:output:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2
dense_23/BiasAddw
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2
dense_23/Relu±
!dense_24/Tensordot/ReadVariableOpReadVariableOp*dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02#
!dense_24/Tensordot/ReadVariableOp|
dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_24/Tensordot/axes
dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_24/Tensordot/free
dense_24/Tensordot/ShapeShapedense_23/Relu:activations:0*
T0*
_output_shapes
:2
dense_24/Tensordot/Shape
 dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_24/Tensordot/GatherV2/axisþ
dense_24/Tensordot/GatherV2GatherV2!dense_24/Tensordot/Shape:output:0 dense_24/Tensordot/free:output:0)dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_24/Tensordot/GatherV2
"dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_24/Tensordot/GatherV2_1/axis
dense_24/Tensordot/GatherV2_1GatherV2!dense_24/Tensordot/Shape:output:0 dense_24/Tensordot/axes:output:0+dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_24/Tensordot/GatherV2_1~
dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const¤
dense_24/Tensordot/ProdProd$dense_24/Tensordot/GatherV2:output:0!dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_24/Tensordot/Prod
dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const_1¬
dense_24/Tensordot/Prod_1Prod&dense_24/Tensordot/GatherV2_1:output:0#dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_24/Tensordot/Prod_1
dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_24/Tensordot/concat/axisÝ
dense_24/Tensordot/concatConcatV2 dense_24/Tensordot/free:output:0 dense_24/Tensordot/axes:output:0'dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/concat°
dense_24/Tensordot/stackPack dense_24/Tensordot/Prod:output:0"dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/stackÀ
dense_24/Tensordot/transpose	Transposedense_23/Relu:activations:0"dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2
dense_24/Tensordot/transposeÃ
dense_24/Tensordot/ReshapeReshape dense_24/Tensordot/transpose:y:0!dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_24/Tensordot/ReshapeÂ
dense_24/Tensordot/MatMulMatMul#dense_24/Tensordot/Reshape:output:0)dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_24/Tensordot/MatMul
dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const_2
 dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_24/Tensordot/concat_1/axisê
dense_24/Tensordot/concat_1ConcatV2$dense_24/Tensordot/GatherV2:output:0#dense_24/Tensordot/Const_2:output:0)dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/concat_1´
dense_24/TensordotReshape#dense_24/Tensordot/MatMul:product:0$dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dense_24/Tensordot§
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_24/BiasAdd/ReadVariableOp«
dense_24/BiasAddBiasAdddense_24/Tensordot:output:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
dense_24/BiasAddý
IdentityIdentitydense_24/BiasAdd:output:0 ^dense_23/BiasAdd/ReadVariableOp"^dense_23/Tensordot/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp"^dense_24/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2F
!dense_23/Tensordot/ReadVariableOp!dense_23/Tensordot/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2F
!dense_24/Tensordot/ReadVariableOp!dense_24/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
è

Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_194989

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
:ÿÿÿÿÿÿÿÿÿB 2
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
:ÿÿÿÿÿÿÿÿÿB 2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
Ø
¤
(__inference_model_3_layer_call_fn_196844
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
C__inference_model_3_layer_call_and_return_conditional_losses_1958782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Í
_input_shapes»
¸:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ß
~
)__inference_dense_27_layer_call_fn_197831

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
D__inference_dense_27_layer_call_and_return_conditional_losses_1956692
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


Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_194504

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
6__inference_batch_normalization_6_layer_call_fn_197087

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
:ÿÿÿÿÿÿÿÿÿB *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1949892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
è

Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_197307

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
:ÿÿÿÿÿÿÿÿÿB 2
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
:ÿÿÿÿÿÿÿÿÿB 2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
É
d
F__inference_dropout_22_layer_call_and_return_conditional_losses_195589

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

e
F__inference_dropout_22_layer_call_and_return_conditional_losses_197750

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

seed*2&
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


?__inference_token_and_position_embedding_3_layer_call_fn_196955
x
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_1948512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:L H
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 

_user_specified_namex
ü

Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_196946
x'
#embedding_7_embedding_lookup_196933'
#embedding_6_embedding_lookup_196939
identity¢embedding_6/embedding_lookup¢embedding_7/embedding_lookup?
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
embedding_7/embedding_lookupResourceGather#embedding_7_embedding_lookup_196933range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_7/embedding_lookup/196933*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02
embedding_7/embedding_lookup
%embedding_7/embedding_lookup/IdentityIdentity%embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_7/embedding_lookup/196933*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%embedding_7/embedding_lookup/IdentityÀ
'embedding_7/embedding_lookup/Identity_1Identity.embedding_7/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'embedding_7/embedding_lookup/Identity_1r
embedding_6/CastCastx*

DstT0*

SrcT0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
embedding_6/Cast»
embedding_6/embedding_lookupResourceGather#embedding_6_embedding_lookup_196939embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_6/embedding_lookup/196939*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype02
embedding_6/embedding_lookup
%embedding_6/embedding_lookup/IdentityIdentity%embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_6/embedding_lookup/196939*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2'
%embedding_6/embedding_lookup/IdentityÆ
'embedding_6/embedding_lookup/Identity_1Identity.embedding_6/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2)
'embedding_6/embedding_lookup/Identity_1¯
addAddV20embedding_6/embedding_lookup/Identity_1:output:00embedding_7/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
add
IdentityIdentityadd:z:0^embedding_6/embedding_lookup^embedding_7/embedding_lookup*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::2<
embedding_6/embedding_lookupembedding_6/embedding_lookup2<
embedding_7/embedding_lookupembedding_7/embedding_lookup:L H
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 

_user_specified_namex
¼0
È
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_197041

inputs
assignmovingavg_197016
assignmovingavg_1_197022)
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
:ÿÿÿÿÿÿÿÿÿB 2
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
loc:@AssignMovingAvg/197016*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_197016*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/197016*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/197016*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_197016AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/197016*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/197022*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_197022*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/197022*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/197022*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_197022AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/197022*
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
:ÿÿÿÿÿÿÿÿÿB 2
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
:ÿÿÿÿÿÿÿÿÿB 2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
¨
Z
.__inference_concatenate_3_layer_call_fn_197718
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
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_1955362
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÀ:ÿÿÿÿÿÿÿÿÿ:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
	
Ý
D__inference_dense_27_layer_call_and_return_conditional_losses_195669

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
á
~
)__inference_dense_25_layer_call_fn_197738

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
D__inference_dense_25_layer_call_and_return_conditional_losses_1955562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
± 
ã
D__inference_dense_23_layer_call_and_return_conditional_losses_198002

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
:ÿÿÿÿÿÿÿÿÿB 2
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
:ÿÿÿÿÿÿÿÿÿB@2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿB ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
ì
©
6__inference_batch_normalization_6_layer_call_fn_197156

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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1944712
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
¿
m
A__inference_add_3_layer_call_and_return_conditional_losses_197339
inputs_0
inputs_1
identity]
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿB :ÿÿÿÿÿÿÿÿÿB :U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
"
_user_specified_name
inputs/1
Ü
¤
(__inference_model_3_layer_call_fn_196922
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
C__inference_model_3_layer_call_and_return_conditional_losses_1960502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Í
_input_shapes»
¸:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ð
¨
-__inference_sequential_7_layer_call_fn_194795
dense_23_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCalldense_23_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_1947842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿB ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
(
_user_specified_namedense_23_input
Ñ
ã
D__inference_dense_24_layer_call_and_return_conditional_losses_194736

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
:ÿÿÿÿÿÿÿÿÿB@2
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
:ÿÿÿÿÿÿÿÿÿB 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿB@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@
 
_user_specified_nameinputs
Ñ
ã
D__inference_dense_24_layer_call_and_return_conditional_losses_198041

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
:ÿÿÿÿÿÿÿÿÿB@2
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
:ÿÿÿÿÿÿÿÿÿB 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿB@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB@
 
_user_specified_nameinputs
Ö
¢
(__inference_model_3_layer_call_fn_196125
input_7
input_8
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
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinput_7input_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
C__inference_model_3_layer_call_and_return_conditional_losses_1960502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Í
_input_shapes»
¸:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!
_user_specified_name	input_7:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_8
Ð

à
4__inference_transformer_block_7_layer_call_fn_197657

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
:ÿÿÿÿÿÿÿÿÿB *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_1952792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿB ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB 
 
_user_specified_nameinputs
	
Ý
D__inference_dense_27_layer_call_and_return_conditional_losses_197822

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


Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_197225

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
ô[
è
C__inference_model_3_layer_call_and_return_conditional_losses_195878

inputs
inputs_1)
%token_and_position_embedding_3_195788)
%token_and_position_embedding_3_195790
conv1d_6_195793
conv1d_6_195795
conv1d_7_195799
conv1d_7_195801 
batch_normalization_6_195806 
batch_normalization_6_195808 
batch_normalization_6_195810 
batch_normalization_6_195812 
batch_normalization_7_195815 
batch_normalization_7_195817 
batch_normalization_7_195819 
batch_normalization_7_195821
transformer_block_7_195825
transformer_block_7_195827
transformer_block_7_195829
transformer_block_7_195831
transformer_block_7_195833
transformer_block_7_195835
transformer_block_7_195837
transformer_block_7_195839
transformer_block_7_195841
transformer_block_7_195843
transformer_block_7_195845
transformer_block_7_195847
transformer_block_7_195849
transformer_block_7_195851
transformer_block_7_195853
transformer_block_7_195855
dense_25_195860
dense_25_195862
dense_26_195866
dense_26_195868
dense_27_195872
dense_27_195874
identity¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢ conv1d_6/StatefulPartitionedCall¢ conv1d_7/StatefulPartitionedCall¢ dense_25/StatefulPartitionedCall¢ dense_26/StatefulPartitionedCall¢ dense_27/StatefulPartitionedCall¢"dropout_22/StatefulPartitionedCall¢"dropout_23/StatefulPartitionedCall¢6token_and_position_embedding_3/StatefulPartitionedCall¢+transformer_block_7/StatefulPartitionedCall
6token_and_position_embedding_3/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_3_195788%token_and_position_embedding_3_195790*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_19485128
6token_and_position_embedding_3/StatefulPartitionedCallÖ
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0conv1d_6_195793conv1d_6_195795*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_1948832"
 conv1d_6/StatefulPartitionedCall 
#average_pooling1d_9/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_1943392%
#average_pooling1d_9/PartitionedCallÂ
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_9/PartitionedCall:output:0conv1d_7_195799conv1d_7_195801*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_1949162"
 conv1d_7/StatefulPartitionedCall¸
$average_pooling1d_11/PartitionedCallPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_1943692&
$average_pooling1d_11/PartitionedCall¢
$average_pooling1d_10/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_1943542&
$average_pooling1d_10/PartitionedCallÁ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_10/PartitionedCall:output:0batch_normalization_6_195806batch_normalization_6_195808batch_normalization_6_195810batch_normalization_6_195812*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1949692/
-batch_normalization_6/StatefulPartitionedCallÁ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_11/PartitionedCall:output:0batch_normalization_7_195815batch_normalization_7_195817batch_normalization_7_195819batch_normalization_7_195821*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1950602/
-batch_normalization_7/StatefulPartitionedCall»
add_3/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_1951222
add_3/PartitionedCall
+transformer_block_7/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0transformer_block_7_195825transformer_block_7_195827transformer_block_7_195829transformer_block_7_195831transformer_block_7_195833transformer_block_7_195835transformer_block_7_195837transformer_block_7_195839transformer_block_7_195841transformer_block_7_195843transformer_block_7_195845transformer_block_7_195847transformer_block_7_195849transformer_block_7_195851transformer_block_7_195853transformer_block_7_195855*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_1952792-
+transformer_block_7/StatefulPartitionedCall
flatten_3/PartitionedCallPartitionedCall4transformer_block_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1955212
flatten_3/PartitionedCall
concatenate_3/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_1955362
concatenate_3/PartitionedCall·
 dense_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_25_195860dense_25_195862*
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
D__inference_dense_25_layer_call_and_return_conditional_losses_1955562"
 dense_25/StatefulPartitionedCall
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
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
F__inference_dropout_22_layer_call_and_return_conditional_losses_1955842$
"dropout_22/StatefulPartitionedCall¼
 dense_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_26_195866dense_26_195868*
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
D__inference_dense_26_layer_call_and_return_conditional_losses_1956132"
 dense_26/StatefulPartitionedCall½
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
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
F__inference_dropout_23_layer_call_and_return_conditional_losses_1956412$
"dropout_23/StatefulPartitionedCall¼
 dense_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_27_195872dense_27_195874*
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
D__inference_dense_27_layer_call_and_return_conditional_losses_1956692"
 dense_27/StatefulPartitionedCall½
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall7^token_and_position_embedding_3/StatefulPartitionedCall,^transformer_block_7/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Í
_input_shapes»
¸:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall2p
6token_and_position_embedding_3/StatefulPartitionedCall6token_and_position_embedding_3/StatefulPartitionedCall2Z
+transformer_block_7/StatefulPartitionedCall+transformer_block_7/StatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£ß
ü9
__inference__traced_save_198395
file_prefix.
*savev2_conv1d_6_kernel_read_readvariableop,
(savev2_conv1d_6_bias_read_readvariableop.
*savev2_conv1d_7_kernel_read_readvariableop,
(savev2_conv1d_7_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	T
Psavev2_token_and_position_embedding_3_embedding_6_embeddings_read_readvariableopT
Psavev2_token_and_position_embedding_3_embedding_7_embeddings_read_readvariableopV
Rsavev2_transformer_block_7_multi_head_attention_7_query_kernel_read_readvariableopT
Psavev2_transformer_block_7_multi_head_attention_7_query_bias_read_readvariableopT
Psavev2_transformer_block_7_multi_head_attention_7_key_kernel_read_readvariableopR
Nsavev2_transformer_block_7_multi_head_attention_7_key_bias_read_readvariableopV
Rsavev2_transformer_block_7_multi_head_attention_7_value_kernel_read_readvariableopT
Psavev2_transformer_block_7_multi_head_attention_7_value_bias_read_readvariableopa
]savev2_transformer_block_7_multi_head_attention_7_attention_output_kernel_read_readvariableop_
[savev2_transformer_block_7_multi_head_attention_7_attention_output_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableopO
Ksavev2_transformer_block_7_layer_normalization_14_gamma_read_readvariableopN
Jsavev2_transformer_block_7_layer_normalization_14_beta_read_readvariableopO
Ksavev2_transformer_block_7_layer_normalization_15_gamma_read_readvariableopN
Jsavev2_transformer_block_7_layer_normalization_15_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_conv1d_6_kernel_m_read_readvariableop3
/savev2_adam_conv1d_6_bias_m_read_readvariableop5
1savev2_adam_conv1d_7_kernel_m_read_readvariableop3
/savev2_adam_conv1d_7_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_m_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_m_read_readvariableop5
1savev2_adam_dense_25_kernel_m_read_readvariableop3
/savev2_adam_dense_25_bias_m_read_readvariableop5
1savev2_adam_dense_26_kernel_m_read_readvariableop3
/savev2_adam_dense_26_bias_m_read_readvariableop5
1savev2_adam_dense_27_kernel_m_read_readvariableop3
/savev2_adam_dense_27_bias_m_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_3_embedding_6_embeddings_m_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_3_embedding_7_embeddings_m_read_readvariableop]
Ysavev2_adam_transformer_block_7_multi_head_attention_7_query_kernel_m_read_readvariableop[
Wsavev2_adam_transformer_block_7_multi_head_attention_7_query_bias_m_read_readvariableop[
Wsavev2_adam_transformer_block_7_multi_head_attention_7_key_kernel_m_read_readvariableopY
Usavev2_adam_transformer_block_7_multi_head_attention_7_key_bias_m_read_readvariableop]
Ysavev2_adam_transformer_block_7_multi_head_attention_7_value_kernel_m_read_readvariableop[
Wsavev2_adam_transformer_block_7_multi_head_attention_7_value_bias_m_read_readvariableoph
dsavev2_adam_transformer_block_7_multi_head_attention_7_attention_output_kernel_m_read_readvariableopf
bsavev2_adam_transformer_block_7_multi_head_attention_7_attention_output_bias_m_read_readvariableop5
1savev2_adam_dense_23_kernel_m_read_readvariableop3
/savev2_adam_dense_23_bias_m_read_readvariableop5
1savev2_adam_dense_24_kernel_m_read_readvariableop3
/savev2_adam_dense_24_bias_m_read_readvariableopV
Rsavev2_adam_transformer_block_7_layer_normalization_14_gamma_m_read_readvariableopU
Qsavev2_adam_transformer_block_7_layer_normalization_14_beta_m_read_readvariableopV
Rsavev2_adam_transformer_block_7_layer_normalization_15_gamma_m_read_readvariableopU
Qsavev2_adam_transformer_block_7_layer_normalization_15_beta_m_read_readvariableop5
1savev2_adam_conv1d_6_kernel_v_read_readvariableop3
/savev2_adam_conv1d_6_bias_v_read_readvariableop5
1savev2_adam_conv1d_7_kernel_v_read_readvariableop3
/savev2_adam_conv1d_7_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_v_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_v_read_readvariableop5
1savev2_adam_dense_25_kernel_v_read_readvariableop3
/savev2_adam_dense_25_bias_v_read_readvariableop5
1savev2_adam_dense_26_kernel_v_read_readvariableop3
/savev2_adam_dense_26_bias_v_read_readvariableop5
1savev2_adam_dense_27_kernel_v_read_readvariableop3
/savev2_adam_dense_27_bias_v_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_3_embedding_6_embeddings_v_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_3_embedding_7_embeddings_v_read_readvariableop]
Ysavev2_adam_transformer_block_7_multi_head_attention_7_query_kernel_v_read_readvariableop[
Wsavev2_adam_transformer_block_7_multi_head_attention_7_query_bias_v_read_readvariableop[
Wsavev2_adam_transformer_block_7_multi_head_attention_7_key_kernel_v_read_readvariableopY
Usavev2_adam_transformer_block_7_multi_head_attention_7_key_bias_v_read_readvariableop]
Ysavev2_adam_transformer_block_7_multi_head_attention_7_value_kernel_v_read_readvariableop[
Wsavev2_adam_transformer_block_7_multi_head_attention_7_value_bias_v_read_readvariableoph
dsavev2_adam_transformer_block_7_multi_head_attention_7_attention_output_kernel_v_read_readvariableopf
bsavev2_adam_transformer_block_7_multi_head_attention_7_attention_output_bias_v_read_readvariableop5
1savev2_adam_dense_23_kernel_v_read_readvariableop3
/savev2_adam_dense_23_bias_v_read_readvariableop5
1savev2_adam_dense_24_kernel_v_read_readvariableop3
/savev2_adam_dense_24_bias_v_read_readvariableopV
Rsavev2_adam_transformer_block_7_layer_normalization_14_gamma_v_read_readvariableopU
Qsavev2_adam_transformer_block_7_layer_normalization_14_beta_v_read_readvariableopV
Rsavev2_adam_transformer_block_7_layer_normalization_15_gamma_v_read_readvariableopU
Qsavev2_adam_transformer_block_7_layer_normalization_15_beta_v_read_readvariableop
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
ShardedFilename7
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:l*
dtype0* 6
value6B6lB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesã
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:l*
dtype0*í
valueãBàlB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices8
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_6_kernel_read_readvariableop(savev2_conv1d_6_bias_read_readvariableop*savev2_conv1d_7_kernel_read_readvariableop(savev2_conv1d_7_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableopPsavev2_token_and_position_embedding_3_embedding_6_embeddings_read_readvariableopPsavev2_token_and_position_embedding_3_embedding_7_embeddings_read_readvariableopRsavev2_transformer_block_7_multi_head_attention_7_query_kernel_read_readvariableopPsavev2_transformer_block_7_multi_head_attention_7_query_bias_read_readvariableopPsavev2_transformer_block_7_multi_head_attention_7_key_kernel_read_readvariableopNsavev2_transformer_block_7_multi_head_attention_7_key_bias_read_readvariableopRsavev2_transformer_block_7_multi_head_attention_7_value_kernel_read_readvariableopPsavev2_transformer_block_7_multi_head_attention_7_value_bias_read_readvariableop]savev2_transformer_block_7_multi_head_attention_7_attention_output_kernel_read_readvariableop[savev2_transformer_block_7_multi_head_attention_7_attention_output_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableopKsavev2_transformer_block_7_layer_normalization_14_gamma_read_readvariableopJsavev2_transformer_block_7_layer_normalization_14_beta_read_readvariableopKsavev2_transformer_block_7_layer_normalization_15_gamma_read_readvariableopJsavev2_transformer_block_7_layer_normalization_15_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_conv1d_6_kernel_m_read_readvariableop/savev2_adam_conv1d_6_bias_m_read_readvariableop1savev2_adam_conv1d_7_kernel_m_read_readvariableop/savev2_adam_conv1d_7_bias_m_read_readvariableop=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop<savev2_adam_batch_normalization_6_beta_m_read_readvariableop=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop<savev2_adam_batch_normalization_7_beta_m_read_readvariableop1savev2_adam_dense_25_kernel_m_read_readvariableop/savev2_adam_dense_25_bias_m_read_readvariableop1savev2_adam_dense_26_kernel_m_read_readvariableop/savev2_adam_dense_26_bias_m_read_readvariableop1savev2_adam_dense_27_kernel_m_read_readvariableop/savev2_adam_dense_27_bias_m_read_readvariableopWsavev2_adam_token_and_position_embedding_3_embedding_6_embeddings_m_read_readvariableopWsavev2_adam_token_and_position_embedding_3_embedding_7_embeddings_m_read_readvariableopYsavev2_adam_transformer_block_7_multi_head_attention_7_query_kernel_m_read_readvariableopWsavev2_adam_transformer_block_7_multi_head_attention_7_query_bias_m_read_readvariableopWsavev2_adam_transformer_block_7_multi_head_attention_7_key_kernel_m_read_readvariableopUsavev2_adam_transformer_block_7_multi_head_attention_7_key_bias_m_read_readvariableopYsavev2_adam_transformer_block_7_multi_head_attention_7_value_kernel_m_read_readvariableopWsavev2_adam_transformer_block_7_multi_head_attention_7_value_bias_m_read_readvariableopdsavev2_adam_transformer_block_7_multi_head_attention_7_attention_output_kernel_m_read_readvariableopbsavev2_adam_transformer_block_7_multi_head_attention_7_attention_output_bias_m_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableop1savev2_adam_dense_24_kernel_m_read_readvariableop/savev2_adam_dense_24_bias_m_read_readvariableopRsavev2_adam_transformer_block_7_layer_normalization_14_gamma_m_read_readvariableopQsavev2_adam_transformer_block_7_layer_normalization_14_beta_m_read_readvariableopRsavev2_adam_transformer_block_7_layer_normalization_15_gamma_m_read_readvariableopQsavev2_adam_transformer_block_7_layer_normalization_15_beta_m_read_readvariableop1savev2_adam_conv1d_6_kernel_v_read_readvariableop/savev2_adam_conv1d_6_bias_v_read_readvariableop1savev2_adam_conv1d_7_kernel_v_read_readvariableop/savev2_adam_conv1d_7_bias_v_read_readvariableop=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop<savev2_adam_batch_normalization_6_beta_v_read_readvariableop=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop<savev2_adam_batch_normalization_7_beta_v_read_readvariableop1savev2_adam_dense_25_kernel_v_read_readvariableop/savev2_adam_dense_25_bias_v_read_readvariableop1savev2_adam_dense_26_kernel_v_read_readvariableop/savev2_adam_dense_26_bias_v_read_readvariableop1savev2_adam_dense_27_kernel_v_read_readvariableop/savev2_adam_dense_27_bias_v_read_readvariableopWsavev2_adam_token_and_position_embedding_3_embedding_6_embeddings_v_read_readvariableopWsavev2_adam_token_and_position_embedding_3_embedding_7_embeddings_v_read_readvariableopYsavev2_adam_transformer_block_7_multi_head_attention_7_query_kernel_v_read_readvariableopWsavev2_adam_transformer_block_7_multi_head_attention_7_query_bias_v_read_readvariableopWsavev2_adam_transformer_block_7_multi_head_attention_7_key_kernel_v_read_readvariableopUsavev2_adam_transformer_block_7_multi_head_attention_7_key_bias_v_read_readvariableopYsavev2_adam_transformer_block_7_multi_head_attention_7_value_kernel_v_read_readvariableopWsavev2_adam_transformer_block_7_multi_head_attention_7_value_bias_v_read_readvariableopdsavev2_adam_transformer_block_7_multi_head_attention_7_attention_output_kernel_v_read_readvariableopbsavev2_adam_transformer_block_7_multi_head_attention_7_attention_output_bias_v_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableop1savev2_adam_dense_24_kernel_v_read_readvariableop/savev2_adam_dense_24_bias_v_read_readvariableopRsavev2_adam_transformer_block_7_layer_normalization_14_gamma_v_read_readvariableopQsavev2_adam_transformer_block_7_layer_normalization_14_beta_v_read_readvariableopRsavev2_adam_transformer_block_7_layer_normalization_15_gamma_v_read_readvariableopQsavev2_adam_transformer_block_7_layer_normalization_15_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *z
dtypesp
n2l	2
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

identity_1Identity_1:output:0*
_input_shapesþ
û: :  : :	  : : : : : : : : : :	È@:@:@@:@:@:: : : : : : :
  :  : :  : :  : :  : : @:@:@ : : : : : : : :  : :	  : : : : : :	È@:@:@@:@:@:: :
  :  : :  : :  : :  : : @:@:@ : : : : : :  : :	  : : : : : :	È@:@:@@:@:@:: :
  :  : :  : :  : :  : : @:@:@ : : : : : : 2(
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
:	È@: 
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
: :

_output_shapes
: :$ 

_output_shapes

: :&"
 
_output_shapes
:
  :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$ 

_output_shapes

: :( $
"
_output_shapes
:  : !

_output_shapes
: :$" 

_output_shapes

: @: #

_output_shapes
:@:$$ 

_output_shapes

:@ : %
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
: : )

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :(,$
"
_output_shapes
:  : -

_output_shapes
: :(.$
"
_output_shapes
:	  : /
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
: : 3

_output_shapes
: :%4!

_output_shapes
:	È@: 5

_output_shapes
:@:$6 

_output_shapes

:@@: 7

_output_shapes
:@:$8 

_output_shapes

:@: 9

_output_shapes
::$: 

_output_shapes

: :&;"
 
_output_shapes
:
  :(<$
"
_output_shapes
:  :$= 

_output_shapes

: :(>$
"
_output_shapes
:  :$? 

_output_shapes

: :(@$
"
_output_shapes
:  :$A 

_output_shapes

: :(B$
"
_output_shapes
:  : C

_output_shapes
: :$D 

_output_shapes

: @: E

_output_shapes
:@:$F 

_output_shapes

:@ : G
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
: : K

_output_shapes
: :(L$
"
_output_shapes
:  : M

_output_shapes
: :(N$
"
_output_shapes
:	  : O

_output_shapes
: : P

_output_shapes
: : Q

_output_shapes
: : R

_output_shapes
: : S

_output_shapes
: :%T!

_output_shapes
:	È@: U

_output_shapes
:@:$V 

_output_shapes

:@@: W

_output_shapes
:@:$X 

_output_shapes

:@: Y

_output_shapes
::$Z 

_output_shapes

: :&["
 
_output_shapes
:
  :(\$
"
_output_shapes
:  :$] 

_output_shapes

: :(^$
"
_output_shapes
:  :$_ 

_output_shapes

: :(`$
"
_output_shapes
:  :$a 

_output_shapes

: :(b$
"
_output_shapes
:  : c

_output_shapes
: :$d 

_output_shapes

: @: e

_output_shapes
:@:$f 

_output_shapes

:@ : g

_output_shapes
: : h

_output_shapes
: : i

_output_shapes
: : j

_output_shapes
: : k

_output_shapes
: :l

_output_shapes
: "±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ê
serving_defaultÖ
=
input_72
serving_default_input_7:0ÿÿÿÿÿÿÿÿÿ 
;
input_80
serving_default_input_8:0ÿÿÿÿÿÿÿÿÿ<
dense_270
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:
H
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
Ò__call__
Ó_default_save_signature
+Ô&call_and_return_all_conditional_losses"ÉB
_tf_keras_network­B{"class_name": "Functional", "name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}, "name": "token_and_position_embedding_3", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_6", "inbound_nodes": [[["token_and_position_embedding_3", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_9", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_9", "inbound_nodes": [[["conv1d_6", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_7", "inbound_nodes": [[["average_pooling1d_9", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_10", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_10", "inbound_nodes": [[["conv1d_7", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_11", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_11", "inbound_nodes": [[["token_and_position_embedding_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["average_pooling1d_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["average_pooling1d_11", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}], ["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block_7", "inbound_nodes": [[["add_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["transformer_block_7", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_3", "inbound_nodes": [[["flatten_3", 0, 0, {}], ["input_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_25", "inbound_nodes": [[["concatenate_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_22", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_22", "inbound_nodes": [[["dense_25", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_26", "inbound_nodes": [[["dropout_22", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_23", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_23", "inbound_nodes": [[["dense_26", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_27", "inbound_nodes": [[["dropout_23", 0, 0, {}]]]}], "input_layers": [["input_7", 0, 0], ["input_8", 0, 0]], "output_layers": [["dense_27", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 20000]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 20000]}, {"class_name": "TensorShape", "items": [null, 8]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ñ"î
_tf_keras_input_layerÎ{"class_name": "InputLayer", "name": "input_7", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20000]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}}
ç
	token_emb
pos_emb
	variables
trainable_variables
regularization_losses
	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses"º
_tf_keras_layer {"class_name": "TokenAndPositionEmbedding", "name": "token_and_position_embedding_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
é	

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses"Â
_tf_keras_layer¨{"class_name": "Conv1D", "name": "conv1d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20000, 32]}}

&	variables
'trainable_variables
(regularization_losses
)	keras_api
Ù__call__
+Ú&call_and_return_all_conditional_losses"ø
_tf_keras_layerÞ{"class_name": "AveragePooling1D", "name": "average_pooling1d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_9", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ç	

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
Û__call__
+Ü&call_and_return_all_conditional_losses"À
_tf_keras_layer¦{"class_name": "Conv1D", "name": "conv1d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 666, 32]}}

0	variables
1trainable_variables
2regularization_losses
3	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "AveragePooling1D", "name": "average_pooling1d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_10", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}

4	variables
5trainable_variables
6regularization_losses
7	keras_api
ß__call__
+à&call_and_return_all_conditional_losses"ü
_tf_keras_layerâ{"class_name": "AveragePooling1D", "name": "average_pooling1d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_11", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
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
á__call__
+â&call_and_return_all_conditional_losses"â
_tf_keras_layerÈ{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
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
ã__call__
+ä&call_and_return_all_conditional_losses"â
_tf_keras_layerÈ{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
³
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
å__call__
+æ&call_and_return_all_conditional_losses"¢
_tf_keras_layer{"class_name": "Add", "name": "add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 66, 32]}, {"class_name": "TensorShape", "items": [null, 66, 32]}]}
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
ç__call__
+è&call_and_return_all_conditional_losses"¥
_tf_keras_layer{"class_name": "TransformerBlock", "name": "transformer_block_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
è
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
é__call__
+ê&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
é"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_8", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}}
Ð
\	variables
]trainable_variables
^regularization_losses
_	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses"¿
_tf_keras_layer¥{"class_name": "Concatenate", "name": "concatenate_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 2112]}, {"class_name": "TensorShape", "items": [null, 8]}]}
ø

`kernel
abias
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
í__call__
+î&call_and_return_all_conditional_losses"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "dense_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2120]}}
é
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
ï__call__
+ð&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_22", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
ô

jkernel
kbias
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
ñ__call__
+ò&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
é
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_23", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
õ

tkernel
ubias
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
õ

zbeta_1

{beta_2
	|decay
}learning_rate
~iter m!m*m+m9m:mBmCm`mamjmkmtmumm 	m¡	m¢	m£	m¤	m¥	m¦	m§	m¨	m©	mª	m«	m¬	m­	m®	m¯	m°	m± v²!v³*v´+vµ9v¶:v·Bv¸Cv¹`vºav»jv¼kv½tv¾uv¿vÀ	vÁ	vÂ	vÃ	vÄ	vÅ	vÆ	vÇ	vÈ	vÉ	vÊ	vË	vÌ	vÍ	vÎ	vÏ	vÐ	vÑ"
	optimizer
Ç
0
1
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
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
`30
a31
j32
k33
t34
u35"
trackable_list_wrapper
§
0
1
 2
!3
*4
+5
96
:7
B8
C9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
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
layer_metrics
metrics
 layer_regularization_losses
non_trainable_variables
	variables
layers
trainable_variables
regularization_losses
Ò__call__
Ó_default_save_signature
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
-
÷serving_default"
signature_map
´

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
ø__call__
+ù&call_and_return_all_conditional_losses"
_tf_keras_layerõ{"class_name": "Embedding", "name": "embedding_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 5, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20000]}}
²

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
ú__call__
+û&call_and_return_all_conditional_losses"
_tf_keras_layerò{"class_name": "Embedding", "name": "embedding_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 20000, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
/
0
1"
trackable_list_wrapper
/
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
metrics
  layer_regularization_losses
	variables
trainable_variables
¡layers
¢non_trainable_variables
regularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
%:#  2conv1d_6/kernel
: 2conv1d_6/bias
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
£layer_metrics
¤metrics
 ¥layer_regularization_losses
"	variables
#trainable_variables
¦layers
§non_trainable_variables
$regularization_losses
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
¨layer_metrics
©metrics
 ªlayer_regularization_losses
&	variables
'trainable_variables
«layers
¬non_trainable_variables
(regularization_losses
Ù__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
%:#	  2conv1d_7/kernel
: 2conv1d_7/bias
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
­layer_metrics
®metrics
 ¯layer_regularization_losses
,	variables
-trainable_variables
°layers
±non_trainable_variables
.regularization_losses
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
²layer_metrics
³metrics
 ´layer_regularization_losses
0	variables
1trainable_variables
µlayers
¶non_trainable_variables
2regularization_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
·layer_metrics
¸metrics
 ¹layer_regularization_losses
4	variables
5trainable_variables
ºlayers
»non_trainable_variables
6regularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_6/gamma
(:& 2batch_normalization_6/beta
1:/  (2!batch_normalization_6/moving_mean
5:3  (2%batch_normalization_6/moving_variance
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
¼layer_metrics
½metrics
 ¾layer_regularization_losses
=	variables
>trainable_variables
¿layers
Ànon_trainable_variables
?regularization_losses
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_7/gamma
(:& 2batch_normalization_7/beta
1:/  (2!batch_normalization_7/moving_mean
5:3  (2%batch_normalization_7/moving_variance
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
Álayer_metrics
Âmetrics
 Ãlayer_regularization_losses
F	variables
Gtrainable_variables
Älayers
Ånon_trainable_variables
Hregularization_losses
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
Ælayer_metrics
Çmetrics
 Èlayer_regularization_losses
J	variables
Ktrainable_variables
Élayers
Ênon_trainable_variables
Lregularization_losses
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
_generic_user_object

Ë_query_dense
Ì
_key_dense
Í_value_dense
Î_softmax
Ï_dropout_layer
Ð_output_dense
Ñ	variables
Òtrainable_variables
Óregularization_losses
Ô	keras_api
ü__call__
+ý&call_and_return_all_conditional_losses"
_tf_keras_layerê{"class_name": "MultiHeadAttention", "name": "multi_head_attention_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multi_head_attention_7", "trainable": true, "dtype": "float32", "num_heads": 1, "key_dim": 32, "value_dim": 32, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}
¯
Õlayer_with_weights-0
Õlayer-0
Ölayer_with_weights-1
Ölayer-1
×	variables
Øtrainable_variables
Ùregularization_losses
Ú	keras_api
þ__call__
+ÿ&call_and_return_all_conditional_losses"È
_tf_keras_sequential©{"class_name": "Sequential", "name": "sequential_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 66, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_23_input"}}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 66, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_23_input"}}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
ì
	Ûaxis

gamma
	beta
Ü	variables
Ýtrainable_variables
Þregularization_losses
ß	keras_api
__call__
+&call_and_return_all_conditional_losses"µ
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_14", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
ì
	àaxis

gamma
	beta
á	variables
âtrainable_variables
ãregularization_losses
ä	keras_api
__call__
+&call_and_return_all_conditional_losses"µ
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_15", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
í
å	variables
ætrainable_variables
çregularization_losses
è	keras_api
__call__
+&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_20", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
í
é	variables
êtrainable_variables
ëregularization_losses
ì	keras_api
__call__
+&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_21", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
¦
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
¦
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ílayer_metrics
îmetrics
 ïlayer_regularization_losses
T	variables
Utrainable_variables
ðlayers
ñnon_trainable_variables
Vregularization_losses
ç__call__
+è&call_and_return_all_conditional_losses
'è"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
òlayer_metrics
ómetrics
 ôlayer_regularization_losses
X	variables
Ytrainable_variables
õlayers
önon_trainable_variables
Zregularization_losses
é__call__
+ê&call_and_return_all_conditional_losses
'ê"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
÷layer_metrics
ømetrics
 ùlayer_regularization_losses
\	variables
]trainable_variables
úlayers
ûnon_trainable_variables
^regularization_losses
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses"
_generic_user_object
": 	È@2dense_25/kernel
:@2dense_25/bias
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
ülayer_metrics
ýmetrics
 þlayer_regularization_losses
b	variables
ctrainable_variables
ÿlayers
non_trainable_variables
dregularization_losses
í__call__
+î&call_and_return_all_conditional_losses
'î"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
metrics
 layer_regularization_losses
f	variables
gtrainable_variables
layers
non_trainable_variables
hregularization_losses
ï__call__
+ð&call_and_return_all_conditional_losses
'ð"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_26/kernel
:@2dense_26/bias
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
layer_metrics
metrics
 layer_regularization_losses
l	variables
mtrainable_variables
layers
non_trainable_variables
nregularization_losses
ñ__call__
+ò&call_and_return_all_conditional_losses
'ò"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
metrics
 layer_regularization_losses
p	variables
qtrainable_variables
layers
non_trainable_variables
rregularization_losses
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_27/kernel
:2dense_27/bias
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
layer_metrics
metrics
 layer_regularization_losses
v	variables
wtrainable_variables
layers
non_trainable_variables
xregularization_losses
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
G:E 25token_and_position_embedding_3/embedding_6/embeddings
I:G
  25token_and_position_embedding_3/embedding_7/embeddings
M:K  27transformer_block_7/multi_head_attention_7/query/kernel
G:E 25transformer_block_7/multi_head_attention_7/query/bias
K:I  25transformer_block_7/multi_head_attention_7/key/kernel
E:C 23transformer_block_7/multi_head_attention_7/key/bias
M:K  27transformer_block_7/multi_head_attention_7/value/kernel
G:E 25transformer_block_7/multi_head_attention_7/value/bias
X:V  2Btransformer_block_7/multi_head_attention_7/attention_output/kernel
N:L 2@transformer_block_7/multi_head_attention_7/attention_output/bias
!: @2dense_23/kernel
:@2dense_23/bias
!:@ 2dense_24/kernel
: 2dense_24/bias
>:< 20transformer_block_7/layer_normalization_14/gamma
=:; 2/transformer_block_7/layer_normalization_14/beta
>:< 20transformer_block_7/layer_normalization_15/gamma
=:; 2/transformer_block_7/layer_normalization_15/beta
 "
trackable_dict_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
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
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layer_metrics
metrics
 layer_regularization_losses
	variables
trainable_variables
layers
non_trainable_variables
regularization_losses
ø__call__
+ù&call_and_return_all_conditional_losses
'ù"call_and_return_conditional_losses"
_generic_user_object
(
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layer_metrics
metrics
 layer_regularization_losses
	variables
trainable_variables
layers
non_trainable_variables
regularization_losses
ú__call__
+û&call_and_return_all_conditional_losses
'û"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
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
;0
<1"
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
D0
E1"
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
 partial_output_shape
¡full_output_shape
kernel
	bias
¢	variables
£trainable_variables
¤regularization_losses
¥	keras_api
__call__
+&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "EinsumDense", "name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
Ç
¦partial_output_shape
§full_output_shape
kernel
	bias
¨	variables
©trainable_variables
ªregularization_losses
«	keras_api
__call__
+&call_and_return_all_conditional_losses"ç
_tf_keras_layerÍ{"class_name": "EinsumDense", "name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
Ë
¬partial_output_shape
­full_output_shape
kernel
	bias
®	variables
¯trainable_variables
°regularization_losses
±	keras_api
__call__
+&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "EinsumDense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
ë
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
__call__
+&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}}
ç
¶	variables
·trainable_variables
¸regularization_losses
¹	keras_api
__call__
+&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
à
ºpartial_output_shape
»full_output_shape
kernel
	bias
¼	variables
½trainable_variables
¾regularization_losses
¿	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layeræ{"class_name": "EinsumDense", "name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 32], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 1, 32]}}
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Àlayer_metrics
Ámetrics
 Âlayer_regularization_losses
Ñ	variables
Òtrainable_variables
Ãlayers
Änon_trainable_variables
Óregularization_losses
ü__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
þ
kernel
	bias
Å	variables
Ætrainable_variables
Çregularization_losses
È	keras_api
__call__
+&call_and_return_all_conditional_losses"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "dense_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}

kernel
	bias
É	variables
Êtrainable_variables
Ëregularization_losses
Ì	keras_api
__call__
+&call_and_return_all_conditional_losses"Ó
_tf_keras_layer¹{"class_name": "Dense", "name": "dense_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 64]}}
@
0
1
2
3"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ílayer_metrics
Îmetrics
 Ïlayer_regularization_losses
Ðnon_trainable_variables
×	variables
Ñlayers
Øtrainable_variables
Ùregularization_losses
þ__call__
+ÿ&call_and_return_all_conditional_losses
'ÿ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Òlayer_metrics
Ómetrics
 Ôlayer_regularization_losses
Ü	variables
Ýtrainable_variables
Õlayers
Önon_trainable_variables
Þregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
×layer_metrics
Ømetrics
 Ùlayer_regularization_losses
á	variables
âtrainable_variables
Úlayers
Ûnon_trainable_variables
ãregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ülayer_metrics
Ýmetrics
 Þlayer_regularization_losses
å	variables
ætrainable_variables
ßlayers
ànon_trainable_variables
çregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
álayer_metrics
âmetrics
 ãlayer_regularization_losses
é	variables
êtrainable_variables
älayers
ånon_trainable_variables
ëregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
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

ætotal

çcount
è	variables
é	keras_api"
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
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
êlayer_metrics
ëmetrics
 ìlayer_regularization_losses
¢	variables
£trainable_variables
ílayers
înon_trainable_variables
¤regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ïlayer_metrics
ðmetrics
 ñlayer_regularization_losses
¨	variables
©trainable_variables
òlayers
ónon_trainable_variables
ªregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ôlayer_metrics
õmetrics
 ölayer_regularization_losses
®	variables
¯trainable_variables
÷layers
ønon_trainable_variables
°regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ùlayer_metrics
úmetrics
 ûlayer_regularization_losses
²	variables
³trainable_variables
ülayers
ýnon_trainable_variables
´regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
þlayer_metrics
ÿmetrics
 layer_regularization_losses
¶	variables
·trainable_variables
layers
non_trainable_variables
¸regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layer_metrics
metrics
 layer_regularization_losses
¼	variables
½trainable_variables
layers
non_trainable_variables
¾regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
P
Ë0
Ì1
Í2
Î3
Ï4
Ð5"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layer_metrics
metrics
 layer_regularization_losses
Å	variables
Ætrainable_variables
layers
non_trainable_variables
Çregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layer_metrics
metrics
 layer_regularization_losses
É	variables
Êtrainable_variables
layers
non_trainable_variables
Ëregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Õ0
Ö1"
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
æ0
ç1"
trackable_list_wrapper
.
è	variables"
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
*:(  2Adam/conv1d_6/kernel/m
 : 2Adam/conv1d_6/bias/m
*:(	  2Adam/conv1d_7/kernel/m
 : 2Adam/conv1d_7/bias/m
.:, 2"Adam/batch_normalization_6/gamma/m
-:+ 2!Adam/batch_normalization_6/beta/m
.:, 2"Adam/batch_normalization_7/gamma/m
-:+ 2!Adam/batch_normalization_7/beta/m
':%	È@2Adam/dense_25/kernel/m
 :@2Adam/dense_25/bias/m
&:$@@2Adam/dense_26/kernel/m
 :@2Adam/dense_26/bias/m
&:$@2Adam/dense_27/kernel/m
 :2Adam/dense_27/bias/m
L:J 2<Adam/token_and_position_embedding_3/embedding_6/embeddings/m
N:L
  2<Adam/token_and_position_embedding_3/embedding_7/embeddings/m
R:P  2>Adam/transformer_block_7/multi_head_attention_7/query/kernel/m
L:J 2<Adam/transformer_block_7/multi_head_attention_7/query/bias/m
P:N  2<Adam/transformer_block_7/multi_head_attention_7/key/kernel/m
J:H 2:Adam/transformer_block_7/multi_head_attention_7/key/bias/m
R:P  2>Adam/transformer_block_7/multi_head_attention_7/value/kernel/m
L:J 2<Adam/transformer_block_7/multi_head_attention_7/value/bias/m
]:[  2IAdam/transformer_block_7/multi_head_attention_7/attention_output/kernel/m
S:Q 2GAdam/transformer_block_7/multi_head_attention_7/attention_output/bias/m
&:$ @2Adam/dense_23/kernel/m
 :@2Adam/dense_23/bias/m
&:$@ 2Adam/dense_24/kernel/m
 : 2Adam/dense_24/bias/m
C:A 27Adam/transformer_block_7/layer_normalization_14/gamma/m
B:@ 26Adam/transformer_block_7/layer_normalization_14/beta/m
C:A 27Adam/transformer_block_7/layer_normalization_15/gamma/m
B:@ 26Adam/transformer_block_7/layer_normalization_15/beta/m
*:(  2Adam/conv1d_6/kernel/v
 : 2Adam/conv1d_6/bias/v
*:(	  2Adam/conv1d_7/kernel/v
 : 2Adam/conv1d_7/bias/v
.:, 2"Adam/batch_normalization_6/gamma/v
-:+ 2!Adam/batch_normalization_6/beta/v
.:, 2"Adam/batch_normalization_7/gamma/v
-:+ 2!Adam/batch_normalization_7/beta/v
':%	È@2Adam/dense_25/kernel/v
 :@2Adam/dense_25/bias/v
&:$@@2Adam/dense_26/kernel/v
 :@2Adam/dense_26/bias/v
&:$@2Adam/dense_27/kernel/v
 :2Adam/dense_27/bias/v
L:J 2<Adam/token_and_position_embedding_3/embedding_6/embeddings/v
N:L
  2<Adam/token_and_position_embedding_3/embedding_7/embeddings/v
R:P  2>Adam/transformer_block_7/multi_head_attention_7/query/kernel/v
L:J 2<Adam/transformer_block_7/multi_head_attention_7/query/bias/v
P:N  2<Adam/transformer_block_7/multi_head_attention_7/key/kernel/v
J:H 2:Adam/transformer_block_7/multi_head_attention_7/key/bias/v
R:P  2>Adam/transformer_block_7/multi_head_attention_7/value/kernel/v
L:J 2<Adam/transformer_block_7/multi_head_attention_7/value/bias/v
]:[  2IAdam/transformer_block_7/multi_head_attention_7/attention_output/kernel/v
S:Q 2GAdam/transformer_block_7/multi_head_attention_7/attention_output/bias/v
&:$ @2Adam/dense_23/kernel/v
 :@2Adam/dense_23/bias/v
&:$@ 2Adam/dense_24/kernel/v
 : 2Adam/dense_24/bias/v
C:A 27Adam/transformer_block_7/layer_normalization_14/gamma/v
B:@ 26Adam/transformer_block_7/layer_normalization_14/beta/v
C:A 27Adam/transformer_block_7/layer_normalization_15/gamma/v
B:@ 26Adam/transformer_block_7/layer_normalization_15/beta/v
î2ë
(__inference_model_3_layer_call_fn_196125
(__inference_model_3_layer_call_fn_196844
(__inference_model_3_layer_call_fn_195953
(__inference_model_3_layer_call_fn_196922À
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
!__inference__wrapped_model_194330à
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
# 
input_7ÿÿÿÿÿÿÿÿÿ 
!
input_8ÿÿÿÿÿÿÿÿÿ
Ú2×
C__inference_model_3_layer_call_and_return_conditional_losses_196766
C__inference_model_3_layer_call_and_return_conditional_losses_195686
C__inference_model_3_layer_call_and_return_conditional_losses_196523
C__inference_model_3_layer_call_and_return_conditional_losses_195780À
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
ä2á
?__inference_token_and_position_embedding_3_layer_call_fn_196955
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
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_196946
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
)__inference_conv1d_6_layer_call_fn_196980¢
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
D__inference_conv1d_6_layer_call_and_return_conditional_losses_196971¢
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
4__inference_average_pooling1d_9_layer_call_fn_194345Ó
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
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_194339Ó
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
)__inference_conv1d_7_layer_call_fn_197005¢
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
D__inference_conv1d_7_layer_call_and_return_conditional_losses_196996¢
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
2
5__inference_average_pooling1d_10_layer_call_fn_194360Ó
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
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_194354Ó
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
5__inference_average_pooling1d_11_layer_call_fn_194375Ó
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
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_194369Ó
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
6__inference_batch_normalization_6_layer_call_fn_197087
6__inference_batch_normalization_6_layer_call_fn_197169
6__inference_batch_normalization_6_layer_call_fn_197074
6__inference_batch_normalization_6_layer_call_fn_197156´
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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_197061
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_197143
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_197041
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_197123´
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
6__inference_batch_normalization_7_layer_call_fn_197238
6__inference_batch_normalization_7_layer_call_fn_197320
6__inference_batch_normalization_7_layer_call_fn_197251
6__inference_batch_normalization_7_layer_call_fn_197333´
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_197205
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_197307
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_197225
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_197287´
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
&__inference_add_3_layer_call_fn_197345¢
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
A__inference_add_3_layer_call_and_return_conditional_losses_197339¢
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
4__inference_transformer_block_7_layer_call_fn_197694
4__inference_transformer_block_7_layer_call_fn_197657°
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
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_197620
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_197493°
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
*__inference_flatten_3_layer_call_fn_197705¢
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
E__inference_flatten_3_layer_call_and_return_conditional_losses_197700¢
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
.__inference_concatenate_3_layer_call_fn_197718¢
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
I__inference_concatenate_3_layer_call_and_return_conditional_losses_197712¢
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
)__inference_dense_25_layer_call_fn_197738¢
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
D__inference_dense_25_layer_call_and_return_conditional_losses_197729¢
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
+__inference_dropout_22_layer_call_fn_197760
+__inference_dropout_22_layer_call_fn_197765´
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
F__inference_dropout_22_layer_call_and_return_conditional_losses_197755
F__inference_dropout_22_layer_call_and_return_conditional_losses_197750´
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
)__inference_dense_26_layer_call_fn_197785¢
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
D__inference_dense_26_layer_call_and_return_conditional_losses_197776¢
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
+__inference_dropout_23_layer_call_fn_197812
+__inference_dropout_23_layer_call_fn_197807´
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
F__inference_dropout_23_layer_call_and_return_conditional_losses_197802
F__inference_dropout_23_layer_call_and_return_conditional_losses_197797´
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
)__inference_dense_27_layer_call_fn_197831¢
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
D__inference_dense_27_layer_call_and_return_conditional_losses_197822¢
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
$__inference_signature_wrapper_196213input_7input_8"
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
-__inference_sequential_7_layer_call_fn_194795
-__inference_sequential_7_layer_call_fn_197971
-__inference_sequential_7_layer_call_fn_194822
-__inference_sequential_7_layer_call_fn_197958À
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
H__inference_sequential_7_layer_call_and_return_conditional_losses_197945
H__inference_sequential_7_layer_call_and_return_conditional_losses_194753
H__inference_sequential_7_layer_call_and_return_conditional_losses_197888
H__inference_sequential_7_layer_call_and_return_conditional_losses_194767À
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
)__inference_dense_23_layer_call_fn_198011¢
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
D__inference_dense_23_layer_call_and_return_conditional_losses_198002¢
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
)__inference_dense_24_layer_call_fn_198050¢
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
D__inference_dense_24_layer_call_and_return_conditional_losses_198041¢
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
 î
!__inference__wrapped_model_194330È5 !*+<9;:EBDC`ajktuZ¢W
P¢M
KH
# 
input_7ÿÿÿÿÿÿÿÿÿ 
!
input_8ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_27"
dense_27ÿÿÿÿÿÿÿÿÿÕ
A__inference_add_3_layer_call_and_return_conditional_losses_197339b¢_
X¢U
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿB 
&#
inputs/1ÿÿÿÿÿÿÿÿÿB 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB 
 ­
&__inference_add_3_layer_call_fn_197345b¢_
X¢U
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿB 
&#
inputs/1ÿÿÿÿÿÿÿÿÿB 
ª "ÿÿÿÿÿÿÿÿÿB Ù
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_194354E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 °
5__inference_average_pooling1d_10_layer_call_fn_194360wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÙ
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_194369E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 °
5__inference_average_pooling1d_11_layer_call_fn_194375wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØ
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_194339E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
4__inference_average_pooling1d_9_layer_call_fn_194345wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¿
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_197041j;<9:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB 
 ¿
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_197061j<9;:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB 
 Ñ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_197123|;<9:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ñ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_197143|<9;:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
6__inference_batch_normalization_6_layer_call_fn_197074];<9:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p
ª "ÿÿÿÿÿÿÿÿÿB 
6__inference_batch_normalization_6_layer_call_fn_197087]<9;:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p 
ª "ÿÿÿÿÿÿÿÿÿB ©
6__inference_batch_normalization_6_layer_call_fn_197156o;<9:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ©
6__inference_batch_normalization_6_layer_call_fn_197169o<9;:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ñ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_197205|DEBC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ñ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_197225|EBDC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ¿
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_197287jDEBC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB 
 ¿
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_197307jEBDC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB 
 ©
6__inference_batch_normalization_7_layer_call_fn_197238oDEBC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ©
6__inference_batch_normalization_7_layer_call_fn_197251oEBDC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
6__inference_batch_normalization_7_layer_call_fn_197320]DEBC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p
ª "ÿÿÿÿÿÿÿÿÿB 
6__inference_batch_normalization_7_layer_call_fn_197333]EBDC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p 
ª "ÿÿÿÿÿÿÿÿÿB Ó
I__inference_concatenate_3_layer_call_and_return_conditional_losses_197712[¢X
Q¢N
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿÀ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÈ
 ª
.__inference_concatenate_3_layer_call_fn_197718x[¢X
Q¢N
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿÀ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÈ°
D__inference_conv1d_6_layer_call_and_return_conditional_losses_196971h !5¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿ  
ª "+¢(
!
0ÿÿÿÿÿÿÿÿÿ  
 
)__inference_conv1d_6_layer_call_fn_196980[ !5¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿ  
ª "ÿÿÿÿÿÿÿÿÿ  ®
D__inference_conv1d_7_layer_call_and_return_conditional_losses_196996f*+4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_conv1d_7_layer_call_fn_197005Y*+4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ®
D__inference_dense_23_layer_call_and_return_conditional_losses_198002f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿB 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB@
 
)__inference_dense_23_layer_call_fn_198011Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿB 
ª "ÿÿÿÿÿÿÿÿÿB@®
D__inference_dense_24_layer_call_and_return_conditional_losses_198041f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿB@
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB 
 
)__inference_dense_24_layer_call_fn_198050Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿB@
ª "ÿÿÿÿÿÿÿÿÿB ¥
D__inference_dense_25_layer_call_and_return_conditional_losses_197729]`a0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÈ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 }
)__inference_dense_25_layer_call_fn_197738P`a0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÈ
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dense_26_layer_call_and_return_conditional_losses_197776\jk/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
)__inference_dense_26_layer_call_fn_197785Ojk/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dense_27_layer_call_and_return_conditional_losses_197822\tu/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_27_layer_call_fn_197831Otu/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dropout_22_layer_call_and_return_conditional_losses_197750\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¦
F__inference_dropout_22_layer_call_and_return_conditional_losses_197755\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ~
+__inference_dropout_22_layer_call_fn_197760O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@~
+__inference_dropout_22_layer_call_fn_197765O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@¦
F__inference_dropout_23_layer_call_and_return_conditional_losses_197797\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¦
F__inference_dropout_23_layer_call_and_return_conditional_losses_197802\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ~
+__inference_dropout_23_layer_call_fn_197807O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@~
+__inference_dropout_23_layer_call_fn_197812O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@¦
E__inference_flatten_3_layer_call_and_return_conditional_losses_197700]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿB 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÀ
 ~
*__inference_flatten_3_layer_call_fn_197705P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿB 
ª "ÿÿÿÿÿÿÿÿÿÀ
C__inference_model_3_layer_call_and_return_conditional_losses_195686Â5 !*+;<9:DEBC`ajktub¢_
X¢U
KH
# 
input_7ÿÿÿÿÿÿÿÿÿ 
!
input_8ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
C__inference_model_3_layer_call_and_return_conditional_losses_195780Â5 !*+<9;:EBDC`ajktub¢_
X¢U
KH
# 
input_7ÿÿÿÿÿÿÿÿÿ 
!
input_8ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
C__inference_model_3_layer_call_and_return_conditional_losses_196523Ä5 !*+;<9:DEBC`ajktud¢a
Z¢W
MJ
$!
inputs/0ÿÿÿÿÿÿÿÿÿ 
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
C__inference_model_3_layer_call_and_return_conditional_losses_196766Ä5 !*+<9;:EBDC`ajktud¢a
Z¢W
MJ
$!
inputs/0ÿÿÿÿÿÿÿÿÿ 
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 â
(__inference_model_3_layer_call_fn_195953µ5 !*+;<9:DEBC`ajktub¢_
X¢U
KH
# 
input_7ÿÿÿÿÿÿÿÿÿ 
!
input_8ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿâ
(__inference_model_3_layer_call_fn_196125µ5 !*+<9;:EBDC`ajktub¢_
X¢U
KH
# 
input_7ÿÿÿÿÿÿÿÿÿ 
!
input_8ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿä
(__inference_model_3_layer_call_fn_196844·5 !*+;<9:DEBC`ajktud¢a
Z¢W
MJ
$!
inputs/0ÿÿÿÿÿÿÿÿÿ 
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿä
(__inference_model_3_layer_call_fn_196922·5 !*+<9;:EBDC`ajktud¢a
Z¢W
MJ
$!
inputs/0ÿÿÿÿÿÿÿÿÿ 
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÆ
H__inference_sequential_7_layer_call_and_return_conditional_losses_194753zC¢@
9¢6
,)
dense_23_inputÿÿÿÿÿÿÿÿÿB 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB 
 Æ
H__inference_sequential_7_layer_call_and_return_conditional_losses_194767zC¢@
9¢6
,)
dense_23_inputÿÿÿÿÿÿÿÿÿB 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB 
 ¾
H__inference_sequential_7_layer_call_and_return_conditional_losses_197888r;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿB 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB 
 ¾
H__inference_sequential_7_layer_call_and_return_conditional_losses_197945r;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿB 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB 
 
-__inference_sequential_7_layer_call_fn_194795mC¢@
9¢6
,)
dense_23_inputÿÿÿÿÿÿÿÿÿB 
p

 
ª "ÿÿÿÿÿÿÿÿÿB 
-__inference_sequential_7_layer_call_fn_194822mC¢@
9¢6
,)
dense_23_inputÿÿÿÿÿÿÿÿÿB 
p 

 
ª "ÿÿÿÿÿÿÿÿÿB 
-__inference_sequential_7_layer_call_fn_197958e;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿB 
p

 
ª "ÿÿÿÿÿÿÿÿÿB 
-__inference_sequential_7_layer_call_fn_197971e;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿB 
p 

 
ª "ÿÿÿÿÿÿÿÿÿB 
$__inference_signature_wrapper_196213Ù5 !*+<9;:EBDC`ajktuk¢h
¢ 
aª^
.
input_7# 
input_7ÿÿÿÿÿÿÿÿÿ 
,
input_8!
input_8ÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_27"
dense_27ÿÿÿÿÿÿÿÿÿ¾
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_196946`,¢)
"¢

xÿÿÿÿÿÿÿÿÿ 
ª "+¢(
!
0ÿÿÿÿÿÿÿÿÿ  
 
?__inference_token_and_position_embedding_3_layer_call_fn_196955S,¢)
"¢

xÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ  Ú
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_197493 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB 
 Ú
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_197620 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB 
 ±
4__inference_transformer_block_7_layer_call_fn_197657y 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p
ª "ÿÿÿÿÿÿÿÿÿB ±
4__inference_transformer_block_7_layer_call_fn_197694y 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p 
ª "ÿÿÿÿÿÿÿÿÿB 