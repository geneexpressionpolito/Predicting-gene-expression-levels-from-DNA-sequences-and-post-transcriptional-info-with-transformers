¿0
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
 "serve*2.4.12v2.4.1-0-g85c8b2a817f8ñü)
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
5token_and_position_embedding_3/embedding_6/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75token_and_position_embedding_3/embedding_6/embeddings
¿
Itoken_and_position_embedding_3/embedding_6/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_3/embedding_6/embeddings*
_output_shapes

: *
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

SGD/conv1d_6/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *-
shared_nameSGD/conv1d_6/kernel/momentum

0SGD/conv1d_6/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_6/kernel/momentum*"
_output_shapes
:  *
dtype0

SGD/conv1d_6/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/conv1d_6/bias/momentum

.SGD/conv1d_6/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_6/bias/momentum*
_output_shapes
: *
dtype0

SGD/conv1d_7/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *-
shared_nameSGD/conv1d_7/kernel/momentum

0SGD/conv1d_7/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_7/kernel/momentum*"
_output_shapes
:	  *
dtype0

SGD/conv1d_7/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/conv1d_7/bias/momentum

.SGD/conv1d_7/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_7/bias/momentum*
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

SGD/dense_25/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	È@*-
shared_nameSGD/dense_25/kernel/momentum

0SGD/dense_25/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_25/kernel/momentum*
_output_shapes
:	È@*
dtype0

SGD/dense_25/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_25/bias/momentum

.SGD/dense_25/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_25/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_26/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*-
shared_nameSGD/dense_26/kernel/momentum

0SGD/dense_26/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_26/kernel/momentum*
_output_shapes

:@@*
dtype0

SGD/dense_26/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_26/bias/momentum

.SGD/dense_26/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_26/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_27/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*-
shared_nameSGD/dense_27/kernel/momentum

0SGD/dense_27/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_27/kernel/momentum*
_output_shapes

:@*
dtype0

SGD/dense_27/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/dense_27/bias/momentum

.SGD/dense_27/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_27/bias/momentum*
_output_shapes
:*
dtype0
à
BSGD/token_and_position_embedding_3/embedding_6/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/token_and_position_embedding_3/embedding_6/embeddings/momentum
Ù
VSGD/token_and_position_embedding_3/embedding_6/embeddings/momentum/Read/ReadVariableOpReadVariableOpBSGD/token_and_position_embedding_3/embedding_6/embeddings/momentum*
_output_shapes

: *
dtype0
â
BSGD/token_and_position_embedding_3/embedding_7/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *S
shared_nameDBSGD/token_and_position_embedding_3/embedding_7/embeddings/momentum
Û
VSGD/token_and_position_embedding_3/embedding_7/embeddings/momentum/Read/ReadVariableOpReadVariableOpBSGD/token_and_position_embedding_3/embedding_7/embeddings/momentum* 
_output_shapes
:
  *
dtype0
è
DSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentum
á
XSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentum*"
_output_shapes
:  *
dtype0
à
BSGD/transformer_block_7/multi_head_attention_7/query/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_7/multi_head_attention_7/query/bias/momentum
Ù
VSGD/transformer_block_7/multi_head_attention_7/query/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_7/multi_head_attention_7/query/bias/momentum*
_output_shapes

: *
dtype0
ä
BSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentum
Ý
VSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentum*"
_output_shapes
:  *
dtype0
Ü
@SGD/transformer_block_7/multi_head_attention_7/key/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *Q
shared_nameB@SGD/transformer_block_7/multi_head_attention_7/key/bias/momentum
Õ
TSGD/transformer_block_7/multi_head_attention_7/key/bias/momentum/Read/ReadVariableOpReadVariableOp@SGD/transformer_block_7/multi_head_attention_7/key/bias/momentum*
_output_shapes

: *
dtype0
è
DSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentum
á
XSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentum*"
_output_shapes
:  *
dtype0
à
BSGD/transformer_block_7/multi_head_attention_7/value/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_7/multi_head_attention_7/value/bias/momentum
Ù
VSGD/transformer_block_7/multi_head_attention_7/value/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_7/multi_head_attention_7/value/bias/momentum*
_output_shapes

: *
dtype0
þ
OSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *`
shared_nameQOSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentum
÷
cSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentum/Read/ReadVariableOpReadVariableOpOSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentum*"
_output_shapes
:  *
dtype0
ò
MSGD/transformer_block_7/multi_head_attention_7/attention_output/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *^
shared_nameOMSGD/transformer_block_7/multi_head_attention_7/attention_output/bias/momentum
ë
aSGD/transformer_block_7/multi_head_attention_7/attention_output/bias/momentum/Read/ReadVariableOpReadVariableOpMSGD/transformer_block_7/multi_head_attention_7/attention_output/bias/momentum*
_output_shapes
: *
dtype0

SGD/dense_23/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*-
shared_nameSGD/dense_23/kernel/momentum

0SGD/dense_23/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_23/kernel/momentum*
_output_shapes

: @*
dtype0

SGD/dense_23/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_23/bias/momentum

.SGD/dense_23/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_23/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_24/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *-
shared_nameSGD/dense_24/kernel/momentum

0SGD/dense_24/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_24/kernel/momentum*
_output_shapes

:@ *
dtype0

SGD/dense_24/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/dense_24/bias/momentum

.SGD/dense_24/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_24/bias/momentum*
_output_shapes
: *
dtype0
Ò
=SGD/transformer_block_7/layer_normalization_14/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=SGD/transformer_block_7/layer_normalization_14/gamma/momentum
Ë
QSGD/transformer_block_7/layer_normalization_14/gamma/momentum/Read/ReadVariableOpReadVariableOp=SGD/transformer_block_7/layer_normalization_14/gamma/momentum*
_output_shapes
: *
dtype0
Ð
<SGD/transformer_block_7/layer_normalization_14/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_7/layer_normalization_14/beta/momentum
É
PSGD/transformer_block_7/layer_normalization_14/beta/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_7/layer_normalization_14/beta/momentum*
_output_shapes
: *
dtype0
Ò
=SGD/transformer_block_7/layer_normalization_15/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=SGD/transformer_block_7/layer_normalization_15/gamma/momentum
Ë
QSGD/transformer_block_7/layer_normalization_15/gamma/momentum/Read/ReadVariableOpReadVariableOp=SGD/transformer_block_7/layer_normalization_15/gamma/momentum*
_output_shapes
: *
dtype0
Ð
<SGD/transformer_block_7/layer_normalization_15/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_7/layer_normalization_15/beta/momentum
É
PSGD/transformer_block_7/layer_normalization_15/beta/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_7/layer_normalization_15/beta/momentum*
_output_shapes
: *
dtype0

NoOpNoOp
§¶
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*áµ
valueÖµBÒµ BÊµ
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
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
n
	token_emb
pos_emb
trainable_variables
regularization_losses
	variables
	keras_api
h

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
R
&trainable_variables
'regularization_losses
(	variables
)	keras_api
h

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
R
0trainable_variables
1regularization_losses
2	variables
3	keras_api
R
4trainable_variables
5regularization_losses
6	variables
7	keras_api

8axis
	9gamma
:beta
;moving_mean
<moving_variance
=trainable_variables
>regularization_losses
?	variables
@	keras_api

Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
R
Jtrainable_variables
Kregularization_losses
L	variables
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
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
R
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
 
R
\trainable_variables
]regularization_losses
^	variables
_	keras_api
h

`kernel
abias
btrainable_variables
cregularization_losses
d	variables
e	keras_api
R
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
h

jkernel
kbias
ltrainable_variables
mregularization_losses
n	variables
o	keras_api
R
ptrainable_variables
qregularization_losses
r	variables
s	keras_api
h

tkernel
ubias
vtrainable_variables
wregularization_losses
x	variables
y	keras_api
æ
	zdecay
{learning_rate
|momentum
}iter momentum!momentum*momentum+momentum9momentum:momentumBmomentumCmomentum`momentumamomentumjmomentumkmomentumtmomentumumomentum~momentummomentum momentum¡momentum¢momentum£momentum¤momentum¥momentum¦momentum§momentum¨momentum©momentumªmomentum«momentum¬momentum­momentum®momentum¯momentum°
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
²
trainable_variables
non_trainable_variables
metrics
layers
regularization_losses
layer_metrics
 layer_regularization_losses
	variables
 
f
~
embeddings
trainable_variables
regularization_losses
	variables
	keras_api
f

embeddings
trainable_variables
regularization_losses
	variables
	keras_api

~0
1
 

~0
1
²
metrics
non_trainable_variables
layers
trainable_variables
regularization_losses
 layer_metrics
 ¡layer_regularization_losses
	variables
[Y
VARIABLE_VALUEconv1d_6/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_6/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
²
¢metrics
£non_trainable_variables
¤layers
"trainable_variables
#regularization_losses
¥layer_metrics
 ¦layer_regularization_losses
$	variables
 
 
 
²
§metrics
¨non_trainable_variables
©layers
&trainable_variables
'regularization_losses
ªlayer_metrics
 «layer_regularization_losses
(	variables
[Y
VARIABLE_VALUEconv1d_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1
 

*0
+1
²
¬metrics
­non_trainable_variables
®layers
,trainable_variables
-regularization_losses
¯layer_metrics
 °layer_regularization_losses
.	variables
 
 
 
²
±metrics
²non_trainable_variables
³layers
0trainable_variables
1regularization_losses
´layer_metrics
 µlayer_regularization_losses
2	variables
 
 
 
²
¶metrics
·non_trainable_variables
¸layers
4trainable_variables
5regularization_losses
¹layer_metrics
 ºlayer_regularization_losses
6	variables
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
90
:1
 

90
:1
;2
<3
²
»metrics
¼non_trainable_variables
½layers
=trainable_variables
>regularization_losses
¾layer_metrics
 ¿layer_regularization_losses
?	variables
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
B0
C1
 

B0
C1
D2
E3
²
Àmetrics
Ánon_trainable_variables
Âlayers
Ftrainable_variables
Gregularization_losses
Ãlayer_metrics
 Älayer_regularization_losses
H	variables
 
 
 
²
Åmetrics
Ænon_trainable_variables
Çlayers
Jtrainable_variables
Kregularization_losses
Èlayer_metrics
 Élayer_regularization_losses
L	variables
Å
Ê_query_dense
Ë
_key_dense
Ì_value_dense
Í_softmax
Î_dropout_layer
Ï_output_dense
Ðtrainable_variables
Ñregularization_losses
Ò	variables
Ó	keras_api
¨
Ôlayer_with_weights-0
Ôlayer-0
Õlayer_with_weights-1
Õlayer-1
Ötrainable_variables
×regularization_losses
Ø	variables
Ù	keras_api
x
	Úaxis

gamma
	beta
Ûtrainable_variables
Üregularization_losses
Ý	variables
Þ	keras_api
x
	ßaxis

gamma
	beta
àtrainable_variables
áregularization_losses
â	variables
ã	keras_api
V
ätrainable_variables
åregularization_losses
æ	variables
ç	keras_api
V
ètrainable_variables
éregularization_losses
ê	variables
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
 
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
²
ìmetrics
ínon_trainable_variables
îlayers
Ttrainable_variables
Uregularization_losses
ïlayer_metrics
 ðlayer_regularization_losses
V	variables
 
 
 
²
ñmetrics
ònon_trainable_variables
ólayers
Xtrainable_variables
Yregularization_losses
ôlayer_metrics
 õlayer_regularization_losses
Z	variables
 
 
 
²
ömetrics
÷non_trainable_variables
ølayers
\trainable_variables
]regularization_losses
ùlayer_metrics
 úlayer_regularization_losses
^	variables
[Y
VARIABLE_VALUEdense_25/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_25/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

`0
a1
 

`0
a1
²
ûmetrics
ünon_trainable_variables
ýlayers
btrainable_variables
cregularization_losses
þlayer_metrics
 ÿlayer_regularization_losses
d	variables
 
 
 
²
metrics
non_trainable_variables
layers
ftrainable_variables
gregularization_losses
layer_metrics
 layer_regularization_losses
h	variables
[Y
VARIABLE_VALUEdense_26/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_26/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

j0
k1
 

j0
k1
²
metrics
non_trainable_variables
layers
ltrainable_variables
mregularization_losses
layer_metrics
 layer_regularization_losses
n	variables
 
 
 
²
metrics
non_trainable_variables
layers
ptrainable_variables
qregularization_losses
layer_metrics
 layer_regularization_losses
r	variables
[Y
VARIABLE_VALUEdense_27/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_27/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

t0
u1
 

t0
u1
²
metrics
non_trainable_variables
layers
vtrainable_variables
wregularization_losses
layer_metrics
 layer_regularization_losses
x	variables
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE5token_and_position_embedding_3/embedding_6/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE5token_and_position_embedding_3/embedding_7/embeddings0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE7transformer_block_7/multi_head_attention_7/query/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE5transformer_block_7/multi_head_attention_7/query/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE5transformer_block_7/multi_head_attention_7/key/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE3transformer_block_7/multi_head_attention_7/key/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE7transformer_block_7/multi_head_attention_7/value/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE5transformer_block_7/multi_head_attention_7/value/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEBtransformer_block_7/multi_head_attention_7/attention_output/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE@transformer_block_7/multi_head_attention_7/attention_output/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_23/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_23/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_24/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_24/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0transformer_block_7/layer_normalization_14/gamma1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE/transformer_block_7/layer_normalization_14/beta1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0transformer_block_7/layer_normalization_15/gamma1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE/transformer_block_7/layer_normalization_15/beta1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUE

;0
<1
D2
E3

0
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
 

~0
 

~0
µ
metrics
non_trainable_variables
layers
trainable_variables
regularization_losses
layer_metrics
 layer_regularization_losses
	variables

0
 

0
µ
metrics
non_trainable_variables
layers
trainable_variables
regularization_losses
layer_metrics
 layer_regularization_losses
	variables
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
¡trainable_variables
¢regularization_losses
£	variables
¤	keras_api
¡
¥partial_output_shape
¦full_output_shape
kernel
	bias
§trainable_variables
¨regularization_losses
©	variables
ª	keras_api
¡
«partial_output_shape
¬full_output_shape
kernel
	bias
­trainable_variables
®regularization_losses
¯	variables
°	keras_api
V
±trainable_variables
²regularization_losses
³	variables
´	keras_api
V
µtrainable_variables
¶regularization_losses
·	variables
¸	keras_api
¡
¹partial_output_shape
ºfull_output_shape
kernel
	bias
»trainable_variables
¼regularization_losses
½	variables
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
 
@
0
1
2
3
4
5
6
7
µ
¿metrics
Ànon_trainable_variables
Álayers
Ðtrainable_variables
Ñregularization_losses
Âlayer_metrics
 Ãlayer_regularization_losses
Ò	variables
n
kernel
	bias
Ätrainable_variables
Åregularization_losses
Æ	variables
Ç	keras_api
n
kernel
	bias
Ètrainable_variables
Éregularization_losses
Ê	variables
Ë	keras_api
 
0
1
2
3
 
 
0
1
2
3
µ
Ötrainable_variables
Ìnon_trainable_variables
Ímetrics
Îlayers
×regularization_losses
Ïlayer_metrics
 Ðlayer_regularization_losses
Ø	variables
 

0
1
 

0
1
µ
Ñmetrics
Ònon_trainable_variables
Ólayers
Ûtrainable_variables
Üregularization_losses
Ôlayer_metrics
 Õlayer_regularization_losses
Ý	variables
 

0
1
 

0
1
µ
Ömetrics
×non_trainable_variables
Ølayers
àtrainable_variables
áregularization_losses
Ùlayer_metrics
 Úlayer_regularization_losses
â	variables
 
 
 
µ
Ûmetrics
Ünon_trainable_variables
Ýlayers
ätrainable_variables
åregularization_losses
Þlayer_metrics
 ßlayer_regularization_losses
æ	variables
 
 
 
µ
àmetrics
ánon_trainable_variables
âlayers
ètrainable_variables
éregularization_losses
ãlayer_metrics
 älayer_regularization_losses
ê	variables
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
 

0
1
µ
émetrics
ênon_trainable_variables
ëlayers
¡trainable_variables
¢regularization_losses
ìlayer_metrics
 ílayer_regularization_losses
£	variables
 
 

0
1
 

0
1
µ
îmetrics
ïnon_trainable_variables
ðlayers
§trainable_variables
¨regularization_losses
ñlayer_metrics
 òlayer_regularization_losses
©	variables
 
 

0
1
 

0
1
µ
ómetrics
ônon_trainable_variables
õlayers
­trainable_variables
®regularization_losses
ölayer_metrics
 ÷layer_regularization_losses
¯	variables
 
 
 
µ
ømetrics
ùnon_trainable_variables
úlayers
±trainable_variables
²regularization_losses
ûlayer_metrics
 ülayer_regularization_losses
³	variables
 
 
 
µ
ýmetrics
þnon_trainable_variables
ÿlayers
µtrainable_variables
¶regularization_losses
layer_metrics
 layer_regularization_losses
·	variables
 
 

0
1
 

0
1
µ
metrics
non_trainable_variables
layers
»trainable_variables
¼regularization_losses
layer_metrics
 layer_regularization_losses
½	variables
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
 

0
1
µ
metrics
non_trainable_variables
layers
Ätrainable_variables
Åregularization_losses
layer_metrics
 layer_regularization_losses
Æ	variables

0
1
 

0
1
µ
metrics
non_trainable_variables
layers
Ètrainable_variables
Éregularization_losses
layer_metrics
 layer_regularization_losses
Ê	variables
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
VARIABLE_VALUESGD/conv1d_6/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv1d_6/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv1d_7/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv1d_7/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/batch_normalization_6/gamma/momentumXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'SGD/batch_normalization_6/beta/momentumWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/batch_normalization_7/gamma/momentumXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'SGD/batch_normalization_7/beta/momentumWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_25/kernel/momentumYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_25/bias/momentumWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_26/kernel/momentumYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_26/bias/momentumWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_27/kernel/momentumYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_27/bias/momentumWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¬©
VARIABLE_VALUEBSGD/token_and_position_embedding_3/embedding_6/embeddings/momentumStrainable_variables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¬©
VARIABLE_VALUEBSGD/token_and_position_embedding_3/embedding_7/embeddings/momentumStrainable_variables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¯¬
VARIABLE_VALUEDSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentumTtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
­ª
VARIABLE_VALUEBSGD/transformer_block_7/multi_head_attention_7/query/bias/momentumTtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
­ª
VARIABLE_VALUEBSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentumTtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
«¨
VARIABLE_VALUE@SGD/transformer_block_7/multi_head_attention_7/key/bias/momentumTtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¯¬
VARIABLE_VALUEDSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentumTtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
­ª
VARIABLE_VALUEBSGD/transformer_block_7/multi_head_attention_7/value/bias/momentumTtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
º·
VARIABLE_VALUEOSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentumTtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¸µ
VARIABLE_VALUEMSGD/transformer_block_7/multi_head_attention_7/attention_output/bias/momentumTtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_23/kernel/momentumTtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_23/bias/momentumTtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_24/kernel/momentumTtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_24/bias/momentumTtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¨¥
VARIABLE_VALUE=SGD/transformer_block_7/layer_normalization_14/gamma/momentumTtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
§¤
VARIABLE_VALUE<SGD/transformer_block_7/layer_normalization_14/beta/momentumTtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¨¥
VARIABLE_VALUE=SGD/transformer_block_7/layer_normalization_15/gamma/momentumTtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
§¤
VARIABLE_VALUE<SGD/transformer_block_7/layer_normalization_15/beta/momentumTtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
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
$__inference_signature_wrapper_411175
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ì$
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1d_6/kernel/Read/ReadVariableOp!conv1d_6/bias/Read/ReadVariableOp#conv1d_7/kernel/Read/ReadVariableOp!conv1d_7/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOp#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOp#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpItoken_and_position_embedding_3/embedding_6/embeddings/Read/ReadVariableOpItoken_and_position_embedding_3/embedding_7/embeddings/Read/ReadVariableOpKtransformer_block_7/multi_head_attention_7/query/kernel/Read/ReadVariableOpItransformer_block_7/multi_head_attention_7/query/bias/Read/ReadVariableOpItransformer_block_7/multi_head_attention_7/key/kernel/Read/ReadVariableOpGtransformer_block_7/multi_head_attention_7/key/bias/Read/ReadVariableOpKtransformer_block_7/multi_head_attention_7/value/kernel/Read/ReadVariableOpItransformer_block_7/multi_head_attention_7/value/bias/Read/ReadVariableOpVtransformer_block_7/multi_head_attention_7/attention_output/kernel/Read/ReadVariableOpTtransformer_block_7/multi_head_attention_7/attention_output/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOpDtransformer_block_7/layer_normalization_14/gamma/Read/ReadVariableOpCtransformer_block_7/layer_normalization_14/beta/Read/ReadVariableOpDtransformer_block_7/layer_normalization_15/gamma/Read/ReadVariableOpCtransformer_block_7/layer_normalization_15/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp0SGD/conv1d_6/kernel/momentum/Read/ReadVariableOp.SGD/conv1d_6/bias/momentum/Read/ReadVariableOp0SGD/conv1d_7/kernel/momentum/Read/ReadVariableOp.SGD/conv1d_7/bias/momentum/Read/ReadVariableOp<SGD/batch_normalization_6/gamma/momentum/Read/ReadVariableOp;SGD/batch_normalization_6/beta/momentum/Read/ReadVariableOp<SGD/batch_normalization_7/gamma/momentum/Read/ReadVariableOp;SGD/batch_normalization_7/beta/momentum/Read/ReadVariableOp0SGD/dense_25/kernel/momentum/Read/ReadVariableOp.SGD/dense_25/bias/momentum/Read/ReadVariableOp0SGD/dense_26/kernel/momentum/Read/ReadVariableOp.SGD/dense_26/bias/momentum/Read/ReadVariableOp0SGD/dense_27/kernel/momentum/Read/ReadVariableOp.SGD/dense_27/bias/momentum/Read/ReadVariableOpVSGD/token_and_position_embedding_3/embedding_6/embeddings/momentum/Read/ReadVariableOpVSGD/token_and_position_embedding_3/embedding_7/embeddings/momentum/Read/ReadVariableOpXSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentum/Read/ReadVariableOpVSGD/transformer_block_7/multi_head_attention_7/query/bias/momentum/Read/ReadVariableOpVSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentum/Read/ReadVariableOpTSGD/transformer_block_7/multi_head_attention_7/key/bias/momentum/Read/ReadVariableOpXSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentum/Read/ReadVariableOpVSGD/transformer_block_7/multi_head_attention_7/value/bias/momentum/Read/ReadVariableOpcSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentum/Read/ReadVariableOpaSGD/transformer_block_7/multi_head_attention_7/attention_output/bias/momentum/Read/ReadVariableOp0SGD/dense_23/kernel/momentum/Read/ReadVariableOp.SGD/dense_23/bias/momentum/Read/ReadVariableOp0SGD/dense_24/kernel/momentum/Read/ReadVariableOp.SGD/dense_24/bias/momentum/Read/ReadVariableOpQSGD/transformer_block_7/layer_normalization_14/gamma/momentum/Read/ReadVariableOpPSGD/transformer_block_7/layer_normalization_14/beta/momentum/Read/ReadVariableOpQSGD/transformer_block_7/layer_normalization_15/gamma/momentum/Read/ReadVariableOpPSGD/transformer_block_7/layer_normalization_15/beta/momentum/Read/ReadVariableOpConst*W
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
__inference__traced_save_413258
ÿ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_6/kernelconv1d_6/biasconv1d_7/kernelconv1d_7/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancebatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancedense_25/kerneldense_25/biasdense_26/kerneldense_26/biasdense_27/kerneldense_27/biasdecaylearning_ratemomentumSGD/iter5token_and_position_embedding_3/embedding_6/embeddings5token_and_position_embedding_3/embedding_7/embeddings7transformer_block_7/multi_head_attention_7/query/kernel5transformer_block_7/multi_head_attention_7/query/bias5transformer_block_7/multi_head_attention_7/key/kernel3transformer_block_7/multi_head_attention_7/key/bias7transformer_block_7/multi_head_attention_7/value/kernel5transformer_block_7/multi_head_attention_7/value/biasBtransformer_block_7/multi_head_attention_7/attention_output/kernel@transformer_block_7/multi_head_attention_7/attention_output/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/bias0transformer_block_7/layer_normalization_14/gamma/transformer_block_7/layer_normalization_14/beta0transformer_block_7/layer_normalization_15/gamma/transformer_block_7/layer_normalization_15/betatotalcountSGD/conv1d_6/kernel/momentumSGD/conv1d_6/bias/momentumSGD/conv1d_7/kernel/momentumSGD/conv1d_7/bias/momentum(SGD/batch_normalization_6/gamma/momentum'SGD/batch_normalization_6/beta/momentum(SGD/batch_normalization_7/gamma/momentum'SGD/batch_normalization_7/beta/momentumSGD/dense_25/kernel/momentumSGD/dense_25/bias/momentumSGD/dense_26/kernel/momentumSGD/dense_26/bias/momentumSGD/dense_27/kernel/momentumSGD/dense_27/bias/momentumBSGD/token_and_position_embedding_3/embedding_6/embeddings/momentumBSGD/token_and_position_embedding_3/embedding_7/embeddings/momentumDSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentumBSGD/transformer_block_7/multi_head_attention_7/query/bias/momentumBSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentum@SGD/transformer_block_7/multi_head_attention_7/key/bias/momentumDSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentumBSGD/transformer_block_7/multi_head_attention_7/value/bias/momentumOSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentumMSGD/transformer_block_7/multi_head_attention_7/attention_output/bias/momentumSGD/dense_23/kernel/momentumSGD/dense_23/bias/momentumSGD/dense_24/kernel/momentumSGD/dense_24/bias/momentum=SGD/transformer_block_7/layer_normalization_14/gamma/momentum<SGD/transformer_block_7/layer_normalization_14/beta/momentum=SGD/transformer_block_7/layer_normalization_15/gamma/momentum<SGD/transformer_block_7/layer_normalization_15/beta/momentum*V
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
"__inference__traced_restore_413490¿½&
ø
l
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_409333

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
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_412582

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


Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412187

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
è

Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_409953

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
F__inference_dropout_23_layer_call_and_return_conditional_losses_412764

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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_409575

inputs
assignmovingavg_409550
assignmovingavg_1_409556)
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
loc:@AssignMovingAvg/409550*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_409550*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/409550*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/409550*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_409550AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/409550*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/409556*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_409556*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/409556*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/409556*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_409556AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/409556*
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

e
F__inference_dropout_23_layer_call_and_return_conditional_losses_412759

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ýJ?2
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
 *
×#<2
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
÷
~
)__inference_conv1d_6_layer_call_fn_411942

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
D__inference_conv1d_6_layer_call_and_return_conditional_losses_4098472
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
ó0
È
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412085

inputs
assignmovingavg_412060
assignmovingavg_1_412066)
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
loc:@AssignMovingAvg/412060*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_412060*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/412060*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/412060*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_412060AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/412060*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/412066*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_412066*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/412066*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/412066*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_412066AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/412066*
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

÷
D__inference_conv1d_7_layer_call_and_return_conditional_losses_411958

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
Â
u
I__inference_concatenate_3_layer_call_and_return_conditional_losses_412674
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
ñ	
Ý
D__inference_dense_25_layer_call_and_return_conditional_losses_412691

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
É
d
F__inference_dropout_23_layer_call_and_return_conditional_losses_410610

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
Ü
¤
(__inference_model_3_layer_call_fn_411884
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
C__inference_model_3_layer_call_and_return_conditional_losses_4110142
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
íX

C__inference_model_3_layer_call_and_return_conditional_losses_410744
input_7
input_8)
%token_and_position_embedding_3_410654)
%token_and_position_embedding_3_410656
conv1d_6_410659
conv1d_6_410661
conv1d_7_410665
conv1d_7_410667 
batch_normalization_6_410672 
batch_normalization_6_410674 
batch_normalization_6_410676 
batch_normalization_6_410678 
batch_normalization_7_410681 
batch_normalization_7_410683 
batch_normalization_7_410685 
batch_normalization_7_410687
transformer_block_7_410691
transformer_block_7_410693
transformer_block_7_410695
transformer_block_7_410697
transformer_block_7_410699
transformer_block_7_410701
transformer_block_7_410703
transformer_block_7_410705
transformer_block_7_410707
transformer_block_7_410709
transformer_block_7_410711
transformer_block_7_410713
transformer_block_7_410715
transformer_block_7_410717
transformer_block_7_410719
transformer_block_7_410721
dense_25_410726
dense_25_410728
dense_26_410732
dense_26_410734
dense_27_410738
dense_27_410740
identity¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢ conv1d_6/StatefulPartitionedCall¢ conv1d_7/StatefulPartitionedCall¢ dense_25/StatefulPartitionedCall¢ dense_26/StatefulPartitionedCall¢ dense_27/StatefulPartitionedCall¢6token_and_position_embedding_3/StatefulPartitionedCall¢+transformer_block_7/StatefulPartitionedCall
6token_and_position_embedding_3/StatefulPartitionedCallStatefulPartitionedCallinput_7%token_and_position_embedding_3_410654%token_and_position_embedding_3_410656*
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
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_40981528
6token_and_position_embedding_3/StatefulPartitionedCallÖ
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0conv1d_6_410659conv1d_6_410661*
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
D__inference_conv1d_6_layer_call_and_return_conditional_losses_4098472"
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
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_4093032%
#average_pooling1d_9/PartitionedCallÂ
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_9/PartitionedCall:output:0conv1d_7_410665conv1d_7_410667*
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
D__inference_conv1d_7_layer_call_and_return_conditional_losses_4098802"
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
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_4093332&
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
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_4093182&
$average_pooling1d_10/PartitionedCallÃ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_10/PartitionedCall:output:0batch_normalization_6_410672batch_normalization_6_410674batch_normalization_6_410676batch_normalization_6_410678*
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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4099532/
-batch_normalization_6/StatefulPartitionedCallÃ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_11/PartitionedCall:output:0batch_normalization_7_410681batch_normalization_7_410683batch_normalization_7_410685batch_normalization_7_410687*
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4100442/
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
A__inference_add_3_layer_call_and_return_conditional_losses_4100862
add_3/PartitionedCall
+transformer_block_7/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0transformer_block_7_410691transformer_block_7_410693transformer_block_7_410695transformer_block_7_410697transformer_block_7_410699transformer_block_7_410701transformer_block_7_410703transformer_block_7_410705transformer_block_7_410707transformer_block_7_410709transformer_block_7_410711transformer_block_7_410713transformer_block_7_410715transformer_block_7_410717transformer_block_7_410719transformer_block_7_410721*
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
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_4103702-
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
E__inference_flatten_3_layer_call_and_return_conditional_losses_4104852
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
I__inference_concatenate_3_layer_call_and_return_conditional_losses_4105002
concatenate_3/PartitionedCall·
 dense_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_25_410726dense_25_410728*
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
D__inference_dense_25_layer_call_and_return_conditional_losses_4105202"
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
F__inference_dropout_22_layer_call_and_return_conditional_losses_4105532
dropout_22/PartitionedCall´
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_26_410732dense_26_410734*
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
D__inference_dense_26_layer_call_and_return_conditional_losses_4105772"
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
F__inference_dropout_23_layer_call_and_return_conditional_losses_4106102
dropout_23/PartitionedCall´
 dense_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_27_410738dense_27_410740*
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
D__inference_dense_27_layer_call_and_return_conditional_losses_4106332"
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
¹Þ
â
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_410370

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


Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_409608

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
¢
÷
D__inference_conv1d_6_layer_call_and_return_conditional_losses_409847

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
ö[
è
C__inference_model_3_layer_call_and_return_conditional_losses_410650
input_7
input_8)
%token_and_position_embedding_3_409826)
%token_and_position_embedding_3_409828
conv1d_6_409858
conv1d_6_409860
conv1d_7_409891
conv1d_7_409893 
batch_normalization_6_409980 
batch_normalization_6_409982 
batch_normalization_6_409984 
batch_normalization_6_409986 
batch_normalization_7_410071 
batch_normalization_7_410073 
batch_normalization_7_410075 
batch_normalization_7_410077
transformer_block_7_410446
transformer_block_7_410448
transformer_block_7_410450
transformer_block_7_410452
transformer_block_7_410454
transformer_block_7_410456
transformer_block_7_410458
transformer_block_7_410460
transformer_block_7_410462
transformer_block_7_410464
transformer_block_7_410466
transformer_block_7_410468
transformer_block_7_410470
transformer_block_7_410472
transformer_block_7_410474
transformer_block_7_410476
dense_25_410531
dense_25_410533
dense_26_410588
dense_26_410590
dense_27_410644
dense_27_410646
identity¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢ conv1d_6/StatefulPartitionedCall¢ conv1d_7/StatefulPartitionedCall¢ dense_25/StatefulPartitionedCall¢ dense_26/StatefulPartitionedCall¢ dense_27/StatefulPartitionedCall¢"dropout_22/StatefulPartitionedCall¢"dropout_23/StatefulPartitionedCall¢6token_and_position_embedding_3/StatefulPartitionedCall¢+transformer_block_7/StatefulPartitionedCall
6token_and_position_embedding_3/StatefulPartitionedCallStatefulPartitionedCallinput_7%token_and_position_embedding_3_409826%token_and_position_embedding_3_409828*
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
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_40981528
6token_and_position_embedding_3/StatefulPartitionedCallÖ
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0conv1d_6_409858conv1d_6_409860*
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
D__inference_conv1d_6_layer_call_and_return_conditional_losses_4098472"
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
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_4093032%
#average_pooling1d_9/PartitionedCallÂ
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_9/PartitionedCall:output:0conv1d_7_409891conv1d_7_409893*
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
D__inference_conv1d_7_layer_call_and_return_conditional_losses_4098802"
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
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_4093332&
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
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_4093182&
$average_pooling1d_10/PartitionedCallÁ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_10/PartitionedCall:output:0batch_normalization_6_409980batch_normalization_6_409982batch_normalization_6_409984batch_normalization_6_409986*
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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4099332/
-batch_normalization_6/StatefulPartitionedCallÁ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_11/PartitionedCall:output:0batch_normalization_7_410071batch_normalization_7_410073batch_normalization_7_410075batch_normalization_7_410077*
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4100242/
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
A__inference_add_3_layer_call_and_return_conditional_losses_4100862
add_3/PartitionedCall
+transformer_block_7/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0transformer_block_7_410446transformer_block_7_410448transformer_block_7_410450transformer_block_7_410452transformer_block_7_410454transformer_block_7_410456transformer_block_7_410458transformer_block_7_410460transformer_block_7_410462transformer_block_7_410464transformer_block_7_410466transformer_block_7_410468transformer_block_7_410470transformer_block_7_410472transformer_block_7_410474transformer_block_7_410476*
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
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_4102432-
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
E__inference_flatten_3_layer_call_and_return_conditional_losses_4104852
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
I__inference_concatenate_3_layer_call_and_return_conditional_losses_4105002
concatenate_3/PartitionedCall·
 dense_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_25_410531dense_25_410533*
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
D__inference_dense_25_layer_call_and_return_conditional_losses_4105202"
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
F__inference_dropout_22_layer_call_and_return_conditional_losses_4105482$
"dropout_22/StatefulPartitionedCall¼
 dense_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_26_410588dense_26_410590*
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
D__inference_dense_26_layer_call_and_return_conditional_losses_4105772"
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
F__inference_dropout_23_layer_call_and_return_conditional_losses_4106052$
"dropout_23/StatefulPartitionedCall¼
 dense_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_27_410644dense_27_410646*
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
D__inference_dense_27_layer_call_and_return_conditional_losses_4106332"
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


Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412105

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
D__inference_dense_24_layer_call_and_return_conditional_losses_413003

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
± 
ã
D__inference_dense_23_layer_call_and_return_conditional_losses_409654

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

G
+__inference_dropout_23_layer_call_fn_412774

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
F__inference_dropout_23_layer_call_and_return_conditional_losses_4106102
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
	
Ý
D__inference_dense_27_layer_call_and_return_conditional_losses_412784

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
î	
Ý
D__inference_dense_26_layer_call_and_return_conditional_losses_412738

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

Q
5__inference_average_pooling1d_10_layer_call_fn_409324

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
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_4093182
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
	
Ý
D__inference_dense_27_layer_call_and_return_conditional_losses_410633

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
¨
Z
.__inference_concatenate_3_layer_call_fn_412680
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
I__inference_concatenate_3_layer_call_and_return_conditional_losses_4105002
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
ó0
È
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_409435

inputs
assignmovingavg_409410
assignmovingavg_1_409416)
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
loc:@AssignMovingAvg/409410*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_409410*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/409410*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/409410*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_409410AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/409410*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/409416*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_409416*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/409416*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/409416*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_409416AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/409416*
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
¸
 
-__inference_sequential_7_layer_call_fn_412920

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
H__inference_sequential_7_layer_call_and_return_conditional_losses_4097482
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
á
~
)__inference_dense_25_layer_call_fn_412700

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
D__inference_dense_25_layer_call_and_return_conditional_losses_4105202
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
Ð
¨
-__inference_sequential_7_layer_call_fn_409759
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
H__inference_sequential_7_layer_call_and_return_conditional_losses_4097482
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
± 
ã
D__inference_dense_23_layer_call_and_return_conditional_losses_412964

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

Q
5__inference_average_pooling1d_11_layer_call_fn_409339

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
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_4093332
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
ÂÉ
1
"__inference__traced_restore_413490
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
assignvariableop_18_decay%
!assignvariableop_19_learning_rate 
assignvariableop_20_momentum 
assignvariableop_21_sgd_iterM
Iassignvariableop_22_token_and_position_embedding_3_embedding_6_embeddingsM
Iassignvariableop_23_token_and_position_embedding_3_embedding_7_embeddingsO
Kassignvariableop_24_transformer_block_7_multi_head_attention_7_query_kernelM
Iassignvariableop_25_transformer_block_7_multi_head_attention_7_query_biasM
Iassignvariableop_26_transformer_block_7_multi_head_attention_7_key_kernelK
Gassignvariableop_27_transformer_block_7_multi_head_attention_7_key_biasO
Kassignvariableop_28_transformer_block_7_multi_head_attention_7_value_kernelM
Iassignvariableop_29_transformer_block_7_multi_head_attention_7_value_biasZ
Vassignvariableop_30_transformer_block_7_multi_head_attention_7_attention_output_kernelX
Tassignvariableop_31_transformer_block_7_multi_head_attention_7_attention_output_bias'
#assignvariableop_32_dense_23_kernel%
!assignvariableop_33_dense_23_bias'
#assignvariableop_34_dense_24_kernel%
!assignvariableop_35_dense_24_biasH
Dassignvariableop_36_transformer_block_7_layer_normalization_14_gammaG
Cassignvariableop_37_transformer_block_7_layer_normalization_14_betaH
Dassignvariableop_38_transformer_block_7_layer_normalization_15_gammaG
Cassignvariableop_39_transformer_block_7_layer_normalization_15_beta
assignvariableop_40_total
assignvariableop_41_count4
0assignvariableop_42_sgd_conv1d_6_kernel_momentum2
.assignvariableop_43_sgd_conv1d_6_bias_momentum4
0assignvariableop_44_sgd_conv1d_7_kernel_momentum2
.assignvariableop_45_sgd_conv1d_7_bias_momentum@
<assignvariableop_46_sgd_batch_normalization_6_gamma_momentum?
;assignvariableop_47_sgd_batch_normalization_6_beta_momentum@
<assignvariableop_48_sgd_batch_normalization_7_gamma_momentum?
;assignvariableop_49_sgd_batch_normalization_7_beta_momentum4
0assignvariableop_50_sgd_dense_25_kernel_momentum2
.assignvariableop_51_sgd_dense_25_bias_momentum4
0assignvariableop_52_sgd_dense_26_kernel_momentum2
.assignvariableop_53_sgd_dense_26_bias_momentum4
0assignvariableop_54_sgd_dense_27_kernel_momentum2
.assignvariableop_55_sgd_dense_27_bias_momentumZ
Vassignvariableop_56_sgd_token_and_position_embedding_3_embedding_6_embeddings_momentumZ
Vassignvariableop_57_sgd_token_and_position_embedding_3_embedding_7_embeddings_momentum\
Xassignvariableop_58_sgd_transformer_block_7_multi_head_attention_7_query_kernel_momentumZ
Vassignvariableop_59_sgd_transformer_block_7_multi_head_attention_7_query_bias_momentumZ
Vassignvariableop_60_sgd_transformer_block_7_multi_head_attention_7_key_kernel_momentumX
Tassignvariableop_61_sgd_transformer_block_7_multi_head_attention_7_key_bias_momentum\
Xassignvariableop_62_sgd_transformer_block_7_multi_head_attention_7_value_kernel_momentumZ
Vassignvariableop_63_sgd_transformer_block_7_multi_head_attention_7_value_bias_momentumg
cassignvariableop_64_sgd_transformer_block_7_multi_head_attention_7_attention_output_kernel_momentume
aassignvariableop_65_sgd_transformer_block_7_multi_head_attention_7_attention_output_bias_momentum4
0assignvariableop_66_sgd_dense_23_kernel_momentum2
.assignvariableop_67_sgd_dense_23_bias_momentum4
0assignvariableop_68_sgd_dense_24_kernel_momentum2
.assignvariableop_69_sgd_dense_24_bias_momentumU
Qassignvariableop_70_sgd_transformer_block_7_layer_normalization_14_gamma_momentumT
Passignvariableop_71_sgd_transformer_block_7_layer_normalization_14_beta_momentumU
Qassignvariableop_72_sgd_transformer_block_7_layer_normalization_15_gamma_momentumT
Passignvariableop_73_sgd_transformer_block_7_layer_normalization_15_beta_momentum
identity_75¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_8¢AssignVariableOp_9Ñ(
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*Ý'
valueÓ'BÐ'KB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOp_22AssignVariableOpIassignvariableop_22_token_and_position_embedding_3_embedding_6_embeddingsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ñ
AssignVariableOp_23AssignVariableOpIassignvariableop_23_token_and_position_embedding_3_embedding_7_embeddingsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ó
AssignVariableOp_24AssignVariableOpKassignvariableop_24_transformer_block_7_multi_head_attention_7_query_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ñ
AssignVariableOp_25AssignVariableOpIassignvariableop_25_transformer_block_7_multi_head_attention_7_query_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ñ
AssignVariableOp_26AssignVariableOpIassignvariableop_26_transformer_block_7_multi_head_attention_7_key_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ï
AssignVariableOp_27AssignVariableOpGassignvariableop_27_transformer_block_7_multi_head_attention_7_key_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ó
AssignVariableOp_28AssignVariableOpKassignvariableop_28_transformer_block_7_multi_head_attention_7_value_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ñ
AssignVariableOp_29AssignVariableOpIassignvariableop_29_transformer_block_7_multi_head_attention_7_value_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Þ
AssignVariableOp_30AssignVariableOpVassignvariableop_30_transformer_block_7_multi_head_attention_7_attention_output_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ü
AssignVariableOp_31AssignVariableOpTassignvariableop_31_transformer_block_7_multi_head_attention_7_attention_output_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32«
AssignVariableOp_32AssignVariableOp#assignvariableop_32_dense_23_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33©
AssignVariableOp_33AssignVariableOp!assignvariableop_33_dense_23_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34«
AssignVariableOp_34AssignVariableOp#assignvariableop_34_dense_24_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35©
AssignVariableOp_35AssignVariableOp!assignvariableop_35_dense_24_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ì
AssignVariableOp_36AssignVariableOpDassignvariableop_36_transformer_block_7_layer_normalization_14_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ë
AssignVariableOp_37AssignVariableOpCassignvariableop_37_transformer_block_7_layer_normalization_14_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ì
AssignVariableOp_38AssignVariableOpDassignvariableop_38_transformer_block_7_layer_normalization_15_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Ë
AssignVariableOp_39AssignVariableOpCassignvariableop_39_transformer_block_7_layer_normalization_15_betaIdentity_39:output:0"/device:CPU:0*
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
AssignVariableOp_42AssignVariableOp0assignvariableop_42_sgd_conv1d_6_kernel_momentumIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43¶
AssignVariableOp_43AssignVariableOp.assignvariableop_43_sgd_conv1d_6_bias_momentumIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¸
AssignVariableOp_44AssignVariableOp0assignvariableop_44_sgd_conv1d_7_kernel_momentumIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45¶
AssignVariableOp_45AssignVariableOp.assignvariableop_45_sgd_conv1d_7_bias_momentumIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Ä
AssignVariableOp_46AssignVariableOp<assignvariableop_46_sgd_batch_normalization_6_gamma_momentumIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Ã
AssignVariableOp_47AssignVariableOp;assignvariableop_47_sgd_batch_normalization_6_beta_momentumIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Ä
AssignVariableOp_48AssignVariableOp<assignvariableop_48_sgd_batch_normalization_7_gamma_momentumIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Ã
AssignVariableOp_49AssignVariableOp;assignvariableop_49_sgd_batch_normalization_7_beta_momentumIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50¸
AssignVariableOp_50AssignVariableOp0assignvariableop_50_sgd_dense_25_kernel_momentumIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51¶
AssignVariableOp_51AssignVariableOp.assignvariableop_51_sgd_dense_25_bias_momentumIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52¸
AssignVariableOp_52AssignVariableOp0assignvariableop_52_sgd_dense_26_kernel_momentumIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53¶
AssignVariableOp_53AssignVariableOp.assignvariableop_53_sgd_dense_26_bias_momentumIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54¸
AssignVariableOp_54AssignVariableOp0assignvariableop_54_sgd_dense_27_kernel_momentumIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55¶
AssignVariableOp_55AssignVariableOp.assignvariableop_55_sgd_dense_27_bias_momentumIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Þ
AssignVariableOp_56AssignVariableOpVassignvariableop_56_sgd_token_and_position_embedding_3_embedding_6_embeddings_momentumIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57Þ
AssignVariableOp_57AssignVariableOpVassignvariableop_57_sgd_token_and_position_embedding_3_embedding_7_embeddings_momentumIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58à
AssignVariableOp_58AssignVariableOpXassignvariableop_58_sgd_transformer_block_7_multi_head_attention_7_query_kernel_momentumIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Þ
AssignVariableOp_59AssignVariableOpVassignvariableop_59_sgd_transformer_block_7_multi_head_attention_7_query_bias_momentumIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Þ
AssignVariableOp_60AssignVariableOpVassignvariableop_60_sgd_transformer_block_7_multi_head_attention_7_key_kernel_momentumIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Ü
AssignVariableOp_61AssignVariableOpTassignvariableop_61_sgd_transformer_block_7_multi_head_attention_7_key_bias_momentumIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62à
AssignVariableOp_62AssignVariableOpXassignvariableop_62_sgd_transformer_block_7_multi_head_attention_7_value_kernel_momentumIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63Þ
AssignVariableOp_63AssignVariableOpVassignvariableop_63_sgd_transformer_block_7_multi_head_attention_7_value_bias_momentumIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64ë
AssignVariableOp_64AssignVariableOpcassignvariableop_64_sgd_transformer_block_7_multi_head_attention_7_attention_output_kernel_momentumIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65é
AssignVariableOp_65AssignVariableOpaassignvariableop_65_sgd_transformer_block_7_multi_head_attention_7_attention_output_bias_momentumIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66¸
AssignVariableOp_66AssignVariableOp0assignvariableop_66_sgd_dense_23_kernel_momentumIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67¶
AssignVariableOp_67AssignVariableOp.assignvariableop_67_sgd_dense_23_bias_momentumIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68¸
AssignVariableOp_68AssignVariableOp0assignvariableop_68_sgd_dense_24_kernel_momentumIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69¶
AssignVariableOp_69AssignVariableOp.assignvariableop_69_sgd_dense_24_bias_momentumIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Ù
AssignVariableOp_70AssignVariableOpQassignvariableop_70_sgd_transformer_block_7_layer_normalization_14_gamma_momentumIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71Ø
AssignVariableOp_71AssignVariableOpPassignvariableop_71_sgd_transformer_block_7_layer_normalization_14_beta_momentumIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72Ù
AssignVariableOp_72AssignVariableOpQassignvariableop_72_sgd_transformer_block_7_layer_normalization_15_gamma_momentumIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73Ø
AssignVariableOp_73AssignVariableOpPassignvariableop_73_sgd_transformer_block_7_layer_normalization_15_beta_momentumIdentity_73:output:0"/device:CPU:0*
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
¢
÷
D__inference_conv1d_6_layer_call_and_return_conditional_losses_411933

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

G
+__inference_dropout_22_layer_call_fn_412727

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
F__inference_dropout_22_layer_call_and_return_conditional_losses_4105532
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
ô[
è
C__inference_model_3_layer_call_and_return_conditional_losses_410842

inputs
inputs_1)
%token_and_position_embedding_3_410752)
%token_and_position_embedding_3_410754
conv1d_6_410757
conv1d_6_410759
conv1d_7_410763
conv1d_7_410765 
batch_normalization_6_410770 
batch_normalization_6_410772 
batch_normalization_6_410774 
batch_normalization_6_410776 
batch_normalization_7_410779 
batch_normalization_7_410781 
batch_normalization_7_410783 
batch_normalization_7_410785
transformer_block_7_410789
transformer_block_7_410791
transformer_block_7_410793
transformer_block_7_410795
transformer_block_7_410797
transformer_block_7_410799
transformer_block_7_410801
transformer_block_7_410803
transformer_block_7_410805
transformer_block_7_410807
transformer_block_7_410809
transformer_block_7_410811
transformer_block_7_410813
transformer_block_7_410815
transformer_block_7_410817
transformer_block_7_410819
dense_25_410824
dense_25_410826
dense_26_410830
dense_26_410832
dense_27_410836
dense_27_410838
identity¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢ conv1d_6/StatefulPartitionedCall¢ conv1d_7/StatefulPartitionedCall¢ dense_25/StatefulPartitionedCall¢ dense_26/StatefulPartitionedCall¢ dense_27/StatefulPartitionedCall¢"dropout_22/StatefulPartitionedCall¢"dropout_23/StatefulPartitionedCall¢6token_and_position_embedding_3/StatefulPartitionedCall¢+transformer_block_7/StatefulPartitionedCall
6token_and_position_embedding_3/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_3_410752%token_and_position_embedding_3_410754*
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
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_40981528
6token_and_position_embedding_3/StatefulPartitionedCallÖ
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0conv1d_6_410757conv1d_6_410759*
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
D__inference_conv1d_6_layer_call_and_return_conditional_losses_4098472"
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
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_4093032%
#average_pooling1d_9/PartitionedCallÂ
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_9/PartitionedCall:output:0conv1d_7_410763conv1d_7_410765*
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
D__inference_conv1d_7_layer_call_and_return_conditional_losses_4098802"
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
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_4093332&
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
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_4093182&
$average_pooling1d_10/PartitionedCallÁ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_10/PartitionedCall:output:0batch_normalization_6_410770batch_normalization_6_410772batch_normalization_6_410774batch_normalization_6_410776*
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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4099332/
-batch_normalization_6/StatefulPartitionedCallÁ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_11/PartitionedCall:output:0batch_normalization_7_410779batch_normalization_7_410781batch_normalization_7_410783batch_normalization_7_410785*
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4100242/
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
A__inference_add_3_layer_call_and_return_conditional_losses_4100862
add_3/PartitionedCall
+transformer_block_7/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0transformer_block_7_410789transformer_block_7_410791transformer_block_7_410793transformer_block_7_410795transformer_block_7_410797transformer_block_7_410799transformer_block_7_410801transformer_block_7_410803transformer_block_7_410805transformer_block_7_410807transformer_block_7_410809transformer_block_7_410811transformer_block_7_410813transformer_block_7_410815transformer_block_7_410817transformer_block_7_410819*
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
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_4102432-
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
E__inference_flatten_3_layer_call_and_return_conditional_losses_4104852
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
I__inference_concatenate_3_layer_call_and_return_conditional_losses_4105002
concatenate_3/PartitionedCall·
 dense_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_25_410824dense_25_410826*
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
D__inference_dense_25_layer_call_and_return_conditional_losses_4105202"
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
F__inference_dropout_22_layer_call_and_return_conditional_losses_4105482$
"dropout_22/StatefulPartitionedCall¼
 dense_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_26_410830dense_26_410832*
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
D__inference_dense_26_layer_call_and_return_conditional_losses_4105772"
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
F__inference_dropout_23_layer_call_and_return_conditional_losses_4106052$
"dropout_23/StatefulPartitionedCall¼
 dense_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_27_410836dense_27_410838*
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
D__inference_dense_27_layer_call_and_return_conditional_losses_4106332"
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

e
F__inference_dropout_22_layer_call_and_return_conditional_losses_410548

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ýJ?2
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
 *
×#<2
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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412023

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
¡
F
*__inference_flatten_3_layer_call_fn_412667

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
E__inference_flatten_3_layer_call_and_return_conditional_losses_4104852
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
¬
R
&__inference_add_3_layer_call_fn_412307
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
A__inference_add_3_layer_call_and_return_conditional_losses_4100862
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
µ
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_412662

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
Ê
©
6__inference_batch_normalization_6_layer_call_fn_412049

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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4099532
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
ì
©
6__inference_batch_normalization_6_layer_call_fn_412118

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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4094352
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

è(
!__inference__wrapped_model_409294
input_7
input_8N
Jmodel_3_token_and_position_embedding_3_embedding_7_embedding_lookup_409063N
Jmodel_3_token_and_position_embedding_3_embedding_6_embedding_lookup_409069@
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
Cmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookupResourceGatherJmodel_3_token_and_position_embedding_3_embedding_7_embedding_lookup_4090635model_3/token_and_position_embedding_3/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*]
_classS
QOloc:@model_3/token_and_position_embedding_3/embedding_7/embedding_lookup/409063*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02E
Cmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookupµ
Lmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup/IdentityIdentityLmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*]
_classS
QOloc:@model_3/token_and_position_embedding_3/embedding_7/embedding_lookup/409063*'
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
Cmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookupResourceGatherJmodel_3_token_and_position_embedding_3_embedding_6_embedding_lookup_409069;model_3/token_and_position_embedding_3/embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*]
_classS
QOloc:@model_3/token_and_position_embedding_3/embedding_6/embedding_lookup/409069*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype02E
Cmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup»
Lmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup/IdentityIdentityLmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*]
_classS
QOloc:@model_3/token_and_position_embedding_3/embedding_6/embedding_lookup/409069*-
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
¼0
È
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_409933

inputs
assignmovingavg_409908
assignmovingavg_1_409914)
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
loc:@AssignMovingAvg/409908*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_409908*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/409908*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/409908*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_409908AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/409908*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/409914*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_409914*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/409914*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/409914*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_409914AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/409914*
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
Ñ
ã
D__inference_dense_24_layer_call_and_return_conditional_losses_409700

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
É
d
F__inference_dropout_22_layer_call_and_return_conditional_losses_410553

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
ÿ
â
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_410243

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
,multi_head_attention_7/dropout/dropout/Shape
Cmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_7/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
dtype02E
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
dropout_20/dropout/ShapeÙ
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
dtype021
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
dropout_21/dropout/ShapeÙ
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
dtype021
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
é

H__inference_sequential_7_layer_call_and_return_conditional_losses_409748

inputs
dense_23_409737
dense_23_409739
dense_24_409742
dense_24_409744
identity¢ dense_23/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCall
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinputsdense_23_409737dense_23_409739*
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
D__inference_dense_23_layer_call_and_return_conditional_losses_4096542"
 dense_23/StatefulPartitionedCall¾
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_409742dense_24_409744*
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
D__inference_dense_24_layer_call_and_return_conditional_losses_4097002"
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
¯£
þ(
__inference__traced_save_413258
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
(savev2_dense_27_bias_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop'
#savev2_momentum_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	T
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
 savev2_count_read_readvariableop;
7savev2_sgd_conv1d_6_kernel_momentum_read_readvariableop9
5savev2_sgd_conv1d_6_bias_momentum_read_readvariableop;
7savev2_sgd_conv1d_7_kernel_momentum_read_readvariableop9
5savev2_sgd_conv1d_7_bias_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_6_gamma_momentum_read_readvariableopF
Bsavev2_sgd_batch_normalization_6_beta_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_7_gamma_momentum_read_readvariableopF
Bsavev2_sgd_batch_normalization_7_beta_momentum_read_readvariableop;
7savev2_sgd_dense_25_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_25_bias_momentum_read_readvariableop;
7savev2_sgd_dense_26_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_26_bias_momentum_read_readvariableop;
7savev2_sgd_dense_27_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_27_bias_momentum_read_readvariableopa
]savev2_sgd_token_and_position_embedding_3_embedding_6_embeddings_momentum_read_readvariableopa
]savev2_sgd_token_and_position_embedding_3_embedding_7_embeddings_momentum_read_readvariableopc
_savev2_sgd_transformer_block_7_multi_head_attention_7_query_kernel_momentum_read_readvariableopa
]savev2_sgd_transformer_block_7_multi_head_attention_7_query_bias_momentum_read_readvariableopa
]savev2_sgd_transformer_block_7_multi_head_attention_7_key_kernel_momentum_read_readvariableop_
[savev2_sgd_transformer_block_7_multi_head_attention_7_key_bias_momentum_read_readvariableopc
_savev2_sgd_transformer_block_7_multi_head_attention_7_value_kernel_momentum_read_readvariableopa
]savev2_sgd_transformer_block_7_multi_head_attention_7_value_bias_momentum_read_readvariableopn
jsavev2_sgd_transformer_block_7_multi_head_attention_7_attention_output_kernel_momentum_read_readvariableopl
hsavev2_sgd_transformer_block_7_multi_head_attention_7_attention_output_bias_momentum_read_readvariableop;
7savev2_sgd_dense_23_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_23_bias_momentum_read_readvariableop;
7savev2_sgd_dense_24_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_24_bias_momentum_read_readvariableop\
Xsavev2_sgd_transformer_block_7_layer_normalization_14_gamma_momentum_read_readvariableop[
Wsavev2_sgd_transformer_block_7_layer_normalization_14_beta_momentum_read_readvariableop\
Xsavev2_sgd_transformer_block_7_layer_normalization_15_gamma_momentum_read_readvariableop[
Wsavev2_sgd_transformer_block_7_layer_normalization_15_beta_momentum_read_readvariableop
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
ShardedFilenameË(
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*Ý'
valueÓ'BÐ'KB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¡
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*«
value¡BKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesî'
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_6_kernel_read_readvariableop(savev2_conv1d_6_bias_read_readvariableop*savev2_conv1d_7_kernel_read_readvariableop(savev2_conv1d_7_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableopPsavev2_token_and_position_embedding_3_embedding_6_embeddings_read_readvariableopPsavev2_token_and_position_embedding_3_embedding_7_embeddings_read_readvariableopRsavev2_transformer_block_7_multi_head_attention_7_query_kernel_read_readvariableopPsavev2_transformer_block_7_multi_head_attention_7_query_bias_read_readvariableopPsavev2_transformer_block_7_multi_head_attention_7_key_kernel_read_readvariableopNsavev2_transformer_block_7_multi_head_attention_7_key_bias_read_readvariableopRsavev2_transformer_block_7_multi_head_attention_7_value_kernel_read_readvariableopPsavev2_transformer_block_7_multi_head_attention_7_value_bias_read_readvariableop]savev2_transformer_block_7_multi_head_attention_7_attention_output_kernel_read_readvariableop[savev2_transformer_block_7_multi_head_attention_7_attention_output_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableopKsavev2_transformer_block_7_layer_normalization_14_gamma_read_readvariableopJsavev2_transformer_block_7_layer_normalization_14_beta_read_readvariableopKsavev2_transformer_block_7_layer_normalization_15_gamma_read_readvariableopJsavev2_transformer_block_7_layer_normalization_15_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop7savev2_sgd_conv1d_6_kernel_momentum_read_readvariableop5savev2_sgd_conv1d_6_bias_momentum_read_readvariableop7savev2_sgd_conv1d_7_kernel_momentum_read_readvariableop5savev2_sgd_conv1d_7_bias_momentum_read_readvariableopCsavev2_sgd_batch_normalization_6_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_6_beta_momentum_read_readvariableopCsavev2_sgd_batch_normalization_7_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_7_beta_momentum_read_readvariableop7savev2_sgd_dense_25_kernel_momentum_read_readvariableop5savev2_sgd_dense_25_bias_momentum_read_readvariableop7savev2_sgd_dense_26_kernel_momentum_read_readvariableop5savev2_sgd_dense_26_bias_momentum_read_readvariableop7savev2_sgd_dense_27_kernel_momentum_read_readvariableop5savev2_sgd_dense_27_bias_momentum_read_readvariableop]savev2_sgd_token_and_position_embedding_3_embedding_6_embeddings_momentum_read_readvariableop]savev2_sgd_token_and_position_embedding_3_embedding_7_embeddings_momentum_read_readvariableop_savev2_sgd_transformer_block_7_multi_head_attention_7_query_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_7_multi_head_attention_7_query_bias_momentum_read_readvariableop]savev2_sgd_transformer_block_7_multi_head_attention_7_key_kernel_momentum_read_readvariableop[savev2_sgd_transformer_block_7_multi_head_attention_7_key_bias_momentum_read_readvariableop_savev2_sgd_transformer_block_7_multi_head_attention_7_value_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_7_multi_head_attention_7_value_bias_momentum_read_readvariableopjsavev2_sgd_transformer_block_7_multi_head_attention_7_attention_output_kernel_momentum_read_readvariableophsavev2_sgd_transformer_block_7_multi_head_attention_7_attention_output_bias_momentum_read_readvariableop7savev2_sgd_dense_23_kernel_momentum_read_readvariableop5savev2_sgd_dense_23_bias_momentum_read_readvariableop7savev2_sgd_dense_24_kernel_momentum_read_readvariableop5savev2_sgd_dense_24_bias_momentum_read_readvariableopXsavev2_sgd_transformer_block_7_layer_normalization_14_gamma_momentum_read_readvariableopWsavev2_sgd_transformer_block_7_layer_normalization_14_beta_momentum_read_readvariableopXsavev2_sgd_transformer_block_7_layer_normalization_15_gamma_momentum_read_readvariableopWsavev2_sgd_transformer_block_7_layer_normalization_15_beta_momentum_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*ó
_input_shapesá
Þ: :  : :	  : : : : : : : : : :	È@:@:@@:@:@:: : : : : :
  :  : :  : :  : :  : : @:@:@ : : : : : : : :  : :	  : : : : : :	È@:@:@@:@:@:: :
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
: :$ 

_output_shapes

: :&"
 
_output_shapes
:
  :($
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
:	È@: 4
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

: :&:"
 
_output_shapes
:
  :(;$
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
·
k
A__inference_add_3_layer_call_and_return_conditional_losses_410086

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
õ
k
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_409303

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


H__inference_sequential_7_layer_call_and_return_conditional_losses_409717
dense_23_input
dense_23_409665
dense_23_409667
dense_24_409711
dense_24_409713
identity¢ dense_23/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCall£
 dense_23/StatefulPartitionedCallStatefulPartitionedCalldense_23_inputdense_23_409665dense_23_409667*
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
D__inference_dense_23_layer_call_and_return_conditional_losses_4096542"
 dense_23/StatefulPartitionedCall¾
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_409711dense_24_409713*
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
D__inference_dense_24_layer_call_and_return_conditional_losses_4097002"
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
¼0
È
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_410024

inputs
assignmovingavg_409999
assignmovingavg_1_410005)
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
loc:@AssignMovingAvg/409999*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_409999*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/409999*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/409999*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_409999AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/409999*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/410005*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_410005*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/410005*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/410005*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_410005AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/410005*
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
Ð

à
4__inference_transformer_block_7_layer_call_fn_412619

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
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_4102432
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
¸
 
-__inference_sequential_7_layer_call_fn_412933

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
H__inference_sequential_7_layer_call_and_return_conditional_losses_4097752
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_410044

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
ï
~
)__inference_dense_23_layer_call_fn_412973

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
D__inference_dense_23_layer_call_and_return_conditional_losses_4096542
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
È
©
6__inference_batch_normalization_7_layer_call_fn_412282

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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4100242
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

P
4__inference_average_pooling1d_9_layer_call_fn_409309

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
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_4093032
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
ß
~
)__inference_dense_26_layer_call_fn_412747

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
D__inference_dense_26_layer_call_and_return_conditional_losses_4105772
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
¼0
È
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412003

inputs
assignmovingavg_411978
assignmovingavg_1_411984)
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
loc:@AssignMovingAvg/411978*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_411978*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/411978*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/411978*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_411978AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/411978*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/411984*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_411984*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/411984*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/411984*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_411984AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/411984*
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
¥
d
+__inference_dropout_23_layer_call_fn_412769

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
F__inference_dropout_23_layer_call_and_return_conditional_losses_4106052
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
ü

Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_409815
x'
#embedding_7_embedding_lookup_409802'
#embedding_6_embedding_lookup_409808
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
embedding_7/embedding_lookupResourceGather#embedding_7_embedding_lookup_409802range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_7/embedding_lookup/409802*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02
embedding_7/embedding_lookup
%embedding_7/embedding_lookup/IdentityIdentity%embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_7/embedding_lookup/409802*'
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
embedding_6/embedding_lookupResourceGather#embedding_6_embedding_lookup_409808embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_6/embedding_lookup/409808*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype02
embedding_6/embedding_lookup
%embedding_6/embedding_lookup/IdentityIdentity%embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_6/embedding_lookup/409808*-
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
ÿ
â
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_412455

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
,multi_head_attention_7/dropout/dropout/Shape
Cmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_7/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
dtype02E
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
dropout_20/dropout/ShapeÙ
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
dtype021
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
dropout_21/dropout/ShapeÙ
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
dtype021
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
Ê
©
6__inference_batch_normalization_7_layer_call_fn_412295

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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4100442
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
ö
l
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_409318

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
é

H__inference_sequential_7_layer_call_and_return_conditional_losses_409775

inputs
dense_23_409764
dense_23_409766
dense_24_409769
dense_24_409771
identity¢ dense_23/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCall
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinputsdense_23_409764dense_23_409766*
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
D__inference_dense_23_layer_call_and_return_conditional_losses_4096542"
 dense_23/StatefulPartitionedCall¾
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_409769dense_24_409771*
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
D__inference_dense_24_layer_call_and_return_conditional_losses_4097002"
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
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_411908
x'
#embedding_7_embedding_lookup_411895'
#embedding_6_embedding_lookup_411901
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
embedding_7/embedding_lookupResourceGather#embedding_7_embedding_lookup_411895range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_7/embedding_lookup/411895*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02
embedding_7/embedding_lookup
%embedding_7/embedding_lookup/IdentityIdentity%embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_7/embedding_lookup/411895*'
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
embedding_6/embedding_lookupResourceGather#embedding_6_embedding_lookup_411901embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_6/embedding_lookup/411901*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype02
embedding_6/embedding_lookup
%embedding_6/embedding_lookup/IdentityIdentity%embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_6/embedding_lookup/411901*-
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
µ
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_410485

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
ï
~
)__inference_dense_24_layer_call_fn_413012

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
D__inference_dense_24_layer_call_and_return_conditional_losses_4097002
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
ó
~
)__inference_conv1d_7_layer_call_fn_411967

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
D__inference_conv1d_7_layer_call_and_return_conditional_losses_4098802
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
î	
Ý
D__inference_dense_26_layer_call_and_return_conditional_losses_410577

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
È
©
6__inference_batch_normalization_6_layer_call_fn_412036

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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4099332
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
î
©
6__inference_batch_normalization_7_layer_call_fn_412213

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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4096082
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
A__inference_add_3_layer_call_and_return_conditional_losses_412301
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
J
¯
H__inference_sequential_7_layer_call_and_return_conditional_losses_412850

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

÷
D__inference_conv1d_7_layer_call_and_return_conditional_losses_409880

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
ó0
È
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412167

inputs
assignmovingavg_412142
assignmovingavg_1_412148)
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
loc:@AssignMovingAvg/412142*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_412142*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/412142*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/412142*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_412142AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/412142*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/412148*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_412148*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/412148*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/412148*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_412148AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/412148*
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
Ø
¤
(__inference_model_3_layer_call_fn_411806
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
C__inference_model_3_layer_call_and_return_conditional_losses_4108422
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
è

Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412269

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
Ö
¢
(__inference_model_3_layer_call_fn_411089
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
C__inference_model_3_layer_call_and_return_conditional_losses_4110142
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
ëX

C__inference_model_3_layer_call_and_return_conditional_losses_411014

inputs
inputs_1)
%token_and_position_embedding_3_410924)
%token_and_position_embedding_3_410926
conv1d_6_410929
conv1d_6_410931
conv1d_7_410935
conv1d_7_410937 
batch_normalization_6_410942 
batch_normalization_6_410944 
batch_normalization_6_410946 
batch_normalization_6_410948 
batch_normalization_7_410951 
batch_normalization_7_410953 
batch_normalization_7_410955 
batch_normalization_7_410957
transformer_block_7_410961
transformer_block_7_410963
transformer_block_7_410965
transformer_block_7_410967
transformer_block_7_410969
transformer_block_7_410971
transformer_block_7_410973
transformer_block_7_410975
transformer_block_7_410977
transformer_block_7_410979
transformer_block_7_410981
transformer_block_7_410983
transformer_block_7_410985
transformer_block_7_410987
transformer_block_7_410989
transformer_block_7_410991
dense_25_410996
dense_25_410998
dense_26_411002
dense_26_411004
dense_27_411008
dense_27_411010
identity¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢ conv1d_6/StatefulPartitionedCall¢ conv1d_7/StatefulPartitionedCall¢ dense_25/StatefulPartitionedCall¢ dense_26/StatefulPartitionedCall¢ dense_27/StatefulPartitionedCall¢6token_and_position_embedding_3/StatefulPartitionedCall¢+transformer_block_7/StatefulPartitionedCall
6token_and_position_embedding_3/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_3_410924%token_and_position_embedding_3_410926*
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
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_40981528
6token_and_position_embedding_3/StatefulPartitionedCallÖ
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0conv1d_6_410929conv1d_6_410931*
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
D__inference_conv1d_6_layer_call_and_return_conditional_losses_4098472"
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
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_4093032%
#average_pooling1d_9/PartitionedCallÂ
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_9/PartitionedCall:output:0conv1d_7_410935conv1d_7_410937*
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
D__inference_conv1d_7_layer_call_and_return_conditional_losses_4098802"
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
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_4093332&
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
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_4093182&
$average_pooling1d_10/PartitionedCallÃ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_10/PartitionedCall:output:0batch_normalization_6_410942batch_normalization_6_410944batch_normalization_6_410946batch_normalization_6_410948*
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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4099532/
-batch_normalization_6/StatefulPartitionedCallÃ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_11/PartitionedCall:output:0batch_normalization_7_410951batch_normalization_7_410953batch_normalization_7_410955batch_normalization_7_410957*
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4100442/
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
A__inference_add_3_layer_call_and_return_conditional_losses_4100862
add_3/PartitionedCall
+transformer_block_7/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0transformer_block_7_410961transformer_block_7_410963transformer_block_7_410965transformer_block_7_410967transformer_block_7_410969transformer_block_7_410971transformer_block_7_410973transformer_block_7_410975transformer_block_7_410977transformer_block_7_410979transformer_block_7_410981transformer_block_7_410983transformer_block_7_410985transformer_block_7_410987transformer_block_7_410989transformer_block_7_410991*
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
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_4103702-
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
E__inference_flatten_3_layer_call_and_return_conditional_losses_4104852
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
I__inference_concatenate_3_layer_call_and_return_conditional_losses_4105002
concatenate_3/PartitionedCall·
 dense_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_25_410996dense_25_410998*
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
D__inference_dense_25_layer_call_and_return_conditional_losses_4105202"
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
F__inference_dropout_22_layer_call_and_return_conditional_losses_4105532
dropout_22/PartitionedCall´
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_26_411002dense_26_411004*
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
D__inference_dense_26_layer_call_and_return_conditional_losses_4105772"
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
F__inference_dropout_23_layer_call_and_return_conditional_losses_4106102
dropout_23/PartitionedCall´
 dense_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_27_411008dense_27_411010*
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
D__inference_dense_27_layer_call_and_return_conditional_losses_4106332"
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
î
©
6__inference_batch_normalization_6_layer_call_fn_412131

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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4094682
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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_409468

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
ÔÜ
Ì$
C__inference_model_3_layer_call_and_return_conditional_losses_411728
inputs_0
inputs_1F
Btoken_and_position_embedding_3_embedding_7_embedding_lookup_411497F
Btoken_and_position_embedding_3_embedding_6_embedding_lookup_4115038
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
;token_and_position_embedding_3/embedding_7/embedding_lookupResourceGatherBtoken_and_position_embedding_3_embedding_7_embedding_lookup_411497-token_and_position_embedding_3/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_7/embedding_lookup/411497*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02=
;token_and_position_embedding_3/embedding_7/embedding_lookup
Dtoken_and_position_embedding_3/embedding_7/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_3/embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_7/embedding_lookup/411497*'
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
;token_and_position_embedding_3/embedding_6/embedding_lookupResourceGatherBtoken_and_position_embedding_3_embedding_6_embedding_lookup_4115033token_and_position_embedding_3/embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_6/embedding_lookup/411503*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype02=
;token_and_position_embedding_3/embedding_6/embedding_lookup
Dtoken_and_position_embedding_3/embedding_6/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_3/embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_6/embedding_lookup/411503*-
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
×ê
¤&
C__inference_model_3_layer_call_and_return_conditional_losses_411485
inputs_0
inputs_1F
Btoken_and_position_embedding_3_embedding_7_embedding_lookup_411187F
Btoken_and_position_embedding_3_embedding_6_embedding_lookup_4111938
4conv1d_6_conv1d_expanddims_1_readvariableop_resource,
(conv1d_6_biasadd_readvariableop_resource8
4conv1d_7_conv1d_expanddims_1_readvariableop_resource,
(conv1d_7_biasadd_readvariableop_resource0
,batch_normalization_6_assignmovingavg_4112432
.batch_normalization_6_assignmovingavg_1_411249?
;batch_normalization_6_batchnorm_mul_readvariableop_resource;
7batch_normalization_6_batchnorm_readvariableop_resource0
,batch_normalization_7_assignmovingavg_4112752
.batch_normalization_7_assignmovingavg_1_411281?
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
;token_and_position_embedding_3/embedding_7/embedding_lookupResourceGatherBtoken_and_position_embedding_3_embedding_7_embedding_lookup_411187-token_and_position_embedding_3/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_7/embedding_lookup/411187*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02=
;token_and_position_embedding_3/embedding_7/embedding_lookup
Dtoken_and_position_embedding_3/embedding_7/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_3/embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_7/embedding_lookup/411187*'
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
;token_and_position_embedding_3/embedding_6/embedding_lookupResourceGatherBtoken_and_position_embedding_3_embedding_6_embedding_lookup_4111933token_and_position_embedding_3/embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_6/embedding_lookup/411193*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype02=
;token_and_position_embedding_3/embedding_6/embedding_lookup
Dtoken_and_position_embedding_3/embedding_6/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_3/embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_6/embedding_lookup/411193*-
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
31loc:@batch_normalization_6/AssignMovingAvg/411243*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_6/AssignMovingAvg/decayÕ
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_6_assignmovingavg_411243*
_output_shapes
: *
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOpß
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg/411243*
_output_shapes
: 2+
)batch_normalization_6/AssignMovingAvg/subÖ
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:04batch_normalization_6/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg/411243*
_output_shapes
: 2+
)batch_normalization_6/AssignMovingAvg/mul³
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_6_assignmovingavg_411243-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg/411243*
_output_shapes
 *
dtype02;
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_6/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/411249*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_6/AssignMovingAvg_1/decayÛ
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_6_assignmovingavg_1_411249*
_output_shapes
: *
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpé
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/411249*
_output_shapes
: 2-
+batch_normalization_6/AssignMovingAvg_1/subà
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:06batch_normalization_6/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/411249*
_output_shapes
: 2-
+batch_normalization_6/AssignMovingAvg_1/mul¿
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_6_assignmovingavg_1_411249/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/411249*
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
31loc:@batch_normalization_7/AssignMovingAvg/411275*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_7/AssignMovingAvg/decayÕ
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_7_assignmovingavg_411275*
_output_shapes
: *
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOpß
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg/411275*
_output_shapes
: 2+
)batch_normalization_7/AssignMovingAvg/subÖ
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:04batch_normalization_7/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg/411275*
_output_shapes
: 2+
)batch_normalization_7/AssignMovingAvg/mul³
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_7_assignmovingavg_411275-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg/411275*
_output_shapes
 *
dtype02;
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_7/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/411281*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_7/AssignMovingAvg_1/decayÛ
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_7_assignmovingavg_1_411281*
_output_shapes
: *
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpé
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/411281*
_output_shapes
: 2-
+batch_normalization_7/AssignMovingAvg_1/subà
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:06batch_normalization_7/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/411281*
_output_shapes
: 2-
+batch_normalization_7/AssignMovingAvg_1/mul¿
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_7_assignmovingavg_1_411281/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/411281*
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
@transformer_block_7/multi_head_attention_7/dropout/dropout/ShapeÕ
Wtransformer_block_7/multi_head_attention_7/dropout/dropout/random_uniform/RandomUniformRandomUniformItransformer_block_7/multi_head_attention_7/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
dtype02Y
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
,transformer_block_7/dropout_20/dropout/Shape
Ctransformer_block_7/dropout_20/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_7/dropout_20/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
dtype02E
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
,transformer_block_7/dropout_21/dropout/Shape
Ctransformer_block_7/dropout_21/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_7/dropout_21/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿB *
dtype02E
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
 *ýJ?2
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
dropout_22/dropout/ShapeÕ
/dropout_22/dropout/random_uniform/RandomUniformRandomUniform!dropout_22/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype021
/dropout_22/dropout/random_uniform/RandomUniform
!dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
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
 *ýJ?2
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
dropout_23/dropout/ShapeÕ
/dropout_23/dropout/random_uniform/RandomUniformRandomUniform!dropout_23/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype021
/dropout_23/dropout/random_uniform/RandomUniform
!dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
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
É
d
F__inference_dropout_22_layer_call_and_return_conditional_losses_412717

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

e
F__inference_dropout_22_layer_call_and_return_conditional_losses_412712

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ýJ?2
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
 *
×#<2
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
ñ	
Ý
D__inference_dense_25_layer_call_and_return_conditional_losses_410520

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


H__inference_sequential_7_layer_call_and_return_conditional_losses_409731
dense_23_input
dense_23_409720
dense_23_409722
dense_24_409725
dense_24_409727
identity¢ dense_23/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCall£
 dense_23/StatefulPartitionedCallStatefulPartitionedCalldense_23_inputdense_23_409720dense_23_409722*
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
D__inference_dense_23_layer_call_and_return_conditional_losses_4096542"
 dense_23/StatefulPartitionedCall¾
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_409725dense_24_409727*
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
D__inference_dense_24_layer_call_and_return_conditional_losses_4097002"
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

e
F__inference_dropout_23_layer_call_and_return_conditional_losses_410605

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ýJ?2
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
 *
×#<2
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
¥
d
+__inference_dropout_22_layer_call_fn_412722

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
F__inference_dropout_22_layer_call_and_return_conditional_losses_4105482
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
J
¯
H__inference_sequential_7_layer_call_and_return_conditional_losses_412907

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
°

$__inference_signature_wrapper_411175
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
!__inference__wrapped_model_4092942
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
º
s
I__inference_concatenate_3_layer_call_and_return_conditional_losses_410500

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
ß
~
)__inference_dense_27_layer_call_fn_412793

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
D__inference_dense_27_layer_call_and_return_conditional_losses_4106332
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
Ò
¢
(__inference_model_3_layer_call_fn_410917
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
C__inference_model_3_layer_call_and_return_conditional_losses_4108422
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


?__inference_token_and_position_embedding_3_layer_call_fn_411917
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
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_4098152
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
Ð

à
4__inference_transformer_block_7_layer_call_fn_412656

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
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_4103702
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
Ð
¨
-__inference_sequential_7_layer_call_fn_409786
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
H__inference_sequential_7_layer_call_and_return_conditional_losses_4097752
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
ì
©
6__inference_batch_normalization_7_layer_call_fn_412200

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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4095752
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
¼0
È
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412249

inputs
assignmovingavg_412224
assignmovingavg_1_412230)
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
loc:@AssignMovingAvg/412224*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_412224*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/412224*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/412224*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_412224AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/412224*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/412230*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_412230*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/412230*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/412230*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_412230AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/412230*
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
 
_user_specified_nameinputs"±L
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
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:èü
ÔG
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
trainable_variables
regularization_losses
	variables
	keras_api

signatures
±__call__
+²&call_and_return_all_conditional_losses
³_default_save_signature"B
_tf_keras_networkB{"class_name": "Functional", "name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}, "name": "token_and_position_embedding_3", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_6", "inbound_nodes": [[["token_and_position_embedding_3", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_9", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_9", "inbound_nodes": [[["conv1d_6", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_7", "inbound_nodes": [[["average_pooling1d_9", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_10", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_10", "inbound_nodes": [[["conv1d_7", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_11", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_11", "inbound_nodes": [[["token_and_position_embedding_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["average_pooling1d_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["average_pooling1d_11", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}], ["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block_7", "inbound_nodes": [[["add_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["transformer_block_7", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_3", "inbound_nodes": [[["flatten_3", 0, 0, {}], ["input_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_25", "inbound_nodes": [[["concatenate_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_22", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}, "name": "dropout_22", "inbound_nodes": [[["dense_25", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_26", "inbound_nodes": [[["dropout_22", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_23", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}, "name": "dropout_23", "inbound_nodes": [[["dense_26", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_27", "inbound_nodes": [[["dropout_23", 0, 0, {}]]]}], "input_layers": [["input_7", 0, 0], ["input_8", 0, 0]], "output_layers": [["dense_27", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 20000]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 20000]}, {"class_name": "TensorShape", "items": [null, 8]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.0010000000474974513, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
ñ"î
_tf_keras_input_layerÎ{"class_name": "InputLayer", "name": "input_7", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20000]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}}
ç
	token_emb
pos_emb
trainable_variables
regularization_losses
	variables
	keras_api
´__call__
+µ&call_and_return_all_conditional_losses"º
_tf_keras_layer {"class_name": "TokenAndPositionEmbedding", "name": "token_and_position_embedding_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
é	

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
¶__call__
+·&call_and_return_all_conditional_losses"Â
_tf_keras_layer¨{"class_name": "Conv1D", "name": "conv1d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20000, 32]}}

&trainable_variables
'regularization_losses
(	variables
)	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses"ø
_tf_keras_layerÞ{"class_name": "AveragePooling1D", "name": "average_pooling1d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_9", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ç	

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
º__call__
+»&call_and_return_all_conditional_losses"À
_tf_keras_layer¦{"class_name": "Conv1D", "name": "conv1d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 666, 32]}}

0trainable_variables
1regularization_losses
2	variables
3	keras_api
¼__call__
+½&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "AveragePooling1D", "name": "average_pooling1d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_10", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}

4trainable_variables
5regularization_losses
6	variables
7	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses"ü
_tf_keras_layerâ{"class_name": "AveragePooling1D", "name": "average_pooling1d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_11", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
¸	
8axis
	9gamma
:beta
;moving_mean
<moving_variance
=trainable_variables
>regularization_losses
?	variables
@	keras_api
À__call__
+Á&call_and_return_all_conditional_losses"â
_tf_keras_layerÈ{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
¸	
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
Â__call__
+Ã&call_and_return_all_conditional_losses"â
_tf_keras_layerÈ{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
³
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses"¢
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
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
Æ__call__
+Ç&call_and_return_all_conditional_losses"¥
_tf_keras_layer{"class_name": "TransformerBlock", "name": "transformer_block_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
è
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
È__call__
+É&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
é"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_8", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}}
Ð
\trainable_variables
]regularization_losses
^	variables
_	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses"¿
_tf_keras_layer¥{"class_name": "Concatenate", "name": "concatenate_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 2112]}, {"class_name": "TensorShape", "items": [null, 8]}]}
ø

`kernel
abias
btrainable_variables
cregularization_losses
d	variables
e	keras_api
Ì__call__
+Í&call_and_return_all_conditional_losses"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "dense_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2120]}}
ê
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
Î__call__
+Ï&call_and_return_all_conditional_losses"Ù
_tf_keras_layer¿{"class_name": "Dropout", "name": "dropout_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_22", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}}
ô

jkernel
kbias
ltrainable_variables
mregularization_losses
n	variables
o	keras_api
Ð__call__
+Ñ&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ê
ptrainable_variables
qregularization_losses
r	variables
s	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses"Ù
_tf_keras_layer¿{"class_name": "Dropout", "name": "dropout_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_23", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}}
õ

tkernel
ubias
vtrainable_variables
wregularization_losses
x	variables
y	keras_api
Ô__call__
+Õ&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ù
	zdecay
{learning_rate
|momentum
}iter momentum!momentum*momentum+momentum9momentum:momentumBmomentumCmomentum`momentumamomentumjmomentumkmomentumtmomentumumomentum~momentummomentum momentum¡momentum¢momentum£momentum¤momentum¥momentum¦momentum§momentum¨momentum©momentumªmomentum«momentum¬momentum­momentum®momentum¯momentum°"
	optimizer
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
Ó
trainable_variables
non_trainable_variables
metrics
layers
regularization_losses
layer_metrics
 layer_regularization_losses
	variables
±__call__
³_default_save_signature
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
-
Öserving_default"
signature_map
´
~
embeddings
trainable_variables
regularization_losses
	variables
	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses"
_tf_keras_layerõ{"class_name": "Embedding", "name": "embedding_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20000]}}
±

embeddings
trainable_variables
regularization_losses
	variables
	keras_api
Ù__call__
+Ú&call_and_return_all_conditional_losses"
_tf_keras_layerò{"class_name": "Embedding", "name": "embedding_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 20000, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
µ
metrics
non_trainable_variables
layers
trainable_variables
regularization_losses
 layer_metrics
 ¡layer_regularization_losses
	variables
´__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
%:#  2conv1d_6/kernel
: 2conv1d_6/bias
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
µ
¢metrics
£non_trainable_variables
¤layers
"trainable_variables
#regularization_losses
¥layer_metrics
 ¦layer_regularization_losses
$	variables
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
§metrics
¨non_trainable_variables
©layers
&trainable_variables
'regularization_losses
ªlayer_metrics
 «layer_regularization_losses
(	variables
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
%:#	  2conv1d_7/kernel
: 2conv1d_7/bias
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
µ
¬metrics
­non_trainable_variables
®layers
,trainable_variables
-regularization_losses
¯layer_metrics
 °layer_regularization_losses
.	variables
º__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
±metrics
²non_trainable_variables
³layers
0trainable_variables
1regularization_losses
´layer_metrics
 µlayer_regularization_losses
2	variables
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¶metrics
·non_trainable_variables
¸layers
4trainable_variables
5regularization_losses
¹layer_metrics
 ºlayer_regularization_losses
6	variables
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_6/gamma
(:& 2batch_normalization_6/beta
1:/  (2!batch_normalization_6/moving_mean
5:3  (2%batch_normalization_6/moving_variance
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
90
:1
;2
<3"
trackable_list_wrapper
µ
»metrics
¼non_trainable_variables
½layers
=trainable_variables
>regularization_losses
¾layer_metrics
 ¿layer_regularization_losses
?	variables
À__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_7/gamma
(:& 2batch_normalization_7/beta
1:/  (2!batch_normalization_7/moving_mean
5:3  (2%batch_normalization_7/moving_variance
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
µ
Àmetrics
Ánon_trainable_variables
Âlayers
Ftrainable_variables
Gregularization_losses
Ãlayer_metrics
 Älayer_regularization_losses
H	variables
Â__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Åmetrics
Ænon_trainable_variables
Çlayers
Jtrainable_variables
Kregularization_losses
Èlayer_metrics
 Élayer_regularization_losses
L	variables
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object

Ê_query_dense
Ë
_key_dense
Ì_value_dense
Í_softmax
Î_dropout_layer
Ï_output_dense
Ðtrainable_variables
Ñregularization_losses
Ò	variables
Ó	keras_api
Û__call__
+Ü&call_and_return_all_conditional_losses"
_tf_keras_layerê{"class_name": "MultiHeadAttention", "name": "multi_head_attention_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multi_head_attention_7", "trainable": true, "dtype": "float32", "num_heads": 1, "key_dim": 32, "value_dim": 32, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}
¯
Ôlayer_with_weights-0
Ôlayer-0
Õlayer_with_weights-1
Õlayer-1
Ötrainable_variables
×regularization_losses
Ø	variables
Ù	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses"È
_tf_keras_sequential©{"class_name": "Sequential", "name": "sequential_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 66, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_23_input"}}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 66, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_23_input"}}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
ì
	Úaxis

gamma
	beta
Ûtrainable_variables
Üregularization_losses
Ý	variables
Þ	keras_api
ß__call__
+à&call_and_return_all_conditional_losses"µ
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_14", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
ì
	ßaxis

gamma
	beta
àtrainable_variables
áregularization_losses
â	variables
ã	keras_api
á__call__
+â&call_and_return_all_conditional_losses"µ
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_15", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
í
ätrainable_variables
åregularization_losses
æ	variables
ç	keras_api
ã__call__
+ä&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_20", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
í
ètrainable_variables
éregularization_losses
ê	variables
ë	keras_api
å__call__
+æ&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_21", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
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
µ
ìmetrics
ínon_trainable_variables
îlayers
Ttrainable_variables
Uregularization_losses
ïlayer_metrics
 ðlayer_regularization_losses
V	variables
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ñmetrics
ònon_trainable_variables
ólayers
Xtrainable_variables
Yregularization_losses
ôlayer_metrics
 õlayer_regularization_losses
Z	variables
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ömetrics
÷non_trainable_variables
ølayers
\trainable_variables
]regularization_losses
ùlayer_metrics
 úlayer_regularization_losses
^	variables
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
": 	È@2dense_25/kernel
:@2dense_25/bias
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
µ
ûmetrics
ünon_trainable_variables
ýlayers
btrainable_variables
cregularization_losses
þlayer_metrics
 ÿlayer_regularization_losses
d	variables
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
non_trainable_variables
layers
ftrainable_variables
gregularization_losses
layer_metrics
 layer_regularization_losses
h	variables
Î__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_26/kernel
:@2dense_26/bias
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
µ
metrics
non_trainable_variables
layers
ltrainable_variables
mregularization_losses
layer_metrics
 layer_regularization_losses
n	variables
Ð__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
non_trainable_variables
layers
ptrainable_variables
qregularization_losses
layer_metrics
 layer_regularization_losses
r	variables
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_27/kernel
:2dense_27/bias
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
µ
metrics
non_trainable_variables
layers
vtrainable_variables
wregularization_losses
layer_metrics
 layer_regularization_losses
x	variables
Ô__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
G:E 25token_and_position_embedding_3/embedding_6/embeddings
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
<
;0
<1
D2
E3"
trackable_list_wrapper
(
0"
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
trackable_dict_wrapper
 "
trackable_list_wrapper
'
~0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
~0"
trackable_list_wrapper
¸
metrics
non_trainable_variables
layers
trainable_variables
regularization_losses
layer_metrics
 layer_regularization_losses
	variables
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
¸
metrics
non_trainable_variables
layers
trainable_variables
regularization_losses
layer_metrics
 layer_regularization_losses
	variables
Ù__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
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
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
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
Ë
partial_output_shape
 full_output_shape
kernel
	bias
¡trainable_variables
¢regularization_losses
£	variables
¤	keras_api
ç__call__
+è&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "EinsumDense", "name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
Ç
¥partial_output_shape
¦full_output_shape
kernel
	bias
§trainable_variables
¨regularization_losses
©	variables
ª	keras_api
é__call__
+ê&call_and_return_all_conditional_losses"ç
_tf_keras_layerÍ{"class_name": "EinsumDense", "name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
Ë
«partial_output_shape
¬full_output_shape
kernel
	bias
­trainable_variables
®regularization_losses
¯	variables
°	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "EinsumDense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
ë
±trainable_variables
²regularization_losses
³	variables
´	keras_api
í__call__
+î&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}}
ç
µtrainable_variables
¶regularization_losses
·	variables
¸	keras_api
ï__call__
+ð&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
à
¹partial_output_shape
ºfull_output_shape
kernel
	bias
»trainable_variables
¼regularization_losses
½	variables
¾	keras_api
ñ__call__
+ò&call_and_return_all_conditional_losses"
_tf_keras_layeræ{"class_name": "EinsumDense", "name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 32], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 1, 32]}}
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
¸
¿metrics
Ànon_trainable_variables
Álayers
Ðtrainable_variables
Ñregularization_losses
Âlayer_metrics
 Ãlayer_regularization_losses
Ò	variables
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
þ
kernel
	bias
Ätrainable_variables
Åregularization_losses
Æ	variables
Ç	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "dense_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}

kernel
	bias
Ètrainable_variables
Éregularization_losses
Ê	variables
Ë	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses"Ó
_tf_keras_layer¹{"class_name": "Dense", "name": "dense_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 64]}}
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
¸
Ötrainable_variables
Ìnon_trainable_variables
Ímetrics
Îlayers
×regularization_losses
Ïlayer_metrics
 Ðlayer_regularization_losses
Ø	variables
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
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
Ñmetrics
Ònon_trainable_variables
Ólayers
Ûtrainable_variables
Üregularization_losses
Ôlayer_metrics
 Õlayer_regularization_losses
Ý	variables
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
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
Ömetrics
×non_trainable_variables
Ølayers
àtrainable_variables
áregularization_losses
Ùlayer_metrics
 Úlayer_regularization_losses
â	variables
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ûmetrics
Ünon_trainable_variables
Ýlayers
ätrainable_variables
åregularization_losses
Þlayer_metrics
 ßlayer_regularization_losses
æ	variables
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
¸
àmetrics
ánon_trainable_variables
âlayers
ètrainable_variables
éregularization_losses
ãlayer_metrics
 älayer_regularization_losses
ê	variables
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
_generic_user_object
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
¿

åtotal

æcount
ç	variables
è	keras_api"
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
émetrics
ênon_trainable_variables
ëlayers
¡trainable_variables
¢regularization_losses
ìlayer_metrics
 ílayer_regularization_losses
£	variables
ç__call__
+è&call_and_return_all_conditional_losses
'è"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
îmetrics
ïnon_trainable_variables
ðlayers
§trainable_variables
¨regularization_losses
ñlayer_metrics
 òlayer_regularization_losses
©	variables
é__call__
+ê&call_and_return_all_conditional_losses
'ê"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
ómetrics
ônon_trainable_variables
õlayers
­trainable_variables
®regularization_losses
ölayer_metrics
 ÷layer_regularization_losses
¯	variables
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ømetrics
ùnon_trainable_variables
úlayers
±trainable_variables
²regularization_losses
ûlayer_metrics
 ülayer_regularization_losses
³	variables
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
¸
ýmetrics
þnon_trainable_variables
ÿlayers
µtrainable_variables
¶regularization_losses
layer_metrics
 layer_regularization_losses
·	variables
ï__call__
+ð&call_and_return_all_conditional_losses
'ð"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
metrics
non_trainable_variables
layers
»trainable_variables
¼regularization_losses
layer_metrics
 layer_regularization_losses
½	variables
ñ__call__
+ò&call_and_return_all_conditional_losses
'ò"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
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
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
metrics
non_trainable_variables
layers
Ätrainable_variables
Åregularization_losses
layer_metrics
 layer_regularization_losses
Æ	variables
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses"
_generic_user_object
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
¸
metrics
non_trainable_variables
layers
Ètrainable_variables
Éregularization_losses
layer_metrics
 layer_regularization_losses
Ê	variables
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ô0
Õ1"
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
0:.  2SGD/conv1d_6/kernel/momentum
&:$ 2SGD/conv1d_6/bias/momentum
0:.	  2SGD/conv1d_7/kernel/momentum
&:$ 2SGD/conv1d_7/bias/momentum
4:2 2(SGD/batch_normalization_6/gamma/momentum
3:1 2'SGD/batch_normalization_6/beta/momentum
4:2 2(SGD/batch_normalization_7/gamma/momentum
3:1 2'SGD/batch_normalization_7/beta/momentum
-:+	È@2SGD/dense_25/kernel/momentum
&:$@2SGD/dense_25/bias/momentum
,:*@@2SGD/dense_26/kernel/momentum
&:$@2SGD/dense_26/bias/momentum
,:*@2SGD/dense_27/kernel/momentum
&:$2SGD/dense_27/bias/momentum
R:P 2BSGD/token_and_position_embedding_3/embedding_6/embeddings/momentum
T:R
  2BSGD/token_and_position_embedding_3/embedding_7/embeddings/momentum
X:V  2DSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentum
R:P 2BSGD/transformer_block_7/multi_head_attention_7/query/bias/momentum
V:T  2BSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentum
P:N 2@SGD/transformer_block_7/multi_head_attention_7/key/bias/momentum
X:V  2DSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentum
R:P 2BSGD/transformer_block_7/multi_head_attention_7/value/bias/momentum
c:a  2OSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentum
Y:W 2MSGD/transformer_block_7/multi_head_attention_7/attention_output/bias/momentum
,:* @2SGD/dense_23/kernel/momentum
&:$@2SGD/dense_23/bias/momentum
,:*@ 2SGD/dense_24/kernel/momentum
&:$ 2SGD/dense_24/bias/momentum
I:G 2=SGD/transformer_block_7/layer_normalization_14/gamma/momentum
H:F 2<SGD/transformer_block_7/layer_normalization_14/beta/momentum
I:G 2=SGD/transformer_block_7/layer_normalization_15/gamma/momentum
H:F 2<SGD/transformer_block_7/layer_normalization_15/beta/momentum
î2ë
(__inference_model_3_layer_call_fn_411089
(__inference_model_3_layer_call_fn_411806
(__inference_model_3_layer_call_fn_410917
(__inference_model_3_layer_call_fn_411884À
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
C__inference_model_3_layer_call_and_return_conditional_losses_411728
C__inference_model_3_layer_call_and_return_conditional_losses_411485
C__inference_model_3_layer_call_and_return_conditional_losses_410744
C__inference_model_3_layer_call_and_return_conditional_losses_410650À
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
!__inference__wrapped_model_409294à
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
ä2á
?__inference_token_and_position_embedding_3_layer_call_fn_411917
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
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_411908
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
)__inference_conv1d_6_layer_call_fn_411942¢
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
D__inference_conv1d_6_layer_call_and_return_conditional_losses_411933¢
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
4__inference_average_pooling1d_9_layer_call_fn_409309Ó
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
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_409303Ó
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
)__inference_conv1d_7_layer_call_fn_411967¢
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
D__inference_conv1d_7_layer_call_and_return_conditional_losses_411958¢
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
5__inference_average_pooling1d_10_layer_call_fn_409324Ó
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
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_409318Ó
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
5__inference_average_pooling1d_11_layer_call_fn_409339Ó
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
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_409333Ó
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
6__inference_batch_normalization_6_layer_call_fn_412036
6__inference_batch_normalization_6_layer_call_fn_412049
6__inference_batch_normalization_6_layer_call_fn_412118
6__inference_batch_normalization_6_layer_call_fn_412131´
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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412085
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412003
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412023
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412105´
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
6__inference_batch_normalization_7_layer_call_fn_412282
6__inference_batch_normalization_7_layer_call_fn_412213
6__inference_batch_normalization_7_layer_call_fn_412295
6__inference_batch_normalization_7_layer_call_fn_412200´
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412167
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412249
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412187
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412269´
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
&__inference_add_3_layer_call_fn_412307¢
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
A__inference_add_3_layer_call_and_return_conditional_losses_412301¢
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
4__inference_transformer_block_7_layer_call_fn_412619
4__inference_transformer_block_7_layer_call_fn_412656°
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
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_412582
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_412455°
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
*__inference_flatten_3_layer_call_fn_412667¢
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
E__inference_flatten_3_layer_call_and_return_conditional_losses_412662¢
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
.__inference_concatenate_3_layer_call_fn_412680¢
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
I__inference_concatenate_3_layer_call_and_return_conditional_losses_412674¢
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
)__inference_dense_25_layer_call_fn_412700¢
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
D__inference_dense_25_layer_call_and_return_conditional_losses_412691¢
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
+__inference_dropout_22_layer_call_fn_412727
+__inference_dropout_22_layer_call_fn_412722´
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
F__inference_dropout_22_layer_call_and_return_conditional_losses_412712
F__inference_dropout_22_layer_call_and_return_conditional_losses_412717´
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
)__inference_dense_26_layer_call_fn_412747¢
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
D__inference_dense_26_layer_call_and_return_conditional_losses_412738¢
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
+__inference_dropout_23_layer_call_fn_412769
+__inference_dropout_23_layer_call_fn_412774´
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
F__inference_dropout_23_layer_call_and_return_conditional_losses_412764
F__inference_dropout_23_layer_call_and_return_conditional_losses_412759´
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
)__inference_dense_27_layer_call_fn_412793¢
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
D__inference_dense_27_layer_call_and_return_conditional_losses_412784¢
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
$__inference_signature_wrapper_411175input_7input_8"
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
-__inference_sequential_7_layer_call_fn_409786
-__inference_sequential_7_layer_call_fn_412920
-__inference_sequential_7_layer_call_fn_409759
-__inference_sequential_7_layer_call_fn_412933À
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
H__inference_sequential_7_layer_call_and_return_conditional_losses_409731
H__inference_sequential_7_layer_call_and_return_conditional_losses_412850
H__inference_sequential_7_layer_call_and_return_conditional_losses_409717
H__inference_sequential_7_layer_call_and_return_conditional_losses_412907À
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
)__inference_dense_23_layer_call_fn_412973¢
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
D__inference_dense_23_layer_call_and_return_conditional_losses_412964¢
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
)__inference_dense_24_layer_call_fn_413012¢
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
D__inference_dense_24_layer_call_and_return_conditional_losses_413003¢
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
!__inference__wrapped_model_409294Ç4~ !*+<9;:EBDC`ajktuZ¢W
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
A__inference_add_3_layer_call_and_return_conditional_losses_412301b¢_
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
&__inference_add_3_layer_call_fn_412307b¢_
X¢U
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿB 
&#
inputs/1ÿÿÿÿÿÿÿÿÿB 
ª "ÿÿÿÿÿÿÿÿÿB Ù
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_409318E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 °
5__inference_average_pooling1d_10_layer_call_fn_409324wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÙ
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_409333E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 °
5__inference_average_pooling1d_11_layer_call_fn_409339wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØ
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_409303E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
4__inference_average_pooling1d_9_layer_call_fn_409309wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¿
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412003j;<9:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB 
 ¿
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412023j<9;:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB 
 Ñ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412085|;<9:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ñ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412105|<9;:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
6__inference_batch_normalization_6_layer_call_fn_412036];<9:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p
ª "ÿÿÿÿÿÿÿÿÿB 
6__inference_batch_normalization_6_layer_call_fn_412049]<9;:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p 
ª "ÿÿÿÿÿÿÿÿÿB ©
6__inference_batch_normalization_6_layer_call_fn_412118o;<9:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ©
6__inference_batch_normalization_6_layer_call_fn_412131o<9;:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ñ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412167|DEBC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ñ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412187|EBDC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ¿
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412249jDEBC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB 
 ¿
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412269jEBDC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB 
 ©
6__inference_batch_normalization_7_layer_call_fn_412200oDEBC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ©
6__inference_batch_normalization_7_layer_call_fn_412213oEBDC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
6__inference_batch_normalization_7_layer_call_fn_412282]DEBC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p
ª "ÿÿÿÿÿÿÿÿÿB 
6__inference_batch_normalization_7_layer_call_fn_412295]EBDC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p 
ª "ÿÿÿÿÿÿÿÿÿB Ó
I__inference_concatenate_3_layer_call_and_return_conditional_losses_412674[¢X
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
.__inference_concatenate_3_layer_call_fn_412680x[¢X
Q¢N
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿÀ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÈ°
D__inference_conv1d_6_layer_call_and_return_conditional_losses_411933h !5¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿ  
ª "+¢(
!
0ÿÿÿÿÿÿÿÿÿ  
 
)__inference_conv1d_6_layer_call_fn_411942[ !5¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿ  
ª "ÿÿÿÿÿÿÿÿÿ  ®
D__inference_conv1d_7_layer_call_and_return_conditional_losses_411958f*+4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_conv1d_7_layer_call_fn_411967Y*+4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ®
D__inference_dense_23_layer_call_and_return_conditional_losses_412964f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿB 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB@
 
)__inference_dense_23_layer_call_fn_412973Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿB 
ª "ÿÿÿÿÿÿÿÿÿB@®
D__inference_dense_24_layer_call_and_return_conditional_losses_413003f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿB@
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB 
 
)__inference_dense_24_layer_call_fn_413012Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿB@
ª "ÿÿÿÿÿÿÿÿÿB ¥
D__inference_dense_25_layer_call_and_return_conditional_losses_412691]`a0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÈ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 }
)__inference_dense_25_layer_call_fn_412700P`a0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÈ
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dense_26_layer_call_and_return_conditional_losses_412738\jk/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
)__inference_dense_26_layer_call_fn_412747Ojk/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dense_27_layer_call_and_return_conditional_losses_412784\tu/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_27_layer_call_fn_412793Otu/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dropout_22_layer_call_and_return_conditional_losses_412712\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¦
F__inference_dropout_22_layer_call_and_return_conditional_losses_412717\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ~
+__inference_dropout_22_layer_call_fn_412722O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@~
+__inference_dropout_22_layer_call_fn_412727O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@¦
F__inference_dropout_23_layer_call_and_return_conditional_losses_412759\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¦
F__inference_dropout_23_layer_call_and_return_conditional_losses_412764\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ~
+__inference_dropout_23_layer_call_fn_412769O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@~
+__inference_dropout_23_layer_call_fn_412774O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@¦
E__inference_flatten_3_layer_call_and_return_conditional_losses_412662]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿB 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÀ
 ~
*__inference_flatten_3_layer_call_fn_412667P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿB 
ª "ÿÿÿÿÿÿÿÿÿÀ
C__inference_model_3_layer_call_and_return_conditional_losses_410650Á4~ !*+;<9:DEBC`ajktub¢_
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
 
C__inference_model_3_layer_call_and_return_conditional_losses_410744Á4~ !*+<9;:EBDC`ajktub¢_
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
 
C__inference_model_3_layer_call_and_return_conditional_losses_411485Ã4~ !*+;<9:DEBC`ajktud¢a
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
 
C__inference_model_3_layer_call_and_return_conditional_losses_411728Ã4~ !*+<9;:EBDC`ajktud¢a
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
 á
(__inference_model_3_layer_call_fn_410917´4~ !*+;<9:DEBC`ajktub¢_
X¢U
KH
# 
input_7ÿÿÿÿÿÿÿÿÿ 
!
input_8ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿá
(__inference_model_3_layer_call_fn_411089´4~ !*+<9;:EBDC`ajktub¢_
X¢U
KH
# 
input_7ÿÿÿÿÿÿÿÿÿ 
!
input_8ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿã
(__inference_model_3_layer_call_fn_411806¶4~ !*+;<9:DEBC`ajktud¢a
Z¢W
MJ
$!
inputs/0ÿÿÿÿÿÿÿÿÿ 
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿã
(__inference_model_3_layer_call_fn_411884¶4~ !*+<9;:EBDC`ajktud¢a
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
H__inference_sequential_7_layer_call_and_return_conditional_losses_409717zC¢@
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
H__inference_sequential_7_layer_call_and_return_conditional_losses_409731zC¢@
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
H__inference_sequential_7_layer_call_and_return_conditional_losses_412850r;¢8
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
H__inference_sequential_7_layer_call_and_return_conditional_losses_412907r;¢8
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
-__inference_sequential_7_layer_call_fn_409759mC¢@
9¢6
,)
dense_23_inputÿÿÿÿÿÿÿÿÿB 
p

 
ª "ÿÿÿÿÿÿÿÿÿB 
-__inference_sequential_7_layer_call_fn_409786mC¢@
9¢6
,)
dense_23_inputÿÿÿÿÿÿÿÿÿB 
p 

 
ª "ÿÿÿÿÿÿÿÿÿB 
-__inference_sequential_7_layer_call_fn_412920e;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿB 
p

 
ª "ÿÿÿÿÿÿÿÿÿB 
-__inference_sequential_7_layer_call_fn_412933e;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿB 
p 

 
ª "ÿÿÿÿÿÿÿÿÿB 
$__inference_signature_wrapper_411175Ø4~ !*+<9;:EBDC`ajktuk¢h
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
dense_27ÿÿÿÿÿÿÿÿÿ½
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_411908_~,¢)
"¢

xÿÿÿÿÿÿÿÿÿ 
ª "+¢(
!
0ÿÿÿÿÿÿÿÿÿ  
 
?__inference_token_and_position_embedding_3_layer_call_fn_411917R~,¢)
"¢

xÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ  Ú
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_412455 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB 
 Ú
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_412582 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿB 
 ±
4__inference_transformer_block_7_layer_call_fn_412619y 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p
ª "ÿÿÿÿÿÿÿÿÿB ±
4__inference_transformer_block_7_layer_call_fn_412656y 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿB 
p 
ª "ÿÿÿÿÿÿÿÿÿB 