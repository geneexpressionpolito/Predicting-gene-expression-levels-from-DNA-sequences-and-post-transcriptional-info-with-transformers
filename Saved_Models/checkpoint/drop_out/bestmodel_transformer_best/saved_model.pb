Ã0
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
 "serve*2.4.12v2.4.1-0-g85c8b2a817f8 *

conv1d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv1d_10/kernel
y
$conv1d_10/kernel/Read/ReadVariableOpReadVariableOpconv1d_10/kernel*"
_output_shapes
:  *
dtype0
t
conv1d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_10/bias
m
"conv1d_10/bias/Read/ReadVariableOpReadVariableOpconv1d_10/bias*
_output_shapes
: *
dtype0

conv1d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *!
shared_nameconv1d_11/kernel
y
$conv1d_11/kernel/Read/ReadVariableOpReadVariableOpconv1d_11/kernel*"
_output_shapes
:	  *
dtype0
t
conv1d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_11/bias
m
"conv1d_11/bias/Read/ReadVariableOpReadVariableOpconv1d_11/bias*
_output_shapes
: *
dtype0

batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_10/gamma

0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
: *
dtype0

batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_10/beta

/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
: *
dtype0

"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_10/moving_mean

6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
: *
dtype0
¤
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_10/moving_variance

:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
: *
dtype0

batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_11/gamma

0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes
: *
dtype0

batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_11/beta

/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes
: *
dtype0

"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_11/moving_mean

6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes
: *
dtype0
¤
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_11/moving_variance

:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes
: *
dtype0
{
dense_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	è@* 
shared_namedense_39/kernel
t
#dense_39/kernel/Read/ReadVariableOpReadVariableOpdense_39/kernel*
_output_shapes
:	è@*
dtype0
r
dense_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_39/bias
k
!dense_39/bias/Read/ReadVariableOpReadVariableOpdense_39/bias*
_output_shapes
:@*
dtype0
z
dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_40/kernel
s
#dense_40/kernel/Read/ReadVariableOpReadVariableOpdense_40/kernel*
_output_shapes

:@@*
dtype0
r
dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_40/bias
k
!dense_40/bias/Read/ReadVariableOpReadVariableOpdense_40/bias*
_output_shapes
:@*
dtype0
z
dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_41/kernel
s
#dense_41/kernel/Read/ReadVariableOpReadVariableOpdense_41/kernel*
_output_shapes

:@*
dtype0
r
dense_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_41/bias
k
!dense_41/bias/Read/ReadVariableOpReadVariableOpdense_41/bias*
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
È
6token_and_position_embedding_5/embedding_10/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *G
shared_name86token_and_position_embedding_5/embedding_10/embeddings
Á
Jtoken_and_position_embedding_5/embedding_10/embeddings/Read/ReadVariableOpReadVariableOp6token_and_position_embedding_5/embedding_10/embeddings*
_output_shapes

: *
dtype0
É
6token_and_position_embedding_5/embedding_11/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	R *G
shared_name86token_and_position_embedding_5/embedding_11/embeddings
Â
Jtoken_and_position_embedding_5/embedding_11/embeddings/Read/ReadVariableOpReadVariableOp6token_and_position_embedding_5/embedding_11/embeddings*
_output_shapes
:	R *
dtype0
Ò
9transformer_block_11/multi_head_attention_11/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *J
shared_name;9transformer_block_11/multi_head_attention_11/query/kernel
Ë
Mtransformer_block_11/multi_head_attention_11/query/kernel/Read/ReadVariableOpReadVariableOp9transformer_block_11/multi_head_attention_11/query/kernel*"
_output_shapes
:  *
dtype0
Ê
7transformer_block_11/multi_head_attention_11/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *H
shared_name97transformer_block_11/multi_head_attention_11/query/bias
Ã
Ktransformer_block_11/multi_head_attention_11/query/bias/Read/ReadVariableOpReadVariableOp7transformer_block_11/multi_head_attention_11/query/bias*
_output_shapes

: *
dtype0
Î
7transformer_block_11/multi_head_attention_11/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_11/multi_head_attention_11/key/kernel
Ç
Ktransformer_block_11/multi_head_attention_11/key/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_11/multi_head_attention_11/key/kernel*"
_output_shapes
:  *
dtype0
Æ
5transformer_block_11/multi_head_attention_11/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_11/multi_head_attention_11/key/bias
¿
Itransformer_block_11/multi_head_attention_11/key/bias/Read/ReadVariableOpReadVariableOp5transformer_block_11/multi_head_attention_11/key/bias*
_output_shapes

: *
dtype0
Ò
9transformer_block_11/multi_head_attention_11/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *J
shared_name;9transformer_block_11/multi_head_attention_11/value/kernel
Ë
Mtransformer_block_11/multi_head_attention_11/value/kernel/Read/ReadVariableOpReadVariableOp9transformer_block_11/multi_head_attention_11/value/kernel*"
_output_shapes
:  *
dtype0
Ê
7transformer_block_11/multi_head_attention_11/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *H
shared_name97transformer_block_11/multi_head_attention_11/value/bias
Ã
Ktransformer_block_11/multi_head_attention_11/value/bias/Read/ReadVariableOpReadVariableOp7transformer_block_11/multi_head_attention_11/value/bias*
_output_shapes

: *
dtype0
è
Dtransformer_block_11/multi_head_attention_11/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDtransformer_block_11/multi_head_attention_11/attention_output/kernel
á
Xtransformer_block_11/multi_head_attention_11/attention_output/kernel/Read/ReadVariableOpReadVariableOpDtransformer_block_11/multi_head_attention_11/attention_output/kernel*"
_output_shapes
:  *
dtype0
Ü
Btransformer_block_11/multi_head_attention_11/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *S
shared_nameDBtransformer_block_11/multi_head_attention_11/attention_output/bias
Õ
Vtransformer_block_11/multi_head_attention_11/attention_output/bias/Read/ReadVariableOpReadVariableOpBtransformer_block_11/multi_head_attention_11/attention_output/bias*
_output_shapes
: *
dtype0
z
dense_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @* 
shared_namedense_37/kernel
s
#dense_37/kernel/Read/ReadVariableOpReadVariableOpdense_37/kernel*
_output_shapes

: @*
dtype0
r
dense_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_37/bias
k
!dense_37/bias/Read/ReadVariableOpReadVariableOpdense_37/bias*
_output_shapes
:@*
dtype0
z
dense_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_38/kernel
s
#dense_38/kernel/Read/ReadVariableOpReadVariableOpdense_38/kernel*
_output_shapes

:@ *
dtype0
r
dense_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_38/bias
k
!dense_38/bias/Read/ReadVariableOpReadVariableOpdense_38/bias*
_output_shapes
: *
dtype0
º
1transformer_block_11/layer_normalization_22/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31transformer_block_11/layer_normalization_22/gamma
³
Etransformer_block_11/layer_normalization_22/gamma/Read/ReadVariableOpReadVariableOp1transformer_block_11/layer_normalization_22/gamma*
_output_shapes
: *
dtype0
¸
0transformer_block_11/layer_normalization_22/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20transformer_block_11/layer_normalization_22/beta
±
Dtransformer_block_11/layer_normalization_22/beta/Read/ReadVariableOpReadVariableOp0transformer_block_11/layer_normalization_22/beta*
_output_shapes
: *
dtype0
º
1transformer_block_11/layer_normalization_23/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31transformer_block_11/layer_normalization_23/gamma
³
Etransformer_block_11/layer_normalization_23/gamma/Read/ReadVariableOpReadVariableOp1transformer_block_11/layer_normalization_23/gamma*
_output_shapes
: *
dtype0
¸
0transformer_block_11/layer_normalization_23/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20transformer_block_11/layer_normalization_23/beta
±
Dtransformer_block_11/layer_normalization_23/beta/Read/ReadVariableOpReadVariableOp0transformer_block_11/layer_normalization_23/beta*
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

SGD/conv1d_10/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *.
shared_nameSGD/conv1d_10/kernel/momentum

1SGD/conv1d_10/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_10/kernel/momentum*"
_output_shapes
:  *
dtype0

SGD/conv1d_10/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameSGD/conv1d_10/bias/momentum

/SGD/conv1d_10/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_10/bias/momentum*
_output_shapes
: *
dtype0

SGD/conv1d_11/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *.
shared_nameSGD/conv1d_11/kernel/momentum

1SGD/conv1d_11/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_11/kernel/momentum*"
_output_shapes
:	  *
dtype0

SGD/conv1d_11/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameSGD/conv1d_11/bias/momentum

/SGD/conv1d_11/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_11/bias/momentum*
_output_shapes
: *
dtype0
ª
)SGD/batch_normalization_10/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)SGD/batch_normalization_10/gamma/momentum
£
=SGD/batch_normalization_10/gamma/momentum/Read/ReadVariableOpReadVariableOp)SGD/batch_normalization_10/gamma/momentum*
_output_shapes
: *
dtype0
¨
(SGD/batch_normalization_10/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(SGD/batch_normalization_10/beta/momentum
¡
<SGD/batch_normalization_10/beta/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_10/beta/momentum*
_output_shapes
: *
dtype0
ª
)SGD/batch_normalization_11/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)SGD/batch_normalization_11/gamma/momentum
£
=SGD/batch_normalization_11/gamma/momentum/Read/ReadVariableOpReadVariableOp)SGD/batch_normalization_11/gamma/momentum*
_output_shapes
: *
dtype0
¨
(SGD/batch_normalization_11/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(SGD/batch_normalization_11/beta/momentum
¡
<SGD/batch_normalization_11/beta/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_11/beta/momentum*
_output_shapes
: *
dtype0

SGD/dense_39/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	è@*-
shared_nameSGD/dense_39/kernel/momentum

0SGD/dense_39/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_39/kernel/momentum*
_output_shapes
:	è@*
dtype0

SGD/dense_39/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_39/bias/momentum

.SGD/dense_39/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_39/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_40/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*-
shared_nameSGD/dense_40/kernel/momentum

0SGD/dense_40/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_40/kernel/momentum*
_output_shapes

:@@*
dtype0

SGD/dense_40/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_40/bias/momentum

.SGD/dense_40/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_40/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_41/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*-
shared_nameSGD/dense_41/kernel/momentum

0SGD/dense_41/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_41/kernel/momentum*
_output_shapes

:@*
dtype0

SGD/dense_41/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/dense_41/bias/momentum

.SGD/dense_41/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_41/bias/momentum*
_output_shapes
:*
dtype0
â
CSGD/token_and_position_embedding_5/embedding_10/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *T
shared_nameECSGD/token_and_position_embedding_5/embedding_10/embeddings/momentum
Û
WSGD/token_and_position_embedding_5/embedding_10/embeddings/momentum/Read/ReadVariableOpReadVariableOpCSGD/token_and_position_embedding_5/embedding_10/embeddings/momentum*
_output_shapes

: *
dtype0
ã
CSGD/token_and_position_embedding_5/embedding_11/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	R *T
shared_nameECSGD/token_and_position_embedding_5/embedding_11/embeddings/momentum
Ü
WSGD/token_and_position_embedding_5/embedding_11/embeddings/momentum/Read/ReadVariableOpReadVariableOpCSGD/token_and_position_embedding_5/embedding_11/embeddings/momentum*
_output_shapes
:	R *
dtype0
ì
FSGD/transformer_block_11/multi_head_attention_11/query/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *W
shared_nameHFSGD/transformer_block_11/multi_head_attention_11/query/kernel/momentum
å
ZSGD/transformer_block_11/multi_head_attention_11/query/kernel/momentum/Read/ReadVariableOpReadVariableOpFSGD/transformer_block_11/multi_head_attention_11/query/kernel/momentum*"
_output_shapes
:  *
dtype0
ä
DSGD/transformer_block_11/multi_head_attention_11/query/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *U
shared_nameFDSGD/transformer_block_11/multi_head_attention_11/query/bias/momentum
Ý
XSGD/transformer_block_11/multi_head_attention_11/query/bias/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_11/multi_head_attention_11/query/bias/momentum*
_output_shapes

: *
dtype0
è
DSGD/transformer_block_11/multi_head_attention_11/key/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_11/multi_head_attention_11/key/kernel/momentum
á
XSGD/transformer_block_11/multi_head_attention_11/key/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_11/multi_head_attention_11/key/kernel/momentum*"
_output_shapes
:  *
dtype0
à
BSGD/transformer_block_11/multi_head_attention_11/key/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_11/multi_head_attention_11/key/bias/momentum
Ù
VSGD/transformer_block_11/multi_head_attention_11/key/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_11/multi_head_attention_11/key/bias/momentum*
_output_shapes

: *
dtype0
ì
FSGD/transformer_block_11/multi_head_attention_11/value/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *W
shared_nameHFSGD/transformer_block_11/multi_head_attention_11/value/kernel/momentum
å
ZSGD/transformer_block_11/multi_head_attention_11/value/kernel/momentum/Read/ReadVariableOpReadVariableOpFSGD/transformer_block_11/multi_head_attention_11/value/kernel/momentum*"
_output_shapes
:  *
dtype0
ä
DSGD/transformer_block_11/multi_head_attention_11/value/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *U
shared_nameFDSGD/transformer_block_11/multi_head_attention_11/value/bias/momentum
Ý
XSGD/transformer_block_11/multi_head_attention_11/value/bias/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_11/multi_head_attention_11/value/bias/momentum*
_output_shapes

: *
dtype0

QSGD/transformer_block_11/multi_head_attention_11/attention_output/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *b
shared_nameSQSGD/transformer_block_11/multi_head_attention_11/attention_output/kernel/momentum
û
eSGD/transformer_block_11/multi_head_attention_11/attention_output/kernel/momentum/Read/ReadVariableOpReadVariableOpQSGD/transformer_block_11/multi_head_attention_11/attention_output/kernel/momentum*"
_output_shapes
:  *
dtype0
ö
OSGD/transformer_block_11/multi_head_attention_11/attention_output/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *`
shared_nameQOSGD/transformer_block_11/multi_head_attention_11/attention_output/bias/momentum
ï
cSGD/transformer_block_11/multi_head_attention_11/attention_output/bias/momentum/Read/ReadVariableOpReadVariableOpOSGD/transformer_block_11/multi_head_attention_11/attention_output/bias/momentum*
_output_shapes
: *
dtype0

SGD/dense_37/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*-
shared_nameSGD/dense_37/kernel/momentum

0SGD/dense_37/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_37/kernel/momentum*
_output_shapes

: @*
dtype0

SGD/dense_37/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_37/bias/momentum

.SGD/dense_37/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_37/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_38/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *-
shared_nameSGD/dense_38/kernel/momentum

0SGD/dense_38/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_38/kernel/momentum*
_output_shapes

:@ *
dtype0

SGD/dense_38/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/dense_38/bias/momentum

.SGD/dense_38/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_38/bias/momentum*
_output_shapes
: *
dtype0
Ô
>SGD/transformer_block_11/layer_normalization_22/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>SGD/transformer_block_11/layer_normalization_22/gamma/momentum
Í
RSGD/transformer_block_11/layer_normalization_22/gamma/momentum/Read/ReadVariableOpReadVariableOp>SGD/transformer_block_11/layer_normalization_22/gamma/momentum*
_output_shapes
: *
dtype0
Ò
=SGD/transformer_block_11/layer_normalization_22/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=SGD/transformer_block_11/layer_normalization_22/beta/momentum
Ë
QSGD/transformer_block_11/layer_normalization_22/beta/momentum/Read/ReadVariableOpReadVariableOp=SGD/transformer_block_11/layer_normalization_22/beta/momentum*
_output_shapes
: *
dtype0
Ô
>SGD/transformer_block_11/layer_normalization_23/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>SGD/transformer_block_11/layer_normalization_23/gamma/momentum
Í
RSGD/transformer_block_11/layer_normalization_23/gamma/momentum/Read/ReadVariableOpReadVariableOp>SGD/transformer_block_11/layer_normalization_23/gamma/momentum*
_output_shapes
: *
dtype0
Ò
=SGD/transformer_block_11/layer_normalization_23/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=SGD/transformer_block_11/layer_normalization_23/beta/momentum
Ë
QSGD/transformer_block_11/layer_normalization_23/beta/momentum/Read/ReadVariableOpReadVariableOp=SGD/transformer_block_11/layer_normalization_23/beta/momentum*
_output_shapes
: *
dtype0

NoOpNoOp
ô³
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*®³
value£³B³ B³
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
regularization_losses
trainable_variables
	keras_api

signatures
 
n
	token_emb
pos_emb
	variables
regularization_losses
trainable_variables
	keras_api
h

 kernel
!bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
R
&	variables
'regularization_losses
(trainable_variables
)	keras_api
h

*kernel
+bias
,	variables
-regularization_losses
.trainable_variables
/	keras_api
R
0	variables
1regularization_losses
2trainable_variables
3	keras_api
R
4	variables
5regularization_losses
6trainable_variables
7	keras_api

8axis
	9gamma
:beta
;moving_mean
<moving_variance
=	variables
>regularization_losses
?trainable_variables
@	keras_api

Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
R
J	variables
Kregularization_losses
Ltrainable_variables
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
Uregularization_losses
Vtrainable_variables
W	keras_api
R
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
 
R
\	variables
]regularization_losses
^trainable_variables
_	keras_api
h

`kernel
abias
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
R
f	variables
gregularization_losses
htrainable_variables
i	keras_api
h

jkernel
kbias
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
R
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
h

tkernel
ubias
v	variables
wregularization_losses
xtrainable_variables
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
 
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
²
layers
layer_metrics
	variables
non_trainable_variables
metrics
 layer_regularization_losses
regularization_losses
trainable_variables
 
f
~
embeddings
	variables
regularization_losses
trainable_variables
	keras_api
f

embeddings
	variables
regularization_losses
trainable_variables
	keras_api

~0
1
 

~0
1
²
layers
layer_metrics
non_trainable_variables
 metrics
	variables
 ¡layer_regularization_losses
regularization_losses
trainable_variables
\Z
VARIABLE_VALUEconv1d_10/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_10/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
²
¢layers
£layer_metrics
¤non_trainable_variables
¥metrics
"	variables
 ¦layer_regularization_losses
#regularization_losses
$trainable_variables
 
 
 
²
§layers
¨layer_metrics
©non_trainable_variables
ªmetrics
&	variables
 «layer_regularization_losses
'regularization_losses
(trainable_variables
\Z
VARIABLE_VALUEconv1d_11/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_11/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1
 

*0
+1
²
¬layers
­layer_metrics
®non_trainable_variables
¯metrics
,	variables
 °layer_regularization_losses
-regularization_losses
.trainable_variables
 
 
 
²
±layers
²layer_metrics
³non_trainable_variables
´metrics
0	variables
 µlayer_regularization_losses
1regularization_losses
2trainable_variables
 
 
 
²
¶layers
·layer_metrics
¸non_trainable_variables
¹metrics
4	variables
 ºlayer_regularization_losses
5regularization_losses
6trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_10/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_10/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_10/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_10/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

90
:1
;2
<3
 

90
:1
²
»layers
¼layer_metrics
½non_trainable_variables
¾metrics
=	variables
 ¿layer_regularization_losses
>regularization_losses
?trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_11/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_11/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_11/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_11/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
D2
E3
 

B0
C1
²
Àlayers
Álayer_metrics
Ânon_trainable_variables
Ãmetrics
F	variables
 Älayer_regularization_losses
Gregularization_losses
Htrainable_variables
 
 
 
²
Ålayers
Ælayer_metrics
Çnon_trainable_variables
Èmetrics
J	variables
 Élayer_regularization_losses
Kregularization_losses
Ltrainable_variables
Å
Ê_query_dense
Ë
_key_dense
Ì_value_dense
Í_softmax
Î_dropout_layer
Ï_output_dense
Ð	variables
Ñregularization_losses
Òtrainable_variables
Ó	keras_api
¨
Ôlayer_with_weights-0
Ôlayer-0
Õlayer_with_weights-1
Õlayer-1
Ö	variables
×regularization_losses
Øtrainable_variables
Ù	keras_api
x
	Úaxis

gamma
	beta
Û	variables
Üregularization_losses
Ýtrainable_variables
Þ	keras_api
x
	ßaxis

gamma
	beta
à	variables
áregularization_losses
âtrainable_variables
ã	keras_api
V
ä	variables
åregularization_losses
ætrainable_variables
ç	keras_api
V
è	variables
éregularization_losses
êtrainable_variables
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
ìlayers
ílayer_metrics
înon_trainable_variables
ïmetrics
T	variables
 ðlayer_regularization_losses
Uregularization_losses
Vtrainable_variables
 
 
 
²
ñlayers
òlayer_metrics
ónon_trainable_variables
ômetrics
X	variables
 õlayer_regularization_losses
Yregularization_losses
Ztrainable_variables
 
 
 
²
ölayers
÷layer_metrics
ønon_trainable_variables
ùmetrics
\	variables
 úlayer_regularization_losses
]regularization_losses
^trainable_variables
[Y
VARIABLE_VALUEdense_39/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_39/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

`0
a1
 

`0
a1
²
ûlayers
ülayer_metrics
ýnon_trainable_variables
þmetrics
b	variables
 ÿlayer_regularization_losses
cregularization_losses
dtrainable_variables
 
 
 
²
layers
layer_metrics
non_trainable_variables
metrics
f	variables
 layer_regularization_losses
gregularization_losses
htrainable_variables
[Y
VARIABLE_VALUEdense_40/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_40/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

j0
k1
 

j0
k1
²
layers
layer_metrics
non_trainable_variables
metrics
l	variables
 layer_regularization_losses
mregularization_losses
ntrainable_variables
 
 
 
²
layers
layer_metrics
non_trainable_variables
metrics
p	variables
 layer_regularization_losses
qregularization_losses
rtrainable_variables
[Y
VARIABLE_VALUEdense_41/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_41/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

t0
u1
 

t0
u1
²
layers
layer_metrics
non_trainable_variables
metrics
v	variables
 layer_regularization_losses
wregularization_losses
xtrainable_variables
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6token_and_position_embedding_5/embedding_10/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6token_and_position_embedding_5/embedding_11/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9transformer_block_11/multi_head_attention_11/query/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7transformer_block_11/multi_head_attention_11/query/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7transformer_block_11/multi_head_attention_11/key/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_11/multi_head_attention_11/key/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9transformer_block_11/multi_head_attention_11/value/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7transformer_block_11/multi_head_attention_11/value/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEDtransformer_block_11/multi_head_attention_11/attention_output/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEBtransformer_block_11/multi_head_attention_11/attention_output/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_37/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_37/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_38/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_38/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1transformer_block_11/layer_normalization_22/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0transformer_block_11/layer_normalization_22/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1transformer_block_11/layer_normalization_23/gamma'variables/28/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0transformer_block_11/layer_normalization_23/beta'variables/29/.ATTRIBUTES/VARIABLE_VALUE
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

;0
<1
D2
E3

0
 

~0
 

~0
µ
layers
layer_metrics
non_trainable_variables
metrics
	variables
 layer_regularization_losses
regularization_losses
trainable_variables

0
 

0
µ
layers
layer_metrics
non_trainable_variables
metrics
	variables
 layer_regularization_losses
regularization_losses
trainable_variables
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
¡
partial_output_shape
 full_output_shape
kernel
	bias
¡	variables
¢regularization_losses
£trainable_variables
¤	keras_api
¡
¥partial_output_shape
¦full_output_shape
kernel
	bias
§	variables
¨regularization_losses
©trainable_variables
ª	keras_api
¡
«partial_output_shape
¬full_output_shape
kernel
	bias
­	variables
®regularization_losses
¯trainable_variables
°	keras_api
V
±	variables
²regularization_losses
³trainable_variables
´	keras_api
V
µ	variables
¶regularization_losses
·trainable_variables
¸	keras_api
¡
¹partial_output_shape
ºfull_output_shape
kernel
	bias
»	variables
¼regularization_losses
½trainable_variables
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
¿layers
Àlayer_metrics
Ánon_trainable_variables
Âmetrics
Ð	variables
 Ãlayer_regularization_losses
Ñregularization_losses
Òtrainable_variables
n
kernel
	bias
Ä	variables
Åregularization_losses
Ætrainable_variables
Ç	keras_api
n
kernel
	bias
È	variables
Éregularization_losses
Êtrainable_variables
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
Ìlayers
Ílayer_metrics
Ö	variables
Înon_trainable_variables
Ïmetrics
 Ðlayer_regularization_losses
×regularization_losses
Øtrainable_variables
 

0
1
 

0
1
µ
Ñlayers
Òlayer_metrics
Ónon_trainable_variables
Ômetrics
Û	variables
 Õlayer_regularization_losses
Üregularization_losses
Ýtrainable_variables
 

0
1
 

0
1
µ
Ölayers
×layer_metrics
Ønon_trainable_variables
Ùmetrics
à	variables
 Úlayer_regularization_losses
áregularization_losses
âtrainable_variables
 
 
 
µ
Ûlayers
Ülayer_metrics
Ýnon_trainable_variables
Þmetrics
ä	variables
 ßlayer_regularization_losses
åregularization_losses
ætrainable_variables
 
 
 
µ
àlayers
álayer_metrics
ânon_trainable_variables
ãmetrics
è	variables
 älayer_regularization_losses
éregularization_losses
êtrainable_variables
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
élayers
êlayer_metrics
ënon_trainable_variables
ìmetrics
¡	variables
 ílayer_regularization_losses
¢regularization_losses
£trainable_variables
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
îlayers
ïlayer_metrics
ðnon_trainable_variables
ñmetrics
§	variables
 òlayer_regularization_losses
¨regularization_losses
©trainable_variables
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
ólayers
ôlayer_metrics
õnon_trainable_variables
ömetrics
­	variables
 ÷layer_regularization_losses
®regularization_losses
¯trainable_variables
 
 
 
µ
ølayers
ùlayer_metrics
únon_trainable_variables
ûmetrics
±	variables
 ülayer_regularization_losses
²regularization_losses
³trainable_variables
 
 
 
µ
ýlayers
þlayer_metrics
ÿnon_trainable_variables
metrics
µ	variables
 layer_regularization_losses
¶regularization_losses
·trainable_variables
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
layers
layer_metrics
non_trainable_variables
metrics
»	variables
 layer_regularization_losses
¼regularization_losses
½trainable_variables
0
Ê0
Ë1
Ì2
Í3
Î4
Ï5
 
 
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
layers
layer_metrics
non_trainable_variables
metrics
Ä	variables
 layer_regularization_losses
Åregularization_losses
Ætrainable_variables

0
1
 

0
1
µ
layers
layer_metrics
non_trainable_variables
metrics
È	variables
 layer_regularization_losses
Éregularization_losses
Êtrainable_variables
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

VARIABLE_VALUESGD/conv1d_10/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv1d_10/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv1d_11/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv1d_11/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)SGD/batch_normalization_10/gamma/momentumXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/batch_normalization_10/beta/momentumWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)SGD/batch_normalization_11/gamma/momentumXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/batch_normalization_11/beta/momentumWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_39/kernel/momentumYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_39/bias/momentumWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_40/kernel/momentumYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_40/bias/momentumWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_41/kernel/momentumYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_41/bias/momentumWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUECSGD/token_and_position_embedding_5/embedding_10/embeddings/momentumIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUECSGD/token_and_position_embedding_5/embedding_11/embeddings/momentumIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
§¤
VARIABLE_VALUEFSGD/transformer_block_11/multi_head_attention_11/query/kernel/momentumJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¥¢
VARIABLE_VALUEDSGD/transformer_block_11/multi_head_attention_11/query/bias/momentumJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¥¢
VARIABLE_VALUEDSGD/transformer_block_11/multi_head_attention_11/key/kernel/momentumJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEBSGD/transformer_block_11/multi_head_attention_11/key/bias/momentumJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
§¤
VARIABLE_VALUEFSGD/transformer_block_11/multi_head_attention_11/value/kernel/momentumJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¥¢
VARIABLE_VALUEDSGD/transformer_block_11/multi_head_attention_11/value/bias/momentumJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
²¯
VARIABLE_VALUEQSGD/transformer_block_11/multi_head_attention_11/attention_output/kernel/momentumJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
°­
VARIABLE_VALUEOSGD/transformer_block_11/multi_head_attention_11/attention_output/bias/momentumJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUESGD/dense_37/kernel/momentumJvariables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUESGD/dense_37/bias/momentumJvariables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUESGD/dense_38/kernel/momentumJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUESGD/dense_38/bias/momentumJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>SGD/transformer_block_11/layer_normalization_22/gamma/momentumJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=SGD/transformer_block_11/layer_normalization_22/beta/momentumJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>SGD/transformer_block_11/layer_normalization_23/gamma/momentumJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=SGD/transformer_block_11/layer_normalization_23/beta/momentumJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
}
serving_default_input_11Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿR
{
serving_default_input_12Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
§
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_11serving_default_input_126token_and_position_embedding_5/embedding_11/embeddings6token_and_position_embedding_5/embedding_10/embeddingsconv1d_10/kernelconv1d_10/biasconv1d_11/kernelconv1d_11/bias&batch_normalization_10/moving_variancebatch_normalization_10/gamma"batch_normalization_10/moving_meanbatch_normalization_10/beta&batch_normalization_11/moving_variancebatch_normalization_11/gamma"batch_normalization_11/moving_meanbatch_normalization_11/beta9transformer_block_11/multi_head_attention_11/query/kernel7transformer_block_11/multi_head_attention_11/query/bias7transformer_block_11/multi_head_attention_11/key/kernel5transformer_block_11/multi_head_attention_11/key/bias9transformer_block_11/multi_head_attention_11/value/kernel7transformer_block_11/multi_head_attention_11/value/biasDtransformer_block_11/multi_head_attention_11/attention_output/kernelBtransformer_block_11/multi_head_attention_11/attention_output/bias1transformer_block_11/layer_normalization_22/gamma0transformer_block_11/layer_normalization_22/betadense_37/kerneldense_37/biasdense_38/kerneldense_38/bias1transformer_block_11/layer_normalization_23/gamma0transformer_block_11/layer_normalization_23/betadense_39/kerneldense_39/biasdense_40/kerneldense_40/biasdense_41/kerneldense_41/bias*1
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
$__inference_signature_wrapper_765432
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
%
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv1d_10/kernel/Read/ReadVariableOp"conv1d_10/bias/Read/ReadVariableOp$conv1d_11/kernel/Read/ReadVariableOp"conv1d_11/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp#dense_39/kernel/Read/ReadVariableOp!dense_39/bias/Read/ReadVariableOp#dense_40/kernel/Read/ReadVariableOp!dense_40/bias/Read/ReadVariableOp#dense_41/kernel/Read/ReadVariableOp!dense_41/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpJtoken_and_position_embedding_5/embedding_10/embeddings/Read/ReadVariableOpJtoken_and_position_embedding_5/embedding_11/embeddings/Read/ReadVariableOpMtransformer_block_11/multi_head_attention_11/query/kernel/Read/ReadVariableOpKtransformer_block_11/multi_head_attention_11/query/bias/Read/ReadVariableOpKtransformer_block_11/multi_head_attention_11/key/kernel/Read/ReadVariableOpItransformer_block_11/multi_head_attention_11/key/bias/Read/ReadVariableOpMtransformer_block_11/multi_head_attention_11/value/kernel/Read/ReadVariableOpKtransformer_block_11/multi_head_attention_11/value/bias/Read/ReadVariableOpXtransformer_block_11/multi_head_attention_11/attention_output/kernel/Read/ReadVariableOpVtransformer_block_11/multi_head_attention_11/attention_output/bias/Read/ReadVariableOp#dense_37/kernel/Read/ReadVariableOp!dense_37/bias/Read/ReadVariableOp#dense_38/kernel/Read/ReadVariableOp!dense_38/bias/Read/ReadVariableOpEtransformer_block_11/layer_normalization_22/gamma/Read/ReadVariableOpDtransformer_block_11/layer_normalization_22/beta/Read/ReadVariableOpEtransformer_block_11/layer_normalization_23/gamma/Read/ReadVariableOpDtransformer_block_11/layer_normalization_23/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1SGD/conv1d_10/kernel/momentum/Read/ReadVariableOp/SGD/conv1d_10/bias/momentum/Read/ReadVariableOp1SGD/conv1d_11/kernel/momentum/Read/ReadVariableOp/SGD/conv1d_11/bias/momentum/Read/ReadVariableOp=SGD/batch_normalization_10/gamma/momentum/Read/ReadVariableOp<SGD/batch_normalization_10/beta/momentum/Read/ReadVariableOp=SGD/batch_normalization_11/gamma/momentum/Read/ReadVariableOp<SGD/batch_normalization_11/beta/momentum/Read/ReadVariableOp0SGD/dense_39/kernel/momentum/Read/ReadVariableOp.SGD/dense_39/bias/momentum/Read/ReadVariableOp0SGD/dense_40/kernel/momentum/Read/ReadVariableOp.SGD/dense_40/bias/momentum/Read/ReadVariableOp0SGD/dense_41/kernel/momentum/Read/ReadVariableOp.SGD/dense_41/bias/momentum/Read/ReadVariableOpWSGD/token_and_position_embedding_5/embedding_10/embeddings/momentum/Read/ReadVariableOpWSGD/token_and_position_embedding_5/embedding_11/embeddings/momentum/Read/ReadVariableOpZSGD/transformer_block_11/multi_head_attention_11/query/kernel/momentum/Read/ReadVariableOpXSGD/transformer_block_11/multi_head_attention_11/query/bias/momentum/Read/ReadVariableOpXSGD/transformer_block_11/multi_head_attention_11/key/kernel/momentum/Read/ReadVariableOpVSGD/transformer_block_11/multi_head_attention_11/key/bias/momentum/Read/ReadVariableOpZSGD/transformer_block_11/multi_head_attention_11/value/kernel/momentum/Read/ReadVariableOpXSGD/transformer_block_11/multi_head_attention_11/value/bias/momentum/Read/ReadVariableOpeSGD/transformer_block_11/multi_head_attention_11/attention_output/kernel/momentum/Read/ReadVariableOpcSGD/transformer_block_11/multi_head_attention_11/attention_output/bias/momentum/Read/ReadVariableOp0SGD/dense_37/kernel/momentum/Read/ReadVariableOp.SGD/dense_37/bias/momentum/Read/ReadVariableOp0SGD/dense_38/kernel/momentum/Read/ReadVariableOp.SGD/dense_38/bias/momentum/Read/ReadVariableOpRSGD/transformer_block_11/layer_normalization_22/gamma/momentum/Read/ReadVariableOpQSGD/transformer_block_11/layer_normalization_22/beta/momentum/Read/ReadVariableOpRSGD/transformer_block_11/layer_normalization_23/gamma/momentum/Read/ReadVariableOpQSGD/transformer_block_11/layer_normalization_23/beta/momentum/Read/ReadVariableOpConst*W
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
__inference__traced_save_767515
¿
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_10/kernelconv1d_10/biasconv1d_11/kernelconv1d_11/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_variancebatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_variancedense_39/kerneldense_39/biasdense_40/kerneldense_40/biasdense_41/kerneldense_41/biasdecaylearning_ratemomentumSGD/iter6token_and_position_embedding_5/embedding_10/embeddings6token_and_position_embedding_5/embedding_11/embeddings9transformer_block_11/multi_head_attention_11/query/kernel7transformer_block_11/multi_head_attention_11/query/bias7transformer_block_11/multi_head_attention_11/key/kernel5transformer_block_11/multi_head_attention_11/key/bias9transformer_block_11/multi_head_attention_11/value/kernel7transformer_block_11/multi_head_attention_11/value/biasDtransformer_block_11/multi_head_attention_11/attention_output/kernelBtransformer_block_11/multi_head_attention_11/attention_output/biasdense_37/kerneldense_37/biasdense_38/kerneldense_38/bias1transformer_block_11/layer_normalization_22/gamma0transformer_block_11/layer_normalization_22/beta1transformer_block_11/layer_normalization_23/gamma0transformer_block_11/layer_normalization_23/betatotalcountSGD/conv1d_10/kernel/momentumSGD/conv1d_10/bias/momentumSGD/conv1d_11/kernel/momentumSGD/conv1d_11/bias/momentum)SGD/batch_normalization_10/gamma/momentum(SGD/batch_normalization_10/beta/momentum)SGD/batch_normalization_11/gamma/momentum(SGD/batch_normalization_11/beta/momentumSGD/dense_39/kernel/momentumSGD/dense_39/bias/momentumSGD/dense_40/kernel/momentumSGD/dense_40/bias/momentumSGD/dense_41/kernel/momentumSGD/dense_41/bias/momentumCSGD/token_and_position_embedding_5/embedding_10/embeddings/momentumCSGD/token_and_position_embedding_5/embedding_11/embeddings/momentumFSGD/transformer_block_11/multi_head_attention_11/query/kernel/momentumDSGD/transformer_block_11/multi_head_attention_11/query/bias/momentumDSGD/transformer_block_11/multi_head_attention_11/key/kernel/momentumBSGD/transformer_block_11/multi_head_attention_11/key/bias/momentumFSGD/transformer_block_11/multi_head_attention_11/value/kernel/momentumDSGD/transformer_block_11/multi_head_attention_11/value/bias/momentumQSGD/transformer_block_11/multi_head_attention_11/attention_output/kernel/momentumOSGD/transformer_block_11/multi_head_attention_11/attention_output/bias/momentumSGD/dense_37/kernel/momentumSGD/dense_37/bias/momentumSGD/dense_38/kernel/momentumSGD/dense_38/bias/momentum>SGD/transformer_block_11/layer_normalization_22/gamma/momentum=SGD/transformer_block_11/layer_normalization_22/beta/momentum>SGD/transformer_block_11/layer_normalization_23/gamma/momentum=SGD/transformer_block_11/layer_normalization_23/beta/momentum*V
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
"__inference__traced_restore_767747öß&

Q
5__inference_average_pooling1d_15_layer_call_fn_763566

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
P__inference_average_pooling1d_15_layer_call_and_return_conditional_losses_7635602
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


?__inference_token_and_position_embedding_5_layer_call_fn_766174
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
Z__inference_token_and_position_embedding_5_layer_call_and_return_conditional_losses_7640722
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
ä
 %
C__inference_model_5_layer_call_and_return_conditional_losses_765985
inputs_0
inputs_1G
Ctoken_and_position_embedding_5_embedding_11_embedding_lookup_765754G
Ctoken_and_position_embedding_5_embedding_10_embedding_lookup_7657609
5conv1d_10_conv1d_expanddims_1_readvariableop_resource-
)conv1d_10_biasadd_readvariableop_resource9
5conv1d_11_conv1d_expanddims_1_readvariableop_resource-
)conv1d_11_biasadd_readvariableop_resource<
8batch_normalization_10_batchnorm_readvariableop_resource@
<batch_normalization_10_batchnorm_mul_readvariableop_resource>
:batch_normalization_10_batchnorm_readvariableop_1_resource>
:batch_normalization_10_batchnorm_readvariableop_2_resource<
8batch_normalization_11_batchnorm_readvariableop_resource@
<batch_normalization_11_batchnorm_mul_readvariableop_resource>
:batch_normalization_11_batchnorm_readvariableop_1_resource>
:batch_normalization_11_batchnorm_readvariableop_2_resource\
Xtransformer_block_11_multi_head_attention_11_query_einsum_einsum_readvariableop_resourceR
Ntransformer_block_11_multi_head_attention_11_query_add_readvariableop_resourceZ
Vtransformer_block_11_multi_head_attention_11_key_einsum_einsum_readvariableop_resourceP
Ltransformer_block_11_multi_head_attention_11_key_add_readvariableop_resource\
Xtransformer_block_11_multi_head_attention_11_value_einsum_einsum_readvariableop_resourceR
Ntransformer_block_11_multi_head_attention_11_value_add_readvariableop_resourceg
ctransformer_block_11_multi_head_attention_11_attention_output_einsum_einsum_readvariableop_resource]
Ytransformer_block_11_multi_head_attention_11_attention_output_add_readvariableop_resourceU
Qtransformer_block_11_layer_normalization_22_batchnorm_mul_readvariableop_resourceQ
Mtransformer_block_11_layer_normalization_22_batchnorm_readvariableop_resourceQ
Mtransformer_block_11_sequential_11_dense_37_tensordot_readvariableop_resourceO
Ktransformer_block_11_sequential_11_dense_37_biasadd_readvariableop_resourceQ
Mtransformer_block_11_sequential_11_dense_38_tensordot_readvariableop_resourceO
Ktransformer_block_11_sequential_11_dense_38_biasadd_readvariableop_resourceU
Qtransformer_block_11_layer_normalization_23_batchnorm_mul_readvariableop_resourceQ
Mtransformer_block_11_layer_normalization_23_batchnorm_readvariableop_resource+
'dense_39_matmul_readvariableop_resource,
(dense_39_biasadd_readvariableop_resource+
'dense_40_matmul_readvariableop_resource,
(dense_40_biasadd_readvariableop_resource+
'dense_41_matmul_readvariableop_resource,
(dense_41_biasadd_readvariableop_resource
identity¢/batch_normalization_10/batchnorm/ReadVariableOp¢1batch_normalization_10/batchnorm/ReadVariableOp_1¢1batch_normalization_10/batchnorm/ReadVariableOp_2¢3batch_normalization_10/batchnorm/mul/ReadVariableOp¢/batch_normalization_11/batchnorm/ReadVariableOp¢1batch_normalization_11/batchnorm/ReadVariableOp_1¢1batch_normalization_11/batchnorm/ReadVariableOp_2¢3batch_normalization_11/batchnorm/mul/ReadVariableOp¢ conv1d_10/BiasAdd/ReadVariableOp¢,conv1d_10/conv1d/ExpandDims_1/ReadVariableOp¢ conv1d_11/BiasAdd/ReadVariableOp¢,conv1d_11/conv1d/ExpandDims_1/ReadVariableOp¢dense_39/BiasAdd/ReadVariableOp¢dense_39/MatMul/ReadVariableOp¢dense_40/BiasAdd/ReadVariableOp¢dense_40/MatMul/ReadVariableOp¢dense_41/BiasAdd/ReadVariableOp¢dense_41/MatMul/ReadVariableOp¢<token_and_position_embedding_5/embedding_10/embedding_lookup¢<token_and_position_embedding_5/embedding_11/embedding_lookup¢Dtransformer_block_11/layer_normalization_22/batchnorm/ReadVariableOp¢Htransformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOp¢Dtransformer_block_11/layer_normalization_23/batchnorm/ReadVariableOp¢Htransformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOp¢Ptransformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOp¢Ztransformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp¢Ctransformer_block_11/multi_head_attention_11/key/add/ReadVariableOp¢Mtransformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOp¢Etransformer_block_11/multi_head_attention_11/query/add/ReadVariableOp¢Otransformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp¢Etransformer_block_11/multi_head_attention_11/value/add/ReadVariableOp¢Otransformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp¢Btransformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOp¢Dtransformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOp¢Btransformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOp¢Dtransformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOp
$token_and_position_embedding_5/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_5/Shape»
2token_and_position_embedding_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ24
2token_and_position_embedding_5/strided_slice/stack¶
4token_and_position_embedding_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_5/strided_slice/stack_1¶
4token_and_position_embedding_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_5/strided_slice/stack_2
,token_and_position_embedding_5/strided_sliceStridedSlice-token_and_position_embedding_5/Shape:output:0;token_and_position_embedding_5/strided_slice/stack:output:0=token_and_position_embedding_5/strided_slice/stack_1:output:0=token_and_position_embedding_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_5/strided_slice
*token_and_position_embedding_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_5/range/start
*token_and_position_embedding_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_5/range/delta
$token_and_position_embedding_5/rangeRange3token_and_position_embedding_5/range/start:output:05token_and_position_embedding_5/strided_slice:output:03token_and_position_embedding_5/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$token_and_position_embedding_5/rangeÎ
<token_and_position_embedding_5/embedding_11/embedding_lookupResourceGatherCtoken_and_position_embedding_5_embedding_11_embedding_lookup_765754-token_and_position_embedding_5/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*V
_classL
JHloc:@token_and_position_embedding_5/embedding_11/embedding_lookup/765754*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02>
<token_and_position_embedding_5/embedding_11/embedding_lookup
Etoken_and_position_embedding_5/embedding_11/embedding_lookup/IdentityIdentityEtoken_and_position_embedding_5/embedding_11/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@token_and_position_embedding_5/embedding_11/embedding_lookup/765754*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2G
Etoken_and_position_embedding_5/embedding_11/embedding_lookup/Identity 
Gtoken_and_position_embedding_5/embedding_11/embedding_lookup/Identity_1IdentityNtoken_and_position_embedding_5/embedding_11/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2I
Gtoken_and_position_embedding_5/embedding_11/embedding_lookup/Identity_1¸
0token_and_position_embedding_5/embedding_10/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR22
0token_and_position_embedding_5/embedding_10/CastÚ
<token_and_position_embedding_5/embedding_10/embedding_lookupResourceGatherCtoken_and_position_embedding_5_embedding_10_embedding_lookup_7657604token_and_position_embedding_5/embedding_10/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*V
_classL
JHloc:@token_and_position_embedding_5/embedding_10/embedding_lookup/765760*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02>
<token_and_position_embedding_5/embedding_10/embedding_lookup
Etoken_and_position_embedding_5/embedding_10/embedding_lookup/IdentityIdentityEtoken_and_position_embedding_5/embedding_10/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@token_and_position_embedding_5/embedding_10/embedding_lookup/765760*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2G
Etoken_and_position_embedding_5/embedding_10/embedding_lookup/Identity¥
Gtoken_and_position_embedding_5/embedding_10/embedding_lookup/Identity_1IdentityNtoken_and_position_embedding_5/embedding_10/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2I
Gtoken_and_position_embedding_5/embedding_10/embedding_lookup/Identity_1¬
"token_and_position_embedding_5/addAddV2Ptoken_and_position_embedding_5/embedding_10/embedding_lookup/Identity_1:output:0Ptoken_and_position_embedding_5/embedding_11/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2$
"token_and_position_embedding_5/add
conv1d_10/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2!
conv1d_10/conv1d/ExpandDims/dimÕ
conv1d_10/conv1d/ExpandDims
ExpandDims&token_and_position_embedding_5/add:z:0(conv1d_10/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_10/conv1d/ExpandDimsÖ
,conv1d_10/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_10_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02.
,conv1d_10/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_10/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_10/conv1d/ExpandDims_1/dimß
conv1d_10/conv1d/ExpandDims_1
ExpandDims4conv1d_10/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_10/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_10/conv1d/ExpandDims_1ß
conv1d_10/conv1dConv2D$conv1d_10/conv1d/ExpandDims:output:0&conv1d_10/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
paddingSAME*
strides
2
conv1d_10/conv1d±
conv1d_10/conv1d/SqueezeSqueezeconv1d_10/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_10/conv1d/Squeezeª
 conv1d_10/BiasAdd/ReadVariableOpReadVariableOp)conv1d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv1d_10/BiasAdd/ReadVariableOpµ
conv1d_10/BiasAddBiasAdd!conv1d_10/conv1d/Squeeze:output:0(conv1d_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_10/BiasAdd{
conv1d_10/ReluReluconv1d_10/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_10/Relu
#average_pooling1d_15/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_15/ExpandDims/dim×
average_pooling1d_15/ExpandDims
ExpandDimsconv1d_10/Relu:activations:0,average_pooling1d_15/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2!
average_pooling1d_15/ExpandDimsè
average_pooling1d_15/AvgPoolAvgPool(average_pooling1d_15/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_15/AvgPool¼
average_pooling1d_15/SqueezeSqueeze%average_pooling1d_15/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims
2
average_pooling1d_15/Squeeze
conv1d_11/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2!
conv1d_11/conv1d/ExpandDims/dimÔ
conv1d_11/conv1d/ExpandDims
ExpandDims%average_pooling1d_15/Squeeze:output:0(conv1d_11/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_11/conv1d/ExpandDimsÖ
,conv1d_11/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_11_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02.
,conv1d_11/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_11/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_11/conv1d/ExpandDims_1/dimß
conv1d_11/conv1d/ExpandDims_1
ExpandDims4conv1d_11/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_11/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d_11/conv1d/ExpandDims_1ß
conv1d_11/conv1dConv2D$conv1d_11/conv1d/ExpandDims:output:0&conv1d_11/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
paddingSAME*
strides
2
conv1d_11/conv1d±
conv1d_11/conv1d/SqueezeSqueezeconv1d_11/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_11/conv1d/Squeezeª
 conv1d_11/BiasAdd/ReadVariableOpReadVariableOp)conv1d_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv1d_11/BiasAdd/ReadVariableOpµ
conv1d_11/BiasAddBiasAdd!conv1d_11/conv1d/Squeeze:output:0(conv1d_11/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_11/BiasAdd{
conv1d_11/ReluReluconv1d_11/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_11/Relu
#average_pooling1d_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_17/ExpandDims/dimá
average_pooling1d_17/ExpandDims
ExpandDims&token_and_position_embedding_5/add:z:0,average_pooling1d_17/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2!
average_pooling1d_17/ExpandDimsé
average_pooling1d_17/AvgPoolAvgPool(average_pooling1d_17/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize	
¬*
paddingVALID*
strides	
¬2
average_pooling1d_17/AvgPool»
average_pooling1d_17/SqueezeSqueeze%average_pooling1d_17/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_17/Squeeze
#average_pooling1d_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_16/ExpandDims/dim×
average_pooling1d_16/ExpandDims
ExpandDimsconv1d_11/Relu:activations:0,average_pooling1d_16/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2!
average_pooling1d_16/ExpandDimsç
average_pooling1d_16/AvgPoolAvgPool(average_pooling1d_16/ExpandDims:output:0*
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
average_pooling1d_16/AvgPool»
average_pooling1d_16/SqueezeSqueeze%average_pooling1d_16/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_16/Squeeze×
/batch_normalization_10/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_10/batchnorm/ReadVariableOp
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_10/batchnorm/add/yä
$batch_normalization_10/batchnorm/addAddV27batch_normalization_10/batchnorm/ReadVariableOp:value:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_10/batchnorm/add¨
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_10/batchnorm/Rsqrtã
3batch_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_10/batchnorm/mul/ReadVariableOpá
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:0;batch_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_10/batchnorm/mulÞ
&batch_normalization_10/batchnorm/mul_1Mul%average_pooling1d_16/Squeeze:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&batch_normalization_10/batchnorm/mul_1Ý
1batch_normalization_10/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_10_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype023
1batch_normalization_10/batchnorm/ReadVariableOp_1á
&batch_normalization_10/batchnorm/mul_2Mul9batch_normalization_10/batchnorm/ReadVariableOp_1:value:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_10/batchnorm/mul_2Ý
1batch_normalization_10/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_10_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype023
1batch_normalization_10/batchnorm/ReadVariableOp_2ß
$batch_normalization_10/batchnorm/subSub9batch_normalization_10/batchnorm/ReadVariableOp_2:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_10/batchnorm/subå
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&batch_normalization_10/batchnorm/add_1×
/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_11/batchnorm/ReadVariableOp
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_11/batchnorm/add/yä
$batch_normalization_11/batchnorm/addAddV27batch_normalization_11/batchnorm/ReadVariableOp:value:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_11/batchnorm/add¨
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_11/batchnorm/Rsqrtã
3batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_11/batchnorm/mul/ReadVariableOpá
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:0;batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_11/batchnorm/mulÞ
&batch_normalization_11/batchnorm/mul_1Mul%average_pooling1d_17/Squeeze:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&batch_normalization_11/batchnorm/mul_1Ý
1batch_normalization_11/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_11_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype023
1batch_normalization_11/batchnorm/ReadVariableOp_1á
&batch_normalization_11/batchnorm/mul_2Mul9batch_normalization_11/batchnorm/ReadVariableOp_1:value:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_11/batchnorm/mul_2Ý
1batch_normalization_11/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_11_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype023
1batch_normalization_11/batchnorm/ReadVariableOp_2ß
$batch_normalization_11/batchnorm/subSub9batch_normalization_11/batchnorm/ReadVariableOp_2:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_11/batchnorm/subå
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&batch_normalization_11/batchnorm/add_1­
	add_5/addAddV2*batch_normalization_10/batchnorm/add_1:z:0*batch_normalization_11/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
	add_5/add¿
Otransformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOpReadVariableOpXtransformer_block_11_multi_head_attention_11_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Q
Otransformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOpÖ
@transformer_block_11/multi_head_attention_11/query/einsum/EinsumEinsumadd_5/add:z:0Wtransformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2B
@transformer_block_11/multi_head_attention_11/query/einsum/Einsum
Etransformer_block_11/multi_head_attention_11/query/add/ReadVariableOpReadVariableOpNtransformer_block_11_multi_head_attention_11_query_add_readvariableop_resource*
_output_shapes

: *
dtype02G
Etransformer_block_11/multi_head_attention_11/query/add/ReadVariableOpÍ
6transformer_block_11/multi_head_attention_11/query/addAddV2Itransformer_block_11/multi_head_attention_11/query/einsum/Einsum:output:0Mtransformer_block_11/multi_head_attention_11/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 28
6transformer_block_11/multi_head_attention_11/query/add¹
Mtransformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_11_multi_head_attention_11_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpÐ
>transformer_block_11/multi_head_attention_11/key/einsum/EinsumEinsumadd_5/add:z:0Utransformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2@
>transformer_block_11/multi_head_attention_11/key/einsum/Einsum
Ctransformer_block_11/multi_head_attention_11/key/add/ReadVariableOpReadVariableOpLtransformer_block_11_multi_head_attention_11_key_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_11/multi_head_attention_11/key/add/ReadVariableOpÅ
4transformer_block_11/multi_head_attention_11/key/addAddV2Gtransformer_block_11/multi_head_attention_11/key/einsum/Einsum:output:0Ktransformer_block_11/multi_head_attention_11/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_11/multi_head_attention_11/key/add¿
Otransformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOpReadVariableOpXtransformer_block_11_multi_head_attention_11_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Q
Otransformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOpÖ
@transformer_block_11/multi_head_attention_11/value/einsum/EinsumEinsumadd_5/add:z:0Wtransformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2B
@transformer_block_11/multi_head_attention_11/value/einsum/Einsum
Etransformer_block_11/multi_head_attention_11/value/add/ReadVariableOpReadVariableOpNtransformer_block_11_multi_head_attention_11_value_add_readvariableop_resource*
_output_shapes

: *
dtype02G
Etransformer_block_11/multi_head_attention_11/value/add/ReadVariableOpÍ
6transformer_block_11/multi_head_attention_11/value/addAddV2Itransformer_block_11/multi_head_attention_11/value/einsum/Einsum:output:0Mtransformer_block_11/multi_head_attention_11/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 28
6transformer_block_11/multi_head_attention_11/value/add­
2transformer_block_11/multi_head_attention_11/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>24
2transformer_block_11/multi_head_attention_11/Mul/y
0transformer_block_11/multi_head_attention_11/MulMul:transformer_block_11/multi_head_attention_11/query/add:z:0;transformer_block_11/multi_head_attention_11/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0transformer_block_11/multi_head_attention_11/MulÔ
:transformer_block_11/multi_head_attention_11/einsum/EinsumEinsum8transformer_block_11/multi_head_attention_11/key/add:z:04transformer_block_11/multi_head_attention_11/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2<
:transformer_block_11/multi_head_attention_11/einsum/Einsum
<transformer_block_11/multi_head_attention_11/softmax/SoftmaxSoftmaxCtransformer_block_11/multi_head_attention_11/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2>
<transformer_block_11/multi_head_attention_11/softmax/Softmax
=transformer_block_11/multi_head_attention_11/dropout/IdentityIdentityFtransformer_block_11/multi_head_attention_11/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2?
=transformer_block_11/multi_head_attention_11/dropout/Identityì
<transformer_block_11/multi_head_attention_11/einsum_1/EinsumEinsumFtransformer_block_11/multi_head_attention_11/dropout/Identity:output:0:transformer_block_11/multi_head_attention_11/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2>
<transformer_block_11/multi_head_attention_11/einsum_1/Einsumà
Ztransformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpctransformer_block_11_multi_head_attention_11_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02\
Ztransformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp«
Ktransformer_block_11/multi_head_attention_11/attention_output/einsum/EinsumEinsumEtransformer_block_11/multi_head_attention_11/einsum_1/Einsum:output:0btransformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe2M
Ktransformer_block_11/multi_head_attention_11/attention_output/einsum/Einsumº
Ptransformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOpReadVariableOpYtransformer_block_11_multi_head_attention_11_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02R
Ptransformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOpõ
Atransformer_block_11/multi_head_attention_11/attention_output/addAddV2Ttransformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum:output:0Xtransformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2C
Atransformer_block_11/multi_head_attention_11/attention_output/addÝ
(transformer_block_11/dropout_32/IdentityIdentityEtransformer_block_11/multi_head_attention_11/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2*
(transformer_block_11/dropout_32/Identityµ
transformer_block_11/addAddV2add_5/add:z:01transformer_block_11/dropout_32/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_11/addâ
Jtransformer_block_11/layer_normalization_22/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block_11/layer_normalization_22/moments/mean/reduction_indices¶
8transformer_block_11/layer_normalization_22/moments/meanMeantransformer_block_11/add:z:0Stransformer_block_11/layer_normalization_22/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2:
8transformer_block_11/layer_normalization_22/moments/mean
@transformer_block_11/layer_normalization_22/moments/StopGradientStopGradientAtransformer_block_11/layer_normalization_22/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2B
@transformer_block_11/layer_normalization_22/moments/StopGradientÂ
Etransformer_block_11/layer_normalization_22/moments/SquaredDifferenceSquaredDifferencetransformer_block_11/add:z:0Itransformer_block_11/layer_normalization_22/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2G
Etransformer_block_11/layer_normalization_22/moments/SquaredDifferenceê
Ntransformer_block_11/layer_normalization_22/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
Ntransformer_block_11/layer_normalization_22/moments/variance/reduction_indicesï
<transformer_block_11/layer_normalization_22/moments/varianceMeanItransformer_block_11/layer_normalization_22/moments/SquaredDifference:z:0Wtransformer_block_11/layer_normalization_22/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2>
<transformer_block_11/layer_normalization_22/moments/variance¿
;transformer_block_11/layer_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752=
;transformer_block_11/layer_normalization_22/batchnorm/add/yÂ
9transformer_block_11/layer_normalization_22/batchnorm/addAddV2Etransformer_block_11/layer_normalization_22/moments/variance:output:0Dtransformer_block_11/layer_normalization_22/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2;
9transformer_block_11/layer_normalization_22/batchnorm/addø
;transformer_block_11/layer_normalization_22/batchnorm/RsqrtRsqrt=transformer_block_11/layer_normalization_22/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2=
;transformer_block_11/layer_normalization_22/batchnorm/Rsqrt¢
Htransformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOpQtransformer_block_11_layer_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02J
Htransformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOpÆ
9transformer_block_11/layer_normalization_22/batchnorm/mulMul?transformer_block_11/layer_normalization_22/batchnorm/Rsqrt:y:0Ptransformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_11/layer_normalization_22/batchnorm/mul
;transformer_block_11/layer_normalization_22/batchnorm/mul_1Multransformer_block_11/add:z:0=transformer_block_11/layer_normalization_22/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2=
;transformer_block_11/layer_normalization_22/batchnorm/mul_1¹
;transformer_block_11/layer_normalization_22/batchnorm/mul_2MulAtransformer_block_11/layer_normalization_22/moments/mean:output:0=transformer_block_11/layer_normalization_22/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2=
;transformer_block_11/layer_normalization_22/batchnorm/mul_2
Dtransformer_block_11/layer_normalization_22/batchnorm/ReadVariableOpReadVariableOpMtransformer_block_11_layer_normalization_22_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02F
Dtransformer_block_11/layer_normalization_22/batchnorm/ReadVariableOpÂ
9transformer_block_11/layer_normalization_22/batchnorm/subSubLtransformer_block_11/layer_normalization_22/batchnorm/ReadVariableOp:value:0?transformer_block_11/layer_normalization_22/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_11/layer_normalization_22/batchnorm/sub¹
;transformer_block_11/layer_normalization_22/batchnorm/add_1AddV2?transformer_block_11/layer_normalization_22/batchnorm/mul_1:z:0=transformer_block_11/layer_normalization_22/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2=
;transformer_block_11/layer_normalization_22/batchnorm/add_1
Dtransformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOpReadVariableOpMtransformer_block_11_sequential_11_dense_37_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02F
Dtransformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOpÂ
:transformer_block_11/sequential_11/dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2<
:transformer_block_11/sequential_11/dense_37/Tensordot/axesÉ
:transformer_block_11/sequential_11/dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2<
:transformer_block_11/sequential_11/dense_37/Tensordot/freeé
;transformer_block_11/sequential_11/dense_37/Tensordot/ShapeShape?transformer_block_11/layer_normalization_22/batchnorm/add_1:z:0*
T0*
_output_shapes
:2=
;transformer_block_11/sequential_11/dense_37/Tensordot/ShapeÌ
Ctransformer_block_11/sequential_11/dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_11/sequential_11/dense_37/Tensordot/GatherV2/axis­
>transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2GatherV2Dtransformer_block_11/sequential_11/dense_37/Tensordot/Shape:output:0Ctransformer_block_11/sequential_11/dense_37/Tensordot/free:output:0Ltransformer_block_11/sequential_11/dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2Ð
Etransformer_block_11/sequential_11/dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2G
Etransformer_block_11/sequential_11/dense_37/Tensordot/GatherV2_1/axis³
@transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2_1GatherV2Dtransformer_block_11/sequential_11/dense_37/Tensordot/Shape:output:0Ctransformer_block_11/sequential_11/dense_37/Tensordot/axes:output:0Ntransformer_block_11/sequential_11/dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2B
@transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2_1Ä
;transformer_block_11/sequential_11/dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_11/sequential_11/dense_37/Tensordot/Const°
:transformer_block_11/sequential_11/dense_37/Tensordot/ProdProdGtransformer_block_11/sequential_11/dense_37/Tensordot/GatherV2:output:0Dtransformer_block_11/sequential_11/dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: 2<
:transformer_block_11/sequential_11/dense_37/Tensordot/ProdÈ
=transformer_block_11/sequential_11/dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=transformer_block_11/sequential_11/dense_37/Tensordot/Const_1¸
<transformer_block_11/sequential_11/dense_37/Tensordot/Prod_1ProdItransformer_block_11/sequential_11/dense_37/Tensordot/GatherV2_1:output:0Ftransformer_block_11/sequential_11/dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2>
<transformer_block_11/sequential_11/dense_37/Tensordot/Prod_1È
Atransformer_block_11/sequential_11/dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_11/sequential_11/dense_37/Tensordot/concat/axis
<transformer_block_11/sequential_11/dense_37/Tensordot/concatConcatV2Ctransformer_block_11/sequential_11/dense_37/Tensordot/free:output:0Ctransformer_block_11/sequential_11/dense_37/Tensordot/axes:output:0Jtransformer_block_11/sequential_11/dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_11/sequential_11/dense_37/Tensordot/concat¼
;transformer_block_11/sequential_11/dense_37/Tensordot/stackPackCtransformer_block_11/sequential_11/dense_37/Tensordot/Prod:output:0Etransformer_block_11/sequential_11/dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_11/sequential_11/dense_37/Tensordot/stackÍ
?transformer_block_11/sequential_11/dense_37/Tensordot/transpose	Transpose?transformer_block_11/layer_normalization_22/batchnorm/add_1:z:0Etransformer_block_11/sequential_11/dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?transformer_block_11/sequential_11/dense_37/Tensordot/transposeÏ
=transformer_block_11/sequential_11/dense_37/Tensordot/ReshapeReshapeCtransformer_block_11/sequential_11/dense_37/Tensordot/transpose:y:0Dtransformer_block_11/sequential_11/dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2?
=transformer_block_11/sequential_11/dense_37/Tensordot/ReshapeÎ
<transformer_block_11/sequential_11/dense_37/Tensordot/MatMulMatMulFtransformer_block_11/sequential_11/dense_37/Tensordot/Reshape:output:0Ltransformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2>
<transformer_block_11/sequential_11/dense_37/Tensordot/MatMulÈ
=transformer_block_11/sequential_11/dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2?
=transformer_block_11/sequential_11/dense_37/Tensordot/Const_2Ì
Ctransformer_block_11/sequential_11/dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_11/sequential_11/dense_37/Tensordot/concat_1/axis
>transformer_block_11/sequential_11/dense_37/Tensordot/concat_1ConcatV2Gtransformer_block_11/sequential_11/dense_37/Tensordot/GatherV2:output:0Ftransformer_block_11/sequential_11/dense_37/Tensordot/Const_2:output:0Ltransformer_block_11/sequential_11/dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2@
>transformer_block_11/sequential_11/dense_37/Tensordot/concat_1À
5transformer_block_11/sequential_11/dense_37/TensordotReshapeFtransformer_block_11/sequential_11/dense_37/Tensordot/MatMul:product:0Gtransformer_block_11/sequential_11/dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@27
5transformer_block_11/sequential_11/dense_37/Tensordot
Btransformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOpReadVariableOpKtransformer_block_11_sequential_11_dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02D
Btransformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOp·
3transformer_block_11/sequential_11/dense_37/BiasAddBiasAdd>transformer_block_11/sequential_11/dense_37/Tensordot:output:0Jtransformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@25
3transformer_block_11/sequential_11/dense_37/BiasAddà
0transformer_block_11/sequential_11/dense_37/ReluRelu<transformer_block_11/sequential_11/dense_37/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@22
0transformer_block_11/sequential_11/dense_37/Relu
Dtransformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOpReadVariableOpMtransformer_block_11_sequential_11_dense_38_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02F
Dtransformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOpÂ
:transformer_block_11/sequential_11/dense_38/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2<
:transformer_block_11/sequential_11/dense_38/Tensordot/axesÉ
:transformer_block_11/sequential_11/dense_38/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2<
:transformer_block_11/sequential_11/dense_38/Tensordot/freeè
;transformer_block_11/sequential_11/dense_38/Tensordot/ShapeShape>transformer_block_11/sequential_11/dense_37/Relu:activations:0*
T0*
_output_shapes
:2=
;transformer_block_11/sequential_11/dense_38/Tensordot/ShapeÌ
Ctransformer_block_11/sequential_11/dense_38/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_11/sequential_11/dense_38/Tensordot/GatherV2/axis­
>transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2GatherV2Dtransformer_block_11/sequential_11/dense_38/Tensordot/Shape:output:0Ctransformer_block_11/sequential_11/dense_38/Tensordot/free:output:0Ltransformer_block_11/sequential_11/dense_38/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2Ð
Etransformer_block_11/sequential_11/dense_38/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2G
Etransformer_block_11/sequential_11/dense_38/Tensordot/GatherV2_1/axis³
@transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2_1GatherV2Dtransformer_block_11/sequential_11/dense_38/Tensordot/Shape:output:0Ctransformer_block_11/sequential_11/dense_38/Tensordot/axes:output:0Ntransformer_block_11/sequential_11/dense_38/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2B
@transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2_1Ä
;transformer_block_11/sequential_11/dense_38/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_11/sequential_11/dense_38/Tensordot/Const°
:transformer_block_11/sequential_11/dense_38/Tensordot/ProdProdGtransformer_block_11/sequential_11/dense_38/Tensordot/GatherV2:output:0Dtransformer_block_11/sequential_11/dense_38/Tensordot/Const:output:0*
T0*
_output_shapes
: 2<
:transformer_block_11/sequential_11/dense_38/Tensordot/ProdÈ
=transformer_block_11/sequential_11/dense_38/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=transformer_block_11/sequential_11/dense_38/Tensordot/Const_1¸
<transformer_block_11/sequential_11/dense_38/Tensordot/Prod_1ProdItransformer_block_11/sequential_11/dense_38/Tensordot/GatherV2_1:output:0Ftransformer_block_11/sequential_11/dense_38/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2>
<transformer_block_11/sequential_11/dense_38/Tensordot/Prod_1È
Atransformer_block_11/sequential_11/dense_38/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_11/sequential_11/dense_38/Tensordot/concat/axis
<transformer_block_11/sequential_11/dense_38/Tensordot/concatConcatV2Ctransformer_block_11/sequential_11/dense_38/Tensordot/free:output:0Ctransformer_block_11/sequential_11/dense_38/Tensordot/axes:output:0Jtransformer_block_11/sequential_11/dense_38/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_11/sequential_11/dense_38/Tensordot/concat¼
;transformer_block_11/sequential_11/dense_38/Tensordot/stackPackCtransformer_block_11/sequential_11/dense_38/Tensordot/Prod:output:0Etransformer_block_11/sequential_11/dense_38/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_11/sequential_11/dense_38/Tensordot/stackÌ
?transformer_block_11/sequential_11/dense_38/Tensordot/transpose	Transpose>transformer_block_11/sequential_11/dense_37/Relu:activations:0Etransformer_block_11/sequential_11/dense_38/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2A
?transformer_block_11/sequential_11/dense_38/Tensordot/transposeÏ
=transformer_block_11/sequential_11/dense_38/Tensordot/ReshapeReshapeCtransformer_block_11/sequential_11/dense_38/Tensordot/transpose:y:0Dtransformer_block_11/sequential_11/dense_38/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2?
=transformer_block_11/sequential_11/dense_38/Tensordot/ReshapeÎ
<transformer_block_11/sequential_11/dense_38/Tensordot/MatMulMatMulFtransformer_block_11/sequential_11/dense_38/Tensordot/Reshape:output:0Ltransformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2>
<transformer_block_11/sequential_11/dense_38/Tensordot/MatMulÈ
=transformer_block_11/sequential_11/dense_38/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2?
=transformer_block_11/sequential_11/dense_38/Tensordot/Const_2Ì
Ctransformer_block_11/sequential_11/dense_38/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_11/sequential_11/dense_38/Tensordot/concat_1/axis
>transformer_block_11/sequential_11/dense_38/Tensordot/concat_1ConcatV2Gtransformer_block_11/sequential_11/dense_38/Tensordot/GatherV2:output:0Ftransformer_block_11/sequential_11/dense_38/Tensordot/Const_2:output:0Ltransformer_block_11/sequential_11/dense_38/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2@
>transformer_block_11/sequential_11/dense_38/Tensordot/concat_1À
5transformer_block_11/sequential_11/dense_38/TensordotReshapeFtransformer_block_11/sequential_11/dense_38/Tensordot/MatMul:product:0Gtransformer_block_11/sequential_11/dense_38/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 27
5transformer_block_11/sequential_11/dense_38/Tensordot
Btransformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOpReadVariableOpKtransformer_block_11_sequential_11_dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOp·
3transformer_block_11/sequential_11/dense_38/BiasAddBiasAdd>transformer_block_11/sequential_11/dense_38/Tensordot:output:0Jtransformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 25
3transformer_block_11/sequential_11/dense_38/BiasAddÔ
(transformer_block_11/dropout_33/IdentityIdentity<transformer_block_11/sequential_11/dense_38/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2*
(transformer_block_11/dropout_33/Identityë
transformer_block_11/add_1AddV2?transformer_block_11/layer_normalization_22/batchnorm/add_1:z:01transformer_block_11/dropout_33/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_11/add_1â
Jtransformer_block_11/layer_normalization_23/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block_11/layer_normalization_23/moments/mean/reduction_indices¸
8transformer_block_11/layer_normalization_23/moments/meanMeantransformer_block_11/add_1:z:0Stransformer_block_11/layer_normalization_23/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2:
8transformer_block_11/layer_normalization_23/moments/mean
@transformer_block_11/layer_normalization_23/moments/StopGradientStopGradientAtransformer_block_11/layer_normalization_23/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2B
@transformer_block_11/layer_normalization_23/moments/StopGradientÄ
Etransformer_block_11/layer_normalization_23/moments/SquaredDifferenceSquaredDifferencetransformer_block_11/add_1:z:0Itransformer_block_11/layer_normalization_23/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2G
Etransformer_block_11/layer_normalization_23/moments/SquaredDifferenceê
Ntransformer_block_11/layer_normalization_23/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
Ntransformer_block_11/layer_normalization_23/moments/variance/reduction_indicesï
<transformer_block_11/layer_normalization_23/moments/varianceMeanItransformer_block_11/layer_normalization_23/moments/SquaredDifference:z:0Wtransformer_block_11/layer_normalization_23/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2>
<transformer_block_11/layer_normalization_23/moments/variance¿
;transformer_block_11/layer_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752=
;transformer_block_11/layer_normalization_23/batchnorm/add/yÂ
9transformer_block_11/layer_normalization_23/batchnorm/addAddV2Etransformer_block_11/layer_normalization_23/moments/variance:output:0Dtransformer_block_11/layer_normalization_23/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2;
9transformer_block_11/layer_normalization_23/batchnorm/addø
;transformer_block_11/layer_normalization_23/batchnorm/RsqrtRsqrt=transformer_block_11/layer_normalization_23/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2=
;transformer_block_11/layer_normalization_23/batchnorm/Rsqrt¢
Htransformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOpQtransformer_block_11_layer_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02J
Htransformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOpÆ
9transformer_block_11/layer_normalization_23/batchnorm/mulMul?transformer_block_11/layer_normalization_23/batchnorm/Rsqrt:y:0Ptransformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_11/layer_normalization_23/batchnorm/mul
;transformer_block_11/layer_normalization_23/batchnorm/mul_1Multransformer_block_11/add_1:z:0=transformer_block_11/layer_normalization_23/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2=
;transformer_block_11/layer_normalization_23/batchnorm/mul_1¹
;transformer_block_11/layer_normalization_23/batchnorm/mul_2MulAtransformer_block_11/layer_normalization_23/moments/mean:output:0=transformer_block_11/layer_normalization_23/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2=
;transformer_block_11/layer_normalization_23/batchnorm/mul_2
Dtransformer_block_11/layer_normalization_23/batchnorm/ReadVariableOpReadVariableOpMtransformer_block_11_layer_normalization_23_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02F
Dtransformer_block_11/layer_normalization_23/batchnorm/ReadVariableOpÂ
9transformer_block_11/layer_normalization_23/batchnorm/subSubLtransformer_block_11/layer_normalization_23/batchnorm/ReadVariableOp:value:0?transformer_block_11/layer_normalization_23/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_11/layer_normalization_23/batchnorm/sub¹
;transformer_block_11/layer_normalization_23/batchnorm/add_1AddV2?transformer_block_11/layer_normalization_23/batchnorm/mul_1:z:0=transformer_block_11/layer_normalization_23/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2=
;transformer_block_11/layer_normalization_23/batchnorm/add_1s
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`  2
flatten_5/Const¿
flatten_5/ReshapeReshape?transformer_block_11/layer_normalization_23/batchnorm/add_1:z:0flatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
flatten_5/Reshapex
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_5/concat/axis¾
concatenate_5/concatConcatV2flatten_5/Reshape:output:0inputs_1"concatenate_5/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
concatenate_5/concat©
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	è@*
dtype02 
dense_39/MatMul/ReadVariableOp¥
dense_39/MatMulMatMulconcatenate_5/concat:output:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_39/MatMul§
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_39/BiasAdd/ReadVariableOp¥
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_39/BiasAdds
dense_39/ReluReludense_39/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_39/Relu
dropout_34/IdentityIdentitydense_39/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_34/Identity¨
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_40/MatMul/ReadVariableOp¤
dense_40/MatMulMatMuldropout_34/Identity:output:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_40/MatMul§
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_40/BiasAdd/ReadVariableOp¥
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_40/BiasAdds
dense_40/ReluReludense_40/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_40/Relu
dropout_35/IdentityIdentitydense_40/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_35/Identity¨
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_41/MatMul/ReadVariableOp¤
dense_41/MatMulMatMuldropout_35/Identity:output:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_41/MatMul§
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_41/BiasAdd/ReadVariableOp¥
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_41/BiasAdd®
IdentityIdentitydense_41/BiasAdd:output:00^batch_normalization_10/batchnorm/ReadVariableOp2^batch_normalization_10/batchnorm/ReadVariableOp_12^batch_normalization_10/batchnorm/ReadVariableOp_24^batch_normalization_10/batchnorm/mul/ReadVariableOp0^batch_normalization_11/batchnorm/ReadVariableOp2^batch_normalization_11/batchnorm/ReadVariableOp_12^batch_normalization_11/batchnorm/ReadVariableOp_24^batch_normalization_11/batchnorm/mul/ReadVariableOp!^conv1d_10/BiasAdd/ReadVariableOp-^conv1d_10/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_11/BiasAdd/ReadVariableOp-^conv1d_11/conv1d/ExpandDims_1/ReadVariableOp ^dense_39/BiasAdd/ReadVariableOp^dense_39/MatMul/ReadVariableOp ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp=^token_and_position_embedding_5/embedding_10/embedding_lookup=^token_and_position_embedding_5/embedding_11/embedding_lookupE^transformer_block_11/layer_normalization_22/batchnorm/ReadVariableOpI^transformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOpE^transformer_block_11/layer_normalization_23/batchnorm/ReadVariableOpI^transformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOpQ^transformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOp[^transformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpD^transformer_block_11/multi_head_attention_11/key/add/ReadVariableOpN^transformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpF^transformer_block_11/multi_head_attention_11/query/add/ReadVariableOpP^transformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOpF^transformer_block_11/multi_head_attention_11/value/add/ReadVariableOpP^transformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOpC^transformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOpE^transformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOpC^transformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOpE^transformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2b
/batch_normalization_10/batchnorm/ReadVariableOp/batch_normalization_10/batchnorm/ReadVariableOp2f
1batch_normalization_10/batchnorm/ReadVariableOp_11batch_normalization_10/batchnorm/ReadVariableOp_12f
1batch_normalization_10/batchnorm/ReadVariableOp_21batch_normalization_10/batchnorm/ReadVariableOp_22j
3batch_normalization_10/batchnorm/mul/ReadVariableOp3batch_normalization_10/batchnorm/mul/ReadVariableOp2b
/batch_normalization_11/batchnorm/ReadVariableOp/batch_normalization_11/batchnorm/ReadVariableOp2f
1batch_normalization_11/batchnorm/ReadVariableOp_11batch_normalization_11/batchnorm/ReadVariableOp_12f
1batch_normalization_11/batchnorm/ReadVariableOp_21batch_normalization_11/batchnorm/ReadVariableOp_22j
3batch_normalization_11/batchnorm/mul/ReadVariableOp3batch_normalization_11/batchnorm/mul/ReadVariableOp2D
 conv1d_10/BiasAdd/ReadVariableOp conv1d_10/BiasAdd/ReadVariableOp2\
,conv1d_10/conv1d/ExpandDims_1/ReadVariableOp,conv1d_10/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_11/BiasAdd/ReadVariableOp conv1d_11/BiasAdd/ReadVariableOp2\
,conv1d_11/conv1d/ExpandDims_1/ReadVariableOp,conv1d_11/conv1d/ExpandDims_1/ReadVariableOp2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2|
<token_and_position_embedding_5/embedding_10/embedding_lookup<token_and_position_embedding_5/embedding_10/embedding_lookup2|
<token_and_position_embedding_5/embedding_11/embedding_lookup<token_and_position_embedding_5/embedding_11/embedding_lookup2
Dtransformer_block_11/layer_normalization_22/batchnorm/ReadVariableOpDtransformer_block_11/layer_normalization_22/batchnorm/ReadVariableOp2
Htransformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOpHtransformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOp2
Dtransformer_block_11/layer_normalization_23/batchnorm/ReadVariableOpDtransformer_block_11/layer_normalization_23/batchnorm/ReadVariableOp2
Htransformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOpHtransformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOp2¤
Ptransformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOpPtransformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOp2¸
Ztransformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpZtransformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp2
Ctransformer_block_11/multi_head_attention_11/key/add/ReadVariableOpCtransformer_block_11/multi_head_attention_11/key/add/ReadVariableOp2
Mtransformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpMtransformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOp2
Etransformer_block_11/multi_head_attention_11/query/add/ReadVariableOpEtransformer_block_11/multi_head_attention_11/query/add/ReadVariableOp2¢
Otransformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOpOtransformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp2
Etransformer_block_11/multi_head_attention_11/value/add/ReadVariableOpEtransformer_block_11/multi_head_attention_11/value/add/ReadVariableOp2¢
Otransformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOpOtransformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp2
Btransformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOpBtransformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOp2
Dtransformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOpDtransformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOp2
Btransformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOpBtransformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOp2
Dtransformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOpDtransformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOp:R N
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
î	
Ý
D__inference_dense_40_layer_call_and_return_conditional_losses_766995

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
É
d
F__inference_dropout_35_layer_call_and_return_conditional_losses_764867

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
é

R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_764301

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
Ú
¤
(__inference_model_5_layer_call_fn_765346
input_11
input_12
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
StatefulPartitionedCallStatefulPartitionedCallinput_11input_12unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
C__inference_model_5_layer_call_and_return_conditional_losses_7652712
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
input_11:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_12
º
¡
.__inference_sequential_11_layer_call_fn_767190

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
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
GPU2*0J 8 *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_7640322
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
½0
É
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_766424

inputs
assignmovingavg_766399
assignmovingavg_1_766405)
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
loc:@AssignMovingAvg/766399*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_766399*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/766399*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/766399*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_766399AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/766399*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/766405*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_766405*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/766405*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/766405*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_766405AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/766405*
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
± 
ã
D__inference_dense_37_layer_call_and_return_conditional_losses_763911

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
º
¡
.__inference_sequential_11_layer_call_fn_767177

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
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
GPU2*0J 8 *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_7640052
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
¬
R
&__inference_add_5_layer_call_fn_766564
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
A__inference_add_5_layer_call_and_return_conditional_losses_7643432
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
Ñ
û
P__inference_transformer_block_11_layer_call_and_return_conditional_losses_766712

inputsG
Cmulti_head_attention_11_query_einsum_einsum_readvariableop_resource=
9multi_head_attention_11_query_add_readvariableop_resourceE
Amulti_head_attention_11_key_einsum_einsum_readvariableop_resource;
7multi_head_attention_11_key_add_readvariableop_resourceG
Cmulti_head_attention_11_value_einsum_einsum_readvariableop_resource=
9multi_head_attention_11_value_add_readvariableop_resourceR
Nmulti_head_attention_11_attention_output_einsum_einsum_readvariableop_resourceH
Dmulti_head_attention_11_attention_output_add_readvariableop_resource@
<layer_normalization_22_batchnorm_mul_readvariableop_resource<
8layer_normalization_22_batchnorm_readvariableop_resource<
8sequential_11_dense_37_tensordot_readvariableop_resource:
6sequential_11_dense_37_biasadd_readvariableop_resource<
8sequential_11_dense_38_tensordot_readvariableop_resource:
6sequential_11_dense_38_biasadd_readvariableop_resource@
<layer_normalization_23_batchnorm_mul_readvariableop_resource<
8layer_normalization_23_batchnorm_readvariableop_resource
identity¢/layer_normalization_22/batchnorm/ReadVariableOp¢3layer_normalization_22/batchnorm/mul/ReadVariableOp¢/layer_normalization_23/batchnorm/ReadVariableOp¢3layer_normalization_23/batchnorm/mul/ReadVariableOp¢;multi_head_attention_11/attention_output/add/ReadVariableOp¢Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp¢.multi_head_attention_11/key/add/ReadVariableOp¢8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp¢0multi_head_attention_11/query/add/ReadVariableOp¢:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp¢0multi_head_attention_11/value/add/ReadVariableOp¢:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp¢-sequential_11/dense_37/BiasAdd/ReadVariableOp¢/sequential_11/dense_37/Tensordot/ReadVariableOp¢-sequential_11/dense_38/BiasAdd/ReadVariableOp¢/sequential_11/dense_38/Tensordot/ReadVariableOp
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_11_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02<
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp
+multi_head_attention_11/query/einsum/EinsumEinsuminputsBmulti_head_attention_11/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2-
+multi_head_attention_11/query/einsum/EinsumÞ
0multi_head_attention_11/query/add/ReadVariableOpReadVariableOp9multi_head_attention_11_query_add_readvariableop_resource*
_output_shapes

: *
dtype022
0multi_head_attention_11/query/add/ReadVariableOpù
!multi_head_attention_11/query/addAddV24multi_head_attention_11/query/einsum/Einsum:output:08multi_head_attention_11/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2#
!multi_head_attention_11/query/addú
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOpReadVariableOpAmulti_head_attention_11_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02:
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp
)multi_head_attention_11/key/einsum/EinsumEinsuminputs@multi_head_attention_11/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2+
)multi_head_attention_11/key/einsum/EinsumØ
.multi_head_attention_11/key/add/ReadVariableOpReadVariableOp7multi_head_attention_11_key_add_readvariableop_resource*
_output_shapes

: *
dtype020
.multi_head_attention_11/key/add/ReadVariableOpñ
multi_head_attention_11/key/addAddV22multi_head_attention_11/key/einsum/Einsum:output:06multi_head_attention_11/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
multi_head_attention_11/key/add
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_11_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02<
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp
+multi_head_attention_11/value/einsum/EinsumEinsuminputsBmulti_head_attention_11/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2-
+multi_head_attention_11/value/einsum/EinsumÞ
0multi_head_attention_11/value/add/ReadVariableOpReadVariableOp9multi_head_attention_11_value_add_readvariableop_resource*
_output_shapes

: *
dtype022
0multi_head_attention_11/value/add/ReadVariableOpù
!multi_head_attention_11/value/addAddV24multi_head_attention_11/value/einsum/Einsum:output:08multi_head_attention_11/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2#
!multi_head_attention_11/value/add
multi_head_attention_11/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_11/Mul/yÊ
multi_head_attention_11/MulMul%multi_head_attention_11/query/add:z:0&multi_head_attention_11/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_11/Mul
%multi_head_attention_11/einsum/EinsumEinsum#multi_head_attention_11/key/add:z:0multi_head_attention_11/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2'
%multi_head_attention_11/einsum/EinsumÇ
'multi_head_attention_11/softmax/SoftmaxSoftmax.multi_head_attention_11/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2)
'multi_head_attention_11/softmax/Softmax£
-multi_head_attention_11/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-multi_head_attention_11/dropout/dropout/Const
+multi_head_attention_11/dropout/dropout/MulMul1multi_head_attention_11/softmax/Softmax:softmax:06multi_head_attention_11/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2-
+multi_head_attention_11/dropout/dropout/Mul¿
-multi_head_attention_11/dropout/dropout/ShapeShape1multi_head_attention_11/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2/
-multi_head_attention_11/dropout/dropout/Shape
Dmulti_head_attention_11/dropout/dropout/random_uniform/RandomUniformRandomUniform6multi_head_attention_11/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
dtype02F
Dmulti_head_attention_11/dropout/dropout/random_uniform/RandomUniformµ
6multi_head_attention_11/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6multi_head_attention_11/dropout/dropout/GreaterEqual/yÆ
4multi_head_attention_11/dropout/dropout/GreaterEqualGreaterEqualMmulti_head_attention_11/dropout/dropout/random_uniform/RandomUniform:output:0?multi_head_attention_11/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##26
4multi_head_attention_11/dropout/dropout/GreaterEqualç
,multi_head_attention_11/dropout/dropout/CastCast8multi_head_attention_11/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2.
,multi_head_attention_11/dropout/dropout/Cast
-multi_head_attention_11/dropout/dropout/Mul_1Mul/multi_head_attention_11/dropout/dropout/Mul:z:00multi_head_attention_11/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2/
-multi_head_attention_11/dropout/dropout/Mul_1
'multi_head_attention_11/einsum_1/EinsumEinsum1multi_head_attention_11/dropout/dropout/Mul_1:z:0%multi_head_attention_11/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2)
'multi_head_attention_11/einsum_1/Einsum¡
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpNmulti_head_attention_11_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02G
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp×
6multi_head_attention_11/attention_output/einsum/EinsumEinsum0multi_head_attention_11/einsum_1/Einsum:output:0Mmulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe28
6multi_head_attention_11/attention_output/einsum/Einsumû
;multi_head_attention_11/attention_output/add/ReadVariableOpReadVariableOpDmulti_head_attention_11_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_attention_11/attention_output/add/ReadVariableOp¡
,multi_head_attention_11/attention_output/addAddV2?multi_head_attention_11/attention_output/einsum/Einsum:output:0Cmulti_head_attention_11/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2.
,multi_head_attention_11/attention_output/addy
dropout_32/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_32/dropout/ConstÂ
dropout_32/dropout/MulMul0multi_head_attention_11/attention_output/add:z:0!dropout_32/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_32/dropout/Mul
dropout_32/dropout/ShapeShape0multi_head_attention_11/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_32/dropout/ShapeÙ
/dropout_32/dropout/random_uniform/RandomUniformRandomUniform!dropout_32/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype021
/dropout_32/dropout/random_uniform/RandomUniform
!dropout_32/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_32/dropout/GreaterEqual/yî
dropout_32/dropout/GreaterEqualGreaterEqual8dropout_32/dropout/random_uniform/RandomUniform:output:0*dropout_32/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
dropout_32/dropout/GreaterEqual¤
dropout_32/dropout/CastCast#dropout_32/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_32/dropout/Castª
dropout_32/dropout/Mul_1Muldropout_32/dropout/Mul:z:0dropout_32/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_32/dropout/Mul_1o
addAddV2inputsdropout_32/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¸
5layer_normalization_22/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_22/moments/mean/reduction_indicesâ
#layer_normalization_22/moments/meanMeanadd:z:0>layer_normalization_22/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_22/moments/meanÎ
+layer_normalization_22/moments/StopGradientStopGradient,layer_normalization_22/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_22/moments/StopGradientî
0layer_normalization_22/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_22/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_22/moments/SquaredDifferenceÀ
9layer_normalization_22/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_22/moments/variance/reduction_indices
'layer_normalization_22/moments/varianceMean4layer_normalization_22/moments/SquaredDifference:z:0Blayer_normalization_22/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_22/moments/variance
&layer_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_22/batchnorm/add/yî
$layer_normalization_22/batchnorm/addAddV20layer_normalization_22/moments/variance:output:0/layer_normalization_22/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_22/batchnorm/add¹
&layer_normalization_22/batchnorm/RsqrtRsqrt(layer_normalization_22/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_22/batchnorm/Rsqrtã
3layer_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_22/batchnorm/mul/ReadVariableOpò
$layer_normalization_22/batchnorm/mulMul*layer_normalization_22/batchnorm/Rsqrt:y:0;layer_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_22/batchnorm/mulÀ
&layer_normalization_22/batchnorm/mul_1Muladd:z:0(layer_normalization_22/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_22/batchnorm/mul_1å
&layer_normalization_22/batchnorm/mul_2Mul,layer_normalization_22/moments/mean:output:0(layer_normalization_22/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_22/batchnorm/mul_2×
/layer_normalization_22/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_22_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_22/batchnorm/ReadVariableOpî
$layer_normalization_22/batchnorm/subSub7layer_normalization_22/batchnorm/ReadVariableOp:value:0*layer_normalization_22/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_22/batchnorm/subå
&layer_normalization_22/batchnorm/add_1AddV2*layer_normalization_22/batchnorm/mul_1:z:0(layer_normalization_22/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_22/batchnorm/add_1Û
/sequential_11/dense_37/Tensordot/ReadVariableOpReadVariableOp8sequential_11_dense_37_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype021
/sequential_11/dense_37/Tensordot/ReadVariableOp
%sequential_11/dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_11/dense_37/Tensordot/axes
%sequential_11/dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_11/dense_37/Tensordot/freeª
&sequential_11/dense_37/Tensordot/ShapeShape*layer_normalization_22/batchnorm/add_1:z:0*
T0*
_output_shapes
:2(
&sequential_11/dense_37/Tensordot/Shape¢
.sequential_11/dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_37/Tensordot/GatherV2/axisÄ
)sequential_11/dense_37/Tensordot/GatherV2GatherV2/sequential_11/dense_37/Tensordot/Shape:output:0.sequential_11/dense_37/Tensordot/free:output:07sequential_11/dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_11/dense_37/Tensordot/GatherV2¦
0sequential_11/dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_11/dense_37/Tensordot/GatherV2_1/axisÊ
+sequential_11/dense_37/Tensordot/GatherV2_1GatherV2/sequential_11/dense_37/Tensordot/Shape:output:0.sequential_11/dense_37/Tensordot/axes:output:09sequential_11/dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_11/dense_37/Tensordot/GatherV2_1
&sequential_11/dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_11/dense_37/Tensordot/ConstÜ
%sequential_11/dense_37/Tensordot/ProdProd2sequential_11/dense_37/Tensordot/GatherV2:output:0/sequential_11/dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_11/dense_37/Tensordot/Prod
(sequential_11/dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_11/dense_37/Tensordot/Const_1ä
'sequential_11/dense_37/Tensordot/Prod_1Prod4sequential_11/dense_37/Tensordot/GatherV2_1:output:01sequential_11/dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_11/dense_37/Tensordot/Prod_1
,sequential_11/dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_11/dense_37/Tensordot/concat/axis£
'sequential_11/dense_37/Tensordot/concatConcatV2.sequential_11/dense_37/Tensordot/free:output:0.sequential_11/dense_37/Tensordot/axes:output:05sequential_11/dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_11/dense_37/Tensordot/concatè
&sequential_11/dense_37/Tensordot/stackPack.sequential_11/dense_37/Tensordot/Prod:output:00sequential_11/dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_11/dense_37/Tensordot/stackù
*sequential_11/dense_37/Tensordot/transpose	Transpose*layer_normalization_22/batchnorm/add_1:z:00sequential_11/dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2,
*sequential_11/dense_37/Tensordot/transposeû
(sequential_11/dense_37/Tensordot/ReshapeReshape.sequential_11/dense_37/Tensordot/transpose:y:0/sequential_11/dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_11/dense_37/Tensordot/Reshapeú
'sequential_11/dense_37/Tensordot/MatMulMatMul1sequential_11/dense_37/Tensordot/Reshape:output:07sequential_11/dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2)
'sequential_11/dense_37/Tensordot/MatMul
(sequential_11/dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2*
(sequential_11/dense_37/Tensordot/Const_2¢
.sequential_11/dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_37/Tensordot/concat_1/axis°
)sequential_11/dense_37/Tensordot/concat_1ConcatV22sequential_11/dense_37/Tensordot/GatherV2:output:01sequential_11/dense_37/Tensordot/Const_2:output:07sequential_11/dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_11/dense_37/Tensordot/concat_1ì
 sequential_11/dense_37/TensordotReshape1sequential_11/dense_37/Tensordot/MatMul:product:02sequential_11/dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2"
 sequential_11/dense_37/TensordotÑ
-sequential_11/dense_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_11/dense_37/BiasAdd/ReadVariableOpã
sequential_11/dense_37/BiasAddBiasAdd)sequential_11/dense_37/Tensordot:output:05sequential_11/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2 
sequential_11/dense_37/BiasAdd¡
sequential_11/dense_37/ReluRelu'sequential_11/dense_37/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_11/dense_37/ReluÛ
/sequential_11/dense_38/Tensordot/ReadVariableOpReadVariableOp8sequential_11_dense_38_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype021
/sequential_11/dense_38/Tensordot/ReadVariableOp
%sequential_11/dense_38/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_11/dense_38/Tensordot/axes
%sequential_11/dense_38/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_11/dense_38/Tensordot/free©
&sequential_11/dense_38/Tensordot/ShapeShape)sequential_11/dense_37/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_11/dense_38/Tensordot/Shape¢
.sequential_11/dense_38/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_38/Tensordot/GatherV2/axisÄ
)sequential_11/dense_38/Tensordot/GatherV2GatherV2/sequential_11/dense_38/Tensordot/Shape:output:0.sequential_11/dense_38/Tensordot/free:output:07sequential_11/dense_38/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_11/dense_38/Tensordot/GatherV2¦
0sequential_11/dense_38/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_11/dense_38/Tensordot/GatherV2_1/axisÊ
+sequential_11/dense_38/Tensordot/GatherV2_1GatherV2/sequential_11/dense_38/Tensordot/Shape:output:0.sequential_11/dense_38/Tensordot/axes:output:09sequential_11/dense_38/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_11/dense_38/Tensordot/GatherV2_1
&sequential_11/dense_38/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_11/dense_38/Tensordot/ConstÜ
%sequential_11/dense_38/Tensordot/ProdProd2sequential_11/dense_38/Tensordot/GatherV2:output:0/sequential_11/dense_38/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_11/dense_38/Tensordot/Prod
(sequential_11/dense_38/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_11/dense_38/Tensordot/Const_1ä
'sequential_11/dense_38/Tensordot/Prod_1Prod4sequential_11/dense_38/Tensordot/GatherV2_1:output:01sequential_11/dense_38/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_11/dense_38/Tensordot/Prod_1
,sequential_11/dense_38/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_11/dense_38/Tensordot/concat/axis£
'sequential_11/dense_38/Tensordot/concatConcatV2.sequential_11/dense_38/Tensordot/free:output:0.sequential_11/dense_38/Tensordot/axes:output:05sequential_11/dense_38/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_11/dense_38/Tensordot/concatè
&sequential_11/dense_38/Tensordot/stackPack.sequential_11/dense_38/Tensordot/Prod:output:00sequential_11/dense_38/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_11/dense_38/Tensordot/stackø
*sequential_11/dense_38/Tensordot/transpose	Transpose)sequential_11/dense_37/Relu:activations:00sequential_11/dense_38/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2,
*sequential_11/dense_38/Tensordot/transposeû
(sequential_11/dense_38/Tensordot/ReshapeReshape.sequential_11/dense_38/Tensordot/transpose:y:0/sequential_11/dense_38/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_11/dense_38/Tensordot/Reshapeú
'sequential_11/dense_38/Tensordot/MatMulMatMul1sequential_11/dense_38/Tensordot/Reshape:output:07sequential_11/dense_38/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential_11/dense_38/Tensordot/MatMul
(sequential_11/dense_38/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_11/dense_38/Tensordot/Const_2¢
.sequential_11/dense_38/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_38/Tensordot/concat_1/axis°
)sequential_11/dense_38/Tensordot/concat_1ConcatV22sequential_11/dense_38/Tensordot/GatherV2:output:01sequential_11/dense_38/Tensordot/Const_2:output:07sequential_11/dense_38/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_11/dense_38/Tensordot/concat_1ì
 sequential_11/dense_38/TensordotReshape1sequential_11/dense_38/Tensordot/MatMul:product:02sequential_11/dense_38/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 sequential_11/dense_38/TensordotÑ
-sequential_11/dense_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_11/dense_38/BiasAdd/ReadVariableOpã
sequential_11/dense_38/BiasAddBiasAdd)sequential_11/dense_38/Tensordot:output:05sequential_11/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
sequential_11/dense_38/BiasAddy
dropout_33/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_33/dropout/Const¹
dropout_33/dropout/MulMul'sequential_11/dense_38/BiasAdd:output:0!dropout_33/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_33/dropout/Mul
dropout_33/dropout/ShapeShape'sequential_11/dense_38/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_33/dropout/ShapeÙ
/dropout_33/dropout/random_uniform/RandomUniformRandomUniform!dropout_33/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype021
/dropout_33/dropout/random_uniform/RandomUniform
!dropout_33/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_33/dropout/GreaterEqual/yî
dropout_33/dropout/GreaterEqualGreaterEqual8dropout_33/dropout/random_uniform/RandomUniform:output:0*dropout_33/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
dropout_33/dropout/GreaterEqual¤
dropout_33/dropout/CastCast#dropout_33/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_33/dropout/Castª
dropout_33/dropout/Mul_1Muldropout_33/dropout/Mul:z:0dropout_33/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_33/dropout/Mul_1
add_1AddV2*layer_normalization_22/batchnorm/add_1:z:0dropout_33/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¸
5layer_normalization_23/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_23/moments/mean/reduction_indicesä
#layer_normalization_23/moments/meanMean	add_1:z:0>layer_normalization_23/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_23/moments/meanÎ
+layer_normalization_23/moments/StopGradientStopGradient,layer_normalization_23/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_23/moments/StopGradientð
0layer_normalization_23/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_23/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_23/moments/SquaredDifferenceÀ
9layer_normalization_23/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_23/moments/variance/reduction_indices
'layer_normalization_23/moments/varianceMean4layer_normalization_23/moments/SquaredDifference:z:0Blayer_normalization_23/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_23/moments/variance
&layer_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_23/batchnorm/add/yî
$layer_normalization_23/batchnorm/addAddV20layer_normalization_23/moments/variance:output:0/layer_normalization_23/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_23/batchnorm/add¹
&layer_normalization_23/batchnorm/RsqrtRsqrt(layer_normalization_23/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_23/batchnorm/Rsqrtã
3layer_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_23/batchnorm/mul/ReadVariableOpò
$layer_normalization_23/batchnorm/mulMul*layer_normalization_23/batchnorm/Rsqrt:y:0;layer_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_23/batchnorm/mulÂ
&layer_normalization_23/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_23/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_23/batchnorm/mul_1å
&layer_normalization_23/batchnorm/mul_2Mul,layer_normalization_23/moments/mean:output:0(layer_normalization_23/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_23/batchnorm/mul_2×
/layer_normalization_23/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_23_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_23/batchnorm/ReadVariableOpî
$layer_normalization_23/batchnorm/subSub7layer_normalization_23/batchnorm/ReadVariableOp:value:0*layer_normalization_23/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_23/batchnorm/subå
&layer_normalization_23/batchnorm/add_1AddV2*layer_normalization_23/batchnorm/mul_1:z:0(layer_normalization_23/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_23/batchnorm/add_1è
IdentityIdentity*layer_normalization_23/batchnorm/add_1:z:00^layer_normalization_22/batchnorm/ReadVariableOp4^layer_normalization_22/batchnorm/mul/ReadVariableOp0^layer_normalization_23/batchnorm/ReadVariableOp4^layer_normalization_23/batchnorm/mul/ReadVariableOp<^multi_head_attention_11/attention_output/add/ReadVariableOpF^multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp/^multi_head_attention_11/key/add/ReadVariableOp9^multi_head_attention_11/key/einsum/Einsum/ReadVariableOp1^multi_head_attention_11/query/add/ReadVariableOp;^multi_head_attention_11/query/einsum/Einsum/ReadVariableOp1^multi_head_attention_11/value/add/ReadVariableOp;^multi_head_attention_11/value/einsum/Einsum/ReadVariableOp.^sequential_11/dense_37/BiasAdd/ReadVariableOp0^sequential_11/dense_37/Tensordot/ReadVariableOp.^sequential_11/dense_38/BiasAdd/ReadVariableOp0^sequential_11/dense_38/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2b
/layer_normalization_22/batchnorm/ReadVariableOp/layer_normalization_22/batchnorm/ReadVariableOp2j
3layer_normalization_22/batchnorm/mul/ReadVariableOp3layer_normalization_22/batchnorm/mul/ReadVariableOp2b
/layer_normalization_23/batchnorm/ReadVariableOp/layer_normalization_23/batchnorm/ReadVariableOp2j
3layer_normalization_23/batchnorm/mul/ReadVariableOp3layer_normalization_23/batchnorm/mul/ReadVariableOp2z
;multi_head_attention_11/attention_output/add/ReadVariableOp;multi_head_attention_11/attention_output/add/ReadVariableOp2
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpEmulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp2`
.multi_head_attention_11/key/add/ReadVariableOp.multi_head_attention_11/key/add/ReadVariableOp2t
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_11/query/add/ReadVariableOp0multi_head_attention_11/query/add/ReadVariableOp2x
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_11/value/add/ReadVariableOp0multi_head_attention_11/value/add/ReadVariableOp2x
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp2^
-sequential_11/dense_37/BiasAdd/ReadVariableOp-sequential_11/dense_37/BiasAdd/ReadVariableOp2b
/sequential_11/dense_37/Tensordot/ReadVariableOp/sequential_11/dense_37/Tensordot/ReadVariableOp2^
-sequential_11/dense_38/BiasAdd/ReadVariableOp-sequential_11/dense_38/BiasAdd/ReadVariableOp2b
/sequential_11/dense_38/Tensordot/ReadVariableOp/sequential_11/dense_38/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
é

R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_764210

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
ö
l
P__inference_average_pooling1d_16_layer_call_and_return_conditional_losses_763575

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
àà
û
P__inference_transformer_block_11_layer_call_and_return_conditional_losses_766839

inputsG
Cmulti_head_attention_11_query_einsum_einsum_readvariableop_resource=
9multi_head_attention_11_query_add_readvariableop_resourceE
Amulti_head_attention_11_key_einsum_einsum_readvariableop_resource;
7multi_head_attention_11_key_add_readvariableop_resourceG
Cmulti_head_attention_11_value_einsum_einsum_readvariableop_resource=
9multi_head_attention_11_value_add_readvariableop_resourceR
Nmulti_head_attention_11_attention_output_einsum_einsum_readvariableop_resourceH
Dmulti_head_attention_11_attention_output_add_readvariableop_resource@
<layer_normalization_22_batchnorm_mul_readvariableop_resource<
8layer_normalization_22_batchnorm_readvariableop_resource<
8sequential_11_dense_37_tensordot_readvariableop_resource:
6sequential_11_dense_37_biasadd_readvariableop_resource<
8sequential_11_dense_38_tensordot_readvariableop_resource:
6sequential_11_dense_38_biasadd_readvariableop_resource@
<layer_normalization_23_batchnorm_mul_readvariableop_resource<
8layer_normalization_23_batchnorm_readvariableop_resource
identity¢/layer_normalization_22/batchnorm/ReadVariableOp¢3layer_normalization_22/batchnorm/mul/ReadVariableOp¢/layer_normalization_23/batchnorm/ReadVariableOp¢3layer_normalization_23/batchnorm/mul/ReadVariableOp¢;multi_head_attention_11/attention_output/add/ReadVariableOp¢Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp¢.multi_head_attention_11/key/add/ReadVariableOp¢8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp¢0multi_head_attention_11/query/add/ReadVariableOp¢:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp¢0multi_head_attention_11/value/add/ReadVariableOp¢:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp¢-sequential_11/dense_37/BiasAdd/ReadVariableOp¢/sequential_11/dense_37/Tensordot/ReadVariableOp¢-sequential_11/dense_38/BiasAdd/ReadVariableOp¢/sequential_11/dense_38/Tensordot/ReadVariableOp
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_11_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02<
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp
+multi_head_attention_11/query/einsum/EinsumEinsuminputsBmulti_head_attention_11/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2-
+multi_head_attention_11/query/einsum/EinsumÞ
0multi_head_attention_11/query/add/ReadVariableOpReadVariableOp9multi_head_attention_11_query_add_readvariableop_resource*
_output_shapes

: *
dtype022
0multi_head_attention_11/query/add/ReadVariableOpù
!multi_head_attention_11/query/addAddV24multi_head_attention_11/query/einsum/Einsum:output:08multi_head_attention_11/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2#
!multi_head_attention_11/query/addú
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOpReadVariableOpAmulti_head_attention_11_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02:
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp
)multi_head_attention_11/key/einsum/EinsumEinsuminputs@multi_head_attention_11/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2+
)multi_head_attention_11/key/einsum/EinsumØ
.multi_head_attention_11/key/add/ReadVariableOpReadVariableOp7multi_head_attention_11_key_add_readvariableop_resource*
_output_shapes

: *
dtype020
.multi_head_attention_11/key/add/ReadVariableOpñ
multi_head_attention_11/key/addAddV22multi_head_attention_11/key/einsum/Einsum:output:06multi_head_attention_11/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
multi_head_attention_11/key/add
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_11_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02<
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp
+multi_head_attention_11/value/einsum/EinsumEinsuminputsBmulti_head_attention_11/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2-
+multi_head_attention_11/value/einsum/EinsumÞ
0multi_head_attention_11/value/add/ReadVariableOpReadVariableOp9multi_head_attention_11_value_add_readvariableop_resource*
_output_shapes

: *
dtype022
0multi_head_attention_11/value/add/ReadVariableOpù
!multi_head_attention_11/value/addAddV24multi_head_attention_11/value/einsum/Einsum:output:08multi_head_attention_11/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2#
!multi_head_attention_11/value/add
multi_head_attention_11/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_11/Mul/yÊ
multi_head_attention_11/MulMul%multi_head_attention_11/query/add:z:0&multi_head_attention_11/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_11/Mul
%multi_head_attention_11/einsum/EinsumEinsum#multi_head_attention_11/key/add:z:0multi_head_attention_11/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2'
%multi_head_attention_11/einsum/EinsumÇ
'multi_head_attention_11/softmax/SoftmaxSoftmax.multi_head_attention_11/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2)
'multi_head_attention_11/softmax/SoftmaxÍ
(multi_head_attention_11/dropout/IdentityIdentity1multi_head_attention_11/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2*
(multi_head_attention_11/dropout/Identity
'multi_head_attention_11/einsum_1/EinsumEinsum1multi_head_attention_11/dropout/Identity:output:0%multi_head_attention_11/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2)
'multi_head_attention_11/einsum_1/Einsum¡
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpNmulti_head_attention_11_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02G
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp×
6multi_head_attention_11/attention_output/einsum/EinsumEinsum0multi_head_attention_11/einsum_1/Einsum:output:0Mmulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe28
6multi_head_attention_11/attention_output/einsum/Einsumû
;multi_head_attention_11/attention_output/add/ReadVariableOpReadVariableOpDmulti_head_attention_11_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_attention_11/attention_output/add/ReadVariableOp¡
,multi_head_attention_11/attention_output/addAddV2?multi_head_attention_11/attention_output/einsum/Einsum:output:0Cmulti_head_attention_11/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2.
,multi_head_attention_11/attention_output/add
dropout_32/IdentityIdentity0multi_head_attention_11/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_32/Identityo
addAddV2inputsdropout_32/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¸
5layer_normalization_22/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_22/moments/mean/reduction_indicesâ
#layer_normalization_22/moments/meanMeanadd:z:0>layer_normalization_22/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_22/moments/meanÎ
+layer_normalization_22/moments/StopGradientStopGradient,layer_normalization_22/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_22/moments/StopGradientî
0layer_normalization_22/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_22/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_22/moments/SquaredDifferenceÀ
9layer_normalization_22/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_22/moments/variance/reduction_indices
'layer_normalization_22/moments/varianceMean4layer_normalization_22/moments/SquaredDifference:z:0Blayer_normalization_22/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_22/moments/variance
&layer_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_22/batchnorm/add/yî
$layer_normalization_22/batchnorm/addAddV20layer_normalization_22/moments/variance:output:0/layer_normalization_22/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_22/batchnorm/add¹
&layer_normalization_22/batchnorm/RsqrtRsqrt(layer_normalization_22/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_22/batchnorm/Rsqrtã
3layer_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_22/batchnorm/mul/ReadVariableOpò
$layer_normalization_22/batchnorm/mulMul*layer_normalization_22/batchnorm/Rsqrt:y:0;layer_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_22/batchnorm/mulÀ
&layer_normalization_22/batchnorm/mul_1Muladd:z:0(layer_normalization_22/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_22/batchnorm/mul_1å
&layer_normalization_22/batchnorm/mul_2Mul,layer_normalization_22/moments/mean:output:0(layer_normalization_22/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_22/batchnorm/mul_2×
/layer_normalization_22/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_22_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_22/batchnorm/ReadVariableOpî
$layer_normalization_22/batchnorm/subSub7layer_normalization_22/batchnorm/ReadVariableOp:value:0*layer_normalization_22/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_22/batchnorm/subå
&layer_normalization_22/batchnorm/add_1AddV2*layer_normalization_22/batchnorm/mul_1:z:0(layer_normalization_22/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_22/batchnorm/add_1Û
/sequential_11/dense_37/Tensordot/ReadVariableOpReadVariableOp8sequential_11_dense_37_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype021
/sequential_11/dense_37/Tensordot/ReadVariableOp
%sequential_11/dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_11/dense_37/Tensordot/axes
%sequential_11/dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_11/dense_37/Tensordot/freeª
&sequential_11/dense_37/Tensordot/ShapeShape*layer_normalization_22/batchnorm/add_1:z:0*
T0*
_output_shapes
:2(
&sequential_11/dense_37/Tensordot/Shape¢
.sequential_11/dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_37/Tensordot/GatherV2/axisÄ
)sequential_11/dense_37/Tensordot/GatherV2GatherV2/sequential_11/dense_37/Tensordot/Shape:output:0.sequential_11/dense_37/Tensordot/free:output:07sequential_11/dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_11/dense_37/Tensordot/GatherV2¦
0sequential_11/dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_11/dense_37/Tensordot/GatherV2_1/axisÊ
+sequential_11/dense_37/Tensordot/GatherV2_1GatherV2/sequential_11/dense_37/Tensordot/Shape:output:0.sequential_11/dense_37/Tensordot/axes:output:09sequential_11/dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_11/dense_37/Tensordot/GatherV2_1
&sequential_11/dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_11/dense_37/Tensordot/ConstÜ
%sequential_11/dense_37/Tensordot/ProdProd2sequential_11/dense_37/Tensordot/GatherV2:output:0/sequential_11/dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_11/dense_37/Tensordot/Prod
(sequential_11/dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_11/dense_37/Tensordot/Const_1ä
'sequential_11/dense_37/Tensordot/Prod_1Prod4sequential_11/dense_37/Tensordot/GatherV2_1:output:01sequential_11/dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_11/dense_37/Tensordot/Prod_1
,sequential_11/dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_11/dense_37/Tensordot/concat/axis£
'sequential_11/dense_37/Tensordot/concatConcatV2.sequential_11/dense_37/Tensordot/free:output:0.sequential_11/dense_37/Tensordot/axes:output:05sequential_11/dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_11/dense_37/Tensordot/concatè
&sequential_11/dense_37/Tensordot/stackPack.sequential_11/dense_37/Tensordot/Prod:output:00sequential_11/dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_11/dense_37/Tensordot/stackù
*sequential_11/dense_37/Tensordot/transpose	Transpose*layer_normalization_22/batchnorm/add_1:z:00sequential_11/dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2,
*sequential_11/dense_37/Tensordot/transposeû
(sequential_11/dense_37/Tensordot/ReshapeReshape.sequential_11/dense_37/Tensordot/transpose:y:0/sequential_11/dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_11/dense_37/Tensordot/Reshapeú
'sequential_11/dense_37/Tensordot/MatMulMatMul1sequential_11/dense_37/Tensordot/Reshape:output:07sequential_11/dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2)
'sequential_11/dense_37/Tensordot/MatMul
(sequential_11/dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2*
(sequential_11/dense_37/Tensordot/Const_2¢
.sequential_11/dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_37/Tensordot/concat_1/axis°
)sequential_11/dense_37/Tensordot/concat_1ConcatV22sequential_11/dense_37/Tensordot/GatherV2:output:01sequential_11/dense_37/Tensordot/Const_2:output:07sequential_11/dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_11/dense_37/Tensordot/concat_1ì
 sequential_11/dense_37/TensordotReshape1sequential_11/dense_37/Tensordot/MatMul:product:02sequential_11/dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2"
 sequential_11/dense_37/TensordotÑ
-sequential_11/dense_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_11/dense_37/BiasAdd/ReadVariableOpã
sequential_11/dense_37/BiasAddBiasAdd)sequential_11/dense_37/Tensordot:output:05sequential_11/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2 
sequential_11/dense_37/BiasAdd¡
sequential_11/dense_37/ReluRelu'sequential_11/dense_37/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_11/dense_37/ReluÛ
/sequential_11/dense_38/Tensordot/ReadVariableOpReadVariableOp8sequential_11_dense_38_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype021
/sequential_11/dense_38/Tensordot/ReadVariableOp
%sequential_11/dense_38/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_11/dense_38/Tensordot/axes
%sequential_11/dense_38/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_11/dense_38/Tensordot/free©
&sequential_11/dense_38/Tensordot/ShapeShape)sequential_11/dense_37/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_11/dense_38/Tensordot/Shape¢
.sequential_11/dense_38/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_38/Tensordot/GatherV2/axisÄ
)sequential_11/dense_38/Tensordot/GatherV2GatherV2/sequential_11/dense_38/Tensordot/Shape:output:0.sequential_11/dense_38/Tensordot/free:output:07sequential_11/dense_38/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_11/dense_38/Tensordot/GatherV2¦
0sequential_11/dense_38/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_11/dense_38/Tensordot/GatherV2_1/axisÊ
+sequential_11/dense_38/Tensordot/GatherV2_1GatherV2/sequential_11/dense_38/Tensordot/Shape:output:0.sequential_11/dense_38/Tensordot/axes:output:09sequential_11/dense_38/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_11/dense_38/Tensordot/GatherV2_1
&sequential_11/dense_38/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_11/dense_38/Tensordot/ConstÜ
%sequential_11/dense_38/Tensordot/ProdProd2sequential_11/dense_38/Tensordot/GatherV2:output:0/sequential_11/dense_38/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_11/dense_38/Tensordot/Prod
(sequential_11/dense_38/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_11/dense_38/Tensordot/Const_1ä
'sequential_11/dense_38/Tensordot/Prod_1Prod4sequential_11/dense_38/Tensordot/GatherV2_1:output:01sequential_11/dense_38/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_11/dense_38/Tensordot/Prod_1
,sequential_11/dense_38/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_11/dense_38/Tensordot/concat/axis£
'sequential_11/dense_38/Tensordot/concatConcatV2.sequential_11/dense_38/Tensordot/free:output:0.sequential_11/dense_38/Tensordot/axes:output:05sequential_11/dense_38/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_11/dense_38/Tensordot/concatè
&sequential_11/dense_38/Tensordot/stackPack.sequential_11/dense_38/Tensordot/Prod:output:00sequential_11/dense_38/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_11/dense_38/Tensordot/stackø
*sequential_11/dense_38/Tensordot/transpose	Transpose)sequential_11/dense_37/Relu:activations:00sequential_11/dense_38/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2,
*sequential_11/dense_38/Tensordot/transposeû
(sequential_11/dense_38/Tensordot/ReshapeReshape.sequential_11/dense_38/Tensordot/transpose:y:0/sequential_11/dense_38/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_11/dense_38/Tensordot/Reshapeú
'sequential_11/dense_38/Tensordot/MatMulMatMul1sequential_11/dense_38/Tensordot/Reshape:output:07sequential_11/dense_38/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential_11/dense_38/Tensordot/MatMul
(sequential_11/dense_38/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_11/dense_38/Tensordot/Const_2¢
.sequential_11/dense_38/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_38/Tensordot/concat_1/axis°
)sequential_11/dense_38/Tensordot/concat_1ConcatV22sequential_11/dense_38/Tensordot/GatherV2:output:01sequential_11/dense_38/Tensordot/Const_2:output:07sequential_11/dense_38/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_11/dense_38/Tensordot/concat_1ì
 sequential_11/dense_38/TensordotReshape1sequential_11/dense_38/Tensordot/MatMul:product:02sequential_11/dense_38/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 sequential_11/dense_38/TensordotÑ
-sequential_11/dense_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_11/dense_38/BiasAdd/ReadVariableOpã
sequential_11/dense_38/BiasAddBiasAdd)sequential_11/dense_38/Tensordot:output:05sequential_11/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
sequential_11/dense_38/BiasAdd
dropout_33/IdentityIdentity'sequential_11/dense_38/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_33/Identity
add_1AddV2*layer_normalization_22/batchnorm/add_1:z:0dropout_33/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¸
5layer_normalization_23/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_23/moments/mean/reduction_indicesä
#layer_normalization_23/moments/meanMean	add_1:z:0>layer_normalization_23/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_23/moments/meanÎ
+layer_normalization_23/moments/StopGradientStopGradient,layer_normalization_23/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_23/moments/StopGradientð
0layer_normalization_23/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_23/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_23/moments/SquaredDifferenceÀ
9layer_normalization_23/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_23/moments/variance/reduction_indices
'layer_normalization_23/moments/varianceMean4layer_normalization_23/moments/SquaredDifference:z:0Blayer_normalization_23/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_23/moments/variance
&layer_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_23/batchnorm/add/yî
$layer_normalization_23/batchnorm/addAddV20layer_normalization_23/moments/variance:output:0/layer_normalization_23/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_23/batchnorm/add¹
&layer_normalization_23/batchnorm/RsqrtRsqrt(layer_normalization_23/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_23/batchnorm/Rsqrtã
3layer_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_23/batchnorm/mul/ReadVariableOpò
$layer_normalization_23/batchnorm/mulMul*layer_normalization_23/batchnorm/Rsqrt:y:0;layer_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_23/batchnorm/mulÂ
&layer_normalization_23/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_23/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_23/batchnorm/mul_1å
&layer_normalization_23/batchnorm/mul_2Mul,layer_normalization_23/moments/mean:output:0(layer_normalization_23/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_23/batchnorm/mul_2×
/layer_normalization_23/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_23_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_23/batchnorm/ReadVariableOpî
$layer_normalization_23/batchnorm/subSub7layer_normalization_23/batchnorm/ReadVariableOp:value:0*layer_normalization_23/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_23/batchnorm/subå
&layer_normalization_23/batchnorm/add_1AddV2*layer_normalization_23/batchnorm/mul_1:z:0(layer_normalization_23/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_23/batchnorm/add_1è
IdentityIdentity*layer_normalization_23/batchnorm/add_1:z:00^layer_normalization_22/batchnorm/ReadVariableOp4^layer_normalization_22/batchnorm/mul/ReadVariableOp0^layer_normalization_23/batchnorm/ReadVariableOp4^layer_normalization_23/batchnorm/mul/ReadVariableOp<^multi_head_attention_11/attention_output/add/ReadVariableOpF^multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp/^multi_head_attention_11/key/add/ReadVariableOp9^multi_head_attention_11/key/einsum/Einsum/ReadVariableOp1^multi_head_attention_11/query/add/ReadVariableOp;^multi_head_attention_11/query/einsum/Einsum/ReadVariableOp1^multi_head_attention_11/value/add/ReadVariableOp;^multi_head_attention_11/value/einsum/Einsum/ReadVariableOp.^sequential_11/dense_37/BiasAdd/ReadVariableOp0^sequential_11/dense_37/Tensordot/ReadVariableOp.^sequential_11/dense_38/BiasAdd/ReadVariableOp0^sequential_11/dense_38/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2b
/layer_normalization_22/batchnorm/ReadVariableOp/layer_normalization_22/batchnorm/ReadVariableOp2j
3layer_normalization_22/batchnorm/mul/ReadVariableOp3layer_normalization_22/batchnorm/mul/ReadVariableOp2b
/layer_normalization_23/batchnorm/ReadVariableOp/layer_normalization_23/batchnorm/ReadVariableOp2j
3layer_normalization_23/batchnorm/mul/ReadVariableOp3layer_normalization_23/batchnorm/mul/ReadVariableOp2z
;multi_head_attention_11/attention_output/add/ReadVariableOp;multi_head_attention_11/attention_output/add/ReadVariableOp2
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpEmulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp2`
.multi_head_attention_11/key/add/ReadVariableOp.multi_head_attention_11/key/add/ReadVariableOp2t
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_11/query/add/ReadVariableOp0multi_head_attention_11/query/add/ReadVariableOp2x
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_11/value/add/ReadVariableOp0multi_head_attention_11/value/add/ReadVariableOp2x
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp2^
-sequential_11/dense_37/BiasAdd/ReadVariableOp-sequential_11/dense_37/BiasAdd/ReadVariableOp2b
/sequential_11/dense_37/Tensordot/ReadVariableOp/sequential_11/dense_37/Tensordot/ReadVariableOp2^
-sequential_11/dense_38/BiasAdd/ReadVariableOp-sequential_11/dense_38/BiasAdd/ReadVariableOp2b
/sequential_11/dense_38/Tensordot/ReadVariableOp/sequential_11/dense_38/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
½0
É
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_766342

inputs
assignmovingavg_766317
assignmovingavg_1_766323)
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
loc:@AssignMovingAvg/766317*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_766317*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/766317*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/766317*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_766317AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/766317*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/766323*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_766323*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/766323*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/766323*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_766323AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/766323*
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
Ô\

C__inference_model_5_layer_call_and_return_conditional_losses_765099

inputs
inputs_1)
%token_and_position_embedding_5_765009)
%token_and_position_embedding_5_765011
conv1d_10_765014
conv1d_10_765016
conv1d_11_765020
conv1d_11_765022!
batch_normalization_10_765027!
batch_normalization_10_765029!
batch_normalization_10_765031!
batch_normalization_10_765033!
batch_normalization_11_765036!
batch_normalization_11_765038!
batch_normalization_11_765040!
batch_normalization_11_765042
transformer_block_11_765046
transformer_block_11_765048
transformer_block_11_765050
transformer_block_11_765052
transformer_block_11_765054
transformer_block_11_765056
transformer_block_11_765058
transformer_block_11_765060
transformer_block_11_765062
transformer_block_11_765064
transformer_block_11_765066
transformer_block_11_765068
transformer_block_11_765070
transformer_block_11_765072
transformer_block_11_765074
transformer_block_11_765076
dense_39_765081
dense_39_765083
dense_40_765087
dense_40_765089
dense_41_765093
dense_41_765095
identity¢.batch_normalization_10/StatefulPartitionedCall¢.batch_normalization_11/StatefulPartitionedCall¢!conv1d_10/StatefulPartitionedCall¢!conv1d_11/StatefulPartitionedCall¢ dense_39/StatefulPartitionedCall¢ dense_40/StatefulPartitionedCall¢ dense_41/StatefulPartitionedCall¢"dropout_34/StatefulPartitionedCall¢"dropout_35/StatefulPartitionedCall¢6token_and_position_embedding_5/StatefulPartitionedCall¢,transformer_block_11/StatefulPartitionedCall
6token_and_position_embedding_5/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_5_765009%token_and_position_embedding_5_765011*
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
Z__inference_token_and_position_embedding_5_layer_call_and_return_conditional_losses_76407228
6token_and_position_embedding_5/StatefulPartitionedCallÚ
!conv1d_10/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_5/StatefulPartitionedCall:output:0conv1d_10_765014conv1d_10_765016*
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
GPU2*0J 8 *N
fIRG
E__inference_conv1d_10_layer_call_and_return_conditional_losses_7641042#
!conv1d_10/StatefulPartitionedCall¤
$average_pooling1d_15/PartitionedCallPartitionedCall*conv1d_10/StatefulPartitionedCall:output:0*
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
P__inference_average_pooling1d_15_layer_call_and_return_conditional_losses_7635602&
$average_pooling1d_15/PartitionedCallÈ
!conv1d_11/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_15/PartitionedCall:output:0conv1d_11_765020conv1d_11_765022*
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
GPU2*0J 8 *N
fIRG
E__inference_conv1d_11_layer_call_and_return_conditional_losses_7641372#
!conv1d_11/StatefulPartitionedCall¸
$average_pooling1d_17/PartitionedCallPartitionedCall?token_and_position_embedding_5/StatefulPartitionedCall:output:0*
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
P__inference_average_pooling1d_17_layer_call_and_return_conditional_losses_7635902&
$average_pooling1d_17/PartitionedCall£
$average_pooling1d_16/PartitionedCallPartitionedCall*conv1d_11/StatefulPartitionedCall:output:0*
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
P__inference_average_pooling1d_16_layer_call_and_return_conditional_losses_7635752&
$average_pooling1d_16/PartitionedCallÈ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_16/PartitionedCall:output:0batch_normalization_10_765027batch_normalization_10_765029batch_normalization_10_765031batch_normalization_10_765033*
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
GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_76419020
.batch_normalization_10/StatefulPartitionedCallÈ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_17/PartitionedCall:output:0batch_normalization_11_765036batch_normalization_11_765038batch_normalization_11_765040batch_normalization_11_765042*
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
GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_76428120
.batch_normalization_11/StatefulPartitionedCall½
add_5/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:07batch_normalization_11/StatefulPartitionedCall:output:0*
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
A__inference_add_5_layer_call_and_return_conditional_losses_7643432
add_5/PartitionedCall¡
,transformer_block_11/StatefulPartitionedCallStatefulPartitionedCalladd_5/PartitionedCall:output:0transformer_block_11_765046transformer_block_11_765048transformer_block_11_765050transformer_block_11_765052transformer_block_11_765054transformer_block_11_765056transformer_block_11_765058transformer_block_11_765060transformer_block_11_765062transformer_block_11_765064transformer_block_11_765066transformer_block_11_765068transformer_block_11_765070transformer_block_11_765072transformer_block_11_765074transformer_block_11_765076*
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
GPU2*0J 8 *Y
fTRR
P__inference_transformer_block_11_layer_call_and_return_conditional_losses_7645002.
,transformer_block_11/StatefulPartitionedCall
flatten_5/PartitionedCallPartitionedCall5transformer_block_11/StatefulPartitionedCall:output:0*
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_7647422
flatten_5/PartitionedCall
concatenate_5/PartitionedCallPartitionedCall"flatten_5/PartitionedCall:output:0inputs_1*
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
I__inference_concatenate_5_layer_call_and_return_conditional_losses_7647572
concatenate_5/PartitionedCall·
 dense_39/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_39_765081dense_39_765083*
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
D__inference_dense_39_layer_call_and_return_conditional_losses_7647772"
 dense_39/StatefulPartitionedCall
"dropout_34/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0*
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
F__inference_dropout_34_layer_call_and_return_conditional_losses_7648052$
"dropout_34/StatefulPartitionedCall¼
 dense_40/StatefulPartitionedCallStatefulPartitionedCall+dropout_34/StatefulPartitionedCall:output:0dense_40_765087dense_40_765089*
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
D__inference_dense_40_layer_call_and_return_conditional_losses_7648342"
 dense_40/StatefulPartitionedCall½
"dropout_35/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0#^dropout_34/StatefulPartitionedCall*
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
F__inference_dropout_35_layer_call_and_return_conditional_losses_7648622$
"dropout_35/StatefulPartitionedCall¼
 dense_41/StatefulPartitionedCallStatefulPartitionedCall+dropout_35/StatefulPartitionedCall:output:0dense_41_765093dense_41_765095*
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
D__inference_dense_41_layer_call_and_return_conditional_losses_7648902"
 dense_41/StatefulPartitionedCallÂ
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall"^conv1d_10/StatefulPartitionedCall"^conv1d_11/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall#^dropout_34/StatefulPartitionedCall#^dropout_35/StatefulPartitionedCall7^token_and_position_embedding_5/StatefulPartitionedCall-^transformer_block_11/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2F
!conv1d_10/StatefulPartitionedCall!conv1d_10/StatefulPartitionedCall2F
!conv1d_11/StatefulPartitionedCall!conv1d_11/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2H
"dropout_34/StatefulPartitionedCall"dropout_34/StatefulPartitionedCall2H
"dropout_35/StatefulPartitionedCall"dropout_35/StatefulPartitionedCall2p
6token_and_position_embedding_5/StatefulPartitionedCall6token_and_position_embedding_5/StatefulPartitionedCall2\
,transformer_block_11/StatefulPartitionedCall,transformer_block_11/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
àà
û
P__inference_transformer_block_11_layer_call_and_return_conditional_losses_764627

inputsG
Cmulti_head_attention_11_query_einsum_einsum_readvariableop_resource=
9multi_head_attention_11_query_add_readvariableop_resourceE
Amulti_head_attention_11_key_einsum_einsum_readvariableop_resource;
7multi_head_attention_11_key_add_readvariableop_resourceG
Cmulti_head_attention_11_value_einsum_einsum_readvariableop_resource=
9multi_head_attention_11_value_add_readvariableop_resourceR
Nmulti_head_attention_11_attention_output_einsum_einsum_readvariableop_resourceH
Dmulti_head_attention_11_attention_output_add_readvariableop_resource@
<layer_normalization_22_batchnorm_mul_readvariableop_resource<
8layer_normalization_22_batchnorm_readvariableop_resource<
8sequential_11_dense_37_tensordot_readvariableop_resource:
6sequential_11_dense_37_biasadd_readvariableop_resource<
8sequential_11_dense_38_tensordot_readvariableop_resource:
6sequential_11_dense_38_biasadd_readvariableop_resource@
<layer_normalization_23_batchnorm_mul_readvariableop_resource<
8layer_normalization_23_batchnorm_readvariableop_resource
identity¢/layer_normalization_22/batchnorm/ReadVariableOp¢3layer_normalization_22/batchnorm/mul/ReadVariableOp¢/layer_normalization_23/batchnorm/ReadVariableOp¢3layer_normalization_23/batchnorm/mul/ReadVariableOp¢;multi_head_attention_11/attention_output/add/ReadVariableOp¢Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp¢.multi_head_attention_11/key/add/ReadVariableOp¢8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp¢0multi_head_attention_11/query/add/ReadVariableOp¢:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp¢0multi_head_attention_11/value/add/ReadVariableOp¢:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp¢-sequential_11/dense_37/BiasAdd/ReadVariableOp¢/sequential_11/dense_37/Tensordot/ReadVariableOp¢-sequential_11/dense_38/BiasAdd/ReadVariableOp¢/sequential_11/dense_38/Tensordot/ReadVariableOp
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_11_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02<
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp
+multi_head_attention_11/query/einsum/EinsumEinsuminputsBmulti_head_attention_11/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2-
+multi_head_attention_11/query/einsum/EinsumÞ
0multi_head_attention_11/query/add/ReadVariableOpReadVariableOp9multi_head_attention_11_query_add_readvariableop_resource*
_output_shapes

: *
dtype022
0multi_head_attention_11/query/add/ReadVariableOpù
!multi_head_attention_11/query/addAddV24multi_head_attention_11/query/einsum/Einsum:output:08multi_head_attention_11/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2#
!multi_head_attention_11/query/addú
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOpReadVariableOpAmulti_head_attention_11_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02:
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp
)multi_head_attention_11/key/einsum/EinsumEinsuminputs@multi_head_attention_11/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2+
)multi_head_attention_11/key/einsum/EinsumØ
.multi_head_attention_11/key/add/ReadVariableOpReadVariableOp7multi_head_attention_11_key_add_readvariableop_resource*
_output_shapes

: *
dtype020
.multi_head_attention_11/key/add/ReadVariableOpñ
multi_head_attention_11/key/addAddV22multi_head_attention_11/key/einsum/Einsum:output:06multi_head_attention_11/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
multi_head_attention_11/key/add
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_11_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02<
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp
+multi_head_attention_11/value/einsum/EinsumEinsuminputsBmulti_head_attention_11/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2-
+multi_head_attention_11/value/einsum/EinsumÞ
0multi_head_attention_11/value/add/ReadVariableOpReadVariableOp9multi_head_attention_11_value_add_readvariableop_resource*
_output_shapes

: *
dtype022
0multi_head_attention_11/value/add/ReadVariableOpù
!multi_head_attention_11/value/addAddV24multi_head_attention_11/value/einsum/Einsum:output:08multi_head_attention_11/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2#
!multi_head_attention_11/value/add
multi_head_attention_11/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_11/Mul/yÊ
multi_head_attention_11/MulMul%multi_head_attention_11/query/add:z:0&multi_head_attention_11/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_11/Mul
%multi_head_attention_11/einsum/EinsumEinsum#multi_head_attention_11/key/add:z:0multi_head_attention_11/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2'
%multi_head_attention_11/einsum/EinsumÇ
'multi_head_attention_11/softmax/SoftmaxSoftmax.multi_head_attention_11/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2)
'multi_head_attention_11/softmax/SoftmaxÍ
(multi_head_attention_11/dropout/IdentityIdentity1multi_head_attention_11/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2*
(multi_head_attention_11/dropout/Identity
'multi_head_attention_11/einsum_1/EinsumEinsum1multi_head_attention_11/dropout/Identity:output:0%multi_head_attention_11/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2)
'multi_head_attention_11/einsum_1/Einsum¡
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpNmulti_head_attention_11_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02G
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp×
6multi_head_attention_11/attention_output/einsum/EinsumEinsum0multi_head_attention_11/einsum_1/Einsum:output:0Mmulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe28
6multi_head_attention_11/attention_output/einsum/Einsumû
;multi_head_attention_11/attention_output/add/ReadVariableOpReadVariableOpDmulti_head_attention_11_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_attention_11/attention_output/add/ReadVariableOp¡
,multi_head_attention_11/attention_output/addAddV2?multi_head_attention_11/attention_output/einsum/Einsum:output:0Cmulti_head_attention_11/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2.
,multi_head_attention_11/attention_output/add
dropout_32/IdentityIdentity0multi_head_attention_11/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_32/Identityo
addAddV2inputsdropout_32/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¸
5layer_normalization_22/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_22/moments/mean/reduction_indicesâ
#layer_normalization_22/moments/meanMeanadd:z:0>layer_normalization_22/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_22/moments/meanÎ
+layer_normalization_22/moments/StopGradientStopGradient,layer_normalization_22/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_22/moments/StopGradientî
0layer_normalization_22/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_22/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_22/moments/SquaredDifferenceÀ
9layer_normalization_22/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_22/moments/variance/reduction_indices
'layer_normalization_22/moments/varianceMean4layer_normalization_22/moments/SquaredDifference:z:0Blayer_normalization_22/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_22/moments/variance
&layer_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_22/batchnorm/add/yî
$layer_normalization_22/batchnorm/addAddV20layer_normalization_22/moments/variance:output:0/layer_normalization_22/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_22/batchnorm/add¹
&layer_normalization_22/batchnorm/RsqrtRsqrt(layer_normalization_22/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_22/batchnorm/Rsqrtã
3layer_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_22/batchnorm/mul/ReadVariableOpò
$layer_normalization_22/batchnorm/mulMul*layer_normalization_22/batchnorm/Rsqrt:y:0;layer_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_22/batchnorm/mulÀ
&layer_normalization_22/batchnorm/mul_1Muladd:z:0(layer_normalization_22/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_22/batchnorm/mul_1å
&layer_normalization_22/batchnorm/mul_2Mul,layer_normalization_22/moments/mean:output:0(layer_normalization_22/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_22/batchnorm/mul_2×
/layer_normalization_22/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_22_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_22/batchnorm/ReadVariableOpî
$layer_normalization_22/batchnorm/subSub7layer_normalization_22/batchnorm/ReadVariableOp:value:0*layer_normalization_22/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_22/batchnorm/subå
&layer_normalization_22/batchnorm/add_1AddV2*layer_normalization_22/batchnorm/mul_1:z:0(layer_normalization_22/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_22/batchnorm/add_1Û
/sequential_11/dense_37/Tensordot/ReadVariableOpReadVariableOp8sequential_11_dense_37_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype021
/sequential_11/dense_37/Tensordot/ReadVariableOp
%sequential_11/dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_11/dense_37/Tensordot/axes
%sequential_11/dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_11/dense_37/Tensordot/freeª
&sequential_11/dense_37/Tensordot/ShapeShape*layer_normalization_22/batchnorm/add_1:z:0*
T0*
_output_shapes
:2(
&sequential_11/dense_37/Tensordot/Shape¢
.sequential_11/dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_37/Tensordot/GatherV2/axisÄ
)sequential_11/dense_37/Tensordot/GatherV2GatherV2/sequential_11/dense_37/Tensordot/Shape:output:0.sequential_11/dense_37/Tensordot/free:output:07sequential_11/dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_11/dense_37/Tensordot/GatherV2¦
0sequential_11/dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_11/dense_37/Tensordot/GatherV2_1/axisÊ
+sequential_11/dense_37/Tensordot/GatherV2_1GatherV2/sequential_11/dense_37/Tensordot/Shape:output:0.sequential_11/dense_37/Tensordot/axes:output:09sequential_11/dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_11/dense_37/Tensordot/GatherV2_1
&sequential_11/dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_11/dense_37/Tensordot/ConstÜ
%sequential_11/dense_37/Tensordot/ProdProd2sequential_11/dense_37/Tensordot/GatherV2:output:0/sequential_11/dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_11/dense_37/Tensordot/Prod
(sequential_11/dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_11/dense_37/Tensordot/Const_1ä
'sequential_11/dense_37/Tensordot/Prod_1Prod4sequential_11/dense_37/Tensordot/GatherV2_1:output:01sequential_11/dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_11/dense_37/Tensordot/Prod_1
,sequential_11/dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_11/dense_37/Tensordot/concat/axis£
'sequential_11/dense_37/Tensordot/concatConcatV2.sequential_11/dense_37/Tensordot/free:output:0.sequential_11/dense_37/Tensordot/axes:output:05sequential_11/dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_11/dense_37/Tensordot/concatè
&sequential_11/dense_37/Tensordot/stackPack.sequential_11/dense_37/Tensordot/Prod:output:00sequential_11/dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_11/dense_37/Tensordot/stackù
*sequential_11/dense_37/Tensordot/transpose	Transpose*layer_normalization_22/batchnorm/add_1:z:00sequential_11/dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2,
*sequential_11/dense_37/Tensordot/transposeû
(sequential_11/dense_37/Tensordot/ReshapeReshape.sequential_11/dense_37/Tensordot/transpose:y:0/sequential_11/dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_11/dense_37/Tensordot/Reshapeú
'sequential_11/dense_37/Tensordot/MatMulMatMul1sequential_11/dense_37/Tensordot/Reshape:output:07sequential_11/dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2)
'sequential_11/dense_37/Tensordot/MatMul
(sequential_11/dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2*
(sequential_11/dense_37/Tensordot/Const_2¢
.sequential_11/dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_37/Tensordot/concat_1/axis°
)sequential_11/dense_37/Tensordot/concat_1ConcatV22sequential_11/dense_37/Tensordot/GatherV2:output:01sequential_11/dense_37/Tensordot/Const_2:output:07sequential_11/dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_11/dense_37/Tensordot/concat_1ì
 sequential_11/dense_37/TensordotReshape1sequential_11/dense_37/Tensordot/MatMul:product:02sequential_11/dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2"
 sequential_11/dense_37/TensordotÑ
-sequential_11/dense_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_11/dense_37/BiasAdd/ReadVariableOpã
sequential_11/dense_37/BiasAddBiasAdd)sequential_11/dense_37/Tensordot:output:05sequential_11/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2 
sequential_11/dense_37/BiasAdd¡
sequential_11/dense_37/ReluRelu'sequential_11/dense_37/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_11/dense_37/ReluÛ
/sequential_11/dense_38/Tensordot/ReadVariableOpReadVariableOp8sequential_11_dense_38_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype021
/sequential_11/dense_38/Tensordot/ReadVariableOp
%sequential_11/dense_38/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_11/dense_38/Tensordot/axes
%sequential_11/dense_38/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_11/dense_38/Tensordot/free©
&sequential_11/dense_38/Tensordot/ShapeShape)sequential_11/dense_37/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_11/dense_38/Tensordot/Shape¢
.sequential_11/dense_38/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_38/Tensordot/GatherV2/axisÄ
)sequential_11/dense_38/Tensordot/GatherV2GatherV2/sequential_11/dense_38/Tensordot/Shape:output:0.sequential_11/dense_38/Tensordot/free:output:07sequential_11/dense_38/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_11/dense_38/Tensordot/GatherV2¦
0sequential_11/dense_38/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_11/dense_38/Tensordot/GatherV2_1/axisÊ
+sequential_11/dense_38/Tensordot/GatherV2_1GatherV2/sequential_11/dense_38/Tensordot/Shape:output:0.sequential_11/dense_38/Tensordot/axes:output:09sequential_11/dense_38/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_11/dense_38/Tensordot/GatherV2_1
&sequential_11/dense_38/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_11/dense_38/Tensordot/ConstÜ
%sequential_11/dense_38/Tensordot/ProdProd2sequential_11/dense_38/Tensordot/GatherV2:output:0/sequential_11/dense_38/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_11/dense_38/Tensordot/Prod
(sequential_11/dense_38/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_11/dense_38/Tensordot/Const_1ä
'sequential_11/dense_38/Tensordot/Prod_1Prod4sequential_11/dense_38/Tensordot/GatherV2_1:output:01sequential_11/dense_38/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_11/dense_38/Tensordot/Prod_1
,sequential_11/dense_38/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_11/dense_38/Tensordot/concat/axis£
'sequential_11/dense_38/Tensordot/concatConcatV2.sequential_11/dense_38/Tensordot/free:output:0.sequential_11/dense_38/Tensordot/axes:output:05sequential_11/dense_38/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_11/dense_38/Tensordot/concatè
&sequential_11/dense_38/Tensordot/stackPack.sequential_11/dense_38/Tensordot/Prod:output:00sequential_11/dense_38/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_11/dense_38/Tensordot/stackø
*sequential_11/dense_38/Tensordot/transpose	Transpose)sequential_11/dense_37/Relu:activations:00sequential_11/dense_38/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2,
*sequential_11/dense_38/Tensordot/transposeû
(sequential_11/dense_38/Tensordot/ReshapeReshape.sequential_11/dense_38/Tensordot/transpose:y:0/sequential_11/dense_38/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_11/dense_38/Tensordot/Reshapeú
'sequential_11/dense_38/Tensordot/MatMulMatMul1sequential_11/dense_38/Tensordot/Reshape:output:07sequential_11/dense_38/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential_11/dense_38/Tensordot/MatMul
(sequential_11/dense_38/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_11/dense_38/Tensordot/Const_2¢
.sequential_11/dense_38/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_38/Tensordot/concat_1/axis°
)sequential_11/dense_38/Tensordot/concat_1ConcatV22sequential_11/dense_38/Tensordot/GatherV2:output:01sequential_11/dense_38/Tensordot/Const_2:output:07sequential_11/dense_38/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_11/dense_38/Tensordot/concat_1ì
 sequential_11/dense_38/TensordotReshape1sequential_11/dense_38/Tensordot/MatMul:product:02sequential_11/dense_38/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 sequential_11/dense_38/TensordotÑ
-sequential_11/dense_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_11/dense_38/BiasAdd/ReadVariableOpã
sequential_11/dense_38/BiasAddBiasAdd)sequential_11/dense_38/Tensordot:output:05sequential_11/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
sequential_11/dense_38/BiasAdd
dropout_33/IdentityIdentity'sequential_11/dense_38/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_33/Identity
add_1AddV2*layer_normalization_22/batchnorm/add_1:z:0dropout_33/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¸
5layer_normalization_23/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_23/moments/mean/reduction_indicesä
#layer_normalization_23/moments/meanMean	add_1:z:0>layer_normalization_23/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_23/moments/meanÎ
+layer_normalization_23/moments/StopGradientStopGradient,layer_normalization_23/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_23/moments/StopGradientð
0layer_normalization_23/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_23/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_23/moments/SquaredDifferenceÀ
9layer_normalization_23/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_23/moments/variance/reduction_indices
'layer_normalization_23/moments/varianceMean4layer_normalization_23/moments/SquaredDifference:z:0Blayer_normalization_23/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_23/moments/variance
&layer_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_23/batchnorm/add/yî
$layer_normalization_23/batchnorm/addAddV20layer_normalization_23/moments/variance:output:0/layer_normalization_23/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_23/batchnorm/add¹
&layer_normalization_23/batchnorm/RsqrtRsqrt(layer_normalization_23/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_23/batchnorm/Rsqrtã
3layer_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_23/batchnorm/mul/ReadVariableOpò
$layer_normalization_23/batchnorm/mulMul*layer_normalization_23/batchnorm/Rsqrt:y:0;layer_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_23/batchnorm/mulÂ
&layer_normalization_23/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_23/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_23/batchnorm/mul_1å
&layer_normalization_23/batchnorm/mul_2Mul,layer_normalization_23/moments/mean:output:0(layer_normalization_23/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_23/batchnorm/mul_2×
/layer_normalization_23/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_23_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_23/batchnorm/ReadVariableOpî
$layer_normalization_23/batchnorm/subSub7layer_normalization_23/batchnorm/ReadVariableOp:value:0*layer_normalization_23/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_23/batchnorm/subå
&layer_normalization_23/batchnorm/add_1AddV2*layer_normalization_23/batchnorm/mul_1:z:0(layer_normalization_23/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_23/batchnorm/add_1è
IdentityIdentity*layer_normalization_23/batchnorm/add_1:z:00^layer_normalization_22/batchnorm/ReadVariableOp4^layer_normalization_22/batchnorm/mul/ReadVariableOp0^layer_normalization_23/batchnorm/ReadVariableOp4^layer_normalization_23/batchnorm/mul/ReadVariableOp<^multi_head_attention_11/attention_output/add/ReadVariableOpF^multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp/^multi_head_attention_11/key/add/ReadVariableOp9^multi_head_attention_11/key/einsum/Einsum/ReadVariableOp1^multi_head_attention_11/query/add/ReadVariableOp;^multi_head_attention_11/query/einsum/Einsum/ReadVariableOp1^multi_head_attention_11/value/add/ReadVariableOp;^multi_head_attention_11/value/einsum/Einsum/ReadVariableOp.^sequential_11/dense_37/BiasAdd/ReadVariableOp0^sequential_11/dense_37/Tensordot/ReadVariableOp.^sequential_11/dense_38/BiasAdd/ReadVariableOp0^sequential_11/dense_38/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2b
/layer_normalization_22/batchnorm/ReadVariableOp/layer_normalization_22/batchnorm/ReadVariableOp2j
3layer_normalization_22/batchnorm/mul/ReadVariableOp3layer_normalization_22/batchnorm/mul/ReadVariableOp2b
/layer_normalization_23/batchnorm/ReadVariableOp/layer_normalization_23/batchnorm/ReadVariableOp2j
3layer_normalization_23/batchnorm/mul/ReadVariableOp3layer_normalization_23/batchnorm/mul/ReadVariableOp2z
;multi_head_attention_11/attention_output/add/ReadVariableOp;multi_head_attention_11/attention_output/add/ReadVariableOp2
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpEmulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp2`
.multi_head_attention_11/key/add/ReadVariableOp.multi_head_attention_11/key/add/ReadVariableOp2t
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_11/query/add/ReadVariableOp0multi_head_attention_11/query/add/ReadVariableOp2x
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_11/value/add/ReadVariableOp0multi_head_attention_11/value/add/ReadVariableOp2x
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp2^
-sequential_11/dense_37/BiasAdd/ReadVariableOp-sequential_11/dense_37/BiasAdd/ReadVariableOp2b
/sequential_11/dense_37/Tensordot/ReadVariableOp/sequential_11/dense_37/Tensordot/ReadVariableOp2^
-sequential_11/dense_38/BiasAdd/ReadVariableOp-sequential_11/dense_38/BiasAdd/ReadVariableOp2b
/sequential_11/dense_38/Tensordot/ReadVariableOp/sequential_11/dense_38/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs

e
F__inference_dropout_34_layer_call_and_return_conditional_losses_764805

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *d!?2
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
×£=2
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
ß
~
)__inference_dense_40_layer_call_fn_767004

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
D__inference_dense_40_layer_call_and_return_conditional_losses_7648342
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

ø
E__inference_conv1d_11_layer_call_and_return_conditional_losses_764137

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
ô0
É
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_766260

inputs
assignmovingavg_766235
assignmovingavg_1_766241)
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
loc:@AssignMovingAvg/766235*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_766235*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/766235*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/766235*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_766235AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/766235*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/766241*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_766241*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/766241*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/766241*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_766241AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/766241*
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
´
 
$__inference_signature_wrapper_765432
input_11
input_12
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
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinput_11input_12unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
!__inference__wrapped_model_7635512
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
input_11:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_12
î	
Ý
D__inference_dense_40_layer_call_and_return_conditional_losses_764834

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
ô0
É
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_766506

inputs
assignmovingavg_766481
assignmovingavg_1_766487)
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
loc:@AssignMovingAvg/766481*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_766481*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/766481*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/766481*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_766481AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/766481*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/766487*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_766487*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/766487*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/766487*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_766487AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/766487*
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
F__inference_dropout_34_layer_call_and_return_conditional_losses_764810

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
ËY
¿
C__inference_model_5_layer_call_and_return_conditional_losses_765271

inputs
inputs_1)
%token_and_position_embedding_5_765181)
%token_and_position_embedding_5_765183
conv1d_10_765186
conv1d_10_765188
conv1d_11_765192
conv1d_11_765194!
batch_normalization_10_765199!
batch_normalization_10_765201!
batch_normalization_10_765203!
batch_normalization_10_765205!
batch_normalization_11_765208!
batch_normalization_11_765210!
batch_normalization_11_765212!
batch_normalization_11_765214
transformer_block_11_765218
transformer_block_11_765220
transformer_block_11_765222
transformer_block_11_765224
transformer_block_11_765226
transformer_block_11_765228
transformer_block_11_765230
transformer_block_11_765232
transformer_block_11_765234
transformer_block_11_765236
transformer_block_11_765238
transformer_block_11_765240
transformer_block_11_765242
transformer_block_11_765244
transformer_block_11_765246
transformer_block_11_765248
dense_39_765253
dense_39_765255
dense_40_765259
dense_40_765261
dense_41_765265
dense_41_765267
identity¢.batch_normalization_10/StatefulPartitionedCall¢.batch_normalization_11/StatefulPartitionedCall¢!conv1d_10/StatefulPartitionedCall¢!conv1d_11/StatefulPartitionedCall¢ dense_39/StatefulPartitionedCall¢ dense_40/StatefulPartitionedCall¢ dense_41/StatefulPartitionedCall¢6token_and_position_embedding_5/StatefulPartitionedCall¢,transformer_block_11/StatefulPartitionedCall
6token_and_position_embedding_5/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_5_765181%token_and_position_embedding_5_765183*
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
Z__inference_token_and_position_embedding_5_layer_call_and_return_conditional_losses_76407228
6token_and_position_embedding_5/StatefulPartitionedCallÚ
!conv1d_10/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_5/StatefulPartitionedCall:output:0conv1d_10_765186conv1d_10_765188*
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
GPU2*0J 8 *N
fIRG
E__inference_conv1d_10_layer_call_and_return_conditional_losses_7641042#
!conv1d_10/StatefulPartitionedCall¤
$average_pooling1d_15/PartitionedCallPartitionedCall*conv1d_10/StatefulPartitionedCall:output:0*
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
P__inference_average_pooling1d_15_layer_call_and_return_conditional_losses_7635602&
$average_pooling1d_15/PartitionedCallÈ
!conv1d_11/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_15/PartitionedCall:output:0conv1d_11_765192conv1d_11_765194*
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
GPU2*0J 8 *N
fIRG
E__inference_conv1d_11_layer_call_and_return_conditional_losses_7641372#
!conv1d_11/StatefulPartitionedCall¸
$average_pooling1d_17/PartitionedCallPartitionedCall?token_and_position_embedding_5/StatefulPartitionedCall:output:0*
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
P__inference_average_pooling1d_17_layer_call_and_return_conditional_losses_7635902&
$average_pooling1d_17/PartitionedCall£
$average_pooling1d_16/PartitionedCallPartitionedCall*conv1d_11/StatefulPartitionedCall:output:0*
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
P__inference_average_pooling1d_16_layer_call_and_return_conditional_losses_7635752&
$average_pooling1d_16/PartitionedCallÊ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_16/PartitionedCall:output:0batch_normalization_10_765199batch_normalization_10_765201batch_normalization_10_765203batch_normalization_10_765205*
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
GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_76421020
.batch_normalization_10/StatefulPartitionedCallÊ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_17/PartitionedCall:output:0batch_normalization_11_765208batch_normalization_11_765210batch_normalization_11_765212batch_normalization_11_765214*
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
GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_76430120
.batch_normalization_11/StatefulPartitionedCall½
add_5/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:07batch_normalization_11/StatefulPartitionedCall:output:0*
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
A__inference_add_5_layer_call_and_return_conditional_losses_7643432
add_5/PartitionedCall¡
,transformer_block_11/StatefulPartitionedCallStatefulPartitionedCalladd_5/PartitionedCall:output:0transformer_block_11_765218transformer_block_11_765220transformer_block_11_765222transformer_block_11_765224transformer_block_11_765226transformer_block_11_765228transformer_block_11_765230transformer_block_11_765232transformer_block_11_765234transformer_block_11_765236transformer_block_11_765238transformer_block_11_765240transformer_block_11_765242transformer_block_11_765244transformer_block_11_765246transformer_block_11_765248*
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
GPU2*0J 8 *Y
fTRR
P__inference_transformer_block_11_layer_call_and_return_conditional_losses_7646272.
,transformer_block_11/StatefulPartitionedCall
flatten_5/PartitionedCallPartitionedCall5transformer_block_11/StatefulPartitionedCall:output:0*
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_7647422
flatten_5/PartitionedCall
concatenate_5/PartitionedCallPartitionedCall"flatten_5/PartitionedCall:output:0inputs_1*
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
I__inference_concatenate_5_layer_call_and_return_conditional_losses_7647572
concatenate_5/PartitionedCall·
 dense_39/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_39_765253dense_39_765255*
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
D__inference_dense_39_layer_call_and_return_conditional_losses_7647772"
 dense_39/StatefulPartitionedCall
dropout_34/PartitionedCallPartitionedCall)dense_39/StatefulPartitionedCall:output:0*
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
F__inference_dropout_34_layer_call_and_return_conditional_losses_7648102
dropout_34/PartitionedCall´
 dense_40/StatefulPartitionedCallStatefulPartitionedCall#dropout_34/PartitionedCall:output:0dense_40_765259dense_40_765261*
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
D__inference_dense_40_layer_call_and_return_conditional_losses_7648342"
 dense_40/StatefulPartitionedCall
dropout_35/PartitionedCallPartitionedCall)dense_40/StatefulPartitionedCall:output:0*
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
F__inference_dropout_35_layer_call_and_return_conditional_losses_7648672
dropout_35/PartitionedCall´
 dense_41/StatefulPartitionedCallStatefulPartitionedCall#dropout_35/PartitionedCall:output:0dense_41_765265dense_41_765267*
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
D__inference_dense_41_layer_call_and_return_conditional_losses_7648902"
 dense_41/StatefulPartitionedCallø
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall"^conv1d_10/StatefulPartitionedCall"^conv1d_11/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall7^token_and_position_embedding_5/StatefulPartitionedCall-^transformer_block_11/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2F
!conv1d_10/StatefulPartitionedCall!conv1d_10/StatefulPartitionedCall2F
!conv1d_11/StatefulPartitionedCall!conv1d_11/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2p
6token_and_position_embedding_5/StatefulPartitionedCall6token_and_position_embedding_5/StatefulPartitionedCall2\
,transformer_block_11/StatefulPartitionedCall,transformer_block_11/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
ª
7__inference_batch_normalization_10_layer_call_fn_766306

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¬
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
GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7637252
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
Ö
¤
(__inference_model_5_layer_call_fn_765174
input_11
input_12
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
StatefulPartitionedCallStatefulPartitionedCallinput_11input_12unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
C__inference_model_5_layer_call_and_return_conditional_losses_7650992
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
input_11:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_12
ô0
É
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_763692

inputs
assignmovingavg_763667
assignmovingavg_1_763673)
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
loc:@AssignMovingAvg/763667*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_763667*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/763667*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/763667*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_763667AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/763667*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/763673*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_763673*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/763673*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/763673*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_763673AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/763673*
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
Ê
ª
7__inference_batch_normalization_10_layer_call_fn_766375

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¡
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
GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7641902
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
)__inference_dense_41_layer_call_fn_767050

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
D__inference_dense_41_layer_call_and_return_conditional_losses_7648902
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
·
k
A__inference_add_5_layer_call_and_return_conditional_losses_764343

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
í
¾)
!__inference__wrapped_model_763551
input_11
input_12O
Kmodel_5_token_and_position_embedding_5_embedding_11_embedding_lookup_763320O
Kmodel_5_token_and_position_embedding_5_embedding_10_embedding_lookup_763326A
=model_5_conv1d_10_conv1d_expanddims_1_readvariableop_resource5
1model_5_conv1d_10_biasadd_readvariableop_resourceA
=model_5_conv1d_11_conv1d_expanddims_1_readvariableop_resource5
1model_5_conv1d_11_biasadd_readvariableop_resourceD
@model_5_batch_normalization_10_batchnorm_readvariableop_resourceH
Dmodel_5_batch_normalization_10_batchnorm_mul_readvariableop_resourceF
Bmodel_5_batch_normalization_10_batchnorm_readvariableop_1_resourceF
Bmodel_5_batch_normalization_10_batchnorm_readvariableop_2_resourceD
@model_5_batch_normalization_11_batchnorm_readvariableop_resourceH
Dmodel_5_batch_normalization_11_batchnorm_mul_readvariableop_resourceF
Bmodel_5_batch_normalization_11_batchnorm_readvariableop_1_resourceF
Bmodel_5_batch_normalization_11_batchnorm_readvariableop_2_resourced
`model_5_transformer_block_11_multi_head_attention_11_query_einsum_einsum_readvariableop_resourceZ
Vmodel_5_transformer_block_11_multi_head_attention_11_query_add_readvariableop_resourceb
^model_5_transformer_block_11_multi_head_attention_11_key_einsum_einsum_readvariableop_resourceX
Tmodel_5_transformer_block_11_multi_head_attention_11_key_add_readvariableop_resourced
`model_5_transformer_block_11_multi_head_attention_11_value_einsum_einsum_readvariableop_resourceZ
Vmodel_5_transformer_block_11_multi_head_attention_11_value_add_readvariableop_resourceo
kmodel_5_transformer_block_11_multi_head_attention_11_attention_output_einsum_einsum_readvariableop_resourcee
amodel_5_transformer_block_11_multi_head_attention_11_attention_output_add_readvariableop_resource]
Ymodel_5_transformer_block_11_layer_normalization_22_batchnorm_mul_readvariableop_resourceY
Umodel_5_transformer_block_11_layer_normalization_22_batchnorm_readvariableop_resourceY
Umodel_5_transformer_block_11_sequential_11_dense_37_tensordot_readvariableop_resourceW
Smodel_5_transformer_block_11_sequential_11_dense_37_biasadd_readvariableop_resourceY
Umodel_5_transformer_block_11_sequential_11_dense_38_tensordot_readvariableop_resourceW
Smodel_5_transformer_block_11_sequential_11_dense_38_biasadd_readvariableop_resource]
Ymodel_5_transformer_block_11_layer_normalization_23_batchnorm_mul_readvariableop_resourceY
Umodel_5_transformer_block_11_layer_normalization_23_batchnorm_readvariableop_resource3
/model_5_dense_39_matmul_readvariableop_resource4
0model_5_dense_39_biasadd_readvariableop_resource3
/model_5_dense_40_matmul_readvariableop_resource4
0model_5_dense_40_biasadd_readvariableop_resource3
/model_5_dense_41_matmul_readvariableop_resource4
0model_5_dense_41_biasadd_readvariableop_resource
identity¢7model_5/batch_normalization_10/batchnorm/ReadVariableOp¢9model_5/batch_normalization_10/batchnorm/ReadVariableOp_1¢9model_5/batch_normalization_10/batchnorm/ReadVariableOp_2¢;model_5/batch_normalization_10/batchnorm/mul/ReadVariableOp¢7model_5/batch_normalization_11/batchnorm/ReadVariableOp¢9model_5/batch_normalization_11/batchnorm/ReadVariableOp_1¢9model_5/batch_normalization_11/batchnorm/ReadVariableOp_2¢;model_5/batch_normalization_11/batchnorm/mul/ReadVariableOp¢(model_5/conv1d_10/BiasAdd/ReadVariableOp¢4model_5/conv1d_10/conv1d/ExpandDims_1/ReadVariableOp¢(model_5/conv1d_11/BiasAdd/ReadVariableOp¢4model_5/conv1d_11/conv1d/ExpandDims_1/ReadVariableOp¢'model_5/dense_39/BiasAdd/ReadVariableOp¢&model_5/dense_39/MatMul/ReadVariableOp¢'model_5/dense_40/BiasAdd/ReadVariableOp¢&model_5/dense_40/MatMul/ReadVariableOp¢'model_5/dense_41/BiasAdd/ReadVariableOp¢&model_5/dense_41/MatMul/ReadVariableOp¢Dmodel_5/token_and_position_embedding_5/embedding_10/embedding_lookup¢Dmodel_5/token_and_position_embedding_5/embedding_11/embedding_lookup¢Lmodel_5/transformer_block_11/layer_normalization_22/batchnorm/ReadVariableOp¢Pmodel_5/transformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOp¢Lmodel_5/transformer_block_11/layer_normalization_23/batchnorm/ReadVariableOp¢Pmodel_5/transformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOp¢Xmodel_5/transformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOp¢bmodel_5/transformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp¢Kmodel_5/transformer_block_11/multi_head_attention_11/key/add/ReadVariableOp¢Umodel_5/transformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOp¢Mmodel_5/transformer_block_11/multi_head_attention_11/query/add/ReadVariableOp¢Wmodel_5/transformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp¢Mmodel_5/transformer_block_11/multi_head_attention_11/value/add/ReadVariableOp¢Wmodel_5/transformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp¢Jmodel_5/transformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOp¢Lmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOp¢Jmodel_5/transformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOp¢Lmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOp
,model_5/token_and_position_embedding_5/ShapeShapeinput_11*
T0*
_output_shapes
:2.
,model_5/token_and_position_embedding_5/ShapeË
:model_5/token_and_position_embedding_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2<
:model_5/token_and_position_embedding_5/strided_slice/stackÆ
<model_5/token_and_position_embedding_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<model_5/token_and_position_embedding_5/strided_slice/stack_1Æ
<model_5/token_and_position_embedding_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model_5/token_and_position_embedding_5/strided_slice/stack_2Ì
4model_5/token_and_position_embedding_5/strided_sliceStridedSlice5model_5/token_and_position_embedding_5/Shape:output:0Cmodel_5/token_and_position_embedding_5/strided_slice/stack:output:0Emodel_5/token_and_position_embedding_5/strided_slice/stack_1:output:0Emodel_5/token_and_position_embedding_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4model_5/token_and_position_embedding_5/strided_sliceª
2model_5/token_and_position_embedding_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : 24
2model_5/token_and_position_embedding_5/range/startª
2model_5/token_and_position_embedding_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :24
2model_5/token_and_position_embedding_5/range/deltaÃ
,model_5/token_and_position_embedding_5/rangeRange;model_5/token_and_position_embedding_5/range/start:output:0=model_5/token_and_position_embedding_5/strided_slice:output:0;model_5/token_and_position_embedding_5/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,model_5/token_and_position_embedding_5/rangeö
Dmodel_5/token_and_position_embedding_5/embedding_11/embedding_lookupResourceGatherKmodel_5_token_and_position_embedding_5_embedding_11_embedding_lookup_7633205model_5/token_and_position_embedding_5/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*^
_classT
RPloc:@model_5/token_and_position_embedding_5/embedding_11/embedding_lookup/763320*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02F
Dmodel_5/token_and_position_embedding_5/embedding_11/embedding_lookup¹
Mmodel_5/token_and_position_embedding_5/embedding_11/embedding_lookup/IdentityIdentityMmodel_5/token_and_position_embedding_5/embedding_11/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*^
_classT
RPloc:@model_5/token_and_position_embedding_5/embedding_11/embedding_lookup/763320*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2O
Mmodel_5/token_and_position_embedding_5/embedding_11/embedding_lookup/Identity¸
Omodel_5/token_and_position_embedding_5/embedding_11/embedding_lookup/Identity_1IdentityVmodel_5/token_and_position_embedding_5/embedding_11/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2Q
Omodel_5/token_and_position_embedding_5/embedding_11/embedding_lookup/Identity_1È
8model_5/token_and_position_embedding_5/embedding_10/CastCastinput_11*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2:
8model_5/token_and_position_embedding_5/embedding_10/Cast
Dmodel_5/token_and_position_embedding_5/embedding_10/embedding_lookupResourceGatherKmodel_5_token_and_position_embedding_5_embedding_10_embedding_lookup_763326<model_5/token_and_position_embedding_5/embedding_10/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*^
_classT
RPloc:@model_5/token_and_position_embedding_5/embedding_10/embedding_lookup/763326*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02F
Dmodel_5/token_and_position_embedding_5/embedding_10/embedding_lookup¾
Mmodel_5/token_and_position_embedding_5/embedding_10/embedding_lookup/IdentityIdentityMmodel_5/token_and_position_embedding_5/embedding_10/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*^
_classT
RPloc:@model_5/token_and_position_embedding_5/embedding_10/embedding_lookup/763326*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2O
Mmodel_5/token_and_position_embedding_5/embedding_10/embedding_lookup/Identity½
Omodel_5/token_and_position_embedding_5/embedding_10/embedding_lookup/Identity_1IdentityVmodel_5/token_and_position_embedding_5/embedding_10/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2Q
Omodel_5/token_and_position_embedding_5/embedding_10/embedding_lookup/Identity_1Ì
*model_5/token_and_position_embedding_5/addAddV2Xmodel_5/token_and_position_embedding_5/embedding_10/embedding_lookup/Identity_1:output:0Xmodel_5/token_and_position_embedding_5/embedding_11/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2,
*model_5/token_and_position_embedding_5/add
'model_5/conv1d_10/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2)
'model_5/conv1d_10/conv1d/ExpandDims/dimõ
#model_5/conv1d_10/conv1d/ExpandDims
ExpandDims.model_5/token_and_position_embedding_5/add:z:00model_5/conv1d_10/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2%
#model_5/conv1d_10/conv1d/ExpandDimsî
4model_5/conv1d_10/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_5_conv1d_10_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype026
4model_5/conv1d_10/conv1d/ExpandDims_1/ReadVariableOp
)model_5/conv1d_10/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_5/conv1d_10/conv1d/ExpandDims_1/dimÿ
%model_5/conv1d_10/conv1d/ExpandDims_1
ExpandDims<model_5/conv1d_10/conv1d/ExpandDims_1/ReadVariableOp:value:02model_5/conv1d_10/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2'
%model_5/conv1d_10/conv1d/ExpandDims_1ÿ
model_5/conv1d_10/conv1dConv2D,model_5/conv1d_10/conv1d/ExpandDims:output:0.model_5/conv1d_10/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
paddingSAME*
strides
2
model_5/conv1d_10/conv1dÉ
 model_5/conv1d_10/conv1d/SqueezeSqueeze!model_5/conv1d_10/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2"
 model_5/conv1d_10/conv1d/SqueezeÂ
(model_5/conv1d_10/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv1d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_5/conv1d_10/BiasAdd/ReadVariableOpÕ
model_5/conv1d_10/BiasAddBiasAdd)model_5/conv1d_10/conv1d/Squeeze:output:00model_5/conv1d_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
model_5/conv1d_10/BiasAdd
model_5/conv1d_10/ReluRelu"model_5/conv1d_10/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
model_5/conv1d_10/Relu
+model_5/average_pooling1d_15/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+model_5/average_pooling1d_15/ExpandDims/dim÷
'model_5/average_pooling1d_15/ExpandDims
ExpandDims$model_5/conv1d_10/Relu:activations:04model_5/average_pooling1d_15/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2)
'model_5/average_pooling1d_15/ExpandDims
$model_5/average_pooling1d_15/AvgPoolAvgPool0model_5/average_pooling1d_15/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
ksize
*
paddingVALID*
strides
2&
$model_5/average_pooling1d_15/AvgPoolÔ
$model_5/average_pooling1d_15/SqueezeSqueeze-model_5/average_pooling1d_15/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims
2&
$model_5/average_pooling1d_15/Squeeze
'model_5/conv1d_11/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2)
'model_5/conv1d_11/conv1d/ExpandDims/dimô
#model_5/conv1d_11/conv1d/ExpandDims
ExpandDims-model_5/average_pooling1d_15/Squeeze:output:00model_5/conv1d_11/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2%
#model_5/conv1d_11/conv1d/ExpandDimsî
4model_5/conv1d_11/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_5_conv1d_11_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype026
4model_5/conv1d_11/conv1d/ExpandDims_1/ReadVariableOp
)model_5/conv1d_11/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_5/conv1d_11/conv1d/ExpandDims_1/dimÿ
%model_5/conv1d_11/conv1d/ExpandDims_1
ExpandDims<model_5/conv1d_11/conv1d/ExpandDims_1/ReadVariableOp:value:02model_5/conv1d_11/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2'
%model_5/conv1d_11/conv1d/ExpandDims_1ÿ
model_5/conv1d_11/conv1dConv2D,model_5/conv1d_11/conv1d/ExpandDims:output:0.model_5/conv1d_11/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
paddingSAME*
strides
2
model_5/conv1d_11/conv1dÉ
 model_5/conv1d_11/conv1d/SqueezeSqueeze!model_5/conv1d_11/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2"
 model_5/conv1d_11/conv1d/SqueezeÂ
(model_5/conv1d_11/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv1d_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_5/conv1d_11/BiasAdd/ReadVariableOpÕ
model_5/conv1d_11/BiasAddBiasAdd)model_5/conv1d_11/conv1d/Squeeze:output:00model_5/conv1d_11/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
model_5/conv1d_11/BiasAdd
model_5/conv1d_11/ReluRelu"model_5/conv1d_11/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
model_5/conv1d_11/Relu
+model_5/average_pooling1d_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+model_5/average_pooling1d_17/ExpandDims/dim
'model_5/average_pooling1d_17/ExpandDims
ExpandDims.model_5/token_and_position_embedding_5/add:z:04model_5/average_pooling1d_17/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2)
'model_5/average_pooling1d_17/ExpandDims
$model_5/average_pooling1d_17/AvgPoolAvgPool0model_5/average_pooling1d_17/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize	
¬*
paddingVALID*
strides	
¬2&
$model_5/average_pooling1d_17/AvgPoolÓ
$model_5/average_pooling1d_17/SqueezeSqueeze-model_5/average_pooling1d_17/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2&
$model_5/average_pooling1d_17/Squeeze
+model_5/average_pooling1d_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+model_5/average_pooling1d_16/ExpandDims/dim÷
'model_5/average_pooling1d_16/ExpandDims
ExpandDims$model_5/conv1d_11/Relu:activations:04model_5/average_pooling1d_16/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2)
'model_5/average_pooling1d_16/ExpandDimsÿ
$model_5/average_pooling1d_16/AvgPoolAvgPool0model_5/average_pooling1d_16/ExpandDims:output:0*
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
$model_5/average_pooling1d_16/AvgPoolÓ
$model_5/average_pooling1d_16/SqueezeSqueeze-model_5/average_pooling1d_16/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2&
$model_5/average_pooling1d_16/Squeezeï
7model_5/batch_normalization_10/batchnorm/ReadVariableOpReadVariableOp@model_5_batch_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype029
7model_5/batch_normalization_10/batchnorm/ReadVariableOp¥
.model_5/batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.model_5/batch_normalization_10/batchnorm/add/y
,model_5/batch_normalization_10/batchnorm/addAddV2?model_5/batch_normalization_10/batchnorm/ReadVariableOp:value:07model_5/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2.
,model_5/batch_normalization_10/batchnorm/addÀ
.model_5/batch_normalization_10/batchnorm/RsqrtRsqrt0model_5/batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
: 20
.model_5/batch_normalization_10/batchnorm/Rsqrtû
;model_5/batch_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_5_batch_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02=
;model_5/batch_normalization_10/batchnorm/mul/ReadVariableOp
,model_5/batch_normalization_10/batchnorm/mulMul2model_5/batch_normalization_10/batchnorm/Rsqrt:y:0Cmodel_5/batch_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,model_5/batch_normalization_10/batchnorm/mulþ
.model_5/batch_normalization_10/batchnorm/mul_1Mul-model_5/average_pooling1d_16/Squeeze:output:00model_5/batch_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 20
.model_5/batch_normalization_10/batchnorm/mul_1õ
9model_5/batch_normalization_10/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_5_batch_normalization_10_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02;
9model_5/batch_normalization_10/batchnorm/ReadVariableOp_1
.model_5/batch_normalization_10/batchnorm/mul_2MulAmodel_5/batch_normalization_10/batchnorm/ReadVariableOp_1:value:00model_5/batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
: 20
.model_5/batch_normalization_10/batchnorm/mul_2õ
9model_5/batch_normalization_10/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_5_batch_normalization_10_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02;
9model_5/batch_normalization_10/batchnorm/ReadVariableOp_2ÿ
,model_5/batch_normalization_10/batchnorm/subSubAmodel_5/batch_normalization_10/batchnorm/ReadVariableOp_2:value:02model_5/batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2.
,model_5/batch_normalization_10/batchnorm/sub
.model_5/batch_normalization_10/batchnorm/add_1AddV22model_5/batch_normalization_10/batchnorm/mul_1:z:00model_5/batch_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 20
.model_5/batch_normalization_10/batchnorm/add_1ï
7model_5/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp@model_5_batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype029
7model_5/batch_normalization_11/batchnorm/ReadVariableOp¥
.model_5/batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.model_5/batch_normalization_11/batchnorm/add/y
,model_5/batch_normalization_11/batchnorm/addAddV2?model_5/batch_normalization_11/batchnorm/ReadVariableOp:value:07model_5/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2.
,model_5/batch_normalization_11/batchnorm/addÀ
.model_5/batch_normalization_11/batchnorm/RsqrtRsqrt0model_5/batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
: 20
.model_5/batch_normalization_11/batchnorm/Rsqrtû
;model_5/batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_5_batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02=
;model_5/batch_normalization_11/batchnorm/mul/ReadVariableOp
,model_5/batch_normalization_11/batchnorm/mulMul2model_5/batch_normalization_11/batchnorm/Rsqrt:y:0Cmodel_5/batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,model_5/batch_normalization_11/batchnorm/mulþ
.model_5/batch_normalization_11/batchnorm/mul_1Mul-model_5/average_pooling1d_17/Squeeze:output:00model_5/batch_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 20
.model_5/batch_normalization_11/batchnorm/mul_1õ
9model_5/batch_normalization_11/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_5_batch_normalization_11_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02;
9model_5/batch_normalization_11/batchnorm/ReadVariableOp_1
.model_5/batch_normalization_11/batchnorm/mul_2MulAmodel_5/batch_normalization_11/batchnorm/ReadVariableOp_1:value:00model_5/batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
: 20
.model_5/batch_normalization_11/batchnorm/mul_2õ
9model_5/batch_normalization_11/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_5_batch_normalization_11_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02;
9model_5/batch_normalization_11/batchnorm/ReadVariableOp_2ÿ
,model_5/batch_normalization_11/batchnorm/subSubAmodel_5/batch_normalization_11/batchnorm/ReadVariableOp_2:value:02model_5/batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2.
,model_5/batch_normalization_11/batchnorm/sub
.model_5/batch_normalization_11/batchnorm/add_1AddV22model_5/batch_normalization_11/batchnorm/mul_1:z:00model_5/batch_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 20
.model_5/batch_normalization_11/batchnorm/add_1Í
model_5/add_5/addAddV22model_5/batch_normalization_10/batchnorm/add_1:z:02model_5/batch_normalization_11/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
model_5/add_5/add×
Wmodel_5/transformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOpReadVariableOp`model_5_transformer_block_11_multi_head_attention_11_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Y
Wmodel_5/transformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOpö
Hmodel_5/transformer_block_11/multi_head_attention_11/query/einsum/EinsumEinsummodel_5/add_5/add:z:0_model_5/transformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2J
Hmodel_5/transformer_block_11/multi_head_attention_11/query/einsum/Einsumµ
Mmodel_5/transformer_block_11/multi_head_attention_11/query/add/ReadVariableOpReadVariableOpVmodel_5_transformer_block_11_multi_head_attention_11_query_add_readvariableop_resource*
_output_shapes

: *
dtype02O
Mmodel_5/transformer_block_11/multi_head_attention_11/query/add/ReadVariableOpí
>model_5/transformer_block_11/multi_head_attention_11/query/addAddV2Qmodel_5/transformer_block_11/multi_head_attention_11/query/einsum/Einsum:output:0Umodel_5/transformer_block_11/multi_head_attention_11/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2@
>model_5/transformer_block_11/multi_head_attention_11/query/addÑ
Umodel_5/transformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpReadVariableOp^model_5_transformer_block_11_multi_head_attention_11_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_5/transformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpð
Fmodel_5/transformer_block_11/multi_head_attention_11/key/einsum/EinsumEinsummodel_5/add_5/add:z:0]model_5/transformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2H
Fmodel_5/transformer_block_11/multi_head_attention_11/key/einsum/Einsum¯
Kmodel_5/transformer_block_11/multi_head_attention_11/key/add/ReadVariableOpReadVariableOpTmodel_5_transformer_block_11_multi_head_attention_11_key_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_5/transformer_block_11/multi_head_attention_11/key/add/ReadVariableOpå
<model_5/transformer_block_11/multi_head_attention_11/key/addAddV2Omodel_5/transformer_block_11/multi_head_attention_11/key/einsum/Einsum:output:0Smodel_5/transformer_block_11/multi_head_attention_11/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2>
<model_5/transformer_block_11/multi_head_attention_11/key/add×
Wmodel_5/transformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOpReadVariableOp`model_5_transformer_block_11_multi_head_attention_11_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Y
Wmodel_5/transformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOpö
Hmodel_5/transformer_block_11/multi_head_attention_11/value/einsum/EinsumEinsummodel_5/add_5/add:z:0_model_5/transformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2J
Hmodel_5/transformer_block_11/multi_head_attention_11/value/einsum/Einsumµ
Mmodel_5/transformer_block_11/multi_head_attention_11/value/add/ReadVariableOpReadVariableOpVmodel_5_transformer_block_11_multi_head_attention_11_value_add_readvariableop_resource*
_output_shapes

: *
dtype02O
Mmodel_5/transformer_block_11/multi_head_attention_11/value/add/ReadVariableOpí
>model_5/transformer_block_11/multi_head_attention_11/value/addAddV2Qmodel_5/transformer_block_11/multi_head_attention_11/value/einsum/Einsum:output:0Umodel_5/transformer_block_11/multi_head_attention_11/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2@
>model_5/transformer_block_11/multi_head_attention_11/value/add½
:model_5/transformer_block_11/multi_head_attention_11/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2<
:model_5/transformer_block_11/multi_head_attention_11/Mul/y¾
8model_5/transformer_block_11/multi_head_attention_11/MulMulBmodel_5/transformer_block_11/multi_head_attention_11/query/add:z:0Cmodel_5/transformer_block_11/multi_head_attention_11/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2:
8model_5/transformer_block_11/multi_head_attention_11/Mulô
Bmodel_5/transformer_block_11/multi_head_attention_11/einsum/EinsumEinsum@model_5/transformer_block_11/multi_head_attention_11/key/add:z:0<model_5/transformer_block_11/multi_head_attention_11/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2D
Bmodel_5/transformer_block_11/multi_head_attention_11/einsum/Einsum
Dmodel_5/transformer_block_11/multi_head_attention_11/softmax/SoftmaxSoftmaxKmodel_5/transformer_block_11/multi_head_attention_11/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2F
Dmodel_5/transformer_block_11/multi_head_attention_11/softmax/Softmax¤
Emodel_5/transformer_block_11/multi_head_attention_11/dropout/IdentityIdentityNmodel_5/transformer_block_11/multi_head_attention_11/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2G
Emodel_5/transformer_block_11/multi_head_attention_11/dropout/Identity
Dmodel_5/transformer_block_11/multi_head_attention_11/einsum_1/EinsumEinsumNmodel_5/transformer_block_11/multi_head_attention_11/dropout/Identity:output:0Bmodel_5/transformer_block_11/multi_head_attention_11/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2F
Dmodel_5/transformer_block_11/multi_head_attention_11/einsum_1/Einsumø
bmodel_5/transformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpkmodel_5_transformer_block_11_multi_head_attention_11_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02d
bmodel_5/transformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpË
Smodel_5/transformer_block_11/multi_head_attention_11/attention_output/einsum/EinsumEinsumMmodel_5/transformer_block_11/multi_head_attention_11/einsum_1/Einsum:output:0jmodel_5/transformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe2U
Smodel_5/transformer_block_11/multi_head_attention_11/attention_output/einsum/EinsumÒ
Xmodel_5/transformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOpReadVariableOpamodel_5_transformer_block_11_multi_head_attention_11_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02Z
Xmodel_5/transformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOp
Imodel_5/transformer_block_11/multi_head_attention_11/attention_output/addAddV2\model_5/transformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum:output:0`model_5/transformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2K
Imodel_5/transformer_block_11/multi_head_attention_11/attention_output/addõ
0model_5/transformer_block_11/dropout_32/IdentityIdentityMmodel_5/transformer_block_11/multi_head_attention_11/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0model_5/transformer_block_11/dropout_32/IdentityÕ
 model_5/transformer_block_11/addAddV2model_5/add_5/add:z:09model_5/transformer_block_11/dropout_32/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 model_5/transformer_block_11/addò
Rmodel_5/transformer_block_11/layer_normalization_22/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2T
Rmodel_5/transformer_block_11/layer_normalization_22/moments/mean/reduction_indicesÖ
@model_5/transformer_block_11/layer_normalization_22/moments/meanMean$model_5/transformer_block_11/add:z:0[model_5/transformer_block_11/layer_normalization_22/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2B
@model_5/transformer_block_11/layer_normalization_22/moments/mean¥
Hmodel_5/transformer_block_11/layer_normalization_22/moments/StopGradientStopGradientImodel_5/transformer_block_11/layer_normalization_22/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2J
Hmodel_5/transformer_block_11/layer_normalization_22/moments/StopGradientâ
Mmodel_5/transformer_block_11/layer_normalization_22/moments/SquaredDifferenceSquaredDifference$model_5/transformer_block_11/add:z:0Qmodel_5/transformer_block_11/layer_normalization_22/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2O
Mmodel_5/transformer_block_11/layer_normalization_22/moments/SquaredDifferenceú
Vmodel_5/transformer_block_11/layer_normalization_22/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2X
Vmodel_5/transformer_block_11/layer_normalization_22/moments/variance/reduction_indices
Dmodel_5/transformer_block_11/layer_normalization_22/moments/varianceMeanQmodel_5/transformer_block_11/layer_normalization_22/moments/SquaredDifference:z:0_model_5/transformer_block_11/layer_normalization_22/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2F
Dmodel_5/transformer_block_11/layer_normalization_22/moments/varianceÏ
Cmodel_5/transformer_block_11/layer_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752E
Cmodel_5/transformer_block_11/layer_normalization_22/batchnorm/add/yâ
Amodel_5/transformer_block_11/layer_normalization_22/batchnorm/addAddV2Mmodel_5/transformer_block_11/layer_normalization_22/moments/variance:output:0Lmodel_5/transformer_block_11/layer_normalization_22/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2C
Amodel_5/transformer_block_11/layer_normalization_22/batchnorm/add
Cmodel_5/transformer_block_11/layer_normalization_22/batchnorm/RsqrtRsqrtEmodel_5/transformer_block_11/layer_normalization_22/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2E
Cmodel_5/transformer_block_11/layer_normalization_22/batchnorm/Rsqrtº
Pmodel_5/transformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOpYmodel_5_transformer_block_11_layer_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02R
Pmodel_5/transformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOpæ
Amodel_5/transformer_block_11/layer_normalization_22/batchnorm/mulMulGmodel_5/transformer_block_11/layer_normalization_22/batchnorm/Rsqrt:y:0Xmodel_5/transformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2C
Amodel_5/transformer_block_11/layer_normalization_22/batchnorm/mul´
Cmodel_5/transformer_block_11/layer_normalization_22/batchnorm/mul_1Mul$model_5/transformer_block_11/add:z:0Emodel_5/transformer_block_11/layer_normalization_22/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2E
Cmodel_5/transformer_block_11/layer_normalization_22/batchnorm/mul_1Ù
Cmodel_5/transformer_block_11/layer_normalization_22/batchnorm/mul_2MulImodel_5/transformer_block_11/layer_normalization_22/moments/mean:output:0Emodel_5/transformer_block_11/layer_normalization_22/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2E
Cmodel_5/transformer_block_11/layer_normalization_22/batchnorm/mul_2®
Lmodel_5/transformer_block_11/layer_normalization_22/batchnorm/ReadVariableOpReadVariableOpUmodel_5_transformer_block_11_layer_normalization_22_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02N
Lmodel_5/transformer_block_11/layer_normalization_22/batchnorm/ReadVariableOpâ
Amodel_5/transformer_block_11/layer_normalization_22/batchnorm/subSubTmodel_5/transformer_block_11/layer_normalization_22/batchnorm/ReadVariableOp:value:0Gmodel_5/transformer_block_11/layer_normalization_22/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2C
Amodel_5/transformer_block_11/layer_normalization_22/batchnorm/subÙ
Cmodel_5/transformer_block_11/layer_normalization_22/batchnorm/add_1AddV2Gmodel_5/transformer_block_11/layer_normalization_22/batchnorm/mul_1:z:0Emodel_5/transformer_block_11/layer_normalization_22/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2E
Cmodel_5/transformer_block_11/layer_normalization_22/batchnorm/add_1²
Lmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOpReadVariableOpUmodel_5_transformer_block_11_sequential_11_dense_37_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02N
Lmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOpÒ
Bmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2D
Bmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/axesÙ
Bmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/free
Cmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/ShapeShapeGmodel_5/transformer_block_11/layer_normalization_22/batchnorm/add_1:z:0*
T0*
_output_shapes
:2E
Cmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/ShapeÜ
Kmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2/axisÕ
Fmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2GatherV2Lmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/Shape:output:0Kmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/free:output:0Tmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2à
Mmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2_1/axisÛ
Hmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2_1GatherV2Lmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/Shape:output:0Kmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/axes:output:0Vmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2J
Hmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2_1Ô
Cmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/ConstÐ
Bmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/ProdProdOmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2:output:0Lmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: 2D
Bmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/ProdØ
Emodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
Emodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/Const_1Ø
Dmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/Prod_1ProdQmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2_1:output:0Nmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2F
Dmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/Prod_1Ø
Imodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/concat/axis´
Dmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/concatConcatV2Kmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/free:output:0Kmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/axes:output:0Rmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/concatÜ
Cmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/stackPackKmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/Prod:output:0Mmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2E
Cmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/stackí
Gmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/transpose	TransposeGmodel_5/transformer_block_11/layer_normalization_22/batchnorm/add_1:z:0Mmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2I
Gmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/transposeï
Emodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/ReshapeReshapeKmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/transpose:y:0Lmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2G
Emodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/Reshapeî
Dmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/MatMulMatMulNmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/Reshape:output:0Tmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2F
Dmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/MatMulØ
Emodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2G
Emodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/Const_2Ü
Kmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/concat_1/axisÁ
Fmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/concat_1ConcatV2Omodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2:output:0Nmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/Const_2:output:0Tmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2H
Fmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/concat_1à
=model_5/transformer_block_11/sequential_11/dense_37/TensordotReshapeNmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/MatMul:product:0Omodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2?
=model_5/transformer_block_11/sequential_11/dense_37/Tensordot¨
Jmodel_5/transformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOpReadVariableOpSmodel_5_transformer_block_11_sequential_11_dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02L
Jmodel_5/transformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOp×
;model_5/transformer_block_11/sequential_11/dense_37/BiasAddBiasAddFmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot:output:0Rmodel_5/transformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2=
;model_5/transformer_block_11/sequential_11/dense_37/BiasAddø
8model_5/transformer_block_11/sequential_11/dense_37/ReluReluDmodel_5/transformer_block_11/sequential_11/dense_37/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2:
8model_5/transformer_block_11/sequential_11/dense_37/Relu²
Lmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOpReadVariableOpUmodel_5_transformer_block_11_sequential_11_dense_38_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02N
Lmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOpÒ
Bmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2D
Bmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/axesÙ
Bmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/free
Cmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/ShapeShapeFmodel_5/transformer_block_11/sequential_11/dense_37/Relu:activations:0*
T0*
_output_shapes
:2E
Cmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/ShapeÜ
Kmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2/axisÕ
Fmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2GatherV2Lmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/Shape:output:0Kmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/free:output:0Tmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2à
Mmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2_1/axisÛ
Hmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2_1GatherV2Lmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/Shape:output:0Kmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/axes:output:0Vmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2J
Hmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2_1Ô
Cmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/ConstÐ
Bmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/ProdProdOmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2:output:0Lmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/Const:output:0*
T0*
_output_shapes
: 2D
Bmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/ProdØ
Emodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
Emodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/Const_1Ø
Dmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/Prod_1ProdQmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2_1:output:0Nmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2F
Dmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/Prod_1Ø
Imodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/concat/axis´
Dmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/concatConcatV2Kmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/free:output:0Kmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/axes:output:0Rmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/concatÜ
Cmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/stackPackKmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/Prod:output:0Mmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2E
Cmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/stackì
Gmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/transpose	TransposeFmodel_5/transformer_block_11/sequential_11/dense_37/Relu:activations:0Mmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2I
Gmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/transposeï
Emodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/ReshapeReshapeKmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/transpose:y:0Lmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2G
Emodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/Reshapeî
Dmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/MatMulMatMulNmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/Reshape:output:0Tmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2F
Dmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/MatMulØ
Emodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2G
Emodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/Const_2Ü
Kmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/concat_1/axisÁ
Fmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/concat_1ConcatV2Omodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2:output:0Nmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/Const_2:output:0Tmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2H
Fmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/concat_1à
=model_5/transformer_block_11/sequential_11/dense_38/TensordotReshapeNmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/MatMul:product:0Omodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2?
=model_5/transformer_block_11/sequential_11/dense_38/Tensordot¨
Jmodel_5/transformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOpReadVariableOpSmodel_5_transformer_block_11_sequential_11_dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02L
Jmodel_5/transformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOp×
;model_5/transformer_block_11/sequential_11/dense_38/BiasAddBiasAddFmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot:output:0Rmodel_5/transformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2=
;model_5/transformer_block_11/sequential_11/dense_38/BiasAddì
0model_5/transformer_block_11/dropout_33/IdentityIdentityDmodel_5/transformer_block_11/sequential_11/dense_38/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0model_5/transformer_block_11/dropout_33/Identity
"model_5/transformer_block_11/add_1AddV2Gmodel_5/transformer_block_11/layer_normalization_22/batchnorm/add_1:z:09model_5/transformer_block_11/dropout_33/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2$
"model_5/transformer_block_11/add_1ò
Rmodel_5/transformer_block_11/layer_normalization_23/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2T
Rmodel_5/transformer_block_11/layer_normalization_23/moments/mean/reduction_indicesØ
@model_5/transformer_block_11/layer_normalization_23/moments/meanMean&model_5/transformer_block_11/add_1:z:0[model_5/transformer_block_11/layer_normalization_23/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2B
@model_5/transformer_block_11/layer_normalization_23/moments/mean¥
Hmodel_5/transformer_block_11/layer_normalization_23/moments/StopGradientStopGradientImodel_5/transformer_block_11/layer_normalization_23/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2J
Hmodel_5/transformer_block_11/layer_normalization_23/moments/StopGradientä
Mmodel_5/transformer_block_11/layer_normalization_23/moments/SquaredDifferenceSquaredDifference&model_5/transformer_block_11/add_1:z:0Qmodel_5/transformer_block_11/layer_normalization_23/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2O
Mmodel_5/transformer_block_11/layer_normalization_23/moments/SquaredDifferenceú
Vmodel_5/transformer_block_11/layer_normalization_23/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2X
Vmodel_5/transformer_block_11/layer_normalization_23/moments/variance/reduction_indices
Dmodel_5/transformer_block_11/layer_normalization_23/moments/varianceMeanQmodel_5/transformer_block_11/layer_normalization_23/moments/SquaredDifference:z:0_model_5/transformer_block_11/layer_normalization_23/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2F
Dmodel_5/transformer_block_11/layer_normalization_23/moments/varianceÏ
Cmodel_5/transformer_block_11/layer_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752E
Cmodel_5/transformer_block_11/layer_normalization_23/batchnorm/add/yâ
Amodel_5/transformer_block_11/layer_normalization_23/batchnorm/addAddV2Mmodel_5/transformer_block_11/layer_normalization_23/moments/variance:output:0Lmodel_5/transformer_block_11/layer_normalization_23/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2C
Amodel_5/transformer_block_11/layer_normalization_23/batchnorm/add
Cmodel_5/transformer_block_11/layer_normalization_23/batchnorm/RsqrtRsqrtEmodel_5/transformer_block_11/layer_normalization_23/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2E
Cmodel_5/transformer_block_11/layer_normalization_23/batchnorm/Rsqrtº
Pmodel_5/transformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOpYmodel_5_transformer_block_11_layer_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02R
Pmodel_5/transformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOpæ
Amodel_5/transformer_block_11/layer_normalization_23/batchnorm/mulMulGmodel_5/transformer_block_11/layer_normalization_23/batchnorm/Rsqrt:y:0Xmodel_5/transformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2C
Amodel_5/transformer_block_11/layer_normalization_23/batchnorm/mul¶
Cmodel_5/transformer_block_11/layer_normalization_23/batchnorm/mul_1Mul&model_5/transformer_block_11/add_1:z:0Emodel_5/transformer_block_11/layer_normalization_23/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2E
Cmodel_5/transformer_block_11/layer_normalization_23/batchnorm/mul_1Ù
Cmodel_5/transformer_block_11/layer_normalization_23/batchnorm/mul_2MulImodel_5/transformer_block_11/layer_normalization_23/moments/mean:output:0Emodel_5/transformer_block_11/layer_normalization_23/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2E
Cmodel_5/transformer_block_11/layer_normalization_23/batchnorm/mul_2®
Lmodel_5/transformer_block_11/layer_normalization_23/batchnorm/ReadVariableOpReadVariableOpUmodel_5_transformer_block_11_layer_normalization_23_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02N
Lmodel_5/transformer_block_11/layer_normalization_23/batchnorm/ReadVariableOpâ
Amodel_5/transformer_block_11/layer_normalization_23/batchnorm/subSubTmodel_5/transformer_block_11/layer_normalization_23/batchnorm/ReadVariableOp:value:0Gmodel_5/transformer_block_11/layer_normalization_23/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2C
Amodel_5/transformer_block_11/layer_normalization_23/batchnorm/subÙ
Cmodel_5/transformer_block_11/layer_normalization_23/batchnorm/add_1AddV2Gmodel_5/transformer_block_11/layer_normalization_23/batchnorm/mul_1:z:0Emodel_5/transformer_block_11/layer_normalization_23/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2E
Cmodel_5/transformer_block_11/layer_normalization_23/batchnorm/add_1
model_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`  2
model_5/flatten_5/Constß
model_5/flatten_5/ReshapeReshapeGmodel_5/transformer_block_11/layer_normalization_23/batchnorm/add_1:z:0 model_5/flatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
model_5/flatten_5/Reshape
!model_5/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_5/concatenate_5/concat/axisÞ
model_5/concatenate_5/concatConcatV2"model_5/flatten_5/Reshape:output:0input_12*model_5/concatenate_5/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
model_5/concatenate_5/concatÁ
&model_5/dense_39/MatMul/ReadVariableOpReadVariableOp/model_5_dense_39_matmul_readvariableop_resource*
_output_shapes
:	è@*
dtype02(
&model_5/dense_39/MatMul/ReadVariableOpÅ
model_5/dense_39/MatMulMatMul%model_5/concatenate_5/concat:output:0.model_5/dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_5/dense_39/MatMul¿
'model_5/dense_39/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_39_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_5/dense_39/BiasAdd/ReadVariableOpÅ
model_5/dense_39/BiasAddBiasAdd!model_5/dense_39/MatMul:product:0/model_5/dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_5/dense_39/BiasAdd
model_5/dense_39/ReluRelu!model_5/dense_39/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_5/dense_39/Relu
model_5/dropout_34/IdentityIdentity#model_5/dense_39/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_5/dropout_34/IdentityÀ
&model_5/dense_40/MatMul/ReadVariableOpReadVariableOp/model_5_dense_40_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02(
&model_5/dense_40/MatMul/ReadVariableOpÄ
model_5/dense_40/MatMulMatMul$model_5/dropout_34/Identity:output:0.model_5/dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_5/dense_40/MatMul¿
'model_5/dense_40/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_40_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_5/dense_40/BiasAdd/ReadVariableOpÅ
model_5/dense_40/BiasAddBiasAdd!model_5/dense_40/MatMul:product:0/model_5/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_5/dense_40/BiasAdd
model_5/dense_40/ReluRelu!model_5/dense_40/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_5/dense_40/Relu
model_5/dropout_35/IdentityIdentity#model_5/dense_40/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_5/dropout_35/IdentityÀ
&model_5/dense_41/MatMul/ReadVariableOpReadVariableOp/model_5_dense_41_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&model_5/dense_41/MatMul/ReadVariableOpÄ
model_5/dense_41/MatMulMatMul$model_5/dropout_35/Identity:output:0.model_5/dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_5/dense_41/MatMul¿
'model_5/dense_41/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_41_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_5/dense_41/BiasAdd/ReadVariableOpÅ
model_5/dense_41/BiasAddBiasAdd!model_5/dense_41/MatMul:product:0/model_5/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_5/dense_41/BiasAddÖ
IdentityIdentity!model_5/dense_41/BiasAdd:output:08^model_5/batch_normalization_10/batchnorm/ReadVariableOp:^model_5/batch_normalization_10/batchnorm/ReadVariableOp_1:^model_5/batch_normalization_10/batchnorm/ReadVariableOp_2<^model_5/batch_normalization_10/batchnorm/mul/ReadVariableOp8^model_5/batch_normalization_11/batchnorm/ReadVariableOp:^model_5/batch_normalization_11/batchnorm/ReadVariableOp_1:^model_5/batch_normalization_11/batchnorm/ReadVariableOp_2<^model_5/batch_normalization_11/batchnorm/mul/ReadVariableOp)^model_5/conv1d_10/BiasAdd/ReadVariableOp5^model_5/conv1d_10/conv1d/ExpandDims_1/ReadVariableOp)^model_5/conv1d_11/BiasAdd/ReadVariableOp5^model_5/conv1d_11/conv1d/ExpandDims_1/ReadVariableOp(^model_5/dense_39/BiasAdd/ReadVariableOp'^model_5/dense_39/MatMul/ReadVariableOp(^model_5/dense_40/BiasAdd/ReadVariableOp'^model_5/dense_40/MatMul/ReadVariableOp(^model_5/dense_41/BiasAdd/ReadVariableOp'^model_5/dense_41/MatMul/ReadVariableOpE^model_5/token_and_position_embedding_5/embedding_10/embedding_lookupE^model_5/token_and_position_embedding_5/embedding_11/embedding_lookupM^model_5/transformer_block_11/layer_normalization_22/batchnorm/ReadVariableOpQ^model_5/transformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOpM^model_5/transformer_block_11/layer_normalization_23/batchnorm/ReadVariableOpQ^model_5/transformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOpY^model_5/transformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOpc^model_5/transformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpL^model_5/transformer_block_11/multi_head_attention_11/key/add/ReadVariableOpV^model_5/transformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpN^model_5/transformer_block_11/multi_head_attention_11/query/add/ReadVariableOpX^model_5/transformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOpN^model_5/transformer_block_11/multi_head_attention_11/value/add/ReadVariableOpX^model_5/transformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOpK^model_5/transformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOpM^model_5/transformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOpK^model_5/transformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOpM^model_5/transformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2r
7model_5/batch_normalization_10/batchnorm/ReadVariableOp7model_5/batch_normalization_10/batchnorm/ReadVariableOp2v
9model_5/batch_normalization_10/batchnorm/ReadVariableOp_19model_5/batch_normalization_10/batchnorm/ReadVariableOp_12v
9model_5/batch_normalization_10/batchnorm/ReadVariableOp_29model_5/batch_normalization_10/batchnorm/ReadVariableOp_22z
;model_5/batch_normalization_10/batchnorm/mul/ReadVariableOp;model_5/batch_normalization_10/batchnorm/mul/ReadVariableOp2r
7model_5/batch_normalization_11/batchnorm/ReadVariableOp7model_5/batch_normalization_11/batchnorm/ReadVariableOp2v
9model_5/batch_normalization_11/batchnorm/ReadVariableOp_19model_5/batch_normalization_11/batchnorm/ReadVariableOp_12v
9model_5/batch_normalization_11/batchnorm/ReadVariableOp_29model_5/batch_normalization_11/batchnorm/ReadVariableOp_22z
;model_5/batch_normalization_11/batchnorm/mul/ReadVariableOp;model_5/batch_normalization_11/batchnorm/mul/ReadVariableOp2T
(model_5/conv1d_10/BiasAdd/ReadVariableOp(model_5/conv1d_10/BiasAdd/ReadVariableOp2l
4model_5/conv1d_10/conv1d/ExpandDims_1/ReadVariableOp4model_5/conv1d_10/conv1d/ExpandDims_1/ReadVariableOp2T
(model_5/conv1d_11/BiasAdd/ReadVariableOp(model_5/conv1d_11/BiasAdd/ReadVariableOp2l
4model_5/conv1d_11/conv1d/ExpandDims_1/ReadVariableOp4model_5/conv1d_11/conv1d/ExpandDims_1/ReadVariableOp2R
'model_5/dense_39/BiasAdd/ReadVariableOp'model_5/dense_39/BiasAdd/ReadVariableOp2P
&model_5/dense_39/MatMul/ReadVariableOp&model_5/dense_39/MatMul/ReadVariableOp2R
'model_5/dense_40/BiasAdd/ReadVariableOp'model_5/dense_40/BiasAdd/ReadVariableOp2P
&model_5/dense_40/MatMul/ReadVariableOp&model_5/dense_40/MatMul/ReadVariableOp2R
'model_5/dense_41/BiasAdd/ReadVariableOp'model_5/dense_41/BiasAdd/ReadVariableOp2P
&model_5/dense_41/MatMul/ReadVariableOp&model_5/dense_41/MatMul/ReadVariableOp2
Dmodel_5/token_and_position_embedding_5/embedding_10/embedding_lookupDmodel_5/token_and_position_embedding_5/embedding_10/embedding_lookup2
Dmodel_5/token_and_position_embedding_5/embedding_11/embedding_lookupDmodel_5/token_and_position_embedding_5/embedding_11/embedding_lookup2
Lmodel_5/transformer_block_11/layer_normalization_22/batchnorm/ReadVariableOpLmodel_5/transformer_block_11/layer_normalization_22/batchnorm/ReadVariableOp2¤
Pmodel_5/transformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOpPmodel_5/transformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOp2
Lmodel_5/transformer_block_11/layer_normalization_23/batchnorm/ReadVariableOpLmodel_5/transformer_block_11/layer_normalization_23/batchnorm/ReadVariableOp2¤
Pmodel_5/transformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOpPmodel_5/transformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOp2´
Xmodel_5/transformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOpXmodel_5/transformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOp2È
bmodel_5/transformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpbmodel_5/transformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp2
Kmodel_5/transformer_block_11/multi_head_attention_11/key/add/ReadVariableOpKmodel_5/transformer_block_11/multi_head_attention_11/key/add/ReadVariableOp2®
Umodel_5/transformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpUmodel_5/transformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOp2
Mmodel_5/transformer_block_11/multi_head_attention_11/query/add/ReadVariableOpMmodel_5/transformer_block_11/multi_head_attention_11/query/add/ReadVariableOp2²
Wmodel_5/transformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOpWmodel_5/transformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp2
Mmodel_5/transformer_block_11/multi_head_attention_11/value/add/ReadVariableOpMmodel_5/transformer_block_11/multi_head_attention_11/value/add/ReadVariableOp2²
Wmodel_5/transformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOpWmodel_5/transformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp2
Jmodel_5/transformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOpJmodel_5/transformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOp2
Lmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOpLmodel_5/transformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOp2
Jmodel_5/transformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOpJmodel_5/transformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOp2
Lmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOpLmodel_5/transformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOp:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
"
_user_specified_name
input_11:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_12


Z__inference_token_and_position_embedding_5_layer_call_and_return_conditional_losses_764072
x(
$embedding_11_embedding_lookup_764059(
$embedding_10_embedding_lookup_764065
identity¢embedding_10/embedding_lookup¢embedding_11/embedding_lookup?
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
range³
embedding_11/embedding_lookupResourceGather$embedding_11_embedding_lookup_764059range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_11/embedding_lookup/764059*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02
embedding_11/embedding_lookup
&embedding_11/embedding_lookup/IdentityIdentity&embedding_11/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_11/embedding_lookup/764059*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&embedding_11/embedding_lookup/IdentityÃ
(embedding_11/embedding_lookup/Identity_1Identity/embedding_11/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(embedding_11/embedding_lookup/Identity_1s
embedding_10/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2
embedding_10/Cast¿
embedding_10/embedding_lookupResourceGather$embedding_10_embedding_lookup_764065embedding_10/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_10/embedding_lookup/764065*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02
embedding_10/embedding_lookup¢
&embedding_10/embedding_lookup/IdentityIdentity&embedding_10/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_10/embedding_lookup/764065*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2(
&embedding_10/embedding_lookup/IdentityÈ
(embedding_10/embedding_lookup/Identity_1Identity/embedding_10/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2*
(embedding_10/embedding_lookup/Identity_1°
addAddV21embedding_10/embedding_lookup/Identity_1:output:01embedding_11/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
add 
IdentityIdentityadd:z:0^embedding_10/embedding_lookup^embedding_11/embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿR::2>
embedding_10/embedding_lookupembedding_10/embedding_lookup2>
embedding_11/embedding_lookupembedding_11/embedding_lookup:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR

_user_specified_namex
Ö
¤
(__inference_model_5_layer_call_fn_766063
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
C__inference_model_5_layer_call_and_return_conditional_losses_7650992
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
Ü\

C__inference_model_5_layer_call_and_return_conditional_losses_764907
input_11
input_12)
%token_and_position_embedding_5_764083)
%token_and_position_embedding_5_764085
conv1d_10_764115
conv1d_10_764117
conv1d_11_764148
conv1d_11_764150!
batch_normalization_10_764237!
batch_normalization_10_764239!
batch_normalization_10_764241!
batch_normalization_10_764243!
batch_normalization_11_764328!
batch_normalization_11_764330!
batch_normalization_11_764332!
batch_normalization_11_764334
transformer_block_11_764703
transformer_block_11_764705
transformer_block_11_764707
transformer_block_11_764709
transformer_block_11_764711
transformer_block_11_764713
transformer_block_11_764715
transformer_block_11_764717
transformer_block_11_764719
transformer_block_11_764721
transformer_block_11_764723
transformer_block_11_764725
transformer_block_11_764727
transformer_block_11_764729
transformer_block_11_764731
transformer_block_11_764733
dense_39_764788
dense_39_764790
dense_40_764845
dense_40_764847
dense_41_764901
dense_41_764903
identity¢.batch_normalization_10/StatefulPartitionedCall¢.batch_normalization_11/StatefulPartitionedCall¢!conv1d_10/StatefulPartitionedCall¢!conv1d_11/StatefulPartitionedCall¢ dense_39/StatefulPartitionedCall¢ dense_40/StatefulPartitionedCall¢ dense_41/StatefulPartitionedCall¢"dropout_34/StatefulPartitionedCall¢"dropout_35/StatefulPartitionedCall¢6token_and_position_embedding_5/StatefulPartitionedCall¢,transformer_block_11/StatefulPartitionedCall
6token_and_position_embedding_5/StatefulPartitionedCallStatefulPartitionedCallinput_11%token_and_position_embedding_5_764083%token_and_position_embedding_5_764085*
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
Z__inference_token_and_position_embedding_5_layer_call_and_return_conditional_losses_76407228
6token_and_position_embedding_5/StatefulPartitionedCallÚ
!conv1d_10/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_5/StatefulPartitionedCall:output:0conv1d_10_764115conv1d_10_764117*
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
GPU2*0J 8 *N
fIRG
E__inference_conv1d_10_layer_call_and_return_conditional_losses_7641042#
!conv1d_10/StatefulPartitionedCall¤
$average_pooling1d_15/PartitionedCallPartitionedCall*conv1d_10/StatefulPartitionedCall:output:0*
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
P__inference_average_pooling1d_15_layer_call_and_return_conditional_losses_7635602&
$average_pooling1d_15/PartitionedCallÈ
!conv1d_11/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_15/PartitionedCall:output:0conv1d_11_764148conv1d_11_764150*
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
GPU2*0J 8 *N
fIRG
E__inference_conv1d_11_layer_call_and_return_conditional_losses_7641372#
!conv1d_11/StatefulPartitionedCall¸
$average_pooling1d_17/PartitionedCallPartitionedCall?token_and_position_embedding_5/StatefulPartitionedCall:output:0*
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
P__inference_average_pooling1d_17_layer_call_and_return_conditional_losses_7635902&
$average_pooling1d_17/PartitionedCall£
$average_pooling1d_16/PartitionedCallPartitionedCall*conv1d_11/StatefulPartitionedCall:output:0*
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
P__inference_average_pooling1d_16_layer_call_and_return_conditional_losses_7635752&
$average_pooling1d_16/PartitionedCallÈ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_16/PartitionedCall:output:0batch_normalization_10_764237batch_normalization_10_764239batch_normalization_10_764241batch_normalization_10_764243*
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
GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_76419020
.batch_normalization_10/StatefulPartitionedCallÈ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_17/PartitionedCall:output:0batch_normalization_11_764328batch_normalization_11_764330batch_normalization_11_764332batch_normalization_11_764334*
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
GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_76428120
.batch_normalization_11/StatefulPartitionedCall½
add_5/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:07batch_normalization_11/StatefulPartitionedCall:output:0*
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
A__inference_add_5_layer_call_and_return_conditional_losses_7643432
add_5/PartitionedCall¡
,transformer_block_11/StatefulPartitionedCallStatefulPartitionedCalladd_5/PartitionedCall:output:0transformer_block_11_764703transformer_block_11_764705transformer_block_11_764707transformer_block_11_764709transformer_block_11_764711transformer_block_11_764713transformer_block_11_764715transformer_block_11_764717transformer_block_11_764719transformer_block_11_764721transformer_block_11_764723transformer_block_11_764725transformer_block_11_764727transformer_block_11_764729transformer_block_11_764731transformer_block_11_764733*
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
GPU2*0J 8 *Y
fTRR
P__inference_transformer_block_11_layer_call_and_return_conditional_losses_7645002.
,transformer_block_11/StatefulPartitionedCall
flatten_5/PartitionedCallPartitionedCall5transformer_block_11/StatefulPartitionedCall:output:0*
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_7647422
flatten_5/PartitionedCall
concatenate_5/PartitionedCallPartitionedCall"flatten_5/PartitionedCall:output:0input_12*
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
I__inference_concatenate_5_layer_call_and_return_conditional_losses_7647572
concatenate_5/PartitionedCall·
 dense_39/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_39_764788dense_39_764790*
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
D__inference_dense_39_layer_call_and_return_conditional_losses_7647772"
 dense_39/StatefulPartitionedCall
"dropout_34/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0*
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
F__inference_dropout_34_layer_call_and_return_conditional_losses_7648052$
"dropout_34/StatefulPartitionedCall¼
 dense_40/StatefulPartitionedCallStatefulPartitionedCall+dropout_34/StatefulPartitionedCall:output:0dense_40_764845dense_40_764847*
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
D__inference_dense_40_layer_call_and_return_conditional_losses_7648342"
 dense_40/StatefulPartitionedCall½
"dropout_35/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0#^dropout_34/StatefulPartitionedCall*
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
F__inference_dropout_35_layer_call_and_return_conditional_losses_7648622$
"dropout_35/StatefulPartitionedCall¼
 dense_41/StatefulPartitionedCallStatefulPartitionedCall+dropout_35/StatefulPartitionedCall:output:0dense_41_764901dense_41_764903*
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
D__inference_dense_41_layer_call_and_return_conditional_losses_7648902"
 dense_41/StatefulPartitionedCallÂ
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall"^conv1d_10/StatefulPartitionedCall"^conv1d_11/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall#^dropout_34/StatefulPartitionedCall#^dropout_35/StatefulPartitionedCall7^token_and_position_embedding_5/StatefulPartitionedCall-^transformer_block_11/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2F
!conv1d_10/StatefulPartitionedCall!conv1d_10/StatefulPartitionedCall2F
!conv1d_11/StatefulPartitionedCall!conv1d_11/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2H
"dropout_34/StatefulPartitionedCall"dropout_34/StatefulPartitionedCall2H
"dropout_35/StatefulPartitionedCall"dropout_35/StatefulPartitionedCall2p
6token_and_position_embedding_5/StatefulPartitionedCall6token_and_position_embedding_5/StatefulPartitionedCall2\
,transformer_block_11/StatefulPartitionedCall,transformer_block_11/StatefulPartitionedCall:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
"
_user_specified_name
input_11:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_12
J
°
I__inference_sequential_11_layer_call_and_return_conditional_losses_767164

inputs.
*dense_37_tensordot_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource.
*dense_38_tensordot_readvariableop_resource,
(dense_38_biasadd_readvariableop_resource
identity¢dense_37/BiasAdd/ReadVariableOp¢!dense_37/Tensordot/ReadVariableOp¢dense_38/BiasAdd/ReadVariableOp¢!dense_38/Tensordot/ReadVariableOp±
!dense_37/Tensordot/ReadVariableOpReadVariableOp*dense_37_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02#
!dense_37/Tensordot/ReadVariableOp|
dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_37/Tensordot/axes
dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_37/Tensordot/freej
dense_37/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_37/Tensordot/Shape
 dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_37/Tensordot/GatherV2/axisþ
dense_37/Tensordot/GatherV2GatherV2!dense_37/Tensordot/Shape:output:0 dense_37/Tensordot/free:output:0)dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_37/Tensordot/GatherV2
"dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_37/Tensordot/GatherV2_1/axis
dense_37/Tensordot/GatherV2_1GatherV2!dense_37/Tensordot/Shape:output:0 dense_37/Tensordot/axes:output:0+dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_37/Tensordot/GatherV2_1~
dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_37/Tensordot/Const¤
dense_37/Tensordot/ProdProd$dense_37/Tensordot/GatherV2:output:0!dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_37/Tensordot/Prod
dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_37/Tensordot/Const_1¬
dense_37/Tensordot/Prod_1Prod&dense_37/Tensordot/GatherV2_1:output:0#dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/Tensordot/Prod_1
dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_37/Tensordot/concat/axisÝ
dense_37/Tensordot/concatConcatV2 dense_37/Tensordot/free:output:0 dense_37/Tensordot/axes:output:0'dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_37/Tensordot/concat°
dense_37/Tensordot/stackPack dense_37/Tensordot/Prod:output:0"dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_37/Tensordot/stack«
dense_37/Tensordot/transpose	Transposeinputs"dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_37/Tensordot/transposeÃ
dense_37/Tensordot/ReshapeReshape dense_37/Tensordot/transpose:y:0!dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_37/Tensordot/ReshapeÂ
dense_37/Tensordot/MatMulMatMul#dense_37/Tensordot/Reshape:output:0)dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_37/Tensordot/MatMul
dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_37/Tensordot/Const_2
 dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_37/Tensordot/concat_1/axisê
dense_37/Tensordot/concat_1ConcatV2$dense_37/Tensordot/GatherV2:output:0#dense_37/Tensordot/Const_2:output:0)dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_37/Tensordot/concat_1´
dense_37/TensordotReshape#dense_37/Tensordot/MatMul:product:0$dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_37/Tensordot§
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_37/BiasAdd/ReadVariableOp«
dense_37/BiasAddBiasAdddense_37/Tensordot:output:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_37/BiasAddw
dense_37/ReluReludense_37/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_37/Relu±
!dense_38/Tensordot/ReadVariableOpReadVariableOp*dense_38_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02#
!dense_38/Tensordot/ReadVariableOp|
dense_38/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_38/Tensordot/axes
dense_38/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_38/Tensordot/free
dense_38/Tensordot/ShapeShapedense_37/Relu:activations:0*
T0*
_output_shapes
:2
dense_38/Tensordot/Shape
 dense_38/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_38/Tensordot/GatherV2/axisþ
dense_38/Tensordot/GatherV2GatherV2!dense_38/Tensordot/Shape:output:0 dense_38/Tensordot/free:output:0)dense_38/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_38/Tensordot/GatherV2
"dense_38/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_38/Tensordot/GatherV2_1/axis
dense_38/Tensordot/GatherV2_1GatherV2!dense_38/Tensordot/Shape:output:0 dense_38/Tensordot/axes:output:0+dense_38/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_38/Tensordot/GatherV2_1~
dense_38/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_38/Tensordot/Const¤
dense_38/Tensordot/ProdProd$dense_38/Tensordot/GatherV2:output:0!dense_38/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_38/Tensordot/Prod
dense_38/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_38/Tensordot/Const_1¬
dense_38/Tensordot/Prod_1Prod&dense_38/Tensordot/GatherV2_1:output:0#dense_38/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_38/Tensordot/Prod_1
dense_38/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_38/Tensordot/concat/axisÝ
dense_38/Tensordot/concatConcatV2 dense_38/Tensordot/free:output:0 dense_38/Tensordot/axes:output:0'dense_38/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_38/Tensordot/concat°
dense_38/Tensordot/stackPack dense_38/Tensordot/Prod:output:0"dense_38/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_38/Tensordot/stackÀ
dense_38/Tensordot/transpose	Transposedense_37/Relu:activations:0"dense_38/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_38/Tensordot/transposeÃ
dense_38/Tensordot/ReshapeReshape dense_38/Tensordot/transpose:y:0!dense_38/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_38/Tensordot/ReshapeÂ
dense_38/Tensordot/MatMulMatMul#dense_38/Tensordot/Reshape:output:0)dense_38/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_38/Tensordot/MatMul
dense_38/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_38/Tensordot/Const_2
 dense_38/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_38/Tensordot/concat_1/axisê
dense_38/Tensordot/concat_1ConcatV2$dense_38/Tensordot/GatherV2:output:0#dense_38/Tensordot/Const_2:output:0)dense_38/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_38/Tensordot/concat_1´
dense_38/TensordotReshape#dense_38/Tensordot/MatMul:product:0$dense_38/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_38/Tensordot§
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_38/BiasAdd/ReadVariableOp«
dense_38/BiasAddBiasAdddense_38/Tensordot:output:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_38/BiasAddý
IdentityIdentitydense_38/BiasAdd:output:0 ^dense_37/BiasAdd/ReadVariableOp"^dense_37/Tensordot/ReadVariableOp ^dense_38/BiasAdd/ReadVariableOp"^dense_38/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2F
!dense_37/Tensordot/ReadVariableOp!dense_37/Tensordot/ReadVariableOp2B
dense_38/BiasAdd/ReadVariableOpdense_38/BiasAdd/ReadVariableOp2F
!dense_38/Tensordot/ReadVariableOp!dense_38/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
á
~
)__inference_dense_39_layer_call_fn_766957

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
D__inference_dense_39_layer_call_and_return_conditional_losses_7647772
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
Ñ
û
P__inference_transformer_block_11_layer_call_and_return_conditional_losses_764500

inputsG
Cmulti_head_attention_11_query_einsum_einsum_readvariableop_resource=
9multi_head_attention_11_query_add_readvariableop_resourceE
Amulti_head_attention_11_key_einsum_einsum_readvariableop_resource;
7multi_head_attention_11_key_add_readvariableop_resourceG
Cmulti_head_attention_11_value_einsum_einsum_readvariableop_resource=
9multi_head_attention_11_value_add_readvariableop_resourceR
Nmulti_head_attention_11_attention_output_einsum_einsum_readvariableop_resourceH
Dmulti_head_attention_11_attention_output_add_readvariableop_resource@
<layer_normalization_22_batchnorm_mul_readvariableop_resource<
8layer_normalization_22_batchnorm_readvariableop_resource<
8sequential_11_dense_37_tensordot_readvariableop_resource:
6sequential_11_dense_37_biasadd_readvariableop_resource<
8sequential_11_dense_38_tensordot_readvariableop_resource:
6sequential_11_dense_38_biasadd_readvariableop_resource@
<layer_normalization_23_batchnorm_mul_readvariableop_resource<
8layer_normalization_23_batchnorm_readvariableop_resource
identity¢/layer_normalization_22/batchnorm/ReadVariableOp¢3layer_normalization_22/batchnorm/mul/ReadVariableOp¢/layer_normalization_23/batchnorm/ReadVariableOp¢3layer_normalization_23/batchnorm/mul/ReadVariableOp¢;multi_head_attention_11/attention_output/add/ReadVariableOp¢Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp¢.multi_head_attention_11/key/add/ReadVariableOp¢8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp¢0multi_head_attention_11/query/add/ReadVariableOp¢:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp¢0multi_head_attention_11/value/add/ReadVariableOp¢:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp¢-sequential_11/dense_37/BiasAdd/ReadVariableOp¢/sequential_11/dense_37/Tensordot/ReadVariableOp¢-sequential_11/dense_38/BiasAdd/ReadVariableOp¢/sequential_11/dense_38/Tensordot/ReadVariableOp
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_11_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02<
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp
+multi_head_attention_11/query/einsum/EinsumEinsuminputsBmulti_head_attention_11/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2-
+multi_head_attention_11/query/einsum/EinsumÞ
0multi_head_attention_11/query/add/ReadVariableOpReadVariableOp9multi_head_attention_11_query_add_readvariableop_resource*
_output_shapes

: *
dtype022
0multi_head_attention_11/query/add/ReadVariableOpù
!multi_head_attention_11/query/addAddV24multi_head_attention_11/query/einsum/Einsum:output:08multi_head_attention_11/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2#
!multi_head_attention_11/query/addú
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOpReadVariableOpAmulti_head_attention_11_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02:
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp
)multi_head_attention_11/key/einsum/EinsumEinsuminputs@multi_head_attention_11/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2+
)multi_head_attention_11/key/einsum/EinsumØ
.multi_head_attention_11/key/add/ReadVariableOpReadVariableOp7multi_head_attention_11_key_add_readvariableop_resource*
_output_shapes

: *
dtype020
.multi_head_attention_11/key/add/ReadVariableOpñ
multi_head_attention_11/key/addAddV22multi_head_attention_11/key/einsum/Einsum:output:06multi_head_attention_11/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
multi_head_attention_11/key/add
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_11_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02<
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp
+multi_head_attention_11/value/einsum/EinsumEinsuminputsBmulti_head_attention_11/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2-
+multi_head_attention_11/value/einsum/EinsumÞ
0multi_head_attention_11/value/add/ReadVariableOpReadVariableOp9multi_head_attention_11_value_add_readvariableop_resource*
_output_shapes

: *
dtype022
0multi_head_attention_11/value/add/ReadVariableOpù
!multi_head_attention_11/value/addAddV24multi_head_attention_11/value/einsum/Einsum:output:08multi_head_attention_11/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2#
!multi_head_attention_11/value/add
multi_head_attention_11/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_11/Mul/yÊ
multi_head_attention_11/MulMul%multi_head_attention_11/query/add:z:0&multi_head_attention_11/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_11/Mul
%multi_head_attention_11/einsum/EinsumEinsum#multi_head_attention_11/key/add:z:0multi_head_attention_11/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2'
%multi_head_attention_11/einsum/EinsumÇ
'multi_head_attention_11/softmax/SoftmaxSoftmax.multi_head_attention_11/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2)
'multi_head_attention_11/softmax/Softmax£
-multi_head_attention_11/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-multi_head_attention_11/dropout/dropout/Const
+multi_head_attention_11/dropout/dropout/MulMul1multi_head_attention_11/softmax/Softmax:softmax:06multi_head_attention_11/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2-
+multi_head_attention_11/dropout/dropout/Mul¿
-multi_head_attention_11/dropout/dropout/ShapeShape1multi_head_attention_11/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2/
-multi_head_attention_11/dropout/dropout/Shape
Dmulti_head_attention_11/dropout/dropout/random_uniform/RandomUniformRandomUniform6multi_head_attention_11/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
dtype02F
Dmulti_head_attention_11/dropout/dropout/random_uniform/RandomUniformµ
6multi_head_attention_11/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6multi_head_attention_11/dropout/dropout/GreaterEqual/yÆ
4multi_head_attention_11/dropout/dropout/GreaterEqualGreaterEqualMmulti_head_attention_11/dropout/dropout/random_uniform/RandomUniform:output:0?multi_head_attention_11/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##26
4multi_head_attention_11/dropout/dropout/GreaterEqualç
,multi_head_attention_11/dropout/dropout/CastCast8multi_head_attention_11/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2.
,multi_head_attention_11/dropout/dropout/Cast
-multi_head_attention_11/dropout/dropout/Mul_1Mul/multi_head_attention_11/dropout/dropout/Mul:z:00multi_head_attention_11/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2/
-multi_head_attention_11/dropout/dropout/Mul_1
'multi_head_attention_11/einsum_1/EinsumEinsum1multi_head_attention_11/dropout/dropout/Mul_1:z:0%multi_head_attention_11/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2)
'multi_head_attention_11/einsum_1/Einsum¡
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpNmulti_head_attention_11_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02G
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp×
6multi_head_attention_11/attention_output/einsum/EinsumEinsum0multi_head_attention_11/einsum_1/Einsum:output:0Mmulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe28
6multi_head_attention_11/attention_output/einsum/Einsumû
;multi_head_attention_11/attention_output/add/ReadVariableOpReadVariableOpDmulti_head_attention_11_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_attention_11/attention_output/add/ReadVariableOp¡
,multi_head_attention_11/attention_output/addAddV2?multi_head_attention_11/attention_output/einsum/Einsum:output:0Cmulti_head_attention_11/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2.
,multi_head_attention_11/attention_output/addy
dropout_32/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_32/dropout/ConstÂ
dropout_32/dropout/MulMul0multi_head_attention_11/attention_output/add:z:0!dropout_32/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_32/dropout/Mul
dropout_32/dropout/ShapeShape0multi_head_attention_11/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_32/dropout/ShapeÙ
/dropout_32/dropout/random_uniform/RandomUniformRandomUniform!dropout_32/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype021
/dropout_32/dropout/random_uniform/RandomUniform
!dropout_32/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_32/dropout/GreaterEqual/yî
dropout_32/dropout/GreaterEqualGreaterEqual8dropout_32/dropout/random_uniform/RandomUniform:output:0*dropout_32/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
dropout_32/dropout/GreaterEqual¤
dropout_32/dropout/CastCast#dropout_32/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_32/dropout/Castª
dropout_32/dropout/Mul_1Muldropout_32/dropout/Mul:z:0dropout_32/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_32/dropout/Mul_1o
addAddV2inputsdropout_32/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¸
5layer_normalization_22/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_22/moments/mean/reduction_indicesâ
#layer_normalization_22/moments/meanMeanadd:z:0>layer_normalization_22/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_22/moments/meanÎ
+layer_normalization_22/moments/StopGradientStopGradient,layer_normalization_22/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_22/moments/StopGradientî
0layer_normalization_22/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_22/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_22/moments/SquaredDifferenceÀ
9layer_normalization_22/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_22/moments/variance/reduction_indices
'layer_normalization_22/moments/varianceMean4layer_normalization_22/moments/SquaredDifference:z:0Blayer_normalization_22/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_22/moments/variance
&layer_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_22/batchnorm/add/yî
$layer_normalization_22/batchnorm/addAddV20layer_normalization_22/moments/variance:output:0/layer_normalization_22/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_22/batchnorm/add¹
&layer_normalization_22/batchnorm/RsqrtRsqrt(layer_normalization_22/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_22/batchnorm/Rsqrtã
3layer_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_22/batchnorm/mul/ReadVariableOpò
$layer_normalization_22/batchnorm/mulMul*layer_normalization_22/batchnorm/Rsqrt:y:0;layer_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_22/batchnorm/mulÀ
&layer_normalization_22/batchnorm/mul_1Muladd:z:0(layer_normalization_22/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_22/batchnorm/mul_1å
&layer_normalization_22/batchnorm/mul_2Mul,layer_normalization_22/moments/mean:output:0(layer_normalization_22/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_22/batchnorm/mul_2×
/layer_normalization_22/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_22_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_22/batchnorm/ReadVariableOpî
$layer_normalization_22/batchnorm/subSub7layer_normalization_22/batchnorm/ReadVariableOp:value:0*layer_normalization_22/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_22/batchnorm/subå
&layer_normalization_22/batchnorm/add_1AddV2*layer_normalization_22/batchnorm/mul_1:z:0(layer_normalization_22/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_22/batchnorm/add_1Û
/sequential_11/dense_37/Tensordot/ReadVariableOpReadVariableOp8sequential_11_dense_37_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype021
/sequential_11/dense_37/Tensordot/ReadVariableOp
%sequential_11/dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_11/dense_37/Tensordot/axes
%sequential_11/dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_11/dense_37/Tensordot/freeª
&sequential_11/dense_37/Tensordot/ShapeShape*layer_normalization_22/batchnorm/add_1:z:0*
T0*
_output_shapes
:2(
&sequential_11/dense_37/Tensordot/Shape¢
.sequential_11/dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_37/Tensordot/GatherV2/axisÄ
)sequential_11/dense_37/Tensordot/GatherV2GatherV2/sequential_11/dense_37/Tensordot/Shape:output:0.sequential_11/dense_37/Tensordot/free:output:07sequential_11/dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_11/dense_37/Tensordot/GatherV2¦
0sequential_11/dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_11/dense_37/Tensordot/GatherV2_1/axisÊ
+sequential_11/dense_37/Tensordot/GatherV2_1GatherV2/sequential_11/dense_37/Tensordot/Shape:output:0.sequential_11/dense_37/Tensordot/axes:output:09sequential_11/dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_11/dense_37/Tensordot/GatherV2_1
&sequential_11/dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_11/dense_37/Tensordot/ConstÜ
%sequential_11/dense_37/Tensordot/ProdProd2sequential_11/dense_37/Tensordot/GatherV2:output:0/sequential_11/dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_11/dense_37/Tensordot/Prod
(sequential_11/dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_11/dense_37/Tensordot/Const_1ä
'sequential_11/dense_37/Tensordot/Prod_1Prod4sequential_11/dense_37/Tensordot/GatherV2_1:output:01sequential_11/dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_11/dense_37/Tensordot/Prod_1
,sequential_11/dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_11/dense_37/Tensordot/concat/axis£
'sequential_11/dense_37/Tensordot/concatConcatV2.sequential_11/dense_37/Tensordot/free:output:0.sequential_11/dense_37/Tensordot/axes:output:05sequential_11/dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_11/dense_37/Tensordot/concatè
&sequential_11/dense_37/Tensordot/stackPack.sequential_11/dense_37/Tensordot/Prod:output:00sequential_11/dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_11/dense_37/Tensordot/stackù
*sequential_11/dense_37/Tensordot/transpose	Transpose*layer_normalization_22/batchnorm/add_1:z:00sequential_11/dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2,
*sequential_11/dense_37/Tensordot/transposeû
(sequential_11/dense_37/Tensordot/ReshapeReshape.sequential_11/dense_37/Tensordot/transpose:y:0/sequential_11/dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_11/dense_37/Tensordot/Reshapeú
'sequential_11/dense_37/Tensordot/MatMulMatMul1sequential_11/dense_37/Tensordot/Reshape:output:07sequential_11/dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2)
'sequential_11/dense_37/Tensordot/MatMul
(sequential_11/dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2*
(sequential_11/dense_37/Tensordot/Const_2¢
.sequential_11/dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_37/Tensordot/concat_1/axis°
)sequential_11/dense_37/Tensordot/concat_1ConcatV22sequential_11/dense_37/Tensordot/GatherV2:output:01sequential_11/dense_37/Tensordot/Const_2:output:07sequential_11/dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_11/dense_37/Tensordot/concat_1ì
 sequential_11/dense_37/TensordotReshape1sequential_11/dense_37/Tensordot/MatMul:product:02sequential_11/dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2"
 sequential_11/dense_37/TensordotÑ
-sequential_11/dense_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_11/dense_37/BiasAdd/ReadVariableOpã
sequential_11/dense_37/BiasAddBiasAdd)sequential_11/dense_37/Tensordot:output:05sequential_11/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2 
sequential_11/dense_37/BiasAdd¡
sequential_11/dense_37/ReluRelu'sequential_11/dense_37/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_11/dense_37/ReluÛ
/sequential_11/dense_38/Tensordot/ReadVariableOpReadVariableOp8sequential_11_dense_38_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype021
/sequential_11/dense_38/Tensordot/ReadVariableOp
%sequential_11/dense_38/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_11/dense_38/Tensordot/axes
%sequential_11/dense_38/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_11/dense_38/Tensordot/free©
&sequential_11/dense_38/Tensordot/ShapeShape)sequential_11/dense_37/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_11/dense_38/Tensordot/Shape¢
.sequential_11/dense_38/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_38/Tensordot/GatherV2/axisÄ
)sequential_11/dense_38/Tensordot/GatherV2GatherV2/sequential_11/dense_38/Tensordot/Shape:output:0.sequential_11/dense_38/Tensordot/free:output:07sequential_11/dense_38/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_11/dense_38/Tensordot/GatherV2¦
0sequential_11/dense_38/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_11/dense_38/Tensordot/GatherV2_1/axisÊ
+sequential_11/dense_38/Tensordot/GatherV2_1GatherV2/sequential_11/dense_38/Tensordot/Shape:output:0.sequential_11/dense_38/Tensordot/axes:output:09sequential_11/dense_38/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_11/dense_38/Tensordot/GatherV2_1
&sequential_11/dense_38/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_11/dense_38/Tensordot/ConstÜ
%sequential_11/dense_38/Tensordot/ProdProd2sequential_11/dense_38/Tensordot/GatherV2:output:0/sequential_11/dense_38/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_11/dense_38/Tensordot/Prod
(sequential_11/dense_38/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_11/dense_38/Tensordot/Const_1ä
'sequential_11/dense_38/Tensordot/Prod_1Prod4sequential_11/dense_38/Tensordot/GatherV2_1:output:01sequential_11/dense_38/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_11/dense_38/Tensordot/Prod_1
,sequential_11/dense_38/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_11/dense_38/Tensordot/concat/axis£
'sequential_11/dense_38/Tensordot/concatConcatV2.sequential_11/dense_38/Tensordot/free:output:0.sequential_11/dense_38/Tensordot/axes:output:05sequential_11/dense_38/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_11/dense_38/Tensordot/concatè
&sequential_11/dense_38/Tensordot/stackPack.sequential_11/dense_38/Tensordot/Prod:output:00sequential_11/dense_38/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_11/dense_38/Tensordot/stackø
*sequential_11/dense_38/Tensordot/transpose	Transpose)sequential_11/dense_37/Relu:activations:00sequential_11/dense_38/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2,
*sequential_11/dense_38/Tensordot/transposeû
(sequential_11/dense_38/Tensordot/ReshapeReshape.sequential_11/dense_38/Tensordot/transpose:y:0/sequential_11/dense_38/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_11/dense_38/Tensordot/Reshapeú
'sequential_11/dense_38/Tensordot/MatMulMatMul1sequential_11/dense_38/Tensordot/Reshape:output:07sequential_11/dense_38/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential_11/dense_38/Tensordot/MatMul
(sequential_11/dense_38/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_11/dense_38/Tensordot/Const_2¢
.sequential_11/dense_38/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_38/Tensordot/concat_1/axis°
)sequential_11/dense_38/Tensordot/concat_1ConcatV22sequential_11/dense_38/Tensordot/GatherV2:output:01sequential_11/dense_38/Tensordot/Const_2:output:07sequential_11/dense_38/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_11/dense_38/Tensordot/concat_1ì
 sequential_11/dense_38/TensordotReshape1sequential_11/dense_38/Tensordot/MatMul:product:02sequential_11/dense_38/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 sequential_11/dense_38/TensordotÑ
-sequential_11/dense_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_11/dense_38/BiasAdd/ReadVariableOpã
sequential_11/dense_38/BiasAddBiasAdd)sequential_11/dense_38/Tensordot:output:05sequential_11/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
sequential_11/dense_38/BiasAddy
dropout_33/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_33/dropout/Const¹
dropout_33/dropout/MulMul'sequential_11/dense_38/BiasAdd:output:0!dropout_33/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_33/dropout/Mul
dropout_33/dropout/ShapeShape'sequential_11/dense_38/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_33/dropout/ShapeÙ
/dropout_33/dropout/random_uniform/RandomUniformRandomUniform!dropout_33/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype021
/dropout_33/dropout/random_uniform/RandomUniform
!dropout_33/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_33/dropout/GreaterEqual/yî
dropout_33/dropout/GreaterEqualGreaterEqual8dropout_33/dropout/random_uniform/RandomUniform:output:0*dropout_33/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
dropout_33/dropout/GreaterEqual¤
dropout_33/dropout/CastCast#dropout_33/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_33/dropout/Castª
dropout_33/dropout/Mul_1Muldropout_33/dropout/Mul:z:0dropout_33/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_33/dropout/Mul_1
add_1AddV2*layer_normalization_22/batchnorm/add_1:z:0dropout_33/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¸
5layer_normalization_23/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_23/moments/mean/reduction_indicesä
#layer_normalization_23/moments/meanMean	add_1:z:0>layer_normalization_23/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2%
#layer_normalization_23/moments/meanÎ
+layer_normalization_23/moments/StopGradientStopGradient,layer_normalization_23/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2-
+layer_normalization_23/moments/StopGradientð
0layer_normalization_23/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_23/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0layer_normalization_23/moments/SquaredDifferenceÀ
9layer_normalization_23/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_23/moments/variance/reduction_indices
'layer_normalization_23/moments/varianceMean4layer_normalization_23/moments/SquaredDifference:z:0Blayer_normalization_23/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2)
'layer_normalization_23/moments/variance
&layer_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_23/batchnorm/add/yî
$layer_normalization_23/batchnorm/addAddV20layer_normalization_23/moments/variance:output:0/layer_normalization_23/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2&
$layer_normalization_23/batchnorm/add¹
&layer_normalization_23/batchnorm/RsqrtRsqrt(layer_normalization_23/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2(
&layer_normalization_23/batchnorm/Rsqrtã
3layer_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_23/batchnorm/mul/ReadVariableOpò
$layer_normalization_23/batchnorm/mulMul*layer_normalization_23/batchnorm/Rsqrt:y:0;layer_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_23/batchnorm/mulÂ
&layer_normalization_23/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_23/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_23/batchnorm/mul_1å
&layer_normalization_23/batchnorm/mul_2Mul,layer_normalization_23/moments/mean:output:0(layer_normalization_23/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_23/batchnorm/mul_2×
/layer_normalization_23/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_23_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_23/batchnorm/ReadVariableOpî
$layer_normalization_23/batchnorm/subSub7layer_normalization_23/batchnorm/ReadVariableOp:value:0*layer_normalization_23/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2&
$layer_normalization_23/batchnorm/subå
&layer_normalization_23/batchnorm/add_1AddV2*layer_normalization_23/batchnorm/mul_1:z:0(layer_normalization_23/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&layer_normalization_23/batchnorm/add_1è
IdentityIdentity*layer_normalization_23/batchnorm/add_1:z:00^layer_normalization_22/batchnorm/ReadVariableOp4^layer_normalization_22/batchnorm/mul/ReadVariableOp0^layer_normalization_23/batchnorm/ReadVariableOp4^layer_normalization_23/batchnorm/mul/ReadVariableOp<^multi_head_attention_11/attention_output/add/ReadVariableOpF^multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp/^multi_head_attention_11/key/add/ReadVariableOp9^multi_head_attention_11/key/einsum/Einsum/ReadVariableOp1^multi_head_attention_11/query/add/ReadVariableOp;^multi_head_attention_11/query/einsum/Einsum/ReadVariableOp1^multi_head_attention_11/value/add/ReadVariableOp;^multi_head_attention_11/value/einsum/Einsum/ReadVariableOp.^sequential_11/dense_37/BiasAdd/ReadVariableOp0^sequential_11/dense_37/Tensordot/ReadVariableOp.^sequential_11/dense_38/BiasAdd/ReadVariableOp0^sequential_11/dense_38/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2b
/layer_normalization_22/batchnorm/ReadVariableOp/layer_normalization_22/batchnorm/ReadVariableOp2j
3layer_normalization_22/batchnorm/mul/ReadVariableOp3layer_normalization_22/batchnorm/mul/ReadVariableOp2b
/layer_normalization_23/batchnorm/ReadVariableOp/layer_normalization_23/batchnorm/ReadVariableOp2j
3layer_normalization_23/batchnorm/mul/ReadVariableOp3layer_normalization_23/batchnorm/mul/ReadVariableOp2z
;multi_head_attention_11/attention_output/add/ReadVariableOp;multi_head_attention_11/attention_output/add/ReadVariableOp2
Emulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpEmulti_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp2`
.multi_head_attention_11/key/add/ReadVariableOp.multi_head_attention_11/key/add/ReadVariableOp2t
8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp8multi_head_attention_11/key/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_11/query/add/ReadVariableOp0multi_head_attention_11/query/add/ReadVariableOp2x
:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp:multi_head_attention_11/query/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_11/value/add/ReadVariableOp0multi_head_attention_11/value/add/ReadVariableOp2x
:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp:multi_head_attention_11/value/einsum/Einsum/ReadVariableOp2^
-sequential_11/dense_37/BiasAdd/ReadVariableOp-sequential_11/dense_37/BiasAdd/ReadVariableOp2b
/sequential_11/dense_37/Tensordot/ReadVariableOp/sequential_11/dense_37/Tensordot/ReadVariableOp2^
-sequential_11/dense_38/BiasAdd/ReadVariableOp-sequential_11/dense_38/BiasAdd/ReadVariableOp2b
/sequential_11/dense_38/Tensordot/ReadVariableOp/sequential_11/dense_38/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs

Q
5__inference_average_pooling1d_17_layer_call_fn_763596

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
P__inference_average_pooling1d_17_layer_call_and_return_conditional_losses_7635902
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
õ

*__inference_conv1d_10_layer_call_fn_766199

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
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1d_10_layer_call_and_return_conditional_losses_7641042
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

e
F__inference_dropout_35_layer_call_and_return_conditional_losses_767016

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *d!?2
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
×£=2
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
+__inference_dropout_35_layer_call_fn_767026

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
F__inference_dropout_35_layer_call_and_return_conditional_losses_7648622
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

ø
E__inference_conv1d_11_layer_call_and_return_conditional_losses_766215

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
± 
ã
D__inference_dense_37_layer_call_and_return_conditional_losses_767221

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


R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_766526

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
ª
7__inference_batch_normalization_11_layer_call_fn_766457

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¡
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
GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_7642812
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
ï
~
)__inference_dense_37_layer_call_fn_767230

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
D__inference_dense_37_layer_call_and_return_conditional_losses_7639112
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
¡
F
*__inference_flatten_5_layer_call_fn_766924

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
E__inference_flatten_5_layer_call_and_return_conditional_losses_7647422
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
é

R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_766362

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

ø
E__inference_conv1d_10_layer_call_and_return_conditional_losses_766190

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
Ã¡
¾)
__inference__traced_save_767515
file_prefix/
+savev2_conv1d_10_kernel_read_readvariableop-
)savev2_conv1d_10_bias_read_readvariableop/
+savev2_conv1d_11_kernel_read_readvariableop-
)savev2_conv1d_11_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop.
*savev2_dense_39_kernel_read_readvariableop,
(savev2_dense_39_bias_read_readvariableop.
*savev2_dense_40_kernel_read_readvariableop,
(savev2_dense_40_bias_read_readvariableop.
*savev2_dense_41_kernel_read_readvariableop,
(savev2_dense_41_bias_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop'
#savev2_momentum_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	U
Qsavev2_token_and_position_embedding_5_embedding_10_embeddings_read_readvariableopU
Qsavev2_token_and_position_embedding_5_embedding_11_embeddings_read_readvariableopX
Tsavev2_transformer_block_11_multi_head_attention_11_query_kernel_read_readvariableopV
Rsavev2_transformer_block_11_multi_head_attention_11_query_bias_read_readvariableopV
Rsavev2_transformer_block_11_multi_head_attention_11_key_kernel_read_readvariableopT
Psavev2_transformer_block_11_multi_head_attention_11_key_bias_read_readvariableopX
Tsavev2_transformer_block_11_multi_head_attention_11_value_kernel_read_readvariableopV
Rsavev2_transformer_block_11_multi_head_attention_11_value_bias_read_readvariableopc
_savev2_transformer_block_11_multi_head_attention_11_attention_output_kernel_read_readvariableopa
]savev2_transformer_block_11_multi_head_attention_11_attention_output_bias_read_readvariableop.
*savev2_dense_37_kernel_read_readvariableop,
(savev2_dense_37_bias_read_readvariableop.
*savev2_dense_38_kernel_read_readvariableop,
(savev2_dense_38_bias_read_readvariableopP
Lsavev2_transformer_block_11_layer_normalization_22_gamma_read_readvariableopO
Ksavev2_transformer_block_11_layer_normalization_22_beta_read_readvariableopP
Lsavev2_transformer_block_11_layer_normalization_23_gamma_read_readvariableopO
Ksavev2_transformer_block_11_layer_normalization_23_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_sgd_conv1d_10_kernel_momentum_read_readvariableop:
6savev2_sgd_conv1d_10_bias_momentum_read_readvariableop<
8savev2_sgd_conv1d_11_kernel_momentum_read_readvariableop:
6savev2_sgd_conv1d_11_bias_momentum_read_readvariableopH
Dsavev2_sgd_batch_normalization_10_gamma_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_10_beta_momentum_read_readvariableopH
Dsavev2_sgd_batch_normalization_11_gamma_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_11_beta_momentum_read_readvariableop;
7savev2_sgd_dense_39_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_39_bias_momentum_read_readvariableop;
7savev2_sgd_dense_40_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_40_bias_momentum_read_readvariableop;
7savev2_sgd_dense_41_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_41_bias_momentum_read_readvariableopb
^savev2_sgd_token_and_position_embedding_5_embedding_10_embeddings_momentum_read_readvariableopb
^savev2_sgd_token_and_position_embedding_5_embedding_11_embeddings_momentum_read_readvariableope
asavev2_sgd_transformer_block_11_multi_head_attention_11_query_kernel_momentum_read_readvariableopc
_savev2_sgd_transformer_block_11_multi_head_attention_11_query_bias_momentum_read_readvariableopc
_savev2_sgd_transformer_block_11_multi_head_attention_11_key_kernel_momentum_read_readvariableopa
]savev2_sgd_transformer_block_11_multi_head_attention_11_key_bias_momentum_read_readvariableope
asavev2_sgd_transformer_block_11_multi_head_attention_11_value_kernel_momentum_read_readvariableopc
_savev2_sgd_transformer_block_11_multi_head_attention_11_value_bias_momentum_read_readvariableopp
lsavev2_sgd_transformer_block_11_multi_head_attention_11_attention_output_kernel_momentum_read_readvariableopn
jsavev2_sgd_transformer_block_11_multi_head_attention_11_attention_output_bias_momentum_read_readvariableop;
7savev2_sgd_dense_37_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_37_bias_momentum_read_readvariableop;
7savev2_sgd_dense_38_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_38_bias_momentum_read_readvariableop]
Ysavev2_sgd_transformer_block_11_layer_normalization_22_gamma_momentum_read_readvariableop\
Xsavev2_sgd_transformer_block_11_layer_normalization_22_beta_momentum_read_readvariableop]
Ysavev2_sgd_transformer_block_11_layer_normalization_23_gamma_momentum_read_readvariableop\
Xsavev2_sgd_transformer_block_11_layer_normalization_23_beta_momentum_read_readvariableop
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
SaveV2/shape_and_slices®(
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv1d_10_kernel_read_readvariableop)savev2_conv1d_10_bias_read_readvariableop+savev2_conv1d_11_kernel_read_readvariableop)savev2_conv1d_11_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop*savev2_dense_39_kernel_read_readvariableop(savev2_dense_39_bias_read_readvariableop*savev2_dense_40_kernel_read_readvariableop(savev2_dense_40_bias_read_readvariableop*savev2_dense_41_kernel_read_readvariableop(savev2_dense_41_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableopQsavev2_token_and_position_embedding_5_embedding_10_embeddings_read_readvariableopQsavev2_token_and_position_embedding_5_embedding_11_embeddings_read_readvariableopTsavev2_transformer_block_11_multi_head_attention_11_query_kernel_read_readvariableopRsavev2_transformer_block_11_multi_head_attention_11_query_bias_read_readvariableopRsavev2_transformer_block_11_multi_head_attention_11_key_kernel_read_readvariableopPsavev2_transformer_block_11_multi_head_attention_11_key_bias_read_readvariableopTsavev2_transformer_block_11_multi_head_attention_11_value_kernel_read_readvariableopRsavev2_transformer_block_11_multi_head_attention_11_value_bias_read_readvariableop_savev2_transformer_block_11_multi_head_attention_11_attention_output_kernel_read_readvariableop]savev2_transformer_block_11_multi_head_attention_11_attention_output_bias_read_readvariableop*savev2_dense_37_kernel_read_readvariableop(savev2_dense_37_bias_read_readvariableop*savev2_dense_38_kernel_read_readvariableop(savev2_dense_38_bias_read_readvariableopLsavev2_transformer_block_11_layer_normalization_22_gamma_read_readvariableopKsavev2_transformer_block_11_layer_normalization_22_beta_read_readvariableopLsavev2_transformer_block_11_layer_normalization_23_gamma_read_readvariableopKsavev2_transformer_block_11_layer_normalization_23_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_sgd_conv1d_10_kernel_momentum_read_readvariableop6savev2_sgd_conv1d_10_bias_momentum_read_readvariableop8savev2_sgd_conv1d_11_kernel_momentum_read_readvariableop6savev2_sgd_conv1d_11_bias_momentum_read_readvariableopDsavev2_sgd_batch_normalization_10_gamma_momentum_read_readvariableopCsavev2_sgd_batch_normalization_10_beta_momentum_read_readvariableopDsavev2_sgd_batch_normalization_11_gamma_momentum_read_readvariableopCsavev2_sgd_batch_normalization_11_beta_momentum_read_readvariableop7savev2_sgd_dense_39_kernel_momentum_read_readvariableop5savev2_sgd_dense_39_bias_momentum_read_readvariableop7savev2_sgd_dense_40_kernel_momentum_read_readvariableop5savev2_sgd_dense_40_bias_momentum_read_readvariableop7savev2_sgd_dense_41_kernel_momentum_read_readvariableop5savev2_sgd_dense_41_bias_momentum_read_readvariableop^savev2_sgd_token_and_position_embedding_5_embedding_10_embeddings_momentum_read_readvariableop^savev2_sgd_token_and_position_embedding_5_embedding_11_embeddings_momentum_read_readvariableopasavev2_sgd_transformer_block_11_multi_head_attention_11_query_kernel_momentum_read_readvariableop_savev2_sgd_transformer_block_11_multi_head_attention_11_query_bias_momentum_read_readvariableop_savev2_sgd_transformer_block_11_multi_head_attention_11_key_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_11_multi_head_attention_11_key_bias_momentum_read_readvariableopasavev2_sgd_transformer_block_11_multi_head_attention_11_value_kernel_momentum_read_readvariableop_savev2_sgd_transformer_block_11_multi_head_attention_11_value_bias_momentum_read_readvariableoplsavev2_sgd_transformer_block_11_multi_head_attention_11_attention_output_kernel_momentum_read_readvariableopjsavev2_sgd_transformer_block_11_multi_head_attention_11_attention_output_bias_momentum_read_readvariableop7savev2_sgd_dense_37_kernel_momentum_read_readvariableop5savev2_sgd_dense_37_bias_momentum_read_readvariableop7savev2_sgd_dense_38_kernel_momentum_read_readvariableop5savev2_sgd_dense_38_bias_momentum_read_readvariableopYsavev2_sgd_transformer_block_11_layer_normalization_22_gamma_momentum_read_readvariableopXsavev2_sgd_transformer_block_11_layer_normalization_22_beta_momentum_read_readvariableopYsavev2_sgd_transformer_block_11_layer_normalization_23_gamma_momentum_read_readvariableopXsavev2_sgd_transformer_block_11_layer_normalization_23_beta_momentum_read_readvariableopsavev2_const"/device:CPU:0*
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
î
ª
7__inference_batch_normalization_10_layer_call_fn_766293

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallª
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
GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7636922
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

ø
E__inference_conv1d_10_layer_call_and_return_conditional_losses_764104

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
½0
É
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_764190

inputs
assignmovingavg_764165
assignmovingavg_1_764171)
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
loc:@AssignMovingAvg/764165*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_764165*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/764165*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/764165*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_764165AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/764165*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/764171*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_764171*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/764171*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/764171*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_764171AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/764171*
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
.__inference_concatenate_5_layer_call_fn_766937
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
I__inference_concatenate_5_layer_call_and_return_conditional_losses_7647572
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
	
Ý
D__inference_dense_41_layer_call_and_return_conditional_losses_764890

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
ÓY
Á
C__inference_model_5_layer_call_and_return_conditional_losses_765001
input_11
input_12)
%token_and_position_embedding_5_764911)
%token_and_position_embedding_5_764913
conv1d_10_764916
conv1d_10_764918
conv1d_11_764922
conv1d_11_764924!
batch_normalization_10_764929!
batch_normalization_10_764931!
batch_normalization_10_764933!
batch_normalization_10_764935!
batch_normalization_11_764938!
batch_normalization_11_764940!
batch_normalization_11_764942!
batch_normalization_11_764944
transformer_block_11_764948
transformer_block_11_764950
transformer_block_11_764952
transformer_block_11_764954
transformer_block_11_764956
transformer_block_11_764958
transformer_block_11_764960
transformer_block_11_764962
transformer_block_11_764964
transformer_block_11_764966
transformer_block_11_764968
transformer_block_11_764970
transformer_block_11_764972
transformer_block_11_764974
transformer_block_11_764976
transformer_block_11_764978
dense_39_764983
dense_39_764985
dense_40_764989
dense_40_764991
dense_41_764995
dense_41_764997
identity¢.batch_normalization_10/StatefulPartitionedCall¢.batch_normalization_11/StatefulPartitionedCall¢!conv1d_10/StatefulPartitionedCall¢!conv1d_11/StatefulPartitionedCall¢ dense_39/StatefulPartitionedCall¢ dense_40/StatefulPartitionedCall¢ dense_41/StatefulPartitionedCall¢6token_and_position_embedding_5/StatefulPartitionedCall¢,transformer_block_11/StatefulPartitionedCall
6token_and_position_embedding_5/StatefulPartitionedCallStatefulPartitionedCallinput_11%token_and_position_embedding_5_764911%token_and_position_embedding_5_764913*
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
Z__inference_token_and_position_embedding_5_layer_call_and_return_conditional_losses_76407228
6token_and_position_embedding_5/StatefulPartitionedCallÚ
!conv1d_10/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_5/StatefulPartitionedCall:output:0conv1d_10_764916conv1d_10_764918*
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
GPU2*0J 8 *N
fIRG
E__inference_conv1d_10_layer_call_and_return_conditional_losses_7641042#
!conv1d_10/StatefulPartitionedCall¤
$average_pooling1d_15/PartitionedCallPartitionedCall*conv1d_10/StatefulPartitionedCall:output:0*
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
P__inference_average_pooling1d_15_layer_call_and_return_conditional_losses_7635602&
$average_pooling1d_15/PartitionedCallÈ
!conv1d_11/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_15/PartitionedCall:output:0conv1d_11_764922conv1d_11_764924*
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
GPU2*0J 8 *N
fIRG
E__inference_conv1d_11_layer_call_and_return_conditional_losses_7641372#
!conv1d_11/StatefulPartitionedCall¸
$average_pooling1d_17/PartitionedCallPartitionedCall?token_and_position_embedding_5/StatefulPartitionedCall:output:0*
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
P__inference_average_pooling1d_17_layer_call_and_return_conditional_losses_7635902&
$average_pooling1d_17/PartitionedCall£
$average_pooling1d_16/PartitionedCallPartitionedCall*conv1d_11/StatefulPartitionedCall:output:0*
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
P__inference_average_pooling1d_16_layer_call_and_return_conditional_losses_7635752&
$average_pooling1d_16/PartitionedCallÊ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_16/PartitionedCall:output:0batch_normalization_10_764929batch_normalization_10_764931batch_normalization_10_764933batch_normalization_10_764935*
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
GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_76421020
.batch_normalization_10/StatefulPartitionedCallÊ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_17/PartitionedCall:output:0batch_normalization_11_764938batch_normalization_11_764940batch_normalization_11_764942batch_normalization_11_764944*
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
GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_76430120
.batch_normalization_11/StatefulPartitionedCall½
add_5/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:07batch_normalization_11/StatefulPartitionedCall:output:0*
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
A__inference_add_5_layer_call_and_return_conditional_losses_7643432
add_5/PartitionedCall¡
,transformer_block_11/StatefulPartitionedCallStatefulPartitionedCalladd_5/PartitionedCall:output:0transformer_block_11_764948transformer_block_11_764950transformer_block_11_764952transformer_block_11_764954transformer_block_11_764956transformer_block_11_764958transformer_block_11_764960transformer_block_11_764962transformer_block_11_764964transformer_block_11_764966transformer_block_11_764968transformer_block_11_764970transformer_block_11_764972transformer_block_11_764974transformer_block_11_764976transformer_block_11_764978*
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
GPU2*0J 8 *Y
fTRR
P__inference_transformer_block_11_layer_call_and_return_conditional_losses_7646272.
,transformer_block_11/StatefulPartitionedCall
flatten_5/PartitionedCallPartitionedCall5transformer_block_11/StatefulPartitionedCall:output:0*
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_7647422
flatten_5/PartitionedCall
concatenate_5/PartitionedCallPartitionedCall"flatten_5/PartitionedCall:output:0input_12*
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
I__inference_concatenate_5_layer_call_and_return_conditional_losses_7647572
concatenate_5/PartitionedCall·
 dense_39/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_39_764983dense_39_764985*
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
D__inference_dense_39_layer_call_and_return_conditional_losses_7647772"
 dense_39/StatefulPartitionedCall
dropout_34/PartitionedCallPartitionedCall)dense_39/StatefulPartitionedCall:output:0*
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
F__inference_dropout_34_layer_call_and_return_conditional_losses_7648102
dropout_34/PartitionedCall´
 dense_40/StatefulPartitionedCallStatefulPartitionedCall#dropout_34/PartitionedCall:output:0dense_40_764989dense_40_764991*
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
D__inference_dense_40_layer_call_and_return_conditional_losses_7648342"
 dense_40/StatefulPartitionedCall
dropout_35/PartitionedCallPartitionedCall)dense_40/StatefulPartitionedCall:output:0*
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
F__inference_dropout_35_layer_call_and_return_conditional_losses_7648672
dropout_35/PartitionedCall´
 dense_41/StatefulPartitionedCallStatefulPartitionedCall#dropout_35/PartitionedCall:output:0dense_41_764995dense_41_764997*
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
D__inference_dense_41_layer_call_and_return_conditional_losses_7648902"
 dense_41/StatefulPartitionedCallø
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall"^conv1d_10/StatefulPartitionedCall"^conv1d_11/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall7^token_and_position_embedding_5/StatefulPartitionedCall-^transformer_block_11/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2F
!conv1d_10/StatefulPartitionedCall!conv1d_10/StatefulPartitionedCall2F
!conv1d_11/StatefulPartitionedCall!conv1d_11/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2p
6token_and_position_embedding_5/StatefulPartitionedCall6token_and_position_embedding_5/StatefulPartitionedCall2\
,transformer_block_11/StatefulPartitionedCall,transformer_block_11/StatefulPartitionedCall:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
"
_user_specified_name
input_11:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_12
Â
u
I__inference_concatenate_5_layer_call_and_return_conditional_losses_766931
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
ö
l
P__inference_average_pooling1d_15_layer_call_and_return_conditional_losses_763560

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
ê

I__inference_sequential_11_layer_call_and_return_conditional_losses_764032

inputs
dense_37_764021
dense_37_764023
dense_38_764026
dense_38_764028
identity¢ dense_37/StatefulPartitionedCall¢ dense_38/StatefulPartitionedCall
 dense_37/StatefulPartitionedCallStatefulPartitionedCallinputsdense_37_764021dense_37_764023*
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
D__inference_dense_37_layer_call_and_return_conditional_losses_7639112"
 dense_37/StatefulPartitionedCall¾
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_764026dense_38_764028*
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
D__inference_dense_38_layer_call_and_return_conditional_losses_7639572"
 dense_38/StatefulPartitionedCallÇ
IdentityIdentity)dense_38/StatefulPartitionedCall:output:0!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
ð
ª
7__inference_batch_normalization_11_layer_call_fn_766552

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¬
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
GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_7638652
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
Ò

á
5__inference_transformer_block_11_layer_call_fn_766876

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
identity¢StatefulPartitionedCallÂ
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
GPU2*0J 8 *Y
fTRR
P__inference_transformer_block_11_layer_call_and_return_conditional_losses_7645002
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
ñ	
Ý
D__inference_dense_39_layer_call_and_return_conditional_losses_766948

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
º
s
I__inference_concatenate_5_layer_call_and_return_conditional_losses_764757

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
ô
ü&
C__inference_model_5_layer_call_and_return_conditional_losses_765742
inputs_0
inputs_1G
Ctoken_and_position_embedding_5_embedding_11_embedding_lookup_765444G
Ctoken_and_position_embedding_5_embedding_10_embedding_lookup_7654509
5conv1d_10_conv1d_expanddims_1_readvariableop_resource-
)conv1d_10_biasadd_readvariableop_resource9
5conv1d_11_conv1d_expanddims_1_readvariableop_resource-
)conv1d_11_biasadd_readvariableop_resource1
-batch_normalization_10_assignmovingavg_7655003
/batch_normalization_10_assignmovingavg_1_765506@
<batch_normalization_10_batchnorm_mul_readvariableop_resource<
8batch_normalization_10_batchnorm_readvariableop_resource1
-batch_normalization_11_assignmovingavg_7655323
/batch_normalization_11_assignmovingavg_1_765538@
<batch_normalization_11_batchnorm_mul_readvariableop_resource<
8batch_normalization_11_batchnorm_readvariableop_resource\
Xtransformer_block_11_multi_head_attention_11_query_einsum_einsum_readvariableop_resourceR
Ntransformer_block_11_multi_head_attention_11_query_add_readvariableop_resourceZ
Vtransformer_block_11_multi_head_attention_11_key_einsum_einsum_readvariableop_resourceP
Ltransformer_block_11_multi_head_attention_11_key_add_readvariableop_resource\
Xtransformer_block_11_multi_head_attention_11_value_einsum_einsum_readvariableop_resourceR
Ntransformer_block_11_multi_head_attention_11_value_add_readvariableop_resourceg
ctransformer_block_11_multi_head_attention_11_attention_output_einsum_einsum_readvariableop_resource]
Ytransformer_block_11_multi_head_attention_11_attention_output_add_readvariableop_resourceU
Qtransformer_block_11_layer_normalization_22_batchnorm_mul_readvariableop_resourceQ
Mtransformer_block_11_layer_normalization_22_batchnorm_readvariableop_resourceQ
Mtransformer_block_11_sequential_11_dense_37_tensordot_readvariableop_resourceO
Ktransformer_block_11_sequential_11_dense_37_biasadd_readvariableop_resourceQ
Mtransformer_block_11_sequential_11_dense_38_tensordot_readvariableop_resourceO
Ktransformer_block_11_sequential_11_dense_38_biasadd_readvariableop_resourceU
Qtransformer_block_11_layer_normalization_23_batchnorm_mul_readvariableop_resourceQ
Mtransformer_block_11_layer_normalization_23_batchnorm_readvariableop_resource+
'dense_39_matmul_readvariableop_resource,
(dense_39_biasadd_readvariableop_resource+
'dense_40_matmul_readvariableop_resource,
(dense_40_biasadd_readvariableop_resource+
'dense_41_matmul_readvariableop_resource,
(dense_41_biasadd_readvariableop_resource
identity¢:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_10/AssignMovingAvg/ReadVariableOp¢<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_10/batchnorm/ReadVariableOp¢3batch_normalization_10/batchnorm/mul/ReadVariableOp¢:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_11/AssignMovingAvg/ReadVariableOp¢<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_11/batchnorm/ReadVariableOp¢3batch_normalization_11/batchnorm/mul/ReadVariableOp¢ conv1d_10/BiasAdd/ReadVariableOp¢,conv1d_10/conv1d/ExpandDims_1/ReadVariableOp¢ conv1d_11/BiasAdd/ReadVariableOp¢,conv1d_11/conv1d/ExpandDims_1/ReadVariableOp¢dense_39/BiasAdd/ReadVariableOp¢dense_39/MatMul/ReadVariableOp¢dense_40/BiasAdd/ReadVariableOp¢dense_40/MatMul/ReadVariableOp¢dense_41/BiasAdd/ReadVariableOp¢dense_41/MatMul/ReadVariableOp¢<token_and_position_embedding_5/embedding_10/embedding_lookup¢<token_and_position_embedding_5/embedding_11/embedding_lookup¢Dtransformer_block_11/layer_normalization_22/batchnorm/ReadVariableOp¢Htransformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOp¢Dtransformer_block_11/layer_normalization_23/batchnorm/ReadVariableOp¢Htransformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOp¢Ptransformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOp¢Ztransformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp¢Ctransformer_block_11/multi_head_attention_11/key/add/ReadVariableOp¢Mtransformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOp¢Etransformer_block_11/multi_head_attention_11/query/add/ReadVariableOp¢Otransformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp¢Etransformer_block_11/multi_head_attention_11/value/add/ReadVariableOp¢Otransformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp¢Btransformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOp¢Dtransformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOp¢Btransformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOp¢Dtransformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOp
$token_and_position_embedding_5/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_5/Shape»
2token_and_position_embedding_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ24
2token_and_position_embedding_5/strided_slice/stack¶
4token_and_position_embedding_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_5/strided_slice/stack_1¶
4token_and_position_embedding_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_5/strided_slice/stack_2
,token_and_position_embedding_5/strided_sliceStridedSlice-token_and_position_embedding_5/Shape:output:0;token_and_position_embedding_5/strided_slice/stack:output:0=token_and_position_embedding_5/strided_slice/stack_1:output:0=token_and_position_embedding_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_5/strided_slice
*token_and_position_embedding_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_5/range/start
*token_and_position_embedding_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_5/range/delta
$token_and_position_embedding_5/rangeRange3token_and_position_embedding_5/range/start:output:05token_and_position_embedding_5/strided_slice:output:03token_and_position_embedding_5/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$token_and_position_embedding_5/rangeÎ
<token_and_position_embedding_5/embedding_11/embedding_lookupResourceGatherCtoken_and_position_embedding_5_embedding_11_embedding_lookup_765444-token_and_position_embedding_5/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*V
_classL
JHloc:@token_and_position_embedding_5/embedding_11/embedding_lookup/765444*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02>
<token_and_position_embedding_5/embedding_11/embedding_lookup
Etoken_and_position_embedding_5/embedding_11/embedding_lookup/IdentityIdentityEtoken_and_position_embedding_5/embedding_11/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@token_and_position_embedding_5/embedding_11/embedding_lookup/765444*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2G
Etoken_and_position_embedding_5/embedding_11/embedding_lookup/Identity 
Gtoken_and_position_embedding_5/embedding_11/embedding_lookup/Identity_1IdentityNtoken_and_position_embedding_5/embedding_11/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2I
Gtoken_and_position_embedding_5/embedding_11/embedding_lookup/Identity_1¸
0token_and_position_embedding_5/embedding_10/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR22
0token_and_position_embedding_5/embedding_10/CastÚ
<token_and_position_embedding_5/embedding_10/embedding_lookupResourceGatherCtoken_and_position_embedding_5_embedding_10_embedding_lookup_7654504token_and_position_embedding_5/embedding_10/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*V
_classL
JHloc:@token_and_position_embedding_5/embedding_10/embedding_lookup/765450*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02>
<token_and_position_embedding_5/embedding_10/embedding_lookup
Etoken_and_position_embedding_5/embedding_10/embedding_lookup/IdentityIdentityEtoken_and_position_embedding_5/embedding_10/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@token_and_position_embedding_5/embedding_10/embedding_lookup/765450*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2G
Etoken_and_position_embedding_5/embedding_10/embedding_lookup/Identity¥
Gtoken_and_position_embedding_5/embedding_10/embedding_lookup/Identity_1IdentityNtoken_and_position_embedding_5/embedding_10/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2I
Gtoken_and_position_embedding_5/embedding_10/embedding_lookup/Identity_1¬
"token_and_position_embedding_5/addAddV2Ptoken_and_position_embedding_5/embedding_10/embedding_lookup/Identity_1:output:0Ptoken_and_position_embedding_5/embedding_11/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2$
"token_and_position_embedding_5/add
conv1d_10/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2!
conv1d_10/conv1d/ExpandDims/dimÕ
conv1d_10/conv1d/ExpandDims
ExpandDims&token_and_position_embedding_5/add:z:0(conv1d_10/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_10/conv1d/ExpandDimsÖ
,conv1d_10/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_10_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02.
,conv1d_10/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_10/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_10/conv1d/ExpandDims_1/dimß
conv1d_10/conv1d/ExpandDims_1
ExpandDims4conv1d_10/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_10/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_10/conv1d/ExpandDims_1ß
conv1d_10/conv1dConv2D$conv1d_10/conv1d/ExpandDims:output:0&conv1d_10/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
paddingSAME*
strides
2
conv1d_10/conv1d±
conv1d_10/conv1d/SqueezeSqueezeconv1d_10/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_10/conv1d/Squeezeª
 conv1d_10/BiasAdd/ReadVariableOpReadVariableOp)conv1d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv1d_10/BiasAdd/ReadVariableOpµ
conv1d_10/BiasAddBiasAdd!conv1d_10/conv1d/Squeeze:output:0(conv1d_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_10/BiasAdd{
conv1d_10/ReluReluconv1d_10/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d_10/Relu
#average_pooling1d_15/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_15/ExpandDims/dim×
average_pooling1d_15/ExpandDims
ExpandDimsconv1d_10/Relu:activations:0,average_pooling1d_15/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2!
average_pooling1d_15/ExpandDimsè
average_pooling1d_15/AvgPoolAvgPool(average_pooling1d_15/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_15/AvgPool¼
average_pooling1d_15/SqueezeSqueeze%average_pooling1d_15/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims
2
average_pooling1d_15/Squeeze
conv1d_11/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2!
conv1d_11/conv1d/ExpandDims/dimÔ
conv1d_11/conv1d/ExpandDims
ExpandDims%average_pooling1d_15/Squeeze:output:0(conv1d_11/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_11/conv1d/ExpandDimsÖ
,conv1d_11/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_11_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02.
,conv1d_11/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_11/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_11/conv1d/ExpandDims_1/dimß
conv1d_11/conv1d/ExpandDims_1
ExpandDims4conv1d_11/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_11/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d_11/conv1d/ExpandDims_1ß
conv1d_11/conv1dConv2D$conv1d_11/conv1d/ExpandDims:output:0&conv1d_11/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
paddingSAME*
strides
2
conv1d_11/conv1d±
conv1d_11/conv1d/SqueezeSqueezeconv1d_11/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_11/conv1d/Squeezeª
 conv1d_11/BiasAdd/ReadVariableOpReadVariableOp)conv1d_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv1d_11/BiasAdd/ReadVariableOpµ
conv1d_11/BiasAddBiasAdd!conv1d_11/conv1d/Squeeze:output:0(conv1d_11/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_11/BiasAdd{
conv1d_11/ReluReluconv1d_11/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_11/Relu
#average_pooling1d_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_17/ExpandDims/dimá
average_pooling1d_17/ExpandDims
ExpandDims&token_and_position_embedding_5/add:z:0,average_pooling1d_17/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2!
average_pooling1d_17/ExpandDimsé
average_pooling1d_17/AvgPoolAvgPool(average_pooling1d_17/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize	
¬*
paddingVALID*
strides	
¬2
average_pooling1d_17/AvgPool»
average_pooling1d_17/SqueezeSqueeze%average_pooling1d_17/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_17/Squeeze
#average_pooling1d_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_16/ExpandDims/dim×
average_pooling1d_16/ExpandDims
ExpandDimsconv1d_11/Relu:activations:0,average_pooling1d_16/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2!
average_pooling1d_16/ExpandDimsç
average_pooling1d_16/AvgPoolAvgPool(average_pooling1d_16/ExpandDims:output:0*
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
average_pooling1d_16/AvgPool»
average_pooling1d_16/SqueezeSqueeze%average_pooling1d_16/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_16/Squeeze¿
5batch_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_10/moments/mean/reduction_indices÷
#batch_normalization_10/moments/meanMean%average_pooling1d_16/Squeeze:output:0>batch_normalization_10/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2%
#batch_normalization_10/moments/meanÅ
+batch_normalization_10/moments/StopGradientStopGradient,batch_normalization_10/moments/mean:output:0*
T0*"
_output_shapes
: 2-
+batch_normalization_10/moments/StopGradient
0batch_normalization_10/moments/SquaredDifferenceSquaredDifference%average_pooling1d_16/Squeeze:output:04batch_normalization_10/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0batch_normalization_10/moments/SquaredDifferenceÇ
9batch_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_10/moments/variance/reduction_indices
'batch_normalization_10/moments/varianceMean4batch_normalization_10/moments/SquaredDifference:z:0Bbatch_normalization_10/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2)
'batch_normalization_10/moments/varianceÆ
&batch_normalization_10/moments/SqueezeSqueeze,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2(
&batch_normalization_10/moments/SqueezeÎ
(batch_normalization_10/moments/Squeeze_1Squeeze0batch_normalization_10/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2*
(batch_normalization_10/moments/Squeeze_1
,batch_normalization_10/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_10/AssignMovingAvg/765500*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_10/AssignMovingAvg/decayØ
5batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_10_assignmovingavg_765500*
_output_shapes
: *
dtype027
5batch_normalization_10/AssignMovingAvg/ReadVariableOpä
*batch_normalization_10/AssignMovingAvg/subSub=batch_normalization_10/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_10/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_10/AssignMovingAvg/765500*
_output_shapes
: 2,
*batch_normalization_10/AssignMovingAvg/subÛ
*batch_normalization_10/AssignMovingAvg/mulMul.batch_normalization_10/AssignMovingAvg/sub:z:05batch_normalization_10/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_10/AssignMovingAvg/765500*
_output_shapes
: 2,
*batch_normalization_10/AssignMovingAvg/mul¹
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_10_assignmovingavg_765500.batch_normalization_10/AssignMovingAvg/mul:z:06^batch_normalization_10/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_10/AssignMovingAvg/765500*
_output_shapes
 *
dtype02<
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_10/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg_1/765506*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_10/AssignMovingAvg_1/decayÞ
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_10_assignmovingavg_1_765506*
_output_shapes
: *
dtype029
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpî
,batch_normalization_10/AssignMovingAvg_1/subSub?batch_normalization_10/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_10/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg_1/765506*
_output_shapes
: 2.
,batch_normalization_10/AssignMovingAvg_1/subå
,batch_normalization_10/AssignMovingAvg_1/mulMul0batch_normalization_10/AssignMovingAvg_1/sub:z:07batch_normalization_10/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg_1/765506*
_output_shapes
: 2.
,batch_normalization_10/AssignMovingAvg_1/mulÅ
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_10_assignmovingavg_1_7655060batch_normalization_10/AssignMovingAvg_1/mul:z:08^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg_1/765506*
_output_shapes
 *
dtype02>
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_10/batchnorm/add/yÞ
$batch_normalization_10/batchnorm/addAddV21batch_normalization_10/moments/Squeeze_1:output:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_10/batchnorm/add¨
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_10/batchnorm/Rsqrtã
3batch_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_10/batchnorm/mul/ReadVariableOpá
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:0;batch_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_10/batchnorm/mulÞ
&batch_normalization_10/batchnorm/mul_1Mul%average_pooling1d_16/Squeeze:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&batch_normalization_10/batchnorm/mul_1×
&batch_normalization_10/batchnorm/mul_2Mul/batch_normalization_10/moments/Squeeze:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_10/batchnorm/mul_2×
/batch_normalization_10/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_10/batchnorm/ReadVariableOpÝ
$batch_normalization_10/batchnorm/subSub7batch_normalization_10/batchnorm/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_10/batchnorm/subå
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&batch_normalization_10/batchnorm/add_1¿
5batch_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_11/moments/mean/reduction_indices÷
#batch_normalization_11/moments/meanMean%average_pooling1d_17/Squeeze:output:0>batch_normalization_11/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2%
#batch_normalization_11/moments/meanÅ
+batch_normalization_11/moments/StopGradientStopGradient,batch_normalization_11/moments/mean:output:0*
T0*"
_output_shapes
: 2-
+batch_normalization_11/moments/StopGradient
0batch_normalization_11/moments/SquaredDifferenceSquaredDifference%average_pooling1d_17/Squeeze:output:04batch_normalization_11/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0batch_normalization_11/moments/SquaredDifferenceÇ
9batch_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_11/moments/variance/reduction_indices
'batch_normalization_11/moments/varianceMean4batch_normalization_11/moments/SquaredDifference:z:0Bbatch_normalization_11/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2)
'batch_normalization_11/moments/varianceÆ
&batch_normalization_11/moments/SqueezeSqueeze,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2(
&batch_normalization_11/moments/SqueezeÎ
(batch_normalization_11/moments/Squeeze_1Squeeze0batch_normalization_11/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2*
(batch_normalization_11/moments/Squeeze_1
,batch_normalization_11/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_11/AssignMovingAvg/765532*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_11/AssignMovingAvg/decayØ
5batch_normalization_11/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_11_assignmovingavg_765532*
_output_shapes
: *
dtype027
5batch_normalization_11/AssignMovingAvg/ReadVariableOpä
*batch_normalization_11/AssignMovingAvg/subSub=batch_normalization_11/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_11/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_11/AssignMovingAvg/765532*
_output_shapes
: 2,
*batch_normalization_11/AssignMovingAvg/subÛ
*batch_normalization_11/AssignMovingAvg/mulMul.batch_normalization_11/AssignMovingAvg/sub:z:05batch_normalization_11/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_11/AssignMovingAvg/765532*
_output_shapes
: 2,
*batch_normalization_11/AssignMovingAvg/mul¹
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_11_assignmovingavg_765532.batch_normalization_11/AssignMovingAvg/mul:z:06^batch_normalization_11/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_11/AssignMovingAvg/765532*
_output_shapes
 *
dtype02<
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_11/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg_1/765538*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_11/AssignMovingAvg_1/decayÞ
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_11_assignmovingavg_1_765538*
_output_shapes
: *
dtype029
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpî
,batch_normalization_11/AssignMovingAvg_1/subSub?batch_normalization_11/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_11/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg_1/765538*
_output_shapes
: 2.
,batch_normalization_11/AssignMovingAvg_1/subå
,batch_normalization_11/AssignMovingAvg_1/mulMul0batch_normalization_11/AssignMovingAvg_1/sub:z:07batch_normalization_11/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg_1/765538*
_output_shapes
: 2.
,batch_normalization_11/AssignMovingAvg_1/mulÅ
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_11_assignmovingavg_1_7655380batch_normalization_11/AssignMovingAvg_1/mul:z:08^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg_1/765538*
_output_shapes
 *
dtype02>
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_11/batchnorm/add/yÞ
$batch_normalization_11/batchnorm/addAddV21batch_normalization_11/moments/Squeeze_1:output:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_11/batchnorm/add¨
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_11/batchnorm/Rsqrtã
3batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_11/batchnorm/mul/ReadVariableOpá
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:0;batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_11/batchnorm/mulÞ
&batch_normalization_11/batchnorm/mul_1Mul%average_pooling1d_17/Squeeze:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&batch_normalization_11/batchnorm/mul_1×
&batch_normalization_11/batchnorm/mul_2Mul/batch_normalization_11/moments/Squeeze:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_11/batchnorm/mul_2×
/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_11/batchnorm/ReadVariableOpÝ
$batch_normalization_11/batchnorm/subSub7batch_normalization_11/batchnorm/ReadVariableOp:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_11/batchnorm/subå
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&batch_normalization_11/batchnorm/add_1­
	add_5/addAddV2*batch_normalization_10/batchnorm/add_1:z:0*batch_normalization_11/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
	add_5/add¿
Otransformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOpReadVariableOpXtransformer_block_11_multi_head_attention_11_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Q
Otransformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOpÖ
@transformer_block_11/multi_head_attention_11/query/einsum/EinsumEinsumadd_5/add:z:0Wtransformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2B
@transformer_block_11/multi_head_attention_11/query/einsum/Einsum
Etransformer_block_11/multi_head_attention_11/query/add/ReadVariableOpReadVariableOpNtransformer_block_11_multi_head_attention_11_query_add_readvariableop_resource*
_output_shapes

: *
dtype02G
Etransformer_block_11/multi_head_attention_11/query/add/ReadVariableOpÍ
6transformer_block_11/multi_head_attention_11/query/addAddV2Itransformer_block_11/multi_head_attention_11/query/einsum/Einsum:output:0Mtransformer_block_11/multi_head_attention_11/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 28
6transformer_block_11/multi_head_attention_11/query/add¹
Mtransformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_11_multi_head_attention_11_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpÐ
>transformer_block_11/multi_head_attention_11/key/einsum/EinsumEinsumadd_5/add:z:0Utransformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2@
>transformer_block_11/multi_head_attention_11/key/einsum/Einsum
Ctransformer_block_11/multi_head_attention_11/key/add/ReadVariableOpReadVariableOpLtransformer_block_11_multi_head_attention_11_key_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_11/multi_head_attention_11/key/add/ReadVariableOpÅ
4transformer_block_11/multi_head_attention_11/key/addAddV2Gtransformer_block_11/multi_head_attention_11/key/einsum/Einsum:output:0Ktransformer_block_11/multi_head_attention_11/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_11/multi_head_attention_11/key/add¿
Otransformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOpReadVariableOpXtransformer_block_11_multi_head_attention_11_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Q
Otransformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOpÖ
@transformer_block_11/multi_head_attention_11/value/einsum/EinsumEinsumadd_5/add:z:0Wtransformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2B
@transformer_block_11/multi_head_attention_11/value/einsum/Einsum
Etransformer_block_11/multi_head_attention_11/value/add/ReadVariableOpReadVariableOpNtransformer_block_11_multi_head_attention_11_value_add_readvariableop_resource*
_output_shapes

: *
dtype02G
Etransformer_block_11/multi_head_attention_11/value/add/ReadVariableOpÍ
6transformer_block_11/multi_head_attention_11/value/addAddV2Itransformer_block_11/multi_head_attention_11/value/einsum/Einsum:output:0Mtransformer_block_11/multi_head_attention_11/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 28
6transformer_block_11/multi_head_attention_11/value/add­
2transformer_block_11/multi_head_attention_11/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>24
2transformer_block_11/multi_head_attention_11/Mul/y
0transformer_block_11/multi_head_attention_11/MulMul:transformer_block_11/multi_head_attention_11/query/add:z:0;transformer_block_11/multi_head_attention_11/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0transformer_block_11/multi_head_attention_11/MulÔ
:transformer_block_11/multi_head_attention_11/einsum/EinsumEinsum8transformer_block_11/multi_head_attention_11/key/add:z:04transformer_block_11/multi_head_attention_11/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2<
:transformer_block_11/multi_head_attention_11/einsum/Einsum
<transformer_block_11/multi_head_attention_11/softmax/SoftmaxSoftmaxCtransformer_block_11/multi_head_attention_11/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2>
<transformer_block_11/multi_head_attention_11/softmax/SoftmaxÍ
Btransformer_block_11/multi_head_attention_11/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2D
Btransformer_block_11/multi_head_attention_11/dropout/dropout/ConstÚ
@transformer_block_11/multi_head_attention_11/dropout/dropout/MulMulFtransformer_block_11/multi_head_attention_11/softmax/Softmax:softmax:0Ktransformer_block_11/multi_head_attention_11/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2B
@transformer_block_11/multi_head_attention_11/dropout/dropout/Mulþ
Btransformer_block_11/multi_head_attention_11/dropout/dropout/ShapeShapeFtransformer_block_11/multi_head_attention_11/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2D
Btransformer_block_11/multi_head_attention_11/dropout/dropout/ShapeÛ
Ytransformer_block_11/multi_head_attention_11/dropout/dropout/random_uniform/RandomUniformRandomUniformKtransformer_block_11/multi_head_attention_11/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
dtype02[
Ytransformer_block_11/multi_head_attention_11/dropout/dropout/random_uniform/RandomUniformß
Ktransformer_block_11/multi_head_attention_11/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2M
Ktransformer_block_11/multi_head_attention_11/dropout/dropout/GreaterEqual/y
Itransformer_block_11/multi_head_attention_11/dropout/dropout/GreaterEqualGreaterEqualbtransformer_block_11/multi_head_attention_11/dropout/dropout/random_uniform/RandomUniform:output:0Ttransformer_block_11/multi_head_attention_11/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2K
Itransformer_block_11/multi_head_attention_11/dropout/dropout/GreaterEqual¦
Atransformer_block_11/multi_head_attention_11/dropout/dropout/CastCastMtransformer_block_11/multi_head_attention_11/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2C
Atransformer_block_11/multi_head_attention_11/dropout/dropout/CastÖ
Btransformer_block_11/multi_head_attention_11/dropout/dropout/Mul_1MulDtransformer_block_11/multi_head_attention_11/dropout/dropout/Mul:z:0Etransformer_block_11/multi_head_attention_11/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2D
Btransformer_block_11/multi_head_attention_11/dropout/dropout/Mul_1ì
<transformer_block_11/multi_head_attention_11/einsum_1/EinsumEinsumFtransformer_block_11/multi_head_attention_11/dropout/dropout/Mul_1:z:0:transformer_block_11/multi_head_attention_11/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2>
<transformer_block_11/multi_head_attention_11/einsum_1/Einsumà
Ztransformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpctransformer_block_11_multi_head_attention_11_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02\
Ztransformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp«
Ktransformer_block_11/multi_head_attention_11/attention_output/einsum/EinsumEinsumEtransformer_block_11/multi_head_attention_11/einsum_1/Einsum:output:0btransformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe2M
Ktransformer_block_11/multi_head_attention_11/attention_output/einsum/Einsumº
Ptransformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOpReadVariableOpYtransformer_block_11_multi_head_attention_11_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02R
Ptransformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOpõ
Atransformer_block_11/multi_head_attention_11/attention_output/addAddV2Ttransformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum:output:0Xtransformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2C
Atransformer_block_11/multi_head_attention_11/attention_output/add£
-transformer_block_11/dropout_32/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2/
-transformer_block_11/dropout_32/dropout/Const
+transformer_block_11/dropout_32/dropout/MulMulEtransformer_block_11/multi_head_attention_11/attention_output/add:z:06transformer_block_11/dropout_32/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+transformer_block_11/dropout_32/dropout/MulÓ
-transformer_block_11/dropout_32/dropout/ShapeShapeEtransformer_block_11/multi_head_attention_11/attention_output/add:z:0*
T0*
_output_shapes
:2/
-transformer_block_11/dropout_32/dropout/Shape
Dtransformer_block_11/dropout_32/dropout/random_uniform/RandomUniformRandomUniform6transformer_block_11/dropout_32/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype02F
Dtransformer_block_11/dropout_32/dropout/random_uniform/RandomUniformµ
6transformer_block_11/dropout_32/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=28
6transformer_block_11/dropout_32/dropout/GreaterEqual/yÂ
4transformer_block_11/dropout_32/dropout/GreaterEqualGreaterEqualMtransformer_block_11/dropout_32/dropout/random_uniform/RandomUniform:output:0?transformer_block_11/dropout_32/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_11/dropout_32/dropout/GreaterEqualã
,transformer_block_11/dropout_32/dropout/CastCast8transformer_block_11/dropout_32/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2.
,transformer_block_11/dropout_32/dropout/Castþ
-transformer_block_11/dropout_32/dropout/Mul_1Mul/transformer_block_11/dropout_32/dropout/Mul:z:00transformer_block_11/dropout_32/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2/
-transformer_block_11/dropout_32/dropout/Mul_1µ
transformer_block_11/addAddV2add_5/add:z:01transformer_block_11/dropout_32/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_11/addâ
Jtransformer_block_11/layer_normalization_22/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block_11/layer_normalization_22/moments/mean/reduction_indices¶
8transformer_block_11/layer_normalization_22/moments/meanMeantransformer_block_11/add:z:0Stransformer_block_11/layer_normalization_22/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2:
8transformer_block_11/layer_normalization_22/moments/mean
@transformer_block_11/layer_normalization_22/moments/StopGradientStopGradientAtransformer_block_11/layer_normalization_22/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2B
@transformer_block_11/layer_normalization_22/moments/StopGradientÂ
Etransformer_block_11/layer_normalization_22/moments/SquaredDifferenceSquaredDifferencetransformer_block_11/add:z:0Itransformer_block_11/layer_normalization_22/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2G
Etransformer_block_11/layer_normalization_22/moments/SquaredDifferenceê
Ntransformer_block_11/layer_normalization_22/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
Ntransformer_block_11/layer_normalization_22/moments/variance/reduction_indicesï
<transformer_block_11/layer_normalization_22/moments/varianceMeanItransformer_block_11/layer_normalization_22/moments/SquaredDifference:z:0Wtransformer_block_11/layer_normalization_22/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2>
<transformer_block_11/layer_normalization_22/moments/variance¿
;transformer_block_11/layer_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752=
;transformer_block_11/layer_normalization_22/batchnorm/add/yÂ
9transformer_block_11/layer_normalization_22/batchnorm/addAddV2Etransformer_block_11/layer_normalization_22/moments/variance:output:0Dtransformer_block_11/layer_normalization_22/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2;
9transformer_block_11/layer_normalization_22/batchnorm/addø
;transformer_block_11/layer_normalization_22/batchnorm/RsqrtRsqrt=transformer_block_11/layer_normalization_22/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2=
;transformer_block_11/layer_normalization_22/batchnorm/Rsqrt¢
Htransformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOpQtransformer_block_11_layer_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02J
Htransformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOpÆ
9transformer_block_11/layer_normalization_22/batchnorm/mulMul?transformer_block_11/layer_normalization_22/batchnorm/Rsqrt:y:0Ptransformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_11/layer_normalization_22/batchnorm/mul
;transformer_block_11/layer_normalization_22/batchnorm/mul_1Multransformer_block_11/add:z:0=transformer_block_11/layer_normalization_22/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2=
;transformer_block_11/layer_normalization_22/batchnorm/mul_1¹
;transformer_block_11/layer_normalization_22/batchnorm/mul_2MulAtransformer_block_11/layer_normalization_22/moments/mean:output:0=transformer_block_11/layer_normalization_22/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2=
;transformer_block_11/layer_normalization_22/batchnorm/mul_2
Dtransformer_block_11/layer_normalization_22/batchnorm/ReadVariableOpReadVariableOpMtransformer_block_11_layer_normalization_22_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02F
Dtransformer_block_11/layer_normalization_22/batchnorm/ReadVariableOpÂ
9transformer_block_11/layer_normalization_22/batchnorm/subSubLtransformer_block_11/layer_normalization_22/batchnorm/ReadVariableOp:value:0?transformer_block_11/layer_normalization_22/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_11/layer_normalization_22/batchnorm/sub¹
;transformer_block_11/layer_normalization_22/batchnorm/add_1AddV2?transformer_block_11/layer_normalization_22/batchnorm/mul_1:z:0=transformer_block_11/layer_normalization_22/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2=
;transformer_block_11/layer_normalization_22/batchnorm/add_1
Dtransformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOpReadVariableOpMtransformer_block_11_sequential_11_dense_37_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02F
Dtransformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOpÂ
:transformer_block_11/sequential_11/dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2<
:transformer_block_11/sequential_11/dense_37/Tensordot/axesÉ
:transformer_block_11/sequential_11/dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2<
:transformer_block_11/sequential_11/dense_37/Tensordot/freeé
;transformer_block_11/sequential_11/dense_37/Tensordot/ShapeShape?transformer_block_11/layer_normalization_22/batchnorm/add_1:z:0*
T0*
_output_shapes
:2=
;transformer_block_11/sequential_11/dense_37/Tensordot/ShapeÌ
Ctransformer_block_11/sequential_11/dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_11/sequential_11/dense_37/Tensordot/GatherV2/axis­
>transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2GatherV2Dtransformer_block_11/sequential_11/dense_37/Tensordot/Shape:output:0Ctransformer_block_11/sequential_11/dense_37/Tensordot/free:output:0Ltransformer_block_11/sequential_11/dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2Ð
Etransformer_block_11/sequential_11/dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2G
Etransformer_block_11/sequential_11/dense_37/Tensordot/GatherV2_1/axis³
@transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2_1GatherV2Dtransformer_block_11/sequential_11/dense_37/Tensordot/Shape:output:0Ctransformer_block_11/sequential_11/dense_37/Tensordot/axes:output:0Ntransformer_block_11/sequential_11/dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2B
@transformer_block_11/sequential_11/dense_37/Tensordot/GatherV2_1Ä
;transformer_block_11/sequential_11/dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_11/sequential_11/dense_37/Tensordot/Const°
:transformer_block_11/sequential_11/dense_37/Tensordot/ProdProdGtransformer_block_11/sequential_11/dense_37/Tensordot/GatherV2:output:0Dtransformer_block_11/sequential_11/dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: 2<
:transformer_block_11/sequential_11/dense_37/Tensordot/ProdÈ
=transformer_block_11/sequential_11/dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=transformer_block_11/sequential_11/dense_37/Tensordot/Const_1¸
<transformer_block_11/sequential_11/dense_37/Tensordot/Prod_1ProdItransformer_block_11/sequential_11/dense_37/Tensordot/GatherV2_1:output:0Ftransformer_block_11/sequential_11/dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2>
<transformer_block_11/sequential_11/dense_37/Tensordot/Prod_1È
Atransformer_block_11/sequential_11/dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_11/sequential_11/dense_37/Tensordot/concat/axis
<transformer_block_11/sequential_11/dense_37/Tensordot/concatConcatV2Ctransformer_block_11/sequential_11/dense_37/Tensordot/free:output:0Ctransformer_block_11/sequential_11/dense_37/Tensordot/axes:output:0Jtransformer_block_11/sequential_11/dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_11/sequential_11/dense_37/Tensordot/concat¼
;transformer_block_11/sequential_11/dense_37/Tensordot/stackPackCtransformer_block_11/sequential_11/dense_37/Tensordot/Prod:output:0Etransformer_block_11/sequential_11/dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_11/sequential_11/dense_37/Tensordot/stackÍ
?transformer_block_11/sequential_11/dense_37/Tensordot/transpose	Transpose?transformer_block_11/layer_normalization_22/batchnorm/add_1:z:0Etransformer_block_11/sequential_11/dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?transformer_block_11/sequential_11/dense_37/Tensordot/transposeÏ
=transformer_block_11/sequential_11/dense_37/Tensordot/ReshapeReshapeCtransformer_block_11/sequential_11/dense_37/Tensordot/transpose:y:0Dtransformer_block_11/sequential_11/dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2?
=transformer_block_11/sequential_11/dense_37/Tensordot/ReshapeÎ
<transformer_block_11/sequential_11/dense_37/Tensordot/MatMulMatMulFtransformer_block_11/sequential_11/dense_37/Tensordot/Reshape:output:0Ltransformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2>
<transformer_block_11/sequential_11/dense_37/Tensordot/MatMulÈ
=transformer_block_11/sequential_11/dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2?
=transformer_block_11/sequential_11/dense_37/Tensordot/Const_2Ì
Ctransformer_block_11/sequential_11/dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_11/sequential_11/dense_37/Tensordot/concat_1/axis
>transformer_block_11/sequential_11/dense_37/Tensordot/concat_1ConcatV2Gtransformer_block_11/sequential_11/dense_37/Tensordot/GatherV2:output:0Ftransformer_block_11/sequential_11/dense_37/Tensordot/Const_2:output:0Ltransformer_block_11/sequential_11/dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2@
>transformer_block_11/sequential_11/dense_37/Tensordot/concat_1À
5transformer_block_11/sequential_11/dense_37/TensordotReshapeFtransformer_block_11/sequential_11/dense_37/Tensordot/MatMul:product:0Gtransformer_block_11/sequential_11/dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@27
5transformer_block_11/sequential_11/dense_37/Tensordot
Btransformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOpReadVariableOpKtransformer_block_11_sequential_11_dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02D
Btransformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOp·
3transformer_block_11/sequential_11/dense_37/BiasAddBiasAdd>transformer_block_11/sequential_11/dense_37/Tensordot:output:0Jtransformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@25
3transformer_block_11/sequential_11/dense_37/BiasAddà
0transformer_block_11/sequential_11/dense_37/ReluRelu<transformer_block_11/sequential_11/dense_37/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@22
0transformer_block_11/sequential_11/dense_37/Relu
Dtransformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOpReadVariableOpMtransformer_block_11_sequential_11_dense_38_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02F
Dtransformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOpÂ
:transformer_block_11/sequential_11/dense_38/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2<
:transformer_block_11/sequential_11/dense_38/Tensordot/axesÉ
:transformer_block_11/sequential_11/dense_38/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2<
:transformer_block_11/sequential_11/dense_38/Tensordot/freeè
;transformer_block_11/sequential_11/dense_38/Tensordot/ShapeShape>transformer_block_11/sequential_11/dense_37/Relu:activations:0*
T0*
_output_shapes
:2=
;transformer_block_11/sequential_11/dense_38/Tensordot/ShapeÌ
Ctransformer_block_11/sequential_11/dense_38/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_11/sequential_11/dense_38/Tensordot/GatherV2/axis­
>transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2GatherV2Dtransformer_block_11/sequential_11/dense_38/Tensordot/Shape:output:0Ctransformer_block_11/sequential_11/dense_38/Tensordot/free:output:0Ltransformer_block_11/sequential_11/dense_38/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2Ð
Etransformer_block_11/sequential_11/dense_38/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2G
Etransformer_block_11/sequential_11/dense_38/Tensordot/GatherV2_1/axis³
@transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2_1GatherV2Dtransformer_block_11/sequential_11/dense_38/Tensordot/Shape:output:0Ctransformer_block_11/sequential_11/dense_38/Tensordot/axes:output:0Ntransformer_block_11/sequential_11/dense_38/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2B
@transformer_block_11/sequential_11/dense_38/Tensordot/GatherV2_1Ä
;transformer_block_11/sequential_11/dense_38/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_11/sequential_11/dense_38/Tensordot/Const°
:transformer_block_11/sequential_11/dense_38/Tensordot/ProdProdGtransformer_block_11/sequential_11/dense_38/Tensordot/GatherV2:output:0Dtransformer_block_11/sequential_11/dense_38/Tensordot/Const:output:0*
T0*
_output_shapes
: 2<
:transformer_block_11/sequential_11/dense_38/Tensordot/ProdÈ
=transformer_block_11/sequential_11/dense_38/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=transformer_block_11/sequential_11/dense_38/Tensordot/Const_1¸
<transformer_block_11/sequential_11/dense_38/Tensordot/Prod_1ProdItransformer_block_11/sequential_11/dense_38/Tensordot/GatherV2_1:output:0Ftransformer_block_11/sequential_11/dense_38/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2>
<transformer_block_11/sequential_11/dense_38/Tensordot/Prod_1È
Atransformer_block_11/sequential_11/dense_38/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_11/sequential_11/dense_38/Tensordot/concat/axis
<transformer_block_11/sequential_11/dense_38/Tensordot/concatConcatV2Ctransformer_block_11/sequential_11/dense_38/Tensordot/free:output:0Ctransformer_block_11/sequential_11/dense_38/Tensordot/axes:output:0Jtransformer_block_11/sequential_11/dense_38/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_11/sequential_11/dense_38/Tensordot/concat¼
;transformer_block_11/sequential_11/dense_38/Tensordot/stackPackCtransformer_block_11/sequential_11/dense_38/Tensordot/Prod:output:0Etransformer_block_11/sequential_11/dense_38/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_11/sequential_11/dense_38/Tensordot/stackÌ
?transformer_block_11/sequential_11/dense_38/Tensordot/transpose	Transpose>transformer_block_11/sequential_11/dense_37/Relu:activations:0Etransformer_block_11/sequential_11/dense_38/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2A
?transformer_block_11/sequential_11/dense_38/Tensordot/transposeÏ
=transformer_block_11/sequential_11/dense_38/Tensordot/ReshapeReshapeCtransformer_block_11/sequential_11/dense_38/Tensordot/transpose:y:0Dtransformer_block_11/sequential_11/dense_38/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2?
=transformer_block_11/sequential_11/dense_38/Tensordot/ReshapeÎ
<transformer_block_11/sequential_11/dense_38/Tensordot/MatMulMatMulFtransformer_block_11/sequential_11/dense_38/Tensordot/Reshape:output:0Ltransformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2>
<transformer_block_11/sequential_11/dense_38/Tensordot/MatMulÈ
=transformer_block_11/sequential_11/dense_38/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2?
=transformer_block_11/sequential_11/dense_38/Tensordot/Const_2Ì
Ctransformer_block_11/sequential_11/dense_38/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_11/sequential_11/dense_38/Tensordot/concat_1/axis
>transformer_block_11/sequential_11/dense_38/Tensordot/concat_1ConcatV2Gtransformer_block_11/sequential_11/dense_38/Tensordot/GatherV2:output:0Ftransformer_block_11/sequential_11/dense_38/Tensordot/Const_2:output:0Ltransformer_block_11/sequential_11/dense_38/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2@
>transformer_block_11/sequential_11/dense_38/Tensordot/concat_1À
5transformer_block_11/sequential_11/dense_38/TensordotReshapeFtransformer_block_11/sequential_11/dense_38/Tensordot/MatMul:product:0Gtransformer_block_11/sequential_11/dense_38/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 27
5transformer_block_11/sequential_11/dense_38/Tensordot
Btransformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOpReadVariableOpKtransformer_block_11_sequential_11_dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOp·
3transformer_block_11/sequential_11/dense_38/BiasAddBiasAdd>transformer_block_11/sequential_11/dense_38/Tensordot:output:0Jtransformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 25
3transformer_block_11/sequential_11/dense_38/BiasAdd£
-transformer_block_11/dropout_33/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2/
-transformer_block_11/dropout_33/dropout/Const
+transformer_block_11/dropout_33/dropout/MulMul<transformer_block_11/sequential_11/dense_38/BiasAdd:output:06transformer_block_11/dropout_33/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+transformer_block_11/dropout_33/dropout/MulÊ
-transformer_block_11/dropout_33/dropout/ShapeShape<transformer_block_11/sequential_11/dense_38/BiasAdd:output:0*
T0*
_output_shapes
:2/
-transformer_block_11/dropout_33/dropout/Shape
Dtransformer_block_11/dropout_33/dropout/random_uniform/RandomUniformRandomUniform6transformer_block_11/dropout_33/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype02F
Dtransformer_block_11/dropout_33/dropout/random_uniform/RandomUniformµ
6transformer_block_11/dropout_33/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=28
6transformer_block_11/dropout_33/dropout/GreaterEqual/yÂ
4transformer_block_11/dropout_33/dropout/GreaterEqualGreaterEqualMtransformer_block_11/dropout_33/dropout/random_uniform/RandomUniform:output:0?transformer_block_11/dropout_33/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_11/dropout_33/dropout/GreaterEqualã
,transformer_block_11/dropout_33/dropout/CastCast8transformer_block_11/dropout_33/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2.
,transformer_block_11/dropout_33/dropout/Castþ
-transformer_block_11/dropout_33/dropout/Mul_1Mul/transformer_block_11/dropout_33/dropout/Mul:z:00transformer_block_11/dropout_33/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2/
-transformer_block_11/dropout_33/dropout/Mul_1ë
transformer_block_11/add_1AddV2?transformer_block_11/layer_normalization_22/batchnorm/add_1:z:01transformer_block_11/dropout_33/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_11/add_1â
Jtransformer_block_11/layer_normalization_23/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block_11/layer_normalization_23/moments/mean/reduction_indices¸
8transformer_block_11/layer_normalization_23/moments/meanMeantransformer_block_11/add_1:z:0Stransformer_block_11/layer_normalization_23/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2:
8transformer_block_11/layer_normalization_23/moments/mean
@transformer_block_11/layer_normalization_23/moments/StopGradientStopGradientAtransformer_block_11/layer_normalization_23/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2B
@transformer_block_11/layer_normalization_23/moments/StopGradientÄ
Etransformer_block_11/layer_normalization_23/moments/SquaredDifferenceSquaredDifferencetransformer_block_11/add_1:z:0Itransformer_block_11/layer_normalization_23/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2G
Etransformer_block_11/layer_normalization_23/moments/SquaredDifferenceê
Ntransformer_block_11/layer_normalization_23/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
Ntransformer_block_11/layer_normalization_23/moments/variance/reduction_indicesï
<transformer_block_11/layer_normalization_23/moments/varianceMeanItransformer_block_11/layer_normalization_23/moments/SquaredDifference:z:0Wtransformer_block_11/layer_normalization_23/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2>
<transformer_block_11/layer_normalization_23/moments/variance¿
;transformer_block_11/layer_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752=
;transformer_block_11/layer_normalization_23/batchnorm/add/yÂ
9transformer_block_11/layer_normalization_23/batchnorm/addAddV2Etransformer_block_11/layer_normalization_23/moments/variance:output:0Dtransformer_block_11/layer_normalization_23/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2;
9transformer_block_11/layer_normalization_23/batchnorm/addø
;transformer_block_11/layer_normalization_23/batchnorm/RsqrtRsqrt=transformer_block_11/layer_normalization_23/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2=
;transformer_block_11/layer_normalization_23/batchnorm/Rsqrt¢
Htransformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOpQtransformer_block_11_layer_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02J
Htransformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOpÆ
9transformer_block_11/layer_normalization_23/batchnorm/mulMul?transformer_block_11/layer_normalization_23/batchnorm/Rsqrt:y:0Ptransformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_11/layer_normalization_23/batchnorm/mul
;transformer_block_11/layer_normalization_23/batchnorm/mul_1Multransformer_block_11/add_1:z:0=transformer_block_11/layer_normalization_23/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2=
;transformer_block_11/layer_normalization_23/batchnorm/mul_1¹
;transformer_block_11/layer_normalization_23/batchnorm/mul_2MulAtransformer_block_11/layer_normalization_23/moments/mean:output:0=transformer_block_11/layer_normalization_23/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2=
;transformer_block_11/layer_normalization_23/batchnorm/mul_2
Dtransformer_block_11/layer_normalization_23/batchnorm/ReadVariableOpReadVariableOpMtransformer_block_11_layer_normalization_23_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02F
Dtransformer_block_11/layer_normalization_23/batchnorm/ReadVariableOpÂ
9transformer_block_11/layer_normalization_23/batchnorm/subSubLtransformer_block_11/layer_normalization_23/batchnorm/ReadVariableOp:value:0?transformer_block_11/layer_normalization_23/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_11/layer_normalization_23/batchnorm/sub¹
;transformer_block_11/layer_normalization_23/batchnorm/add_1AddV2?transformer_block_11/layer_normalization_23/batchnorm/mul_1:z:0=transformer_block_11/layer_normalization_23/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2=
;transformer_block_11/layer_normalization_23/batchnorm/add_1s
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`  2
flatten_5/Const¿
flatten_5/ReshapeReshape?transformer_block_11/layer_normalization_23/batchnorm/add_1:z:0flatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
flatten_5/Reshapex
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_5/concat/axis¾
concatenate_5/concatConcatV2flatten_5/Reshape:output:0inputs_1"concatenate_5/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
concatenate_5/concat©
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	è@*
dtype02 
dense_39/MatMul/ReadVariableOp¥
dense_39/MatMulMatMulconcatenate_5/concat:output:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_39/MatMul§
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_39/BiasAdd/ReadVariableOp¥
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_39/BiasAdds
dense_39/ReluReludense_39/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_39/Reluy
dropout_34/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *d!?2
dropout_34/dropout/Const©
dropout_34/dropout/MulMuldense_39/Relu:activations:0!dropout_34/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_34/dropout/Mul
dropout_34/dropout/ShapeShapedense_39/Relu:activations:0*
T0*
_output_shapes
:2
dropout_34/dropout/ShapeÕ
/dropout_34/dropout/random_uniform/RandomUniformRandomUniform!dropout_34/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype021
/dropout_34/dropout/random_uniform/RandomUniform
!dropout_34/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
×£=2#
!dropout_34/dropout/GreaterEqual/yê
dropout_34/dropout/GreaterEqualGreaterEqual8dropout_34/dropout/random_uniform/RandomUniform:output:0*dropout_34/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
dropout_34/dropout/GreaterEqual 
dropout_34/dropout/CastCast#dropout_34/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_34/dropout/Cast¦
dropout_34/dropout/Mul_1Muldropout_34/dropout/Mul:z:0dropout_34/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_34/dropout/Mul_1¨
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_40/MatMul/ReadVariableOp¤
dense_40/MatMulMatMuldropout_34/dropout/Mul_1:z:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_40/MatMul§
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_40/BiasAdd/ReadVariableOp¥
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_40/BiasAdds
dense_40/ReluReludense_40/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_40/Reluy
dropout_35/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *d!?2
dropout_35/dropout/Const©
dropout_35/dropout/MulMuldense_40/Relu:activations:0!dropout_35/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_35/dropout/Mul
dropout_35/dropout/ShapeShapedense_40/Relu:activations:0*
T0*
_output_shapes
:2
dropout_35/dropout/ShapeÕ
/dropout_35/dropout/random_uniform/RandomUniformRandomUniform!dropout_35/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype021
/dropout_35/dropout/random_uniform/RandomUniform
!dropout_35/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
×£=2#
!dropout_35/dropout/GreaterEqual/yê
dropout_35/dropout/GreaterEqualGreaterEqual8dropout_35/dropout/random_uniform/RandomUniform:output:0*dropout_35/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
dropout_35/dropout/GreaterEqual 
dropout_35/dropout/CastCast#dropout_35/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_35/dropout/Cast¦
dropout_35/dropout/Mul_1Muldropout_35/dropout/Mul:z:0dropout_35/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_35/dropout/Mul_1¨
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_41/MatMul/ReadVariableOp¤
dense_41/MatMulMatMuldropout_35/dropout/Mul_1:z:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_41/MatMul§
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_41/BiasAdd/ReadVariableOp¥
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_41/BiasAddº
IdentityIdentitydense_41/BiasAdd:output:0;^batch_normalization_10/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_10/AssignMovingAvg/ReadVariableOp=^batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_10/batchnorm/ReadVariableOp4^batch_normalization_10/batchnorm/mul/ReadVariableOp;^batch_normalization_11/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_11/AssignMovingAvg/ReadVariableOp=^batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_11/batchnorm/ReadVariableOp4^batch_normalization_11/batchnorm/mul/ReadVariableOp!^conv1d_10/BiasAdd/ReadVariableOp-^conv1d_10/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_11/BiasAdd/ReadVariableOp-^conv1d_11/conv1d/ExpandDims_1/ReadVariableOp ^dense_39/BiasAdd/ReadVariableOp^dense_39/MatMul/ReadVariableOp ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp=^token_and_position_embedding_5/embedding_10/embedding_lookup=^token_and_position_embedding_5/embedding_11/embedding_lookupE^transformer_block_11/layer_normalization_22/batchnorm/ReadVariableOpI^transformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOpE^transformer_block_11/layer_normalization_23/batchnorm/ReadVariableOpI^transformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOpQ^transformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOp[^transformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpD^transformer_block_11/multi_head_attention_11/key/add/ReadVariableOpN^transformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpF^transformer_block_11/multi_head_attention_11/query/add/ReadVariableOpP^transformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOpF^transformer_block_11/multi_head_attention_11/value/add/ReadVariableOpP^transformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOpC^transformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOpE^transformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOpC^transformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOpE^transformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2x
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_10/AssignMovingAvg/ReadVariableOp5batch_normalization_10/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_10/batchnorm/ReadVariableOp/batch_normalization_10/batchnorm/ReadVariableOp2j
3batch_normalization_10/batchnorm/mul/ReadVariableOp3batch_normalization_10/batchnorm/mul/ReadVariableOp2x
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_11/AssignMovingAvg/ReadVariableOp5batch_normalization_11/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_11/batchnorm/ReadVariableOp/batch_normalization_11/batchnorm/ReadVariableOp2j
3batch_normalization_11/batchnorm/mul/ReadVariableOp3batch_normalization_11/batchnorm/mul/ReadVariableOp2D
 conv1d_10/BiasAdd/ReadVariableOp conv1d_10/BiasAdd/ReadVariableOp2\
,conv1d_10/conv1d/ExpandDims_1/ReadVariableOp,conv1d_10/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_11/BiasAdd/ReadVariableOp conv1d_11/BiasAdd/ReadVariableOp2\
,conv1d_11/conv1d/ExpandDims_1/ReadVariableOp,conv1d_11/conv1d/ExpandDims_1/ReadVariableOp2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2|
<token_and_position_embedding_5/embedding_10/embedding_lookup<token_and_position_embedding_5/embedding_10/embedding_lookup2|
<token_and_position_embedding_5/embedding_11/embedding_lookup<token_and_position_embedding_5/embedding_11/embedding_lookup2
Dtransformer_block_11/layer_normalization_22/batchnorm/ReadVariableOpDtransformer_block_11/layer_normalization_22/batchnorm/ReadVariableOp2
Htransformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOpHtransformer_block_11/layer_normalization_22/batchnorm/mul/ReadVariableOp2
Dtransformer_block_11/layer_normalization_23/batchnorm/ReadVariableOpDtransformer_block_11/layer_normalization_23/batchnorm/ReadVariableOp2
Htransformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOpHtransformer_block_11/layer_normalization_23/batchnorm/mul/ReadVariableOp2¤
Ptransformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOpPtransformer_block_11/multi_head_attention_11/attention_output/add/ReadVariableOp2¸
Ztransformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOpZtransformer_block_11/multi_head_attention_11/attention_output/einsum/Einsum/ReadVariableOp2
Ctransformer_block_11/multi_head_attention_11/key/add/ReadVariableOpCtransformer_block_11/multi_head_attention_11/key/add/ReadVariableOp2
Mtransformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOpMtransformer_block_11/multi_head_attention_11/key/einsum/Einsum/ReadVariableOp2
Etransformer_block_11/multi_head_attention_11/query/add/ReadVariableOpEtransformer_block_11/multi_head_attention_11/query/add/ReadVariableOp2¢
Otransformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOpOtransformer_block_11/multi_head_attention_11/query/einsum/Einsum/ReadVariableOp2
Etransformer_block_11/multi_head_attention_11/value/add/ReadVariableOpEtransformer_block_11/multi_head_attention_11/value/add/ReadVariableOp2¢
Otransformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOpOtransformer_block_11/multi_head_attention_11/value/einsum/Einsum/ReadVariableOp2
Btransformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOpBtransformer_block_11/sequential_11/dense_37/BiasAdd/ReadVariableOp2
Dtransformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOpDtransformer_block_11/sequential_11/dense_37/Tensordot/ReadVariableOp2
Btransformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOpBtransformer_block_11/sequential_11/dense_38/BiasAdd/ReadVariableOp2
Dtransformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOpDtransformer_block_11/sequential_11/dense_38/Tensordot/ReadVariableOp:R N
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
Ì
ª
7__inference_batch_normalization_10_layer_call_fn_766388

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall£
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
GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_7642102
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

e
F__inference_dropout_35_layer_call_and_return_conditional_losses_764862

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *d!?2
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
×£=2
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
Ò
©
.__inference_sequential_11_layer_call_fn_764043
dense_37_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCalldense_37_inputunknown	unknown_0	unknown_1	unknown_2*
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
GPU2*0J 8 *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_7640322
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
_user_specified_namedense_37_input
ï
~
)__inference_dense_38_layer_call_fn_767269

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
D__inference_dense_38_layer_call_and_return_conditional_losses_7639572
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


R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_766280

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
µ
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_764742

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


Z__inference_token_and_position_embedding_5_layer_call_and_return_conditional_losses_766165
x(
$embedding_11_embedding_lookup_766152(
$embedding_10_embedding_lookup_766158
identity¢embedding_10/embedding_lookup¢embedding_11/embedding_lookup?
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
range³
embedding_11/embedding_lookupResourceGather$embedding_11_embedding_lookup_766152range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_11/embedding_lookup/766152*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02
embedding_11/embedding_lookup
&embedding_11/embedding_lookup/IdentityIdentity&embedding_11/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_11/embedding_lookup/766152*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&embedding_11/embedding_lookup/IdentityÃ
(embedding_11/embedding_lookup/Identity_1Identity/embedding_11/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(embedding_11/embedding_lookup/Identity_1s
embedding_10/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2
embedding_10/Cast¿
embedding_10/embedding_lookupResourceGather$embedding_10_embedding_lookup_766158embedding_10/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_10/embedding_lookup/766158*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02
embedding_10/embedding_lookup¢
&embedding_10/embedding_lookup/IdentityIdentity&embedding_10/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_10/embedding_lookup/766158*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2(
&embedding_10/embedding_lookup/IdentityÈ
(embedding_10/embedding_lookup/Identity_1Identity/embedding_10/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2*
(embedding_10/embedding_lookup/Identity_1°
addAddV21embedding_10/embedding_lookup/Identity_1:output:01embedding_11/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
add 
IdentityIdentityadd:z:0^embedding_10/embedding_lookup^embedding_11/embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿR::2>
embedding_10/embedding_lookupembedding_10/embedding_lookup2>
embedding_11/embedding_lookupembedding_11/embedding_lookup:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR

_user_specified_namex
Ì
ª
7__inference_batch_normalization_11_layer_call_fn_766470

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall£
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
GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_7643012
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_766919

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
Ñ
ã
D__inference_dense_38_layer_call_and_return_conditional_losses_767260

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
¥
d
+__inference_dropout_34_layer_call_fn_766979

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
F__inference_dropout_34_layer_call_and_return_conditional_losses_7648052
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
ø
l
P__inference_average_pooling1d_17_layer_call_and_return_conditional_losses_763590

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
Ñ
ã
D__inference_dense_38_layer_call_and_return_conditional_losses_763957

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


I__inference_sequential_11_layer_call_and_return_conditional_losses_763974
dense_37_input
dense_37_763922
dense_37_763924
dense_38_763968
dense_38_763970
identity¢ dense_37/StatefulPartitionedCall¢ dense_38/StatefulPartitionedCall£
 dense_37/StatefulPartitionedCallStatefulPartitionedCalldense_37_inputdense_37_763922dense_37_763924*
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
D__inference_dense_37_layer_call_and_return_conditional_losses_7639112"
 dense_37/StatefulPartitionedCall¾
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_763968dense_38_763970*
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
D__inference_dense_38_layer_call_and_return_conditional_losses_7639572"
 dense_38/StatefulPartitionedCallÇ
IdentityIdentity)dense_38/StatefulPartitionedCall:output:0!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
(
_user_specified_namedense_37_input
É
d
F__inference_dropout_35_layer_call_and_return_conditional_losses_767021

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
J
°
I__inference_sequential_11_layer_call_and_return_conditional_losses_767107

inputs.
*dense_37_tensordot_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource.
*dense_38_tensordot_readvariableop_resource,
(dense_38_biasadd_readvariableop_resource
identity¢dense_37/BiasAdd/ReadVariableOp¢!dense_37/Tensordot/ReadVariableOp¢dense_38/BiasAdd/ReadVariableOp¢!dense_38/Tensordot/ReadVariableOp±
!dense_37/Tensordot/ReadVariableOpReadVariableOp*dense_37_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02#
!dense_37/Tensordot/ReadVariableOp|
dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_37/Tensordot/axes
dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_37/Tensordot/freej
dense_37/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_37/Tensordot/Shape
 dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_37/Tensordot/GatherV2/axisþ
dense_37/Tensordot/GatherV2GatherV2!dense_37/Tensordot/Shape:output:0 dense_37/Tensordot/free:output:0)dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_37/Tensordot/GatherV2
"dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_37/Tensordot/GatherV2_1/axis
dense_37/Tensordot/GatherV2_1GatherV2!dense_37/Tensordot/Shape:output:0 dense_37/Tensordot/axes:output:0+dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_37/Tensordot/GatherV2_1~
dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_37/Tensordot/Const¤
dense_37/Tensordot/ProdProd$dense_37/Tensordot/GatherV2:output:0!dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_37/Tensordot/Prod
dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_37/Tensordot/Const_1¬
dense_37/Tensordot/Prod_1Prod&dense_37/Tensordot/GatherV2_1:output:0#dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/Tensordot/Prod_1
dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_37/Tensordot/concat/axisÝ
dense_37/Tensordot/concatConcatV2 dense_37/Tensordot/free:output:0 dense_37/Tensordot/axes:output:0'dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_37/Tensordot/concat°
dense_37/Tensordot/stackPack dense_37/Tensordot/Prod:output:0"dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_37/Tensordot/stack«
dense_37/Tensordot/transpose	Transposeinputs"dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_37/Tensordot/transposeÃ
dense_37/Tensordot/ReshapeReshape dense_37/Tensordot/transpose:y:0!dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_37/Tensordot/ReshapeÂ
dense_37/Tensordot/MatMulMatMul#dense_37/Tensordot/Reshape:output:0)dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_37/Tensordot/MatMul
dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_37/Tensordot/Const_2
 dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_37/Tensordot/concat_1/axisê
dense_37/Tensordot/concat_1ConcatV2$dense_37/Tensordot/GatherV2:output:0#dense_37/Tensordot/Const_2:output:0)dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_37/Tensordot/concat_1´
dense_37/TensordotReshape#dense_37/Tensordot/MatMul:product:0$dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_37/Tensordot§
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_37/BiasAdd/ReadVariableOp«
dense_37/BiasAddBiasAdddense_37/Tensordot:output:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_37/BiasAddw
dense_37/ReluReludense_37/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_37/Relu±
!dense_38/Tensordot/ReadVariableOpReadVariableOp*dense_38_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02#
!dense_38/Tensordot/ReadVariableOp|
dense_38/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_38/Tensordot/axes
dense_38/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_38/Tensordot/free
dense_38/Tensordot/ShapeShapedense_37/Relu:activations:0*
T0*
_output_shapes
:2
dense_38/Tensordot/Shape
 dense_38/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_38/Tensordot/GatherV2/axisþ
dense_38/Tensordot/GatherV2GatherV2!dense_38/Tensordot/Shape:output:0 dense_38/Tensordot/free:output:0)dense_38/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_38/Tensordot/GatherV2
"dense_38/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_38/Tensordot/GatherV2_1/axis
dense_38/Tensordot/GatherV2_1GatherV2!dense_38/Tensordot/Shape:output:0 dense_38/Tensordot/axes:output:0+dense_38/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_38/Tensordot/GatherV2_1~
dense_38/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_38/Tensordot/Const¤
dense_38/Tensordot/ProdProd$dense_38/Tensordot/GatherV2:output:0!dense_38/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_38/Tensordot/Prod
dense_38/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_38/Tensordot/Const_1¬
dense_38/Tensordot/Prod_1Prod&dense_38/Tensordot/GatherV2_1:output:0#dense_38/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_38/Tensordot/Prod_1
dense_38/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_38/Tensordot/concat/axisÝ
dense_38/Tensordot/concatConcatV2 dense_38/Tensordot/free:output:0 dense_38/Tensordot/axes:output:0'dense_38/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_38/Tensordot/concat°
dense_38/Tensordot/stackPack dense_38/Tensordot/Prod:output:0"dense_38/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_38/Tensordot/stackÀ
dense_38/Tensordot/transpose	Transposedense_37/Relu:activations:0"dense_38/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_38/Tensordot/transposeÃ
dense_38/Tensordot/ReshapeReshape dense_38/Tensordot/transpose:y:0!dense_38/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_38/Tensordot/ReshapeÂ
dense_38/Tensordot/MatMulMatMul#dense_38/Tensordot/Reshape:output:0)dense_38/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_38/Tensordot/MatMul
dense_38/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_38/Tensordot/Const_2
 dense_38/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_38/Tensordot/concat_1/axisê
dense_38/Tensordot/concat_1ConcatV2$dense_38/Tensordot/GatherV2:output:0#dense_38/Tensordot/Const_2:output:0)dense_38/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_38/Tensordot/concat_1´
dense_38/TensordotReshape#dense_38/Tensordot/MatMul:product:0$dense_38/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_38/Tensordot§
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_38/BiasAdd/ReadVariableOp«
dense_38/BiasAddBiasAdddense_38/Tensordot:output:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_38/BiasAddý
IdentityIdentitydense_38/BiasAdd:output:0 ^dense_37/BiasAdd/ReadVariableOp"^dense_37/Tensordot/ReadVariableOp ^dense_38/BiasAdd/ReadVariableOp"^dense_38/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2F
!dense_37/Tensordot/ReadVariableOp!dense_37/Tensordot/ReadVariableOp2B
dense_38/BiasAdd/ReadVariableOpdense_38/BiasAdd/ReadVariableOp2F
!dense_38/Tensordot/ReadVariableOp!dense_38/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
Ú
¤
(__inference_model_5_layer_call_fn_766141
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
C__inference_model_5_layer_call_and_return_conditional_losses_7652712
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
É
d
F__inference_dropout_34_layer_call_and_return_conditional_losses_766974

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

Q
5__inference_average_pooling1d_16_layer_call_fn_763581

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
P__inference_average_pooling1d_16_layer_call_and_return_conditional_losses_7635752
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
D__inference_dense_41_layer_call_and_return_conditional_losses_767041

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
F__inference_dropout_34_layer_call_and_return_conditional_losses_766969

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *d!?2
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
×£=2
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
ê

I__inference_sequential_11_layer_call_and_return_conditional_losses_764005

inputs
dense_37_763994
dense_37_763996
dense_38_763999
dense_38_764001
identity¢ dense_37/StatefulPartitionedCall¢ dense_38/StatefulPartitionedCall
 dense_37/StatefulPartitionedCallStatefulPartitionedCallinputsdense_37_763994dense_37_763996*
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
D__inference_dense_37_layer_call_and_return_conditional_losses_7639112"
 dense_37/StatefulPartitionedCall¾
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_763999dense_38_764001*
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
D__inference_dense_38_layer_call_and_return_conditional_losses_7639572"
 dense_38/StatefulPartitionedCallÇ
IdentityIdentity)dense_38/StatefulPartitionedCall:output:0!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs


R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_763725

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


I__inference_sequential_11_layer_call_and_return_conditional_losses_763988
dense_37_input
dense_37_763977
dense_37_763979
dense_38_763982
dense_38_763984
identity¢ dense_37/StatefulPartitionedCall¢ dense_38/StatefulPartitionedCall£
 dense_37/StatefulPartitionedCallStatefulPartitionedCalldense_37_inputdense_37_763977dense_37_763979*
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
D__inference_dense_37_layer_call_and_return_conditional_losses_7639112"
 dense_37/StatefulPartitionedCall¾
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_763982dense_38_763984*
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
D__inference_dense_38_layer_call_and_return_conditional_losses_7639572"
 dense_38/StatefulPartitionedCallÇ
IdentityIdentity)dense_38/StatefulPartitionedCall:output:0!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
(
_user_specified_namedense_37_input


R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_763865

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
Ò
©
.__inference_sequential_11_layer_call_fn_764016
dense_37_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCalldense_37_inputunknown	unknown_0	unknown_1	unknown_2*
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
GPU2*0J 8 *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_7640052
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
_user_specified_namedense_37_input
ñ	
Ý
D__inference_dense_39_layer_call_and_return_conditional_losses_764777

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
î
ª
7__inference_batch_normalization_11_layer_call_fn_766539

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallª
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
GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_7638322
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
é

R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_766444

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
+__inference_dropout_34_layer_call_fn_766984

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
F__inference_dropout_34_layer_call_and_return_conditional_losses_7648102
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
¿
m
A__inference_add_5_layer_call_and_return_conditional_losses_766558
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
ô0
É
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_763832

inputs
assignmovingavg_763807
assignmovingavg_1_763813)
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
loc:@AssignMovingAvg/763807*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_763807*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/763807*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/763807*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_763807AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/763807*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/763813*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_763813*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/763813*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/763813*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_763813AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/763813*
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
ÚÇ
Ù1
"__inference__traced_restore_767747
file_prefix%
!assignvariableop_conv1d_10_kernel%
!assignvariableop_1_conv1d_10_bias'
#assignvariableop_2_conv1d_11_kernel%
!assignvariableop_3_conv1d_11_bias3
/assignvariableop_4_batch_normalization_10_gamma2
.assignvariableop_5_batch_normalization_10_beta9
5assignvariableop_6_batch_normalization_10_moving_mean=
9assignvariableop_7_batch_normalization_10_moving_variance3
/assignvariableop_8_batch_normalization_11_gamma2
.assignvariableop_9_batch_normalization_11_beta:
6assignvariableop_10_batch_normalization_11_moving_mean>
:assignvariableop_11_batch_normalization_11_moving_variance'
#assignvariableop_12_dense_39_kernel%
!assignvariableop_13_dense_39_bias'
#assignvariableop_14_dense_40_kernel%
!assignvariableop_15_dense_40_bias'
#assignvariableop_16_dense_41_kernel%
!assignvariableop_17_dense_41_bias
assignvariableop_18_decay%
!assignvariableop_19_learning_rate 
assignvariableop_20_momentum 
assignvariableop_21_sgd_iterN
Jassignvariableop_22_token_and_position_embedding_5_embedding_10_embeddingsN
Jassignvariableop_23_token_and_position_embedding_5_embedding_11_embeddingsQ
Massignvariableop_24_transformer_block_11_multi_head_attention_11_query_kernelO
Kassignvariableop_25_transformer_block_11_multi_head_attention_11_query_biasO
Kassignvariableop_26_transformer_block_11_multi_head_attention_11_key_kernelM
Iassignvariableop_27_transformer_block_11_multi_head_attention_11_key_biasQ
Massignvariableop_28_transformer_block_11_multi_head_attention_11_value_kernelO
Kassignvariableop_29_transformer_block_11_multi_head_attention_11_value_bias\
Xassignvariableop_30_transformer_block_11_multi_head_attention_11_attention_output_kernelZ
Vassignvariableop_31_transformer_block_11_multi_head_attention_11_attention_output_bias'
#assignvariableop_32_dense_37_kernel%
!assignvariableop_33_dense_37_bias'
#assignvariableop_34_dense_38_kernel%
!assignvariableop_35_dense_38_biasI
Eassignvariableop_36_transformer_block_11_layer_normalization_22_gammaH
Dassignvariableop_37_transformer_block_11_layer_normalization_22_betaI
Eassignvariableop_38_transformer_block_11_layer_normalization_23_gammaH
Dassignvariableop_39_transformer_block_11_layer_normalization_23_beta
assignvariableop_40_total
assignvariableop_41_count5
1assignvariableop_42_sgd_conv1d_10_kernel_momentum3
/assignvariableop_43_sgd_conv1d_10_bias_momentum5
1assignvariableop_44_sgd_conv1d_11_kernel_momentum3
/assignvariableop_45_sgd_conv1d_11_bias_momentumA
=assignvariableop_46_sgd_batch_normalization_10_gamma_momentum@
<assignvariableop_47_sgd_batch_normalization_10_beta_momentumA
=assignvariableop_48_sgd_batch_normalization_11_gamma_momentum@
<assignvariableop_49_sgd_batch_normalization_11_beta_momentum4
0assignvariableop_50_sgd_dense_39_kernel_momentum2
.assignvariableop_51_sgd_dense_39_bias_momentum4
0assignvariableop_52_sgd_dense_40_kernel_momentum2
.assignvariableop_53_sgd_dense_40_bias_momentum4
0assignvariableop_54_sgd_dense_41_kernel_momentum2
.assignvariableop_55_sgd_dense_41_bias_momentum[
Wassignvariableop_56_sgd_token_and_position_embedding_5_embedding_10_embeddings_momentum[
Wassignvariableop_57_sgd_token_and_position_embedding_5_embedding_11_embeddings_momentum^
Zassignvariableop_58_sgd_transformer_block_11_multi_head_attention_11_query_kernel_momentum\
Xassignvariableop_59_sgd_transformer_block_11_multi_head_attention_11_query_bias_momentum\
Xassignvariableop_60_sgd_transformer_block_11_multi_head_attention_11_key_kernel_momentumZ
Vassignvariableop_61_sgd_transformer_block_11_multi_head_attention_11_key_bias_momentum^
Zassignvariableop_62_sgd_transformer_block_11_multi_head_attention_11_value_kernel_momentum\
Xassignvariableop_63_sgd_transformer_block_11_multi_head_attention_11_value_bias_momentumi
eassignvariableop_64_sgd_transformer_block_11_multi_head_attention_11_attention_output_kernel_momentumg
cassignvariableop_65_sgd_transformer_block_11_multi_head_attention_11_attention_output_bias_momentum4
0assignvariableop_66_sgd_dense_37_kernel_momentum2
.assignvariableop_67_sgd_dense_37_bias_momentum4
0assignvariableop_68_sgd_dense_38_kernel_momentum2
.assignvariableop_69_sgd_dense_38_bias_momentumV
Rassignvariableop_70_sgd_transformer_block_11_layer_normalization_22_gamma_momentumU
Qassignvariableop_71_sgd_transformer_block_11_layer_normalization_22_beta_momentumV
Rassignvariableop_72_sgd_transformer_block_11_layer_normalization_23_gamma_momentumU
Qassignvariableop_73_sgd_transformer_block_11_layer_normalization_23_beta_momentum
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

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_conv1d_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv1d_10_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv1d_11_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv1d_11_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4´
AssignVariableOp_4AssignVariableOp/assignvariableop_4_batch_normalization_10_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5³
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batch_normalization_10_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6º
AssignVariableOp_6AssignVariableOp5assignvariableop_6_batch_normalization_10_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¾
AssignVariableOp_7AssignVariableOp9assignvariableop_7_batch_normalization_10_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8´
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_11_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9³
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_11_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¾
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_11_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Â
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_11_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_39_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_39_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_40_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_40_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16«
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_41_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17©
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_41_biasIdentity_17:output:0"/device:CPU:0*
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
Identity_22Ò
AssignVariableOp_22AssignVariableOpJassignvariableop_22_token_and_position_embedding_5_embedding_10_embeddingsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ò
AssignVariableOp_23AssignVariableOpJassignvariableop_23_token_and_position_embedding_5_embedding_11_embeddingsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Õ
AssignVariableOp_24AssignVariableOpMassignvariableop_24_transformer_block_11_multi_head_attention_11_query_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ó
AssignVariableOp_25AssignVariableOpKassignvariableop_25_transformer_block_11_multi_head_attention_11_query_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ó
AssignVariableOp_26AssignVariableOpKassignvariableop_26_transformer_block_11_multi_head_attention_11_key_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ñ
AssignVariableOp_27AssignVariableOpIassignvariableop_27_transformer_block_11_multi_head_attention_11_key_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Õ
AssignVariableOp_28AssignVariableOpMassignvariableop_28_transformer_block_11_multi_head_attention_11_value_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ó
AssignVariableOp_29AssignVariableOpKassignvariableop_29_transformer_block_11_multi_head_attention_11_value_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30à
AssignVariableOp_30AssignVariableOpXassignvariableop_30_transformer_block_11_multi_head_attention_11_attention_output_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Þ
AssignVariableOp_31AssignVariableOpVassignvariableop_31_transformer_block_11_multi_head_attention_11_attention_output_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32«
AssignVariableOp_32AssignVariableOp#assignvariableop_32_dense_37_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33©
AssignVariableOp_33AssignVariableOp!assignvariableop_33_dense_37_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34«
AssignVariableOp_34AssignVariableOp#assignvariableop_34_dense_38_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35©
AssignVariableOp_35AssignVariableOp!assignvariableop_35_dense_38_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Í
AssignVariableOp_36AssignVariableOpEassignvariableop_36_transformer_block_11_layer_normalization_22_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ì
AssignVariableOp_37AssignVariableOpDassignvariableop_37_transformer_block_11_layer_normalization_22_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Í
AssignVariableOp_38AssignVariableOpEassignvariableop_38_transformer_block_11_layer_normalization_23_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Ì
AssignVariableOp_39AssignVariableOpDassignvariableop_39_transformer_block_11_layer_normalization_23_betaIdentity_39:output:0"/device:CPU:0*
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
Identity_42¹
AssignVariableOp_42AssignVariableOp1assignvariableop_42_sgd_conv1d_10_kernel_momentumIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43·
AssignVariableOp_43AssignVariableOp/assignvariableop_43_sgd_conv1d_10_bias_momentumIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¹
AssignVariableOp_44AssignVariableOp1assignvariableop_44_sgd_conv1d_11_kernel_momentumIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45·
AssignVariableOp_45AssignVariableOp/assignvariableop_45_sgd_conv1d_11_bias_momentumIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Å
AssignVariableOp_46AssignVariableOp=assignvariableop_46_sgd_batch_normalization_10_gamma_momentumIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Ä
AssignVariableOp_47AssignVariableOp<assignvariableop_47_sgd_batch_normalization_10_beta_momentumIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Å
AssignVariableOp_48AssignVariableOp=assignvariableop_48_sgd_batch_normalization_11_gamma_momentumIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Ä
AssignVariableOp_49AssignVariableOp<assignvariableop_49_sgd_batch_normalization_11_beta_momentumIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50¸
AssignVariableOp_50AssignVariableOp0assignvariableop_50_sgd_dense_39_kernel_momentumIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51¶
AssignVariableOp_51AssignVariableOp.assignvariableop_51_sgd_dense_39_bias_momentumIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52¸
AssignVariableOp_52AssignVariableOp0assignvariableop_52_sgd_dense_40_kernel_momentumIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53¶
AssignVariableOp_53AssignVariableOp.assignvariableop_53_sgd_dense_40_bias_momentumIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54¸
AssignVariableOp_54AssignVariableOp0assignvariableop_54_sgd_dense_41_kernel_momentumIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55¶
AssignVariableOp_55AssignVariableOp.assignvariableop_55_sgd_dense_41_bias_momentumIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56ß
AssignVariableOp_56AssignVariableOpWassignvariableop_56_sgd_token_and_position_embedding_5_embedding_10_embeddings_momentumIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57ß
AssignVariableOp_57AssignVariableOpWassignvariableop_57_sgd_token_and_position_embedding_5_embedding_11_embeddings_momentumIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58â
AssignVariableOp_58AssignVariableOpZassignvariableop_58_sgd_transformer_block_11_multi_head_attention_11_query_kernel_momentumIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59à
AssignVariableOp_59AssignVariableOpXassignvariableop_59_sgd_transformer_block_11_multi_head_attention_11_query_bias_momentumIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60à
AssignVariableOp_60AssignVariableOpXassignvariableop_60_sgd_transformer_block_11_multi_head_attention_11_key_kernel_momentumIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Þ
AssignVariableOp_61AssignVariableOpVassignvariableop_61_sgd_transformer_block_11_multi_head_attention_11_key_bias_momentumIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62â
AssignVariableOp_62AssignVariableOpZassignvariableop_62_sgd_transformer_block_11_multi_head_attention_11_value_kernel_momentumIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63à
AssignVariableOp_63AssignVariableOpXassignvariableop_63_sgd_transformer_block_11_multi_head_attention_11_value_bias_momentumIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64í
AssignVariableOp_64AssignVariableOpeassignvariableop_64_sgd_transformer_block_11_multi_head_attention_11_attention_output_kernel_momentumIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65ë
AssignVariableOp_65AssignVariableOpcassignvariableop_65_sgd_transformer_block_11_multi_head_attention_11_attention_output_bias_momentumIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66¸
AssignVariableOp_66AssignVariableOp0assignvariableop_66_sgd_dense_37_kernel_momentumIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67¶
AssignVariableOp_67AssignVariableOp.assignvariableop_67_sgd_dense_37_bias_momentumIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68¸
AssignVariableOp_68AssignVariableOp0assignvariableop_68_sgd_dense_38_kernel_momentumIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69¶
AssignVariableOp_69AssignVariableOp.assignvariableop_69_sgd_dense_38_bias_momentumIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Ú
AssignVariableOp_70AssignVariableOpRassignvariableop_70_sgd_transformer_block_11_layer_normalization_22_gamma_momentumIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71Ù
AssignVariableOp_71AssignVariableOpQassignvariableop_71_sgd_transformer_block_11_layer_normalization_22_beta_momentumIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72Ú
AssignVariableOp_72AssignVariableOpRassignvariableop_72_sgd_transformer_block_11_layer_normalization_23_gamma_momentumIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73Ù
AssignVariableOp_73AssignVariableOpQassignvariableop_73_sgd_transformer_block_11_layer_normalization_23_beta_momentumIdentity_73:output:0"/device:CPU:0*
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
½0
É
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_764281

inputs
assignmovingavg_764256
assignmovingavg_1_764262)
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
loc:@AssignMovingAvg/764256*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_764256*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/764256*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/764256*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_764256AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/764256*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/764262*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_764262*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/764262*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/764262*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_764262AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/764262*
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
õ

*__inference_conv1d_11_layer_call_fn_766224

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
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1d_11_layer_call_and_return_conditional_losses_7641372
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
Ò

á
5__inference_transformer_block_11_layer_call_fn_766913

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
identity¢StatefulPartitionedCallÂ
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
GPU2*0J 8 *Y
fTRR
P__inference_transformer_block_11_layer_call_and_return_conditional_losses_7646272
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

G
+__inference_dropout_35_layer_call_fn_767031

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
F__inference_dropout_35_layer_call_and_return_conditional_losses_7648672
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
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*í
serving_defaultÙ
>
input_112
serving_default_input_11:0ÿÿÿÿÿÿÿÿÿR
=
input_121
serving_default_input_12:0ÿÿÿÿÿÿÿÿÿ<
dense_410
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:þ
íG
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
regularization_losses
trainable_variables
	keras_api

signatures
+±&call_and_return_all_conditional_losses
²__call__
³_default_save_signature"µB
_tf_keras_networkB{"class_name": "Functional", "name": "model_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}, "name": "input_11", "inbound_nodes": []}, {"class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}, "name": "token_and_position_embedding_5", "inbound_nodes": [[["input_11", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_10", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_10", "inbound_nodes": [[["token_and_position_embedding_5", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_15", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_15", "inbound_nodes": [[["conv1d_10", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_11", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_11", "inbound_nodes": [[["average_pooling1d_15", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_16", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_16", "inbound_nodes": [[["conv1d_11", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_17", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_17", "inbound_nodes": [[["token_and_position_embedding_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_10", "inbound_nodes": [[["average_pooling1d_16", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_11", "inbound_nodes": [[["average_pooling1d_17", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "name": "add_5", "inbound_nodes": [[["batch_normalization_10", 0, 0, {}], ["batch_normalization_11", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block_11", "inbound_nodes": [[["add_5", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["transformer_block_11", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_12"}, "name": "input_12", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_5", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_5", "inbound_nodes": [[["flatten_5", 0, 0, {}], ["input_12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_39", "inbound_nodes": [[["concatenate_5", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_34", "trainable": true, "dtype": "float32", "rate": 0.08, "noise_shape": null, "seed": null}, "name": "dropout_34", "inbound_nodes": [[["dense_39", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_40", "inbound_nodes": [[["dropout_34", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_35", "trainable": true, "dtype": "float32", "rate": 0.08, "noise_shape": null, "seed": null}, "name": "dropout_35", "inbound_nodes": [[["dense_40", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_41", "inbound_nodes": [[["dropout_35", 0, 0, {}]]]}], "input_layers": [["input_11", 0, 0], ["input_12", 0, 0]], "output_layers": [["dense_41", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 10500]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 10500]}, {"class_name": "TensorShape", "items": [null, 8]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.0010000000474974513, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
ó"ð
_tf_keras_input_layerÐ{"class_name": "InputLayer", "name": "input_11", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}}
ç
	token_emb
pos_emb
	variables
regularization_losses
trainable_variables
	keras_api
+´&call_and_return_all_conditional_losses
µ__call__"º
_tf_keras_layer {"class_name": "TokenAndPositionEmbedding", "name": "token_and_position_embedding_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
ë	

 kernel
!bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
+¶&call_and_return_all_conditional_losses
·__call__"Ä
_tf_keras_layerª{"class_name": "Conv1D", "name": "conv1d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_10", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500, 32]}}

&	variables
'regularization_losses
(trainable_variables
)	keras_api
+¸&call_and_return_all_conditional_losses
¹__call__"ú
_tf_keras_layerà{"class_name": "AveragePooling1D", "name": "average_pooling1d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_15", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
é	

*kernel
+bias
,	variables
-regularization_losses
.trainable_variables
/	keras_api
+º&call_and_return_all_conditional_losses
»__call__"Â
_tf_keras_layer¨{"class_name": "Conv1D", "name": "conv1d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_11", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 350, 32]}}

0	variables
1regularization_losses
2trainable_variables
3	keras_api
+¼&call_and_return_all_conditional_losses
½__call__"ú
_tf_keras_layerà{"class_name": "AveragePooling1D", "name": "average_pooling1d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_16", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}

4	variables
5regularization_losses
6trainable_variables
7	keras_api
+¾&call_and_return_all_conditional_losses
¿__call__"ü
_tf_keras_layerâ{"class_name": "AveragePooling1D", "name": "average_pooling1d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_17", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
º	
8axis
	9gamma
:beta
;moving_mean
<moving_variance
=	variables
>regularization_losses
?trainable_variables
@	keras_api
+À&call_and_return_all_conditional_losses
Á__call__"ä
_tf_keras_layerÊ{"class_name": "BatchNormalization", "name": "batch_normalization_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
º	
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
+Â&call_and_return_all_conditional_losses
Ã__call__"ä
_tf_keras_layerÊ{"class_name": "BatchNormalization", "name": "batch_normalization_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
³
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
+Ä&call_and_return_all_conditional_losses
Å__call__"¢
_tf_keras_layer{"class_name": "Add", "name": "add_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 35, 32]}, {"class_name": "TensorShape", "items": [null, 35, 32]}]}

Natt
Offn
P
layernorm1
Q
layernorm2
Rdropout1
Sdropout2
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
+Æ&call_and_return_all_conditional_losses
Ç__call__"¦
_tf_keras_layer{"class_name": "TransformerBlock", "name": "transformer_block_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
è
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
+È&call_and_return_all_conditional_losses
É__call__"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ë"è
_tf_keras_input_layerÈ{"class_name": "InputLayer", "name": "input_12", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_12"}}
Ð
\	variables
]regularization_losses
^trainable_variables
_	keras_api
+Ê&call_and_return_all_conditional_losses
Ë__call__"¿
_tf_keras_layer¥{"class_name": "Concatenate", "name": "concatenate_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_5", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1120]}, {"class_name": "TensorShape", "items": [null, 8]}]}
ø

`kernel
abias
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
+Ì&call_and_return_all_conditional_losses
Í__call__"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "dense_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1128]}}
ê
f	variables
gregularization_losses
htrainable_variables
i	keras_api
+Î&call_and_return_all_conditional_losses
Ï__call__"Ù
_tf_keras_layer¿{"class_name": "Dropout", "name": "dropout_34", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_34", "trainable": true, "dtype": "float32", "rate": 0.08, "noise_shape": null, "seed": null}}
ô

jkernel
kbias
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
+Ð&call_and_return_all_conditional_losses
Ñ__call__"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ê
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
+Ò&call_and_return_all_conditional_losses
Ó__call__"Ù
_tf_keras_layer¿{"class_name": "Dropout", "name": "dropout_35", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_35", "trainable": true, "dtype": "float32", "rate": 0.08, "noise_shape": null, "seed": null}}
õ

tkernel
ubias
v	variables
wregularization_losses
xtrainable_variables
y	keras_api
+Ô&call_and_return_all_conditional_losses
Õ__call__"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
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
 "
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
Ó
layers
layer_metrics
	variables
non_trainable_variables
metrics
 layer_regularization_losses
regularization_losses
trainable_variables
²__call__
³_default_save_signature
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
-
Öserving_default"
signature_map
¶
~
embeddings
	variables
regularization_losses
trainable_variables
	keras_api
+×&call_and_return_all_conditional_losses
Ø__call__"
_tf_keras_layer÷{"class_name": "Embedding", "name": "embedding_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500]}}
³

embeddings
	variables
regularization_losses
trainable_variables
	keras_api
+Ù&call_and_return_all_conditional_losses
Ú__call__"
_tf_keras_layerô{"class_name": "Embedding", "name": "embedding_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_11", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 10500, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
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
layers
layer_metrics
non_trainable_variables
 metrics
	variables
 ¡layer_regularization_losses
regularization_losses
trainable_variables
µ__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
&:$  2conv1d_10/kernel
: 2conv1d_10/bias
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
¢layers
£layer_metrics
¤non_trainable_variables
¥metrics
"	variables
 ¦layer_regularization_losses
#regularization_losses
$trainable_variables
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
§layers
¨layer_metrics
©non_trainable_variables
ªmetrics
&	variables
 «layer_regularization_losses
'regularization_losses
(trainable_variables
¹__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
&:$	  2conv1d_11/kernel
: 2conv1d_11/bias
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
¬layers
­layer_metrics
®non_trainable_variables
¯metrics
,	variables
 °layer_regularization_losses
-regularization_losses
.trainable_variables
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
±layers
²layer_metrics
³non_trainable_variables
´metrics
0	variables
 µlayer_regularization_losses
1regularization_losses
2trainable_variables
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
¶layers
·layer_metrics
¸non_trainable_variables
¹metrics
4	variables
 ºlayer_regularization_losses
5regularization_losses
6trainable_variables
¿__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_10/gamma
):' 2batch_normalization_10/beta
2:0  (2"batch_normalization_10/moving_mean
6:4  (2&batch_normalization_10/moving_variance
<
90
:1
;2
<3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
µ
»layers
¼layer_metrics
½non_trainable_variables
¾metrics
=	variables
 ¿layer_regularization_losses
>regularization_losses
?trainable_variables
Á__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_11/gamma
):' 2batch_normalization_11/beta
2:0  (2"batch_normalization_11/moving_mean
6:4  (2&batch_normalization_11/moving_variance
<
B0
C1
D2
E3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
µ
Àlayers
Álayer_metrics
Ânon_trainable_variables
Ãmetrics
F	variables
 Älayer_regularization_losses
Gregularization_losses
Htrainable_variables
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
Ålayers
Ælayer_metrics
Çnon_trainable_variables
Èmetrics
J	variables
 Élayer_regularization_losses
Kregularization_losses
Ltrainable_variables
Å__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object

Ê_query_dense
Ë
_key_dense
Ì_value_dense
Í_softmax
Î_dropout_layer
Ï_output_dense
Ð	variables
Ñregularization_losses
Òtrainable_variables
Ó	keras_api
+Û&call_and_return_all_conditional_losses
Ü__call__"
_tf_keras_layerì{"class_name": "MultiHeadAttention", "name": "multi_head_attention_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multi_head_attention_11", "trainable": true, "dtype": "float32", "num_heads": 1, "key_dim": 32, "value_dim": 32, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}
²
Ôlayer_with_weights-0
Ôlayer-0
Õlayer_with_weights-1
Õlayer-1
Ö	variables
×regularization_losses
Øtrainable_variables
Ù	keras_api
+Ý&call_and_return_all_conditional_losses
Þ__call__"Ë
_tf_keras_sequential¬{"class_name": "Sequential", "name": "sequential_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_11", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_37_input"}}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_11", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_37_input"}}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
ì
	Úaxis

gamma
	beta
Û	variables
Üregularization_losses
Ýtrainable_variables
Þ	keras_api
+ß&call_and_return_all_conditional_losses
à__call__"µ
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_22", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
ì
	ßaxis

gamma
	beta
à	variables
áregularization_losses
âtrainable_variables
ã	keras_api
+á&call_and_return_all_conditional_losses
â__call__"µ
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_23", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
í
ä	variables
åregularization_losses
ætrainable_variables
ç	keras_api
+ã&call_and_return_all_conditional_losses
ä__call__"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_32", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_32", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
í
è	variables
éregularization_losses
êtrainable_variables
ë	keras_api
+å&call_and_return_all_conditional_losses
æ__call__"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_33", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_33", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
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
ìlayers
ílayer_metrics
înon_trainable_variables
ïmetrics
T	variables
 ðlayer_regularization_losses
Uregularization_losses
Vtrainable_variables
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
ñlayers
òlayer_metrics
ónon_trainable_variables
ômetrics
X	variables
 õlayer_regularization_losses
Yregularization_losses
Ztrainable_variables
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
ölayers
÷layer_metrics
ønon_trainable_variables
ùmetrics
\	variables
 úlayer_regularization_losses
]regularization_losses
^trainable_variables
Ë__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
": 	è@2dense_39/kernel
:@2dense_39/bias
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
ûlayers
ülayer_metrics
ýnon_trainable_variables
þmetrics
b	variables
 ÿlayer_regularization_losses
cregularization_losses
dtrainable_variables
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
layers
layer_metrics
non_trainable_variables
metrics
f	variables
 layer_regularization_losses
gregularization_losses
htrainable_variables
Ï__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_40/kernel
:@2dense_40/bias
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
layers
layer_metrics
non_trainable_variables
metrics
l	variables
 layer_regularization_losses
mregularization_losses
ntrainable_variables
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
layers
layer_metrics
non_trainable_variables
metrics
p	variables
 layer_regularization_losses
qregularization_losses
rtrainable_variables
Ó__call__
+Ò&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_41/kernel
:2dense_41/bias
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
layers
layer_metrics
non_trainable_variables
metrics
v	variables
 layer_regularization_losses
wregularization_losses
xtrainable_variables
Õ__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
H:F 26token_and_position_embedding_5/embedding_10/embeddings
I:G	R 26token_and_position_embedding_5/embedding_11/embeddings
O:M  29transformer_block_11/multi_head_attention_11/query/kernel
I:G 27transformer_block_11/multi_head_attention_11/query/bias
M:K  27transformer_block_11/multi_head_attention_11/key/kernel
G:E 25transformer_block_11/multi_head_attention_11/key/bias
O:M  29transformer_block_11/multi_head_attention_11/value/kernel
I:G 27transformer_block_11/multi_head_attention_11/value/bias
Z:X  2Dtransformer_block_11/multi_head_attention_11/attention_output/kernel
P:N 2Btransformer_block_11/multi_head_attention_11/attention_output/bias
!: @2dense_37/kernel
:@2dense_37/bias
!:@ 2dense_38/kernel
: 2dense_38/bias
?:= 21transformer_block_11/layer_normalization_22/gamma
>:< 20transformer_block_11/layer_normalization_22/beta
?:= 21transformer_block_11/layer_normalization_23/gamma
>:< 20transformer_block_11/layer_normalization_23/beta
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
<
;0
<1
D2
E3"
trackable_list_wrapper
(
0"
trackable_list_wrapper
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
layers
layer_metrics
non_trainable_variables
metrics
	variables
 layer_regularization_losses
regularization_losses
trainable_variables
Ø__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses"
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
layers
layer_metrics
non_trainable_variables
metrics
	variables
 layer_regularization_losses
regularization_losses
trainable_variables
Ú__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses"
_generic_user_object
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
Ë
partial_output_shape
 full_output_shape
kernel
	bias
¡	variables
¢regularization_losses
£trainable_variables
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
¨regularization_losses
©trainable_variables
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
®regularization_losses
¯trainable_variables
°	keras_api
+ë&call_and_return_all_conditional_losses
ì__call__"ë
_tf_keras_layerÑ{"class_name": "EinsumDense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
ë
±	variables
²regularization_losses
³trainable_variables
´	keras_api
+í&call_and_return_all_conditional_losses
î__call__"Ö
_tf_keras_layer¼{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}}
ç
µ	variables
¶regularization_losses
·trainable_variables
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
¼regularization_losses
½trainable_variables
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
¿layers
Àlayer_metrics
Ánon_trainable_variables
Âmetrics
Ð	variables
 Ãlayer_regularization_losses
Ñregularization_losses
Òtrainable_variables
Ü__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
þ
kernel
	bias
Ä	variables
Åregularization_losses
Ætrainable_variables
Ç	keras_api
+ó&call_and_return_all_conditional_losses
ô__call__"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "dense_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}

kernel
	bias
È	variables
Éregularization_losses
Êtrainable_variables
Ë	keras_api
+õ&call_and_return_all_conditional_losses
ö__call__"Ó
_tf_keras_layer¹{"class_name": "Dense", "name": "dense_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 64]}}
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
Ìlayers
Ílayer_metrics
Ö	variables
Înon_trainable_variables
Ïmetrics
 Ðlayer_regularization_losses
×regularization_losses
Øtrainable_variables
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
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
Ñlayers
Òlayer_metrics
Ónon_trainable_variables
Ômetrics
Û	variables
 Õlayer_regularization_losses
Üregularization_losses
Ýtrainable_variables
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
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
Ölayers
×layer_metrics
Ønon_trainable_variables
Ùmetrics
à	variables
 Úlayer_regularization_losses
áregularization_losses
âtrainable_variables
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
Ûlayers
Ülayer_metrics
Ýnon_trainable_variables
Þmetrics
ä	variables
 ßlayer_regularization_losses
åregularization_losses
ætrainable_variables
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
àlayers
álayer_metrics
ânon_trainable_variables
ãmetrics
è	variables
 älayer_regularization_losses
éregularization_losses
êtrainable_variables
æ__call__
+å&call_and_return_all_conditional_losses
'å"call_and_return_conditional_losses"
_generic_user_object
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
élayers
êlayer_metrics
ënon_trainable_variables
ìmetrics
¡	variables
 ílayer_regularization_losses
¢regularization_losses
£trainable_variables
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
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
îlayers
ïlayer_metrics
ðnon_trainable_variables
ñmetrics
§	variables
 òlayer_regularization_losses
¨regularization_losses
©trainable_variables
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
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
ólayers
ôlayer_metrics
õnon_trainable_variables
ömetrics
­	variables
 ÷layer_regularization_losses
®regularization_losses
¯trainable_variables
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
ølayers
ùlayer_metrics
únon_trainable_variables
ûmetrics
±	variables
 ülayer_regularization_losses
²regularization_losses
³trainable_variables
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
ýlayers
þlayer_metrics
ÿnon_trainable_variables
metrics
µ	variables
 layer_regularization_losses
¶regularization_losses
·trainable_variables
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
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
layers
layer_metrics
non_trainable_variables
metrics
»	variables
 layer_regularization_losses
¼regularization_losses
½trainable_variables
ò__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
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
layers
layer_metrics
non_trainable_variables
metrics
Ä	variables
 layer_regularization_losses
Åregularization_losses
Ætrainable_variables
ô__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
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
layers
layer_metrics
non_trainable_variables
metrics
È	variables
 layer_regularization_losses
Éregularization_losses
Êtrainable_variables
ö__call__
+õ&call_and_return_all_conditional_losses
'õ"call_and_return_conditional_losses"
_generic_user_object
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
1:/  2SGD/conv1d_10/kernel/momentum
':% 2SGD/conv1d_10/bias/momentum
1:/	  2SGD/conv1d_11/kernel/momentum
':% 2SGD/conv1d_11/bias/momentum
5:3 2)SGD/batch_normalization_10/gamma/momentum
4:2 2(SGD/batch_normalization_10/beta/momentum
5:3 2)SGD/batch_normalization_11/gamma/momentum
4:2 2(SGD/batch_normalization_11/beta/momentum
-:+	è@2SGD/dense_39/kernel/momentum
&:$@2SGD/dense_39/bias/momentum
,:*@@2SGD/dense_40/kernel/momentum
&:$@2SGD/dense_40/bias/momentum
,:*@2SGD/dense_41/kernel/momentum
&:$2SGD/dense_41/bias/momentum
S:Q 2CSGD/token_and_position_embedding_5/embedding_10/embeddings/momentum
T:R	R 2CSGD/token_and_position_embedding_5/embedding_11/embeddings/momentum
Z:X  2FSGD/transformer_block_11/multi_head_attention_11/query/kernel/momentum
T:R 2DSGD/transformer_block_11/multi_head_attention_11/query/bias/momentum
X:V  2DSGD/transformer_block_11/multi_head_attention_11/key/kernel/momentum
R:P 2BSGD/transformer_block_11/multi_head_attention_11/key/bias/momentum
Z:X  2FSGD/transformer_block_11/multi_head_attention_11/value/kernel/momentum
T:R 2DSGD/transformer_block_11/multi_head_attention_11/value/bias/momentum
e:c  2QSGD/transformer_block_11/multi_head_attention_11/attention_output/kernel/momentum
[:Y 2OSGD/transformer_block_11/multi_head_attention_11/attention_output/bias/momentum
,:* @2SGD/dense_37/kernel/momentum
&:$@2SGD/dense_37/bias/momentum
,:*@ 2SGD/dense_38/kernel/momentum
&:$ 2SGD/dense_38/bias/momentum
J:H 2>SGD/transformer_block_11/layer_normalization_22/gamma/momentum
I:G 2=SGD/transformer_block_11/layer_normalization_22/beta/momentum
J:H 2>SGD/transformer_block_11/layer_normalization_23/gamma/momentum
I:G 2=SGD/transformer_block_11/layer_normalization_23/beta/momentum
Ú2×
C__inference_model_5_layer_call_and_return_conditional_losses_765001
C__inference_model_5_layer_call_and_return_conditional_losses_764907
C__inference_model_5_layer_call_and_return_conditional_losses_765742
C__inference_model_5_layer_call_and_return_conditional_losses_765985À
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
î2ë
(__inference_model_5_layer_call_fn_765174
(__inference_model_5_layer_call_fn_766063
(__inference_model_5_layer_call_fn_766141
(__inference_model_5_layer_call_fn_765346À
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
2
!__inference__wrapped_model_763551á
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
annotationsª *Q¢N
LI
# 
input_11ÿÿÿÿÿÿÿÿÿR
"
input_12ÿÿÿÿÿÿÿÿÿ
ÿ2ü
Z__inference_token_and_position_embedding_5_layer_call_and_return_conditional_losses_766165
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
?__inference_token_and_position_embedding_5_layer_call_fn_766174
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
ï2ì
E__inference_conv1d_10_layer_call_and_return_conditional_losses_766190¢
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
*__inference_conv1d_10_layer_call_fn_766199¢
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
P__inference_average_pooling1d_15_layer_call_and_return_conditional_losses_763560Ó
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
5__inference_average_pooling1d_15_layer_call_fn_763566Ó
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
ï2ì
E__inference_conv1d_11_layer_call_and_return_conditional_losses_766215¢
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
*__inference_conv1d_11_layer_call_fn_766224¢
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
P__inference_average_pooling1d_16_layer_call_and_return_conditional_losses_763575Ó
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
5__inference_average_pooling1d_16_layer_call_fn_763581Ó
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
P__inference_average_pooling1d_17_layer_call_and_return_conditional_losses_763590Ó
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
5__inference_average_pooling1d_17_layer_call_fn_763596Ó
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
2
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_766280
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_766362
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_766342
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_766260´
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
2
7__inference_batch_normalization_10_layer_call_fn_766388
7__inference_batch_normalization_10_layer_call_fn_766306
7__inference_batch_normalization_10_layer_call_fn_766293
7__inference_batch_normalization_10_layer_call_fn_766375´
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
2
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_766444
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_766506
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_766526
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_766424´
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
2
7__inference_batch_normalization_11_layer_call_fn_766552
7__inference_batch_normalization_11_layer_call_fn_766457
7__inference_batch_normalization_11_layer_call_fn_766539
7__inference_batch_normalization_11_layer_call_fn_766470´
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
A__inference_add_5_layer_call_and_return_conditional_losses_766558¢
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
&__inference_add_5_layer_call_fn_766564¢
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
Ú2×
P__inference_transformer_block_11_layer_call_and_return_conditional_losses_766712
P__inference_transformer_block_11_layer_call_and_return_conditional_losses_766839°
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
¤2¡
5__inference_transformer_block_11_layer_call_fn_766876
5__inference_transformer_block_11_layer_call_fn_766913°
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_766919¢
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
*__inference_flatten_5_layer_call_fn_766924¢
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
I__inference_concatenate_5_layer_call_and_return_conditional_losses_766931¢
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
.__inference_concatenate_5_layer_call_fn_766937¢
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
D__inference_dense_39_layer_call_and_return_conditional_losses_766948¢
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
)__inference_dense_39_layer_call_fn_766957¢
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
F__inference_dropout_34_layer_call_and_return_conditional_losses_766969
F__inference_dropout_34_layer_call_and_return_conditional_losses_766974´
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
+__inference_dropout_34_layer_call_fn_766984
+__inference_dropout_34_layer_call_fn_766979´
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
D__inference_dense_40_layer_call_and_return_conditional_losses_766995¢
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
)__inference_dense_40_layer_call_fn_767004¢
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
F__inference_dropout_35_layer_call_and_return_conditional_losses_767016
F__inference_dropout_35_layer_call_and_return_conditional_losses_767021´
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
+__inference_dropout_35_layer_call_fn_767031
+__inference_dropout_35_layer_call_fn_767026´
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
D__inference_dense_41_layer_call_and_return_conditional_losses_767041¢
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
)__inference_dense_41_layer_call_fn_767050¢
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
ÔBÑ
$__inference_signature_wrapper_765432input_11input_12"
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
ò2ï
I__inference_sequential_11_layer_call_and_return_conditional_losses_763974
I__inference_sequential_11_layer_call_and_return_conditional_losses_767164
I__inference_sequential_11_layer_call_and_return_conditional_losses_767107
I__inference_sequential_11_layer_call_and_return_conditional_losses_763988À
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
2
.__inference_sequential_11_layer_call_fn_767190
.__inference_sequential_11_layer_call_fn_764043
.__inference_sequential_11_layer_call_fn_764016
.__inference_sequential_11_layer_call_fn_767177À
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
D__inference_dense_37_layer_call_and_return_conditional_losses_767221¢
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
)__inference_dense_37_layer_call_fn_767230¢
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
D__inference_dense_38_layer_call_and_return_conditional_losses_767260¢
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
)__inference_dense_38_layer_call_fn_767269¢
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
!__inference__wrapped_model_763551È4~ !*+<9;:EBDC`ajktu[¢X
Q¢N
LI
# 
input_11ÿÿÿÿÿÿÿÿÿR
"
input_12ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_41"
dense_41ÿÿÿÿÿÿÿÿÿÕ
A__inference_add_5_layer_call_and_return_conditional_losses_766558b¢_
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
&__inference_add_5_layer_call_fn_766564b¢_
X¢U
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿ# 
&#
inputs/1ÿÿÿÿÿÿÿÿÿ# 
ª "ÿÿÿÿÿÿÿÿÿ# Ù
P__inference_average_pooling1d_15_layer_call_and_return_conditional_losses_763560E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 °
5__inference_average_pooling1d_15_layer_call_fn_763566wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÙ
P__inference_average_pooling1d_16_layer_call_and_return_conditional_losses_763575E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 °
5__inference_average_pooling1d_16_layer_call_fn_763581wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÙ
P__inference_average_pooling1d_17_layer_call_and_return_conditional_losses_763590E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 °
5__inference_average_pooling1d_17_layer_call_fn_763596wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÒ
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_766260|;<9:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ò
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_766280|<9;:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 À
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_766342j;<9:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 À
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_766362j<9;:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ª
7__inference_batch_normalization_10_layer_call_fn_766293o;<9:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ª
7__inference_batch_normalization_10_layer_call_fn_766306o<9;:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
7__inference_batch_normalization_10_layer_call_fn_766375];<9:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª "ÿÿÿÿÿÿÿÿÿ# 
7__inference_batch_normalization_10_layer_call_fn_766388]<9;:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª "ÿÿÿÿÿÿÿÿÿ# À
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_766424jDEBC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 À
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_766444jEBDC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 Ò
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_766506|DEBC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ò
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_766526|EBDC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
7__inference_batch_normalization_11_layer_call_fn_766457]DEBC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª "ÿÿÿÿÿÿÿÿÿ# 
7__inference_batch_normalization_11_layer_call_fn_766470]EBDC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª "ÿÿÿÿÿÿÿÿÿ# ª
7__inference_batch_normalization_11_layer_call_fn_766539oDEBC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ª
7__inference_batch_normalization_11_layer_call_fn_766552oEBDC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ó
I__inference_concatenate_5_layer_call_and_return_conditional_losses_766931[¢X
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
.__inference_concatenate_5_layer_call_fn_766937x[¢X
Q¢N
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿà
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿè¯
E__inference_conv1d_10_layer_call_and_return_conditional_losses_766190f !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿR 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿR 
 
*__inference_conv1d_10_layer_call_fn_766199Y !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿR 
ª "ÿÿÿÿÿÿÿÿÿR ¯
E__inference_conv1d_11_layer_call_and_return_conditional_losses_766215f*+4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÞ 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÞ 
 
*__inference_conv1d_11_layer_call_fn_766224Y*+4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÞ 
ª "ÿÿÿÿÿÿÿÿÿÞ ®
D__inference_dense_37_layer_call_and_return_conditional_losses_767221f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ#@
 
)__inference_dense_37_layer_call_fn_767230Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª "ÿÿÿÿÿÿÿÿÿ#@®
D__inference_dense_38_layer_call_and_return_conditional_losses_767260f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ#@
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 
)__inference_dense_38_layer_call_fn_767269Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ#@
ª "ÿÿÿÿÿÿÿÿÿ# ¥
D__inference_dense_39_layer_call_and_return_conditional_losses_766948]`a0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿè
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 }
)__inference_dense_39_layer_call_fn_766957P`a0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿè
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dense_40_layer_call_and_return_conditional_losses_766995\jk/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
)__inference_dense_40_layer_call_fn_767004Ojk/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dense_41_layer_call_and_return_conditional_losses_767041\tu/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_41_layer_call_fn_767050Otu/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dropout_34_layer_call_and_return_conditional_losses_766969\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¦
F__inference_dropout_34_layer_call_and_return_conditional_losses_766974\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ~
+__inference_dropout_34_layer_call_fn_766979O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@~
+__inference_dropout_34_layer_call_fn_766984O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@¦
F__inference_dropout_35_layer_call_and_return_conditional_losses_767016\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¦
F__inference_dropout_35_layer_call_and_return_conditional_losses_767021\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ~
+__inference_dropout_35_layer_call_fn_767026O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@~
+__inference_dropout_35_layer_call_fn_767031O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@¦
E__inference_flatten_5_layer_call_and_return_conditional_losses_766919]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿà
 ~
*__inference_flatten_5_layer_call_fn_766924P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª "ÿÿÿÿÿÿÿÿÿà
C__inference_model_5_layer_call_and_return_conditional_losses_764907Â4~ !*+;<9:DEBC`ajktuc¢`
Y¢V
LI
# 
input_11ÿÿÿÿÿÿÿÿÿR
"
input_12ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
C__inference_model_5_layer_call_and_return_conditional_losses_765001Â4~ !*+<9;:EBDC`ajktuc¢`
Y¢V
LI
# 
input_11ÿÿÿÿÿÿÿÿÿR
"
input_12ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
C__inference_model_5_layer_call_and_return_conditional_losses_765742Â4~ !*+;<9:DEBC`ajktuc¢`
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
C__inference_model_5_layer_call_and_return_conditional_losses_765985Â4~ !*+<9;:EBDC`ajktuc¢`
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
 â
(__inference_model_5_layer_call_fn_765174µ4~ !*+;<9:DEBC`ajktuc¢`
Y¢V
LI
# 
input_11ÿÿÿÿÿÿÿÿÿR
"
input_12ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿâ
(__inference_model_5_layer_call_fn_765346µ4~ !*+<9;:EBDC`ajktuc¢`
Y¢V
LI
# 
input_11ÿÿÿÿÿÿÿÿÿR
"
input_12ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿâ
(__inference_model_5_layer_call_fn_766063µ4~ !*+;<9:DEBC`ajktuc¢`
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
(__inference_model_5_layer_call_fn_766141µ4~ !*+<9;:EBDC`ajktuc¢`
Y¢V
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿR
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÇ
I__inference_sequential_11_layer_call_and_return_conditional_losses_763974zC¢@
9¢6
,)
dense_37_inputÿÿÿÿÿÿÿÿÿ# 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 Ç
I__inference_sequential_11_layer_call_and_return_conditional_losses_763988zC¢@
9¢6
,)
dense_37_inputÿÿÿÿÿÿÿÿÿ# 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ¿
I__inference_sequential_11_layer_call_and_return_conditional_losses_767107r;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ¿
I__inference_sequential_11_layer_call_and_return_conditional_losses_767164r;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 
.__inference_sequential_11_layer_call_fn_764016mC¢@
9¢6
,)
dense_37_inputÿÿÿÿÿÿÿÿÿ# 
p

 
ª "ÿÿÿÿÿÿÿÿÿ# 
.__inference_sequential_11_layer_call_fn_764043mC¢@
9¢6
,)
dense_37_inputÿÿÿÿÿÿÿÿÿ# 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ# 
.__inference_sequential_11_layer_call_fn_767177e;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p

 
ª "ÿÿÿÿÿÿÿÿÿ# 
.__inference_sequential_11_layer_call_fn_767190e;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ# 
$__inference_signature_wrapper_765432Û4~ !*+<9;:EBDC`ajktun¢k
¢ 
dªa
/
input_11# 
input_11ÿÿÿÿÿÿÿÿÿR
.
input_12"
input_12ÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_41"
dense_41ÿÿÿÿÿÿÿÿÿ»
Z__inference_token_and_position_embedding_5_layer_call_and_return_conditional_losses_766165]~+¢(
!¢

xÿÿÿÿÿÿÿÿÿR
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿR 
 
?__inference_token_and_position_embedding_5_layer_call_fn_766174P~+¢(
!¢

xÿÿÿÿÿÿÿÿÿR
ª "ÿÿÿÿÿÿÿÿÿR Û
P__inference_transformer_block_11_layer_call_and_return_conditional_losses_766712 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 Û
P__inference_transformer_block_11_layer_call_and_return_conditional_losses_766839 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ²
5__inference_transformer_block_11_layer_call_fn_766876y 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª "ÿÿÿÿÿÿÿÿÿ# ²
5__inference_transformer_block_11_layer_call_fn_766913y 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª "ÿÿÿÿÿÿÿÿÿ# 