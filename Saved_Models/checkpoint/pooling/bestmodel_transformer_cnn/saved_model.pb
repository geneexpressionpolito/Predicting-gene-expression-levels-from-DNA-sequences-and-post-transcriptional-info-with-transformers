εδ0
Ώ!!
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
Ό
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
₯
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
Ύ
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
φ
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
 "serve*2.4.12v2.4.1-0-g85c8b2a817f8§¬*
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
’
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
’
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
z
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(@* 
shared_namedense_25/kernel
s
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel*
_output_shapes

:(@*
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
Ζ
5token_and_position_embedding_3/embedding_6/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75token_and_position_embedding_3/embedding_6/embeddings
Ώ
Itoken_and_position_embedding_3/embedding_6/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_3/embedding_6/embeddings*
_output_shapes

: *
dtype0
Η
5token_and_position_embedding_3/embedding_7/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	R *F
shared_name75token_and_position_embedding_3/embedding_7/embeddings
ΐ
Itoken_and_position_embedding_3/embedding_7/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_3/embedding_7/embeddings*
_output_shapes
:	R *
dtype0
Ξ
7transformer_block_7/multi_head_attention_7/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_7/multi_head_attention_7/query/kernel
Η
Ktransformer_block_7/multi_head_attention_7/query/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_7/multi_head_attention_7/query/kernel*"
_output_shapes
:  *
dtype0
Ζ
5transformer_block_7/multi_head_attention_7/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_7/multi_head_attention_7/query/bias
Ώ
Itransformer_block_7/multi_head_attention_7/query/bias/Read/ReadVariableOpReadVariableOp5transformer_block_7/multi_head_attention_7/query/bias*
_output_shapes

: *
dtype0
Κ
5transformer_block_7/multi_head_attention_7/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *F
shared_name75transformer_block_7/multi_head_attention_7/key/kernel
Γ
Itransformer_block_7/multi_head_attention_7/key/kernel/Read/ReadVariableOpReadVariableOp5transformer_block_7/multi_head_attention_7/key/kernel*"
_output_shapes
:  *
dtype0
Β
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
Ξ
7transformer_block_7/multi_head_attention_7/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_7/multi_head_attention_7/value/kernel
Η
Ktransformer_block_7/multi_head_attention_7/value/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_7/multi_head_attention_7/value/kernel*"
_output_shapes
:  *
dtype0
Ζ
5transformer_block_7/multi_head_attention_7/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_7/multi_head_attention_7/value/bias
Ώ
Itransformer_block_7/multi_head_attention_7/value/bias/Read/ReadVariableOpReadVariableOp5transformer_block_7/multi_head_attention_7/value/bias*
_output_shapes

: *
dtype0
δ
Btransformer_block_7/multi_head_attention_7/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBtransformer_block_7/multi_head_attention_7/attention_output/kernel
έ
Vtransformer_block_7/multi_head_attention_7/attention_output/kernel/Read/ReadVariableOpReadVariableOpBtransformer_block_7/multi_head_attention_7/attention_output/kernel*"
_output_shapes
:  *
dtype0
Ψ
@transformer_block_7/multi_head_attention_7/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@transformer_block_7/multi_head_attention_7/attention_output/bias
Ρ
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
Έ
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
Ά
/transformer_block_7/layer_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_7/layer_normalization_14/beta
―
Ctransformer_block_7/layer_normalization_14/beta/Read/ReadVariableOpReadVariableOp/transformer_block_7/layer_normalization_14/beta*
_output_shapes
: *
dtype0
Έ
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
Ά
/transformer_block_7/layer_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_7/layer_normalization_15/beta
―
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
‘
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
‘
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

SGD/dense_25/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(@*-
shared_nameSGD/dense_25/kernel/momentum

0SGD/dense_25/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_25/kernel/momentum*
_output_shapes

:(@*
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
ΰ
BSGD/token_and_position_embedding_3/embedding_6/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/token_and_position_embedding_3/embedding_6/embeddings/momentum
Ω
VSGD/token_and_position_embedding_3/embedding_6/embeddings/momentum/Read/ReadVariableOpReadVariableOpBSGD/token_and_position_embedding_3/embedding_6/embeddings/momentum*
_output_shapes

: *
dtype0
α
BSGD/token_and_position_embedding_3/embedding_7/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	R *S
shared_nameDBSGD/token_and_position_embedding_3/embedding_7/embeddings/momentum
Ϊ
VSGD/token_and_position_embedding_3/embedding_7/embeddings/momentum/Read/ReadVariableOpReadVariableOpBSGD/token_and_position_embedding_3/embedding_7/embeddings/momentum*
_output_shapes
:	R *
dtype0
θ
DSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentum
α
XSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentum*"
_output_shapes
:  *
dtype0
ΰ
BSGD/transformer_block_7/multi_head_attention_7/query/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_7/multi_head_attention_7/query/bias/momentum
Ω
VSGD/transformer_block_7/multi_head_attention_7/query/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_7/multi_head_attention_7/query/bias/momentum*
_output_shapes

: *
dtype0
δ
BSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentum
έ
VSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentum*"
_output_shapes
:  *
dtype0
ά
@SGD/transformer_block_7/multi_head_attention_7/key/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *Q
shared_nameB@SGD/transformer_block_7/multi_head_attention_7/key/bias/momentum
Υ
TSGD/transformer_block_7/multi_head_attention_7/key/bias/momentum/Read/ReadVariableOpReadVariableOp@SGD/transformer_block_7/multi_head_attention_7/key/bias/momentum*
_output_shapes

: *
dtype0
θ
DSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentum
α
XSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentum*"
_output_shapes
:  *
dtype0
ΰ
BSGD/transformer_block_7/multi_head_attention_7/value/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_7/multi_head_attention_7/value/bias/momentum
Ω
VSGD/transformer_block_7/multi_head_attention_7/value/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_7/multi_head_attention_7/value/bias/momentum*
_output_shapes

: *
dtype0
ώ
OSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *`
shared_nameQOSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentum
χ
cSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentum/Read/ReadVariableOpReadVariableOpOSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentum*"
_output_shapes
:  *
dtype0
ς
MSGD/transformer_block_7/multi_head_attention_7/attention_output/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *^
shared_nameOMSGD/transformer_block_7/multi_head_attention_7/attention_output/bias/momentum
λ
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
?
=SGD/transformer_block_7/layer_normalization_14/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=SGD/transformer_block_7/layer_normalization_14/gamma/momentum
Λ
QSGD/transformer_block_7/layer_normalization_14/gamma/momentum/Read/ReadVariableOpReadVariableOp=SGD/transformer_block_7/layer_normalization_14/gamma/momentum*
_output_shapes
: *
dtype0
Π
<SGD/transformer_block_7/layer_normalization_14/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_7/layer_normalization_14/beta/momentum
Ι
PSGD/transformer_block_7/layer_normalization_14/beta/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_7/layer_normalization_14/beta/momentum*
_output_shapes
: *
dtype0
?
=SGD/transformer_block_7/layer_normalization_15/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=SGD/transformer_block_7/layer_normalization_15/gamma/momentum
Λ
QSGD/transformer_block_7/layer_normalization_15/gamma/momentum/Read/ReadVariableOpReadVariableOp=SGD/transformer_block_7/layer_normalization_15/gamma/momentum*
_output_shapes
: *
dtype0
Π
<SGD/transformer_block_7/layer_normalization_15/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_7/layer_normalization_15/beta/momentum
Ι
PSGD/transformer_block_7/layer_normalization_15/beta/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_7/layer_normalization_15/beta/momentum*
_output_shapes
: *
dtype0

NoOpNoOp
υ΅
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*―΅
value€΅B ΅ B΅
ι
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
layer-14
layer_with_weights-6
layer-15
layer-16
layer_with_weights-7
layer-17
layer-18
layer_with_weights-8
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
	variables
regularization_losses
trainable_variables
 	keras_api
h

!kernel
"bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
R
'	variables
(regularization_losses
)trainable_variables
*	keras_api
h

+kernel
,bias
-	variables
.regularization_losses
/trainable_variables
0	keras_api
R
1	variables
2regularization_losses
3trainable_variables
4	keras_api
R
5	variables
6regularization_losses
7trainable_variables
8	keras_api

9axis
	:gamma
;beta
<moving_mean
=moving_variance
>	variables
?regularization_losses
@trainable_variables
A	keras_api

Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
R
K	variables
Lregularization_losses
Mtrainable_variables
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
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
R
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
R
]	variables
^regularization_losses
_trainable_variables
`	keras_api
 
R
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
h

ekernel
fbias
g	variables
hregularization_losses
itrainable_variables
j	keras_api
R
k	variables
lregularization_losses
mtrainable_variables
n	keras_api
h

okernel
pbias
q	variables
rregularization_losses
strainable_variables
t	keras_api
R
u	variables
vregularization_losses
wtrainable_variables
x	keras_api
h

ykernel
zbias
{	variables
|regularization_losses
}trainable_variables
~	keras_api
λ
	decay
learning_rate
momentum
	iter!momentum"momentum+momentum,momentum:momentum;momentum Cmomentum‘Dmomentum’emomentum£fmomentum€omomentum₯pmomentum¦ymomentum§zmomentum¨momentum©momentumͺmomentum«momentum¬momentum­momentum?momentum―momentum°momentum±momentum²momentum³momentum΄momentum΅momentumΆmomentum·momentumΈmomentumΉmomentumΊ
¨
0
1
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
26
27
28
29
e30
f31
o32
p33
y34
z35
 

0
1
!2
"3
+4
,5
:6
;7
C8
D9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
e26
f27
o28
p29
y30
z31
²
metrics
	variables
 layer_regularization_losses
regularization_losses
trainable_variables
layer_metrics
layers
non_trainable_variables
 
g

embeddings
	variables
regularization_losses
trainable_variables
	keras_api
g

embeddings
	variables
regularization_losses
 trainable_variables
‘	keras_api

0
1
 

0
1
²
’metrics
	variables
 £layer_regularization_losses
regularization_losses
trainable_variables
€layer_metrics
₯layers
¦non_trainable_variables
[Y
VARIABLE_VALUEconv1d_6/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_6/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
 

!0
"1
²
§metrics
#	variables
 ¨layer_regularization_losses
$regularization_losses
%trainable_variables
©layer_metrics
ͺlayers
«non_trainable_variables
 
 
 
²
¬metrics
'	variables
 ­layer_regularization_losses
(regularization_losses
)trainable_variables
?layer_metrics
―layers
°non_trainable_variables
[Y
VARIABLE_VALUEconv1d_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1
 

+0
,1
²
±metrics
-	variables
 ²layer_regularization_losses
.regularization_losses
/trainable_variables
³layer_metrics
΄layers
΅non_trainable_variables
 
 
 
²
Άmetrics
1	variables
 ·layer_regularization_losses
2regularization_losses
3trainable_variables
Έlayer_metrics
Ήlayers
Ίnon_trainable_variables
 
 
 
²
»metrics
5	variables
 Όlayer_regularization_losses
6regularization_losses
7trainable_variables
½layer_metrics
Ύlayers
Ώnon_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_6/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_6/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_6/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_6/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
<2
=3
 

:0
;1
²
ΐmetrics
>	variables
 Αlayer_regularization_losses
?regularization_losses
@trainable_variables
Βlayer_metrics
Γlayers
Δnon_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_7/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_7/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_7/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_7/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

C0
D1
E2
F3
 

C0
D1
²
Εmetrics
G	variables
 Ζlayer_regularization_losses
Hregularization_losses
Itrainable_variables
Ηlayer_metrics
Θlayers
Ιnon_trainable_variables
 
 
 
²
Κmetrics
K	variables
 Λlayer_regularization_losses
Lregularization_losses
Mtrainable_variables
Μlayer_metrics
Νlayers
Ξnon_trainable_variables
Ε
Ο_query_dense
Π
_key_dense
Ρ_value_dense
?_softmax
Σ_dropout_layer
Τ_output_dense
Υ	variables
Φregularization_losses
Χtrainable_variables
Ψ	keras_api
¨
Ωlayer_with_weights-0
Ωlayer-0
Ϊlayer_with_weights-1
Ϊlayer-1
Ϋ	variables
άregularization_losses
έtrainable_variables
ή	keras_api
x
	ίaxis

gamma
	beta
ΰ	variables
αregularization_losses
βtrainable_variables
γ	keras_api
x
	δaxis

gamma
	beta
ε	variables
ζregularization_losses
ηtrainable_variables
θ	keras_api
V
ι	variables
κregularization_losses
λtrainable_variables
μ	keras_api
V
ν	variables
ξregularization_losses
οtrainable_variables
π	keras_api

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
 

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
²
ρmetrics
U	variables
 ςlayer_regularization_losses
Vregularization_losses
Wtrainable_variables
σlayer_metrics
τlayers
υnon_trainable_variables
 
 
 
²
φmetrics
Y	variables
 χlayer_regularization_losses
Zregularization_losses
[trainable_variables
ψlayer_metrics
ωlayers
ϊnon_trainable_variables
 
 
 
²
ϋmetrics
]	variables
 όlayer_regularization_losses
^regularization_losses
_trainable_variables
ύlayer_metrics
ώlayers
?non_trainable_variables
 
 
 
²
metrics
a	variables
 layer_regularization_losses
bregularization_losses
ctrainable_variables
layer_metrics
layers
non_trainable_variables
[Y
VARIABLE_VALUEdense_25/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_25/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

e0
f1
 

e0
f1
²
metrics
g	variables
 layer_regularization_losses
hregularization_losses
itrainable_variables
layer_metrics
layers
non_trainable_variables
 
 
 
²
metrics
k	variables
 layer_regularization_losses
lregularization_losses
mtrainable_variables
layer_metrics
layers
non_trainable_variables
[Y
VARIABLE_VALUEdense_26/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_26/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

o0
p1
 

o0
p1
²
metrics
q	variables
 layer_regularization_losses
rregularization_losses
strainable_variables
layer_metrics
layers
non_trainable_variables
 
 
 
²
metrics
u	variables
 layer_regularization_losses
vregularization_losses
wtrainable_variables
layer_metrics
layers
non_trainable_variables
[Y
VARIABLE_VALUEdense_27/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_27/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

y0
z1
 

y0
z1
²
metrics
{	variables
 layer_regularization_losses
|regularization_losses
}trainable_variables
layer_metrics
layers
non_trainable_variables
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
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

0
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

<0
=1
E2
F3

0
 

0
΅
metrics
	variables
  layer_regularization_losses
regularization_losses
trainable_variables
‘layer_metrics
’layers
£non_trainable_variables

0
 

0
΅
€metrics
	variables
 ₯layer_regularization_losses
regularization_losses
 trainable_variables
¦layer_metrics
§layers
¨non_trainable_variables
 
 
 
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
‘
©partial_output_shape
ͺfull_output_shape
kernel
	bias
«	variables
¬regularization_losses
­trainable_variables
?	keras_api
‘
―partial_output_shape
°full_output_shape
kernel
	bias
±	variables
²regularization_losses
³trainable_variables
΄	keras_api
‘
΅partial_output_shape
Άfull_output_shape
kernel
	bias
·	variables
Έregularization_losses
Ήtrainable_variables
Ί	keras_api
V
»	variables
Όregularization_losses
½trainable_variables
Ύ	keras_api
V
Ώ	variables
ΐregularization_losses
Αtrainable_variables
Β	keras_api
‘
Γpartial_output_shape
Δfull_output_shape
kernel
	bias
Ε	variables
Ζregularization_losses
Ηtrainable_variables
Θ	keras_api
@
0
1
2
3
4
5
6
7
 
@
0
1
2
3
4
5
6
7
΅
Ιmetrics
Υ	variables
 Κlayer_regularization_losses
Φregularization_losses
Χtrainable_variables
Λlayer_metrics
Μlayers
Νnon_trainable_variables
n
kernel
	bias
Ξ	variables
Οregularization_losses
Πtrainable_variables
Ρ	keras_api
n
kernel
	bias
?	variables
Σregularization_losses
Τtrainable_variables
Υ	keras_api
 
0
1
2
3
 
 
0
1
2
3
΅
Φmetrics
Ϋ	variables
 Χlayer_regularization_losses
άregularization_losses
έtrainable_variables
Ψlayer_metrics
Ωlayers
Ϊnon_trainable_variables
 

0
1
 

0
1
΅
Ϋmetrics
ΰ	variables
 άlayer_regularization_losses
αregularization_losses
βtrainable_variables
έlayer_metrics
ήlayers
ίnon_trainable_variables
 

0
1
 

0
1
΅
ΰmetrics
ε	variables
 αlayer_regularization_losses
ζregularization_losses
ηtrainable_variables
βlayer_metrics
γlayers
δnon_trainable_variables
 
 
 
΅
εmetrics
ι	variables
 ζlayer_regularization_losses
κregularization_losses
λtrainable_variables
ηlayer_metrics
θlayers
ιnon_trainable_variables
 
 
 
΅
κmetrics
ν	variables
 λlayer_regularization_losses
ξregularization_losses
οtrainable_variables
μlayer_metrics
νlayers
ξnon_trainable_variables
 
 
 
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
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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

οtotal

πcount
ρ	variables
ς	keras_api
 
 
 
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
0
1
 

0
1
΅
σmetrics
«	variables
 τlayer_regularization_losses
¬regularization_losses
­trainable_variables
υlayer_metrics
φlayers
χnon_trainable_variables
 
 

0
1
 

0
1
΅
ψmetrics
±	variables
 ωlayer_regularization_losses
²regularization_losses
³trainable_variables
ϊlayer_metrics
ϋlayers
όnon_trainable_variables
 
 

0
1
 

0
1
΅
ύmetrics
·	variables
 ώlayer_regularization_losses
Έregularization_losses
Ήtrainable_variables
?layer_metrics
layers
non_trainable_variables
 
 
 
΅
metrics
»	variables
 layer_regularization_losses
Όregularization_losses
½trainable_variables
layer_metrics
layers
non_trainable_variables
 
 
 
΅
metrics
Ώ	variables
 layer_regularization_losses
ΐregularization_losses
Αtrainable_variables
layer_metrics
layers
non_trainable_variables
 
 

0
1
 

0
1
΅
metrics
Ε	variables
 layer_regularization_losses
Ζregularization_losses
Ηtrainable_variables
layer_metrics
layers
non_trainable_variables
 
 
 
0
Ο0
Π1
Ρ2
?3
Σ4
Τ5
 

0
1
 

0
1
΅
metrics
Ξ	variables
 layer_regularization_losses
Οregularization_losses
Πtrainable_variables
layer_metrics
layers
non_trainable_variables

0
1
 

0
1
΅
metrics
?	variables
 layer_regularization_losses
Σregularization_losses
Τtrainable_variables
layer_metrics
layers
non_trainable_variables
 
 
 

Ω0
Ϊ1
 
 
 
 
 
 
 
 
 
 
 
 
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
ο0
π1

ρ	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
’
VARIABLE_VALUEBSGD/token_and_position_embedding_3/embedding_6/embeddings/momentumIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
’
VARIABLE_VALUEBSGD/token_and_position_embedding_3/embedding_7/embeddings/momentumIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
₯’
VARIABLE_VALUEDSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentumJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEBSGD/transformer_block_7/multi_head_attention_7/query/bias/momentumJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEBSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentumJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
‘
VARIABLE_VALUE@SGD/transformer_block_7/multi_head_attention_7/key/bias/momentumJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
₯’
VARIABLE_VALUEDSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentumJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEBSGD/transformer_block_7/multi_head_attention_7/value/bias/momentumJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
°­
VARIABLE_VALUEOSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentumJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
?«
VARIABLE_VALUEMSGD/transformer_block_7/multi_head_attention_7/attention_output/bias/momentumJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUESGD/dense_23/kernel/momentumJvariables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUESGD/dense_23/bias/momentumJvariables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUESGD/dense_24/kernel/momentumJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUESGD/dense_24/bias/momentumJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=SGD/transformer_block_7/layer_normalization_14/gamma/momentumJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<SGD/transformer_block_7/layer_normalization_14/beta/momentumJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=SGD/transformer_block_7/layer_normalization_15/gamma/momentumJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<SGD/transformer_block_7/layer_normalization_15/beta/momentumJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_7Placeholder*(
_output_shapes
:?????????R*
dtype0*
shape:?????????R
z
serving_default_input_8Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_7serving_default_input_85token_and_position_embedding_3/embedding_7/embeddings5token_and_position_embedding_3/embedding_6/embeddingsconv1d_6/kernelconv1d_6/biasconv1d_7/kernelconv1d_7/bias%batch_normalization_6/moving_variancebatch_normalization_6/gamma!batch_normalization_6/moving_meanbatch_normalization_6/beta%batch_normalization_7/moving_variancebatch_normalization_7/gamma!batch_normalization_7/moving_meanbatch_normalization_7/beta7transformer_block_7/multi_head_attention_7/query/kernel5transformer_block_7/multi_head_attention_7/query/bias5transformer_block_7/multi_head_attention_7/key/kernel3transformer_block_7/multi_head_attention_7/key/bias7transformer_block_7/multi_head_attention_7/value/kernel5transformer_block_7/multi_head_attention_7/value/biasBtransformer_block_7/multi_head_attention_7/attention_output/kernel@transformer_block_7/multi_head_attention_7/attention_output/bias0transformer_block_7/layer_normalization_14/gamma/transformer_block_7/layer_normalization_14/betadense_23/kerneldense_23/biasdense_24/kerneldense_24/bias0transformer_block_7/layer_normalization_15/gamma/transformer_block_7/layer_normalization_15/betadense_25/kerneldense_25/biasdense_26/kerneldense_26/biasdense_27/kerneldense_27/bias*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_307008
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Μ$
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
__inference__traced_save_309117
?
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
"__inference__traced_restore_309349±ν&

χ
D__inference_conv1d_6_layer_call_and_return_conditional_losses_305665

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????R 2
conv1d/ExpandDimsΈ
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
:?????????R *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:?????????R *
squeeze_dims

ύ????????2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????R 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????R 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:?????????R 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????R ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????R 
 
_user_specified_nameinputs
μ
©
6__inference_batch_normalization_6_layer_call_fn_307873

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3052342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs


Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_308024

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOp
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
 :?????????????????? 2
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
 :?????????????????? 2
batchnorm/add_1θ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????????????? ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
ψ
l
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_305132

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
-:+???????????????????????????2

ExpandDimsΌ
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+???????????????????????????*
ksize	
¬*
paddingVALID*
strides	
¬2	
AvgPool
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
ξ
©
6__inference_batch_normalization_6_layer_call_fn_307886

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3052672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
¬
R
&__inference_add_3_layer_call_fn_308144
inputs_0
inputs_1
identityΣ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
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
GPU2*0J 8 *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_3059042
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:?????????# :?????????# :U Q
+
_output_shapes
:?????????# 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????# 
"
_user_specified_name
inputs/1

e
F__inference_dropout_23_layer_call_and_return_conditional_losses_308618

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?2
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
dropout/Shape΄
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
 *ΝΜΜ=2
dropout/GreaterEqual/yΎ
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
Έ
 
-__inference_sequential_7_layer_call_fn_308792

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall
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
GPU2*0J 8 *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_3055742
StatefulPartitionedCall
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
τ

Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_307745
x'
#embedding_7_embedding_lookup_307732'
#embedding_6_embedding_lookup_307738
identity’embedding_6/embedding_lookup’embedding_7/embedding_lookup?
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
strided_slice/stack_2β
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
:?????????2
range―
embedding_7/embedding_lookupResourceGather#embedding_7_embedding_lookup_307732range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_7/embedding_lookup/307732*'
_output_shapes
:????????? *
dtype02
embedding_7/embedding_lookup
%embedding_7/embedding_lookup/IdentityIdentity%embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_7/embedding_lookup/307732*'
_output_shapes
:????????? 2'
%embedding_7/embedding_lookup/Identityΐ
'embedding_7/embedding_lookup/Identity_1Identity.embedding_7/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2)
'embedding_7/embedding_lookup/Identity_1q
embedding_6/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:?????????R2
embedding_6/CastΊ
embedding_6/embedding_lookupResourceGather#embedding_6_embedding_lookup_307738embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_6/embedding_lookup/307738*,
_output_shapes
:?????????R *
dtype02
embedding_6/embedding_lookup
%embedding_6/embedding_lookup/IdentityIdentity%embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_6/embedding_lookup/307738*,
_output_shapes
:?????????R 2'
%embedding_6/embedding_lookup/IdentityΕ
'embedding_6/embedding_lookup/Identity_1Identity.embedding_6/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:?????????R 2)
'embedding_6/embedding_lookup/Identity_1?
addAddV20embedding_6/embedding_lookup/Identity_1:output:00embedding_7/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:?????????R 2
add
IdentityIdentityadd:z:0^embedding_6/embedding_lookup^embedding_7/embedding_lookup*
T0*,
_output_shapes
:?????????R 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????R::2<
embedding_6/embedding_lookupembedding_6/embedding_lookup2<
embedding_7/embedding_lookupembedding_7/embedding_lookup:K G
(
_output_shapes
:?????????R

_user_specified_namex
Ήή
β
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_308419

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
identity’/layer_normalization_14/batchnorm/ReadVariableOp’3layer_normalization_14/batchnorm/mul/ReadVariableOp’/layer_normalization_15/batchnorm/ReadVariableOp’3layer_normalization_15/batchnorm/mul/ReadVariableOp’:multi_head_attention_7/attention_output/add/ReadVariableOp’Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp’-multi_head_attention_7/key/add/ReadVariableOp’7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp’/multi_head_attention_7/query/add/ReadVariableOp’9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp’/multi_head_attention_7/value/add/ReadVariableOp’9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp’,sequential_7/dense_23/BiasAdd/ReadVariableOp’.sequential_7/dense_23/Tensordot/ReadVariableOp’,sequential_7/dense_24/BiasAdd/ReadVariableOp’.sequential_7/dense_24/Tensordot/ReadVariableOpύ
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_7/query/einsum/EinsumEinsuminputsAmulti_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2,
*multi_head_attention_7/query/einsum/EinsumΫ
/multi_head_attention_7/query/add/ReadVariableOpReadVariableOp8multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/query/add/ReadVariableOpυ
 multi_head_attention_7/query/addAddV23multi_head_attention_7/query/einsum/Einsum:output:07multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2"
 multi_head_attention_7/query/addχ
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_7/key/einsum/EinsumEinsuminputs?multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2*
(multi_head_attention_7/key/einsum/EinsumΥ
-multi_head_attention_7/key/add/ReadVariableOpReadVariableOp6multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_7/key/add/ReadVariableOpν
multi_head_attention_7/key/addAddV21multi_head_attention_7/key/einsum/Einsum:output:05multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2 
multi_head_attention_7/key/addύ
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_7/value/einsum/EinsumEinsuminputsAmulti_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2,
*multi_head_attention_7/value/einsum/EinsumΫ
/multi_head_attention_7/value/add/ReadVariableOpReadVariableOp8multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/value/add/ReadVariableOpυ
 multi_head_attention_7/value/addAddV23multi_head_attention_7/value/einsum/Einsum:output:07multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2"
 multi_head_attention_7/value/add
multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *σ5>2
multi_head_attention_7/Mul/yΖ
multi_head_attention_7/MulMul$multi_head_attention_7/query/add:z:0%multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:?????????# 2
multi_head_attention_7/Mulό
$multi_head_attention_7/einsum/EinsumEinsum"multi_head_attention_7/key/add:z:0multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:?????????##*
equationaecd,abcd->acbe2&
$multi_head_attention_7/einsum/EinsumΔ
&multi_head_attention_7/softmax/SoftmaxSoftmax-multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????##2(
&multi_head_attention_7/softmax/SoftmaxΚ
'multi_head_attention_7/dropout/IdentityIdentity0multi_head_attention_7/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????##2)
'multi_head_attention_7/dropout/Identity
&multi_head_attention_7/einsum_1/EinsumEinsum0multi_head_attention_7/dropout/Identity:output:0$multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:?????????# *
equationacbe,aecd->abcd2(
&multi_head_attention_7/einsum_1/Einsum
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpΣ
5multi_head_attention_7/attention_output/einsum/EinsumEinsum/multi_head_attention_7/einsum_1/Einsum:output:0Lmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????# *
equationabcd,cde->abe27
5multi_head_attention_7/attention_output/einsum/Einsumψ
:multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_7/attention_output/add/ReadVariableOp
+multi_head_attention_7/attention_output/addAddV2>multi_head_attention_7/attention_output/einsum/Einsum:output:0Bmulti_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2-
+multi_head_attention_7/attention_output/add
dropout_20/IdentityIdentity/multi_head_attention_7/attention_output/add:z:0*
T0*+
_output_shapes
:?????????# 2
dropout_20/Identityo
addAddV2inputsdropout_20/Identity:output:0*
T0*+
_output_shapes
:?????????# 2
addΈ
5layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_14/moments/mean/reduction_indicesβ
#layer_normalization_14/moments/meanMeanadd:z:0>layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2%
#layer_normalization_14/moments/meanΞ
+layer_normalization_14/moments/StopGradientStopGradient,layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2-
+layer_normalization_14/moments/StopGradientξ
0layer_normalization_14/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 22
0layer_normalization_14/moments/SquaredDifferenceΐ
9layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_14/moments/variance/reduction_indices
'layer_normalization_14/moments/varianceMean4layer_normalization_14/moments/SquaredDifference:z:0Blayer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2)
'layer_normalization_14/moments/variance
&layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_14/batchnorm/add/yξ
$layer_normalization_14/batchnorm/addAddV20layer_normalization_14/moments/variance:output:0/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2&
$layer_normalization_14/batchnorm/addΉ
&layer_normalization_14/batchnorm/RsqrtRsqrt(layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2(
&layer_normalization_14/batchnorm/Rsqrtγ
3layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_14/batchnorm/mul/ReadVariableOpς
$layer_normalization_14/batchnorm/mulMul*layer_normalization_14/batchnorm/Rsqrt:y:0;layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2&
$layer_normalization_14/batchnorm/mulΐ
&layer_normalization_14/batchnorm/mul_1Muladd:z:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_14/batchnorm/mul_1ε
&layer_normalization_14/batchnorm/mul_2Mul,layer_normalization_14/moments/mean:output:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_14/batchnorm/mul_2Χ
/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_14/batchnorm/ReadVariableOpξ
$layer_normalization_14/batchnorm/subSub7layer_normalization_14/batchnorm/ReadVariableOp:value:0*layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2&
$layer_normalization_14/batchnorm/subε
&layer_normalization_14/batchnorm/add_1AddV2*layer_normalization_14/batchnorm/mul_1:z:0(layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_14/batchnorm/add_1Ψ
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
-sequential_7/dense_23/Tensordot/GatherV2/axisΏ
(sequential_7/dense_23/Tensordot/GatherV2GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/free:output:06sequential_7/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/GatherV2€
/sequential_7/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_23/Tensordot/GatherV2_1/axisΕ
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
%sequential_7/dense_23/Tensordot/ConstΨ
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
'sequential_7/dense_23/Tensordot/Const_1ΰ
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
&sequential_7/dense_23/Tensordot/concatδ
%sequential_7/dense_23/Tensordot/stackPack-sequential_7/dense_23/Tensordot/Prod:output:0/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/stackφ
)sequential_7/dense_23/Tensordot/transpose	Transpose*layer_normalization_14/batchnorm/add_1:z:0/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????# 2+
)sequential_7/dense_23/Tensordot/transposeχ
'sequential_7/dense_23/Tensordot/ReshapeReshape-sequential_7/dense_23/Tensordot/transpose:y:0.sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_7/dense_23/Tensordot/Reshapeφ
&sequential_7/dense_23/Tensordot/MatMulMatMul0sequential_7/dense_23/Tensordot/Reshape:output:06sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2(
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
(sequential_7/dense_23/Tensordot/concat_1θ
sequential_7/dense_23/TensordotReshape0sequential_7/dense_23/Tensordot/MatMul:product:01sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@2!
sequential_7/dense_23/TensordotΞ
,sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_7/dense_23/BiasAdd/ReadVariableOpί
sequential_7/dense_23/BiasAddBiasAdd(sequential_7/dense_23/Tensordot:output:04sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@2
sequential_7/dense_23/BiasAdd
sequential_7/dense_23/ReluRelu&sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#@2
sequential_7/dense_23/ReluΨ
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
-sequential_7/dense_24/Tensordot/GatherV2/axisΏ
(sequential_7/dense_24/Tensordot/GatherV2GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/free:output:06sequential_7/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/GatherV2€
/sequential_7/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_24/Tensordot/GatherV2_1/axisΕ
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
%sequential_7/dense_24/Tensordot/ConstΨ
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
'sequential_7/dense_24/Tensordot/Const_1ΰ
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
&sequential_7/dense_24/Tensordot/concatδ
%sequential_7/dense_24/Tensordot/stackPack-sequential_7/dense_24/Tensordot/Prod:output:0/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/stackτ
)sequential_7/dense_24/Tensordot/transpose	Transpose(sequential_7/dense_23/Relu:activations:0/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2+
)sequential_7/dense_24/Tensordot/transposeχ
'sequential_7/dense_24/Tensordot/ReshapeReshape-sequential_7/dense_24/Tensordot/transpose:y:0.sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_7/dense_24/Tensordot/Reshapeφ
&sequential_7/dense_24/Tensordot/MatMulMatMul0sequential_7/dense_24/Tensordot/Reshape:output:06sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
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
(sequential_7/dense_24/Tensordot/concat_1θ
sequential_7/dense_24/TensordotReshape0sequential_7/dense_24/Tensordot/MatMul:product:01sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 2!
sequential_7/dense_24/TensordotΞ
,sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_7/dense_24/BiasAdd/ReadVariableOpί
sequential_7/dense_24/BiasAddBiasAdd(sequential_7/dense_24/Tensordot:output:04sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2
sequential_7/dense_24/BiasAdd
dropout_21/IdentityIdentity&sequential_7/dense_24/BiasAdd:output:0*
T0*+
_output_shapes
:?????????# 2
dropout_21/Identity
add_1AddV2*layer_normalization_14/batchnorm/add_1:z:0dropout_21/Identity:output:0*
T0*+
_output_shapes
:?????????# 2
add_1Έ
5layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_15/moments/mean/reduction_indicesδ
#layer_normalization_15/moments/meanMean	add_1:z:0>layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2%
#layer_normalization_15/moments/meanΞ
+layer_normalization_15/moments/StopGradientStopGradient,layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2-
+layer_normalization_15/moments/StopGradientπ
0layer_normalization_15/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 22
0layer_normalization_15/moments/SquaredDifferenceΐ
9layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_15/moments/variance/reduction_indices
'layer_normalization_15/moments/varianceMean4layer_normalization_15/moments/SquaredDifference:z:0Blayer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2)
'layer_normalization_15/moments/variance
&layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_15/batchnorm/add/yξ
$layer_normalization_15/batchnorm/addAddV20layer_normalization_15/moments/variance:output:0/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2&
$layer_normalization_15/batchnorm/addΉ
&layer_normalization_15/batchnorm/RsqrtRsqrt(layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2(
&layer_normalization_15/batchnorm/Rsqrtγ
3layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_15/batchnorm/mul/ReadVariableOpς
$layer_normalization_15/batchnorm/mulMul*layer_normalization_15/batchnorm/Rsqrt:y:0;layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2&
$layer_normalization_15/batchnorm/mulΒ
&layer_normalization_15/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_15/batchnorm/mul_1ε
&layer_normalization_15/batchnorm/mul_2Mul,layer_normalization_15/moments/mean:output:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_15/batchnorm/mul_2Χ
/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_15/batchnorm/ReadVariableOpξ
$layer_normalization_15/batchnorm/subSub7layer_normalization_15/batchnorm/ReadVariableOp:value:0*layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2&
$layer_normalization_15/batchnorm/subε
&layer_normalization_15/batchnorm/add_1AddV2*layer_normalization_15/batchnorm/mul_1:z:0(layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_15/batchnorm/add_1ά
IdentityIdentity*layer_normalization_15/batchnorm/add_1:z:00^layer_normalization_14/batchnorm/ReadVariableOp4^layer_normalization_14/batchnorm/mul/ReadVariableOp0^layer_normalization_15/batchnorm/ReadVariableOp4^layer_normalization_15/batchnorm/mul/ReadVariableOp;^multi_head_attention_7/attention_output/add/ReadVariableOpE^multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_7/key/add/ReadVariableOp8^multi_head_attention_7/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/query/add/ReadVariableOp:^multi_head_attention_7/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/value/add/ReadVariableOp:^multi_head_attention_7/value/einsum/Einsum/ReadVariableOp-^sequential_7/dense_23/BiasAdd/ReadVariableOp/^sequential_7/dense_23/Tensordot/ReadVariableOp-^sequential_7/dense_24/BiasAdd/ReadVariableOp/^sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:?????????# ::::::::::::::::2b
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
:?????????# 
 
_user_specified_nameinputs
Π

ΰ
4__inference_transformer_block_7_layer_call_fn_308456

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
identity’StatefulPartitionedCallΑ
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
GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_3060612
StatefulPartitionedCall
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
Ήή
β
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_306188

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
identity’/layer_normalization_14/batchnorm/ReadVariableOp’3layer_normalization_14/batchnorm/mul/ReadVariableOp’/layer_normalization_15/batchnorm/ReadVariableOp’3layer_normalization_15/batchnorm/mul/ReadVariableOp’:multi_head_attention_7/attention_output/add/ReadVariableOp’Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp’-multi_head_attention_7/key/add/ReadVariableOp’7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp’/multi_head_attention_7/query/add/ReadVariableOp’9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp’/multi_head_attention_7/value/add/ReadVariableOp’9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp’,sequential_7/dense_23/BiasAdd/ReadVariableOp’.sequential_7/dense_23/Tensordot/ReadVariableOp’,sequential_7/dense_24/BiasAdd/ReadVariableOp’.sequential_7/dense_24/Tensordot/ReadVariableOpύ
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_7/query/einsum/EinsumEinsuminputsAmulti_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2,
*multi_head_attention_7/query/einsum/EinsumΫ
/multi_head_attention_7/query/add/ReadVariableOpReadVariableOp8multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/query/add/ReadVariableOpυ
 multi_head_attention_7/query/addAddV23multi_head_attention_7/query/einsum/Einsum:output:07multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2"
 multi_head_attention_7/query/addχ
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_7/key/einsum/EinsumEinsuminputs?multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2*
(multi_head_attention_7/key/einsum/EinsumΥ
-multi_head_attention_7/key/add/ReadVariableOpReadVariableOp6multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_7/key/add/ReadVariableOpν
multi_head_attention_7/key/addAddV21multi_head_attention_7/key/einsum/Einsum:output:05multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2 
multi_head_attention_7/key/addύ
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_7/value/einsum/EinsumEinsuminputsAmulti_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2,
*multi_head_attention_7/value/einsum/EinsumΫ
/multi_head_attention_7/value/add/ReadVariableOpReadVariableOp8multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/value/add/ReadVariableOpυ
 multi_head_attention_7/value/addAddV23multi_head_attention_7/value/einsum/Einsum:output:07multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2"
 multi_head_attention_7/value/add
multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *σ5>2
multi_head_attention_7/Mul/yΖ
multi_head_attention_7/MulMul$multi_head_attention_7/query/add:z:0%multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:?????????# 2
multi_head_attention_7/Mulό
$multi_head_attention_7/einsum/EinsumEinsum"multi_head_attention_7/key/add:z:0multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:?????????##*
equationaecd,abcd->acbe2&
$multi_head_attention_7/einsum/EinsumΔ
&multi_head_attention_7/softmax/SoftmaxSoftmax-multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????##2(
&multi_head_attention_7/softmax/SoftmaxΚ
'multi_head_attention_7/dropout/IdentityIdentity0multi_head_attention_7/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????##2)
'multi_head_attention_7/dropout/Identity
&multi_head_attention_7/einsum_1/EinsumEinsum0multi_head_attention_7/dropout/Identity:output:0$multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:?????????# *
equationacbe,aecd->abcd2(
&multi_head_attention_7/einsum_1/Einsum
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpΣ
5multi_head_attention_7/attention_output/einsum/EinsumEinsum/multi_head_attention_7/einsum_1/Einsum:output:0Lmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????# *
equationabcd,cde->abe27
5multi_head_attention_7/attention_output/einsum/Einsumψ
:multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_7/attention_output/add/ReadVariableOp
+multi_head_attention_7/attention_output/addAddV2>multi_head_attention_7/attention_output/einsum/Einsum:output:0Bmulti_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2-
+multi_head_attention_7/attention_output/add
dropout_20/IdentityIdentity/multi_head_attention_7/attention_output/add:z:0*
T0*+
_output_shapes
:?????????# 2
dropout_20/Identityo
addAddV2inputsdropout_20/Identity:output:0*
T0*+
_output_shapes
:?????????# 2
addΈ
5layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_14/moments/mean/reduction_indicesβ
#layer_normalization_14/moments/meanMeanadd:z:0>layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2%
#layer_normalization_14/moments/meanΞ
+layer_normalization_14/moments/StopGradientStopGradient,layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2-
+layer_normalization_14/moments/StopGradientξ
0layer_normalization_14/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 22
0layer_normalization_14/moments/SquaredDifferenceΐ
9layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_14/moments/variance/reduction_indices
'layer_normalization_14/moments/varianceMean4layer_normalization_14/moments/SquaredDifference:z:0Blayer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2)
'layer_normalization_14/moments/variance
&layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_14/batchnorm/add/yξ
$layer_normalization_14/batchnorm/addAddV20layer_normalization_14/moments/variance:output:0/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2&
$layer_normalization_14/batchnorm/addΉ
&layer_normalization_14/batchnorm/RsqrtRsqrt(layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2(
&layer_normalization_14/batchnorm/Rsqrtγ
3layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_14/batchnorm/mul/ReadVariableOpς
$layer_normalization_14/batchnorm/mulMul*layer_normalization_14/batchnorm/Rsqrt:y:0;layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2&
$layer_normalization_14/batchnorm/mulΐ
&layer_normalization_14/batchnorm/mul_1Muladd:z:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_14/batchnorm/mul_1ε
&layer_normalization_14/batchnorm/mul_2Mul,layer_normalization_14/moments/mean:output:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_14/batchnorm/mul_2Χ
/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_14/batchnorm/ReadVariableOpξ
$layer_normalization_14/batchnorm/subSub7layer_normalization_14/batchnorm/ReadVariableOp:value:0*layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2&
$layer_normalization_14/batchnorm/subε
&layer_normalization_14/batchnorm/add_1AddV2*layer_normalization_14/batchnorm/mul_1:z:0(layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_14/batchnorm/add_1Ψ
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
-sequential_7/dense_23/Tensordot/GatherV2/axisΏ
(sequential_7/dense_23/Tensordot/GatherV2GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/free:output:06sequential_7/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/GatherV2€
/sequential_7/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_23/Tensordot/GatherV2_1/axisΕ
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
%sequential_7/dense_23/Tensordot/ConstΨ
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
'sequential_7/dense_23/Tensordot/Const_1ΰ
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
&sequential_7/dense_23/Tensordot/concatδ
%sequential_7/dense_23/Tensordot/stackPack-sequential_7/dense_23/Tensordot/Prod:output:0/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/stackφ
)sequential_7/dense_23/Tensordot/transpose	Transpose*layer_normalization_14/batchnorm/add_1:z:0/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????# 2+
)sequential_7/dense_23/Tensordot/transposeχ
'sequential_7/dense_23/Tensordot/ReshapeReshape-sequential_7/dense_23/Tensordot/transpose:y:0.sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_7/dense_23/Tensordot/Reshapeφ
&sequential_7/dense_23/Tensordot/MatMulMatMul0sequential_7/dense_23/Tensordot/Reshape:output:06sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2(
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
(sequential_7/dense_23/Tensordot/concat_1θ
sequential_7/dense_23/TensordotReshape0sequential_7/dense_23/Tensordot/MatMul:product:01sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@2!
sequential_7/dense_23/TensordotΞ
,sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_7/dense_23/BiasAdd/ReadVariableOpί
sequential_7/dense_23/BiasAddBiasAdd(sequential_7/dense_23/Tensordot:output:04sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@2
sequential_7/dense_23/BiasAdd
sequential_7/dense_23/ReluRelu&sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#@2
sequential_7/dense_23/ReluΨ
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
-sequential_7/dense_24/Tensordot/GatherV2/axisΏ
(sequential_7/dense_24/Tensordot/GatherV2GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/free:output:06sequential_7/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/GatherV2€
/sequential_7/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_24/Tensordot/GatherV2_1/axisΕ
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
%sequential_7/dense_24/Tensordot/ConstΨ
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
'sequential_7/dense_24/Tensordot/Const_1ΰ
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
&sequential_7/dense_24/Tensordot/concatδ
%sequential_7/dense_24/Tensordot/stackPack-sequential_7/dense_24/Tensordot/Prod:output:0/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/stackτ
)sequential_7/dense_24/Tensordot/transpose	Transpose(sequential_7/dense_23/Relu:activations:0/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2+
)sequential_7/dense_24/Tensordot/transposeχ
'sequential_7/dense_24/Tensordot/ReshapeReshape-sequential_7/dense_24/Tensordot/transpose:y:0.sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_7/dense_24/Tensordot/Reshapeφ
&sequential_7/dense_24/Tensordot/MatMulMatMul0sequential_7/dense_24/Tensordot/Reshape:output:06sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
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
(sequential_7/dense_24/Tensordot/concat_1θ
sequential_7/dense_24/TensordotReshape0sequential_7/dense_24/Tensordot/MatMul:product:01sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 2!
sequential_7/dense_24/TensordotΞ
,sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_7/dense_24/BiasAdd/ReadVariableOpί
sequential_7/dense_24/BiasAddBiasAdd(sequential_7/dense_24/Tensordot:output:04sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2
sequential_7/dense_24/BiasAdd
dropout_21/IdentityIdentity&sequential_7/dense_24/BiasAdd:output:0*
T0*+
_output_shapes
:?????????# 2
dropout_21/Identity
add_1AddV2*layer_normalization_14/batchnorm/add_1:z:0dropout_21/Identity:output:0*
T0*+
_output_shapes
:?????????# 2
add_1Έ
5layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_15/moments/mean/reduction_indicesδ
#layer_normalization_15/moments/meanMean	add_1:z:0>layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2%
#layer_normalization_15/moments/meanΞ
+layer_normalization_15/moments/StopGradientStopGradient,layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2-
+layer_normalization_15/moments/StopGradientπ
0layer_normalization_15/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 22
0layer_normalization_15/moments/SquaredDifferenceΐ
9layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_15/moments/variance/reduction_indices
'layer_normalization_15/moments/varianceMean4layer_normalization_15/moments/SquaredDifference:z:0Blayer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2)
'layer_normalization_15/moments/variance
&layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_15/batchnorm/add/yξ
$layer_normalization_15/batchnorm/addAddV20layer_normalization_15/moments/variance:output:0/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2&
$layer_normalization_15/batchnorm/addΉ
&layer_normalization_15/batchnorm/RsqrtRsqrt(layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2(
&layer_normalization_15/batchnorm/Rsqrtγ
3layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_15/batchnorm/mul/ReadVariableOpς
$layer_normalization_15/batchnorm/mulMul*layer_normalization_15/batchnorm/Rsqrt:y:0;layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2&
$layer_normalization_15/batchnorm/mulΒ
&layer_normalization_15/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_15/batchnorm/mul_1ε
&layer_normalization_15/batchnorm/mul_2Mul,layer_normalization_15/moments/mean:output:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_15/batchnorm/mul_2Χ
/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_15/batchnorm/ReadVariableOpξ
$layer_normalization_15/batchnorm/subSub7layer_normalization_15/batchnorm/ReadVariableOp:value:0*layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2&
$layer_normalization_15/batchnorm/subε
&layer_normalization_15/batchnorm/add_1AddV2*layer_normalization_15/batchnorm/mul_1:z:0(layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_15/batchnorm/add_1ά
IdentityIdentity*layer_normalization_15/batchnorm/add_1:z:00^layer_normalization_14/batchnorm/ReadVariableOp4^layer_normalization_14/batchnorm/mul/ReadVariableOp0^layer_normalization_15/batchnorm/ReadVariableOp4^layer_normalization_15/batchnorm/mul/ReadVariableOp;^multi_head_attention_7/attention_output/add/ReadVariableOpE^multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_7/key/add/ReadVariableOp8^multi_head_attention_7/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/query/add/ReadVariableOp:^multi_head_attention_7/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/value/add/ReadVariableOp:^multi_head_attention_7/value/einsum/Einsum/ReadVariableOp-^sequential_7/dense_23/BiasAdd/ReadVariableOp/^sequential_7/dense_23/Tensordot/ReadVariableOp-^sequential_7/dense_24/BiasAdd/ReadVariableOp/^sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:?????????# ::::::::::::::::2b
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
:?????????# 
 
_user_specified_nameinputs
ί
~
)__inference_dense_26_layer_call_fn_308606

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallχ
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
GPU2*0J 8 *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_3064072
StatefulPartitionedCall
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
ξ	
έ
D__inference_dense_25_layer_call_and_return_conditional_losses_306350

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
Α
W
;__inference_global_average_pooling1d_3_layer_call_fn_308504

inputs
identityΧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_3063022
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????# :S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs

G
+__inference_dropout_23_layer_call_fn_308633

inputs
identityΗ
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
GPU2*0J 8 *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_3064402
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
?
β
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_306061

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
identity’/layer_normalization_14/batchnorm/ReadVariableOp’3layer_normalization_14/batchnorm/mul/ReadVariableOp’/layer_normalization_15/batchnorm/ReadVariableOp’3layer_normalization_15/batchnorm/mul/ReadVariableOp’:multi_head_attention_7/attention_output/add/ReadVariableOp’Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp’-multi_head_attention_7/key/add/ReadVariableOp’7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp’/multi_head_attention_7/query/add/ReadVariableOp’9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp’/multi_head_attention_7/value/add/ReadVariableOp’9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp’,sequential_7/dense_23/BiasAdd/ReadVariableOp’.sequential_7/dense_23/Tensordot/ReadVariableOp’,sequential_7/dense_24/BiasAdd/ReadVariableOp’.sequential_7/dense_24/Tensordot/ReadVariableOpύ
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_7/query/einsum/EinsumEinsuminputsAmulti_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2,
*multi_head_attention_7/query/einsum/EinsumΫ
/multi_head_attention_7/query/add/ReadVariableOpReadVariableOp8multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/query/add/ReadVariableOpυ
 multi_head_attention_7/query/addAddV23multi_head_attention_7/query/einsum/Einsum:output:07multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2"
 multi_head_attention_7/query/addχ
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_7/key/einsum/EinsumEinsuminputs?multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2*
(multi_head_attention_7/key/einsum/EinsumΥ
-multi_head_attention_7/key/add/ReadVariableOpReadVariableOp6multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_7/key/add/ReadVariableOpν
multi_head_attention_7/key/addAddV21multi_head_attention_7/key/einsum/Einsum:output:05multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2 
multi_head_attention_7/key/addύ
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_7/value/einsum/EinsumEinsuminputsAmulti_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2,
*multi_head_attention_7/value/einsum/EinsumΫ
/multi_head_attention_7/value/add/ReadVariableOpReadVariableOp8multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/value/add/ReadVariableOpυ
 multi_head_attention_7/value/addAddV23multi_head_attention_7/value/einsum/Einsum:output:07multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2"
 multi_head_attention_7/value/add
multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *σ5>2
multi_head_attention_7/Mul/yΖ
multi_head_attention_7/MulMul$multi_head_attention_7/query/add:z:0%multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:?????????# 2
multi_head_attention_7/Mulό
$multi_head_attention_7/einsum/EinsumEinsum"multi_head_attention_7/key/add:z:0multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:?????????##*
equationaecd,abcd->acbe2&
$multi_head_attention_7/einsum/EinsumΔ
&multi_head_attention_7/softmax/SoftmaxSoftmax-multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????##2(
&multi_head_attention_7/softmax/Softmax‘
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
:?????????##2,
*multi_head_attention_7/dropout/dropout/MulΌ
,multi_head_attention_7/dropout/dropout/ShapeShape0multi_head_attention_7/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_7/dropout/dropout/Shape
Cmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_7/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????##*
dtype02E
Cmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniform³
5multi_head_attention_7/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_7/dropout/dropout/GreaterEqual/yΒ
3multi_head_attention_7/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_7/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????##25
3multi_head_attention_7/dropout/dropout/GreaterEqualδ
+multi_head_attention_7/dropout/dropout/CastCast7multi_head_attention_7/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????##2-
+multi_head_attention_7/dropout/dropout/Castώ
,multi_head_attention_7/dropout/dropout/Mul_1Mul.multi_head_attention_7/dropout/dropout/Mul:z:0/multi_head_attention_7/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????##2.
,multi_head_attention_7/dropout/dropout/Mul_1
&multi_head_attention_7/einsum_1/EinsumEinsum0multi_head_attention_7/dropout/dropout/Mul_1:z:0$multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:?????????# *
equationacbe,aecd->abcd2(
&multi_head_attention_7/einsum_1/Einsum
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpΣ
5multi_head_attention_7/attention_output/einsum/EinsumEinsum/multi_head_attention_7/einsum_1/Einsum:output:0Lmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????# *
equationabcd,cde->abe27
5multi_head_attention_7/attention_output/einsum/Einsumψ
:multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_7/attention_output/add/ReadVariableOp
+multi_head_attention_7/attention_output/addAddV2>multi_head_attention_7/attention_output/einsum/Einsum:output:0Bmulti_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2-
+multi_head_attention_7/attention_output/addy
dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?2
dropout_20/dropout/ConstΑ
dropout_20/dropout/MulMul/multi_head_attention_7/attention_output/add:z:0!dropout_20/dropout/Const:output:0*
T0*+
_output_shapes
:?????????# 2
dropout_20/dropout/Mul
dropout_20/dropout/ShapeShape/multi_head_attention_7/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_20/dropout/ShapeΩ
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????# *
dtype021
/dropout_20/dropout/random_uniform/RandomUniform
!dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2#
!dropout_20/dropout/GreaterEqual/yξ
dropout_20/dropout/GreaterEqualGreaterEqual8dropout_20/dropout/random_uniform/RandomUniform:output:0*dropout_20/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????# 2!
dropout_20/dropout/GreaterEqual€
dropout_20/dropout/CastCast#dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????# 2
dropout_20/dropout/Castͺ
dropout_20/dropout/Mul_1Muldropout_20/dropout/Mul:z:0dropout_20/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????# 2
dropout_20/dropout/Mul_1o
addAddV2inputsdropout_20/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????# 2
addΈ
5layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_14/moments/mean/reduction_indicesβ
#layer_normalization_14/moments/meanMeanadd:z:0>layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2%
#layer_normalization_14/moments/meanΞ
+layer_normalization_14/moments/StopGradientStopGradient,layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2-
+layer_normalization_14/moments/StopGradientξ
0layer_normalization_14/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 22
0layer_normalization_14/moments/SquaredDifferenceΐ
9layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_14/moments/variance/reduction_indices
'layer_normalization_14/moments/varianceMean4layer_normalization_14/moments/SquaredDifference:z:0Blayer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2)
'layer_normalization_14/moments/variance
&layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_14/batchnorm/add/yξ
$layer_normalization_14/batchnorm/addAddV20layer_normalization_14/moments/variance:output:0/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2&
$layer_normalization_14/batchnorm/addΉ
&layer_normalization_14/batchnorm/RsqrtRsqrt(layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2(
&layer_normalization_14/batchnorm/Rsqrtγ
3layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_14/batchnorm/mul/ReadVariableOpς
$layer_normalization_14/batchnorm/mulMul*layer_normalization_14/batchnorm/Rsqrt:y:0;layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2&
$layer_normalization_14/batchnorm/mulΐ
&layer_normalization_14/batchnorm/mul_1Muladd:z:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_14/batchnorm/mul_1ε
&layer_normalization_14/batchnorm/mul_2Mul,layer_normalization_14/moments/mean:output:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_14/batchnorm/mul_2Χ
/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_14/batchnorm/ReadVariableOpξ
$layer_normalization_14/batchnorm/subSub7layer_normalization_14/batchnorm/ReadVariableOp:value:0*layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2&
$layer_normalization_14/batchnorm/subε
&layer_normalization_14/batchnorm/add_1AddV2*layer_normalization_14/batchnorm/mul_1:z:0(layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_14/batchnorm/add_1Ψ
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
-sequential_7/dense_23/Tensordot/GatherV2/axisΏ
(sequential_7/dense_23/Tensordot/GatherV2GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/free:output:06sequential_7/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/GatherV2€
/sequential_7/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_23/Tensordot/GatherV2_1/axisΕ
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
%sequential_7/dense_23/Tensordot/ConstΨ
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
'sequential_7/dense_23/Tensordot/Const_1ΰ
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
&sequential_7/dense_23/Tensordot/concatδ
%sequential_7/dense_23/Tensordot/stackPack-sequential_7/dense_23/Tensordot/Prod:output:0/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/stackφ
)sequential_7/dense_23/Tensordot/transpose	Transpose*layer_normalization_14/batchnorm/add_1:z:0/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????# 2+
)sequential_7/dense_23/Tensordot/transposeχ
'sequential_7/dense_23/Tensordot/ReshapeReshape-sequential_7/dense_23/Tensordot/transpose:y:0.sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_7/dense_23/Tensordot/Reshapeφ
&sequential_7/dense_23/Tensordot/MatMulMatMul0sequential_7/dense_23/Tensordot/Reshape:output:06sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2(
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
(sequential_7/dense_23/Tensordot/concat_1θ
sequential_7/dense_23/TensordotReshape0sequential_7/dense_23/Tensordot/MatMul:product:01sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@2!
sequential_7/dense_23/TensordotΞ
,sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_7/dense_23/BiasAdd/ReadVariableOpί
sequential_7/dense_23/BiasAddBiasAdd(sequential_7/dense_23/Tensordot:output:04sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@2
sequential_7/dense_23/BiasAdd
sequential_7/dense_23/ReluRelu&sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#@2
sequential_7/dense_23/ReluΨ
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
-sequential_7/dense_24/Tensordot/GatherV2/axisΏ
(sequential_7/dense_24/Tensordot/GatherV2GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/free:output:06sequential_7/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/GatherV2€
/sequential_7/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_24/Tensordot/GatherV2_1/axisΕ
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
%sequential_7/dense_24/Tensordot/ConstΨ
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
'sequential_7/dense_24/Tensordot/Const_1ΰ
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
&sequential_7/dense_24/Tensordot/concatδ
%sequential_7/dense_24/Tensordot/stackPack-sequential_7/dense_24/Tensordot/Prod:output:0/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/stackτ
)sequential_7/dense_24/Tensordot/transpose	Transpose(sequential_7/dense_23/Relu:activations:0/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2+
)sequential_7/dense_24/Tensordot/transposeχ
'sequential_7/dense_24/Tensordot/ReshapeReshape-sequential_7/dense_24/Tensordot/transpose:y:0.sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_7/dense_24/Tensordot/Reshapeφ
&sequential_7/dense_24/Tensordot/MatMulMatMul0sequential_7/dense_24/Tensordot/Reshape:output:06sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
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
(sequential_7/dense_24/Tensordot/concat_1θ
sequential_7/dense_24/TensordotReshape0sequential_7/dense_24/Tensordot/MatMul:product:01sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 2!
sequential_7/dense_24/TensordotΞ
,sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_7/dense_24/BiasAdd/ReadVariableOpί
sequential_7/dense_24/BiasAddBiasAdd(sequential_7/dense_24/Tensordot:output:04sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2
sequential_7/dense_24/BiasAddy
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?2
dropout_21/dropout/ConstΈ
dropout_21/dropout/MulMul&sequential_7/dense_24/BiasAdd:output:0!dropout_21/dropout/Const:output:0*
T0*+
_output_shapes
:?????????# 2
dropout_21/dropout/Mul
dropout_21/dropout/ShapeShape&sequential_7/dense_24/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_21/dropout/ShapeΩ
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????# *
dtype021
/dropout_21/dropout/random_uniform/RandomUniform
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2#
!dropout_21/dropout/GreaterEqual/yξ
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????# 2!
dropout_21/dropout/GreaterEqual€
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????# 2
dropout_21/dropout/Castͺ
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????# 2
dropout_21/dropout/Mul_1
add_1AddV2*layer_normalization_14/batchnorm/add_1:z:0dropout_21/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????# 2
add_1Έ
5layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_15/moments/mean/reduction_indicesδ
#layer_normalization_15/moments/meanMean	add_1:z:0>layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2%
#layer_normalization_15/moments/meanΞ
+layer_normalization_15/moments/StopGradientStopGradient,layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2-
+layer_normalization_15/moments/StopGradientπ
0layer_normalization_15/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 22
0layer_normalization_15/moments/SquaredDifferenceΐ
9layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_15/moments/variance/reduction_indices
'layer_normalization_15/moments/varianceMean4layer_normalization_15/moments/SquaredDifference:z:0Blayer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2)
'layer_normalization_15/moments/variance
&layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_15/batchnorm/add/yξ
$layer_normalization_15/batchnorm/addAddV20layer_normalization_15/moments/variance:output:0/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2&
$layer_normalization_15/batchnorm/addΉ
&layer_normalization_15/batchnorm/RsqrtRsqrt(layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2(
&layer_normalization_15/batchnorm/Rsqrtγ
3layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_15/batchnorm/mul/ReadVariableOpς
$layer_normalization_15/batchnorm/mulMul*layer_normalization_15/batchnorm/Rsqrt:y:0;layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2&
$layer_normalization_15/batchnorm/mulΒ
&layer_normalization_15/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_15/batchnorm/mul_1ε
&layer_normalization_15/batchnorm/mul_2Mul,layer_normalization_15/moments/mean:output:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_15/batchnorm/mul_2Χ
/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_15/batchnorm/ReadVariableOpξ
$layer_normalization_15/batchnorm/subSub7layer_normalization_15/batchnorm/ReadVariableOp:value:0*layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2&
$layer_normalization_15/batchnorm/subε
&layer_normalization_15/batchnorm/add_1AddV2*layer_normalization_15/batchnorm/mul_1:z:0(layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_15/batchnorm/add_1ά
IdentityIdentity*layer_normalization_15/batchnorm/add_1:z:00^layer_normalization_14/batchnorm/ReadVariableOp4^layer_normalization_14/batchnorm/mul/ReadVariableOp0^layer_normalization_15/batchnorm/ReadVariableOp4^layer_normalization_15/batchnorm/mul/ReadVariableOp;^multi_head_attention_7/attention_output/add/ReadVariableOpE^multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_7/key/add/ReadVariableOp8^multi_head_attention_7/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/query/add/ReadVariableOp:^multi_head_attention_7/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/value/add/ReadVariableOp:^multi_head_attention_7/value/einsum/Einsum/ReadVariableOp-^sequential_7/dense_23/BiasAdd/ReadVariableOp/^sequential_7/dense_23/Tensordot/ReadVariableOp-^sequential_7/dense_24/BiasAdd/ReadVariableOp/^sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:?????????# ::::::::::::::::2b
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
:?????????# 
 
_user_specified_nameinputs
Φ
€
(__inference_model_3_layer_call_fn_307643
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
identity’StatefulPartitionedCall?
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
:?????????*B
_read_only_resource_inputs$
" 
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_3066742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Μ
_input_shapesΊ
·:?????????R:?????????::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:?????????R
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
σ0
Θ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_307840

inputs
assignmovingavg_307815
assignmovingavg_1_307821)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOp
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
 :?????????????????? 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
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
moments/Squeeze_1Μ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/307815*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_307815*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpρ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/307815*
_output_shapes
: 2
AssignMovingAvg/subθ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/307815*
_output_shapes
: 2
AssignMovingAvg/mul―
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_307815AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/307815*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/307821*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_307821*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpϋ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/307821*
_output_shapes
: 2
AssignMovingAvg_1/subς
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/307821*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_307821AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/307821*
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
 :?????????????????? 2
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
 :?????????????????? 2
batchnorm/add_1ΐ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????????????? ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
«
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_308521

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:????????? 2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
ι

H__inference_sequential_7_layer_call_and_return_conditional_losses_305547

inputs
dense_23_305536
dense_23_305538
dense_24_305541
dense_24_305543
identity’ dense_23/StatefulPartitionedCall’ dense_24/StatefulPartitionedCall
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinputsdense_23_305536dense_23_305538*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_3054532"
 dense_23/StatefulPartitionedCallΎ
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_305541dense_24_305543*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_3054992"
 dense_24/StatefulPartitionedCallΗ
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
Ι
d
F__inference_dropout_22_layer_call_and_return_conditional_losses_308576

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


H__inference_sequential_7_layer_call_and_return_conditional_losses_305530
dense_23_input
dense_23_305519
dense_23_305521
dense_24_305524
dense_24_305526
identity’ dense_23/StatefulPartitionedCall’ dense_24/StatefulPartitionedCall£
 dense_23/StatefulPartitionedCallStatefulPartitionedCalldense_23_inputdense_23_305519dense_23_305521*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_3054532"
 dense_23/StatefulPartitionedCallΎ
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_305524dense_24_305526*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_3054992"
 dense_24/StatefulPartitionedCallΗ
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:[ W
+
_output_shapes
:?????????# 
(
_user_specified_namedense_23_input
ο
~
)__inference_dense_23_layer_call_fn_308832

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallϋ
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
GPU2*0J 8 *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_3054532
StatefulPartitionedCall
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
ά
r
V__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_306302

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????# :S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
Ρ
γ
D__inference_dense_24_layer_call_and_return_conditional_losses_308862

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Tensordot/ReadVariableOp
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
Tensordot/GatherV2/axisΡ
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
Tensordot/GatherV2_1/axisΧ
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
:?????????#@2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape
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
:?????????# 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2	
BiasAdd
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
σ
~
)__inference_conv1d_6_layer_call_fn_307779

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallό
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_3056652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????R 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????R ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????R 
 
_user_specified_nameinputs
ξ	
έ
D__inference_dense_26_layer_call_and_return_conditional_losses_306407

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu
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
θ

Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_305862

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOp
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
:?????????# 2
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
:?????????# 2
batchnorm/add_1ί
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs

χ
D__inference_conv1d_7_layer_call_and_return_conditional_losses_307795

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ή 2
conv1d/ExpandDimsΈ
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
:?????????ή *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:?????????ή *
squeeze_dims

ύ????????2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ή 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????ή 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:?????????ή 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????ή ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????ή 
 
_user_specified_nameinputs

P
4__inference_average_pooling1d_9_layer_call_fn_305108

inputs
identityζ
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
GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_3051022
PartitionedCall
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
σ0
Θ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_308004

inputs
assignmovingavg_307979
assignmovingavg_1_307985)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOp
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
 :?????????????????? 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
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
moments/Squeeze_1Μ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/307979*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_307979*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpρ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/307979*
_output_shapes
: 2
AssignMovingAvg/subθ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/307979*
_output_shapes
: 2
AssignMovingAvg/mul―
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_307979AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/307979*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/307985*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_307985*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpϋ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/307985*
_output_shapes
: 2
AssignMovingAvg_1/subς
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/307985*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_307985AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/307985*
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
 :?????????????????? 2
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
 :?????????????????? 2
batchnorm/add_1ΐ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????????????? ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
ο
~
)__inference_dense_24_layer_call_fn_308871

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallϋ
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
GPU2*0J 8 *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_3054992
StatefulPartitionedCall
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
ά
r
V__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_308499

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????# :S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
θ

Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_305771

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOp
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
:?????????# 2
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
:?????????# 2
batchnorm/add_1ί
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
«
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_306315

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:????????? 2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs

Q
5__inference_average_pooling1d_11_layer_call_fn_305138

inputs
identityη
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_3051322
PartitionedCall
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
	
έ
D__inference_dense_27_layer_call_and_return_conditional_losses_306463

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd
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


Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_305407

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOp
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
 :?????????????????? 2
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
 :?????????????????? 2
batchnorm/add_1θ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????????????? ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
·
k
A__inference_add_3_layer_call_and_return_conditional_losses_305904

inputs
inputs_1
identity[
addAddV2inputsinputs_1*
T0*+
_output_shapes
:?????????# 2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:?????????# :?????????# :S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
Π

ΰ
4__inference_transformer_block_7_layer_call_fn_308493

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
identity’StatefulPartitionedCallΑ
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
GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_3061882
StatefulPartitionedCall
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
Ύ
u
I__inference_concatenate_3_layer_call_and_return_conditional_losses_308533
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????(2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????? :?????????:Q M
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
ΪΖ
1
"__inference__traced_restore_309349
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
identity_75’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_36’AssignVariableOp_37’AssignVariableOp_38’AssignVariableOp_39’AssignVariableOp_4’AssignVariableOp_40’AssignVariableOp_41’AssignVariableOp_42’AssignVariableOp_43’AssignVariableOp_44’AssignVariableOp_45’AssignVariableOp_46’AssignVariableOp_47’AssignVariableOp_48’AssignVariableOp_49’AssignVariableOp_5’AssignVariableOp_50’AssignVariableOp_51’AssignVariableOp_52’AssignVariableOp_53’AssignVariableOp_54’AssignVariableOp_55’AssignVariableOp_56’AssignVariableOp_57’AssignVariableOp_58’AssignVariableOp_59’AssignVariableOp_6’AssignVariableOp_60’AssignVariableOp_61’AssignVariableOp_62’AssignVariableOp_63’AssignVariableOp_64’AssignVariableOp_65’AssignVariableOp_66’AssignVariableOp_67’AssignVariableOp_68’AssignVariableOp_69’AssignVariableOp_7’AssignVariableOp_70’AssignVariableOp_71’AssignVariableOp_72’AssignVariableOp_73’AssignVariableOp_8’AssignVariableOp_9ι%
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*υ$
valueλ$Bθ$KB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names§
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*«
value‘BKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices₯
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Β
_output_shapes―
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

Identity_1₯
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

Identity_3₯
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

Identity_6Ή
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
Identity_11Α
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
Identity_18‘
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
Identity_20€
AssignVariableOp_20AssignVariableOpassignvariableop_20_momentumIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_21€
AssignVariableOp_21AssignVariableOpassignvariableop_21_sgd_iterIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ρ
AssignVariableOp_22AssignVariableOpIassignvariableop_22_token_and_position_embedding_3_embedding_6_embeddingsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ρ
AssignVariableOp_23AssignVariableOpIassignvariableop_23_token_and_position_embedding_3_embedding_7_embeddingsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Σ
AssignVariableOp_24AssignVariableOpKassignvariableop_24_transformer_block_7_multi_head_attention_7_query_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ρ
AssignVariableOp_25AssignVariableOpIassignvariableop_25_transformer_block_7_multi_head_attention_7_query_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ρ
AssignVariableOp_26AssignVariableOpIassignvariableop_26_transformer_block_7_multi_head_attention_7_key_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ο
AssignVariableOp_27AssignVariableOpGassignvariableop_27_transformer_block_7_multi_head_attention_7_key_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Σ
AssignVariableOp_28AssignVariableOpKassignvariableop_28_transformer_block_7_multi_head_attention_7_value_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ρ
AssignVariableOp_29AssignVariableOpIassignvariableop_29_transformer_block_7_multi_head_attention_7_value_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30ή
AssignVariableOp_30AssignVariableOpVassignvariableop_30_transformer_block_7_multi_head_attention_7_attention_output_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31ά
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
Identity_36Μ
AssignVariableOp_36AssignVariableOpDassignvariableop_36_transformer_block_7_layer_normalization_14_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Λ
AssignVariableOp_37AssignVariableOpCassignvariableop_37_transformer_block_7_layer_normalization_14_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Μ
AssignVariableOp_38AssignVariableOpDassignvariableop_38_transformer_block_7_layer_normalization_15_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Λ
AssignVariableOp_39AssignVariableOpCassignvariableop_39_transformer_block_7_layer_normalization_15_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40‘
AssignVariableOp_40AssignVariableOpassignvariableop_40_totalIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41‘
AssignVariableOp_41AssignVariableOpassignvariableop_41_countIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Έ
AssignVariableOp_42AssignVariableOp0assignvariableop_42_sgd_conv1d_6_kernel_momentumIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Ά
AssignVariableOp_43AssignVariableOp.assignvariableop_43_sgd_conv1d_6_bias_momentumIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Έ
AssignVariableOp_44AssignVariableOp0assignvariableop_44_sgd_conv1d_7_kernel_momentumIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Ά
AssignVariableOp_45AssignVariableOp.assignvariableop_45_sgd_conv1d_7_bias_momentumIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Δ
AssignVariableOp_46AssignVariableOp<assignvariableop_46_sgd_batch_normalization_6_gamma_momentumIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Γ
AssignVariableOp_47AssignVariableOp;assignvariableop_47_sgd_batch_normalization_6_beta_momentumIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Δ
AssignVariableOp_48AssignVariableOp<assignvariableop_48_sgd_batch_normalization_7_gamma_momentumIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Γ
AssignVariableOp_49AssignVariableOp;assignvariableop_49_sgd_batch_normalization_7_beta_momentumIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Έ
AssignVariableOp_50AssignVariableOp0assignvariableop_50_sgd_dense_25_kernel_momentumIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Ά
AssignVariableOp_51AssignVariableOp.assignvariableop_51_sgd_dense_25_bias_momentumIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Έ
AssignVariableOp_52AssignVariableOp0assignvariableop_52_sgd_dense_26_kernel_momentumIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Ά
AssignVariableOp_53AssignVariableOp.assignvariableop_53_sgd_dense_26_bias_momentumIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54Έ
AssignVariableOp_54AssignVariableOp0assignvariableop_54_sgd_dense_27_kernel_momentumIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55Ά
AssignVariableOp_55AssignVariableOp.assignvariableop_55_sgd_dense_27_bias_momentumIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56ή
AssignVariableOp_56AssignVariableOpVassignvariableop_56_sgd_token_and_position_embedding_3_embedding_6_embeddings_momentumIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57ή
AssignVariableOp_57AssignVariableOpVassignvariableop_57_sgd_token_and_position_embedding_3_embedding_7_embeddings_momentumIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58ΰ
AssignVariableOp_58AssignVariableOpXassignvariableop_58_sgd_transformer_block_7_multi_head_attention_7_query_kernel_momentumIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59ή
AssignVariableOp_59AssignVariableOpVassignvariableop_59_sgd_transformer_block_7_multi_head_attention_7_query_bias_momentumIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60ή
AssignVariableOp_60AssignVariableOpVassignvariableop_60_sgd_transformer_block_7_multi_head_attention_7_key_kernel_momentumIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61ά
AssignVariableOp_61AssignVariableOpTassignvariableop_61_sgd_transformer_block_7_multi_head_attention_7_key_bias_momentumIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62ΰ
AssignVariableOp_62AssignVariableOpXassignvariableop_62_sgd_transformer_block_7_multi_head_attention_7_value_kernel_momentumIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63ή
AssignVariableOp_63AssignVariableOpVassignvariableop_63_sgd_transformer_block_7_multi_head_attention_7_value_bias_momentumIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64λ
AssignVariableOp_64AssignVariableOpcassignvariableop_64_sgd_transformer_block_7_multi_head_attention_7_attention_output_kernel_momentumIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65ι
AssignVariableOp_65AssignVariableOpaassignvariableop_65_sgd_transformer_block_7_multi_head_attention_7_attention_output_bias_momentumIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66Έ
AssignVariableOp_66AssignVariableOp0assignvariableop_66_sgd_dense_23_kernel_momentumIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67Ά
AssignVariableOp_67AssignVariableOp.assignvariableop_67_sgd_dense_23_bias_momentumIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68Έ
AssignVariableOp_68AssignVariableOp0assignvariableop_68_sgd_dense_24_kernel_momentumIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69Ά
AssignVariableOp_69AssignVariableOp.assignvariableop_69_sgd_dense_24_bias_momentumIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Ω
AssignVariableOp_70AssignVariableOpQassignvariableop_70_sgd_transformer_block_7_layer_normalization_14_gamma_momentumIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71Ψ
AssignVariableOp_71AssignVariableOpPassignvariableop_71_sgd_transformer_block_7_layer_normalization_14_beta_momentumIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72Ω
AssignVariableOp_72AssignVariableOpQassignvariableop_72_sgd_transformer_block_7_layer_normalization_15_gamma_momentumIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73Ψ
AssignVariableOp_73AssignVariableOpPassignvariableop_73_sgd_transformer_block_7_layer_normalization_15_beta_momentumIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_739
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpΊ
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
identity_75Identity_75:output:0*Ώ
_input_shapes­
ͺ: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
ί
~
)__inference_dense_25_layer_call_fn_308559

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallχ
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
GPU2*0J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_3063502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?

$__inference_signature_wrapper_307008
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
identity’StatefulPartitionedCall²
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
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_3050932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Μ
_input_shapesΊ
·:?????????R:?????????::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:?????????R
!
_user_specified_name	input_7:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_8
΅
s
I__inference_concatenate_3_layer_call_and_return_conditional_losses_306330

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????(2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????? :?????????:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ξ	
έ
D__inference_dense_25_layer_call_and_return_conditional_losses_308550

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs

G
+__inference_dropout_22_layer_call_fn_308586

inputs
identityΗ
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
GPU2*0J 8 *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_3063832
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
Φν
€&
C__inference_model_3_layer_call_and_return_conditional_losses_307320
inputs_0
inputs_1F
Btoken_and_position_embedding_3_embedding_7_embedding_lookup_307020F
Btoken_and_position_embedding_3_embedding_6_embedding_lookup_3070268
4conv1d_6_conv1d_expanddims_1_readvariableop_resource,
(conv1d_6_biasadd_readvariableop_resource8
4conv1d_7_conv1d_expanddims_1_readvariableop_resource,
(conv1d_7_biasadd_readvariableop_resource0
,batch_normalization_6_assignmovingavg_3070762
.batch_normalization_6_assignmovingavg_1_307082?
;batch_normalization_6_batchnorm_mul_readvariableop_resource;
7batch_normalization_6_batchnorm_readvariableop_resource0
,batch_normalization_7_assignmovingavg_3071082
.batch_normalization_7_assignmovingavg_1_307114?
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
identity’9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp’4batch_normalization_6/AssignMovingAvg/ReadVariableOp’;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp’6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp’.batch_normalization_6/batchnorm/ReadVariableOp’2batch_normalization_6/batchnorm/mul/ReadVariableOp’9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp’4batch_normalization_7/AssignMovingAvg/ReadVariableOp’;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp’6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp’.batch_normalization_7/batchnorm/ReadVariableOp’2batch_normalization_7/batchnorm/mul/ReadVariableOp’conv1d_6/BiasAdd/ReadVariableOp’+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp’conv1d_7/BiasAdd/ReadVariableOp’+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp’dense_25/BiasAdd/ReadVariableOp’dense_25/MatMul/ReadVariableOp’dense_26/BiasAdd/ReadVariableOp’dense_26/MatMul/ReadVariableOp’dense_27/BiasAdd/ReadVariableOp’dense_27/MatMul/ReadVariableOp’;token_and_position_embedding_3/embedding_6/embedding_lookup’;token_and_position_embedding_3/embedding_7/embedding_lookup’Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp’Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp’Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp’Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp’Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp’Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp’Atransformer_block_7/multi_head_attention_7/key/add/ReadVariableOp’Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp’Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOp’Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp’Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOp’Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp’@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp’Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp’@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp’Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp
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
?????????24
2token_and_position_embedding_3/strided_slice/stackΆ
4token_and_position_embedding_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_3/strided_slice/stack_1Ά
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
:?????????2&
$token_and_position_embedding_3/rangeΚ
;token_and_position_embedding_3/embedding_7/embedding_lookupResourceGatherBtoken_and_position_embedding_3_embedding_7_embedding_lookup_307020-token_and_position_embedding_3/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_7/embedding_lookup/307020*'
_output_shapes
:????????? *
dtype02=
;token_and_position_embedding_3/embedding_7/embedding_lookup
Dtoken_and_position_embedding_3/embedding_7/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_3/embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_7/embedding_lookup/307020*'
_output_shapes
:????????? 2F
Dtoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity
Ftoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2H
Ftoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1Ά
/token_and_position_embedding_3/embedding_6/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:?????????R21
/token_and_position_embedding_3/embedding_6/CastΥ
;token_and_position_embedding_3/embedding_6/embedding_lookupResourceGatherBtoken_and_position_embedding_3_embedding_6_embedding_lookup_3070263token_and_position_embedding_3/embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_6/embedding_lookup/307026*,
_output_shapes
:?????????R *
dtype02=
;token_and_position_embedding_3/embedding_6/embedding_lookup
Dtoken_and_position_embedding_3/embedding_6/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_3/embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_6/embedding_lookup/307026*,
_output_shapes
:?????????R 2F
Dtoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity’
Ftoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:?????????R 2H
Ftoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1ͺ
"token_and_position_embedding_3/addAddV2Otoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:?????????R 2$
"token_and_position_embedding_3/add
conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2 
conv1d_6/conv1d/ExpandDims/dim?
conv1d_6/conv1d/ExpandDims
ExpandDims&token_and_position_embedding_3/add:z:0'conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????R 2
conv1d_6/conv1d/ExpandDimsΣ
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
 conv1d_6/conv1d/ExpandDims_1/dimΫ
conv1d_6/conv1d/ExpandDims_1
ExpandDims3conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_6/conv1d/ExpandDims_1Ϋ
conv1d_6/conv1dConv2D#conv1d_6/conv1d/ExpandDims:output:0%conv1d_6/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????R *
paddingSAME*
strides
2
conv1d_6/conv1d?
conv1d_6/conv1d/SqueezeSqueezeconv1d_6/conv1d:output:0*
T0*,
_output_shapes
:?????????R *
squeeze_dims

ύ????????2
conv1d_6/conv1d/Squeeze§
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_6/BiasAdd/ReadVariableOp±
conv1d_6/BiasAddBiasAdd conv1d_6/conv1d/Squeeze:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????R 2
conv1d_6/BiasAddx
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*,
_output_shapes
:?????????R 2
conv1d_6/Relu
"average_pooling1d_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_9/ExpandDims/dimΣ
average_pooling1d_9/ExpandDims
ExpandDimsconv1d_6/Relu:activations:0+average_pooling1d_9/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????R 2 
average_pooling1d_9/ExpandDimsε
average_pooling1d_9/AvgPoolAvgPool'average_pooling1d_9/ExpandDims:output:0*
T0*0
_output_shapes
:?????????ή *
ksize
*
paddingVALID*
strides
2
average_pooling1d_9/AvgPoolΉ
average_pooling1d_9/SqueezeSqueeze$average_pooling1d_9/AvgPool:output:0*
T0*,
_output_shapes
:?????????ή *
squeeze_dims
2
average_pooling1d_9/Squeeze
conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2 
conv1d_7/conv1d/ExpandDims/dimΠ
conv1d_7/conv1d/ExpandDims
ExpandDims$average_pooling1d_9/Squeeze:output:0'conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ή 2
conv1d_7/conv1d/ExpandDimsΣ
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
 conv1d_7/conv1d/ExpandDims_1/dimΫ
conv1d_7/conv1d/ExpandDims_1
ExpandDims3conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d_7/conv1d/ExpandDims_1Ϋ
conv1d_7/conv1dConv2D#conv1d_7/conv1d/ExpandDims:output:0%conv1d_7/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????ή *
paddingSAME*
strides
2
conv1d_7/conv1d?
conv1d_7/conv1d/SqueezeSqueezeconv1d_7/conv1d:output:0*
T0*,
_output_shapes
:?????????ή *
squeeze_dims

ύ????????2
conv1d_7/conv1d/Squeeze§
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_7/BiasAdd/ReadVariableOp±
conv1d_7/BiasAddBiasAdd conv1d_7/conv1d/Squeeze:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ή 2
conv1d_7/BiasAddx
conv1d_7/ReluReluconv1d_7/BiasAdd:output:0*
T0*,
_output_shapes
:?????????ή 2
conv1d_7/Relu
#average_pooling1d_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_11/ExpandDims/dimα
average_pooling1d_11/ExpandDims
ExpandDims&token_and_position_embedding_3/add:z:0,average_pooling1d_11/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????R 2!
average_pooling1d_11/ExpandDimsι
average_pooling1d_11/AvgPoolAvgPool(average_pooling1d_11/ExpandDims:output:0*
T0*/
_output_shapes
:?????????# *
ksize	
¬*
paddingVALID*
strides	
¬2
average_pooling1d_11/AvgPool»
average_pooling1d_11/SqueezeSqueeze%average_pooling1d_11/AvgPool:output:0*
T0*+
_output_shapes
:?????????# *
squeeze_dims
2
average_pooling1d_11/Squeeze
#average_pooling1d_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_10/ExpandDims/dimΦ
average_pooling1d_10/ExpandDims
ExpandDimsconv1d_7/Relu:activations:0,average_pooling1d_10/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ή 2!
average_pooling1d_10/ExpandDimsη
average_pooling1d_10/AvgPoolAvgPool(average_pooling1d_10/ExpandDims:output:0*
T0*/
_output_shapes
:?????????# *
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
:?????????# *
squeeze_dims
2
average_pooling1d_10/Squeeze½
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_6/moments/mean/reduction_indicesτ
"batch_normalization_6/moments/meanMean%average_pooling1d_10/Squeeze:output:0=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_6/moments/meanΒ
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_6/moments/StopGradient
/batch_normalization_6/moments/SquaredDifferenceSquaredDifference%average_pooling1d_10/Squeeze:output:03batch_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 21
/batch_normalization_6/moments/SquaredDifferenceΕ
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
&batch_normalization_6/moments/varianceΓ
%batch_normalization_6/moments/SqueezeSqueeze+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_6/moments/SqueezeΛ
'batch_normalization_6/moments/Squeeze_1Squeeze/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_6/moments/Squeeze_1
+batch_normalization_6/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg/307076*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2-
+batch_normalization_6/AssignMovingAvg/decayΥ
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_6_assignmovingavg_307076*
_output_shapes
: *
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOpί
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg/307076*
_output_shapes
: 2+
)batch_normalization_6/AssignMovingAvg/subΦ
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:04batch_normalization_6/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg/307076*
_output_shapes
: 2+
)batch_normalization_6/AssignMovingAvg/mul³
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_6_assignmovingavg_307076-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg/307076*
_output_shapes
 *
dtype02;
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_6/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/307082*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2/
-batch_normalization_6/AssignMovingAvg_1/decayΫ
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_6_assignmovingavg_1_307082*
_output_shapes
: *
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpι
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/307082*
_output_shapes
: 2-
+batch_normalization_6/AssignMovingAvg_1/subΰ
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:06batch_normalization_6/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/307082*
_output_shapes
: 2-
+batch_normalization_6/AssignMovingAvg_1/mulΏ
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_6_assignmovingavg_1_307082/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/307082*
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
%batch_normalization_6/batchnorm/add/yΪ
#batch_normalization_6/batchnorm/addAddV20batch_normalization_6/moments/Squeeze_1:output:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_6/batchnorm/add₯
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_6/batchnorm/Rsqrtΰ
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_6/batchnorm/mul/ReadVariableOpέ
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_6/batchnorm/mulΫ
%batch_normalization_6/batchnorm/mul_1Mul%average_pooling1d_10/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%batch_normalization_6/batchnorm/mul_1Σ
%batch_normalization_6/batchnorm/mul_2Mul.batch_normalization_6/moments/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_6/batchnorm/mul_2Τ
.batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_6/batchnorm/ReadVariableOpΩ
#batch_normalization_6/batchnorm/subSub6batch_normalization_6/batchnorm/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_6/batchnorm/subα
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2'
%batch_normalization_6/batchnorm/add_1½
4batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_7/moments/mean/reduction_indicesτ
"batch_normalization_7/moments/meanMean%average_pooling1d_11/Squeeze:output:0=batch_normalization_7/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_7/moments/meanΒ
*batch_normalization_7/moments/StopGradientStopGradient+batch_normalization_7/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_7/moments/StopGradient
/batch_normalization_7/moments/SquaredDifferenceSquaredDifference%average_pooling1d_11/Squeeze:output:03batch_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 21
/batch_normalization_7/moments/SquaredDifferenceΕ
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
&batch_normalization_7/moments/varianceΓ
%batch_normalization_7/moments/SqueezeSqueeze+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_7/moments/SqueezeΛ
'batch_normalization_7/moments/Squeeze_1Squeeze/batch_normalization_7/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_7/moments/Squeeze_1
+batch_normalization_7/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg/307108*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2-
+batch_normalization_7/AssignMovingAvg/decayΥ
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_7_assignmovingavg_307108*
_output_shapes
: *
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOpί
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg/307108*
_output_shapes
: 2+
)batch_normalization_7/AssignMovingAvg/subΦ
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:04batch_normalization_7/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg/307108*
_output_shapes
: 2+
)batch_normalization_7/AssignMovingAvg/mul³
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_7_assignmovingavg_307108-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg/307108*
_output_shapes
 *
dtype02;
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_7/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/307114*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2/
-batch_normalization_7/AssignMovingAvg_1/decayΫ
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_7_assignmovingavg_1_307114*
_output_shapes
: *
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpι
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/307114*
_output_shapes
: 2-
+batch_normalization_7/AssignMovingAvg_1/subΰ
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:06batch_normalization_7/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/307114*
_output_shapes
: 2-
+batch_normalization_7/AssignMovingAvg_1/mulΏ
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_7_assignmovingavg_1_307114/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/307114*
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
%batch_normalization_7/batchnorm/add/yΪ
#batch_normalization_7/batchnorm/addAddV20batch_normalization_7/moments/Squeeze_1:output:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_7/batchnorm/add₯
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_7/batchnorm/Rsqrtΰ
2batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_7/batchnorm/mul/ReadVariableOpέ
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:0:batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_7/batchnorm/mulΫ
%batch_normalization_7/batchnorm/mul_1Mul%average_pooling1d_11/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%batch_normalization_7/batchnorm/mul_1Σ
%batch_normalization_7/batchnorm/mul_2Mul.batch_normalization_7/moments/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_7/batchnorm/mul_2Τ
.batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_7/batchnorm/ReadVariableOpΩ
#batch_normalization_7/batchnorm/subSub6batch_normalization_7/batchnorm/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_7/batchnorm/subα
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2'
%batch_normalization_7/batchnorm/add_1«
	add_3/addAddV2)batch_normalization_6/batchnorm/add_1:z:0)batch_normalization_7/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????# 2
	add_3/addΉ
Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_7_multi_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpΠ
>transformer_block_7/multi_head_attention_7/query/einsum/EinsumEinsumadd_3/add:z:0Utransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2@
>transformer_block_7/multi_head_attention_7/query/einsum/Einsum
Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOpReadVariableOpLtransformer_block_7_multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOpΕ
4transformer_block_7/multi_head_attention_7/query/addAddV2Gtransformer_block_7/multi_head_attention_7/query/einsum/Einsum:output:0Ktransformer_block_7/multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 26
4transformer_block_7/multi_head_attention_7/query/add³
Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_7_multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpΚ
<transformer_block_7/multi_head_attention_7/key/einsum/EinsumEinsumadd_3/add:z:0Stransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
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
:?????????# 24
2transformer_block_7/multi_head_attention_7/key/addΉ
Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_7_multi_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpΠ
>transformer_block_7/multi_head_attention_7/value/einsum/EinsumEinsumadd_3/add:z:0Utransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2@
>transformer_block_7/multi_head_attention_7/value/einsum/Einsum
Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOpReadVariableOpLtransformer_block_7_multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOpΕ
4transformer_block_7/multi_head_attention_7/value/addAddV2Gtransformer_block_7/multi_head_attention_7/value/einsum/Einsum:output:0Ktransformer_block_7/multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 26
4transformer_block_7/multi_head_attention_7/value/add©
0transformer_block_7/multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *σ5>22
0transformer_block_7/multi_head_attention_7/Mul/y
.transformer_block_7/multi_head_attention_7/MulMul8transformer_block_7/multi_head_attention_7/query/add:z:09transformer_block_7/multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:?????????# 20
.transformer_block_7/multi_head_attention_7/MulΜ
8transformer_block_7/multi_head_attention_7/einsum/EinsumEinsum6transformer_block_7/multi_head_attention_7/key/add:z:02transformer_block_7/multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:?????????##*
equationaecd,abcd->acbe2:
8transformer_block_7/multi_head_attention_7/einsum/Einsum
:transformer_block_7/multi_head_attention_7/softmax/SoftmaxSoftmaxAtransformer_block_7/multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????##2<
:transformer_block_7/multi_head_attention_7/softmax/SoftmaxΙ
@transformer_block_7/multi_head_attention_7/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2B
@transformer_block_7/multi_head_attention_7/dropout/dropout/Const?
>transformer_block_7/multi_head_attention_7/dropout/dropout/MulMulDtransformer_block_7/multi_head_attention_7/softmax/Softmax:softmax:0Itransformer_block_7/multi_head_attention_7/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????##2@
>transformer_block_7/multi_head_attention_7/dropout/dropout/Mulψ
@transformer_block_7/multi_head_attention_7/dropout/dropout/ShapeShapeDtransformer_block_7/multi_head_attention_7/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2B
@transformer_block_7/multi_head_attention_7/dropout/dropout/ShapeΥ
Wtransformer_block_7/multi_head_attention_7/dropout/dropout/random_uniform/RandomUniformRandomUniformItransformer_block_7/multi_head_attention_7/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????##*
dtype02Y
Wtransformer_block_7/multi_head_attention_7/dropout/dropout/random_uniform/RandomUniformΫ
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
:?????????##2I
Gtransformer_block_7/multi_head_attention_7/dropout/dropout/GreaterEqual 
?transformer_block_7/multi_head_attention_7/dropout/dropout/CastCastKtransformer_block_7/multi_head_attention_7/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????##2A
?transformer_block_7/multi_head_attention_7/dropout/dropout/CastΞ
@transformer_block_7/multi_head_attention_7/dropout/dropout/Mul_1MulBtransformer_block_7/multi_head_attention_7/dropout/dropout/Mul:z:0Ctransformer_block_7/multi_head_attention_7/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????##2B
@transformer_block_7/multi_head_attention_7/dropout/dropout/Mul_1δ
:transformer_block_7/multi_head_attention_7/einsum_1/EinsumEinsumDtransformer_block_7/multi_head_attention_7/dropout/dropout/Mul_1:z:08transformer_block_7/multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:?????????# *
equationacbe,aecd->abcd2<
:transformer_block_7/multi_head_attention_7/einsum_1/EinsumΪ
Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_7_multi_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp£
Itransformer_block_7/multi_head_attention_7/attention_output/einsum/EinsumEinsumCtransformer_block_7/multi_head_attention_7/einsum_1/Einsum:output:0`transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????# *
equationabcd,cde->abe2K
Itransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum΄
Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_7_multi_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpν
?transformer_block_7/multi_head_attention_7/attention_output/addAddV2Rtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum:output:0Vtransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2A
?transformer_block_7/multi_head_attention_7/attention_output/add‘
,transformer_block_7/dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?2.
,transformer_block_7/dropout_20/dropout/Const
*transformer_block_7/dropout_20/dropout/MulMulCtransformer_block_7/multi_head_attention_7/attention_output/add:z:05transformer_block_7/dropout_20/dropout/Const:output:0*
T0*+
_output_shapes
:?????????# 2,
*transformer_block_7/dropout_20/dropout/MulΟ
,transformer_block_7/dropout_20/dropout/ShapeShapeCtransformer_block_7/multi_head_attention_7/attention_output/add:z:0*
T0*
_output_shapes
:2.
,transformer_block_7/dropout_20/dropout/Shape
Ctransformer_block_7/dropout_20/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_7/dropout_20/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????# *
dtype02E
Ctransformer_block_7/dropout_20/dropout/random_uniform/RandomUniform³
5transformer_block_7/dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=27
5transformer_block_7/dropout_20/dropout/GreaterEqual/yΎ
3transformer_block_7/dropout_20/dropout/GreaterEqualGreaterEqualLtransformer_block_7/dropout_20/dropout/random_uniform/RandomUniform:output:0>transformer_block_7/dropout_20/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????# 25
3transformer_block_7/dropout_20/dropout/GreaterEqualΰ
+transformer_block_7/dropout_20/dropout/CastCast7transformer_block_7/dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????# 2-
+transformer_block_7/dropout_20/dropout/Castϊ
,transformer_block_7/dropout_20/dropout/Mul_1Mul.transformer_block_7/dropout_20/dropout/Mul:z:0/transformer_block_7/dropout_20/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????# 2.
,transformer_block_7/dropout_20/dropout/Mul_1²
transformer_block_7/addAddV2add_3/add:z:00transformer_block_7/dropout_20/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????# 2
transformer_block_7/addΰ
Itransformer_block_7/layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_7/layer_normalization_14/moments/mean/reduction_indices²
7transformer_block_7/layer_normalization_14/moments/meanMeantransformer_block_7/add:z:0Rtransformer_block_7/layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(29
7transformer_block_7/layer_normalization_14/moments/mean
?transformer_block_7/layer_normalization_14/moments/StopGradientStopGradient@transformer_block_7/layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2A
?transformer_block_7/layer_normalization_14/moments/StopGradientΎ
Dtransformer_block_7/layer_normalization_14/moments/SquaredDifferenceSquaredDifferencetransformer_block_7/add:z:0Htransformer_block_7/layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 2F
Dtransformer_block_7/layer_normalization_14/moments/SquaredDifferenceθ
Mtransformer_block_7/layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_7/layer_normalization_14/moments/variance/reduction_indicesλ
;transformer_block_7/layer_normalization_14/moments/varianceMeanHtransformer_block_7/layer_normalization_14/moments/SquaredDifference:z:0Vtransformer_block_7/layer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2=
;transformer_block_7/layer_normalization_14/moments/variance½
:transformer_block_7/layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_7/layer_normalization_14/batchnorm/add/yΎ
8transformer_block_7/layer_normalization_14/batchnorm/addAddV2Dtransformer_block_7/layer_normalization_14/moments/variance:output:0Ctransformer_block_7/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2:
8transformer_block_7/layer_normalization_14/batchnorm/addυ
:transformer_block_7/layer_normalization_14/batchnorm/RsqrtRsqrt<transformer_block_7/layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2<
:transformer_block_7/layer_normalization_14/batchnorm/Rsqrt
Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_7_layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpΒ
8transformer_block_7/layer_normalization_14/batchnorm/mulMul>transformer_block_7/layer_normalization_14/batchnorm/Rsqrt:y:0Otransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2:
8transformer_block_7/layer_normalization_14/batchnorm/mul
:transformer_block_7/layer_normalization_14/batchnorm/mul_1Multransformer_block_7/add:z:0<transformer_block_7/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2<
:transformer_block_7/layer_normalization_14/batchnorm/mul_1΅
:transformer_block_7/layer_normalization_14/batchnorm/mul_2Mul@transformer_block_7/layer_normalization_14/moments/mean:output:0<transformer_block_7/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2<
:transformer_block_7/layer_normalization_14/batchnorm/mul_2
Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_7_layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpΎ
8transformer_block_7/layer_normalization_14/batchnorm/subSubKtransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp:value:0>transformer_block_7/layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2:
8transformer_block_7/layer_normalization_14/batchnorm/sub΅
:transformer_block_7/layer_normalization_14/batchnorm/add_1AddV2>transformer_block_7/layer_normalization_14/batchnorm/mul_1:z:0<transformer_block_7/layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2<
:transformer_block_7/layer_normalization_14/batchnorm/add_1
Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_7_sequential_7_dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02D
Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpΎ
8transformer_block_7/sequential_7/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_7/sequential_7/dense_23/Tensordot/axesΕ
8transformer_block_7/sequential_7/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_7/sequential_7/dense_23/Tensordot/freeδ
9transformer_block_7/sequential_7/dense_23/Tensordot/ShapeShape>transformer_block_7/layer_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_23/Tensordot/ShapeΘ
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
<transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2Μ
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
>transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1ΐ
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
8transformer_block_7/sequential_7/dense_23/Tensordot/ProdΔ
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
:transformer_block_7/sequential_7/dense_23/Tensordot/Prod_1Δ
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
:transformer_block_7/sequential_7/dense_23/Tensordot/concat΄
9transformer_block_7/sequential_7/dense_23/Tensordot/stackPackAtransformer_block_7/sequential_7/dense_23/Tensordot/Prod:output:0Ctransformer_block_7/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_23/Tensordot/stackΖ
=transformer_block_7/sequential_7/dense_23/Tensordot/transpose	Transpose>transformer_block_7/layer_normalization_14/batchnorm/add_1:z:0Ctransformer_block_7/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????# 2?
=transformer_block_7/sequential_7/dense_23/Tensordot/transposeΗ
;transformer_block_7/sequential_7/dense_23/Tensordot/ReshapeReshapeAtransformer_block_7/sequential_7/dense_23/Tensordot/transpose:y:0Btransformer_block_7/sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2=
;transformer_block_7/sequential_7/dense_23/Tensordot/ReshapeΖ
:transformer_block_7/sequential_7/dense_23/Tensordot/MatMulMatMulDtransformer_block_7/sequential_7/dense_23/Tensordot/Reshape:output:0Jtransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2<
:transformer_block_7/sequential_7/dense_23/Tensordot/MatMulΔ
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2=
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_2Θ
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
<transformer_block_7/sequential_7/dense_23/Tensordot/concat_1Έ
3transformer_block_7/sequential_7/dense_23/TensordotReshapeDtransformer_block_7/sequential_7/dense_23/Tensordot/MatMul:product:0Etransformer_block_7/sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@25
3transformer_block_7/sequential_7/dense_23/Tensordot
@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_7_sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp―
1transformer_block_7/sequential_7/dense_23/BiasAddBiasAdd<transformer_block_7/sequential_7/dense_23/Tensordot:output:0Htransformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@23
1transformer_block_7/sequential_7/dense_23/BiasAddΪ
.transformer_block_7/sequential_7/dense_23/ReluRelu:transformer_block_7/sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#@20
.transformer_block_7/sequential_7/dense_23/Relu
Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_7_sequential_7_dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02D
Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpΎ
8transformer_block_7/sequential_7/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_7/sequential_7/dense_24/Tensordot/axesΕ
8transformer_block_7/sequential_7/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_7/sequential_7/dense_24/Tensordot/freeβ
9transformer_block_7/sequential_7/dense_24/Tensordot/ShapeShape<transformer_block_7/sequential_7/dense_23/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_24/Tensordot/ShapeΘ
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
<transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2Μ
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
>transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1ΐ
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
8transformer_block_7/sequential_7/dense_24/Tensordot/ProdΔ
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
:transformer_block_7/sequential_7/dense_24/Tensordot/Prod_1Δ
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
:transformer_block_7/sequential_7/dense_24/Tensordot/concat΄
9transformer_block_7/sequential_7/dense_24/Tensordot/stackPackAtransformer_block_7/sequential_7/dense_24/Tensordot/Prod:output:0Ctransformer_block_7/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_24/Tensordot/stackΔ
=transformer_block_7/sequential_7/dense_24/Tensordot/transpose	Transpose<transformer_block_7/sequential_7/dense_23/Relu:activations:0Ctransformer_block_7/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2?
=transformer_block_7/sequential_7/dense_24/Tensordot/transposeΗ
;transformer_block_7/sequential_7/dense_24/Tensordot/ReshapeReshapeAtransformer_block_7/sequential_7/dense_24/Tensordot/transpose:y:0Btransformer_block_7/sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2=
;transformer_block_7/sequential_7/dense_24/Tensordot/ReshapeΖ
:transformer_block_7/sequential_7/dense_24/Tensordot/MatMulMatMulDtransformer_block_7/sequential_7/dense_24/Tensordot/Reshape:output:0Jtransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2<
:transformer_block_7/sequential_7/dense_24/Tensordot/MatMulΔ
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_2Θ
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
<transformer_block_7/sequential_7/dense_24/Tensordot/concat_1Έ
3transformer_block_7/sequential_7/dense_24/TensordotReshapeDtransformer_block_7/sequential_7/dense_24/Tensordot/MatMul:product:0Etransformer_block_7/sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 25
3transformer_block_7/sequential_7/dense_24/Tensordot
@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_7_sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp―
1transformer_block_7/sequential_7/dense_24/BiasAddBiasAdd<transformer_block_7/sequential_7/dense_24/Tensordot:output:0Htransformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 23
1transformer_block_7/sequential_7/dense_24/BiasAdd‘
,transformer_block_7/dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?2.
,transformer_block_7/dropout_21/dropout/Const
*transformer_block_7/dropout_21/dropout/MulMul:transformer_block_7/sequential_7/dense_24/BiasAdd:output:05transformer_block_7/dropout_21/dropout/Const:output:0*
T0*+
_output_shapes
:?????????# 2,
*transformer_block_7/dropout_21/dropout/MulΖ
,transformer_block_7/dropout_21/dropout/ShapeShape:transformer_block_7/sequential_7/dense_24/BiasAdd:output:0*
T0*
_output_shapes
:2.
,transformer_block_7/dropout_21/dropout/Shape
Ctransformer_block_7/dropout_21/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_7/dropout_21/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????# *
dtype02E
Ctransformer_block_7/dropout_21/dropout/random_uniform/RandomUniform³
5transformer_block_7/dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=27
5transformer_block_7/dropout_21/dropout/GreaterEqual/yΎ
3transformer_block_7/dropout_21/dropout/GreaterEqualGreaterEqualLtransformer_block_7/dropout_21/dropout/random_uniform/RandomUniform:output:0>transformer_block_7/dropout_21/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????# 25
3transformer_block_7/dropout_21/dropout/GreaterEqualΰ
+transformer_block_7/dropout_21/dropout/CastCast7transformer_block_7/dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????# 2-
+transformer_block_7/dropout_21/dropout/Castϊ
,transformer_block_7/dropout_21/dropout/Mul_1Mul.transformer_block_7/dropout_21/dropout/Mul:z:0/transformer_block_7/dropout_21/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????# 2.
,transformer_block_7/dropout_21/dropout/Mul_1η
transformer_block_7/add_1AddV2>transformer_block_7/layer_normalization_14/batchnorm/add_1:z:00transformer_block_7/dropout_21/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????# 2
transformer_block_7/add_1ΰ
Itransformer_block_7/layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_7/layer_normalization_15/moments/mean/reduction_indices΄
7transformer_block_7/layer_normalization_15/moments/meanMeantransformer_block_7/add_1:z:0Rtransformer_block_7/layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(29
7transformer_block_7/layer_normalization_15/moments/mean
?transformer_block_7/layer_normalization_15/moments/StopGradientStopGradient@transformer_block_7/layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2A
?transformer_block_7/layer_normalization_15/moments/StopGradientΐ
Dtransformer_block_7/layer_normalization_15/moments/SquaredDifferenceSquaredDifferencetransformer_block_7/add_1:z:0Htransformer_block_7/layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 2F
Dtransformer_block_7/layer_normalization_15/moments/SquaredDifferenceθ
Mtransformer_block_7/layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_7/layer_normalization_15/moments/variance/reduction_indicesλ
;transformer_block_7/layer_normalization_15/moments/varianceMeanHtransformer_block_7/layer_normalization_15/moments/SquaredDifference:z:0Vtransformer_block_7/layer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2=
;transformer_block_7/layer_normalization_15/moments/variance½
:transformer_block_7/layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_7/layer_normalization_15/batchnorm/add/yΎ
8transformer_block_7/layer_normalization_15/batchnorm/addAddV2Dtransformer_block_7/layer_normalization_15/moments/variance:output:0Ctransformer_block_7/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2:
8transformer_block_7/layer_normalization_15/batchnorm/addυ
:transformer_block_7/layer_normalization_15/batchnorm/RsqrtRsqrt<transformer_block_7/layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2<
:transformer_block_7/layer_normalization_15/batchnorm/Rsqrt
Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_7_layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpΒ
8transformer_block_7/layer_normalization_15/batchnorm/mulMul>transformer_block_7/layer_normalization_15/batchnorm/Rsqrt:y:0Otransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2:
8transformer_block_7/layer_normalization_15/batchnorm/mul
:transformer_block_7/layer_normalization_15/batchnorm/mul_1Multransformer_block_7/add_1:z:0<transformer_block_7/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2<
:transformer_block_7/layer_normalization_15/batchnorm/mul_1΅
:transformer_block_7/layer_normalization_15/batchnorm/mul_2Mul@transformer_block_7/layer_normalization_15/moments/mean:output:0<transformer_block_7/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2<
:transformer_block_7/layer_normalization_15/batchnorm/mul_2
Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_7_layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpΎ
8transformer_block_7/layer_normalization_15/batchnorm/subSubKtransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp:value:0>transformer_block_7/layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2:
8transformer_block_7/layer_normalization_15/batchnorm/sub΅
:transformer_block_7/layer_normalization_15/batchnorm/add_1AddV2>transformer_block_7/layer_normalization_15/batchnorm/mul_1:z:0<transformer_block_7/layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2<
:transformer_block_7/layer_normalization_15/batchnorm/add_1¨
1global_average_pooling1d_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_3/Mean/reduction_indicesψ
global_average_pooling1d_3/MeanMean>transformer_block_7/layer_normalization_15/batchnorm/add_1:z:0:global_average_pooling1d_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2!
global_average_pooling1d_3/Means
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
flatten_3/Const§
flatten_3/ReshapeReshape(global_average_pooling1d_3/Mean:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:????????? 2
flatten_3/Reshapex
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axis½
concatenate_3/concatConcatV2flatten_3/Reshape:output:0inputs_1"concatenate_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????(2
concatenate_3/concat¨
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:(@*
dtype02 
dense_25/MatMul/ReadVariableOp₯
dense_25/MatMulMatMulconcatenate_3/concat:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_25/MatMul§
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_25/BiasAdd/ReadVariableOp₯
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_25/BiasAdds
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_25/Reluy
dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?2
dropout_22/dropout/Const©
dropout_22/dropout/MulMuldense_25/Relu:activations:0!dropout_22/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout_22/dropout/Mul
dropout_22/dropout/ShapeShapedense_25/Relu:activations:0*
T0*
_output_shapes
:2
dropout_22/dropout/ShapeΥ
/dropout_22/dropout/random_uniform/RandomUniformRandomUniform!dropout_22/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype021
/dropout_22/dropout/random_uniform/RandomUniform
!dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2#
!dropout_22/dropout/GreaterEqual/yκ
dropout_22/dropout/GreaterEqualGreaterEqual8dropout_22/dropout/random_uniform/RandomUniform:output:0*dropout_22/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2!
dropout_22/dropout/GreaterEqual 
dropout_22/dropout/CastCast#dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout_22/dropout/Cast¦
dropout_22/dropout/Mul_1Muldropout_22/dropout/Mul:z:0dropout_22/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout_22/dropout/Mul_1¨
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_26/MatMul/ReadVariableOp€
dense_26/MatMulMatMuldropout_22/dropout/Mul_1:z:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_26/MatMul§
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_26/BiasAdd/ReadVariableOp₯
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_26/BiasAdds
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_26/Reluy
dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?2
dropout_23/dropout/Const©
dropout_23/dropout/MulMuldense_26/Relu:activations:0!dropout_23/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout_23/dropout/Mul
dropout_23/dropout/ShapeShapedense_26/Relu:activations:0*
T0*
_output_shapes
:2
dropout_23/dropout/ShapeΥ
/dropout_23/dropout/random_uniform/RandomUniformRandomUniform!dropout_23/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype021
/dropout_23/dropout/random_uniform/RandomUniform
!dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2#
!dropout_23/dropout/GreaterEqual/yκ
dropout_23/dropout/GreaterEqualGreaterEqual8dropout_23/dropout/random_uniform/RandomUniform:output:0*dropout_23/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2!
dropout_23/dropout/GreaterEqual 
dropout_23/dropout/CastCast#dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout_23/dropout/Cast¦
dropout_23/dropout/Mul_1Muldropout_23/dropout/Mul:z:0dropout_23/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout_23/dropout/Mul_1¨
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_27/MatMul/ReadVariableOp€
dense_27/MatMulMatMuldropout_23/dropout/Mul_1:z:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_27/MatMul§
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_27/BiasAdd/ReadVariableOp₯
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_27/BiasAdd
IdentityIdentitydense_27/BiasAdd:output:0:^batch_normalization_6/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_6/AssignMovingAvg/ReadVariableOp<^batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_6/batchnorm/ReadVariableOp3^batch_normalization_6/batchnorm/mul/ReadVariableOp:^batch_normalization_7/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_7/AssignMovingAvg/ReadVariableOp<^batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_7/batchnorm/ReadVariableOp3^batch_normalization_7/batchnorm/mul/ReadVariableOp ^conv1d_6/BiasAdd/ReadVariableOp,^conv1d_6/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_7/BiasAdd/ReadVariableOp,^conv1d_7/conv1d/ExpandDims_1/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp<^token_and_position_embedding_3/embedding_6/embedding_lookup<^token_and_position_embedding_3/embedding_7/embedding_lookupD^transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpH^transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpD^transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpH^transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpO^transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpY^transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_7/multi_head_attention_7/key/add/ReadVariableOpL^transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpD^transformer_block_7/multi_head_attention_7/query/add/ReadVariableOpN^transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpD^transformer_block_7/multi_head_attention_7/value/add/ReadVariableOpN^transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpA^transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpC^transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpA^transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpC^transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Μ
_input_shapesΊ
·:?????????R:?????????::::::::::::::::::::::::::::::::::::2v
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
Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpNtransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp2΄
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
Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpBtransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp:R N
(
_output_shapes
:?????????R
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1


Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_307860

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOp
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
 :?????????????????? 2
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
 :?????????????????? 2
batchnorm/add_1θ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????????????? ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
₯
d
+__inference_dropout_23_layer_call_fn_308628

inputs
identity’StatefulPartitionedCallί
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
GPU2*0J 8 *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_3064352
StatefulPartitionedCall
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
σ
~
)__inference_conv1d_7_layer_call_fn_307804

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallό
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????ή *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_3056982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????ή 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????ή ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????ή 
 
_user_specified_nameinputs
ξ	
έ
D__inference_dense_26_layer_call_and_return_conditional_losses_308597

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu
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
₯
d
+__inference_dropout_22_layer_call_fn_308581

inputs
identity’StatefulPartitionedCallί
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
GPU2*0J 8 *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_3063782
StatefulPartitionedCall
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
Π
¨
-__inference_sequential_7_layer_call_fn_305558
dense_23_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCalldense_23_inputunknown	unknown_0	unknown_1	unknown_2*
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
GPU2*0J 8 *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_3055472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:?????????# 
(
_user_specified_namedense_23_input
± 
γ
D__inference_dense_23_layer_call_and_return_conditional_losses_308823

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Tensordot/ReadVariableOp
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
Tensordot/GatherV2/axisΡ
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
Tensordot/GatherV2_1/axisΧ
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
:?????????# 2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape
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
:?????????#@2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????#@2
Relu
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
± 
γ
D__inference_dense_23_layer_call_and_return_conditional_losses_305453

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Tensordot/ReadVariableOp
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
Tensordot/GatherV2/axisΡ
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
Tensordot/GatherV2_1/axisΧ
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
:?????????# 2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape
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
:?????????#@2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????#@2
Relu
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
Σί
Μ$
C__inference_model_3_layer_call_and_return_conditional_losses_307565
inputs_0
inputs_1F
Btoken_and_position_embedding_3_embedding_7_embedding_lookup_307332F
Btoken_and_position_embedding_3_embedding_6_embedding_lookup_3073388
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
identity’.batch_normalization_6/batchnorm/ReadVariableOp’0batch_normalization_6/batchnorm/ReadVariableOp_1’0batch_normalization_6/batchnorm/ReadVariableOp_2’2batch_normalization_6/batchnorm/mul/ReadVariableOp’.batch_normalization_7/batchnorm/ReadVariableOp’0batch_normalization_7/batchnorm/ReadVariableOp_1’0batch_normalization_7/batchnorm/ReadVariableOp_2’2batch_normalization_7/batchnorm/mul/ReadVariableOp’conv1d_6/BiasAdd/ReadVariableOp’+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp’conv1d_7/BiasAdd/ReadVariableOp’+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp’dense_25/BiasAdd/ReadVariableOp’dense_25/MatMul/ReadVariableOp’dense_26/BiasAdd/ReadVariableOp’dense_26/MatMul/ReadVariableOp’dense_27/BiasAdd/ReadVariableOp’dense_27/MatMul/ReadVariableOp’;token_and_position_embedding_3/embedding_6/embedding_lookup’;token_and_position_embedding_3/embedding_7/embedding_lookup’Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp’Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp’Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp’Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp’Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp’Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp’Atransformer_block_7/multi_head_attention_7/key/add/ReadVariableOp’Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp’Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOp’Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp’Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOp’Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp’@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp’Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp’@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp’Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp
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
?????????24
2token_and_position_embedding_3/strided_slice/stackΆ
4token_and_position_embedding_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_3/strided_slice/stack_1Ά
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
:?????????2&
$token_and_position_embedding_3/rangeΚ
;token_and_position_embedding_3/embedding_7/embedding_lookupResourceGatherBtoken_and_position_embedding_3_embedding_7_embedding_lookup_307332-token_and_position_embedding_3/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_7/embedding_lookup/307332*'
_output_shapes
:????????? *
dtype02=
;token_and_position_embedding_3/embedding_7/embedding_lookup
Dtoken_and_position_embedding_3/embedding_7/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_3/embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_7/embedding_lookup/307332*'
_output_shapes
:????????? 2F
Dtoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity
Ftoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2H
Ftoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1Ά
/token_and_position_embedding_3/embedding_6/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:?????????R21
/token_and_position_embedding_3/embedding_6/CastΥ
;token_and_position_embedding_3/embedding_6/embedding_lookupResourceGatherBtoken_and_position_embedding_3_embedding_6_embedding_lookup_3073383token_and_position_embedding_3/embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_6/embedding_lookup/307338*,
_output_shapes
:?????????R *
dtype02=
;token_and_position_embedding_3/embedding_6/embedding_lookup
Dtoken_and_position_embedding_3/embedding_6/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_3/embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_6/embedding_lookup/307338*,
_output_shapes
:?????????R 2F
Dtoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity’
Ftoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:?????????R 2H
Ftoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1ͺ
"token_and_position_embedding_3/addAddV2Otoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:?????????R 2$
"token_and_position_embedding_3/add
conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2 
conv1d_6/conv1d/ExpandDims/dim?
conv1d_6/conv1d/ExpandDims
ExpandDims&token_and_position_embedding_3/add:z:0'conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????R 2
conv1d_6/conv1d/ExpandDimsΣ
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
 conv1d_6/conv1d/ExpandDims_1/dimΫ
conv1d_6/conv1d/ExpandDims_1
ExpandDims3conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_6/conv1d/ExpandDims_1Ϋ
conv1d_6/conv1dConv2D#conv1d_6/conv1d/ExpandDims:output:0%conv1d_6/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????R *
paddingSAME*
strides
2
conv1d_6/conv1d?
conv1d_6/conv1d/SqueezeSqueezeconv1d_6/conv1d:output:0*
T0*,
_output_shapes
:?????????R *
squeeze_dims

ύ????????2
conv1d_6/conv1d/Squeeze§
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_6/BiasAdd/ReadVariableOp±
conv1d_6/BiasAddBiasAdd conv1d_6/conv1d/Squeeze:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????R 2
conv1d_6/BiasAddx
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*,
_output_shapes
:?????????R 2
conv1d_6/Relu
"average_pooling1d_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_9/ExpandDims/dimΣ
average_pooling1d_9/ExpandDims
ExpandDimsconv1d_6/Relu:activations:0+average_pooling1d_9/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????R 2 
average_pooling1d_9/ExpandDimsε
average_pooling1d_9/AvgPoolAvgPool'average_pooling1d_9/ExpandDims:output:0*
T0*0
_output_shapes
:?????????ή *
ksize
*
paddingVALID*
strides
2
average_pooling1d_9/AvgPoolΉ
average_pooling1d_9/SqueezeSqueeze$average_pooling1d_9/AvgPool:output:0*
T0*,
_output_shapes
:?????????ή *
squeeze_dims
2
average_pooling1d_9/Squeeze
conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2 
conv1d_7/conv1d/ExpandDims/dimΠ
conv1d_7/conv1d/ExpandDims
ExpandDims$average_pooling1d_9/Squeeze:output:0'conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ή 2
conv1d_7/conv1d/ExpandDimsΣ
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
 conv1d_7/conv1d/ExpandDims_1/dimΫ
conv1d_7/conv1d/ExpandDims_1
ExpandDims3conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d_7/conv1d/ExpandDims_1Ϋ
conv1d_7/conv1dConv2D#conv1d_7/conv1d/ExpandDims:output:0%conv1d_7/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????ή *
paddingSAME*
strides
2
conv1d_7/conv1d?
conv1d_7/conv1d/SqueezeSqueezeconv1d_7/conv1d:output:0*
T0*,
_output_shapes
:?????????ή *
squeeze_dims

ύ????????2
conv1d_7/conv1d/Squeeze§
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_7/BiasAdd/ReadVariableOp±
conv1d_7/BiasAddBiasAdd conv1d_7/conv1d/Squeeze:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ή 2
conv1d_7/BiasAddx
conv1d_7/ReluReluconv1d_7/BiasAdd:output:0*
T0*,
_output_shapes
:?????????ή 2
conv1d_7/Relu
#average_pooling1d_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_11/ExpandDims/dimα
average_pooling1d_11/ExpandDims
ExpandDims&token_and_position_embedding_3/add:z:0,average_pooling1d_11/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????R 2!
average_pooling1d_11/ExpandDimsι
average_pooling1d_11/AvgPoolAvgPool(average_pooling1d_11/ExpandDims:output:0*
T0*/
_output_shapes
:?????????# *
ksize	
¬*
paddingVALID*
strides	
¬2
average_pooling1d_11/AvgPool»
average_pooling1d_11/SqueezeSqueeze%average_pooling1d_11/AvgPool:output:0*
T0*+
_output_shapes
:?????????# *
squeeze_dims
2
average_pooling1d_11/Squeeze
#average_pooling1d_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_10/ExpandDims/dimΦ
average_pooling1d_10/ExpandDims
ExpandDimsconv1d_7/Relu:activations:0,average_pooling1d_10/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ή 2!
average_pooling1d_10/ExpandDimsη
average_pooling1d_10/AvgPoolAvgPool(average_pooling1d_10/ExpandDims:output:0*
T0*/
_output_shapes
:?????????# *
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
:?????????# *
squeeze_dims
2
average_pooling1d_10/SqueezeΤ
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
%batch_normalization_6/batchnorm/add/yΰ
#batch_normalization_6/batchnorm/addAddV26batch_normalization_6/batchnorm/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_6/batchnorm/add₯
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_6/batchnorm/Rsqrtΰ
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_6/batchnorm/mul/ReadVariableOpέ
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_6/batchnorm/mulΫ
%batch_normalization_6/batchnorm/mul_1Mul%average_pooling1d_10/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%batch_normalization_6/batchnorm/mul_1Ϊ
0batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_6/batchnorm/ReadVariableOp_1έ
%batch_normalization_6/batchnorm/mul_2Mul8batch_normalization_6/batchnorm/ReadVariableOp_1:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_6/batchnorm/mul_2Ϊ
0batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_6/batchnorm/ReadVariableOp_2Ϋ
#batch_normalization_6/batchnorm/subSub8batch_normalization_6/batchnorm/ReadVariableOp_2:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_6/batchnorm/subα
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2'
%batch_normalization_6/batchnorm/add_1Τ
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
%batch_normalization_7/batchnorm/add/yΰ
#batch_normalization_7/batchnorm/addAddV26batch_normalization_7/batchnorm/ReadVariableOp:value:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_7/batchnorm/add₯
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_7/batchnorm/Rsqrtΰ
2batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_7/batchnorm/mul/ReadVariableOpέ
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:0:batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_7/batchnorm/mulΫ
%batch_normalization_7/batchnorm/mul_1Mul%average_pooling1d_11/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2'
%batch_normalization_7/batchnorm/mul_1Ϊ
0batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_7/batchnorm/ReadVariableOp_1έ
%batch_normalization_7/batchnorm/mul_2Mul8batch_normalization_7/batchnorm/ReadVariableOp_1:value:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_7/batchnorm/mul_2Ϊ
0batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_7/batchnorm/ReadVariableOp_2Ϋ
#batch_normalization_7/batchnorm/subSub8batch_normalization_7/batchnorm/ReadVariableOp_2:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_7/batchnorm/subα
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2'
%batch_normalization_7/batchnorm/add_1«
	add_3/addAddV2)batch_normalization_6/batchnorm/add_1:z:0)batch_normalization_7/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????# 2
	add_3/addΉ
Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_7_multi_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpΠ
>transformer_block_7/multi_head_attention_7/query/einsum/EinsumEinsumadd_3/add:z:0Utransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2@
>transformer_block_7/multi_head_attention_7/query/einsum/Einsum
Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOpReadVariableOpLtransformer_block_7_multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOpΕ
4transformer_block_7/multi_head_attention_7/query/addAddV2Gtransformer_block_7/multi_head_attention_7/query/einsum/Einsum:output:0Ktransformer_block_7/multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 26
4transformer_block_7/multi_head_attention_7/query/add³
Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_7_multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpΚ
<transformer_block_7/multi_head_attention_7/key/einsum/EinsumEinsumadd_3/add:z:0Stransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
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
:?????????# 24
2transformer_block_7/multi_head_attention_7/key/addΉ
Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_7_multi_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpΠ
>transformer_block_7/multi_head_attention_7/value/einsum/EinsumEinsumadd_3/add:z:0Utransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2@
>transformer_block_7/multi_head_attention_7/value/einsum/Einsum
Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOpReadVariableOpLtransformer_block_7_multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOpΕ
4transformer_block_7/multi_head_attention_7/value/addAddV2Gtransformer_block_7/multi_head_attention_7/value/einsum/Einsum:output:0Ktransformer_block_7/multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 26
4transformer_block_7/multi_head_attention_7/value/add©
0transformer_block_7/multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *σ5>22
0transformer_block_7/multi_head_attention_7/Mul/y
.transformer_block_7/multi_head_attention_7/MulMul8transformer_block_7/multi_head_attention_7/query/add:z:09transformer_block_7/multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:?????????# 20
.transformer_block_7/multi_head_attention_7/MulΜ
8transformer_block_7/multi_head_attention_7/einsum/EinsumEinsum6transformer_block_7/multi_head_attention_7/key/add:z:02transformer_block_7/multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:?????????##*
equationaecd,abcd->acbe2:
8transformer_block_7/multi_head_attention_7/einsum/Einsum
:transformer_block_7/multi_head_attention_7/softmax/SoftmaxSoftmaxAtransformer_block_7/multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????##2<
:transformer_block_7/multi_head_attention_7/softmax/Softmax
;transformer_block_7/multi_head_attention_7/dropout/IdentityIdentityDtransformer_block_7/multi_head_attention_7/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????##2=
;transformer_block_7/multi_head_attention_7/dropout/Identityδ
:transformer_block_7/multi_head_attention_7/einsum_1/EinsumEinsumDtransformer_block_7/multi_head_attention_7/dropout/Identity:output:08transformer_block_7/multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:?????????# *
equationacbe,aecd->abcd2<
:transformer_block_7/multi_head_attention_7/einsum_1/EinsumΪ
Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_7_multi_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp£
Itransformer_block_7/multi_head_attention_7/attention_output/einsum/EinsumEinsumCtransformer_block_7/multi_head_attention_7/einsum_1/Einsum:output:0`transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????# *
equationabcd,cde->abe2K
Itransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum΄
Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_7_multi_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpν
?transformer_block_7/multi_head_attention_7/attention_output/addAddV2Rtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum:output:0Vtransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2A
?transformer_block_7/multi_head_attention_7/attention_output/addΩ
'transformer_block_7/dropout_20/IdentityIdentityCtransformer_block_7/multi_head_attention_7/attention_output/add:z:0*
T0*+
_output_shapes
:?????????# 2)
'transformer_block_7/dropout_20/Identity²
transformer_block_7/addAddV2add_3/add:z:00transformer_block_7/dropout_20/Identity:output:0*
T0*+
_output_shapes
:?????????# 2
transformer_block_7/addΰ
Itransformer_block_7/layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_7/layer_normalization_14/moments/mean/reduction_indices²
7transformer_block_7/layer_normalization_14/moments/meanMeantransformer_block_7/add:z:0Rtransformer_block_7/layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(29
7transformer_block_7/layer_normalization_14/moments/mean
?transformer_block_7/layer_normalization_14/moments/StopGradientStopGradient@transformer_block_7/layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2A
?transformer_block_7/layer_normalization_14/moments/StopGradientΎ
Dtransformer_block_7/layer_normalization_14/moments/SquaredDifferenceSquaredDifferencetransformer_block_7/add:z:0Htransformer_block_7/layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 2F
Dtransformer_block_7/layer_normalization_14/moments/SquaredDifferenceθ
Mtransformer_block_7/layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_7/layer_normalization_14/moments/variance/reduction_indicesλ
;transformer_block_7/layer_normalization_14/moments/varianceMeanHtransformer_block_7/layer_normalization_14/moments/SquaredDifference:z:0Vtransformer_block_7/layer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2=
;transformer_block_7/layer_normalization_14/moments/variance½
:transformer_block_7/layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_7/layer_normalization_14/batchnorm/add/yΎ
8transformer_block_7/layer_normalization_14/batchnorm/addAddV2Dtransformer_block_7/layer_normalization_14/moments/variance:output:0Ctransformer_block_7/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2:
8transformer_block_7/layer_normalization_14/batchnorm/addυ
:transformer_block_7/layer_normalization_14/batchnorm/RsqrtRsqrt<transformer_block_7/layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2<
:transformer_block_7/layer_normalization_14/batchnorm/Rsqrt
Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_7_layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpΒ
8transformer_block_7/layer_normalization_14/batchnorm/mulMul>transformer_block_7/layer_normalization_14/batchnorm/Rsqrt:y:0Otransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2:
8transformer_block_7/layer_normalization_14/batchnorm/mul
:transformer_block_7/layer_normalization_14/batchnorm/mul_1Multransformer_block_7/add:z:0<transformer_block_7/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2<
:transformer_block_7/layer_normalization_14/batchnorm/mul_1΅
:transformer_block_7/layer_normalization_14/batchnorm/mul_2Mul@transformer_block_7/layer_normalization_14/moments/mean:output:0<transformer_block_7/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2<
:transformer_block_7/layer_normalization_14/batchnorm/mul_2
Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_7_layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpΎ
8transformer_block_7/layer_normalization_14/batchnorm/subSubKtransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp:value:0>transformer_block_7/layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2:
8transformer_block_7/layer_normalization_14/batchnorm/sub΅
:transformer_block_7/layer_normalization_14/batchnorm/add_1AddV2>transformer_block_7/layer_normalization_14/batchnorm/mul_1:z:0<transformer_block_7/layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2<
:transformer_block_7/layer_normalization_14/batchnorm/add_1
Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_7_sequential_7_dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02D
Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpΎ
8transformer_block_7/sequential_7/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_7/sequential_7/dense_23/Tensordot/axesΕ
8transformer_block_7/sequential_7/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_7/sequential_7/dense_23/Tensordot/freeδ
9transformer_block_7/sequential_7/dense_23/Tensordot/ShapeShape>transformer_block_7/layer_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_23/Tensordot/ShapeΘ
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
<transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2Μ
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
>transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1ΐ
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
8transformer_block_7/sequential_7/dense_23/Tensordot/ProdΔ
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
:transformer_block_7/sequential_7/dense_23/Tensordot/Prod_1Δ
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
:transformer_block_7/sequential_7/dense_23/Tensordot/concat΄
9transformer_block_7/sequential_7/dense_23/Tensordot/stackPackAtransformer_block_7/sequential_7/dense_23/Tensordot/Prod:output:0Ctransformer_block_7/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_23/Tensordot/stackΖ
=transformer_block_7/sequential_7/dense_23/Tensordot/transpose	Transpose>transformer_block_7/layer_normalization_14/batchnorm/add_1:z:0Ctransformer_block_7/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????# 2?
=transformer_block_7/sequential_7/dense_23/Tensordot/transposeΗ
;transformer_block_7/sequential_7/dense_23/Tensordot/ReshapeReshapeAtransformer_block_7/sequential_7/dense_23/Tensordot/transpose:y:0Btransformer_block_7/sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2=
;transformer_block_7/sequential_7/dense_23/Tensordot/ReshapeΖ
:transformer_block_7/sequential_7/dense_23/Tensordot/MatMulMatMulDtransformer_block_7/sequential_7/dense_23/Tensordot/Reshape:output:0Jtransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2<
:transformer_block_7/sequential_7/dense_23/Tensordot/MatMulΔ
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2=
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_2Θ
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
<transformer_block_7/sequential_7/dense_23/Tensordot/concat_1Έ
3transformer_block_7/sequential_7/dense_23/TensordotReshapeDtransformer_block_7/sequential_7/dense_23/Tensordot/MatMul:product:0Etransformer_block_7/sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@25
3transformer_block_7/sequential_7/dense_23/Tensordot
@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_7_sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp―
1transformer_block_7/sequential_7/dense_23/BiasAddBiasAdd<transformer_block_7/sequential_7/dense_23/Tensordot:output:0Htransformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@23
1transformer_block_7/sequential_7/dense_23/BiasAddΪ
.transformer_block_7/sequential_7/dense_23/ReluRelu:transformer_block_7/sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#@20
.transformer_block_7/sequential_7/dense_23/Relu
Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_7_sequential_7_dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02D
Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpΎ
8transformer_block_7/sequential_7/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_7/sequential_7/dense_24/Tensordot/axesΕ
8transformer_block_7/sequential_7/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_7/sequential_7/dense_24/Tensordot/freeβ
9transformer_block_7/sequential_7/dense_24/Tensordot/ShapeShape<transformer_block_7/sequential_7/dense_23/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_24/Tensordot/ShapeΘ
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
<transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2Μ
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
>transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1ΐ
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
8transformer_block_7/sequential_7/dense_24/Tensordot/ProdΔ
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
:transformer_block_7/sequential_7/dense_24/Tensordot/Prod_1Δ
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
:transformer_block_7/sequential_7/dense_24/Tensordot/concat΄
9transformer_block_7/sequential_7/dense_24/Tensordot/stackPackAtransformer_block_7/sequential_7/dense_24/Tensordot/Prod:output:0Ctransformer_block_7/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_24/Tensordot/stackΔ
=transformer_block_7/sequential_7/dense_24/Tensordot/transpose	Transpose<transformer_block_7/sequential_7/dense_23/Relu:activations:0Ctransformer_block_7/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2?
=transformer_block_7/sequential_7/dense_24/Tensordot/transposeΗ
;transformer_block_7/sequential_7/dense_24/Tensordot/ReshapeReshapeAtransformer_block_7/sequential_7/dense_24/Tensordot/transpose:y:0Btransformer_block_7/sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2=
;transformer_block_7/sequential_7/dense_24/Tensordot/ReshapeΖ
:transformer_block_7/sequential_7/dense_24/Tensordot/MatMulMatMulDtransformer_block_7/sequential_7/dense_24/Tensordot/Reshape:output:0Jtransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2<
:transformer_block_7/sequential_7/dense_24/Tensordot/MatMulΔ
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_2Θ
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
<transformer_block_7/sequential_7/dense_24/Tensordot/concat_1Έ
3transformer_block_7/sequential_7/dense_24/TensordotReshapeDtransformer_block_7/sequential_7/dense_24/Tensordot/MatMul:product:0Etransformer_block_7/sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 25
3transformer_block_7/sequential_7/dense_24/Tensordot
@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_7_sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp―
1transformer_block_7/sequential_7/dense_24/BiasAddBiasAdd<transformer_block_7/sequential_7/dense_24/Tensordot:output:0Htransformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 23
1transformer_block_7/sequential_7/dense_24/BiasAddΠ
'transformer_block_7/dropout_21/IdentityIdentity:transformer_block_7/sequential_7/dense_24/BiasAdd:output:0*
T0*+
_output_shapes
:?????????# 2)
'transformer_block_7/dropout_21/Identityη
transformer_block_7/add_1AddV2>transformer_block_7/layer_normalization_14/batchnorm/add_1:z:00transformer_block_7/dropout_21/Identity:output:0*
T0*+
_output_shapes
:?????????# 2
transformer_block_7/add_1ΰ
Itransformer_block_7/layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_7/layer_normalization_15/moments/mean/reduction_indices΄
7transformer_block_7/layer_normalization_15/moments/meanMeantransformer_block_7/add_1:z:0Rtransformer_block_7/layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(29
7transformer_block_7/layer_normalization_15/moments/mean
?transformer_block_7/layer_normalization_15/moments/StopGradientStopGradient@transformer_block_7/layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2A
?transformer_block_7/layer_normalization_15/moments/StopGradientΐ
Dtransformer_block_7/layer_normalization_15/moments/SquaredDifferenceSquaredDifferencetransformer_block_7/add_1:z:0Htransformer_block_7/layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 2F
Dtransformer_block_7/layer_normalization_15/moments/SquaredDifferenceθ
Mtransformer_block_7/layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_7/layer_normalization_15/moments/variance/reduction_indicesλ
;transformer_block_7/layer_normalization_15/moments/varianceMeanHtransformer_block_7/layer_normalization_15/moments/SquaredDifference:z:0Vtransformer_block_7/layer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2=
;transformer_block_7/layer_normalization_15/moments/variance½
:transformer_block_7/layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_7/layer_normalization_15/batchnorm/add/yΎ
8transformer_block_7/layer_normalization_15/batchnorm/addAddV2Dtransformer_block_7/layer_normalization_15/moments/variance:output:0Ctransformer_block_7/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2:
8transformer_block_7/layer_normalization_15/batchnorm/addυ
:transformer_block_7/layer_normalization_15/batchnorm/RsqrtRsqrt<transformer_block_7/layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2<
:transformer_block_7/layer_normalization_15/batchnorm/Rsqrt
Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_7_layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpΒ
8transformer_block_7/layer_normalization_15/batchnorm/mulMul>transformer_block_7/layer_normalization_15/batchnorm/Rsqrt:y:0Otransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2:
8transformer_block_7/layer_normalization_15/batchnorm/mul
:transformer_block_7/layer_normalization_15/batchnorm/mul_1Multransformer_block_7/add_1:z:0<transformer_block_7/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2<
:transformer_block_7/layer_normalization_15/batchnorm/mul_1΅
:transformer_block_7/layer_normalization_15/batchnorm/mul_2Mul@transformer_block_7/layer_normalization_15/moments/mean:output:0<transformer_block_7/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2<
:transformer_block_7/layer_normalization_15/batchnorm/mul_2
Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_7_layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpΎ
8transformer_block_7/layer_normalization_15/batchnorm/subSubKtransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp:value:0>transformer_block_7/layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2:
8transformer_block_7/layer_normalization_15/batchnorm/sub΅
:transformer_block_7/layer_normalization_15/batchnorm/add_1AddV2>transformer_block_7/layer_normalization_15/batchnorm/mul_1:z:0<transformer_block_7/layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2<
:transformer_block_7/layer_normalization_15/batchnorm/add_1¨
1global_average_pooling1d_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_3/Mean/reduction_indicesψ
global_average_pooling1d_3/MeanMean>transformer_block_7/layer_normalization_15/batchnorm/add_1:z:0:global_average_pooling1d_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2!
global_average_pooling1d_3/Means
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
flatten_3/Const§
flatten_3/ReshapeReshape(global_average_pooling1d_3/Mean:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:????????? 2
flatten_3/Reshapex
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axis½
concatenate_3/concatConcatV2flatten_3/Reshape:output:0inputs_1"concatenate_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????(2
concatenate_3/concat¨
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:(@*
dtype02 
dense_25/MatMul/ReadVariableOp₯
dense_25/MatMulMatMulconcatenate_3/concat:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_25/MatMul§
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_25/BiasAdd/ReadVariableOp₯
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_25/BiasAdds
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_25/Relu
dropout_22/IdentityIdentitydense_25/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
dropout_22/Identity¨
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_26/MatMul/ReadVariableOp€
dense_26/MatMulMatMuldropout_22/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_26/MatMul§
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_26/BiasAdd/ReadVariableOp₯
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_26/BiasAdds
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_26/Relu
dropout_23/IdentityIdentitydense_26/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
dropout_23/Identity¨
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_27/MatMul/ReadVariableOp€
dense_27/MatMulMatMuldropout_23/Identity:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_27/MatMul§
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_27/BiasAdd/ReadVariableOp₯
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_27/BiasAdd
IdentityIdentitydense_27/BiasAdd:output:0/^batch_normalization_6/batchnorm/ReadVariableOp1^batch_normalization_6/batchnorm/ReadVariableOp_11^batch_normalization_6/batchnorm/ReadVariableOp_23^batch_normalization_6/batchnorm/mul/ReadVariableOp/^batch_normalization_7/batchnorm/ReadVariableOp1^batch_normalization_7/batchnorm/ReadVariableOp_11^batch_normalization_7/batchnorm/ReadVariableOp_23^batch_normalization_7/batchnorm/mul/ReadVariableOp ^conv1d_6/BiasAdd/ReadVariableOp,^conv1d_6/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_7/BiasAdd/ReadVariableOp,^conv1d_7/conv1d/ExpandDims_1/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp<^token_and_position_embedding_3/embedding_6/embedding_lookup<^token_and_position_embedding_3/embedding_7/embedding_lookupD^transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpH^transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpD^transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpH^transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpO^transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpY^transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_7/multi_head_attention_7/key/add/ReadVariableOpL^transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpD^transformer_block_7/multi_head_attention_7/query/add/ReadVariableOpN^transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpD^transformer_block_7/multi_head_attention_7/value/add/ReadVariableOpN^transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpA^transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpC^transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpA^transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpC^transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Μ
_input_shapesΊ
·:?????????R:?????????::::::::::::::::::::::::::::::::::::2`
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
Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpNtransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp2΄
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
Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpBtransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp:R N
(
_output_shapes
:?????????R
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
Ρ
γ
D__inference_dense_24_layer_call_and_return_conditional_losses_305499

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Tensordot/ReadVariableOp
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
Tensordot/GatherV2/axisΡ
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
Tensordot/GatherV2_1/axisΧ
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
:?????????#@2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape
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
:?????????# 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2	
BiasAdd
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

r
V__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_308510

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
Έ
 
-__inference_sequential_7_layer_call_fn_308779

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall
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
GPU2*0J 8 *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_3055472
StatefulPartitionedCall
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
?
β
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_308292

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
identity’/layer_normalization_14/batchnorm/ReadVariableOp’3layer_normalization_14/batchnorm/mul/ReadVariableOp’/layer_normalization_15/batchnorm/ReadVariableOp’3layer_normalization_15/batchnorm/mul/ReadVariableOp’:multi_head_attention_7/attention_output/add/ReadVariableOp’Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp’-multi_head_attention_7/key/add/ReadVariableOp’7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp’/multi_head_attention_7/query/add/ReadVariableOp’9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp’/multi_head_attention_7/value/add/ReadVariableOp’9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp’,sequential_7/dense_23/BiasAdd/ReadVariableOp’.sequential_7/dense_23/Tensordot/ReadVariableOp’,sequential_7/dense_24/BiasAdd/ReadVariableOp’.sequential_7/dense_24/Tensordot/ReadVariableOpύ
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_7/query/einsum/EinsumEinsuminputsAmulti_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2,
*multi_head_attention_7/query/einsum/EinsumΫ
/multi_head_attention_7/query/add/ReadVariableOpReadVariableOp8multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/query/add/ReadVariableOpυ
 multi_head_attention_7/query/addAddV23multi_head_attention_7/query/einsum/Einsum:output:07multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2"
 multi_head_attention_7/query/addχ
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_7/key/einsum/EinsumEinsuminputs?multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2*
(multi_head_attention_7/key/einsum/EinsumΥ
-multi_head_attention_7/key/add/ReadVariableOpReadVariableOp6multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_7/key/add/ReadVariableOpν
multi_head_attention_7/key/addAddV21multi_head_attention_7/key/einsum/Einsum:output:05multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2 
multi_head_attention_7/key/addύ
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_7/value/einsum/EinsumEinsuminputsAmulti_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2,
*multi_head_attention_7/value/einsum/EinsumΫ
/multi_head_attention_7/value/add/ReadVariableOpReadVariableOp8multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/value/add/ReadVariableOpυ
 multi_head_attention_7/value/addAddV23multi_head_attention_7/value/einsum/Einsum:output:07multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2"
 multi_head_attention_7/value/add
multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *σ5>2
multi_head_attention_7/Mul/yΖ
multi_head_attention_7/MulMul$multi_head_attention_7/query/add:z:0%multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:?????????# 2
multi_head_attention_7/Mulό
$multi_head_attention_7/einsum/EinsumEinsum"multi_head_attention_7/key/add:z:0multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:?????????##*
equationaecd,abcd->acbe2&
$multi_head_attention_7/einsum/EinsumΔ
&multi_head_attention_7/softmax/SoftmaxSoftmax-multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????##2(
&multi_head_attention_7/softmax/Softmax‘
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
:?????????##2,
*multi_head_attention_7/dropout/dropout/MulΌ
,multi_head_attention_7/dropout/dropout/ShapeShape0multi_head_attention_7/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_7/dropout/dropout/Shape
Cmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_7/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????##*
dtype02E
Cmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniform³
5multi_head_attention_7/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_7/dropout/dropout/GreaterEqual/yΒ
3multi_head_attention_7/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_7/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????##25
3multi_head_attention_7/dropout/dropout/GreaterEqualδ
+multi_head_attention_7/dropout/dropout/CastCast7multi_head_attention_7/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????##2-
+multi_head_attention_7/dropout/dropout/Castώ
,multi_head_attention_7/dropout/dropout/Mul_1Mul.multi_head_attention_7/dropout/dropout/Mul:z:0/multi_head_attention_7/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????##2.
,multi_head_attention_7/dropout/dropout/Mul_1
&multi_head_attention_7/einsum_1/EinsumEinsum0multi_head_attention_7/dropout/dropout/Mul_1:z:0$multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:?????????# *
equationacbe,aecd->abcd2(
&multi_head_attention_7/einsum_1/Einsum
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpΣ
5multi_head_attention_7/attention_output/einsum/EinsumEinsum/multi_head_attention_7/einsum_1/Einsum:output:0Lmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????# *
equationabcd,cde->abe27
5multi_head_attention_7/attention_output/einsum/Einsumψ
:multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_7/attention_output/add/ReadVariableOp
+multi_head_attention_7/attention_output/addAddV2>multi_head_attention_7/attention_output/einsum/Einsum:output:0Bmulti_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2-
+multi_head_attention_7/attention_output/addy
dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?2
dropout_20/dropout/ConstΑ
dropout_20/dropout/MulMul/multi_head_attention_7/attention_output/add:z:0!dropout_20/dropout/Const:output:0*
T0*+
_output_shapes
:?????????# 2
dropout_20/dropout/Mul
dropout_20/dropout/ShapeShape/multi_head_attention_7/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_20/dropout/ShapeΩ
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????# *
dtype021
/dropout_20/dropout/random_uniform/RandomUniform
!dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2#
!dropout_20/dropout/GreaterEqual/yξ
dropout_20/dropout/GreaterEqualGreaterEqual8dropout_20/dropout/random_uniform/RandomUniform:output:0*dropout_20/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????# 2!
dropout_20/dropout/GreaterEqual€
dropout_20/dropout/CastCast#dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????# 2
dropout_20/dropout/Castͺ
dropout_20/dropout/Mul_1Muldropout_20/dropout/Mul:z:0dropout_20/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????# 2
dropout_20/dropout/Mul_1o
addAddV2inputsdropout_20/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????# 2
addΈ
5layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_14/moments/mean/reduction_indicesβ
#layer_normalization_14/moments/meanMeanadd:z:0>layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2%
#layer_normalization_14/moments/meanΞ
+layer_normalization_14/moments/StopGradientStopGradient,layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2-
+layer_normalization_14/moments/StopGradientξ
0layer_normalization_14/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 22
0layer_normalization_14/moments/SquaredDifferenceΐ
9layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_14/moments/variance/reduction_indices
'layer_normalization_14/moments/varianceMean4layer_normalization_14/moments/SquaredDifference:z:0Blayer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2)
'layer_normalization_14/moments/variance
&layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_14/batchnorm/add/yξ
$layer_normalization_14/batchnorm/addAddV20layer_normalization_14/moments/variance:output:0/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2&
$layer_normalization_14/batchnorm/addΉ
&layer_normalization_14/batchnorm/RsqrtRsqrt(layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2(
&layer_normalization_14/batchnorm/Rsqrtγ
3layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_14/batchnorm/mul/ReadVariableOpς
$layer_normalization_14/batchnorm/mulMul*layer_normalization_14/batchnorm/Rsqrt:y:0;layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2&
$layer_normalization_14/batchnorm/mulΐ
&layer_normalization_14/batchnorm/mul_1Muladd:z:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_14/batchnorm/mul_1ε
&layer_normalization_14/batchnorm/mul_2Mul,layer_normalization_14/moments/mean:output:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_14/batchnorm/mul_2Χ
/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_14/batchnorm/ReadVariableOpξ
$layer_normalization_14/batchnorm/subSub7layer_normalization_14/batchnorm/ReadVariableOp:value:0*layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2&
$layer_normalization_14/batchnorm/subε
&layer_normalization_14/batchnorm/add_1AddV2*layer_normalization_14/batchnorm/mul_1:z:0(layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_14/batchnorm/add_1Ψ
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
-sequential_7/dense_23/Tensordot/GatherV2/axisΏ
(sequential_7/dense_23/Tensordot/GatherV2GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/free:output:06sequential_7/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/GatherV2€
/sequential_7/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_23/Tensordot/GatherV2_1/axisΕ
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
%sequential_7/dense_23/Tensordot/ConstΨ
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
'sequential_7/dense_23/Tensordot/Const_1ΰ
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
&sequential_7/dense_23/Tensordot/concatδ
%sequential_7/dense_23/Tensordot/stackPack-sequential_7/dense_23/Tensordot/Prod:output:0/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/stackφ
)sequential_7/dense_23/Tensordot/transpose	Transpose*layer_normalization_14/batchnorm/add_1:z:0/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????# 2+
)sequential_7/dense_23/Tensordot/transposeχ
'sequential_7/dense_23/Tensordot/ReshapeReshape-sequential_7/dense_23/Tensordot/transpose:y:0.sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_7/dense_23/Tensordot/Reshapeφ
&sequential_7/dense_23/Tensordot/MatMulMatMul0sequential_7/dense_23/Tensordot/Reshape:output:06sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2(
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
(sequential_7/dense_23/Tensordot/concat_1θ
sequential_7/dense_23/TensordotReshape0sequential_7/dense_23/Tensordot/MatMul:product:01sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@2!
sequential_7/dense_23/TensordotΞ
,sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_7/dense_23/BiasAdd/ReadVariableOpί
sequential_7/dense_23/BiasAddBiasAdd(sequential_7/dense_23/Tensordot:output:04sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@2
sequential_7/dense_23/BiasAdd
sequential_7/dense_23/ReluRelu&sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#@2
sequential_7/dense_23/ReluΨ
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
-sequential_7/dense_24/Tensordot/GatherV2/axisΏ
(sequential_7/dense_24/Tensordot/GatherV2GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/free:output:06sequential_7/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/GatherV2€
/sequential_7/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_24/Tensordot/GatherV2_1/axisΕ
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
%sequential_7/dense_24/Tensordot/ConstΨ
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
'sequential_7/dense_24/Tensordot/Const_1ΰ
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
&sequential_7/dense_24/Tensordot/concatδ
%sequential_7/dense_24/Tensordot/stackPack-sequential_7/dense_24/Tensordot/Prod:output:0/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/stackτ
)sequential_7/dense_24/Tensordot/transpose	Transpose(sequential_7/dense_23/Relu:activations:0/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2+
)sequential_7/dense_24/Tensordot/transposeχ
'sequential_7/dense_24/Tensordot/ReshapeReshape-sequential_7/dense_24/Tensordot/transpose:y:0.sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_7/dense_24/Tensordot/Reshapeφ
&sequential_7/dense_24/Tensordot/MatMulMatMul0sequential_7/dense_24/Tensordot/Reshape:output:06sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
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
(sequential_7/dense_24/Tensordot/concat_1θ
sequential_7/dense_24/TensordotReshape0sequential_7/dense_24/Tensordot/MatMul:product:01sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 2!
sequential_7/dense_24/TensordotΞ
,sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_7/dense_24/BiasAdd/ReadVariableOpί
sequential_7/dense_24/BiasAddBiasAdd(sequential_7/dense_24/Tensordot:output:04sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2
sequential_7/dense_24/BiasAddy
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?2
dropout_21/dropout/ConstΈ
dropout_21/dropout/MulMul&sequential_7/dense_24/BiasAdd:output:0!dropout_21/dropout/Const:output:0*
T0*+
_output_shapes
:?????????# 2
dropout_21/dropout/Mul
dropout_21/dropout/ShapeShape&sequential_7/dense_24/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_21/dropout/ShapeΩ
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????# *
dtype021
/dropout_21/dropout/random_uniform/RandomUniform
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2#
!dropout_21/dropout/GreaterEqual/yξ
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????# 2!
dropout_21/dropout/GreaterEqual€
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????# 2
dropout_21/dropout/Castͺ
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????# 2
dropout_21/dropout/Mul_1
add_1AddV2*layer_normalization_14/batchnorm/add_1:z:0dropout_21/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????# 2
add_1Έ
5layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_15/moments/mean/reduction_indicesδ
#layer_normalization_15/moments/meanMean	add_1:z:0>layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2%
#layer_normalization_15/moments/meanΞ
+layer_normalization_15/moments/StopGradientStopGradient,layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2-
+layer_normalization_15/moments/StopGradientπ
0layer_normalization_15/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 22
0layer_normalization_15/moments/SquaredDifferenceΐ
9layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_15/moments/variance/reduction_indices
'layer_normalization_15/moments/varianceMean4layer_normalization_15/moments/SquaredDifference:z:0Blayer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2)
'layer_normalization_15/moments/variance
&layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_15/batchnorm/add/yξ
$layer_normalization_15/batchnorm/addAddV20layer_normalization_15/moments/variance:output:0/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2&
$layer_normalization_15/batchnorm/addΉ
&layer_normalization_15/batchnorm/RsqrtRsqrt(layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2(
&layer_normalization_15/batchnorm/Rsqrtγ
3layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_15/batchnorm/mul/ReadVariableOpς
$layer_normalization_15/batchnorm/mulMul*layer_normalization_15/batchnorm/Rsqrt:y:0;layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2&
$layer_normalization_15/batchnorm/mulΒ
&layer_normalization_15/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_15/batchnorm/mul_1ε
&layer_normalization_15/batchnorm/mul_2Mul,layer_normalization_15/moments/mean:output:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_15/batchnorm/mul_2Χ
/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_15/batchnorm/ReadVariableOpξ
$layer_normalization_15/batchnorm/subSub7layer_normalization_15/batchnorm/ReadVariableOp:value:0*layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2&
$layer_normalization_15/batchnorm/subε
&layer_normalization_15/batchnorm/add_1AddV2*layer_normalization_15/batchnorm/mul_1:z:0(layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2(
&layer_normalization_15/batchnorm/add_1ά
IdentityIdentity*layer_normalization_15/batchnorm/add_1:z:00^layer_normalization_14/batchnorm/ReadVariableOp4^layer_normalization_14/batchnorm/mul/ReadVariableOp0^layer_normalization_15/batchnorm/ReadVariableOp4^layer_normalization_15/batchnorm/mul/ReadVariableOp;^multi_head_attention_7/attention_output/add/ReadVariableOpE^multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_7/key/add/ReadVariableOp8^multi_head_attention_7/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/query/add/ReadVariableOp:^multi_head_attention_7/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/value/add/ReadVariableOp:^multi_head_attention_7/value/einsum/Einsum/ReadVariableOp-^sequential_7/dense_23/BiasAdd/ReadVariableOp/^sequential_7/dense_23/Tensordot/ReadVariableOp-^sequential_7/dense_24/BiasAdd/ReadVariableOp/^sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:?????????# ::::::::::::::::2b
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
:?????????# 
 
_user_specified_nameinputs
’\

C__inference_model_3_layer_call_and_return_conditional_losses_306847

inputs
inputs_1)
%token_and_position_embedding_3_306756)
%token_and_position_embedding_3_306758
conv1d_6_306761
conv1d_6_306763
conv1d_7_306767
conv1d_7_306769 
batch_normalization_6_306774 
batch_normalization_6_306776 
batch_normalization_6_306778 
batch_normalization_6_306780 
batch_normalization_7_306783 
batch_normalization_7_306785 
batch_normalization_7_306787 
batch_normalization_7_306789
transformer_block_7_306793
transformer_block_7_306795
transformer_block_7_306797
transformer_block_7_306799
transformer_block_7_306801
transformer_block_7_306803
transformer_block_7_306805
transformer_block_7_306807
transformer_block_7_306809
transformer_block_7_306811
transformer_block_7_306813
transformer_block_7_306815
transformer_block_7_306817
transformer_block_7_306819
transformer_block_7_306821
transformer_block_7_306823
dense_25_306829
dense_25_306831
dense_26_306835
dense_26_306837
dense_27_306841
dense_27_306843
identity’-batch_normalization_6/StatefulPartitionedCall’-batch_normalization_7/StatefulPartitionedCall’ conv1d_6/StatefulPartitionedCall’ conv1d_7/StatefulPartitionedCall’ dense_25/StatefulPartitionedCall’ dense_26/StatefulPartitionedCall’ dense_27/StatefulPartitionedCall’6token_and_position_embedding_3/StatefulPartitionedCall’+transformer_block_7/StatefulPartitionedCall
6token_and_position_embedding_3/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_3_306756%token_and_position_embedding_3_306758*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_30563328
6token_and_position_embedding_3/StatefulPartitionedCallΥ
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0conv1d_6_306761conv1d_6_306763*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_3056652"
 conv1d_6/StatefulPartitionedCall 
#average_pooling1d_9/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????ή * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_3051022%
#average_pooling1d_9/PartitionedCallΒ
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_9/PartitionedCall:output:0conv1d_7_306767conv1d_7_306769*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????ή *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_3056982"
 conv1d_7/StatefulPartitionedCallΈ
$average_pooling1d_11/PartitionedCallPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_3051322&
$average_pooling1d_11/PartitionedCall’
$average_pooling1d_10/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_3051172&
$average_pooling1d_10/PartitionedCallΓ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_10/PartitionedCall:output:0batch_normalization_6_306774batch_normalization_6_306776batch_normalization_6_306778batch_normalization_6_306780*
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
GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3057712/
-batch_normalization_6/StatefulPartitionedCallΓ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_11/PartitionedCall:output:0batch_normalization_7_306783batch_normalization_7_306785batch_normalization_7_306787batch_normalization_7_306789*
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
GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3058622/
-batch_normalization_7/StatefulPartitionedCall»
add_3/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8 *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_3059042
add_3/PartitionedCall
+transformer_block_7/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0transformer_block_7_306793transformer_block_7_306795transformer_block_7_306797transformer_block_7_306799transformer_block_7_306801transformer_block_7_306803transformer_block_7_306805transformer_block_7_306807transformer_block_7_306809transformer_block_7_306811transformer_block_7_306813transformer_block_7_306815transformer_block_7_306817transformer_block_7_306819transformer_block_7_306821transformer_block_7_306823*
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
GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_3061882-
+transformer_block_7/StatefulPartitionedCall»
*global_average_pooling1d_3/PartitionedCallPartitionedCall4transformer_block_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_3063022,
*global_average_pooling1d_3/PartitionedCall
flatten_3/PartitionedCallPartitionedCall3global_average_pooling1d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_3063152
flatten_3/PartitionedCall
concatenate_3/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_3063302
concatenate_3/PartitionedCall·
 dense_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_25_306829dense_25_306831*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_3063502"
 dense_25/StatefulPartitionedCall
dropout_22/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_3063832
dropout_22/PartitionedCall΄
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_26_306835dense_26_306837*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_3064072"
 dense_26/StatefulPartitionedCall
dropout_23/PartitionedCallPartitionedCall)dense_26/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_3064402
dropout_23/PartitionedCall΄
 dense_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_27_306841dense_27_306843*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_3064632"
 dense_27/StatefulPartitionedCallσ
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall7^token_and_position_embedding_3/StatefulPartitionedCall,^transformer_block_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Μ
_input_shapesΊ
·:?????????R:?????????::::::::::::::::::::::::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2p
6token_and_position_embedding_3/StatefulPartitionedCall6token_and_position_embedding_3/StatefulPartitionedCall2Z
+transformer_block_7/StatefulPartitionedCall+transformer_block_7/StatefulPartitionedCall:P L
(
_output_shapes
:?????????R
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
μ
©
6__inference_batch_normalization_7_layer_call_fn_308037

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3053742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
σ0
Θ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_305374

inputs
assignmovingavg_305349
assignmovingavg_1_305355)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOp
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
 :?????????????????? 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
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
moments/Squeeze_1Μ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/305349*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_305349*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpρ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/305349*
_output_shapes
: 2
AssignMovingAvg/subθ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/305349*
_output_shapes
: 2
AssignMovingAvg/mul―
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_305349AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/305349*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/305355*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_305355*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpϋ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/305355*
_output_shapes
: 2
AssignMovingAvg_1/subς
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/305355*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_305355AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/305355*
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
 :?????????????????? 2
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
 :?????????????????? 2
batchnorm/add_1ΐ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????????????? ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
υ
k
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_305102

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
-:+???????????????????????????2

ExpandDimsΊ
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool
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
ξ
©
6__inference_batch_normalization_7_layer_call_fn_308050

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3054072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs


H__inference_sequential_7_layer_call_and_return_conditional_losses_305516
dense_23_input
dense_23_305464
dense_23_305466
dense_24_305510
dense_24_305512
identity’ dense_23/StatefulPartitionedCall’ dense_24/StatefulPartitionedCall£
 dense_23/StatefulPartitionedCallStatefulPartitionedCalldense_23_inputdense_23_305464dense_23_305466*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_3054532"
 dense_23/StatefulPartitionedCallΎ
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_305510dense_24_305512*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_3054992"
 dense_24/StatefulPartitionedCallΗ
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:[ W
+
_output_shapes
:?????????# 
(
_user_specified_namedense_23_input
­_
θ
C__inference_model_3_layer_call_and_return_conditional_losses_306480
input_7
input_8)
%token_and_position_embedding_3_305644)
%token_and_position_embedding_3_305646
conv1d_6_305676
conv1d_6_305678
conv1d_7_305709
conv1d_7_305711 
batch_normalization_6_305798 
batch_normalization_6_305800 
batch_normalization_6_305802 
batch_normalization_6_305804 
batch_normalization_7_305889 
batch_normalization_7_305891 
batch_normalization_7_305893 
batch_normalization_7_305895
transformer_block_7_306264
transformer_block_7_306266
transformer_block_7_306268
transformer_block_7_306270
transformer_block_7_306272
transformer_block_7_306274
transformer_block_7_306276
transformer_block_7_306278
transformer_block_7_306280
transformer_block_7_306282
transformer_block_7_306284
transformer_block_7_306286
transformer_block_7_306288
transformer_block_7_306290
transformer_block_7_306292
transformer_block_7_306294
dense_25_306361
dense_25_306363
dense_26_306418
dense_26_306420
dense_27_306474
dense_27_306476
identity’-batch_normalization_6/StatefulPartitionedCall’-batch_normalization_7/StatefulPartitionedCall’ conv1d_6/StatefulPartitionedCall’ conv1d_7/StatefulPartitionedCall’ dense_25/StatefulPartitionedCall’ dense_26/StatefulPartitionedCall’ dense_27/StatefulPartitionedCall’"dropout_22/StatefulPartitionedCall’"dropout_23/StatefulPartitionedCall’6token_and_position_embedding_3/StatefulPartitionedCall’+transformer_block_7/StatefulPartitionedCall
6token_and_position_embedding_3/StatefulPartitionedCallStatefulPartitionedCallinput_7%token_and_position_embedding_3_305644%token_and_position_embedding_3_305646*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_30563328
6token_and_position_embedding_3/StatefulPartitionedCallΥ
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0conv1d_6_305676conv1d_6_305678*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_3056652"
 conv1d_6/StatefulPartitionedCall 
#average_pooling1d_9/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????ή * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_3051022%
#average_pooling1d_9/PartitionedCallΒ
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_9/PartitionedCall:output:0conv1d_7_305709conv1d_7_305711*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????ή *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_3056982"
 conv1d_7/StatefulPartitionedCallΈ
$average_pooling1d_11/PartitionedCallPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_3051322&
$average_pooling1d_11/PartitionedCall’
$average_pooling1d_10/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_3051172&
$average_pooling1d_10/PartitionedCallΑ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_10/PartitionedCall:output:0batch_normalization_6_305798batch_normalization_6_305800batch_normalization_6_305802batch_normalization_6_305804*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3057512/
-batch_normalization_6/StatefulPartitionedCallΑ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_11/PartitionedCall:output:0batch_normalization_7_305889batch_normalization_7_305891batch_normalization_7_305893batch_normalization_7_305895*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3058422/
-batch_normalization_7/StatefulPartitionedCall»
add_3/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8 *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_3059042
add_3/PartitionedCall
+transformer_block_7/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0transformer_block_7_306264transformer_block_7_306266transformer_block_7_306268transformer_block_7_306270transformer_block_7_306272transformer_block_7_306274transformer_block_7_306276transformer_block_7_306278transformer_block_7_306280transformer_block_7_306282transformer_block_7_306284transformer_block_7_306286transformer_block_7_306288transformer_block_7_306290transformer_block_7_306292transformer_block_7_306294*
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
GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_3060612-
+transformer_block_7/StatefulPartitionedCall»
*global_average_pooling1d_3/PartitionedCallPartitionedCall4transformer_block_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_3063022,
*global_average_pooling1d_3/PartitionedCall
flatten_3/PartitionedCallPartitionedCall3global_average_pooling1d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_3063152
flatten_3/PartitionedCall
concatenate_3/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0input_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_3063302
concatenate_3/PartitionedCall·
 dense_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_25_306361dense_25_306363*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_3063502"
 dense_25/StatefulPartitionedCall
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_3063782$
"dropout_22/StatefulPartitionedCallΌ
 dense_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_26_306418dense_26_306420*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_3064072"
 dense_26/StatefulPartitionedCall½
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
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
GPU2*0J 8 *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_3064352$
"dropout_23/StatefulPartitionedCallΌ
 dense_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_27_306474dense_27_306476*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_3064632"
 dense_27/StatefulPartitionedCall½
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall7^token_and_position_embedding_3/StatefulPartitionedCall,^transformer_block_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Μ
_input_shapesΊ
·:?????????R:?????????::::::::::::::::::::::::::::::::::::2^
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
(
_output_shapes
:?????????R
!
_user_specified_name	input_7:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_8

r
V__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_305601

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
τ

Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_305633
x'
#embedding_7_embedding_lookup_305620'
#embedding_6_embedding_lookup_305626
identity’embedding_6/embedding_lookup’embedding_7/embedding_lookup?
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
strided_slice/stack_2β
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
:?????????2
range―
embedding_7/embedding_lookupResourceGather#embedding_7_embedding_lookup_305620range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_7/embedding_lookup/305620*'
_output_shapes
:????????? *
dtype02
embedding_7/embedding_lookup
%embedding_7/embedding_lookup/IdentityIdentity%embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_7/embedding_lookup/305620*'
_output_shapes
:????????? 2'
%embedding_7/embedding_lookup/Identityΐ
'embedding_7/embedding_lookup/Identity_1Identity.embedding_7/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2)
'embedding_7/embedding_lookup/Identity_1q
embedding_6/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:?????????R2
embedding_6/CastΊ
embedding_6/embedding_lookupResourceGather#embedding_6_embedding_lookup_305626embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_6/embedding_lookup/305626*,
_output_shapes
:?????????R *
dtype02
embedding_6/embedding_lookup
%embedding_6/embedding_lookup/IdentityIdentity%embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_6/embedding_lookup/305626*,
_output_shapes
:?????????R 2'
%embedding_6/embedding_lookup/IdentityΕ
'embedding_6/embedding_lookup/Identity_1Identity.embedding_6/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:?????????R 2)
'embedding_6/embedding_lookup/Identity_1?
addAddV20embedding_6/embedding_lookup/Identity_1:output:00embedding_7/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:?????????R 2
add
IdentityIdentityadd:z:0^embedding_6/embedding_lookup^embedding_7/embedding_lookup*
T0*,
_output_shapes
:?????????R 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????R::2<
embedding_6/embedding_lookupembedding_6/embedding_lookup2<
embedding_7/embedding_lookupembedding_7/embedding_lookup:K G
(
_output_shapes
:?????????R

_user_specified_namex
Ό0
Θ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_305842

inputs
assignmovingavg_305817
assignmovingavg_1_305823)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOp
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
:?????????# 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
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
moments/Squeeze_1Μ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/305817*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_305817*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpρ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/305817*
_output_shapes
: 2
AssignMovingAvg/subθ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/305817*
_output_shapes
: 2
AssignMovingAvg/mul―
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_305817AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/305817*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/305823*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_305823*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpϋ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/305823*
_output_shapes
: 2
AssignMovingAvg_1/subς
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/305823*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_305823AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/305823*
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
:?????????# 2
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
:?????????# 2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
Π
’
(__inference_model_3_layer_call_fn_306749
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
identity’StatefulPartitionedCallΠ
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
:?????????*B
_read_only_resource_inputs$
" 
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_3066742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Μ
_input_shapesΊ
·:?????????R:?????????::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:?????????R
!
_user_specified_name	input_7:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_8
€
Z
.__inference_concatenate_3_layer_call_fn_308539
inputs_0
inputs_1
identityΧ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_3063302
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????? :?????????:Q M
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
Ϊ
€
(__inference_model_3_layer_call_fn_307721
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
identity’StatefulPartitionedCallΦ
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
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_3068472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Μ
_input_shapesΊ
·:?????????R:?????????::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:?????????R
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
σ0
Θ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_305234

inputs
assignmovingavg_305209
assignmovingavg_1_305215)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOp
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
 :?????????????????? 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
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
moments/Squeeze_1Μ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/305209*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_305209*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpρ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/305209*
_output_shapes
: 2
AssignMovingAvg/subθ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/305209*
_output_shapes
: 2
AssignMovingAvg/mul―
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_305209AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/305209*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/305215*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_305215*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpϋ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/305215*
_output_shapes
: 2
AssignMovingAvg_1/subς
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/305215*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_305215AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/305215*
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
 :?????????????????? 2
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
 :?????????????????? 2
batchnorm/add_1ΐ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????????????? ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs

e
F__inference_dropout_22_layer_call_and_return_conditional_losses_306378

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?2
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
dropout/Shape΄
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
 *ΝΜΜ=2
dropout/GreaterEqual/yΎ
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
Θ
©
6__inference_batch_normalization_6_layer_call_fn_307955

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3057512
StatefulPartitionedCall
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
θ

Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_307942

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOp
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
:?????????# 2
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
:?????????# 2
batchnorm/add_1ί
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
J
―
H__inference_sequential_7_layer_call_and_return_conditional_losses_308766

inputs.
*dense_23_tensordot_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource.
*dense_24_tensordot_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource
identity’dense_23/BiasAdd/ReadVariableOp’!dense_23/Tensordot/ReadVariableOp’dense_24/BiasAdd/ReadVariableOp’!dense_24/Tensordot/ReadVariableOp±
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
 dense_23/Tensordot/GatherV2/axisώ
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
dense_23/Tensordot/Const€
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
dense_23/Tensordot/concat/axisέ
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
:?????????# 2
dense_23/Tensordot/transposeΓ
dense_23/Tensordot/ReshapeReshape dense_23/Tensordot/transpose:y:0!dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_23/Tensordot/ReshapeΒ
dense_23/Tensordot/MatMulMatMul#dense_23/Tensordot/Reshape:output:0)dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
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
 dense_23/Tensordot/concat_1/axisκ
dense_23/Tensordot/concat_1ConcatV2$dense_23/Tensordot/GatherV2:output:0#dense_23/Tensordot/Const_2:output:0)dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/concat_1΄
dense_23/TensordotReshape#dense_23/Tensordot/MatMul:product:0$dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@2
dense_23/Tensordot§
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_23/BiasAdd/ReadVariableOp«
dense_23/BiasAddBiasAdddense_23/Tensordot:output:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@2
dense_23/BiasAddw
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#@2
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
 dense_24/Tensordot/GatherV2/axisώ
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
dense_24/Tensordot/Const€
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
dense_24/Tensordot/concat/axisέ
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
dense_24/Tensordot/stackΐ
dense_24/Tensordot/transpose	Transposedense_23/Relu:activations:0"dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2
dense_24/Tensordot/transposeΓ
dense_24/Tensordot/ReshapeReshape dense_24/Tensordot/transpose:y:0!dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_24/Tensordot/ReshapeΒ
dense_24/Tensordot/MatMulMatMul#dense_24/Tensordot/Reshape:output:0)dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
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
 dense_24/Tensordot/concat_1/axisκ
dense_24/Tensordot/concat_1ConcatV2$dense_24/Tensordot/GatherV2:output:0#dense_24/Tensordot/Const_2:output:0)dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/concat_1΄
dense_24/TensordotReshape#dense_24/Tensordot/MatMul:product:0$dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 2
dense_24/Tensordot§
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_24/BiasAdd/ReadVariableOp«
dense_24/BiasAddBiasAdddense_24/Tensordot:output:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2
dense_24/BiasAddύ
IdentityIdentitydense_24/BiasAdd:output:0 ^dense_23/BiasAdd/ReadVariableOp"^dense_23/Tensordot/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp"^dense_24/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2F
!dense_23/Tensordot/ReadVariableOp!dense_23/Tensordot/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2F
!dense_24/Tensordot/ReadVariableOp!dense_24/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
φ
l
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_305117

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
-:+???????????????????????????2

ExpandDimsΊ
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+???????????????????????????*
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
Ώ
m
A__inference_add_3_layer_call_and_return_conditional_losses_308138
inputs_0
inputs_1
identity]
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:?????????# 2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:?????????# :?????????# :U Q
+
_output_shapes
:?????????# 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????# 
"
_user_specified_name
inputs/1


?__inference_token_and_position_embedding_3_layer_call_fn_307754
x
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_3056332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????R 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????R::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:?????????R

_user_specified_namex
Κ
©
6__inference_batch_normalization_6_layer_call_fn_307968

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall’
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
GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3057712
StatefulPartitionedCall
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
Π
¨
-__inference_sequential_7_layer_call_fn_305585
dense_23_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCalldense_23_inputunknown	unknown_0	unknown_1	unknown_2*
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
GPU2*0J 8 *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_3055742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:?????????# 
(
_user_specified_namedense_23_input

χ
D__inference_conv1d_6_layer_call_and_return_conditional_losses_307770

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????R 2
conv1d/ExpandDimsΈ
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
:?????????R *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:?????????R *
squeeze_dims

ύ????????2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????R 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????R 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:?????????R 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????R ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????R 
 
_user_specified_nameinputs
Τ
’
(__inference_model_3_layer_call_fn_306922
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
identity’StatefulPartitionedCallΤ
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
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_3068472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Μ
_input_shapesΊ
·:?????????R:?????????::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:?????????R
!
_user_specified_name	input_7:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_8
Ι
d
F__inference_dropout_23_layer_call_and_return_conditional_losses_308623

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
ί
~
)__inference_dense_27_layer_call_fn_308652

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallχ
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
GPU2*0J 8 *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_3064632
StatefulPartitionedCall
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
Κ
θ(
!__inference__wrapped_model_305093
input_7
input_8N
Jmodel_3_token_and_position_embedding_3_embedding_7_embedding_lookup_304860N
Jmodel_3_token_and_position_embedding_3_embedding_6_embedding_lookup_304866@
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
identity’6model_3/batch_normalization_6/batchnorm/ReadVariableOp’8model_3/batch_normalization_6/batchnorm/ReadVariableOp_1’8model_3/batch_normalization_6/batchnorm/ReadVariableOp_2’:model_3/batch_normalization_6/batchnorm/mul/ReadVariableOp’6model_3/batch_normalization_7/batchnorm/ReadVariableOp’8model_3/batch_normalization_7/batchnorm/ReadVariableOp_1’8model_3/batch_normalization_7/batchnorm/ReadVariableOp_2’:model_3/batch_normalization_7/batchnorm/mul/ReadVariableOp’'model_3/conv1d_6/BiasAdd/ReadVariableOp’3model_3/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp’'model_3/conv1d_7/BiasAdd/ReadVariableOp’3model_3/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp’'model_3/dense_25/BiasAdd/ReadVariableOp’&model_3/dense_25/MatMul/ReadVariableOp’'model_3/dense_26/BiasAdd/ReadVariableOp’&model_3/dense_26/MatMul/ReadVariableOp’'model_3/dense_27/BiasAdd/ReadVariableOp’&model_3/dense_27/MatMul/ReadVariableOp’Cmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup’Cmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup’Kmodel_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp’Omodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp’Kmodel_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp’Omodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp’Vmodel_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp’`model_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp’Imodel_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOp’Smodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp’Kmodel_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOp’Umodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp’Kmodel_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOp’Umodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp’Hmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp’Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp’Hmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp’Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp
,model_3/token_and_position_embedding_3/ShapeShapeinput_7*
T0*
_output_shapes
:2.
,model_3/token_and_position_embedding_3/ShapeΛ
:model_3/token_and_position_embedding_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2<
:model_3/token_and_position_embedding_3/strided_slice/stackΖ
<model_3/token_and_position_embedding_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<model_3/token_and_position_embedding_3/strided_slice/stack_1Ζ
<model_3/token_and_position_embedding_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model_3/token_and_position_embedding_3/strided_slice/stack_2Μ
4model_3/token_and_position_embedding_3/strided_sliceStridedSlice5model_3/token_and_position_embedding_3/Shape:output:0Cmodel_3/token_and_position_embedding_3/strided_slice/stack:output:0Emodel_3/token_and_position_embedding_3/strided_slice/stack_1:output:0Emodel_3/token_and_position_embedding_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4model_3/token_and_position_embedding_3/strided_sliceͺ
2model_3/token_and_position_embedding_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : 24
2model_3/token_and_position_embedding_3/range/startͺ
2model_3/token_and_position_embedding_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :24
2model_3/token_and_position_embedding_3/range/deltaΓ
,model_3/token_and_position_embedding_3/rangeRange;model_3/token_and_position_embedding_3/range/start:output:0=model_3/token_and_position_embedding_3/strided_slice:output:0;model_3/token_and_position_embedding_3/range/delta:output:0*#
_output_shapes
:?????????2.
,model_3/token_and_position_embedding_3/rangeς
Cmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookupResourceGatherJmodel_3_token_and_position_embedding_3_embedding_7_embedding_lookup_3048605model_3/token_and_position_embedding_3/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*]
_classS
QOloc:@model_3/token_and_position_embedding_3/embedding_7/embedding_lookup/304860*'
_output_shapes
:????????? *
dtype02E
Cmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup΅
Lmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup/IdentityIdentityLmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*]
_classS
QOloc:@model_3/token_and_position_embedding_3/embedding_7/embedding_lookup/304860*'
_output_shapes
:????????? 2N
Lmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup/Identity΅
Nmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1IdentityUmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2P
Nmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1Ε
7model_3/token_and_position_embedding_3/embedding_6/CastCastinput_7*

DstT0*

SrcT0*(
_output_shapes
:?????????R29
7model_3/token_and_position_embedding_3/embedding_6/Castύ
Cmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookupResourceGatherJmodel_3_token_and_position_embedding_3_embedding_6_embedding_lookup_304866;model_3/token_and_position_embedding_3/embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*]
_classS
QOloc:@model_3/token_and_position_embedding_3/embedding_6/embedding_lookup/304866*,
_output_shapes
:?????????R *
dtype02E
Cmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookupΊ
Lmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup/IdentityIdentityLmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*]
_classS
QOloc:@model_3/token_and_position_embedding_3/embedding_6/embedding_lookup/304866*,
_output_shapes
:?????????R 2N
Lmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup/IdentityΊ
Nmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1IdentityUmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:?????????R 2P
Nmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1Κ
*model_3/token_and_position_embedding_3/addAddV2Wmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1:output:0Wmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:?????????R 2,
*model_3/token_and_position_embedding_3/add
&model_3/conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2(
&model_3/conv1d_6/conv1d/ExpandDims/dimς
"model_3/conv1d_6/conv1d/ExpandDims
ExpandDims.model_3/token_and_position_embedding_3/add:z:0/model_3/conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????R 2$
"model_3/conv1d_6/conv1d/ExpandDimsλ
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
(model_3/conv1d_6/conv1d/ExpandDims_1/dimϋ
$model_3/conv1d_6/conv1d/ExpandDims_1
ExpandDims;model_3/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:01model_3/conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2&
$model_3/conv1d_6/conv1d/ExpandDims_1ϋ
model_3/conv1d_6/conv1dConv2D+model_3/conv1d_6/conv1d/ExpandDims:output:0-model_3/conv1d_6/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????R *
paddingSAME*
strides
2
model_3/conv1d_6/conv1dΖ
model_3/conv1d_6/conv1d/SqueezeSqueeze model_3/conv1d_6/conv1d:output:0*
T0*,
_output_shapes
:?????????R *
squeeze_dims

ύ????????2!
model_3/conv1d_6/conv1d/SqueezeΏ
'model_3/conv1d_6/BiasAdd/ReadVariableOpReadVariableOp0model_3_conv1d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_3/conv1d_6/BiasAdd/ReadVariableOpΡ
model_3/conv1d_6/BiasAddBiasAdd(model_3/conv1d_6/conv1d/Squeeze:output:0/model_3/conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????R 2
model_3/conv1d_6/BiasAdd
model_3/conv1d_6/ReluRelu!model_3/conv1d_6/BiasAdd:output:0*
T0*,
_output_shapes
:?????????R 2
model_3/conv1d_6/Relu
*model_3/average_pooling1d_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_3/average_pooling1d_9/ExpandDims/dimσ
&model_3/average_pooling1d_9/ExpandDims
ExpandDims#model_3/conv1d_6/Relu:activations:03model_3/average_pooling1d_9/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????R 2(
&model_3/average_pooling1d_9/ExpandDimsύ
#model_3/average_pooling1d_9/AvgPoolAvgPool/model_3/average_pooling1d_9/ExpandDims:output:0*
T0*0
_output_shapes
:?????????ή *
ksize
*
paddingVALID*
strides
2%
#model_3/average_pooling1d_9/AvgPoolΡ
#model_3/average_pooling1d_9/SqueezeSqueeze,model_3/average_pooling1d_9/AvgPool:output:0*
T0*,
_output_shapes
:?????????ή *
squeeze_dims
2%
#model_3/average_pooling1d_9/Squeeze
&model_3/conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2(
&model_3/conv1d_7/conv1d/ExpandDims/dimπ
"model_3/conv1d_7/conv1d/ExpandDims
ExpandDims,model_3/average_pooling1d_9/Squeeze:output:0/model_3/conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ή 2$
"model_3/conv1d_7/conv1d/ExpandDimsλ
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
(model_3/conv1d_7/conv1d/ExpandDims_1/dimϋ
$model_3/conv1d_7/conv1d/ExpandDims_1
ExpandDims;model_3/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:01model_3/conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2&
$model_3/conv1d_7/conv1d/ExpandDims_1ϋ
model_3/conv1d_7/conv1dConv2D+model_3/conv1d_7/conv1d/ExpandDims:output:0-model_3/conv1d_7/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????ή *
paddingSAME*
strides
2
model_3/conv1d_7/conv1dΖ
model_3/conv1d_7/conv1d/SqueezeSqueeze model_3/conv1d_7/conv1d:output:0*
T0*,
_output_shapes
:?????????ή *
squeeze_dims

ύ????????2!
model_3/conv1d_7/conv1d/SqueezeΏ
'model_3/conv1d_7/BiasAdd/ReadVariableOpReadVariableOp0model_3_conv1d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_3/conv1d_7/BiasAdd/ReadVariableOpΡ
model_3/conv1d_7/BiasAddBiasAdd(model_3/conv1d_7/conv1d/Squeeze:output:0/model_3/conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ή 2
model_3/conv1d_7/BiasAdd
model_3/conv1d_7/ReluRelu!model_3/conv1d_7/BiasAdd:output:0*
T0*,
_output_shapes
:?????????ή 2
model_3/conv1d_7/Relu
+model_3/average_pooling1d_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+model_3/average_pooling1d_11/ExpandDims/dim
'model_3/average_pooling1d_11/ExpandDims
ExpandDims.model_3/token_and_position_embedding_3/add:z:04model_3/average_pooling1d_11/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????R 2)
'model_3/average_pooling1d_11/ExpandDims
$model_3/average_pooling1d_11/AvgPoolAvgPool0model_3/average_pooling1d_11/ExpandDims:output:0*
T0*/
_output_shapes
:?????????# *
ksize	
¬*
paddingVALID*
strides	
¬2&
$model_3/average_pooling1d_11/AvgPoolΣ
$model_3/average_pooling1d_11/SqueezeSqueeze-model_3/average_pooling1d_11/AvgPool:output:0*
T0*+
_output_shapes
:?????????# *
squeeze_dims
2&
$model_3/average_pooling1d_11/Squeeze
+model_3/average_pooling1d_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+model_3/average_pooling1d_10/ExpandDims/dimφ
'model_3/average_pooling1d_10/ExpandDims
ExpandDims#model_3/conv1d_7/Relu:activations:04model_3/average_pooling1d_10/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ή 2)
'model_3/average_pooling1d_10/ExpandDims?
$model_3/average_pooling1d_10/AvgPoolAvgPool0model_3/average_pooling1d_10/ExpandDims:output:0*
T0*/
_output_shapes
:?????????# *
ksize

*
paddingVALID*
strides

2&
$model_3/average_pooling1d_10/AvgPoolΣ
$model_3/average_pooling1d_10/SqueezeSqueeze-model_3/average_pooling1d_10/AvgPool:output:0*
T0*+
_output_shapes
:?????????# *
squeeze_dims
2&
$model_3/average_pooling1d_10/Squeezeμ
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
-model_3/batch_normalization_6/batchnorm/Rsqrtψ
:model_3/batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_3_batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:model_3/batch_normalization_6/batchnorm/mul/ReadVariableOpύ
+model_3/batch_normalization_6/batchnorm/mulMul1model_3/batch_normalization_6/batchnorm/Rsqrt:y:0Bmodel_3/batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+model_3/batch_normalization_6/batchnorm/mulϋ
-model_3/batch_normalization_6/batchnorm/mul_1Mul-model_3/average_pooling1d_10/Squeeze:output:0/model_3/batch_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2/
-model_3/batch_normalization_6/batchnorm/mul_1ς
8model_3/batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_3_batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model_3/batch_normalization_6/batchnorm/ReadVariableOp_1ύ
-model_3/batch_normalization_6/batchnorm/mul_2Mul@model_3/batch_normalization_6/batchnorm/ReadVariableOp_1:value:0/model_3/batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-model_3/batch_normalization_6/batchnorm/mul_2ς
8model_3/batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_3_batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8model_3/batch_normalization_6/batchnorm/ReadVariableOp_2ϋ
+model_3/batch_normalization_6/batchnorm/subSub@model_3/batch_normalization_6/batchnorm/ReadVariableOp_2:value:01model_3/batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+model_3/batch_normalization_6/batchnorm/sub
-model_3/batch_normalization_6/batchnorm/add_1AddV21model_3/batch_normalization_6/batchnorm/mul_1:z:0/model_3/batch_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2/
-model_3/batch_normalization_6/batchnorm/add_1μ
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
-model_3/batch_normalization_7/batchnorm/Rsqrtψ
:model_3/batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_3_batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:model_3/batch_normalization_7/batchnorm/mul/ReadVariableOpύ
+model_3/batch_normalization_7/batchnorm/mulMul1model_3/batch_normalization_7/batchnorm/Rsqrt:y:0Bmodel_3/batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+model_3/batch_normalization_7/batchnorm/mulϋ
-model_3/batch_normalization_7/batchnorm/mul_1Mul-model_3/average_pooling1d_11/Squeeze:output:0/model_3/batch_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2/
-model_3/batch_normalization_7/batchnorm/mul_1ς
8model_3/batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_3_batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model_3/batch_normalization_7/batchnorm/ReadVariableOp_1ύ
-model_3/batch_normalization_7/batchnorm/mul_2Mul@model_3/batch_normalization_7/batchnorm/ReadVariableOp_1:value:0/model_3/batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-model_3/batch_normalization_7/batchnorm/mul_2ς
8model_3/batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_3_batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8model_3/batch_normalization_7/batchnorm/ReadVariableOp_2ϋ
+model_3/batch_normalization_7/batchnorm/subSub@model_3/batch_normalization_7/batchnorm/ReadVariableOp_2:value:01model_3/batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+model_3/batch_normalization_7/batchnorm/sub
-model_3/batch_normalization_7/batchnorm/add_1AddV21model_3/batch_normalization_7/batchnorm/mul_1:z:0/model_3/batch_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2/
-model_3/batch_normalization_7/batchnorm/add_1Λ
model_3/add_3/addAddV21model_3/batch_normalization_6/batchnorm/add_1:z:01model_3/batch_normalization_7/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????# 2
model_3/add_3/addΡ
Umodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOp^model_3_transformer_block_7_multi_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpπ
Fmodel_3/transformer_block_7/multi_head_attention_7/query/einsum/EinsumEinsummodel_3/add_3/add:z:0]model_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2H
Fmodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum―
Kmodel_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOpReadVariableOpTmodel_3_transformer_block_7_multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOpε
<model_3/transformer_block_7/multi_head_attention_7/query/addAddV2Omodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum:output:0Smodel_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2>
<model_3/transformer_block_7/multi_head_attention_7/query/addΛ
Smodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOp\model_3_transformer_block_7_multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02U
Smodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpκ
Dmodel_3/transformer_block_7/multi_head_attention_7/key/einsum/EinsumEinsummodel_3/add_3/add:z:0[model_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2F
Dmodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum©
Imodel_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOpReadVariableOpRmodel_3_transformer_block_7_multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

: *
dtype02K
Imodel_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOpέ
:model_3/transformer_block_7/multi_head_attention_7/key/addAddV2Mmodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum:output:0Qmodel_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2<
:model_3/transformer_block_7/multi_head_attention_7/key/addΡ
Umodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOp^model_3_transformer_block_7_multi_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpπ
Fmodel_3/transformer_block_7/multi_head_attention_7/value/einsum/EinsumEinsummodel_3/add_3/add:z:0]model_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????# *
equationabc,cde->abde2H
Fmodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum―
Kmodel_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOpReadVariableOpTmodel_3_transformer_block_7_multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOpε
<model_3/transformer_block_7/multi_head_attention_7/value/addAddV2Omodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum:output:0Smodel_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????# 2>
<model_3/transformer_block_7/multi_head_attention_7/value/addΉ
8model_3/transformer_block_7/multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *σ5>2:
8model_3/transformer_block_7/multi_head_attention_7/Mul/yΆ
6model_3/transformer_block_7/multi_head_attention_7/MulMul@model_3/transformer_block_7/multi_head_attention_7/query/add:z:0Amodel_3/transformer_block_7/multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:?????????# 28
6model_3/transformer_block_7/multi_head_attention_7/Mulμ
@model_3/transformer_block_7/multi_head_attention_7/einsum/EinsumEinsum>model_3/transformer_block_7/multi_head_attention_7/key/add:z:0:model_3/transformer_block_7/multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:?????????##*
equationaecd,abcd->acbe2B
@model_3/transformer_block_7/multi_head_attention_7/einsum/Einsum
Bmodel_3/transformer_block_7/multi_head_attention_7/softmax/SoftmaxSoftmaxImodel_3/transformer_block_7/multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????##2D
Bmodel_3/transformer_block_7/multi_head_attention_7/softmax/Softmax
Cmodel_3/transformer_block_7/multi_head_attention_7/dropout/IdentityIdentityLmodel_3/transformer_block_7/multi_head_attention_7/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????##2E
Cmodel_3/transformer_block_7/multi_head_attention_7/dropout/Identity
Bmodel_3/transformer_block_7/multi_head_attention_7/einsum_1/EinsumEinsumLmodel_3/transformer_block_7/multi_head_attention_7/dropout/Identity:output:0@model_3/transformer_block_7/multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:?????????# *
equationacbe,aecd->abcd2D
Bmodel_3/transformer_block_7/multi_head_attention_7/einsum_1/Einsumς
`model_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpimodel_3_transformer_block_7_multi_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02b
`model_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpΓ
Qmodel_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/EinsumEinsumKmodel_3/transformer_block_7/multi_head_attention_7/einsum_1/Einsum:output:0hmodel_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????# *
equationabcd,cde->abe2S
Qmodel_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/EinsumΜ
Vmodel_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOp_model_3_transformer_block_7_multi_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02X
Vmodel_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp
Gmodel_3/transformer_block_7/multi_head_attention_7/attention_output/addAddV2Zmodel_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum:output:0^model_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2I
Gmodel_3/transformer_block_7/multi_head_attention_7/attention_output/addρ
/model_3/transformer_block_7/dropout_20/IdentityIdentityKmodel_3/transformer_block_7/multi_head_attention_7/attention_output/add:z:0*
T0*+
_output_shapes
:?????????# 21
/model_3/transformer_block_7/dropout_20/Identity?
model_3/transformer_block_7/addAddV2model_3/add_3/add:z:08model_3/transformer_block_7/dropout_20/Identity:output:0*
T0*+
_output_shapes
:?????????# 2!
model_3/transformer_block_7/addπ
Qmodel_3/transformer_block_7/layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_3/transformer_block_7/layer_normalization_14/moments/mean/reduction_indices?
?model_3/transformer_block_7/layer_normalization_14/moments/meanMean#model_3/transformer_block_7/add:z:0Zmodel_3/transformer_block_7/layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2A
?model_3/transformer_block_7/layer_normalization_14/moments/mean’
Gmodel_3/transformer_block_7/layer_normalization_14/moments/StopGradientStopGradientHmodel_3/transformer_block_7/layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2I
Gmodel_3/transformer_block_7/layer_normalization_14/moments/StopGradientή
Lmodel_3/transformer_block_7/layer_normalization_14/moments/SquaredDifferenceSquaredDifference#model_3/transformer_block_7/add:z:0Pmodel_3/transformer_block_7/layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 2N
Lmodel_3/transformer_block_7/layer_normalization_14/moments/SquaredDifferenceψ
Umodel_3/transformer_block_7/layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_3/transformer_block_7/layer_normalization_14/moments/variance/reduction_indices
Cmodel_3/transformer_block_7/layer_normalization_14/moments/varianceMeanPmodel_3/transformer_block_7/layer_normalization_14/moments/SquaredDifference:z:0^model_3/transformer_block_7/layer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2E
Cmodel_3/transformer_block_7/layer_normalization_14/moments/varianceΝ
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752D
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add/yή
@model_3/transformer_block_7/layer_normalization_14/batchnorm/addAddV2Lmodel_3/transformer_block_7/layer_normalization_14/moments/variance:output:0Kmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2B
@model_3/transformer_block_7/layer_normalization_14/batchnorm/add
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/RsqrtRsqrtDmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2D
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/Rsqrt·
Omodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_3_transformer_block_7_layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02Q
Omodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpβ
@model_3/transformer_block_7/layer_normalization_14/batchnorm/mulMulFmodel_3/transformer_block_7/layer_normalization_14/batchnorm/Rsqrt:y:0Wmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2B
@model_3/transformer_block_7/layer_normalization_14/batchnorm/mul°
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul_1Mul#model_3/transformer_block_7/add:z:0Dmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2D
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul_1Υ
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul_2MulHmodel_3/transformer_block_7/layer_normalization_14/moments/mean:output:0Dmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2D
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul_2«
Kmodel_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOpTmodel_3_transformer_block_7_layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02M
Kmodel_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpή
@model_3/transformer_block_7/layer_normalization_14/batchnorm/subSubSmodel_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp:value:0Fmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2B
@model_3/transformer_block_7/layer_normalization_14/batchnorm/subΥ
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add_1AddV2Fmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul_1:z:0Dmodel_3/transformer_block_7/layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2D
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add_1¬
Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpReadVariableOpSmodel_3_transformer_block_7_sequential_7_dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02L
Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpΞ
@model_3/transformer_block_7/sequential_7/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_3/transformer_block_7/sequential_7/dense_23/Tensordot/axesΥ
@model_3/transformer_block_7/sequential_7/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_3/transformer_block_7/sequential_7/dense_23/Tensordot/freeό
Amodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ShapeShapeFmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2C
Amodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ShapeΨ
Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axisΛ
Dmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2GatherV2Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Shape:output:0Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/free:output:0Rmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2ά
Kmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axisΡ
Fmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1GatherV2Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Shape:output:0Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/axes:output:0Tmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1Π
Amodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ConstΘ
@model_3/transformer_block_7/sequential_7/dense_23/Tensordot/ProdProdMmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2:output:0Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_3/transformer_block_7/sequential_7/dense_23/Tensordot/ProdΤ
Cmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const_1Π
Bmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Prod_1ProdOmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1:output:0Lmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Prod_1Τ
Gmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat/axisͺ
Bmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concatConcatV2Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/free:output:0Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/axes:output:0Pmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concatΤ
Amodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/stackPackImodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Prod:output:0Kmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/stackζ
Emodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/transpose	TransposeFmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add_1:z:0Kmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????# 2G
Emodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/transposeη
Cmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReshapeReshapeImodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/transpose:y:0Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2E
Cmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Reshapeζ
Bmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/MatMulMatMulLmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Reshape:output:0Rmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2D
Bmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/MatMulΤ
Cmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2E
Cmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const_2Ψ
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
Dmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat_1Ψ
;model_3/transformer_block_7/sequential_7/dense_23/TensordotReshapeLmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/MatMul:product:0Mmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@2=
;model_3/transformer_block_7/sequential_7/dense_23/Tensordot’
Hmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOpQmodel_3_transformer_block_7_sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02J
Hmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpΟ
9model_3/transformer_block_7/sequential_7/dense_23/BiasAddBiasAddDmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot:output:0Pmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@2;
9model_3/transformer_block_7/sequential_7/dense_23/BiasAddς
6model_3/transformer_block_7/sequential_7/dense_23/ReluReluBmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#@28
6model_3/transformer_block_7/sequential_7/dense_23/Relu¬
Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpReadVariableOpSmodel_3_transformer_block_7_sequential_7_dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02L
Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpΞ
@model_3/transformer_block_7/sequential_7/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_3/transformer_block_7/sequential_7/dense_24/Tensordot/axesΥ
@model_3/transformer_block_7/sequential_7/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_3/transformer_block_7/sequential_7/dense_24/Tensordot/freeϊ
Amodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ShapeShapeDmodel_3/transformer_block_7/sequential_7/dense_23/Relu:activations:0*
T0*
_output_shapes
:2C
Amodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ShapeΨ
Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axisΛ
Dmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2GatherV2Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Shape:output:0Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/free:output:0Rmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2ά
Kmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axisΡ
Fmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1GatherV2Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Shape:output:0Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/axes:output:0Tmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1Π
Amodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ConstΘ
@model_3/transformer_block_7/sequential_7/dense_24/Tensordot/ProdProdMmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2:output:0Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_3/transformer_block_7/sequential_7/dense_24/Tensordot/ProdΤ
Cmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const_1Π
Bmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Prod_1ProdOmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1:output:0Lmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Prod_1Τ
Gmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat/axisͺ
Bmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concatConcatV2Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/free:output:0Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/axes:output:0Pmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concatΤ
Amodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/stackPackImodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Prod:output:0Kmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/stackδ
Emodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/transpose	TransposeDmodel_3/transformer_block_7/sequential_7/dense_23/Relu:activations:0Kmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2G
Emodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/transposeη
Cmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReshapeReshapeImodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/transpose:y:0Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2E
Cmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Reshapeζ
Bmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/MatMulMatMulLmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Reshape:output:0Rmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2D
Bmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/MatMulΤ
Cmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const_2Ψ
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
Dmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat_1Ψ
;model_3/transformer_block_7/sequential_7/dense_24/TensordotReshapeLmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/MatMul:product:0Mmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 2=
;model_3/transformer_block_7/sequential_7/dense_24/Tensordot’
Hmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOpQmodel_3_transformer_block_7_sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpΟ
9model_3/transformer_block_7/sequential_7/dense_24/BiasAddBiasAddDmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot:output:0Pmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2;
9model_3/transformer_block_7/sequential_7/dense_24/BiasAddθ
/model_3/transformer_block_7/dropout_21/IdentityIdentityBmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd:output:0*
T0*+
_output_shapes
:?????????# 21
/model_3/transformer_block_7/dropout_21/Identity
!model_3/transformer_block_7/add_1AddV2Fmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add_1:z:08model_3/transformer_block_7/dropout_21/Identity:output:0*
T0*+
_output_shapes
:?????????# 2#
!model_3/transformer_block_7/add_1π
Qmodel_3/transformer_block_7/layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_3/transformer_block_7/layer_normalization_15/moments/mean/reduction_indicesΤ
?model_3/transformer_block_7/layer_normalization_15/moments/meanMean%model_3/transformer_block_7/add_1:z:0Zmodel_3/transformer_block_7/layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2A
?model_3/transformer_block_7/layer_normalization_15/moments/mean’
Gmodel_3/transformer_block_7/layer_normalization_15/moments/StopGradientStopGradientHmodel_3/transformer_block_7/layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:?????????#2I
Gmodel_3/transformer_block_7/layer_normalization_15/moments/StopGradientΰ
Lmodel_3/transformer_block_7/layer_normalization_15/moments/SquaredDifferenceSquaredDifference%model_3/transformer_block_7/add_1:z:0Pmodel_3/transformer_block_7/layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????# 2N
Lmodel_3/transformer_block_7/layer_normalization_15/moments/SquaredDifferenceψ
Umodel_3/transformer_block_7/layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_3/transformer_block_7/layer_normalization_15/moments/variance/reduction_indices
Cmodel_3/transformer_block_7/layer_normalization_15/moments/varianceMeanPmodel_3/transformer_block_7/layer_normalization_15/moments/SquaredDifference:z:0^model_3/transformer_block_7/layer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????#*
	keep_dims(2E
Cmodel_3/transformer_block_7/layer_normalization_15/moments/varianceΝ
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752D
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add/yή
@model_3/transformer_block_7/layer_normalization_15/batchnorm/addAddV2Lmodel_3/transformer_block_7/layer_normalization_15/moments/variance:output:0Kmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????#2B
@model_3/transformer_block_7/layer_normalization_15/batchnorm/add
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/RsqrtRsqrtDmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????#2D
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/Rsqrt·
Omodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_3_transformer_block_7_layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02Q
Omodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpβ
@model_3/transformer_block_7/layer_normalization_15/batchnorm/mulMulFmodel_3/transformer_block_7/layer_normalization_15/batchnorm/Rsqrt:y:0Wmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2B
@model_3/transformer_block_7/layer_normalization_15/batchnorm/mul²
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul_1Mul%model_3/transformer_block_7/add_1:z:0Dmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2D
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul_1Υ
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul_2MulHmodel_3/transformer_block_7/layer_normalization_15/moments/mean:output:0Dmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????# 2D
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul_2«
Kmodel_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOpTmodel_3_transformer_block_7_layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02M
Kmodel_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpή
@model_3/transformer_block_7/layer_normalization_15/batchnorm/subSubSmodel_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp:value:0Fmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????# 2B
@model_3/transformer_block_7/layer_normalization_15/batchnorm/subΥ
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add_1AddV2Fmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul_1:z:0Dmodel_3/transformer_block_7/layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????# 2D
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add_1Έ
9model_3/global_average_pooling1d_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9model_3/global_average_pooling1d_3/Mean/reduction_indices
'model_3/global_average_pooling1d_3/MeanMeanFmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add_1:z:0Bmodel_3/global_average_pooling1d_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2)
'model_3/global_average_pooling1d_3/Mean
model_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
model_3/flatten_3/ConstΗ
model_3/flatten_3/ReshapeReshape0model_3/global_average_pooling1d_3/Mean:output:0 model_3/flatten_3/Const:output:0*
T0*'
_output_shapes
:????????? 2
model_3/flatten_3/Reshape
!model_3/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_3/concatenate_3/concat/axisά
model_3/concatenate_3/concatConcatV2"model_3/flatten_3/Reshape:output:0input_8*model_3/concatenate_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????(2
model_3/concatenate_3/concatΐ
&model_3/dense_25/MatMul/ReadVariableOpReadVariableOp/model_3_dense_25_matmul_readvariableop_resource*
_output_shapes

:(@*
dtype02(
&model_3/dense_25/MatMul/ReadVariableOpΕ
model_3/dense_25/MatMulMatMul%model_3/concatenate_3/concat:output:0.model_3/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_3/dense_25/MatMulΏ
'model_3/dense_25/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_3/dense_25/BiasAdd/ReadVariableOpΕ
model_3/dense_25/BiasAddBiasAdd!model_3/dense_25/MatMul:product:0/model_3/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_3/dense_25/BiasAdd
model_3/dense_25/ReluRelu!model_3/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_3/dense_25/Relu
model_3/dropout_22/IdentityIdentity#model_3/dense_25/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
model_3/dropout_22/Identityΐ
&model_3/dense_26/MatMul/ReadVariableOpReadVariableOp/model_3_dense_26_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02(
&model_3/dense_26/MatMul/ReadVariableOpΔ
model_3/dense_26/MatMulMatMul$model_3/dropout_22/Identity:output:0.model_3/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_3/dense_26/MatMulΏ
'model_3/dense_26/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_3/dense_26/BiasAdd/ReadVariableOpΕ
model_3/dense_26/BiasAddBiasAdd!model_3/dense_26/MatMul:product:0/model_3/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_3/dense_26/BiasAdd
model_3/dense_26/ReluRelu!model_3/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_3/dense_26/Relu
model_3/dropout_23/IdentityIdentity#model_3/dense_26/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
model_3/dropout_23/Identityΐ
&model_3/dense_27/MatMul/ReadVariableOpReadVariableOp/model_3_dense_27_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&model_3/dense_27/MatMul/ReadVariableOpΔ
model_3/dense_27/MatMulMatMul$model_3/dropout_23/Identity:output:0.model_3/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_27/MatMulΏ
'model_3/dense_27/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_3/dense_27/BiasAdd/ReadVariableOpΕ
model_3/dense_27/BiasAddBiasAdd!model_3/dense_27/MatMul:product:0/model_3/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_27/BiasAdd¬
IdentityIdentity!model_3/dense_27/BiasAdd:output:07^model_3/batch_normalization_6/batchnorm/ReadVariableOp9^model_3/batch_normalization_6/batchnorm/ReadVariableOp_19^model_3/batch_normalization_6/batchnorm/ReadVariableOp_2;^model_3/batch_normalization_6/batchnorm/mul/ReadVariableOp7^model_3/batch_normalization_7/batchnorm/ReadVariableOp9^model_3/batch_normalization_7/batchnorm/ReadVariableOp_19^model_3/batch_normalization_7/batchnorm/ReadVariableOp_2;^model_3/batch_normalization_7/batchnorm/mul/ReadVariableOp(^model_3/conv1d_6/BiasAdd/ReadVariableOp4^model_3/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp(^model_3/conv1d_7/BiasAdd/ReadVariableOp4^model_3/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp(^model_3/dense_25/BiasAdd/ReadVariableOp'^model_3/dense_25/MatMul/ReadVariableOp(^model_3/dense_26/BiasAdd/ReadVariableOp'^model_3/dense_26/MatMul/ReadVariableOp(^model_3/dense_27/BiasAdd/ReadVariableOp'^model_3/dense_27/MatMul/ReadVariableOpD^model_3/token_and_position_embedding_3/embedding_6/embedding_lookupD^model_3/token_and_position_embedding_3/embedding_7/embedding_lookupL^model_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpP^model_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpL^model_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpP^model_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpW^model_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpa^model_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpJ^model_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOpT^model_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpL^model_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOpV^model_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpL^model_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOpV^model_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpI^model_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpK^model_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpI^model_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpK^model_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Μ
_input_shapesΊ
·:?????????R:?????????::::::::::::::::::::::::::::::::::::2p
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
Kmodel_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpKmodel_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp2’
Omodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpOmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp2
Kmodel_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpKmodel_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp2’
Omodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpOmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp2°
Vmodel_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpVmodel_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp2Δ
`model_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp`model_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp2
Imodel_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOpImodel_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOp2ͺ
Smodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpSmodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp2
Kmodel_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOpKmodel_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOp2?
Umodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpUmodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp2
Kmodel_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOpKmodel_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOp2?
Umodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpUmodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp2
Hmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpHmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp2
Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpJmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp2
Hmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpHmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp2
Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpJmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp:Q M
(
_output_shapes
:?????????R
!
_user_specified_name	input_7:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_8
	
έ
D__inference_dense_27_layer_call_and_return_conditional_losses_308643

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd
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
Θ
©
6__inference_batch_normalization_7_layer_call_fn_308119

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3058422
StatefulPartitionedCall
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
θ

Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_308106

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOp
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
:?????????# 2
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
:?????????# 2
batchnorm/add_1ί
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
Ό0
Θ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_308086

inputs
assignmovingavg_308061
assignmovingavg_1_308067)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOp
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
:?????????# 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
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
moments/Squeeze_1Μ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/308061*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_308061*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpρ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/308061*
_output_shapes
: 2
AssignMovingAvg/subθ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/308061*
_output_shapes
: 2
AssignMovingAvg/mul―
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_308061AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/308061*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/308067*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_308067*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpϋ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/308067*
_output_shapes
: 2
AssignMovingAvg_1/subς
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/308067*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_308067AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/308067*
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
:?????????# 2
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
:?????????# 2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
J
―
H__inference_sequential_7_layer_call_and_return_conditional_losses_308709

inputs.
*dense_23_tensordot_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource.
*dense_24_tensordot_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource
identity’dense_23/BiasAdd/ReadVariableOp’!dense_23/Tensordot/ReadVariableOp’dense_24/BiasAdd/ReadVariableOp’!dense_24/Tensordot/ReadVariableOp±
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
 dense_23/Tensordot/GatherV2/axisώ
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
dense_23/Tensordot/Const€
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
dense_23/Tensordot/concat/axisέ
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
:?????????# 2
dense_23/Tensordot/transposeΓ
dense_23/Tensordot/ReshapeReshape dense_23/Tensordot/transpose:y:0!dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_23/Tensordot/ReshapeΒ
dense_23/Tensordot/MatMulMatMul#dense_23/Tensordot/Reshape:output:0)dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
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
 dense_23/Tensordot/concat_1/axisκ
dense_23/Tensordot/concat_1ConcatV2$dense_23/Tensordot/GatherV2:output:0#dense_23/Tensordot/Const_2:output:0)dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/concat_1΄
dense_23/TensordotReshape#dense_23/Tensordot/MatMul:product:0$dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????#@2
dense_23/Tensordot§
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_23/BiasAdd/ReadVariableOp«
dense_23/BiasAddBiasAdddense_23/Tensordot:output:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#@2
dense_23/BiasAddw
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#@2
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
 dense_24/Tensordot/GatherV2/axisώ
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
dense_24/Tensordot/Const€
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
dense_24/Tensordot/concat/axisέ
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
dense_24/Tensordot/stackΐ
dense_24/Tensordot/transpose	Transposedense_23/Relu:activations:0"dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????#@2
dense_24/Tensordot/transposeΓ
dense_24/Tensordot/ReshapeReshape dense_24/Tensordot/transpose:y:0!dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_24/Tensordot/ReshapeΒ
dense_24/Tensordot/MatMulMatMul#dense_24/Tensordot/Reshape:output:0)dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
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
 dense_24/Tensordot/concat_1/axisκ
dense_24/Tensordot/concat_1ConcatV2$dense_24/Tensordot/GatherV2:output:0#dense_24/Tensordot/Const_2:output:0)dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/concat_1΄
dense_24/TensordotReshape#dense_24/Tensordot/MatMul:product:0$dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????# 2
dense_24/Tensordot§
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_24/BiasAdd/ReadVariableOp«
dense_24/BiasAddBiasAdddense_24/Tensordot:output:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????# 2
dense_24/BiasAddύ
IdentityIdentitydense_24/BiasAdd:output:0 ^dense_23/BiasAdd/ReadVariableOp"^dense_23/Tensordot/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp"^dense_24/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2F
!dense_23/Tensordot/ReadVariableOp!dense_23/Tensordot/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2F
!dense_24/Tensordot/ReadVariableOp!dense_24/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
ι

H__inference_sequential_7_layer_call_and_return_conditional_losses_305574

inputs
dense_23_305563
dense_23_305565
dense_24_305568
dense_24_305570
identity’ dense_23/StatefulPartitionedCall’ dense_24/StatefulPartitionedCall
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinputsdense_23_305563dense_23_305565*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_3054532"
 dense_23/StatefulPartitionedCallΎ
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_305568dense_24_305570*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_3054992"
 dense_24/StatefulPartitionedCallΗ
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
Κ
©
6__inference_batch_normalization_7_layer_call_fn_308132

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall’
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
GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3058622
StatefulPartitionedCall
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
Ι
d
F__inference_dropout_23_layer_call_and_return_conditional_losses_306440

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
χ
W
;__inference_global_average_pooling1d_3_layer_call_fn_308515

inputs
identityΰ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_3056012
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
«_
θ
C__inference_model_3_layer_call_and_return_conditional_losses_306674

inputs
inputs_1)
%token_and_position_embedding_3_306583)
%token_and_position_embedding_3_306585
conv1d_6_306588
conv1d_6_306590
conv1d_7_306594
conv1d_7_306596 
batch_normalization_6_306601 
batch_normalization_6_306603 
batch_normalization_6_306605 
batch_normalization_6_306607 
batch_normalization_7_306610 
batch_normalization_7_306612 
batch_normalization_7_306614 
batch_normalization_7_306616
transformer_block_7_306620
transformer_block_7_306622
transformer_block_7_306624
transformer_block_7_306626
transformer_block_7_306628
transformer_block_7_306630
transformer_block_7_306632
transformer_block_7_306634
transformer_block_7_306636
transformer_block_7_306638
transformer_block_7_306640
transformer_block_7_306642
transformer_block_7_306644
transformer_block_7_306646
transformer_block_7_306648
transformer_block_7_306650
dense_25_306656
dense_25_306658
dense_26_306662
dense_26_306664
dense_27_306668
dense_27_306670
identity’-batch_normalization_6/StatefulPartitionedCall’-batch_normalization_7/StatefulPartitionedCall’ conv1d_6/StatefulPartitionedCall’ conv1d_7/StatefulPartitionedCall’ dense_25/StatefulPartitionedCall’ dense_26/StatefulPartitionedCall’ dense_27/StatefulPartitionedCall’"dropout_22/StatefulPartitionedCall’"dropout_23/StatefulPartitionedCall’6token_and_position_embedding_3/StatefulPartitionedCall’+transformer_block_7/StatefulPartitionedCall
6token_and_position_embedding_3/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_3_306583%token_and_position_embedding_3_306585*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_30563328
6token_and_position_embedding_3/StatefulPartitionedCallΥ
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0conv1d_6_306588conv1d_6_306590*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_3056652"
 conv1d_6/StatefulPartitionedCall 
#average_pooling1d_9/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????ή * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_3051022%
#average_pooling1d_9/PartitionedCallΒ
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_9/PartitionedCall:output:0conv1d_7_306594conv1d_7_306596*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????ή *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_3056982"
 conv1d_7/StatefulPartitionedCallΈ
$average_pooling1d_11/PartitionedCallPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_3051322&
$average_pooling1d_11/PartitionedCall’
$average_pooling1d_10/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_3051172&
$average_pooling1d_10/PartitionedCallΑ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_10/PartitionedCall:output:0batch_normalization_6_306601batch_normalization_6_306603batch_normalization_6_306605batch_normalization_6_306607*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3057512/
-batch_normalization_6/StatefulPartitionedCallΑ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_11/PartitionedCall:output:0batch_normalization_7_306610batch_normalization_7_306612batch_normalization_7_306614batch_normalization_7_306616*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3058422/
-batch_normalization_7/StatefulPartitionedCall»
add_3/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8 *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_3059042
add_3/PartitionedCall
+transformer_block_7/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0transformer_block_7_306620transformer_block_7_306622transformer_block_7_306624transformer_block_7_306626transformer_block_7_306628transformer_block_7_306630transformer_block_7_306632transformer_block_7_306634transformer_block_7_306636transformer_block_7_306638transformer_block_7_306640transformer_block_7_306642transformer_block_7_306644transformer_block_7_306646transformer_block_7_306648transformer_block_7_306650*
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
GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_3060612-
+transformer_block_7/StatefulPartitionedCall»
*global_average_pooling1d_3/PartitionedCallPartitionedCall4transformer_block_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_3063022,
*global_average_pooling1d_3/PartitionedCall
flatten_3/PartitionedCallPartitionedCall3global_average_pooling1d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_3063152
flatten_3/PartitionedCall
concatenate_3/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_3063302
concatenate_3/PartitionedCall·
 dense_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_25_306656dense_25_306658*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_3063502"
 dense_25/StatefulPartitionedCall
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_3063782$
"dropout_22/StatefulPartitionedCallΌ
 dense_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_26_306662dense_26_306664*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_3064072"
 dense_26/StatefulPartitionedCall½
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
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
GPU2*0J 8 *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_3064352$
"dropout_23/StatefulPartitionedCallΌ
 dense_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_27_306668dense_27_306670*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_3064632"
 dense_27/StatefulPartitionedCall½
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall7^token_and_position_embedding_3/StatefulPartitionedCall,^transformer_block_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Μ
_input_shapesΊ
·:?????????R:?????????::::::::::::::::::::::::::::::::::::2^
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
+transformer_block_7/StatefulPartitionedCall+transformer_block_7/StatefulPartitionedCall:P L
(
_output_shapes
:?????????R
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Ό0
Θ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_305751

inputs
assignmovingavg_305726
assignmovingavg_1_305732)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOp
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
:?????????# 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
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
moments/Squeeze_1Μ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/305726*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_305726*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpρ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/305726*
_output_shapes
: 2
AssignMovingAvg/subθ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/305726*
_output_shapes
: 2
AssignMovingAvg/mul―
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_305726AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/305726*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/305732*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_305732*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpϋ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/305732*
_output_shapes
: 2
AssignMovingAvg_1/subς
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/305732*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_305732AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/305732*
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
:?????????# 2
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
:?????????# 2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs
€\

C__inference_model_3_layer_call_and_return_conditional_losses_306575
input_7
input_8)
%token_and_position_embedding_3_306484)
%token_and_position_embedding_3_306486
conv1d_6_306489
conv1d_6_306491
conv1d_7_306495
conv1d_7_306497 
batch_normalization_6_306502 
batch_normalization_6_306504 
batch_normalization_6_306506 
batch_normalization_6_306508 
batch_normalization_7_306511 
batch_normalization_7_306513 
batch_normalization_7_306515 
batch_normalization_7_306517
transformer_block_7_306521
transformer_block_7_306523
transformer_block_7_306525
transformer_block_7_306527
transformer_block_7_306529
transformer_block_7_306531
transformer_block_7_306533
transformer_block_7_306535
transformer_block_7_306537
transformer_block_7_306539
transformer_block_7_306541
transformer_block_7_306543
transformer_block_7_306545
transformer_block_7_306547
transformer_block_7_306549
transformer_block_7_306551
dense_25_306557
dense_25_306559
dense_26_306563
dense_26_306565
dense_27_306569
dense_27_306571
identity’-batch_normalization_6/StatefulPartitionedCall’-batch_normalization_7/StatefulPartitionedCall’ conv1d_6/StatefulPartitionedCall’ conv1d_7/StatefulPartitionedCall’ dense_25/StatefulPartitionedCall’ dense_26/StatefulPartitionedCall’ dense_27/StatefulPartitionedCall’6token_and_position_embedding_3/StatefulPartitionedCall’+transformer_block_7/StatefulPartitionedCall
6token_and_position_embedding_3/StatefulPartitionedCallStatefulPartitionedCallinput_7%token_and_position_embedding_3_306484%token_and_position_embedding_3_306486*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_30563328
6token_and_position_embedding_3/StatefulPartitionedCallΥ
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0conv1d_6_306489conv1d_6_306491*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_3056652"
 conv1d_6/StatefulPartitionedCall 
#average_pooling1d_9/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????ή * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_3051022%
#average_pooling1d_9/PartitionedCallΒ
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_9/PartitionedCall:output:0conv1d_7_306495conv1d_7_306497*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????ή *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_3056982"
 conv1d_7/StatefulPartitionedCallΈ
$average_pooling1d_11/PartitionedCallPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_3051322&
$average_pooling1d_11/PartitionedCall’
$average_pooling1d_10/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_3051172&
$average_pooling1d_10/PartitionedCallΓ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_10/PartitionedCall:output:0batch_normalization_6_306502batch_normalization_6_306504batch_normalization_6_306506batch_normalization_6_306508*
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
GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3057712/
-batch_normalization_6/StatefulPartitionedCallΓ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_11/PartitionedCall:output:0batch_normalization_7_306511batch_normalization_7_306513batch_normalization_7_306515batch_normalization_7_306517*
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
GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3058622/
-batch_normalization_7/StatefulPartitionedCall»
add_3/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8 *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_3059042
add_3/PartitionedCall
+transformer_block_7/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0transformer_block_7_306521transformer_block_7_306523transformer_block_7_306525transformer_block_7_306527transformer_block_7_306529transformer_block_7_306531transformer_block_7_306533transformer_block_7_306535transformer_block_7_306537transformer_block_7_306539transformer_block_7_306541transformer_block_7_306543transformer_block_7_306545transformer_block_7_306547transformer_block_7_306549transformer_block_7_306551*
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
GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_3061882-
+transformer_block_7/StatefulPartitionedCall»
*global_average_pooling1d_3/PartitionedCallPartitionedCall4transformer_block_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_3063022,
*global_average_pooling1d_3/PartitionedCall
flatten_3/PartitionedCallPartitionedCall3global_average_pooling1d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_3063152
flatten_3/PartitionedCall
concatenate_3/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0input_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_3063302
concatenate_3/PartitionedCall·
 dense_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_25_306557dense_25_306559*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_3063502"
 dense_25/StatefulPartitionedCall
dropout_22/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_3063832
dropout_22/PartitionedCall΄
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_26_306563dense_26_306565*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_3064072"
 dense_26/StatefulPartitionedCall
dropout_23/PartitionedCallPartitionedCall)dense_26/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_3064402
dropout_23/PartitionedCall΄
 dense_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_27_306569dense_27_306571*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_3064632"
 dense_27/StatefulPartitionedCallσ
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall7^token_and_position_embedding_3/StatefulPartitionedCall,^transformer_block_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Μ
_input_shapesΊ
·:?????????R:?????????::::::::::::::::::::::::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2p
6token_and_position_embedding_3/StatefulPartitionedCall6token_and_position_embedding_3/StatefulPartitionedCall2Z
+transformer_block_7/StatefulPartitionedCall+transformer_block_7/StatefulPartitionedCall:Q M
(
_output_shapes
:?????????R
!
_user_specified_name	input_7:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_8

e
F__inference_dropout_23_layer_call_and_return_conditional_losses_306435

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?2
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
dropout/Shape΄
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
 *ΝΜΜ=2
dropout/GreaterEqual/yΎ
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

e
F__inference_dropout_22_layer_call_and_return_conditional_losses_308571

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?2
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
dropout/Shape΄
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
 *ΝΜΜ=2
dropout/GreaterEqual/yΎ
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

χ
D__inference_conv1d_7_layer_call_and_return_conditional_losses_305698

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ή 2
conv1d/ExpandDimsΈ
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
:?????????ή *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:?????????ή *
squeeze_dims

ύ????????2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ή 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????ή 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:?????????ή 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????ή ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:?????????ή 
 
_user_specified_nameinputs
Ώ 
ώ(
__inference__traced_save_309117
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

identity_1’MergeV2Checkpoints
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
ShardedFilenameγ%
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*υ$
valueλ$Bθ$KB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names‘
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*«
value‘BKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesξ'
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_6_kernel_read_readvariableop(savev2_conv1d_6_bias_read_readvariableop*savev2_conv1d_7_kernel_read_readvariableop(savev2_conv1d_7_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableopPsavev2_token_and_position_embedding_3_embedding_6_embeddings_read_readvariableopPsavev2_token_and_position_embedding_3_embedding_7_embeddings_read_readvariableopRsavev2_transformer_block_7_multi_head_attention_7_query_kernel_read_readvariableopPsavev2_transformer_block_7_multi_head_attention_7_query_bias_read_readvariableopPsavev2_transformer_block_7_multi_head_attention_7_key_kernel_read_readvariableopNsavev2_transformer_block_7_multi_head_attention_7_key_bias_read_readvariableopRsavev2_transformer_block_7_multi_head_attention_7_value_kernel_read_readvariableopPsavev2_transformer_block_7_multi_head_attention_7_value_bias_read_readvariableop]savev2_transformer_block_7_multi_head_attention_7_attention_output_kernel_read_readvariableop[savev2_transformer_block_7_multi_head_attention_7_attention_output_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableopKsavev2_transformer_block_7_layer_normalization_14_gamma_read_readvariableopJsavev2_transformer_block_7_layer_normalization_14_beta_read_readvariableopKsavev2_transformer_block_7_layer_normalization_15_gamma_read_readvariableopJsavev2_transformer_block_7_layer_normalization_15_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop7savev2_sgd_conv1d_6_kernel_momentum_read_readvariableop5savev2_sgd_conv1d_6_bias_momentum_read_readvariableop7savev2_sgd_conv1d_7_kernel_momentum_read_readvariableop5savev2_sgd_conv1d_7_bias_momentum_read_readvariableopCsavev2_sgd_batch_normalization_6_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_6_beta_momentum_read_readvariableopCsavev2_sgd_batch_normalization_7_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_7_beta_momentum_read_readvariableop7savev2_sgd_dense_25_kernel_momentum_read_readvariableop5savev2_sgd_dense_25_bias_momentum_read_readvariableop7savev2_sgd_dense_26_kernel_momentum_read_readvariableop5savev2_sgd_dense_26_bias_momentum_read_readvariableop7savev2_sgd_dense_27_kernel_momentum_read_readvariableop5savev2_sgd_dense_27_bias_momentum_read_readvariableop]savev2_sgd_token_and_position_embedding_3_embedding_6_embeddings_momentum_read_readvariableop]savev2_sgd_token_and_position_embedding_3_embedding_7_embeddings_momentum_read_readvariableop_savev2_sgd_transformer_block_7_multi_head_attention_7_query_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_7_multi_head_attention_7_query_bias_momentum_read_readvariableop]savev2_sgd_transformer_block_7_multi_head_attention_7_key_kernel_momentum_read_readvariableop[savev2_sgd_transformer_block_7_multi_head_attention_7_key_bias_momentum_read_readvariableop_savev2_sgd_transformer_block_7_multi_head_attention_7_value_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_7_multi_head_attention_7_value_bias_momentum_read_readvariableopjsavev2_sgd_transformer_block_7_multi_head_attention_7_attention_output_kernel_momentum_read_readvariableophsavev2_sgd_transformer_block_7_multi_head_attention_7_attention_output_bias_momentum_read_readvariableop7savev2_sgd_dense_23_kernel_momentum_read_readvariableop5savev2_sgd_dense_23_bias_momentum_read_readvariableop7savev2_sgd_dense_24_kernel_momentum_read_readvariableop5savev2_sgd_dense_24_bias_momentum_read_readvariableopXsavev2_sgd_transformer_block_7_layer_normalization_14_gamma_momentum_read_readvariableopWsavev2_sgd_transformer_block_7_layer_normalization_14_beta_momentum_read_readvariableopXsavev2_sgd_transformer_block_7_layer_normalization_15_gamma_momentum_read_readvariableopWsavev2_sgd_transformer_block_7_layer_normalization_15_beta_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Y
dtypesO
M2K	2
SaveV2Ί
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes‘
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

identity_1Identity_1:output:0*ο
_input_shapesέ
Ϊ: :  : :	  : : : : : : : : : :(@:@:@@:@:@:: : : : : :	R :  : :  : :  : :  : : @:@:@ : : : : : : : :  : :	  : : : : : :(@:@:@@:@:@:: :	R :  : :  : :  : :  : : @:@:@ : : : : : : 2(
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
: :$ 

_output_shapes

:(@: 
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
: :$3 

_output_shapes

:(@: 4
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
Ό0
Θ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_307922

inputs
assignmovingavg_307897
assignmovingavg_1_307903)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOp
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
:?????????# 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesΆ
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
moments/Squeeze_1Μ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/307897*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_307897*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpρ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/307897*
_output_shapes
: 2
AssignMovingAvg/subθ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/307897*
_output_shapes
: 2
AssignMovingAvg/mul―
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_307897AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/307897*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/307903*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_307903*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpϋ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/307903*
_output_shapes
: 2
AssignMovingAvg_1/subς
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/307903*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_307903AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/307903*
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
:?????????# 2
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
:?????????# 2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????# 
 
_user_specified_nameinputs

F
*__inference_flatten_3_layer_call_fn_308526

inputs
identityΖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_3063152
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs


Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_305267

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOp
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
 :?????????????????? 2
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
 :?????????????????? 2
batchnorm/add_1θ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:?????????????????? ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs

Q
5__inference_average_pooling1d_10_layer_call_fn_305123

inputs
identityη
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
GPU2*0J 8 *Y
fTRR
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_3051172
PartitionedCall
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
Ι
d
F__inference_dropout_22_layer_call_and_return_conditional_losses_306383

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
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ι
serving_defaultΥ
<
input_71
serving_default_input_7:0?????????R
;
input_80
serving_default_input_8:0?????????<
dense_270
StatefulPartitionedCall:0?????????tensorflow/serving/predict:Ω
ιI
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
layer-14
layer_with_weights-6
layer-15
layer-16
layer_with_weights-7
layer-17
layer-18
layer_with_weights-8
layer-19
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
»_default_save_signature
+Ό&call_and_return_all_conditional_losses
½__call__"£D
_tf_keras_networkD{"class_name": "Functional", "name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}, "name": "token_and_position_embedding_3", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_6", "inbound_nodes": [[["token_and_position_embedding_3", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_9", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_9", "inbound_nodes": [[["conv1d_6", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_7", "inbound_nodes": [[["average_pooling1d_9", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_10", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_10", "inbound_nodes": [[["conv1d_7", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_11", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_11", "inbound_nodes": [[["token_and_position_embedding_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["average_pooling1d_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["average_pooling1d_11", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}], ["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block_7", "inbound_nodes": [[["add_3", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling1d_3", "inbound_nodes": [[["transformer_block_7", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["global_average_pooling1d_3", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_3", "inbound_nodes": [[["flatten_3", 0, 0, {}], ["input_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_25", "inbound_nodes": [[["concatenate_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_22", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_22", "inbound_nodes": [[["dense_25", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_26", "inbound_nodes": [[["dropout_22", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_23", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_23", "inbound_nodes": [[["dense_26", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_27", "inbound_nodes": [[["dropout_23", 0, 0, {}]]]}], "input_layers": [["input_7", 0, 0], ["input_8", 0, 0]], "output_layers": [["dense_27", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 10500]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 10500]}, {"class_name": "TensorShape", "items": [null, 8]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.00020000000949949026, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
ρ"ξ
_tf_keras_input_layerΞ{"class_name": "InputLayer", "name": "input_7", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}}
η
	token_emb
pos_emb
	variables
regularization_losses
trainable_variables
 	keras_api
+Ύ&call_and_return_all_conditional_losses
Ώ__call__"Ί
_tf_keras_layer {"class_name": "TokenAndPositionEmbedding", "name": "token_and_position_embedding_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
ι	

!kernel
"bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
+ΐ&call_and_return_all_conditional_losses
Α__call__"Β
_tf_keras_layer¨{"class_name": "Conv1D", "name": "conv1d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500, 32]}}

'	variables
(regularization_losses
)trainable_variables
*	keras_api
+Β&call_and_return_all_conditional_losses
Γ__call__"ψ
_tf_keras_layerή{"class_name": "AveragePooling1D", "name": "average_pooling1d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_9", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
η	

+kernel
,bias
-	variables
.regularization_losses
/trainable_variables
0	keras_api
+Δ&call_and_return_all_conditional_losses
Ε__call__"ΐ
_tf_keras_layer¦{"class_name": "Conv1D", "name": "conv1d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 350, 32]}}

1	variables
2regularization_losses
3trainable_variables
4	keras_api
+Ζ&call_and_return_all_conditional_losses
Η__call__"ϊ
_tf_keras_layerΰ{"class_name": "AveragePooling1D", "name": "average_pooling1d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_10", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}

5	variables
6regularization_losses
7trainable_variables
8	keras_api
+Θ&call_and_return_all_conditional_losses
Ι__call__"ό
_tf_keras_layerβ{"class_name": "AveragePooling1D", "name": "average_pooling1d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_11", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Έ	
9axis
	:gamma
;beta
<moving_mean
=moving_variance
>	variables
?regularization_losses
@trainable_variables
A	keras_api
+Κ&call_and_return_all_conditional_losses
Λ__call__"β
_tf_keras_layerΘ{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
Έ	
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
+Μ&call_and_return_all_conditional_losses
Ν__call__"β
_tf_keras_layerΘ{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
³
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
+Ξ&call_and_return_all_conditional_losses
Ο__call__"’
_tf_keras_layer{"class_name": "Add", "name": "add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 35, 32]}, {"class_name": "TensorShape", "items": [null, 35, 32]}]}

Oatt
Pffn
Q
layernorm1
R
layernorm2
Sdropout1
Tdropout2
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
+Π&call_and_return_all_conditional_losses
Ρ__call__"₯
_tf_keras_layer{"class_name": "TransformerBlock", "name": "transformer_block_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}

Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
+?&call_and_return_all_conditional_losses
Σ__call__"
_tf_keras_layerξ{"class_name": "GlobalAveragePooling1D", "name": "global_average_pooling1d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling1d_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
θ
]	variables
^regularization_losses
_trainable_variables
`	keras_api
+Τ&call_and_return_all_conditional_losses
Υ__call__"Χ
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ι"ζ
_tf_keras_input_layerΖ{"class_name": "InputLayer", "name": "input_8", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}}
Ξ
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
+Φ&call_and_return_all_conditional_losses
Χ__call__"½
_tf_keras_layer£{"class_name": "Concatenate", "name": "concatenate_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32]}, {"class_name": "TensorShape", "items": [null, 8]}]}
τ

ekernel
fbias
g	variables
hregularization_losses
itrainable_variables
j	keras_api
+Ψ&call_and_return_all_conditional_losses
Ω__call__"Ν
_tf_keras_layer³{"class_name": "Dense", "name": "dense_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
ι
k	variables
lregularization_losses
mtrainable_variables
n	keras_api
+Ϊ&call_and_return_all_conditional_losses
Ϋ__call__"Ψ
_tf_keras_layerΎ{"class_name": "Dropout", "name": "dropout_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_22", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
τ

okernel
pbias
q	variables
rregularization_losses
strainable_variables
t	keras_api
+ά&call_and_return_all_conditional_losses
έ__call__"Ν
_tf_keras_layer³{"class_name": "Dense", "name": "dense_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ι
u	variables
vregularization_losses
wtrainable_variables
x	keras_api
+ή&call_and_return_all_conditional_losses
ί__call__"Ψ
_tf_keras_layerΎ{"class_name": "Dropout", "name": "dropout_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_23", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
υ

ykernel
zbias
{	variables
|regularization_losses
}trainable_variables
~	keras_api
+ΰ&call_and_return_all_conditional_losses
α__call__"Ξ
_tf_keras_layer΄{"class_name": "Dense", "name": "dense_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ώ
	decay
learning_rate
momentum
	iter!momentum"momentum+momentum,momentum:momentum;momentum Cmomentum‘Dmomentum’emomentum£fmomentum€omomentum₯pmomentum¦ymomentum§zmomentum¨momentum©momentumͺmomentum«momentum¬momentum­momentum?momentum―momentum°momentum±momentum²momentum³momentum΄momentum΅momentumΆmomentum·momentumΈmomentumΉmomentumΊ"
	optimizer
Θ
0
1
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
26
27
28
29
e30
f31
o32
p33
y34
z35"
trackable_list_wrapper
 "
trackable_list_wrapper
¨
0
1
!2
"3
+4
,5
:6
;7
C8
D9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
e26
f27
o28
p29
y30
z31"
trackable_list_wrapper
Σ
metrics
	variables
 layer_regularization_losses
regularization_losses
trainable_variables
layer_metrics
layers
non_trainable_variables
½__call__
»_default_save_signature
+Ό&call_and_return_all_conditional_losses
'Ό"call_and_return_conditional_losses"
_generic_user_object
-
βserving_default"
signature_map
΅

embeddings
	variables
regularization_losses
trainable_variables
	keras_api
+γ&call_and_return_all_conditional_losses
δ__call__"
_tf_keras_layerυ{"class_name": "Embedding", "name": "embedding_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500]}}
²

embeddings
	variables
regularization_losses
 trainable_variables
‘	keras_api
+ε&call_and_return_all_conditional_losses
ζ__call__"
_tf_keras_layerς{"class_name": "Embedding", "name": "embedding_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 10500, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
΅
’metrics
	variables
 £layer_regularization_losses
regularization_losses
trainable_variables
€layer_metrics
₯layers
¦non_trainable_variables
Ώ__call__
+Ύ&call_and_return_all_conditional_losses
'Ύ"call_and_return_conditional_losses"
_generic_user_object
%:#  2conv1d_6/kernel
: 2conv1d_6/bias
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
΅
§metrics
#	variables
 ¨layer_regularization_losses
$regularization_losses
%trainable_variables
©layer_metrics
ͺlayers
«non_trainable_variables
Α__call__
+ΐ&call_and_return_all_conditional_losses
'ΐ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
¬metrics
'	variables
 ­layer_regularization_losses
(regularization_losses
)trainable_variables
?layer_metrics
―layers
°non_trainable_variables
Γ__call__
+Β&call_and_return_all_conditional_losses
'Β"call_and_return_conditional_losses"
_generic_user_object
%:#	  2conv1d_7/kernel
: 2conv1d_7/bias
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
΅
±metrics
-	variables
 ²layer_regularization_losses
.regularization_losses
/trainable_variables
³layer_metrics
΄layers
΅non_trainable_variables
Ε__call__
+Δ&call_and_return_all_conditional_losses
'Δ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
Άmetrics
1	variables
 ·layer_regularization_losses
2regularization_losses
3trainable_variables
Έlayer_metrics
Ήlayers
Ίnon_trainable_variables
Η__call__
+Ζ&call_and_return_all_conditional_losses
'Ζ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
»metrics
5	variables
 Όlayer_regularization_losses
6regularization_losses
7trainable_variables
½layer_metrics
Ύlayers
Ώnon_trainable_variables
Ι__call__
+Θ&call_and_return_all_conditional_losses
'Θ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_6/gamma
(:& 2batch_normalization_6/beta
1:/  (2!batch_normalization_6/moving_mean
5:3  (2%batch_normalization_6/moving_variance
<
:0
;1
<2
=3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
΅
ΐmetrics
>	variables
 Αlayer_regularization_losses
?regularization_losses
@trainable_variables
Βlayer_metrics
Γlayers
Δnon_trainable_variables
Λ__call__
+Κ&call_and_return_all_conditional_losses
'Κ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_7/gamma
(:& 2batch_normalization_7/beta
1:/  (2!batch_normalization_7/moving_mean
5:3  (2%batch_normalization_7/moving_variance
<
C0
D1
E2
F3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
΅
Εmetrics
G	variables
 Ζlayer_regularization_losses
Hregularization_losses
Itrainable_variables
Ηlayer_metrics
Θlayers
Ιnon_trainable_variables
Ν__call__
+Μ&call_and_return_all_conditional_losses
'Μ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
Κmetrics
K	variables
 Λlayer_regularization_losses
Lregularization_losses
Mtrainable_variables
Μlayer_metrics
Νlayers
Ξnon_trainable_variables
Ο__call__
+Ξ&call_and_return_all_conditional_losses
'Ξ"call_and_return_conditional_losses"
_generic_user_object

Ο_query_dense
Π
_key_dense
Ρ_value_dense
?_softmax
Σ_dropout_layer
Τ_output_dense
Υ	variables
Φregularization_losses
Χtrainable_variables
Ψ	keras_api
+η&call_and_return_all_conditional_losses
θ__call__"
_tf_keras_layerκ{"class_name": "MultiHeadAttention", "name": "multi_head_attention_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multi_head_attention_7", "trainable": true, "dtype": "float32", "num_heads": 1, "key_dim": 32, "value_dim": 32, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}
―
Ωlayer_with_weights-0
Ωlayer-0
Ϊlayer_with_weights-1
Ϊlayer-1
Ϋ	variables
άregularization_losses
έtrainable_variables
ή	keras_api
+ι&call_and_return_all_conditional_losses
κ__call__"Θ
_tf_keras_sequential©{"class_name": "Sequential", "name": "sequential_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_23_input"}}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_23_input"}}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
μ
	ίaxis

gamma
	beta
ΰ	variables
αregularization_losses
βtrainable_variables
γ	keras_api
+λ&call_and_return_all_conditional_losses
μ__call__"΅
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_14", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
μ
	δaxis

gamma
	beta
ε	variables
ζregularization_losses
ηtrainable_variables
θ	keras_api
+ν&call_and_return_all_conditional_losses
ξ__call__"΅
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_15", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
ν
ι	variables
κregularization_losses
λtrainable_variables
μ	keras_api
+ο&call_and_return_all_conditional_losses
π__call__"Ψ
_tf_keras_layerΎ{"class_name": "Dropout", "name": "dropout_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_20", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
ν
ν	variables
ξregularization_losses
οtrainable_variables
π	keras_api
+ρ&call_and_return_all_conditional_losses
ς__call__"Ψ
_tf_keras_layerΎ{"class_name": "Dropout", "name": "dropout_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_21", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
¦
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
¦
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
΅
ρmetrics
U	variables
 ςlayer_regularization_losses
Vregularization_losses
Wtrainable_variables
σlayer_metrics
τlayers
υnon_trainable_variables
Ρ__call__
+Π&call_and_return_all_conditional_losses
'Π"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
φmetrics
Y	variables
 χlayer_regularization_losses
Zregularization_losses
[trainable_variables
ψlayer_metrics
ωlayers
ϊnon_trainable_variables
Σ__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
ϋmetrics
]	variables
 όlayer_regularization_losses
^regularization_losses
_trainable_variables
ύlayer_metrics
ώlayers
?non_trainable_variables
Υ__call__
+Τ&call_and_return_all_conditional_losses
'Τ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
metrics
a	variables
 layer_regularization_losses
bregularization_losses
ctrainable_variables
layer_metrics
layers
non_trainable_variables
Χ__call__
+Φ&call_and_return_all_conditional_losses
'Φ"call_and_return_conditional_losses"
_generic_user_object
!:(@2dense_25/kernel
:@2dense_25/bias
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
΅
metrics
g	variables
 layer_regularization_losses
hregularization_losses
itrainable_variables
layer_metrics
layers
non_trainable_variables
Ω__call__
+Ψ&call_and_return_all_conditional_losses
'Ψ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
metrics
k	variables
 layer_regularization_losses
lregularization_losses
mtrainable_variables
layer_metrics
layers
non_trainable_variables
Ϋ__call__
+Ϊ&call_and_return_all_conditional_losses
'Ϊ"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_26/kernel
:@2dense_26/bias
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
΅
metrics
q	variables
 layer_regularization_losses
rregularization_losses
strainable_variables
layer_metrics
layers
non_trainable_variables
έ__call__
+ά&call_and_return_all_conditional_losses
'ά"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
metrics
u	variables
 layer_regularization_losses
vregularization_losses
wtrainable_variables
layer_metrics
layers
non_trainable_variables
ί__call__
+ή&call_and_return_all_conditional_losses
'ή"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_27/kernel
:2dense_27/bias
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
΅
metrics
{	variables
 layer_regularization_losses
|regularization_losses
}trainable_variables
layer_metrics
layers
non_trainable_variables
α__call__
+ΰ&call_and_return_all_conditional_losses
'ΰ"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
G:E 25token_and_position_embedding_3/embedding_6/embeddings
H:F	R 25token_and_position_embedding_3/embedding_7/embeddings
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
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Ά
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
<
<0
=1
E2
F3"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
Έ
metrics
	variables
  layer_regularization_losses
regularization_losses
trainable_variables
‘layer_metrics
’layers
£non_trainable_variables
δ__call__
+γ&call_and_return_all_conditional_losses
'γ"call_and_return_conditional_losses"
_generic_user_object
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
Έ
€metrics
	variables
 ₯layer_regularization_losses
regularization_losses
 trainable_variables
¦layer_metrics
§layers
¨non_trainable_variables
ζ__call__
+ε&call_and_return_all_conditional_losses
'ε"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
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
<0
=1"
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
E0
F1"
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
Λ
©partial_output_shape
ͺfull_output_shape
kernel
	bias
«	variables
¬regularization_losses
­trainable_variables
?	keras_api
+σ&call_and_return_all_conditional_losses
τ__call__"λ
_tf_keras_layerΡ{"class_name": "EinsumDense", "name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
Η
―partial_output_shape
°full_output_shape
kernel
	bias
±	variables
²regularization_losses
³trainable_variables
΄	keras_api
+υ&call_and_return_all_conditional_losses
φ__call__"η
_tf_keras_layerΝ{"class_name": "EinsumDense", "name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
Λ
΅partial_output_shape
Άfull_output_shape
kernel
	bias
·	variables
Έregularization_losses
Ήtrainable_variables
Ί	keras_api
+χ&call_and_return_all_conditional_losses
ψ__call__"λ
_tf_keras_layerΡ{"class_name": "EinsumDense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
λ
»	variables
Όregularization_losses
½trainable_variables
Ύ	keras_api
+ω&call_and_return_all_conditional_losses
ϊ__call__"Φ
_tf_keras_layerΌ{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}}
η
Ώ	variables
ΐregularization_losses
Αtrainable_variables
Β	keras_api
+ϋ&call_and_return_all_conditional_losses
ό__call__"?
_tf_keras_layerΈ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
ΰ
Γpartial_output_shape
Δfull_output_shape
kernel
	bias
Ε	variables
Ζregularization_losses
Ηtrainable_variables
Θ	keras_api
+ύ&call_and_return_all_conditional_losses
ώ__call__"
_tf_keras_layerζ{"class_name": "EinsumDense", "name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 32], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 1, 32]}}
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
Έ
Ιmetrics
Υ	variables
 Κlayer_regularization_losses
Φregularization_losses
Χtrainable_variables
Λlayer_metrics
Μlayers
Νnon_trainable_variables
θ__call__
+η&call_and_return_all_conditional_losses
'η"call_and_return_conditional_losses"
_generic_user_object
ώ
kernel
	bias
Ξ	variables
Οregularization_losses
Πtrainable_variables
Ρ	keras_api
+?&call_and_return_all_conditional_losses
__call__"Ρ
_tf_keras_layer·{"class_name": "Dense", "name": "dense_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}

kernel
	bias
?	variables
Σregularization_losses
Τtrainable_variables
Υ	keras_api
+&call_and_return_all_conditional_losses
__call__"Σ
_tf_keras_layerΉ{"class_name": "Dense", "name": "dense_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 64]}}
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
Έ
Φmetrics
Ϋ	variables
 Χlayer_regularization_losses
άregularization_losses
έtrainable_variables
Ψlayer_metrics
Ωlayers
Ϊnon_trainable_variables
κ__call__
+ι&call_and_return_all_conditional_losses
'ι"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
Έ
Ϋmetrics
ΰ	variables
 άlayer_regularization_losses
αregularization_losses
βtrainable_variables
έlayer_metrics
ήlayers
ίnon_trainable_variables
μ__call__
+λ&call_and_return_all_conditional_losses
'λ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
Έ
ΰmetrics
ε	variables
 αlayer_regularization_losses
ζregularization_losses
ηtrainable_variables
βlayer_metrics
γlayers
δnon_trainable_variables
ξ__call__
+ν&call_and_return_all_conditional_losses
'ν"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
εmetrics
ι	variables
 ζlayer_regularization_losses
κregularization_losses
λtrainable_variables
ηlayer_metrics
θlayers
ιnon_trainable_variables
π__call__
+ο&call_and_return_all_conditional_losses
'ο"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
κmetrics
ν	variables
 λlayer_regularization_losses
ξregularization_losses
οtrainable_variables
μlayer_metrics
νlayers
ξnon_trainable_variables
ς__call__
+ρ&call_and_return_all_conditional_losses
'ρ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
Ώ

οtotal

πcount
ρ	variables
ς	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
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
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
Έ
σmetrics
«	variables
 τlayer_regularization_losses
¬regularization_losses
­trainable_variables
υlayer_metrics
φlayers
χnon_trainable_variables
τ__call__
+σ&call_and_return_all_conditional_losses
'σ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
Έ
ψmetrics
±	variables
 ωlayer_regularization_losses
²regularization_losses
³trainable_variables
ϊlayer_metrics
ϋlayers
όnon_trainable_variables
φ__call__
+υ&call_and_return_all_conditional_losses
'υ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
Έ
ύmetrics
·	variables
 ώlayer_regularization_losses
Έregularization_losses
Ήtrainable_variables
?layer_metrics
layers
non_trainable_variables
ψ__call__
+χ&call_and_return_all_conditional_losses
'χ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
metrics
»	variables
 layer_regularization_losses
Όregularization_losses
½trainable_variables
layer_metrics
layers
non_trainable_variables
ϊ__call__
+ω&call_and_return_all_conditional_losses
'ω"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
metrics
Ώ	variables
 layer_regularization_losses
ΐregularization_losses
Αtrainable_variables
layer_metrics
layers
non_trainable_variables
ό__call__
+ϋ&call_and_return_all_conditional_losses
'ϋ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
Έ
metrics
Ε	variables
 layer_regularization_losses
Ζregularization_losses
Ηtrainable_variables
layer_metrics
layers
non_trainable_variables
ώ__call__
+ύ&call_and_return_all_conditional_losses
'ύ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
P
Ο0
Π1
Ρ2
?3
Σ4
Τ5"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
Έ
metrics
Ξ	variables
 layer_regularization_losses
Οregularization_losses
Πtrainable_variables
layer_metrics
layers
non_trainable_variables
__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
Έ
metrics
?	variables
 layer_regularization_losses
Σregularization_losses
Τtrainable_variables
layer_metrics
layers
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ω0
Ϊ1"
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
:  (2total
:  (2count
0
ο0
π1"
trackable_list_wrapper
.
ρ	variables"
_generic_user_object
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
0:.  2SGD/conv1d_6/kernel/momentum
&:$ 2SGD/conv1d_6/bias/momentum
0:.	  2SGD/conv1d_7/kernel/momentum
&:$ 2SGD/conv1d_7/bias/momentum
4:2 2(SGD/batch_normalization_6/gamma/momentum
3:1 2'SGD/batch_normalization_6/beta/momentum
4:2 2(SGD/batch_normalization_7/gamma/momentum
3:1 2'SGD/batch_normalization_7/beta/momentum
,:*(@2SGD/dense_25/kernel/momentum
&:$@2SGD/dense_25/bias/momentum
,:*@@2SGD/dense_26/kernel/momentum
&:$@2SGD/dense_26/bias/momentum
,:*@2SGD/dense_27/kernel/momentum
&:$2SGD/dense_27/bias/momentum
R:P 2BSGD/token_and_position_embedding_3/embedding_6/embeddings/momentum
S:Q	R 2BSGD/token_and_position_embedding_3/embedding_7/embeddings/momentum
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
2
!__inference__wrapped_model_305093ί
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
annotationsͺ *O’L
JG
"
input_7?????????R
!
input_8?????????
Ϊ2Χ
C__inference_model_3_layer_call_and_return_conditional_losses_307320
C__inference_model_3_layer_call_and_return_conditional_losses_307565
C__inference_model_3_layer_call_and_return_conditional_losses_306575
C__inference_model_3_layer_call_and_return_conditional_losses_306480ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
ξ2λ
(__inference_model_3_layer_call_fn_306749
(__inference_model_3_layer_call_fn_307721
(__inference_model_3_layer_call_fn_306922
(__inference_model_3_layer_call_fn_307643ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
?2ό
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_307745
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
annotationsͺ *
 
δ2α
?__inference_token_and_position_embedding_3_layer_call_fn_307754
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
annotationsͺ *
 
ξ2λ
D__inference_conv1d_6_layer_call_and_return_conditional_losses_307770’
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
annotationsͺ *
 
Σ2Π
)__inference_conv1d_6_layer_call_fn_307779’
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
annotationsͺ *
 
ͺ2§
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_305102Σ
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
annotationsͺ *3’0
.+'???????????????????????????
2
4__inference_average_pooling1d_9_layer_call_fn_305108Σ
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
annotationsͺ *3’0
.+'???????????????????????????
ξ2λ
D__inference_conv1d_7_layer_call_and_return_conditional_losses_307795’
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
annotationsͺ *
 
Σ2Π
)__inference_conv1d_7_layer_call_fn_307804’
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
annotationsͺ *
 
«2¨
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_305117Σ
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
annotationsͺ *3’0
.+'???????????????????????????
2
5__inference_average_pooling1d_10_layer_call_fn_305123Σ
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
annotationsͺ *3’0
.+'???????????????????????????
«2¨
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_305132Σ
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
annotationsͺ *3’0
.+'???????????????????????????
2
5__inference_average_pooling1d_11_layer_call_fn_305138Σ
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
annotationsͺ *3’0
.+'???????????????????????????
2
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_307860
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_307840
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_307922
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_307942΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
6__inference_batch_normalization_6_layer_call_fn_307873
6__inference_batch_normalization_6_layer_call_fn_307955
6__inference_batch_normalization_6_layer_call_fn_307968
6__inference_batch_normalization_6_layer_call_fn_307886΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_308106
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_308024
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_308086
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_308004΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
6__inference_batch_normalization_7_layer_call_fn_308132
6__inference_batch_normalization_7_layer_call_fn_308037
6__inference_batch_normalization_7_layer_call_fn_308050
6__inference_batch_normalization_7_layer_call_fn_308119΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
λ2θ
A__inference_add_3_layer_call_and_return_conditional_losses_308138’
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
annotationsͺ *
 
Π2Ν
&__inference_add_3_layer_call_fn_308144’
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
annotationsͺ *
 
Ψ2Υ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_308292
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_308419°
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
kwonlydefaultsͺ 
annotationsͺ *
 
’2
4__inference_transformer_block_7_layer_call_fn_308456
4__inference_transformer_block_7_layer_call_fn_308493°
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
kwonlydefaultsͺ 
annotationsͺ *
 
ε2β
V__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_308510
V__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_308499―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
―2¬
;__inference_global_average_pooling1d_3_layer_call_fn_308515
;__inference_global_average_pooling1d_3_layer_call_fn_308504―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ο2μ
E__inference_flatten_3_layer_call_and_return_conditional_losses_308521’
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
annotationsͺ *
 
Τ2Ρ
*__inference_flatten_3_layer_call_fn_308526’
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
annotationsͺ *
 
σ2π
I__inference_concatenate_3_layer_call_and_return_conditional_losses_308533’
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
annotationsͺ *
 
Ψ2Υ
.__inference_concatenate_3_layer_call_fn_308539’
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
annotationsͺ *
 
ξ2λ
D__inference_dense_25_layer_call_and_return_conditional_losses_308550’
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
annotationsͺ *
 
Σ2Π
)__inference_dense_25_layer_call_fn_308559’
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
annotationsͺ *
 
Κ2Η
F__inference_dropout_22_layer_call_and_return_conditional_losses_308576
F__inference_dropout_22_layer_call_and_return_conditional_losses_308571΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
+__inference_dropout_22_layer_call_fn_308581
+__inference_dropout_22_layer_call_fn_308586΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
ξ2λ
D__inference_dense_26_layer_call_and_return_conditional_losses_308597’
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
annotationsͺ *
 
Σ2Π
)__inference_dense_26_layer_call_fn_308606’
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
annotationsͺ *
 
Κ2Η
F__inference_dropout_23_layer_call_and_return_conditional_losses_308623
F__inference_dropout_23_layer_call_and_return_conditional_losses_308618΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
+__inference_dropout_23_layer_call_fn_308633
+__inference_dropout_23_layer_call_fn_308628΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
ξ2λ
D__inference_dense_27_layer_call_and_return_conditional_losses_308643’
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
annotationsͺ *
 
Σ2Π
)__inference_dense_27_layer_call_fn_308652’
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
annotationsͺ *
 
?BΟ
$__inference_signature_wrapper_307008input_7input_8"
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
2?ό
σ²ο
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
kwonlydefaultsͺ 
annotationsͺ *
 
2?ό
σ²ο
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
kwonlydefaultsͺ 
annotationsͺ *
 
ξ2λ
H__inference_sequential_7_layer_call_and_return_conditional_losses_308709
H__inference_sequential_7_layer_call_and_return_conditional_losses_308766
H__inference_sequential_7_layer_call_and_return_conditional_losses_305530
H__inference_sequential_7_layer_call_and_return_conditional_losses_305516ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
2?
-__inference_sequential_7_layer_call_fn_305558
-__inference_sequential_7_layer_call_fn_308779
-__inference_sequential_7_layer_call_fn_308792
-__inference_sequential_7_layer_call_fn_305585ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
Ί2·΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
Ί2·΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
Ί2·΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
Ί2·΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
΅2²―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
΅2²―
¦²’
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults’

 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ί2·΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
Ί2·΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
ξ2λ
D__inference_dense_23_layer_call_and_return_conditional_losses_308823’
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
annotationsͺ *
 
Σ2Π
)__inference_dense_23_layer_call_fn_308832’
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
annotationsͺ *
 
ξ2λ
D__inference_dense_24_layer_call_and_return_conditional_losses_308862’
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
annotationsͺ *
 
Σ2Π
)__inference_dense_24_layer_call_fn_308871’
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
annotationsͺ *
 ξ
!__inference__wrapped_model_305093Θ6!"+,=:<;FCEDefopyzY’V
O’L
JG
"
input_7?????????R
!
input_8?????????
ͺ "3ͺ0
.
dense_27"
dense_27?????????Υ
A__inference_add_3_layer_call_and_return_conditional_losses_308138b’_
X’U
SP
&#
inputs/0?????????# 
&#
inputs/1?????????# 
ͺ ")’&

0?????????# 
 ­
&__inference_add_3_layer_call_fn_308144b’_
X’U
SP
&#
inputs/0?????????# 
&#
inputs/1?????????# 
ͺ "?????????# Ω
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_305117E’B
;’8
63
inputs'???????????????????????????
ͺ ";’8
1.
0'???????????????????????????
 °
5__inference_average_pooling1d_10_layer_call_fn_305123wE’B
;’8
63
inputs'???????????????????????????
ͺ ".+'???????????????????????????Ω
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_305132E’B
;’8
63
inputs'???????????????????????????
ͺ ";’8
1.
0'???????????????????????????
 °
5__inference_average_pooling1d_11_layer_call_fn_305138wE’B
;’8
63
inputs'???????????????????????????
ͺ ".+'???????????????????????????Ψ
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_305102E’B
;’8
63
inputs'???????????????????????????
ͺ ";’8
1.
0'???????????????????????????
 ―
4__inference_average_pooling1d_9_layer_call_fn_305108wE’B
;’8
63
inputs'???????????????????????????
ͺ ".+'???????????????????????????Ρ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_307840|<=:;@’=
6’3
-*
inputs?????????????????? 
p
ͺ "2’/
(%
0?????????????????? 
 Ρ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_307860|=:<;@’=
6’3
-*
inputs?????????????????? 
p 
ͺ "2’/
(%
0?????????????????? 
 Ώ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_307922j<=:;7’4
-’*
$!
inputs?????????# 
p
ͺ ")’&

0?????????# 
 Ώ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_307942j=:<;7’4
-’*
$!
inputs?????????# 
p 
ͺ ")’&

0?????????# 
 ©
6__inference_batch_normalization_6_layer_call_fn_307873o<=:;@’=
6’3
-*
inputs?????????????????? 
p
ͺ "%"?????????????????? ©
6__inference_batch_normalization_6_layer_call_fn_307886o=:<;@’=
6’3
-*
inputs?????????????????? 
p 
ͺ "%"?????????????????? 
6__inference_batch_normalization_6_layer_call_fn_307955]<=:;7’4
-’*
$!
inputs?????????# 
p
ͺ "?????????# 
6__inference_batch_normalization_6_layer_call_fn_307968]=:<;7’4
-’*
$!
inputs?????????# 
p 
ͺ "?????????# Ρ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_308004|EFCD@’=
6’3
-*
inputs?????????????????? 
p
ͺ "2’/
(%
0?????????????????? 
 Ρ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_308024|FCED@’=
6’3
-*
inputs?????????????????? 
p 
ͺ "2’/
(%
0?????????????????? 
 Ώ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_308086jEFCD7’4
-’*
$!
inputs?????????# 
p
ͺ ")’&

0?????????# 
 Ώ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_308106jFCED7’4
-’*
$!
inputs?????????# 
p 
ͺ ")’&

0?????????# 
 ©
6__inference_batch_normalization_7_layer_call_fn_308037oEFCD@’=
6’3
-*
inputs?????????????????? 
p
ͺ "%"?????????????????? ©
6__inference_batch_normalization_7_layer_call_fn_308050oFCED@’=
6’3
-*
inputs?????????????????? 
p 
ͺ "%"?????????????????? 
6__inference_batch_normalization_7_layer_call_fn_308119]EFCD7’4
-’*
$!
inputs?????????# 
p
ͺ "?????????# 
6__inference_batch_normalization_7_layer_call_fn_308132]FCED7’4
-’*
$!
inputs?????????# 
p 
ͺ "?????????# Ρ
I__inference_concatenate_3_layer_call_and_return_conditional_losses_308533Z’W
P’M
KH
"
inputs/0????????? 
"
inputs/1?????????
ͺ "%’"

0?????????(
 ¨
.__inference_concatenate_3_layer_call_fn_308539vZ’W
P’M
KH
"
inputs/0????????? 
"
inputs/1?????????
ͺ "?????????(?
D__inference_conv1d_6_layer_call_and_return_conditional_losses_307770f!"4’1
*’'
%"
inputs?????????R 
ͺ "*’'
 
0?????????R 
 
)__inference_conv1d_6_layer_call_fn_307779Y!"4’1
*’'
%"
inputs?????????R 
ͺ "?????????R ?
D__inference_conv1d_7_layer_call_and_return_conditional_losses_307795f+,4’1
*’'
%"
inputs?????????ή 
ͺ "*’'
 
0?????????ή 
 
)__inference_conv1d_7_layer_call_fn_307804Y+,4’1
*’'
%"
inputs?????????ή 
ͺ "?????????ή ?
D__inference_dense_23_layer_call_and_return_conditional_losses_308823f3’0
)’&
$!
inputs?????????# 
ͺ ")’&

0?????????#@
 
)__inference_dense_23_layer_call_fn_308832Y3’0
)’&
$!
inputs?????????# 
ͺ "?????????#@?
D__inference_dense_24_layer_call_and_return_conditional_losses_308862f3’0
)’&
$!
inputs?????????#@
ͺ ")’&

0?????????# 
 
)__inference_dense_24_layer_call_fn_308871Y3’0
)’&
$!
inputs?????????#@
ͺ "?????????# €
D__inference_dense_25_layer_call_and_return_conditional_losses_308550\ef/’,
%’"
 
inputs?????????(
ͺ "%’"

0?????????@
 |
)__inference_dense_25_layer_call_fn_308559Oef/’,
%’"
 
inputs?????????(
ͺ "?????????@€
D__inference_dense_26_layer_call_and_return_conditional_losses_308597\op/’,
%’"
 
inputs?????????@
ͺ "%’"

0?????????@
 |
)__inference_dense_26_layer_call_fn_308606Oop/’,
%’"
 
inputs?????????@
ͺ "?????????@€
D__inference_dense_27_layer_call_and_return_conditional_losses_308643\yz/’,
%’"
 
inputs?????????@
ͺ "%’"

0?????????
 |
)__inference_dense_27_layer_call_fn_308652Oyz/’,
%’"
 
inputs?????????@
ͺ "?????????¦
F__inference_dropout_22_layer_call_and_return_conditional_losses_308571\3’0
)’&
 
inputs?????????@
p
ͺ "%’"

0?????????@
 ¦
F__inference_dropout_22_layer_call_and_return_conditional_losses_308576\3’0
)’&
 
inputs?????????@
p 
ͺ "%’"

0?????????@
 ~
+__inference_dropout_22_layer_call_fn_308581O3’0
)’&
 
inputs?????????@
p
ͺ "?????????@~
+__inference_dropout_22_layer_call_fn_308586O3’0
)’&
 
inputs?????????@
p 
ͺ "?????????@¦
F__inference_dropout_23_layer_call_and_return_conditional_losses_308618\3’0
)’&
 
inputs?????????@
p
ͺ "%’"

0?????????@
 ¦
F__inference_dropout_23_layer_call_and_return_conditional_losses_308623\3’0
)’&
 
inputs?????????@
p 
ͺ "%’"

0?????????@
 ~
+__inference_dropout_23_layer_call_fn_308628O3’0
)’&
 
inputs?????????@
p
ͺ "?????????@~
+__inference_dropout_23_layer_call_fn_308633O3’0
)’&
 
inputs?????????@
p 
ͺ "?????????@‘
E__inference_flatten_3_layer_call_and_return_conditional_losses_308521X/’,
%’"
 
inputs????????? 
ͺ "%’"

0????????? 
 y
*__inference_flatten_3_layer_call_fn_308526K/’,
%’"
 
inputs????????? 
ͺ "????????? Ί
V__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_308499`7’4
-’*
$!
inputs?????????# 

 
ͺ "%’"

0????????? 
 Υ
V__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_308510{I’F
?’<
63
inputs'???????????????????????????

 
ͺ ".’+
$!
0??????????????????
 
;__inference_global_average_pooling1d_3_layer_call_fn_308504S7’4
-’*
$!
inputs?????????# 

 
ͺ "????????? ­
;__inference_global_average_pooling1d_3_layer_call_fn_308515nI’F
?’<
63
inputs'???????????????????????????

 
ͺ "!??????????????????
C__inference_model_3_layer_call_and_return_conditional_losses_306480Β6!"+,<=:;EFCDefopyza’^
W’T
JG
"
input_7?????????R
!
input_8?????????
p

 
ͺ "%’"

0?????????
 
C__inference_model_3_layer_call_and_return_conditional_losses_306575Β6!"+,=:<;FCEDefopyza’^
W’T
JG
"
input_7?????????R
!
input_8?????????
p 

 
ͺ "%’"

0?????????
 
C__inference_model_3_layer_call_and_return_conditional_losses_307320Δ6!"+,<=:;EFCDefopyzc’`
Y’V
LI
# 
inputs/0?????????R
"
inputs/1?????????
p

 
ͺ "%’"

0?????????
 
C__inference_model_3_layer_call_and_return_conditional_losses_307565Δ6!"+,=:<;FCEDefopyzc’`
Y’V
LI
# 
inputs/0?????????R
"
inputs/1?????????
p 

 
ͺ "%’"

0?????????
 β
(__inference_model_3_layer_call_fn_306749΅6!"+,<=:;EFCDefopyza’^
W’T
JG
"
input_7?????????R
!
input_8?????????
p

 
ͺ "?????????β
(__inference_model_3_layer_call_fn_306922΅6!"+,=:<;FCEDefopyza’^
W’T
JG
"
input_7?????????R
!
input_8?????????
p 

 
ͺ "?????????δ
(__inference_model_3_layer_call_fn_307643·6!"+,<=:;EFCDefopyzc’`
Y’V
LI
# 
inputs/0?????????R
"
inputs/1?????????
p

 
ͺ "?????????δ
(__inference_model_3_layer_call_fn_307721·6!"+,=:<;FCEDefopyzc’`
Y’V
LI
# 
inputs/0?????????R
"
inputs/1?????????
p 

 
ͺ "?????????Ζ
H__inference_sequential_7_layer_call_and_return_conditional_losses_305516zC’@
9’6
,)
dense_23_input?????????# 
p

 
ͺ ")’&

0?????????# 
 Ζ
H__inference_sequential_7_layer_call_and_return_conditional_losses_305530zC’@
9’6
,)
dense_23_input?????????# 
p 

 
ͺ ")’&

0?????????# 
 Ύ
H__inference_sequential_7_layer_call_and_return_conditional_losses_308709r;’8
1’.
$!
inputs?????????# 
p

 
ͺ ")’&

0?????????# 
 Ύ
H__inference_sequential_7_layer_call_and_return_conditional_losses_308766r;’8
1’.
$!
inputs?????????# 
p 

 
ͺ ")’&

0?????????# 
 
-__inference_sequential_7_layer_call_fn_305558mC’@
9’6
,)
dense_23_input?????????# 
p

 
ͺ "?????????# 
-__inference_sequential_7_layer_call_fn_305585mC’@
9’6
,)
dense_23_input?????????# 
p 

 
ͺ "?????????# 
-__inference_sequential_7_layer_call_fn_308779e;’8
1’.
$!
inputs?????????# 
p

 
ͺ "?????????# 
-__inference_sequential_7_layer_call_fn_308792e;’8
1’.
$!
inputs?????????# 
p 

 
ͺ "?????????# 
$__inference_signature_wrapper_307008Ω6!"+,=:<;FCEDefopyzj’g
’ 
`ͺ]
-
input_7"
input_7?????????R
,
input_8!
input_8?????????"3ͺ0
.
dense_27"
dense_27?????????½
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_307745_+’(
!’

x?????????R
ͺ "*’'
 
0?????????R 
 
?__inference_token_and_position_embedding_3_layer_call_fn_307754R+’(
!’

x?????????R
ͺ "?????????R Ϊ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_308292 7’4
-’*
$!
inputs?????????# 
p
ͺ ")’&

0?????????# 
 Ϊ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_308419 7’4
-’*
$!
inputs?????????# 
p 
ͺ ")’&

0?????????# 
 ±
4__inference_transformer_block_7_layer_call_fn_308456y 7’4
-’*
$!
inputs?????????# 
p
ͺ "?????????# ±
4__inference_transformer_block_7_layer_call_fn_308493y 7’4
-’*
$!
inputs?????????# 
p 
ͺ "?????????# 