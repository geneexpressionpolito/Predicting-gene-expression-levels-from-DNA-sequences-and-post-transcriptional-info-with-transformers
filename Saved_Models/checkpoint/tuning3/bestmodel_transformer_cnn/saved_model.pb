ä1
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
 "serve*2.4.12v2.4.1-0-g85c8b2a817f8+
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
:  *
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
: *
dtype0

batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_4/gamma

/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
: *
dtype0

batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_4/beta

.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
: *
dtype0

!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_4/moving_mean

5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
: *
dtype0
¢
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_4/moving_variance

9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
: *
dtype0

batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_5/gamma

/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
: *
dtype0

batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_5/beta

.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
: *
dtype0

!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_5/moving_mean

5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
: *
dtype0
¢
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_5/moving_variance

9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
: *
dtype0
z
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( * 
shared_namedense_18/kernel
s
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes

:( *
dtype0
r
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_18/bias
k
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes
: *
dtype0
z
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_19/kernel
s
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes

:  *
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes
: *
dtype0
z
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_20/kernel
s
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes

: *
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
j
Adamax/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdamax/iter
c
Adamax/iter/Read/ReadVariableOpReadVariableOpAdamax/iter*
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
shape:	¼i *F
shared_name75token_and_position_embedding_2/embedding_5/embeddings
À
Itoken_and_position_embedding_2/embedding_5/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_2/embedding_5/embeddings*
_output_shapes
:	¼i *
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
:  * 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

:  *
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
: *
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

:  *
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

Adamax/conv1d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *)
shared_nameAdamax/conv1d_2/kernel/m

,Adamax/conv1d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/conv1d_2/kernel/m*"
_output_shapes
:  *
dtype0

Adamax/conv1d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdamax/conv1d_2/bias/m
}
*Adamax/conv1d_2/bias/m/Read/ReadVariableOpReadVariableOpAdamax/conv1d_2/bias/m*
_output_shapes
: *
dtype0
 
$Adamax/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adamax/batch_normalization_4/gamma/m

8Adamax/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp$Adamax/batch_normalization_4/gamma/m*
_output_shapes
: *
dtype0

#Adamax/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adamax/batch_normalization_4/beta/m

7Adamax/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp#Adamax/batch_normalization_4/beta/m*
_output_shapes
: *
dtype0
 
$Adamax/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adamax/batch_normalization_5/gamma/m

8Adamax/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp$Adamax/batch_normalization_5/gamma/m*
_output_shapes
: *
dtype0

#Adamax/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adamax/batch_normalization_5/beta/m

7Adamax/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp#Adamax/batch_normalization_5/beta/m*
_output_shapes
: *
dtype0

Adamax/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *)
shared_nameAdamax/dense_18/kernel/m

,Adamax/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/dense_18/kernel/m*
_output_shapes

:( *
dtype0

Adamax/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdamax/dense_18/bias/m
}
*Adamax/dense_18/bias/m/Read/ReadVariableOpReadVariableOpAdamax/dense_18/bias/m*
_output_shapes
: *
dtype0

Adamax/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *)
shared_nameAdamax/dense_19/kernel/m

,Adamax/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/dense_19/kernel/m*
_output_shapes

:  *
dtype0

Adamax/dense_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdamax/dense_19/bias/m
}
*Adamax/dense_19/bias/m/Read/ReadVariableOpReadVariableOpAdamax/dense_19/bias/m*
_output_shapes
: *
dtype0

Adamax/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_nameAdamax/dense_20/kernel/m

,Adamax/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/dense_20/kernel/m*
_output_shapes

: *
dtype0

Adamax/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdamax/dense_20/bias/m
}
*Adamax/dense_20/bias/m/Read/ReadVariableOpReadVariableOpAdamax/dense_20/bias/m*
_output_shapes
:*
dtype0
Ø
>Adamax/token_and_position_embedding_2/embedding_4/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *O
shared_name@>Adamax/token_and_position_embedding_2/embedding_4/embeddings/m
Ñ
RAdamax/token_and_position_embedding_2/embedding_4/embeddings/m/Read/ReadVariableOpReadVariableOp>Adamax/token_and_position_embedding_2/embedding_4/embeddings/m*
_output_shapes

: *
dtype0
Ù
>Adamax/token_and_position_embedding_2/embedding_5/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¼i *O
shared_name@>Adamax/token_and_position_embedding_2/embedding_5/embeddings/m
Ò
RAdamax/token_and_position_embedding_2/embedding_5/embeddings/m/Read/ReadVariableOpReadVariableOp>Adamax/token_and_position_embedding_2/embedding_5/embeddings/m*
_output_shapes
:	¼i *
dtype0
à
@Adamax/transformer_block_5/multi_head_attention_5/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *Q
shared_nameB@Adamax/transformer_block_5/multi_head_attention_5/query/kernel/m
Ù
TAdamax/transformer_block_5/multi_head_attention_5/query/kernel/m/Read/ReadVariableOpReadVariableOp@Adamax/transformer_block_5/multi_head_attention_5/query/kernel/m*"
_output_shapes
:  *
dtype0
Ø
>Adamax/transformer_block_5/multi_head_attention_5/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *O
shared_name@>Adamax/transformer_block_5/multi_head_attention_5/query/bias/m
Ñ
RAdamax/transformer_block_5/multi_head_attention_5/query/bias/m/Read/ReadVariableOpReadVariableOp>Adamax/transformer_block_5/multi_head_attention_5/query/bias/m*
_output_shapes

: *
dtype0
Ü
>Adamax/transformer_block_5/multi_head_attention_5/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *O
shared_name@>Adamax/transformer_block_5/multi_head_attention_5/key/kernel/m
Õ
RAdamax/transformer_block_5/multi_head_attention_5/key/kernel/m/Read/ReadVariableOpReadVariableOp>Adamax/transformer_block_5/multi_head_attention_5/key/kernel/m*"
_output_shapes
:  *
dtype0
Ô
<Adamax/transformer_block_5/multi_head_attention_5/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *M
shared_name><Adamax/transformer_block_5/multi_head_attention_5/key/bias/m
Í
PAdamax/transformer_block_5/multi_head_attention_5/key/bias/m/Read/ReadVariableOpReadVariableOp<Adamax/transformer_block_5/multi_head_attention_5/key/bias/m*
_output_shapes

: *
dtype0
à
@Adamax/transformer_block_5/multi_head_attention_5/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *Q
shared_nameB@Adamax/transformer_block_5/multi_head_attention_5/value/kernel/m
Ù
TAdamax/transformer_block_5/multi_head_attention_5/value/kernel/m/Read/ReadVariableOpReadVariableOp@Adamax/transformer_block_5/multi_head_attention_5/value/kernel/m*"
_output_shapes
:  *
dtype0
Ø
>Adamax/transformer_block_5/multi_head_attention_5/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *O
shared_name@>Adamax/transformer_block_5/multi_head_attention_5/value/bias/m
Ñ
RAdamax/transformer_block_5/multi_head_attention_5/value/bias/m/Read/ReadVariableOpReadVariableOp>Adamax/transformer_block_5/multi_head_attention_5/value/bias/m*
_output_shapes

: *
dtype0
ö
KAdamax/transformer_block_5/multi_head_attention_5/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *\
shared_nameMKAdamax/transformer_block_5/multi_head_attention_5/attention_output/kernel/m
ï
_Adamax/transformer_block_5/multi_head_attention_5/attention_output/kernel/m/Read/ReadVariableOpReadVariableOpKAdamax/transformer_block_5/multi_head_attention_5/attention_output/kernel/m*"
_output_shapes
:  *
dtype0
ê
IAdamax/transformer_block_5/multi_head_attention_5/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Z
shared_nameKIAdamax/transformer_block_5/multi_head_attention_5/attention_output/bias/m
ã
]Adamax/transformer_block_5/multi_head_attention_5/attention_output/bias/m/Read/ReadVariableOpReadVariableOpIAdamax/transformer_block_5/multi_head_attention_5/attention_output/bias/m*
_output_shapes
: *
dtype0

Adamax/dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *)
shared_nameAdamax/dense_16/kernel/m

,Adamax/dense_16/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/dense_16/kernel/m*
_output_shapes

:  *
dtype0

Adamax/dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdamax/dense_16/bias/m
}
*Adamax/dense_16/bias/m/Read/ReadVariableOpReadVariableOpAdamax/dense_16/bias/m*
_output_shapes
: *
dtype0

Adamax/dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *)
shared_nameAdamax/dense_17/kernel/m

,Adamax/dense_17/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/dense_17/kernel/m*
_output_shapes

:  *
dtype0

Adamax/dense_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdamax/dense_17/bias/m
}
*Adamax/dense_17/bias/m/Read/ReadVariableOpReadVariableOpAdamax/dense_17/bias/m*
_output_shapes
: *
dtype0
Ê
9Adamax/transformer_block_5/layer_normalization_10/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *J
shared_name;9Adamax/transformer_block_5/layer_normalization_10/gamma/m
Ã
MAdamax/transformer_block_5/layer_normalization_10/gamma/m/Read/ReadVariableOpReadVariableOp9Adamax/transformer_block_5/layer_normalization_10/gamma/m*
_output_shapes
: *
dtype0
È
8Adamax/transformer_block_5/layer_normalization_10/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adamax/transformer_block_5/layer_normalization_10/beta/m
Á
LAdamax/transformer_block_5/layer_normalization_10/beta/m/Read/ReadVariableOpReadVariableOp8Adamax/transformer_block_5/layer_normalization_10/beta/m*
_output_shapes
: *
dtype0
Ê
9Adamax/transformer_block_5/layer_normalization_11/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *J
shared_name;9Adamax/transformer_block_5/layer_normalization_11/gamma/m
Ã
MAdamax/transformer_block_5/layer_normalization_11/gamma/m/Read/ReadVariableOpReadVariableOp9Adamax/transformer_block_5/layer_normalization_11/gamma/m*
_output_shapes
: *
dtype0
È
8Adamax/transformer_block_5/layer_normalization_11/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adamax/transformer_block_5/layer_normalization_11/beta/m
Á
LAdamax/transformer_block_5/layer_normalization_11/beta/m/Read/ReadVariableOpReadVariableOp8Adamax/transformer_block_5/layer_normalization_11/beta/m*
_output_shapes
: *
dtype0

Adamax/conv1d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *)
shared_nameAdamax/conv1d_2/kernel/v

,Adamax/conv1d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/conv1d_2/kernel/v*"
_output_shapes
:  *
dtype0

Adamax/conv1d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdamax/conv1d_2/bias/v
}
*Adamax/conv1d_2/bias/v/Read/ReadVariableOpReadVariableOpAdamax/conv1d_2/bias/v*
_output_shapes
: *
dtype0
 
$Adamax/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adamax/batch_normalization_4/gamma/v

8Adamax/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp$Adamax/batch_normalization_4/gamma/v*
_output_shapes
: *
dtype0

#Adamax/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adamax/batch_normalization_4/beta/v

7Adamax/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp#Adamax/batch_normalization_4/beta/v*
_output_shapes
: *
dtype0
 
$Adamax/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adamax/batch_normalization_5/gamma/v

8Adamax/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp$Adamax/batch_normalization_5/gamma/v*
_output_shapes
: *
dtype0

#Adamax/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adamax/batch_normalization_5/beta/v

7Adamax/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp#Adamax/batch_normalization_5/beta/v*
_output_shapes
: *
dtype0

Adamax/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *)
shared_nameAdamax/dense_18/kernel/v

,Adamax/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/dense_18/kernel/v*
_output_shapes

:( *
dtype0

Adamax/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdamax/dense_18/bias/v
}
*Adamax/dense_18/bias/v/Read/ReadVariableOpReadVariableOpAdamax/dense_18/bias/v*
_output_shapes
: *
dtype0

Adamax/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *)
shared_nameAdamax/dense_19/kernel/v

,Adamax/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/dense_19/kernel/v*
_output_shapes

:  *
dtype0

Adamax/dense_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdamax/dense_19/bias/v
}
*Adamax/dense_19/bias/v/Read/ReadVariableOpReadVariableOpAdamax/dense_19/bias/v*
_output_shapes
: *
dtype0

Adamax/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_nameAdamax/dense_20/kernel/v

,Adamax/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/dense_20/kernel/v*
_output_shapes

: *
dtype0

Adamax/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdamax/dense_20/bias/v
}
*Adamax/dense_20/bias/v/Read/ReadVariableOpReadVariableOpAdamax/dense_20/bias/v*
_output_shapes
:*
dtype0
Ø
>Adamax/token_and_position_embedding_2/embedding_4/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *O
shared_name@>Adamax/token_and_position_embedding_2/embedding_4/embeddings/v
Ñ
RAdamax/token_and_position_embedding_2/embedding_4/embeddings/v/Read/ReadVariableOpReadVariableOp>Adamax/token_and_position_embedding_2/embedding_4/embeddings/v*
_output_shapes

: *
dtype0
Ù
>Adamax/token_and_position_embedding_2/embedding_5/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¼i *O
shared_name@>Adamax/token_and_position_embedding_2/embedding_5/embeddings/v
Ò
RAdamax/token_and_position_embedding_2/embedding_5/embeddings/v/Read/ReadVariableOpReadVariableOp>Adamax/token_and_position_embedding_2/embedding_5/embeddings/v*
_output_shapes
:	¼i *
dtype0
à
@Adamax/transformer_block_5/multi_head_attention_5/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *Q
shared_nameB@Adamax/transformer_block_5/multi_head_attention_5/query/kernel/v
Ù
TAdamax/transformer_block_5/multi_head_attention_5/query/kernel/v/Read/ReadVariableOpReadVariableOp@Adamax/transformer_block_5/multi_head_attention_5/query/kernel/v*"
_output_shapes
:  *
dtype0
Ø
>Adamax/transformer_block_5/multi_head_attention_5/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *O
shared_name@>Adamax/transformer_block_5/multi_head_attention_5/query/bias/v
Ñ
RAdamax/transformer_block_5/multi_head_attention_5/query/bias/v/Read/ReadVariableOpReadVariableOp>Adamax/transformer_block_5/multi_head_attention_5/query/bias/v*
_output_shapes

: *
dtype0
Ü
>Adamax/transformer_block_5/multi_head_attention_5/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *O
shared_name@>Adamax/transformer_block_5/multi_head_attention_5/key/kernel/v
Õ
RAdamax/transformer_block_5/multi_head_attention_5/key/kernel/v/Read/ReadVariableOpReadVariableOp>Adamax/transformer_block_5/multi_head_attention_5/key/kernel/v*"
_output_shapes
:  *
dtype0
Ô
<Adamax/transformer_block_5/multi_head_attention_5/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *M
shared_name><Adamax/transformer_block_5/multi_head_attention_5/key/bias/v
Í
PAdamax/transformer_block_5/multi_head_attention_5/key/bias/v/Read/ReadVariableOpReadVariableOp<Adamax/transformer_block_5/multi_head_attention_5/key/bias/v*
_output_shapes

: *
dtype0
à
@Adamax/transformer_block_5/multi_head_attention_5/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *Q
shared_nameB@Adamax/transformer_block_5/multi_head_attention_5/value/kernel/v
Ù
TAdamax/transformer_block_5/multi_head_attention_5/value/kernel/v/Read/ReadVariableOpReadVariableOp@Adamax/transformer_block_5/multi_head_attention_5/value/kernel/v*"
_output_shapes
:  *
dtype0
Ø
>Adamax/transformer_block_5/multi_head_attention_5/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *O
shared_name@>Adamax/transformer_block_5/multi_head_attention_5/value/bias/v
Ñ
RAdamax/transformer_block_5/multi_head_attention_5/value/bias/v/Read/ReadVariableOpReadVariableOp>Adamax/transformer_block_5/multi_head_attention_5/value/bias/v*
_output_shapes

: *
dtype0
ö
KAdamax/transformer_block_5/multi_head_attention_5/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *\
shared_nameMKAdamax/transformer_block_5/multi_head_attention_5/attention_output/kernel/v
ï
_Adamax/transformer_block_5/multi_head_attention_5/attention_output/kernel/v/Read/ReadVariableOpReadVariableOpKAdamax/transformer_block_5/multi_head_attention_5/attention_output/kernel/v*"
_output_shapes
:  *
dtype0
ê
IAdamax/transformer_block_5/multi_head_attention_5/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Z
shared_nameKIAdamax/transformer_block_5/multi_head_attention_5/attention_output/bias/v
ã
]Adamax/transformer_block_5/multi_head_attention_5/attention_output/bias/v/Read/ReadVariableOpReadVariableOpIAdamax/transformer_block_5/multi_head_attention_5/attention_output/bias/v*
_output_shapes
: *
dtype0

Adamax/dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *)
shared_nameAdamax/dense_16/kernel/v

,Adamax/dense_16/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/dense_16/kernel/v*
_output_shapes

:  *
dtype0

Adamax/dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdamax/dense_16/bias/v
}
*Adamax/dense_16/bias/v/Read/ReadVariableOpReadVariableOpAdamax/dense_16/bias/v*
_output_shapes
: *
dtype0

Adamax/dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *)
shared_nameAdamax/dense_17/kernel/v

,Adamax/dense_17/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/dense_17/kernel/v*
_output_shapes

:  *
dtype0

Adamax/dense_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdamax/dense_17/bias/v
}
*Adamax/dense_17/bias/v/Read/ReadVariableOpReadVariableOpAdamax/dense_17/bias/v*
_output_shapes
: *
dtype0
Ê
9Adamax/transformer_block_5/layer_normalization_10/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *J
shared_name;9Adamax/transformer_block_5/layer_normalization_10/gamma/v
Ã
MAdamax/transformer_block_5/layer_normalization_10/gamma/v/Read/ReadVariableOpReadVariableOp9Adamax/transformer_block_5/layer_normalization_10/gamma/v*
_output_shapes
: *
dtype0
È
8Adamax/transformer_block_5/layer_normalization_10/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adamax/transformer_block_5/layer_normalization_10/beta/v
Á
LAdamax/transformer_block_5/layer_normalization_10/beta/v/Read/ReadVariableOpReadVariableOp8Adamax/transformer_block_5/layer_normalization_10/beta/v*
_output_shapes
: *
dtype0
Ê
9Adamax/transformer_block_5/layer_normalization_11/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *J
shared_name;9Adamax/transformer_block_5/layer_normalization_11/gamma/v
Ã
MAdamax/transformer_block_5/layer_normalization_11/gamma/v/Read/ReadVariableOpReadVariableOp9Adamax/transformer_block_5/layer_normalization_11/gamma/v*
_output_shapes
: *
dtype0
È
8Adamax/transformer_block_5/layer_normalization_11/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adamax/transformer_block_5/layer_normalization_11/beta/v
Á
LAdamax/transformer_block_5/layer_normalization_11/beta/v/Read/ReadVariableOpReadVariableOp8Adamax/transformer_block_5/layer_normalization_11/beta/v*
_output_shapes
: *
dtype0

NoOpNoOp
áÈ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*È
valueÈBÈ BÈ
¥
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
n
	token_emb
pos_emb
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
R
$	variables
%trainable_variables
&regularization_losses
'	keras_api
R
(	variables
)trainable_variables
*regularization_losses
+	keras_api

,axis
	-gamma
.beta
/moving_mean
0moving_variance
1	variables
2trainable_variables
3regularization_losses
4	keras_api

5axis
	6gamma
7beta
8moving_mean
9moving_variance
:	variables
;trainable_variables
<regularization_losses
=	keras_api
R
>	variables
?trainable_variables
@regularization_losses
A	keras_api
 
Batt
Cffn
D
layernorm1
E
layernorm2
Fdropout1
Gdropout2
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
R
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
 
R
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
h

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
R
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
h

^kernel
_bias
`	variables
atrainable_variables
bregularization_losses
c	keras_api
R
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h

hkernel
ibias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
¢

nbeta_1

obeta_2
	pdecay
qlearning_rate
ritermümý-mþ.mÿ6m7mTmUm^m_mhmimsmtmumvmwmxmymzm{m|m}m~mm	m	m	m	m	mvv-v.v6v7vTv Uv¡^v¢_v£hv¤iv¥sv¦tv§uv¨vv©wvªxv«yv¬zv­{v®|v¯}v°~v±v²	v³	v´	vµ	v¶	v·

s0
t1
2
3
-4
.5
/6
07
68
79
810
911
u12
v13
w14
x15
y16
z17
{18
|19
}20
~21
22
23
24
25
26
27
T28
U29
^30
_31
h32
i33
ë
s0
t1
2
3
-4
.5
66
77
u8
v9
w10
x11
y12
z13
{14
|15
}16
~17
18
19
20
21
22
23
T24
U25
^26
_27
h28
i29
 
²
	variables
metrics
trainable_variables
layer_metrics
regularization_losses
non_trainable_variables
layers
 layer_regularization_losses
 
f
s
embeddings
	variables
trainable_variables
regularization_losses
	keras_api
f
t
embeddings
	variables
trainable_variables
regularization_losses
	keras_api

s0
t1

s0
t1
 
²
	variables
metrics
trainable_variables
layer_metrics
regularization_losses
non_trainable_variables
layers
 layer_regularization_losses
[Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
²
 	variables
metrics
!trainable_variables
layer_metrics
"regularization_losses
non_trainable_variables
layers
 layer_regularization_losses
 
 
 
²
$	variables
metrics
%trainable_variables
layer_metrics
&regularization_losses
non_trainable_variables
layers
  layer_regularization_losses
 
 
 
²
(	variables
¡metrics
)trainable_variables
¢layer_metrics
*regularization_losses
£non_trainable_variables
¤layers
 ¥layer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

-0
.1
/2
03

-0
.1
 
²
1	variables
¦metrics
2trainable_variables
§layer_metrics
3regularization_losses
¨non_trainable_variables
©layers
 ªlayer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_5/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_5/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_5/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_5/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

60
71
82
93

60
71
 
²
:	variables
«metrics
;trainable_variables
¬layer_metrics
<regularization_losses
­non_trainable_variables
®layers
 ¯layer_regularization_losses
 
 
 
²
>	variables
°metrics
?trainable_variables
±layer_metrics
@regularization_losses
²non_trainable_variables
³layers
 ´layer_regularization_losses
Å
µ_query_dense
¶
_key_dense
·_value_dense
¸_softmax
¹_dropout_layer
º_output_dense
»	variables
¼trainable_variables
½regularization_losses
¾	keras_api
¨
¿layer_with_weights-0
¿layer-0
Àlayer_with_weights-1
Àlayer-1
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
x
	Åaxis

gamma
	beta
Æ	variables
Çtrainable_variables
Èregularization_losses
É	keras_api
x
	Êaxis

gamma
	beta
Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
V
Ï	variables
Ðtrainable_variables
Ñregularization_losses
Ò	keras_api
V
Ó	variables
Ôtrainable_variables
Õregularization_losses
Ö	keras_api
{
u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
11
12
13
14
15
{
u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
11
12
13
14
15
 
²
H	variables
×metrics
Itrainable_variables
Ølayer_metrics
Jregularization_losses
Ùnon_trainable_variables
Úlayers
 Ûlayer_regularization_losses
 
 
 
²
L	variables
Ümetrics
Mtrainable_variables
Ýlayer_metrics
Nregularization_losses
Þnon_trainable_variables
ßlayers
 àlayer_regularization_losses
 
 
 
²
P	variables
ámetrics
Qtrainable_variables
âlayer_metrics
Rregularization_losses
ãnon_trainable_variables
älayers
 ålayer_regularization_losses
[Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_18/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

T0
U1

T0
U1
 
²
V	variables
æmetrics
Wtrainable_variables
çlayer_metrics
Xregularization_losses
ènon_trainable_variables
élayers
 êlayer_regularization_losses
 
 
 
²
Z	variables
ëmetrics
[trainable_variables
ìlayer_metrics
\regularization_losses
ínon_trainable_variables
îlayers
 ïlayer_regularization_losses
[Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_19/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

^0
_1

^0
_1
 
²
`	variables
ðmetrics
atrainable_variables
ñlayer_metrics
bregularization_losses
ònon_trainable_variables
ólayers
 ôlayer_regularization_losses
 
 
 
²
d	variables
õmetrics
etrainable_variables
ölayer_metrics
fregularization_losses
÷non_trainable_variables
ølayers
 ùlayer_regularization_losses
[Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_20/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

h0
i1

h0
i1
 
²
j	variables
úmetrics
ktrainable_variables
ûlayer_metrics
lregularization_losses
ünon_trainable_variables
ýlayers
 þlayer_regularization_losses
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEAdamax/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5token_and_position_embedding_2/embedding_4/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5token_and_position_embedding_2/embedding_5/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7transformer_block_5/multi_head_attention_5/query/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_5/multi_head_attention_5/query/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_5/multi_head_attention_5/key/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE3transformer_block_5/multi_head_attention_5/key/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7transformer_block_5/multi_head_attention_5/value/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_5/multi_head_attention_5/value/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEBtransformer_block_5/multi_head_attention_5/attention_output/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE@transformer_block_5/multi_head_attention_5/attention_output/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_16/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_16/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_17/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_17/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0transformer_block_5/layer_normalization_10/gamma'variables/24/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_5/layer_normalization_10/beta'variables/25/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0transformer_block_5/layer_normalization_11/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_5/layer_normalization_11/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE

ÿ0
 

/0
01
82
93
~
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
 

s0

s0
 
µ
	variables
metrics
trainable_variables
layer_metrics
regularization_losses
non_trainable_variables
layers
 layer_regularization_losses

t0

t0
 
µ
	variables
metrics
trainable_variables
layer_metrics
regularization_losses
non_trainable_variables
layers
 layer_regularization_losses
 
 
 

0
1
 
 
 
 
 
 
 
 
 
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
/0
01
 
 
 
 

80
91
 
 
 
 
 
 
 

partial_output_shape
full_output_shape

ukernel
vbias
	variables
trainable_variables
regularization_losses
	keras_api

partial_output_shape
full_output_shape

wkernel
xbias
	variables
trainable_variables
regularization_losses
	keras_api

partial_output_shape
full_output_shape

ykernel
zbias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
V
 	variables
¡trainable_variables
¢regularization_losses
£	keras_api

¤partial_output_shape
¥full_output_shape

{kernel
|bias
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
8
u0
v1
w2
x3
y4
z5
{6
|7
8
u0
v1
w2
x3
y4
z5
{6
|7
 
µ
»	variables
ªmetrics
¼trainable_variables
«layer_metrics
½regularization_losses
¬non_trainable_variables
­layers
 ®layer_regularization_losses
l

}kernel
~bias
¯	variables
°trainable_variables
±regularization_losses
²	keras_api
m

kernel
	bias
³	variables
´trainable_variables
µregularization_losses
¶	keras_api

}0
~1
2
3

}0
~1
2
3
 
µ
Á	variables
·metrics
Âtrainable_variables
¸layer_metrics
Ãregularization_losses
¹non_trainable_variables
ºlayers
 »layer_regularization_losses
 

0
1

0
1
 
µ
Æ	variables
¼metrics
Çtrainable_variables
½layer_metrics
Èregularization_losses
¾non_trainable_variables
¿layers
 Àlayer_regularization_losses
 

0
1

0
1
 
µ
Ë	variables
Ámetrics
Ìtrainable_variables
Âlayer_metrics
Íregularization_losses
Ãnon_trainable_variables
Älayers
 Ålayer_regularization_losses
 
 
 
µ
Ï	variables
Æmetrics
Ðtrainable_variables
Çlayer_metrics
Ñregularization_losses
Ènon_trainable_variables
Élayers
 Êlayer_regularization_losses
 
 
 
µ
Ó	variables
Ëmetrics
Ôtrainable_variables
Ìlayer_metrics
Õregularization_losses
Ínon_trainable_variables
Îlayers
 Ïlayer_regularization_losses
 
 
 
*
B0
C1
D2
E3
F4
G5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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

Ðtotal

Ñcount
Ò	variables
Ó	keras_api
 
 
 
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
u0
v1

u0
v1
 
µ
	variables
Ômetrics
trainable_variables
Õlayer_metrics
regularization_losses
Önon_trainable_variables
×layers
 Ølayer_regularization_losses
 
 

w0
x1

w0
x1
 
µ
	variables
Ùmetrics
trainable_variables
Úlayer_metrics
regularization_losses
Ûnon_trainable_variables
Ülayers
 Ýlayer_regularization_losses
 
 

y0
z1

y0
z1
 
µ
	variables
Þmetrics
trainable_variables
ßlayer_metrics
regularization_losses
ànon_trainable_variables
álayers
 âlayer_regularization_losses
 
 
 
µ
	variables
ãmetrics
trainable_variables
älayer_metrics
regularization_losses
ånon_trainable_variables
ælayers
 çlayer_regularization_losses
 
 
 
µ
 	variables
èmetrics
¡trainable_variables
élayer_metrics
¢regularization_losses
ênon_trainable_variables
ëlayers
 ìlayer_regularization_losses
 
 

{0
|1

{0
|1
 
µ
¦	variables
ímetrics
§trainable_variables
îlayer_metrics
¨regularization_losses
ïnon_trainable_variables
ðlayers
 ñlayer_regularization_losses
 
 
 
0
µ0
¶1
·2
¸3
¹4
º5
 

}0
~1

}0
~1
 
µ
¯	variables
òmetrics
°trainable_variables
ólayer_metrics
±regularization_losses
ônon_trainable_variables
õlayers
 ölayer_regularization_losses

0
1

0
1
 
µ
³	variables
÷metrics
´trainable_variables
ølayer_metrics
µregularization_losses
ùnon_trainable_variables
úlayers
 ûlayer_regularization_losses
 
 
 

¿0
À1
 
 
 
 
 
 
 
 
 
 
 
 
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
Ð0
Ñ1

Ò	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
~
VARIABLE_VALUEAdamax/conv1d_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamax/conv1d_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adamax/batch_normalization_4/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adamax/batch_normalization_4/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adamax/batch_normalization_5/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adamax/batch_normalization_5/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdamax/dense_18/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamax/dense_18/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdamax/dense_19/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamax/dense_19/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdamax/dense_20/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamax/dense_20/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adamax/token_and_position_embedding_2/embedding_4/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adamax/token_and_position_embedding_2/embedding_5/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE@Adamax/transformer_block_5/multi_head_attention_5/query/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adamax/transformer_block_5/multi_head_attention_5/query/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adamax/transformer_block_5/multi_head_attention_5/key/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adamax/transformer_block_5/multi_head_attention_5/key/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE@Adamax/transformer_block_5/multi_head_attention_5/value/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adamax/transformer_block_5/multi_head_attention_5/value/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
¥¢
VARIABLE_VALUEKAdamax/transformer_block_5/multi_head_attention_5/attention_output/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEIAdamax/transformer_block_5/multi_head_attention_5/attention_output/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdamax/dense_16/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdamax/dense_16/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdamax/dense_17/kernel/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdamax/dense_17/bias/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE9Adamax/transformer_block_5/layer_normalization_10/gamma/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Adamax/transformer_block_5/layer_normalization_10/beta/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE9Adamax/transformer_block_5/layer_normalization_11/gamma/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Adamax/transformer_block_5/layer_normalization_11/beta/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdamax/conv1d_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamax/conv1d_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adamax/batch_normalization_4/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adamax/batch_normalization_4/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adamax/batch_normalization_5/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adamax/batch_normalization_5/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdamax/dense_18/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamax/dense_18/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdamax/dense_19/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamax/dense_19/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdamax/dense_20/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamax/dense_20/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adamax/token_and_position_embedding_2/embedding_4/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adamax/token_and_position_embedding_2/embedding_5/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE@Adamax/transformer_block_5/multi_head_attention_5/query/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adamax/transformer_block_5/multi_head_attention_5/query/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adamax/transformer_block_5/multi_head_attention_5/key/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<Adamax/transformer_block_5/multi_head_attention_5/key/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE@Adamax/transformer_block_5/multi_head_attention_5/value/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>Adamax/transformer_block_5/multi_head_attention_5/value/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
¥¢
VARIABLE_VALUEKAdamax/transformer_block_5/multi_head_attention_5/attention_output/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEIAdamax/transformer_block_5/multi_head_attention_5/attention_output/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdamax/dense_16/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdamax/dense_16/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdamax/dense_17/kernel/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdamax/dense_17/bias/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE9Adamax/transformer_block_5/layer_normalization_10/gamma/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Adamax/transformer_block_5/layer_normalization_10/beta/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE9Adamax/transformer_block_5/layer_normalization_11/gamma/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Adamax/transformer_block_5/layer_normalization_11/beta/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_5Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ¼i
z
serving_default_input_6Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Þ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5serving_default_input_65token_and_position_embedding_2/embedding_5/embeddings5token_and_position_embedding_2/embedding_4/embeddingsconv1d_2/kernelconv1d_2/bias%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/beta%batch_normalization_5/moving_variancebatch_normalization_5/gamma!batch_normalization_5/moving_meanbatch_normalization_5/beta7transformer_block_5/multi_head_attention_5/query/kernel5transformer_block_5/multi_head_attention_5/query/bias5transformer_block_5/multi_head_attention_5/key/kernel3transformer_block_5/multi_head_attention_5/key/bias7transformer_block_5/multi_head_attention_5/value/kernel5transformer_block_5/multi_head_attention_5/value/biasBtransformer_block_5/multi_head_attention_5/attention_output/kernel@transformer_block_5/multi_head_attention_5/attention_output/bias0transformer_block_5/layer_normalization_10/gamma/transformer_block_5/layer_normalization_10/betadense_16/kerneldense_16/biasdense_17/kerneldense_17/bias0transformer_block_5/layer_normalization_11/gamma/transformer_block_5/layer_normalization_11/betadense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/bias*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*D
_read_only_resource_inputs&
$"	
 !"#*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_49328
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ÿ1
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOp#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdamax/iter/Read/ReadVariableOpItoken_and_position_embedding_2/embedding_4/embeddings/Read/ReadVariableOpItoken_and_position_embedding_2/embedding_5/embeddings/Read/ReadVariableOpKtransformer_block_5/multi_head_attention_5/query/kernel/Read/ReadVariableOpItransformer_block_5/multi_head_attention_5/query/bias/Read/ReadVariableOpItransformer_block_5/multi_head_attention_5/key/kernel/Read/ReadVariableOpGtransformer_block_5/multi_head_attention_5/key/bias/Read/ReadVariableOpKtransformer_block_5/multi_head_attention_5/value/kernel/Read/ReadVariableOpItransformer_block_5/multi_head_attention_5/value/bias/Read/ReadVariableOpVtransformer_block_5/multi_head_attention_5/attention_output/kernel/Read/ReadVariableOpTtransformer_block_5/multi_head_attention_5/attention_output/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOpDtransformer_block_5/layer_normalization_10/gamma/Read/ReadVariableOpCtransformer_block_5/layer_normalization_10/beta/Read/ReadVariableOpDtransformer_block_5/layer_normalization_11/gamma/Read/ReadVariableOpCtransformer_block_5/layer_normalization_11/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,Adamax/conv1d_2/kernel/m/Read/ReadVariableOp*Adamax/conv1d_2/bias/m/Read/ReadVariableOp8Adamax/batch_normalization_4/gamma/m/Read/ReadVariableOp7Adamax/batch_normalization_4/beta/m/Read/ReadVariableOp8Adamax/batch_normalization_5/gamma/m/Read/ReadVariableOp7Adamax/batch_normalization_5/beta/m/Read/ReadVariableOp,Adamax/dense_18/kernel/m/Read/ReadVariableOp*Adamax/dense_18/bias/m/Read/ReadVariableOp,Adamax/dense_19/kernel/m/Read/ReadVariableOp*Adamax/dense_19/bias/m/Read/ReadVariableOp,Adamax/dense_20/kernel/m/Read/ReadVariableOp*Adamax/dense_20/bias/m/Read/ReadVariableOpRAdamax/token_and_position_embedding_2/embedding_4/embeddings/m/Read/ReadVariableOpRAdamax/token_and_position_embedding_2/embedding_5/embeddings/m/Read/ReadVariableOpTAdamax/transformer_block_5/multi_head_attention_5/query/kernel/m/Read/ReadVariableOpRAdamax/transformer_block_5/multi_head_attention_5/query/bias/m/Read/ReadVariableOpRAdamax/transformer_block_5/multi_head_attention_5/key/kernel/m/Read/ReadVariableOpPAdamax/transformer_block_5/multi_head_attention_5/key/bias/m/Read/ReadVariableOpTAdamax/transformer_block_5/multi_head_attention_5/value/kernel/m/Read/ReadVariableOpRAdamax/transformer_block_5/multi_head_attention_5/value/bias/m/Read/ReadVariableOp_Adamax/transformer_block_5/multi_head_attention_5/attention_output/kernel/m/Read/ReadVariableOp]Adamax/transformer_block_5/multi_head_attention_5/attention_output/bias/m/Read/ReadVariableOp,Adamax/dense_16/kernel/m/Read/ReadVariableOp*Adamax/dense_16/bias/m/Read/ReadVariableOp,Adamax/dense_17/kernel/m/Read/ReadVariableOp*Adamax/dense_17/bias/m/Read/ReadVariableOpMAdamax/transformer_block_5/layer_normalization_10/gamma/m/Read/ReadVariableOpLAdamax/transformer_block_5/layer_normalization_10/beta/m/Read/ReadVariableOpMAdamax/transformer_block_5/layer_normalization_11/gamma/m/Read/ReadVariableOpLAdamax/transformer_block_5/layer_normalization_11/beta/m/Read/ReadVariableOp,Adamax/conv1d_2/kernel/v/Read/ReadVariableOp*Adamax/conv1d_2/bias/v/Read/ReadVariableOp8Adamax/batch_normalization_4/gamma/v/Read/ReadVariableOp7Adamax/batch_normalization_4/beta/v/Read/ReadVariableOp8Adamax/batch_normalization_5/gamma/v/Read/ReadVariableOp7Adamax/batch_normalization_5/beta/v/Read/ReadVariableOp,Adamax/dense_18/kernel/v/Read/ReadVariableOp*Adamax/dense_18/bias/v/Read/ReadVariableOp,Adamax/dense_19/kernel/v/Read/ReadVariableOp*Adamax/dense_19/bias/v/Read/ReadVariableOp,Adamax/dense_20/kernel/v/Read/ReadVariableOp*Adamax/dense_20/bias/v/Read/ReadVariableOpRAdamax/token_and_position_embedding_2/embedding_4/embeddings/v/Read/ReadVariableOpRAdamax/token_and_position_embedding_2/embedding_5/embeddings/v/Read/ReadVariableOpTAdamax/transformer_block_5/multi_head_attention_5/query/kernel/v/Read/ReadVariableOpRAdamax/transformer_block_5/multi_head_attention_5/query/bias/v/Read/ReadVariableOpRAdamax/transformer_block_5/multi_head_attention_5/key/kernel/v/Read/ReadVariableOpPAdamax/transformer_block_5/multi_head_attention_5/key/bias/v/Read/ReadVariableOpTAdamax/transformer_block_5/multi_head_attention_5/value/kernel/v/Read/ReadVariableOpRAdamax/transformer_block_5/multi_head_attention_5/value/bias/v/Read/ReadVariableOp_Adamax/transformer_block_5/multi_head_attention_5/attention_output/kernel/v/Read/ReadVariableOp]Adamax/transformer_block_5/multi_head_attention_5/attention_output/bias/v/Read/ReadVariableOp,Adamax/dense_16/kernel/v/Read/ReadVariableOp*Adamax/dense_16/bias/v/Read/ReadVariableOp,Adamax/dense_17/kernel/v/Read/ReadVariableOp*Adamax/dense_17/bias/v/Read/ReadVariableOpMAdamax/transformer_block_5/layer_normalization_10/gamma/v/Read/ReadVariableOpLAdamax/transformer_block_5/layer_normalization_10/beta/v/Read/ReadVariableOpMAdamax/transformer_block_5/layer_normalization_11/gamma/v/Read/ReadVariableOpLAdamax/transformer_block_5/layer_normalization_11/beta/v/Read/ReadVariableOpConst*r
Tink
i2g	*
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
GPU2*0J 8 *'
f"R 
__inference__traced_save_51438
"
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_2/kernelconv1d_2/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancebatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variancedense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/biasbeta_1beta_2decaylearning_rateAdamax/iter5token_and_position_embedding_2/embedding_4/embeddings5token_and_position_embedding_2/embedding_5/embeddings7transformer_block_5/multi_head_attention_5/query/kernel5transformer_block_5/multi_head_attention_5/query/bias5transformer_block_5/multi_head_attention_5/key/kernel3transformer_block_5/multi_head_attention_5/key/bias7transformer_block_5/multi_head_attention_5/value/kernel5transformer_block_5/multi_head_attention_5/value/biasBtransformer_block_5/multi_head_attention_5/attention_output/kernel@transformer_block_5/multi_head_attention_5/attention_output/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias0transformer_block_5/layer_normalization_10/gamma/transformer_block_5/layer_normalization_10/beta0transformer_block_5/layer_normalization_11/gamma/transformer_block_5/layer_normalization_11/betatotalcountAdamax/conv1d_2/kernel/mAdamax/conv1d_2/bias/m$Adamax/batch_normalization_4/gamma/m#Adamax/batch_normalization_4/beta/m$Adamax/batch_normalization_5/gamma/m#Adamax/batch_normalization_5/beta/mAdamax/dense_18/kernel/mAdamax/dense_18/bias/mAdamax/dense_19/kernel/mAdamax/dense_19/bias/mAdamax/dense_20/kernel/mAdamax/dense_20/bias/m>Adamax/token_and_position_embedding_2/embedding_4/embeddings/m>Adamax/token_and_position_embedding_2/embedding_5/embeddings/m@Adamax/transformer_block_5/multi_head_attention_5/query/kernel/m>Adamax/transformer_block_5/multi_head_attention_5/query/bias/m>Adamax/transformer_block_5/multi_head_attention_5/key/kernel/m<Adamax/transformer_block_5/multi_head_attention_5/key/bias/m@Adamax/transformer_block_5/multi_head_attention_5/value/kernel/m>Adamax/transformer_block_5/multi_head_attention_5/value/bias/mKAdamax/transformer_block_5/multi_head_attention_5/attention_output/kernel/mIAdamax/transformer_block_5/multi_head_attention_5/attention_output/bias/mAdamax/dense_16/kernel/mAdamax/dense_16/bias/mAdamax/dense_17/kernel/mAdamax/dense_17/bias/m9Adamax/transformer_block_5/layer_normalization_10/gamma/m8Adamax/transformer_block_5/layer_normalization_10/beta/m9Adamax/transformer_block_5/layer_normalization_11/gamma/m8Adamax/transformer_block_5/layer_normalization_11/beta/mAdamax/conv1d_2/kernel/vAdamax/conv1d_2/bias/v$Adamax/batch_normalization_4/gamma/v#Adamax/batch_normalization_4/beta/v$Adamax/batch_normalization_5/gamma/v#Adamax/batch_normalization_5/beta/vAdamax/dense_18/kernel/vAdamax/dense_18/bias/vAdamax/dense_19/kernel/vAdamax/dense_19/bias/vAdamax/dense_20/kernel/vAdamax/dense_20/bias/v>Adamax/token_and_position_embedding_2/embedding_4/embeddings/v>Adamax/token_and_position_embedding_2/embedding_5/embeddings/v@Adamax/transformer_block_5/multi_head_attention_5/query/kernel/v>Adamax/transformer_block_5/multi_head_attention_5/query/bias/v>Adamax/transformer_block_5/multi_head_attention_5/key/kernel/v<Adamax/transformer_block_5/multi_head_attention_5/key/bias/v@Adamax/transformer_block_5/multi_head_attention_5/value/kernel/v>Adamax/transformer_block_5/multi_head_attention_5/value/bias/vKAdamax/transformer_block_5/multi_head_attention_5/attention_output/kernel/vIAdamax/transformer_block_5/multi_head_attention_5/attention_output/bias/vAdamax/dense_16/kernel/vAdamax/dense_16/bias/vAdamax/dense_17/kernel/vAdamax/dense_17/bias/v9Adamax/transformer_block_5/layer_normalization_10/gamma/v8Adamax/transformer_block_5/layer_normalization_10/beta/v9Adamax/transformer_block_5/layer_normalization_11/gamma/v8Adamax/transformer_block_5/layer_normalization_11/beta/v*q
Tinj
h2f*
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
GPU2*0J 8 **
f%R#
!__inference__traced_restore_51751Ö&
¯P
®
B__inference_model_2_layer_call_and_return_conditional_losses_48919
input_5
input_6(
$token_and_position_embedding_2_48835(
$token_and_position_embedding_2_48837
conv1d_2_48840
conv1d_2_48842
batch_normalization_4_48847
batch_normalization_4_48849
batch_normalization_4_48851
batch_normalization_4_48853
batch_normalization_5_48856
batch_normalization_5_48858
batch_normalization_5_48860
batch_normalization_5_48862
transformer_block_5_48866
transformer_block_5_48868
transformer_block_5_48870
transformer_block_5_48872
transformer_block_5_48874
transformer_block_5_48876
transformer_block_5_48878
transformer_block_5_48880
transformer_block_5_48882
transformer_block_5_48884
transformer_block_5_48886
transformer_block_5_48888
transformer_block_5_48890
transformer_block_5_48892
transformer_block_5_48894
transformer_block_5_48896
dense_18_48901
dense_18_48903
dense_19_48907
dense_19_48909
dense_20_48913
dense_20_48915
identity¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢ dense_20/StatefulPartitionedCall¢6token_and_position_embedding_2/StatefulPartitionedCall¢+transformer_block_5/StatefulPartitionedCall
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_5$token_and_position_embedding_2_48835$token_and_position_embedding_2_48837*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *b
f]R[
Y__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_4803128
6token_and_position_embedding_2/StatefulPartitionedCallÒ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0conv1d_2_48840conv1d_2_48842*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_480632"
 conv1d_2/StatefulPartitionedCallµ
#average_pooling1d_5/PartitionedCallPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_475302%
#average_pooling1d_5/PartitionedCall
#average_pooling1d_4/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_475152%
#average_pooling1d_4/PartitionedCall¾
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_4/PartitionedCall:output:0batch_normalization_4_48847batch_normalization_4_48849batch_normalization_4_48851batch_normalization_4_48853*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_481362/
-batch_normalization_4/StatefulPartitionedCall¾
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_5/PartitionedCall:output:0batch_normalization_5_48856batch_normalization_5_48858batch_normalization_5_48860batch_normalization_5_48862*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_482272/
-batch_normalization_5/StatefulPartitionedCall»
add_2/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:06batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_482692
add_2/PartitionedCallþ
+transformer_block_5/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0transformer_block_5_48866transformer_block_5_48868transformer_block_5_48870transformer_block_5_48872transformer_block_5_48874transformer_block_5_48876transformer_block_5_48878transformer_block_5_48880transformer_block_5_48882transformer_block_5_48884transformer_block_5_48886transformer_block_5_48888transformer_block_5_48890transformer_block_5_48892transformer_block_5_48894transformer_block_5_48896*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_transformer_block_5_layer_call_and_return_conditional_losses_485532-
+transformer_block_5/StatefulPartitionedCallº
*global_average_pooling1d_2/PartitionedCallPartitionedCall4transformer_block_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_486672,
*global_average_pooling1d_2/PartitionedCall
concatenate_2/PartitionedCallPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0input_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_486812
concatenate_2/PartitionedCall´
 dense_18/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_18_48901dense_18_48903*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_487012"
 dense_18/StatefulPartitionedCallÿ
dropout_16/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_487342
dropout_16/PartitionedCall±
 dense_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0dense_19_48907dense_19_48909*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_19_layer_call_and_return_conditional_losses_487582"
 dense_19/StatefulPartitionedCallÿ
dropout_17/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_487912
dropout_17/PartitionedCall±
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0dense_20_48913dense_20_48915*
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
GPU2*0J 8 *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_488142"
 dense_20/StatefulPartitionedCallÐ
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_5/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ¼i:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_5/StatefulPartitionedCall+transformer_block_5/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i
!
_user_specified_name	input_5:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6
Æ
#
B__inference_model_2_layer_call_and_return_conditional_losses_49849
inputs_0
inputs_1E
Atoken_and_position_embedding_2_embedding_5_embedding_lookup_49634E
Atoken_and_position_embedding_2_embedding_4_embedding_lookup_496408
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource?
;batch_normalization_4_batchnorm_mul_readvariableop_resource=
9batch_normalization_4_batchnorm_readvariableop_1_resource=
9batch_normalization_4_batchnorm_readvariableop_2_resource;
7batch_normalization_5_batchnorm_readvariableop_resource?
;batch_normalization_5_batchnorm_mul_readvariableop_resource=
9batch_normalization_5_batchnorm_readvariableop_1_resource=
9batch_normalization_5_batchnorm_readvariableop_2_resourceZ
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
Ltransformer_block_5_layer_normalization_11_batchnorm_readvariableop_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource
identity¢.batch_normalization_4/batchnorm/ReadVariableOp¢0batch_normalization_4/batchnorm/ReadVariableOp_1¢0batch_normalization_4/batchnorm/ReadVariableOp_2¢2batch_normalization_4/batchnorm/mul/ReadVariableOp¢.batch_normalization_5/batchnorm/ReadVariableOp¢0batch_normalization_5/batchnorm/ReadVariableOp_1¢0batch_normalization_5/batchnorm/ReadVariableOp_2¢2batch_normalization_5/batchnorm/mul/ReadVariableOp¢conv1d_2/BiasAdd/ReadVariableOp¢+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp¢dense_18/BiasAdd/ReadVariableOp¢dense_18/MatMul/ReadVariableOp¢dense_19/BiasAdd/ReadVariableOp¢dense_19/MatMul/ReadVariableOp¢dense_20/BiasAdd/ReadVariableOp¢dense_20/MatMul/ReadVariableOp¢;token_and_position_embedding_2/embedding_4/embedding_lookup¢;token_and_position_embedding_2/embedding_5/embedding_lookup¢Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp¢Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp¢Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp¢Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp¢Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp¢Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp¢Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp¢Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp¢Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOp¢Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp¢Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOp¢Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp¢@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp¢Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp¢@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp¢Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp
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
$token_and_position_embedding_2/rangeÈ
;token_and_position_embedding_2/embedding_5/embedding_lookupResourceGatherAtoken_and_position_embedding_2_embedding_5_embedding_lookup_49634-token_and_position_embedding_2/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/49634*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02=
;token_and_position_embedding_2/embedding_5/embedding_lookup
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/49634*'
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
:ÿÿÿÿÿÿÿÿÿ¼i21
/token_and_position_embedding_2/embedding_4/CastÓ
;token_and_position_embedding_2/embedding_4/embedding_lookupResourceGatherAtoken_and_position_embedding_2_embedding_4_embedding_lookup_496403token_and_position_embedding_2/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/49640*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *
dtype02=
;token_and_position_embedding_2/embedding_4/embedding_lookup
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/49640*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2F
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity¢
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2H
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1ª
"token_and_position_embedding_2/addAddV2Otoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2$
"token_and_position_embedding_2/add
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_2/conv1d/ExpandDims/dimÒ
conv1d_2/conv1d/ExpandDims
ExpandDims&token_and_position_embedding_2/add:z:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2
conv1d_2/conv1d/ExpandDimsÓ
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dimÛ
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_2/conv1d/ExpandDims_1Û
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *
paddingSAME*
strides
2
conv1d_2/conv1d®
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_2/conv1d/Squeeze§
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_2/BiasAdd/ReadVariableOp±
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2
conv1d_2/BiasAddx
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2
conv1d_2/Relu
"average_pooling1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_5/ExpandDims/dimÞ
average_pooling1d_5/ExpandDims
ExpandDims&token_and_position_embedding_2/add:z:0+average_pooling1d_5/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2 
average_pooling1d_5/ExpandDimså
average_pooling1d_5/AvgPoolAvgPool'average_pooling1d_5/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_5/AvgPool¹
average_pooling1d_5/SqueezeSqueeze$average_pooling1d_5/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
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
ExpandDimsconv1d_2/Relu:activations:0+average_pooling1d_4/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2 
average_pooling1d_4/ExpandDimså
average_pooling1d_4/AvgPoolAvgPool'average_pooling1d_4/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_4/AvgPool¹
average_pooling1d_4/SqueezeSqueeze$average_pooling1d_4/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
squeeze_dims
2
average_pooling1d_4/SqueezeÔ
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_4/batchnorm/ReadVariableOp
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_4/batchnorm/add/yà
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/add¥
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/Rsqrtà
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/mulÛ
%batch_normalization_4/batchnorm/mul_1Mul$average_pooling1d_4/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2'
%batch_normalization_4/batchnorm/mul_1Ú
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_1Ý
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/mul_2Ú
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_2Û
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/subâ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2'
%batch_normalization_4/batchnorm/add_1Ô
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_5/batchnorm/ReadVariableOp
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_5/batchnorm/add/yà
#batch_normalization_5/batchnorm/addAddV26batch_normalization_5/batchnorm/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_5/batchnorm/add¥
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_5/batchnorm/Rsqrtà
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_5/batchnorm/mulÛ
%batch_normalization_5/batchnorm/mul_1Mul$average_pooling1d_5/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2'
%batch_normalization_5/batchnorm/mul_1Ú
0batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_1Ý
%batch_normalization_5/batchnorm/mul_2Mul8batch_normalization_5/batchnorm/ReadVariableOp_1:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_5/batchnorm/mul_2Ú
0batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_2Û
#batch_normalization_5/batchnorm/subSub8batch_normalization_5/batchnorm/ReadVariableOp_2:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_5/batchnorm/subâ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2'
%batch_normalization_5/batchnorm/add_1¬
	add_2/addAddV2)batch_normalization_4/batchnorm/add_1:z:0)batch_normalization_5/batchnorm/add_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
	add_2/add¹
Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_5_multi_head_attention_5_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpÑ
>transformer_block_5/multi_head_attention_5/query/einsum/EinsumEinsumadd_2/add:z:0Utransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2@
>transformer_block_5/multi_head_attention_5/query/einsum/Einsum
Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOpReadVariableOpLtransformer_block_5_multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOpÆ
4transformer_block_5/multi_head_attention_5/query/addAddV2Gtransformer_block_5/multi_head_attention_5/query/einsum/Einsum:output:0Ktransformer_block_5/multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 26
4transformer_block_5/multi_head_attention_5/query/add³
Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_5_multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpË
<transformer_block_5/multi_head_attention_5/key/einsum/EinsumEinsumadd_2/add:z:0Stransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2>
<transformer_block_5/multi_head_attention_5/key/einsum/Einsum
Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOpReadVariableOpJtransformer_block_5_multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp¾
2transformer_block_5/multi_head_attention_5/key/addAddV2Etransformer_block_5/multi_head_attention_5/key/einsum/Einsum:output:0Itransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 24
2transformer_block_5/multi_head_attention_5/key/add¹
Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_5_multi_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpÑ
>transformer_block_5/multi_head_attention_5/value/einsum/EinsumEinsumadd_2/add:z:0Utransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2@
>transformer_block_5/multi_head_attention_5/value/einsum/Einsum
Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOpReadVariableOpLtransformer_block_5_multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOpÆ
4transformer_block_5/multi_head_attention_5/value/addAddV2Gtransformer_block_5/multi_head_attention_5/value/einsum/Einsum:output:0Ktransformer_block_5/multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 26
4transformer_block_5/multi_head_attention_5/value/add©
0transformer_block_5/multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>22
0transformer_block_5/multi_head_attention_5/Mul/y
.transformer_block_5/multi_head_attention_5/MulMul8transformer_block_5/multi_head_attention_5/query/add:z:09transformer_block_5/multi_head_attention_5/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 20
.transformer_block_5/multi_head_attention_5/MulÎ
8transformer_block_5/multi_head_attention_5/einsum/EinsumEinsum6transformer_block_5/multi_head_attention_5/key/add:z:02transformer_block_5/multi_head_attention_5/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ*
equationaecd,abcd->acbe2:
8transformer_block_5/multi_head_attention_5/einsum/Einsum
:transformer_block_5/multi_head_attention_5/softmax/SoftmaxSoftmaxAtransformer_block_5/multi_head_attention_5/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2<
:transformer_block_5/multi_head_attention_5/softmax/Softmax
;transformer_block_5/multi_head_attention_5/dropout/IdentityIdentityDtransformer_block_5/multi_head_attention_5/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2=
;transformer_block_5/multi_head_attention_5/dropout/Identityå
:transformer_block_5/multi_head_attention_5/einsum_1/EinsumEinsumDtransformer_block_5/multi_head_attention_5/dropout/Identity:output:08transformer_block_5/multi_head_attention_5/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationacbe,aecd->abcd2<
:transformer_block_5/multi_head_attention_5/einsum_1/EinsumÚ
Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_5_multi_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp¤
Itransformer_block_5/multi_head_attention_5/attention_output/einsum/EinsumEinsumCtransformer_block_5/multi_head_attention_5/einsum_1/Einsum:output:0`transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabcd,cde->abe2K
Itransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum´
Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_5_multi_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpî
?transformer_block_5/multi_head_attention_5/attention_output/addAddV2Rtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum:output:0Vtransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2A
?transformer_block_5/multi_head_attention_5/attention_output/addÚ
'transformer_block_5/dropout_14/IdentityIdentityCtransformer_block_5/multi_head_attention_5/attention_output/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2)
'transformer_block_5/dropout_14/Identity³
transformer_block_5/addAddV2add_2/add:z:00transformer_block_5/dropout_14/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
transformer_block_5/addà
Itransformer_block_5/layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_5/layer_normalization_10/moments/mean/reduction_indices³
7transformer_block_5/layer_normalization_10/moments/meanMeantransformer_block_5/add:z:0Rtransformer_block_5/layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(29
7transformer_block_5/layer_normalization_10/moments/mean
?transformer_block_5/layer_normalization_10/moments/StopGradientStopGradient@transformer_block_5/layer_normalization_10/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2A
?transformer_block_5/layer_normalization_10/moments/StopGradient¿
Dtransformer_block_5/layer_normalization_10/moments/SquaredDifferenceSquaredDifferencetransformer_block_5/add:z:0Htransformer_block_5/layer_normalization_10/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2F
Dtransformer_block_5/layer_normalization_10/moments/SquaredDifferenceè
Mtransformer_block_5/layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_5/layer_normalization_10/moments/variance/reduction_indicesì
;transformer_block_5/layer_normalization_10/moments/varianceMeanHtransformer_block_5/layer_normalization_10/moments/SquaredDifference:z:0Vtransformer_block_5/layer_normalization_10/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2=
;transformer_block_5/layer_normalization_10/moments/variance½
:transformer_block_5/layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_5/layer_normalization_10/batchnorm/add/y¿
8transformer_block_5/layer_normalization_10/batchnorm/addAddV2Dtransformer_block_5/layer_normalization_10/moments/variance:output:0Ctransformer_block_5/layer_normalization_10/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2:
8transformer_block_5/layer_normalization_10/batchnorm/addö
:transformer_block_5/layer_normalization_10/batchnorm/RsqrtRsqrt<transformer_block_5/layer_normalization_10/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2<
:transformer_block_5/layer_normalization_10/batchnorm/Rsqrt
Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_5_layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpÃ
8transformer_block_5/layer_normalization_10/batchnorm/mulMul>transformer_block_5/layer_normalization_10/batchnorm/Rsqrt:y:0Otransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2:
8transformer_block_5/layer_normalization_10/batchnorm/mul
:transformer_block_5/layer_normalization_10/batchnorm/mul_1Multransformer_block_5/add:z:0<transformer_block_5/layer_normalization_10/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2<
:transformer_block_5/layer_normalization_10/batchnorm/mul_1¶
:transformer_block_5/layer_normalization_10/batchnorm/mul_2Mul@transformer_block_5/layer_normalization_10/moments/mean:output:0<transformer_block_5/layer_normalization_10/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2<
:transformer_block_5/layer_normalization_10/batchnorm/mul_2
Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_5_layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp¿
8transformer_block_5/layer_normalization_10/batchnorm/subSubKtransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp:value:0>transformer_block_5/layer_normalization_10/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2:
8transformer_block_5/layer_normalization_10/batchnorm/sub¶
:transformer_block_5/layer_normalization_10/batchnorm/add_1AddV2>transformer_block_5/layer_normalization_10/batchnorm/mul_1:z:0<transformer_block_5/layer_normalization_10/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2<
:transformer_block_5/layer_normalization_10/batchnorm/add_1
Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_5_sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

:  *
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
9transformer_block_5/sequential_5/dense_16/Tensordot/stackÇ
=transformer_block_5/sequential_5/dense_16/Tensordot/transpose	Transpose>transformer_block_5/layer_normalization_10/batchnorm/add_1:z:0Ctransformer_block_5/sequential_5/dense_16/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2?
=transformer_block_5/sequential_5/dense_16/Tensordot/transposeÇ
;transformer_block_5/sequential_5/dense_16/Tensordot/ReshapeReshapeAtransformer_block_5/sequential_5/dense_16/Tensordot/transpose:y:0Btransformer_block_5/sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2=
;transformer_block_5/sequential_5/dense_16/Tensordot/ReshapeÆ
:transformer_block_5/sequential_5/dense_16/Tensordot/MatMulMatMulDtransformer_block_5/sequential_5/dense_16/Tensordot/Reshape:output:0Jtransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2<
:transformer_block_5/sequential_5/dense_16/Tensordot/MatMulÄ
;transformer_block_5/sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
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
<transformer_block_5/sequential_5/dense_16/Tensordot/concat_1¹
3transformer_block_5/sequential_5/dense_16/TensordotReshapeDtransformer_block_5/sequential_5/dense_16/Tensordot/MatMul:product:0Etransformer_block_5/sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 25
3transformer_block_5/sequential_5/dense_16/Tensordot
@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_5_sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp°
1transformer_block_5/sequential_5/dense_16/BiasAddBiasAdd<transformer_block_5/sequential_5/dense_16/Tensordot:output:0Htransformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 23
1transformer_block_5/sequential_5/dense_16/BiasAddÛ
.transformer_block_5/sequential_5/dense_16/ReluRelu:transformer_block_5/sequential_5/dense_16/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 20
.transformer_block_5/sequential_5/dense_16/Relu
Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_5_sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:  *
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
9transformer_block_5/sequential_5/dense_17/Tensordot/stackÅ
=transformer_block_5/sequential_5/dense_17/Tensordot/transpose	Transpose<transformer_block_5/sequential_5/dense_16/Relu:activations:0Ctransformer_block_5/sequential_5/dense_17/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2?
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
<transformer_block_5/sequential_5/dense_17/Tensordot/concat_1¹
3transformer_block_5/sequential_5/dense_17/TensordotReshapeDtransformer_block_5/sequential_5/dense_17/Tensordot/MatMul:product:0Etransformer_block_5/sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 25
3transformer_block_5/sequential_5/dense_17/Tensordot
@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_5_sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp°
1transformer_block_5/sequential_5/dense_17/BiasAddBiasAdd<transformer_block_5/sequential_5/dense_17/Tensordot:output:0Htransformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 23
1transformer_block_5/sequential_5/dense_17/BiasAddÑ
'transformer_block_5/dropout_15/IdentityIdentity:transformer_block_5/sequential_5/dense_17/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2)
'transformer_block_5/dropout_15/Identityè
transformer_block_5/add_1AddV2>transformer_block_5/layer_normalization_10/batchnorm/add_1:z:00transformer_block_5/dropout_15/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
transformer_block_5/add_1à
Itransformer_block_5/layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_5/layer_normalization_11/moments/mean/reduction_indicesµ
7transformer_block_5/layer_normalization_11/moments/meanMeantransformer_block_5/add_1:z:0Rtransformer_block_5/layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(29
7transformer_block_5/layer_normalization_11/moments/mean
?transformer_block_5/layer_normalization_11/moments/StopGradientStopGradient@transformer_block_5/layer_normalization_11/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2A
?transformer_block_5/layer_normalization_11/moments/StopGradientÁ
Dtransformer_block_5/layer_normalization_11/moments/SquaredDifferenceSquaredDifferencetransformer_block_5/add_1:z:0Htransformer_block_5/layer_normalization_11/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2F
Dtransformer_block_5/layer_normalization_11/moments/SquaredDifferenceè
Mtransformer_block_5/layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_5/layer_normalization_11/moments/variance/reduction_indicesì
;transformer_block_5/layer_normalization_11/moments/varianceMeanHtransformer_block_5/layer_normalization_11/moments/SquaredDifference:z:0Vtransformer_block_5/layer_normalization_11/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2=
;transformer_block_5/layer_normalization_11/moments/variance½
:transformer_block_5/layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_5/layer_normalization_11/batchnorm/add/y¿
8transformer_block_5/layer_normalization_11/batchnorm/addAddV2Dtransformer_block_5/layer_normalization_11/moments/variance:output:0Ctransformer_block_5/layer_normalization_11/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2:
8transformer_block_5/layer_normalization_11/batchnorm/addö
:transformer_block_5/layer_normalization_11/batchnorm/RsqrtRsqrt<transformer_block_5/layer_normalization_11/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2<
:transformer_block_5/layer_normalization_11/batchnorm/Rsqrt
Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_5_layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpÃ
8transformer_block_5/layer_normalization_11/batchnorm/mulMul>transformer_block_5/layer_normalization_11/batchnorm/Rsqrt:y:0Otransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2:
8transformer_block_5/layer_normalization_11/batchnorm/mul
:transformer_block_5/layer_normalization_11/batchnorm/mul_1Multransformer_block_5/add_1:z:0<transformer_block_5/layer_normalization_11/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2<
:transformer_block_5/layer_normalization_11/batchnorm/mul_1¶
:transformer_block_5/layer_normalization_11/batchnorm/mul_2Mul@transformer_block_5/layer_normalization_11/moments/mean:output:0<transformer_block_5/layer_normalization_11/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2<
:transformer_block_5/layer_normalization_11/batchnorm/mul_2
Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_5_layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp¿
8transformer_block_5/layer_normalization_11/batchnorm/subSubKtransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp:value:0>transformer_block_5/layer_normalization_11/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2:
8transformer_block_5/layer_normalization_11/batchnorm/sub¶
:transformer_block_5/layer_normalization_11/batchnorm/add_1AddV2>transformer_block_5/layer_normalization_11/batchnorm/mul_1:z:0<transformer_block_5/layer_normalization_11/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2<
:transformer_block_5/layer_normalization_11/batchnorm/add_1¨
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indicesø
global_average_pooling1d_2/MeanMean>transformer_block_5/layer_normalization_11/batchnorm/add_1:z:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
global_average_pooling1d_2/Meanx
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axisË
concatenate_2/concatConcatV2(global_average_pooling1d_2/Mean:output:0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
concatenate_2/concat¨
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:( *
dtype02 
dense_18/MatMul/ReadVariableOp¥
dense_18/MatMulMatMulconcatenate_2/concat:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_18/MatMul§
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_18/BiasAdd/ReadVariableOp¥
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_18/BiasAdds
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_18/Relu
dropout_16/IdentityIdentitydense_18/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_16/Identity¨
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_19/MatMul/ReadVariableOp¤
dense_19/MatMulMatMuldropout_16/Identity:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_19/MatMul§
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_19/BiasAdd/ReadVariableOp¥
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_19/BiasAdds
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_19/Relu
dropout_17/IdentityIdentitydense_19/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_17/Identity¨
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

: *
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
dense_20/BiasAdd´
IdentityIdentitydense_20/BiasAdd:output:0/^batch_normalization_4/batchnorm/ReadVariableOp1^batch_normalization_4/batchnorm/ReadVariableOp_11^batch_normalization_4/batchnorm/ReadVariableOp_23^batch_normalization_4/batchnorm/mul/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp1^batch_normalization_5/batchnorm/ReadVariableOp_11^batch_normalization_5/batchnorm/ReadVariableOp_23^batch_normalization_5/batchnorm/mul/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp<^token_and_position_embedding_2/embedding_4/embedding_lookup<^token_and_position_embedding_2/embedding_5/embedding_lookupD^transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpH^transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpD^transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpH^transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpO^transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpY^transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_5/multi_head_attention_5/key/add/ReadVariableOpL^transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpD^transformer_block_5/multi_head_attention_5/query/add/ReadVariableOpN^transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpD^transformer_block_5/multi_head_attention_5/value/add/ReadVariableOpN^transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpA^transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpC^transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpA^transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpC^transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ¼i:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2d
0batch_normalization_4/batchnorm/ReadVariableOp_10batch_normalization_4/batchnorm/ReadVariableOp_12d
0batch_normalization_4/batchnorm/ReadVariableOp_20batch_normalization_4/batchnorm/ReadVariableOp_22h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2`
.batch_normalization_5/batchnorm/ReadVariableOp.batch_normalization_5/batchnorm/ReadVariableOp2d
0batch_normalization_5/batchnorm/ReadVariableOp_10batch_normalization_5/batchnorm/ReadVariableOp_12d
0batch_normalization_5/batchnorm/ReadVariableOp_20batch_normalization_5/batchnorm/ReadVariableOp_22h
2batch_normalization_5/batchnorm/mul/ReadVariableOp2batch_normalization_5/batchnorm/mul/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2B
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
:ÿÿÿÿÿÿÿÿÿ¼i
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
è
ý
#__inference_signature_wrapper_49328
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

unknown_32
identity¢StatefulPartitionedCall
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
unknown_32*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*D
_read_only_resource_inputs&
$"	
 !"#*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_475062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ¼i:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i
!
_user_specified_name	input_5:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6
ì

P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_50111

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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
batchnorm/add_1à
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
¼
j
@__inference_add_2_layer_call_and_return_conditional_losses_48269

inputs
inputs_1
identity\
addAddV2inputsinputs_1*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÂ :ÿÿÿÿÿÿÿÿÿÂ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs:TP
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
Ê
¨
5__inference_batch_normalization_4_layer_call_fn_50124

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
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_481162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
³0
Å
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_50091

inputs
assignmovingavg_50066
assignmovingavg_1_50072)
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
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
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
moments/Squeeze_1Ë
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/50066*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_50066*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpð
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/50066*
_output_shapes
: 2
AssignMovingAvg/subç
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/50066*
_output_shapes
: 2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_50066AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/50066*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÑ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/50072*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_50072*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpú
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/50072*
_output_shapes
: 2
AssignMovingAvg_1/subñ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/50072*
_output_shapes
: 2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_50072AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/50072*
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
batchnorm/add_1¸
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
Á
V
:__inference_global_average_pooling1d_2_layer_call_fn_50755

inputs
identityÖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_486672
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÂ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs


'__inference_model_2_layer_call_fn_49082
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

unknown_32
identity¢StatefulPartitionedCall³
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
unknown_32*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*@
_read_only_resource_inputs"
 	 !"#*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_490112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ¼i:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i
!
_user_specified_name	input_5:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6


>__inference_token_and_position_embedding_2_layer_call_fn_50030
x
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *b
f]R[
Y__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_480312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¼i::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i

_user_specified_namex

O
3__inference_average_pooling1d_4_layer_call_fn_47521

inputs
identityå
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
GPU2*0J 8 *W
fRRP
N__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_475152
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
°
Q
%__inference_add_2_layer_call_fn_50395
inputs_0
inputs_1
identityÓ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_482692
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÂ :ÿÿÿÿÿÿÿÿÿÂ :V R
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
"
_user_specified_name
inputs/1
ì
¨
5__inference_batch_normalization_5_layer_call_fn_50301

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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_478052
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

ß
3__inference_transformer_block_5_layer_call_fn_50707

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
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_transformer_block_5_layer_call_and_return_conditional_losses_484262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:ÿÿÿÿÿÿÿÿÿÂ ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
¸S
ø
B__inference_model_2_layer_call_and_return_conditional_losses_48831
input_5
input_6(
$token_and_position_embedding_2_48042(
$token_and_position_embedding_2_48044
conv1d_2_48074
conv1d_2_48076
batch_normalization_4_48163
batch_normalization_4_48165
batch_normalization_4_48167
batch_normalization_4_48169
batch_normalization_5_48254
batch_normalization_5_48256
batch_normalization_5_48258
batch_normalization_5_48260
transformer_block_5_48629
transformer_block_5_48631
transformer_block_5_48633
transformer_block_5_48635
transformer_block_5_48637
transformer_block_5_48639
transformer_block_5_48641
transformer_block_5_48643
transformer_block_5_48645
transformer_block_5_48647
transformer_block_5_48649
transformer_block_5_48651
transformer_block_5_48653
transformer_block_5_48655
transformer_block_5_48657
transformer_block_5_48659
dense_18_48712
dense_18_48714
dense_19_48769
dense_19_48771
dense_20_48825
dense_20_48827
identity¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢ dense_20/StatefulPartitionedCall¢"dropout_16/StatefulPartitionedCall¢"dropout_17/StatefulPartitionedCall¢6token_and_position_embedding_2/StatefulPartitionedCall¢+transformer_block_5/StatefulPartitionedCall
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_5$token_and_position_embedding_2_48042$token_and_position_embedding_2_48044*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *b
f]R[
Y__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_4803128
6token_and_position_embedding_2/StatefulPartitionedCallÒ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0conv1d_2_48074conv1d_2_48076*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_480632"
 conv1d_2/StatefulPartitionedCallµ
#average_pooling1d_5/PartitionedCallPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_475302%
#average_pooling1d_5/PartitionedCall
#average_pooling1d_4/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_475152%
#average_pooling1d_4/PartitionedCall¼
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_4/PartitionedCall:output:0batch_normalization_4_48163batch_normalization_4_48165batch_normalization_4_48167batch_normalization_4_48169*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_481162/
-batch_normalization_4/StatefulPartitionedCall¼
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_5/PartitionedCall:output:0batch_normalization_5_48254batch_normalization_5_48256batch_normalization_5_48258batch_normalization_5_48260*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_482072/
-batch_normalization_5/StatefulPartitionedCall»
add_2/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:06batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_482692
add_2/PartitionedCallþ
+transformer_block_5/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0transformer_block_5_48629transformer_block_5_48631transformer_block_5_48633transformer_block_5_48635transformer_block_5_48637transformer_block_5_48639transformer_block_5_48641transformer_block_5_48643transformer_block_5_48645transformer_block_5_48647transformer_block_5_48649transformer_block_5_48651transformer_block_5_48653transformer_block_5_48655transformer_block_5_48657transformer_block_5_48659*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_transformer_block_5_layer_call_and_return_conditional_losses_484262-
+transformer_block_5/StatefulPartitionedCallº
*global_average_pooling1d_2/PartitionedCallPartitionedCall4transformer_block_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_486672,
*global_average_pooling1d_2/PartitionedCall
concatenate_2/PartitionedCallPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0input_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_486812
concatenate_2/PartitionedCall´
 dense_18/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_18_48712dense_18_48714*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_487012"
 dense_18/StatefulPartitionedCall
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_487292$
"dropout_16/StatefulPartitionedCall¹
 dense_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0dense_19_48769dense_19_48771*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_19_layer_call_and_return_conditional_losses_487582"
 dense_19/StatefulPartitionedCall¼
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_487862$
"dropout_17/StatefulPartitionedCall¹
 dense_20/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_20_48825dense_20_48827*
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
GPU2*0J 8 *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_488142"
 dense_20/StatefulPartitionedCall
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_5/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ¼i:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_5/StatefulPartitionedCall+transformer_block_5/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i
!
_user_specified_name	input_5:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6
Ý
}
(__inference_dense_18_layer_call_fn_50799

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_487012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
· 
â
C__inference_dense_16_layer_call_and_return_conditional_losses_51063

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
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
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
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
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÂ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
ì

P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_48136

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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
batchnorm/add_1à
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
º

,__inference_sequential_5_layer_call_fn_51032

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
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_479722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs

á
N__inference_transformer_block_5_layer_call_and_return_conditional_losses_50543

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
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_5/query/einsum/EinsumEinsuminputsAmulti_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2,
*multi_head_attention_5/query/einsum/EinsumÛ
/multi_head_attention_5/query/add/ReadVariableOpReadVariableOp8multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/query/add/ReadVariableOpö
 multi_head_attention_5/query/addAddV23multi_head_attention_5/query/einsum/Einsum:output:07multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2"
 multi_head_attention_5/query/add÷
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_5/key/einsum/EinsumEinsuminputs?multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2*
(multi_head_attention_5/key/einsum/EinsumÕ
-multi_head_attention_5/key/add/ReadVariableOpReadVariableOp6multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_5/key/add/ReadVariableOpî
multi_head_attention_5/key/addAddV21multi_head_attention_5/key/einsum/Einsum:output:05multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2 
multi_head_attention_5/key/addý
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_5/value/einsum/EinsumEinsuminputsAmulti_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2,
*multi_head_attention_5/value/einsum/EinsumÛ
/multi_head_attention_5/value/add/ReadVariableOpReadVariableOp8multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/value/add/ReadVariableOpö
 multi_head_attention_5/value/addAddV23multi_head_attention_5/value/einsum/Einsum:output:07multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2"
 multi_head_attention_5/value/add
multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_5/Mul/yÇ
multi_head_attention_5/MulMul$multi_head_attention_5/query/add:z:0%multi_head_attention_5/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
multi_head_attention_5/Mulþ
$multi_head_attention_5/einsum/EinsumEinsum"multi_head_attention_5/key/add:z:0multi_head_attention_5/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ*
equationaecd,abcd->acbe2&
$multi_head_attention_5/einsum/EinsumÆ
&multi_head_attention_5/softmax/SoftmaxSoftmax-multi_head_attention_5/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2(
&multi_head_attention_5/softmax/Softmax¡
,multi_head_attention_5/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,multi_head_attention_5/dropout/dropout/Const
*multi_head_attention_5/dropout/dropout/MulMul0multi_head_attention_5/softmax/Softmax:softmax:05multi_head_attention_5/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2,
*multi_head_attention_5/dropout/dropout/Mul¼
,multi_head_attention_5/dropout/dropout/ShapeShape0multi_head_attention_5/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_5/dropout/dropout/Shape§
Cmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_5/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ*
dtype0*

seed*2E
Cmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniform³
5multi_head_attention_5/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_5/dropout/dropout/GreaterEqual/yÄ
3multi_head_attention_5/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_5/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ25
3multi_head_attention_5/dropout/dropout/GreaterEqualæ
+multi_head_attention_5/dropout/dropout/CastCast7multi_head_attention_5/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2-
+multi_head_attention_5/dropout/dropout/Cast
,multi_head_attention_5/dropout/dropout/Mul_1Mul.multi_head_attention_5/dropout/dropout/Mul:z:0/multi_head_attention_5/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2.
,multi_head_attention_5/dropout/dropout/Mul_1
&multi_head_attention_5/einsum_1/EinsumEinsum0multi_head_attention_5/dropout/dropout/Mul_1:z:0$multi_head_attention_5/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationacbe,aecd->abcd2(
&multi_head_attention_5/einsum_1/Einsum
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpÔ
5multi_head_attention_5/attention_output/einsum/EinsumEinsum/multi_head_attention_5/einsum_1/Einsum:output:0Lmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabcd,cde->abe27
5multi_head_attention_5/attention_output/einsum/Einsumø
:multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_5/attention_output/add/ReadVariableOp
+multi_head_attention_5/attention_output/addAddV2>multi_head_attention_5/attention_output/einsum/Einsum:output:0Bmulti_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2-
+multi_head_attention_5/attention_output/addy
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_14/dropout/ConstÂ
dropout_14/dropout/MulMul/multi_head_attention_5/attention_output/add:z:0!dropout_14/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dropout_14/dropout/Mul
dropout_14/dropout/ShapeShape/multi_head_attention_5/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_14/dropout/Shapeó
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
dtype0*

seed**
seed221
/dropout_14/dropout/random_uniform/RandomUniform
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_14/dropout/GreaterEqual/yï
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2!
dropout_14/dropout/GreaterEqual¥
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dropout_14/dropout/Cast«
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dropout_14/dropout/Mul_1p
addAddV2inputsdropout_14/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
add¸
5layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_10/moments/mean/reduction_indicesã
#layer_normalization_10/moments/meanMeanadd:z:0>layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2%
#layer_normalization_10/moments/meanÏ
+layer_normalization_10/moments/StopGradientStopGradient,layer_normalization_10/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2-
+layer_normalization_10/moments/StopGradientï
0layer_normalization_10/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_10/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 22
0layer_normalization_10/moments/SquaredDifferenceÀ
9layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_10/moments/variance/reduction_indices
'layer_normalization_10/moments/varianceMean4layer_normalization_10/moments/SquaredDifference:z:0Blayer_normalization_10/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2)
'layer_normalization_10/moments/variance
&layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_10/batchnorm/add/yï
$layer_normalization_10/batchnorm/addAddV20layer_normalization_10/moments/variance:output:0/layer_normalization_10/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2&
$layer_normalization_10/batchnorm/addº
&layer_normalization_10/batchnorm/RsqrtRsqrt(layer_normalization_10/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2(
&layer_normalization_10/batchnorm/Rsqrtã
3layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_10/batchnorm/mul/ReadVariableOpó
$layer_normalization_10/batchnorm/mulMul*layer_normalization_10/batchnorm/Rsqrt:y:0;layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2&
$layer_normalization_10/batchnorm/mulÁ
&layer_normalization_10/batchnorm/mul_1Muladd:z:0(layer_normalization_10/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_10/batchnorm/mul_1æ
&layer_normalization_10/batchnorm/mul_2Mul,layer_normalization_10/moments/mean:output:0(layer_normalization_10/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_10/batchnorm/mul_2×
/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_10/batchnorm/ReadVariableOpï
$layer_normalization_10/batchnorm/subSub7layer_normalization_10/batchnorm/ReadVariableOp:value:0*layer_normalization_10/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2&
$layer_normalization_10/batchnorm/subæ
&layer_normalization_10/batchnorm/add_1AddV2*layer_normalization_10/batchnorm/mul_1:z:0(layer_normalization_10/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_10/batchnorm/add_1Ø
.sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

:  *
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
%sequential_5/dense_16/Tensordot/stack÷
)sequential_5/dense_16/Tensordot/transpose	Transpose*layer_normalization_10/batchnorm/add_1:z:0/sequential_5/dense_16/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2+
)sequential_5/dense_16/Tensordot/transpose÷
'sequential_5/dense_16/Tensordot/ReshapeReshape-sequential_5/dense_16/Tensordot/transpose:y:0.sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_5/dense_16/Tensordot/Reshapeö
&sequential_5/dense_16/Tensordot/MatMulMatMul0sequential_5/dense_16/Tensordot/Reshape:output:06sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_5/dense_16/Tensordot/MatMul
'sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
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
(sequential_5/dense_16/Tensordot/concat_1é
sequential_5/dense_16/TensordotReshape0sequential_5/dense_16/Tensordot/MatMul:product:01sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2!
sequential_5/dense_16/TensordotÎ
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOpà
sequential_5/dense_16/BiasAddBiasAdd(sequential_5/dense_16/Tensordot:output:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
sequential_5/dense_16/BiasAdd
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
sequential_5/dense_16/ReluØ
.sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:  *
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
%sequential_5/dense_17/Tensordot/stackõ
)sequential_5/dense_17/Tensordot/transpose	Transpose(sequential_5/dense_16/Relu:activations:0/sequential_5/dense_17/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2+
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
(sequential_5/dense_17/Tensordot/concat_1é
sequential_5/dense_17/TensordotReshape0sequential_5/dense_17/Tensordot/MatMul:product:01sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2!
sequential_5/dense_17/TensordotÎ
,sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_17/BiasAdd/ReadVariableOpà
sequential_5/dense_17/BiasAddBiasAdd(sequential_5/dense_17/Tensordot:output:04sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
sequential_5/dense_17/BiasAddy
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_15/dropout/Const¹
dropout_15/dropout/MulMul&sequential_5/dense_17/BiasAdd:output:0!dropout_15/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dropout_15/dropout/Mul
dropout_15/dropout/ShapeShape&sequential_5/dense_17/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_15/dropout/Shapeó
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
dtype0*

seed**
seed221
/dropout_15/dropout/random_uniform/RandomUniform
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_15/dropout/GreaterEqual/yï
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2!
dropout_15/dropout/GreaterEqual¥
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dropout_15/dropout/Cast«
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dropout_15/dropout/Mul_1
add_1AddV2*layer_normalization_10/batchnorm/add_1:z:0dropout_15/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
add_1¸
5layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_11/moments/mean/reduction_indiceså
#layer_normalization_11/moments/meanMean	add_1:z:0>layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2%
#layer_normalization_11/moments/meanÏ
+layer_normalization_11/moments/StopGradientStopGradient,layer_normalization_11/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2-
+layer_normalization_11/moments/StopGradientñ
0layer_normalization_11/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_11/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 22
0layer_normalization_11/moments/SquaredDifferenceÀ
9layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_11/moments/variance/reduction_indices
'layer_normalization_11/moments/varianceMean4layer_normalization_11/moments/SquaredDifference:z:0Blayer_normalization_11/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2)
'layer_normalization_11/moments/variance
&layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_11/batchnorm/add/yï
$layer_normalization_11/batchnorm/addAddV20layer_normalization_11/moments/variance:output:0/layer_normalization_11/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2&
$layer_normalization_11/batchnorm/addº
&layer_normalization_11/batchnorm/RsqrtRsqrt(layer_normalization_11/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2(
&layer_normalization_11/batchnorm/Rsqrtã
3layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_11/batchnorm/mul/ReadVariableOpó
$layer_normalization_11/batchnorm/mulMul*layer_normalization_11/batchnorm/Rsqrt:y:0;layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2&
$layer_normalization_11/batchnorm/mulÃ
&layer_normalization_11/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_11/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_11/batchnorm/mul_1æ
&layer_normalization_11/batchnorm/mul_2Mul,layer_normalization_11/moments/mean:output:0(layer_normalization_11/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_11/batchnorm/mul_2×
/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_11/batchnorm/ReadVariableOpï
$layer_normalization_11/batchnorm/subSub7layer_normalization_11/batchnorm/ReadVariableOp:value:0*layer_normalization_11/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2&
$layer_normalization_11/batchnorm/subæ
&layer_normalization_11/batchnorm/add_1AddV2*layer_normalization_11/batchnorm/mul_1:z:0(layer_normalization_11/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_11/batchnorm/add_1Ý
IdentityIdentity*layer_normalization_11/batchnorm/add_1:z:00^layer_normalization_10/batchnorm/ReadVariableOp4^layer_normalization_10/batchnorm/mul/ReadVariableOp0^layer_normalization_11/batchnorm/ReadVariableOp4^layer_normalization_11/batchnorm/mul/ReadVariableOp;^multi_head_attention_5/attention_output/add/ReadVariableOpE^multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_5/key/add/ReadVariableOp8^multi_head_attention_5/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/query/add/ReadVariableOp:^multi_head_attention_5/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/value/add/ReadVariableOp:^multi_head_attention_5/value/einsum/Einsum/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp/^sequential_5/dense_16/Tensordot/ReadVariableOp-^sequential_5/dense_17/BiasAdd/ReadVariableOp/^sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:ÿÿÿÿÿÿÿÿÿÂ ::::::::::::::::2b
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
.sequential_5/dense_17/Tensordot/ReadVariableOp.sequential_5/dense_17/Tensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
Ò
§
,__inference_sequential_5_layer_call_fn_47956
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
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_479452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
(
_user_specified_namedense_16_input

d
E__inference_dropout_17_layer_call_and_return_conditional_losses_48786

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
:ÿÿÿÿÿÿÿÿÿ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÀ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
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
:ÿÿÿÿÿÿÿÿÿ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

ö
C__inference_conv1d_2_layer_call_and_return_conditional_losses_50046

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
:ÿÿÿÿÿÿÿÿÿ¼i 2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
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
:  2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *
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
:ÿÿÿÿÿÿÿÿÿ¼i 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ¼i ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 
 
_user_specified_nameinputs


'__inference_model_2_layer_call_fn_49997
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

unknown_32
identity¢StatefulPartitionedCall¹
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
unknown_32*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*D
_read_only_resource_inputs&
$"	
 !"#*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_491732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ¼i:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ö
â
C__inference_dense_17_layer_call_and_return_conditional_losses_51102

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
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
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
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
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÂ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs

d
E__inference_dropout_16_layer_call_and_return_conditional_losses_48729

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
:ÿÿÿÿÿÿÿÿÿ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÀ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
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
:ÿÿÿÿÿÿÿÿÿ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ê
¨
5__inference_batch_normalization_5_layer_call_fn_50288

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¨
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
GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_477722
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
Ì
¨
5__inference_batch_normalization_5_layer_call_fn_50383

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
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_482272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
í	
Ü
C__inference_dense_19_layer_call_and_return_conditional_losses_50837

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
J
®
G__inference_sequential_5_layer_call_and_return_conditional_losses_50949

inputs.
*dense_16_tensordot_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource.
*dense_17_tensordot_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
identity¢dense_16/BiasAdd/ReadVariableOp¢!dense_16/Tensordot/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢!dense_17/Tensordot/ReadVariableOp±
!dense_16/Tensordot/ReadVariableOpReadVariableOp*dense_16_tensordot_readvariableop_resource*
_output_shapes

:  *
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
dense_16/Tensordot/stack¬
dense_16/Tensordot/transpose	Transposeinputs"dense_16/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dense_16/Tensordot/transposeÃ
dense_16/Tensordot/ReshapeReshape dense_16/Tensordot/transpose:y:0!dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_16/Tensordot/ReshapeÂ
dense_16/Tensordot/MatMulMatMul#dense_16/Tensordot/Reshape:output:0)dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_16/Tensordot/MatMul
dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
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
dense_16/Tensordot/concat_1µ
dense_16/TensordotReshape#dense_16/Tensordot/MatMul:product:0$dense_16/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dense_16/Tensordot§
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_16/BiasAdd/ReadVariableOp¬
dense_16/BiasAddBiasAdddense_16/Tensordot:output:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dense_16/BiasAddx
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dense_16/Relu±
!dense_17/Tensordot/ReadVariableOpReadVariableOp*dense_17_tensordot_readvariableop_resource*
_output_shapes

:  *
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
dense_17/Tensordot/stackÁ
dense_17/Tensordot/transpose	Transposedense_16/Relu:activations:0"dense_17/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
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
dense_17/Tensordot/concat_1µ
dense_17/TensordotReshape#dense_17/Tensordot/MatMul:product:0$dense_17/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dense_17/Tensordot§
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_17/BiasAdd/ReadVariableOp¬
dense_17/BiasAddBiasAdddense_17/Tensordot:output:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dense_17/BiasAddþ
IdentityIdentitydense_17/BiasAdd:output:0 ^dense_16/BiasAdd/ReadVariableOp"^dense_16/Tensordot/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp"^dense_17/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2F
!dense_16/Tensordot/ReadVariableOp!dense_16/Tensordot/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2F
!dense_17/Tensordot/ReadVariableOp!dense_17/Tensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
Ý
q
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_48667

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
:ÿÿÿÿÿÿÿÿÿ 2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÂ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs


'__inference_model_2_layer_call_fn_49244
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

unknown_32
identity¢StatefulPartitionedCall·
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
unknown_32*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*D
_read_only_resource_inputs&
$"	
 !"#*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_491732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ¼i:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i
!
_user_specified_name	input_5:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6


P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_50193

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
ä0
Å
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_47632

inputs
assignmovingavg_47607
assignmovingavg_1_47613)
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
moments/Squeeze_1Ë
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/47607*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_47607*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpð
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/47607*
_output_shapes
: 2
AssignMovingAvg/subç
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/47607*
_output_shapes
: 2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_47607AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/47607*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÑ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/47613*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_47613*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpú
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/47613*
_output_shapes
: 2
AssignMovingAvg_1/subñ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/47613*
_output_shapes
: 2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_47613AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/47613*
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
ìÞ
á
N__inference_transformer_block_5_layer_call_and_return_conditional_losses_50670

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
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_5/query/einsum/EinsumEinsuminputsAmulti_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2,
*multi_head_attention_5/query/einsum/EinsumÛ
/multi_head_attention_5/query/add/ReadVariableOpReadVariableOp8multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/query/add/ReadVariableOpö
 multi_head_attention_5/query/addAddV23multi_head_attention_5/query/einsum/Einsum:output:07multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2"
 multi_head_attention_5/query/add÷
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_5/key/einsum/EinsumEinsuminputs?multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2*
(multi_head_attention_5/key/einsum/EinsumÕ
-multi_head_attention_5/key/add/ReadVariableOpReadVariableOp6multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_5/key/add/ReadVariableOpî
multi_head_attention_5/key/addAddV21multi_head_attention_5/key/einsum/Einsum:output:05multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2 
multi_head_attention_5/key/addý
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_5/value/einsum/EinsumEinsuminputsAmulti_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2,
*multi_head_attention_5/value/einsum/EinsumÛ
/multi_head_attention_5/value/add/ReadVariableOpReadVariableOp8multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/value/add/ReadVariableOpö
 multi_head_attention_5/value/addAddV23multi_head_attention_5/value/einsum/Einsum:output:07multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2"
 multi_head_attention_5/value/add
multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_5/Mul/yÇ
multi_head_attention_5/MulMul$multi_head_attention_5/query/add:z:0%multi_head_attention_5/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
multi_head_attention_5/Mulþ
$multi_head_attention_5/einsum/EinsumEinsum"multi_head_attention_5/key/add:z:0multi_head_attention_5/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ*
equationaecd,abcd->acbe2&
$multi_head_attention_5/einsum/EinsumÆ
&multi_head_attention_5/softmax/SoftmaxSoftmax-multi_head_attention_5/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2(
&multi_head_attention_5/softmax/SoftmaxÌ
'multi_head_attention_5/dropout/IdentityIdentity0multi_head_attention_5/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2)
'multi_head_attention_5/dropout/Identity
&multi_head_attention_5/einsum_1/EinsumEinsum0multi_head_attention_5/dropout/Identity:output:0$multi_head_attention_5/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationacbe,aecd->abcd2(
&multi_head_attention_5/einsum_1/Einsum
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpÔ
5multi_head_attention_5/attention_output/einsum/EinsumEinsum/multi_head_attention_5/einsum_1/Einsum:output:0Lmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabcd,cde->abe27
5multi_head_attention_5/attention_output/einsum/Einsumø
:multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_5/attention_output/add/ReadVariableOp
+multi_head_attention_5/attention_output/addAddV2>multi_head_attention_5/attention_output/einsum/Einsum:output:0Bmulti_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2-
+multi_head_attention_5/attention_output/add
dropout_14/IdentityIdentity/multi_head_attention_5/attention_output/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dropout_14/Identityp
addAddV2inputsdropout_14/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
add¸
5layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_10/moments/mean/reduction_indicesã
#layer_normalization_10/moments/meanMeanadd:z:0>layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2%
#layer_normalization_10/moments/meanÏ
+layer_normalization_10/moments/StopGradientStopGradient,layer_normalization_10/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2-
+layer_normalization_10/moments/StopGradientï
0layer_normalization_10/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_10/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 22
0layer_normalization_10/moments/SquaredDifferenceÀ
9layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_10/moments/variance/reduction_indices
'layer_normalization_10/moments/varianceMean4layer_normalization_10/moments/SquaredDifference:z:0Blayer_normalization_10/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2)
'layer_normalization_10/moments/variance
&layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_10/batchnorm/add/yï
$layer_normalization_10/batchnorm/addAddV20layer_normalization_10/moments/variance:output:0/layer_normalization_10/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2&
$layer_normalization_10/batchnorm/addº
&layer_normalization_10/batchnorm/RsqrtRsqrt(layer_normalization_10/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2(
&layer_normalization_10/batchnorm/Rsqrtã
3layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_10/batchnorm/mul/ReadVariableOpó
$layer_normalization_10/batchnorm/mulMul*layer_normalization_10/batchnorm/Rsqrt:y:0;layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2&
$layer_normalization_10/batchnorm/mulÁ
&layer_normalization_10/batchnorm/mul_1Muladd:z:0(layer_normalization_10/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_10/batchnorm/mul_1æ
&layer_normalization_10/batchnorm/mul_2Mul,layer_normalization_10/moments/mean:output:0(layer_normalization_10/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_10/batchnorm/mul_2×
/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_10/batchnorm/ReadVariableOpï
$layer_normalization_10/batchnorm/subSub7layer_normalization_10/batchnorm/ReadVariableOp:value:0*layer_normalization_10/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2&
$layer_normalization_10/batchnorm/subæ
&layer_normalization_10/batchnorm/add_1AddV2*layer_normalization_10/batchnorm/mul_1:z:0(layer_normalization_10/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_10/batchnorm/add_1Ø
.sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

:  *
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
%sequential_5/dense_16/Tensordot/stack÷
)sequential_5/dense_16/Tensordot/transpose	Transpose*layer_normalization_10/batchnorm/add_1:z:0/sequential_5/dense_16/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2+
)sequential_5/dense_16/Tensordot/transpose÷
'sequential_5/dense_16/Tensordot/ReshapeReshape-sequential_5/dense_16/Tensordot/transpose:y:0.sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_5/dense_16/Tensordot/Reshapeö
&sequential_5/dense_16/Tensordot/MatMulMatMul0sequential_5/dense_16/Tensordot/Reshape:output:06sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_5/dense_16/Tensordot/MatMul
'sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
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
(sequential_5/dense_16/Tensordot/concat_1é
sequential_5/dense_16/TensordotReshape0sequential_5/dense_16/Tensordot/MatMul:product:01sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2!
sequential_5/dense_16/TensordotÎ
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOpà
sequential_5/dense_16/BiasAddBiasAdd(sequential_5/dense_16/Tensordot:output:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
sequential_5/dense_16/BiasAdd
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
sequential_5/dense_16/ReluØ
.sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:  *
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
%sequential_5/dense_17/Tensordot/stackõ
)sequential_5/dense_17/Tensordot/transpose	Transpose(sequential_5/dense_16/Relu:activations:0/sequential_5/dense_17/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2+
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
(sequential_5/dense_17/Tensordot/concat_1é
sequential_5/dense_17/TensordotReshape0sequential_5/dense_17/Tensordot/MatMul:product:01sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2!
sequential_5/dense_17/TensordotÎ
,sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_17/BiasAdd/ReadVariableOpà
sequential_5/dense_17/BiasAddBiasAdd(sequential_5/dense_17/Tensordot:output:04sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
sequential_5/dense_17/BiasAdd
dropout_15/IdentityIdentity&sequential_5/dense_17/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dropout_15/Identity
add_1AddV2*layer_normalization_10/batchnorm/add_1:z:0dropout_15/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
add_1¸
5layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_11/moments/mean/reduction_indiceså
#layer_normalization_11/moments/meanMean	add_1:z:0>layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2%
#layer_normalization_11/moments/meanÏ
+layer_normalization_11/moments/StopGradientStopGradient,layer_normalization_11/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2-
+layer_normalization_11/moments/StopGradientñ
0layer_normalization_11/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_11/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 22
0layer_normalization_11/moments/SquaredDifferenceÀ
9layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_11/moments/variance/reduction_indices
'layer_normalization_11/moments/varianceMean4layer_normalization_11/moments/SquaredDifference:z:0Blayer_normalization_11/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2)
'layer_normalization_11/moments/variance
&layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_11/batchnorm/add/yï
$layer_normalization_11/batchnorm/addAddV20layer_normalization_11/moments/variance:output:0/layer_normalization_11/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2&
$layer_normalization_11/batchnorm/addº
&layer_normalization_11/batchnorm/RsqrtRsqrt(layer_normalization_11/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2(
&layer_normalization_11/batchnorm/Rsqrtã
3layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_11/batchnorm/mul/ReadVariableOpó
$layer_normalization_11/batchnorm/mulMul*layer_normalization_11/batchnorm/Rsqrt:y:0;layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2&
$layer_normalization_11/batchnorm/mulÃ
&layer_normalization_11/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_11/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_11/batchnorm/mul_1æ
&layer_normalization_11/batchnorm/mul_2Mul,layer_normalization_11/moments/mean:output:0(layer_normalization_11/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_11/batchnorm/mul_2×
/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_11/batchnorm/ReadVariableOpï
$layer_normalization_11/batchnorm/subSub7layer_normalization_11/batchnorm/ReadVariableOp:value:0*layer_normalization_11/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2&
$layer_normalization_11/batchnorm/subæ
&layer_normalization_11/batchnorm/add_1AddV2*layer_normalization_11/batchnorm/mul_1:z:0(layer_normalization_11/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_11/batchnorm/add_1Ý
IdentityIdentity*layer_normalization_11/batchnorm/add_1:z:00^layer_normalization_10/batchnorm/ReadVariableOp4^layer_normalization_10/batchnorm/mul/ReadVariableOp0^layer_normalization_11/batchnorm/ReadVariableOp4^layer_normalization_11/batchnorm/mul/ReadVariableOp;^multi_head_attention_5/attention_output/add/ReadVariableOpE^multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_5/key/add/ReadVariableOp8^multi_head_attention_5/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/query/add/ReadVariableOp:^multi_head_attention_5/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/value/add/ReadVariableOp:^multi_head_attention_5/value/einsum/Einsum/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp/^sequential_5/dense_16/Tensordot/ReadVariableOp-^sequential_5/dense_17/BiasAdd/ReadVariableOp/^sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:ÿÿÿÿÿÿÿÿÿÂ ::::::::::::::::2b
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
.sequential_5/dense_17/Tensordot/ReadVariableOp.sequential_5/dense_17/Tensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
ä0
Å
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_50173

inputs
assignmovingavg_50148
assignmovingavg_1_50154)
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
moments/Squeeze_1Ë
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/50148*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_50148*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpð
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/50148*
_output_shapes
: 2
AssignMovingAvg/subç
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/50148*
_output_shapes
: 2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_50148AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/50148*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÑ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/50154*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_50154*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpú
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/50154*
_output_shapes
: 2
AssignMovingAvg_1/subñ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/50154*
_output_shapes
: 2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_50154AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/50154*
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
ã
ü
G__inference_sequential_5_layer_call_and_return_conditional_losses_47972

inputs
dense_16_47961
dense_16_47963
dense_17_47966
dense_17_47968
identity¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall
 dense_16/StatefulPartitionedCallStatefulPartitionedCallinputsdense_16_47961dense_16_47963*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_478512"
 dense_16/StatefulPartitionedCall¼
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_47966dense_17_47968*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_478972"
 dense_17/StatefulPartitionedCallÈ
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
ñ
}
(__inference_conv1d_2_layer_call_fn_50055

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
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_480632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ¼i ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 
 
_user_specified_nameinputs
È
c
E__inference_dropout_16_layer_call_and_return_conditional_losses_48734

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
È
c
E__inference_dropout_17_layer_call_and_return_conditional_losses_48791

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ì

P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_50357

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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
batchnorm/add_1à
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
ê
¨
5__inference_batch_normalization_4_layer_call_fn_50206

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¨
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
GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_476322
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
­P
®
B__inference_model_2_layer_call_and_return_conditional_losses_49173

inputs
inputs_1(
$token_and_position_embedding_2_49089(
$token_and_position_embedding_2_49091
conv1d_2_49094
conv1d_2_49096
batch_normalization_4_49101
batch_normalization_4_49103
batch_normalization_4_49105
batch_normalization_4_49107
batch_normalization_5_49110
batch_normalization_5_49112
batch_normalization_5_49114
batch_normalization_5_49116
transformer_block_5_49120
transformer_block_5_49122
transformer_block_5_49124
transformer_block_5_49126
transformer_block_5_49128
transformer_block_5_49130
transformer_block_5_49132
transformer_block_5_49134
transformer_block_5_49136
transformer_block_5_49138
transformer_block_5_49140
transformer_block_5_49142
transformer_block_5_49144
transformer_block_5_49146
transformer_block_5_49148
transformer_block_5_49150
dense_18_49155
dense_18_49157
dense_19_49161
dense_19_49163
dense_20_49167
dense_20_49169
identity¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢ dense_20/StatefulPartitionedCall¢6token_and_position_embedding_2/StatefulPartitionedCall¢+transformer_block_5/StatefulPartitionedCall
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs$token_and_position_embedding_2_49089$token_and_position_embedding_2_49091*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *b
f]R[
Y__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_4803128
6token_and_position_embedding_2/StatefulPartitionedCallÒ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0conv1d_2_49094conv1d_2_49096*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_480632"
 conv1d_2/StatefulPartitionedCallµ
#average_pooling1d_5/PartitionedCallPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_475302%
#average_pooling1d_5/PartitionedCall
#average_pooling1d_4/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_475152%
#average_pooling1d_4/PartitionedCall¾
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_4/PartitionedCall:output:0batch_normalization_4_49101batch_normalization_4_49103batch_normalization_4_49105batch_normalization_4_49107*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_481362/
-batch_normalization_4/StatefulPartitionedCall¾
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_5/PartitionedCall:output:0batch_normalization_5_49110batch_normalization_5_49112batch_normalization_5_49114batch_normalization_5_49116*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_482272/
-batch_normalization_5/StatefulPartitionedCall»
add_2/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:06batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_482692
add_2/PartitionedCallþ
+transformer_block_5/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0transformer_block_5_49120transformer_block_5_49122transformer_block_5_49124transformer_block_5_49126transformer_block_5_49128transformer_block_5_49130transformer_block_5_49132transformer_block_5_49134transformer_block_5_49136transformer_block_5_49138transformer_block_5_49140transformer_block_5_49142transformer_block_5_49144transformer_block_5_49146transformer_block_5_49148transformer_block_5_49150*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_transformer_block_5_layer_call_and_return_conditional_losses_485532-
+transformer_block_5/StatefulPartitionedCallº
*global_average_pooling1d_2/PartitionedCallPartitionedCall4transformer_block_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_486672,
*global_average_pooling1d_2/PartitionedCall
concatenate_2/PartitionedCallPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_486812
concatenate_2/PartitionedCall´
 dense_18/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_18_49155dense_18_49157*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_487012"
 dense_18/StatefulPartitionedCallÿ
dropout_16/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_487342
dropout_16/PartitionedCall±
 dense_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0dense_19_49161dense_19_49163*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_19_layer_call_and_return_conditional_losses_487582"
 dense_19/StatefulPartitionedCallÿ
dropout_17/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_487912
dropout_17/PartitionedCall±
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0dense_20_49167dense_20_49169*
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
GPU2*0J 8 *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_488142"
 dense_20/StatefulPartitionedCallÐ
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_5/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ¼i:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_5/StatefulPartitionedCall+transformer_block_5/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


'__inference_model_2_layer_call_fn_49923
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

unknown_32
identity¢StatefulPartitionedCallµ
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
unknown_32*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*@
_read_only_resource_inputs"
 	 !"#*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_490112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ¼i:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
· 
â
C__inference_dense_16_layer_call_and_return_conditional_losses_47851

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
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
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
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
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÂ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
Ì
¨
5__inference_batch_normalization_4_layer_call_fn_50137

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
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_481362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
J
®
G__inference_sequential_5_layer_call_and_return_conditional_losses_51006

inputs.
*dense_16_tensordot_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource.
*dense_17_tensordot_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
identity¢dense_16/BiasAdd/ReadVariableOp¢!dense_16/Tensordot/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢!dense_17/Tensordot/ReadVariableOp±
!dense_16/Tensordot/ReadVariableOpReadVariableOp*dense_16_tensordot_readvariableop_resource*
_output_shapes

:  *
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
dense_16/Tensordot/stack¬
dense_16/Tensordot/transpose	Transposeinputs"dense_16/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dense_16/Tensordot/transposeÃ
dense_16/Tensordot/ReshapeReshape dense_16/Tensordot/transpose:y:0!dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_16/Tensordot/ReshapeÂ
dense_16/Tensordot/MatMulMatMul#dense_16/Tensordot/Reshape:output:0)dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_16/Tensordot/MatMul
dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
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
dense_16/Tensordot/concat_1µ
dense_16/TensordotReshape#dense_16/Tensordot/MatMul:product:0$dense_16/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dense_16/Tensordot§
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_16/BiasAdd/ReadVariableOp¬
dense_16/BiasAddBiasAdddense_16/Tensordot:output:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dense_16/BiasAddx
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dense_16/Relu±
!dense_17/Tensordot/ReadVariableOpReadVariableOp*dense_17_tensordot_readvariableop_resource*
_output_shapes

:  *
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
dense_17/Tensordot/stackÁ
dense_17/Tensordot/transpose	Transposedense_16/Relu:activations:0"dense_17/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
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
dense_17/Tensordot/concat_1µ
dense_17/TensordotReshape#dense_17/Tensordot/MatMul:product:0$dense_17/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dense_17/Tensordot§
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_17/BiasAdd/ReadVariableOp¬
dense_17/BiasAddBiasAdddense_17/Tensordot:output:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dense_17/BiasAddþ
IdentityIdentitydense_17/BiasAdd:output:0 ^dense_16/BiasAdd/ReadVariableOp"^dense_16/Tensordot/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp"^dense_17/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2F
!dense_16/Tensordot/ReadVariableOp!dense_16/Tensordot/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2F
!dense_17/Tensordot/ReadVariableOp!dense_17/Tensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
ì

P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_48227

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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
batchnorm/add_1à
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
Ö
â
C__inference_dense_17_layer_call_and_return_conditional_losses_47897

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
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
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
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
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÂ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
Ä
l
@__inference_add_2_layer_call_and_return_conditional_losses_50389
inputs_0
inputs_1
identity^
addAddV2inputs_0inputs_1*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÂ :ÿÿÿÿÿÿÿÿÿÂ :V R
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
"
_user_specified_name
inputs/1
õ
V
:__inference_global_average_pooling1d_2_layer_call_fn_50766

inputs
identityß
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_479992
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û

G__inference_sequential_5_layer_call_and_return_conditional_losses_47914
dense_16_input
dense_16_47862
dense_16_47864
dense_17_47908
dense_17_47910
identity¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¡
 dense_16/StatefulPartitionedCallStatefulPartitionedCalldense_16_inputdense_16_47862dense_16_47864*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_478512"
 dense_16/StatefulPartitionedCall¼
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_47908dense_17_47910*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_478972"
 dense_17/StatefulPartitionedCallÈ
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:\ X
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
(
_user_specified_namedense_16_input
û

G__inference_sequential_5_layer_call_and_return_conditional_losses_47928
dense_16_input
dense_16_47917
dense_16_47919
dense_17_47922
dense_17_47924
identity¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¡
 dense_16/StatefulPartitionedCallStatefulPartitionedCalldense_16_inputdense_16_47917dense_16_47919*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_478512"
 dense_16/StatefulPartitionedCall¼
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_47922dense_17_47924*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_478972"
 dense_17/StatefulPartitionedCallÈ
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:\ X
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
(
_user_specified_namedense_16_input
ë

Y__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_48031
x&
"embedding_5_embedding_lookup_48018&
"embedding_4_embedding_lookup_48024
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
range­
embedding_5/embedding_lookupResourceGather"embedding_5_embedding_lookup_48018range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_5/embedding_lookup/48018*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02
embedding_5/embedding_lookup
%embedding_5/embedding_lookup/IdentityIdentity%embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_5/embedding_lookup/48018*'
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
:ÿÿÿÿÿÿÿÿÿ¼i2
embedding_4/Cast¸
embedding_4/embedding_lookupResourceGather"embedding_4_embedding_lookup_48024embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_4/embedding_lookup/48024*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *
dtype02
embedding_4/embedding_lookup
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_4/embedding_lookup/48024*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2'
%embedding_4/embedding_lookup/IdentityÅ
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2)
'embedding_4/embedding_lookup/Identity_1®
addAddV20embedding_4/embedding_lookup/Identity_1:output:00embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2
add
IdentityIdentityadd:z:0^embedding_4/embedding_lookup^embedding_5/embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¼i::2<
embedding_4/embedding_lookupembedding_4/embedding_lookup2<
embedding_5/embedding_lookupembedding_5/embedding_lookup:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i

_user_specified_namex

F
*__inference_dropout_17_layer_call_fn_50873

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_487912
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
Ü
C__inference_dense_20_layer_call_and_return_conditional_losses_50883

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
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
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
È
c
E__inference_dropout_17_layer_call_and_return_conditional_losses_50863

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ý
}
(__inference_dense_20_layer_call_fn_50892

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
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
GPU2*0J 8 *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_488142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ê
¨
5__inference_batch_normalization_5_layer_call_fn_50370

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
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_482072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
³0
Å
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_50337

inputs
assignmovingavg_50312
assignmovingavg_1_50318)
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
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
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
moments/Squeeze_1Ë
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/50312*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_50312*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpð
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/50312*
_output_shapes
: 2
AssignMovingAvg/subç
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/50312*
_output_shapes
: 2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_50312AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/50312*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÑ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/50318*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_50318*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpú
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/50318*
_output_shapes
: 2
AssignMovingAvg_1/subñ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/50318*
_output_shapes
: 2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_50318AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/50318*
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
batchnorm/add_1¸
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
Ý
q
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_50750

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
:ÿÿÿÿÿÿÿÿÿ 2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÂ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
£
c
*__inference_dropout_17_layer_call_fn_50868

inputs
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_487862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
º

,__inference_sequential_5_layer_call_fn_51019

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
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_479452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs


P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_50275

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


P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_47665

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
ÊÖ
¿8
__inference__traced_save_51438
file_prefix.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop*
&savev2_adamax_iter_read_readvariableop	T
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
 savev2_count_read_readvariableop7
3savev2_adamax_conv1d_2_kernel_m_read_readvariableop5
1savev2_adamax_conv1d_2_bias_m_read_readvariableopC
?savev2_adamax_batch_normalization_4_gamma_m_read_readvariableopB
>savev2_adamax_batch_normalization_4_beta_m_read_readvariableopC
?savev2_adamax_batch_normalization_5_gamma_m_read_readvariableopB
>savev2_adamax_batch_normalization_5_beta_m_read_readvariableop7
3savev2_adamax_dense_18_kernel_m_read_readvariableop5
1savev2_adamax_dense_18_bias_m_read_readvariableop7
3savev2_adamax_dense_19_kernel_m_read_readvariableop5
1savev2_adamax_dense_19_bias_m_read_readvariableop7
3savev2_adamax_dense_20_kernel_m_read_readvariableop5
1savev2_adamax_dense_20_bias_m_read_readvariableop]
Ysavev2_adamax_token_and_position_embedding_2_embedding_4_embeddings_m_read_readvariableop]
Ysavev2_adamax_token_and_position_embedding_2_embedding_5_embeddings_m_read_readvariableop_
[savev2_adamax_transformer_block_5_multi_head_attention_5_query_kernel_m_read_readvariableop]
Ysavev2_adamax_transformer_block_5_multi_head_attention_5_query_bias_m_read_readvariableop]
Ysavev2_adamax_transformer_block_5_multi_head_attention_5_key_kernel_m_read_readvariableop[
Wsavev2_adamax_transformer_block_5_multi_head_attention_5_key_bias_m_read_readvariableop_
[savev2_adamax_transformer_block_5_multi_head_attention_5_value_kernel_m_read_readvariableop]
Ysavev2_adamax_transformer_block_5_multi_head_attention_5_value_bias_m_read_readvariableopj
fsavev2_adamax_transformer_block_5_multi_head_attention_5_attention_output_kernel_m_read_readvariableoph
dsavev2_adamax_transformer_block_5_multi_head_attention_5_attention_output_bias_m_read_readvariableop7
3savev2_adamax_dense_16_kernel_m_read_readvariableop5
1savev2_adamax_dense_16_bias_m_read_readvariableop7
3savev2_adamax_dense_17_kernel_m_read_readvariableop5
1savev2_adamax_dense_17_bias_m_read_readvariableopX
Tsavev2_adamax_transformer_block_5_layer_normalization_10_gamma_m_read_readvariableopW
Ssavev2_adamax_transformer_block_5_layer_normalization_10_beta_m_read_readvariableopX
Tsavev2_adamax_transformer_block_5_layer_normalization_11_gamma_m_read_readvariableopW
Ssavev2_adamax_transformer_block_5_layer_normalization_11_beta_m_read_readvariableop7
3savev2_adamax_conv1d_2_kernel_v_read_readvariableop5
1savev2_adamax_conv1d_2_bias_v_read_readvariableopC
?savev2_adamax_batch_normalization_4_gamma_v_read_readvariableopB
>savev2_adamax_batch_normalization_4_beta_v_read_readvariableopC
?savev2_adamax_batch_normalization_5_gamma_v_read_readvariableopB
>savev2_adamax_batch_normalization_5_beta_v_read_readvariableop7
3savev2_adamax_dense_18_kernel_v_read_readvariableop5
1savev2_adamax_dense_18_bias_v_read_readvariableop7
3savev2_adamax_dense_19_kernel_v_read_readvariableop5
1savev2_adamax_dense_19_bias_v_read_readvariableop7
3savev2_adamax_dense_20_kernel_v_read_readvariableop5
1savev2_adamax_dense_20_bias_v_read_readvariableop]
Ysavev2_adamax_token_and_position_embedding_2_embedding_4_embeddings_v_read_readvariableop]
Ysavev2_adamax_token_and_position_embedding_2_embedding_5_embeddings_v_read_readvariableop_
[savev2_adamax_transformer_block_5_multi_head_attention_5_query_kernel_v_read_readvariableop]
Ysavev2_adamax_transformer_block_5_multi_head_attention_5_query_bias_v_read_readvariableop]
Ysavev2_adamax_transformer_block_5_multi_head_attention_5_key_kernel_v_read_readvariableop[
Wsavev2_adamax_transformer_block_5_multi_head_attention_5_key_bias_v_read_readvariableop_
[savev2_adamax_transformer_block_5_multi_head_attention_5_value_kernel_v_read_readvariableop]
Ysavev2_adamax_transformer_block_5_multi_head_attention_5_value_bias_v_read_readvariableopj
fsavev2_adamax_transformer_block_5_multi_head_attention_5_attention_output_kernel_v_read_readvariableoph
dsavev2_adamax_transformer_block_5_multi_head_attention_5_attention_output_bias_v_read_readvariableop7
3savev2_adamax_dense_16_kernel_v_read_readvariableop5
1savev2_adamax_dense_16_bias_v_read_readvariableop7
3savev2_adamax_dense_17_kernel_v_read_readvariableop5
1savev2_adamax_dense_17_bias_v_read_readvariableopX
Tsavev2_adamax_transformer_block_5_layer_normalization_10_gamma_v_read_readvariableopW
Ssavev2_adamax_transformer_block_5_layer_normalization_10_beta_v_read_readvariableopX
Tsavev2_adamax_transformer_block_5_layer_normalization_11_gamma_v_read_readvariableopW
Ssavev2_adamax_transformer_block_5_layer_normalization_11_beta_v_read_readvariableop
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
ShardedFilenameÔ3
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*æ2
valueÜ2BÙ2fB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names×
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*á
value×BÔfB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesß6
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop&savev2_adamax_iter_read_readvariableopPsavev2_token_and_position_embedding_2_embedding_4_embeddings_read_readvariableopPsavev2_token_and_position_embedding_2_embedding_5_embeddings_read_readvariableopRsavev2_transformer_block_5_multi_head_attention_5_query_kernel_read_readvariableopPsavev2_transformer_block_5_multi_head_attention_5_query_bias_read_readvariableopPsavev2_transformer_block_5_multi_head_attention_5_key_kernel_read_readvariableopNsavev2_transformer_block_5_multi_head_attention_5_key_bias_read_readvariableopRsavev2_transformer_block_5_multi_head_attention_5_value_kernel_read_readvariableopPsavev2_transformer_block_5_multi_head_attention_5_value_bias_read_readvariableop]savev2_transformer_block_5_multi_head_attention_5_attention_output_kernel_read_readvariableop[savev2_transformer_block_5_multi_head_attention_5_attention_output_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableopKsavev2_transformer_block_5_layer_normalization_10_gamma_read_readvariableopJsavev2_transformer_block_5_layer_normalization_10_beta_read_readvariableopKsavev2_transformer_block_5_layer_normalization_11_gamma_read_readvariableopJsavev2_transformer_block_5_layer_normalization_11_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_adamax_conv1d_2_kernel_m_read_readvariableop1savev2_adamax_conv1d_2_bias_m_read_readvariableop?savev2_adamax_batch_normalization_4_gamma_m_read_readvariableop>savev2_adamax_batch_normalization_4_beta_m_read_readvariableop?savev2_adamax_batch_normalization_5_gamma_m_read_readvariableop>savev2_adamax_batch_normalization_5_beta_m_read_readvariableop3savev2_adamax_dense_18_kernel_m_read_readvariableop1savev2_adamax_dense_18_bias_m_read_readvariableop3savev2_adamax_dense_19_kernel_m_read_readvariableop1savev2_adamax_dense_19_bias_m_read_readvariableop3savev2_adamax_dense_20_kernel_m_read_readvariableop1savev2_adamax_dense_20_bias_m_read_readvariableopYsavev2_adamax_token_and_position_embedding_2_embedding_4_embeddings_m_read_readvariableopYsavev2_adamax_token_and_position_embedding_2_embedding_5_embeddings_m_read_readvariableop[savev2_adamax_transformer_block_5_multi_head_attention_5_query_kernel_m_read_readvariableopYsavev2_adamax_transformer_block_5_multi_head_attention_5_query_bias_m_read_readvariableopYsavev2_adamax_transformer_block_5_multi_head_attention_5_key_kernel_m_read_readvariableopWsavev2_adamax_transformer_block_5_multi_head_attention_5_key_bias_m_read_readvariableop[savev2_adamax_transformer_block_5_multi_head_attention_5_value_kernel_m_read_readvariableopYsavev2_adamax_transformer_block_5_multi_head_attention_5_value_bias_m_read_readvariableopfsavev2_adamax_transformer_block_5_multi_head_attention_5_attention_output_kernel_m_read_readvariableopdsavev2_adamax_transformer_block_5_multi_head_attention_5_attention_output_bias_m_read_readvariableop3savev2_adamax_dense_16_kernel_m_read_readvariableop1savev2_adamax_dense_16_bias_m_read_readvariableop3savev2_adamax_dense_17_kernel_m_read_readvariableop1savev2_adamax_dense_17_bias_m_read_readvariableopTsavev2_adamax_transformer_block_5_layer_normalization_10_gamma_m_read_readvariableopSsavev2_adamax_transformer_block_5_layer_normalization_10_beta_m_read_readvariableopTsavev2_adamax_transformer_block_5_layer_normalization_11_gamma_m_read_readvariableopSsavev2_adamax_transformer_block_5_layer_normalization_11_beta_m_read_readvariableop3savev2_adamax_conv1d_2_kernel_v_read_readvariableop1savev2_adamax_conv1d_2_bias_v_read_readvariableop?savev2_adamax_batch_normalization_4_gamma_v_read_readvariableop>savev2_adamax_batch_normalization_4_beta_v_read_readvariableop?savev2_adamax_batch_normalization_5_gamma_v_read_readvariableop>savev2_adamax_batch_normalization_5_beta_v_read_readvariableop3savev2_adamax_dense_18_kernel_v_read_readvariableop1savev2_adamax_dense_18_bias_v_read_readvariableop3savev2_adamax_dense_19_kernel_v_read_readvariableop1savev2_adamax_dense_19_bias_v_read_readvariableop3savev2_adamax_dense_20_kernel_v_read_readvariableop1savev2_adamax_dense_20_bias_v_read_readvariableopYsavev2_adamax_token_and_position_embedding_2_embedding_4_embeddings_v_read_readvariableopYsavev2_adamax_token_and_position_embedding_2_embedding_5_embeddings_v_read_readvariableop[savev2_adamax_transformer_block_5_multi_head_attention_5_query_kernel_v_read_readvariableopYsavev2_adamax_transformer_block_5_multi_head_attention_5_query_bias_v_read_readvariableopYsavev2_adamax_transformer_block_5_multi_head_attention_5_key_kernel_v_read_readvariableopWsavev2_adamax_transformer_block_5_multi_head_attention_5_key_bias_v_read_readvariableop[savev2_adamax_transformer_block_5_multi_head_attention_5_value_kernel_v_read_readvariableopYsavev2_adamax_transformer_block_5_multi_head_attention_5_value_bias_v_read_readvariableopfsavev2_adamax_transformer_block_5_multi_head_attention_5_attention_output_kernel_v_read_readvariableopdsavev2_adamax_transformer_block_5_multi_head_attention_5_attention_output_bias_v_read_readvariableop3savev2_adamax_dense_16_kernel_v_read_readvariableop1savev2_adamax_dense_16_bias_v_read_readvariableop3savev2_adamax_dense_17_kernel_v_read_readvariableop1savev2_adamax_dense_17_bias_v_read_readvariableopTsavev2_adamax_transformer_block_5_layer_normalization_10_gamma_v_read_readvariableopSsavev2_adamax_transformer_block_5_layer_normalization_10_beta_v_read_readvariableopTsavev2_adamax_transformer_block_5_layer_normalization_11_gamma_v_read_readvariableopSsavev2_adamax_transformer_block_5_layer_normalization_11_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *t
dtypesj
h2f	2
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

identity_1Identity_1:output:0*Î
_input_shapes¼
¹: :  : : : : : : : : : :( : :  : : :: : : : : : :	¼i :  : :  : :  : :  : :  : :  : : : : : : : :  : : : : : :( : :  : : :: :	¼i :  : :  : :  : :  : :  : :  : : : : : :  : : : : : :( : :  : : :: :	¼i :  : :  : :  : :  : :  : :  : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 
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
: :$ 

_output_shapes

:( : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: :%!

_output_shapes
:	¼i :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
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
:  : 

_output_shapes
: :$  

_output_shapes

:  : !

_output_shapes
: :$" 

_output_shapes

:  : #

_output_shapes
: : $
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
: :(

_output_shapes
: :)

_output_shapes
: :(*$
"
_output_shapes
:  : +

_output_shapes
: : ,

_output_shapes
: : -

_output_shapes
: : .

_output_shapes
: : /

_output_shapes
: :$0 

_output_shapes

:( : 1

_output_shapes
: :$2 

_output_shapes

:  : 3

_output_shapes
: :$4 

_output_shapes

: : 5

_output_shapes
::$6 

_output_shapes

: :%7!

_output_shapes
:	¼i :(8$
"
_output_shapes
:  :$9 

_output_shapes

: :(:$
"
_output_shapes
:  :$; 

_output_shapes

: :(<$
"
_output_shapes
:  :$= 

_output_shapes

: :(>$
"
_output_shapes
:  : ?

_output_shapes
: :$@ 

_output_shapes

:  : A

_output_shapes
: :$B 

_output_shapes

:  : C

_output_shapes
: : D

_output_shapes
: : E

_output_shapes
: : F

_output_shapes
: : G

_output_shapes
: :(H$
"
_output_shapes
:  : I

_output_shapes
: : J
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
: :$N 

_output_shapes

:( : O

_output_shapes
: :$P 

_output_shapes

:  : Q

_output_shapes
: :$R 

_output_shapes

: : S

_output_shapes
::$T 

_output_shapes

: :%U!

_output_shapes
:	¼i :(V$
"
_output_shapes
:  :$W 

_output_shapes

: :(X$
"
_output_shapes
:  :$Y 

_output_shapes

: :(Z$
"
_output_shapes
:  :$[ 

_output_shapes

: :(\$
"
_output_shapes
:  : ]

_output_shapes
: :$^ 

_output_shapes

:  : _

_output_shapes
: :$` 

_output_shapes

:  : a

_output_shapes
: : b

_output_shapes
: : c

_output_shapes
: : d

_output_shapes
: : e

_output_shapes
: :f

_output_shapes
: 
ñ
}
(__inference_dense_16_layer_call_fn_51072

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
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_478512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÂ ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
¢
Y
-__inference_concatenate_2_layer_call_fn_50779
inputs_0
inputs_1
identityÖ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_486812
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
£
c
*__inference_dropout_16_layer_call_fn_50821

inputs
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_487292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_47805

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
ä0
Å
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_50255

inputs
assignmovingavg_50230
assignmovingavg_1_50236)
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
moments/Squeeze_1Ë
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/50230*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_50230*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpð
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/50230*
_output_shapes
: 2
AssignMovingAvg/subç
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/50230*
_output_shapes
: 2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_50230AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/50230*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÑ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/50236*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_50236*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpú
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/50236*
_output_shapes
: 2
AssignMovingAvg_1/subñ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/50236*
_output_shapes
: 2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_50236AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/50236*
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
í	
Ü
C__inference_dense_18_layer_call_and_return_conditional_losses_48701

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:( *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

q
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_50761

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
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³0
Å
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_48116

inputs
assignmovingavg_48091
assignmovingavg_1_48097)
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
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
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
moments/Squeeze_1Ë
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/48091*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_48091*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpð
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/48091*
_output_shapes
: 2
AssignMovingAvg/subç
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/48091*
_output_shapes
: 2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_48091AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/48091*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÑ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/48097*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_48097*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpú
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/48097*
_output_shapes
: 2
AssignMovingAvg_1/subñ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/48097*
_output_shapes
: 2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_48097AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/48097*
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
batchnorm/add_1¸
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
Ò
§
,__inference_sequential_5_layer_call_fn_47983
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
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_479722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
(
_user_specified_namedense_16_input
´
r
H__inference_concatenate_2_layer_call_and_return_conditional_losses_48681

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
:ÿÿÿÿÿÿÿÿÿ(2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
j
N__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_47515

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
È
c
E__inference_dropout_16_layer_call_and_return_conditional_losses_50816

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ý
}
(__inference_dense_19_layer_call_fn_50846

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_19_layer_call_and_return_conditional_losses_487582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Õ
å$
B__inference_model_2_layer_call_and_return_conditional_losses_49622
inputs_0
inputs_1E
Atoken_and_position_embedding_2_embedding_5_embedding_lookup_49340E
Atoken_and_position_embedding_2_embedding_4_embedding_lookup_493468
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource/
+batch_normalization_4_assignmovingavg_493801
-batch_normalization_4_assignmovingavg_1_49386?
;batch_normalization_4_batchnorm_mul_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource/
+batch_normalization_5_assignmovingavg_494121
-batch_normalization_5_assignmovingavg_1_49418?
;batch_normalization_5_batchnorm_mul_readvariableop_resource;
7batch_normalization_5_batchnorm_readvariableop_resourceZ
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
Ltransformer_block_5_layer_normalization_11_batchnorm_readvariableop_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource
identity¢9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_4/AssignMovingAvg/ReadVariableOp¢;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_4/batchnorm/ReadVariableOp¢2batch_normalization_4/batchnorm/mul/ReadVariableOp¢9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_5/AssignMovingAvg/ReadVariableOp¢;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_5/batchnorm/ReadVariableOp¢2batch_normalization_5/batchnorm/mul/ReadVariableOp¢conv1d_2/BiasAdd/ReadVariableOp¢+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp¢dense_18/BiasAdd/ReadVariableOp¢dense_18/MatMul/ReadVariableOp¢dense_19/BiasAdd/ReadVariableOp¢dense_19/MatMul/ReadVariableOp¢dense_20/BiasAdd/ReadVariableOp¢dense_20/MatMul/ReadVariableOp¢;token_and_position_embedding_2/embedding_4/embedding_lookup¢;token_and_position_embedding_2/embedding_5/embedding_lookup¢Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp¢Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp¢Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp¢Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp¢Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp¢Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp¢Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp¢Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp¢Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOp¢Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp¢Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOp¢Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp¢@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp¢Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp¢@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp¢Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp
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
$token_and_position_embedding_2/rangeÈ
;token_and_position_embedding_2/embedding_5/embedding_lookupResourceGatherAtoken_and_position_embedding_2_embedding_5_embedding_lookup_49340-token_and_position_embedding_2/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/49340*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02=
;token_and_position_embedding_2/embedding_5/embedding_lookup
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/49340*'
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
:ÿÿÿÿÿÿÿÿÿ¼i21
/token_and_position_embedding_2/embedding_4/CastÓ
;token_and_position_embedding_2/embedding_4/embedding_lookupResourceGatherAtoken_and_position_embedding_2_embedding_4_embedding_lookup_493463token_and_position_embedding_2/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/49346*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *
dtype02=
;token_and_position_embedding_2/embedding_4/embedding_lookup
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/49346*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2F
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity¢
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2H
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1ª
"token_and_position_embedding_2/addAddV2Otoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2$
"token_and_position_embedding_2/add
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_2/conv1d/ExpandDims/dimÒ
conv1d_2/conv1d/ExpandDims
ExpandDims&token_and_position_embedding_2/add:z:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2
conv1d_2/conv1d/ExpandDimsÓ
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dimÛ
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_2/conv1d/ExpandDims_1Û
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *
paddingSAME*
strides
2
conv1d_2/conv1d®
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_2/conv1d/Squeeze§
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_2/BiasAdd/ReadVariableOp±
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2
conv1d_2/BiasAddx
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2
conv1d_2/Relu
"average_pooling1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_5/ExpandDims/dimÞ
average_pooling1d_5/ExpandDims
ExpandDims&token_and_position_embedding_2/add:z:0+average_pooling1d_5/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2 
average_pooling1d_5/ExpandDimså
average_pooling1d_5/AvgPoolAvgPool'average_pooling1d_5/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_5/AvgPool¹
average_pooling1d_5/SqueezeSqueeze$average_pooling1d_5/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
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
ExpandDimsconv1d_2/Relu:activations:0+average_pooling1d_4/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2 
average_pooling1d_4/ExpandDimså
average_pooling1d_4/AvgPoolAvgPool'average_pooling1d_4/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_4/AvgPool¹
average_pooling1d_4/SqueezeSqueeze$average_pooling1d_4/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
squeeze_dims
2
average_pooling1d_4/Squeeze½
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_4/moments/mean/reduction_indicesó
"batch_normalization_4/moments/meanMean$average_pooling1d_4/Squeeze:output:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_4/moments/meanÂ
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_4/moments/StopGradient
/batch_normalization_4/moments/SquaredDifferenceSquaredDifference$average_pooling1d_4/Squeeze:output:03batch_normalization_4/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 21
/batch_normalization_4/moments/SquaredDifferenceÅ
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_4/moments/variance/reduction_indices
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2(
&batch_normalization_4/moments/varianceÃ
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_4/moments/SqueezeË
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1
+batch_normalization_4/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/49380*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_4/AssignMovingAvg/decayÔ
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_4_assignmovingavg_49380*
_output_shapes
: *
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOpÞ
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/49380*
_output_shapes
: 2+
)batch_normalization_4/AssignMovingAvg/subÕ
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/49380*
_output_shapes
: 2+
)batch_normalization_4/AssignMovingAvg/mul±
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_4_assignmovingavg_49380-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/49380*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_4/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/49386*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_4/AssignMovingAvg_1/decayÚ
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_4_assignmovingavg_1_49386*
_output_shapes
: *
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpè
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/49386*
_output_shapes
: 2-
+batch_normalization_4/AssignMovingAvg_1/subß
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/49386*
_output_shapes
: 2-
+batch_normalization_4/AssignMovingAvg_1/mul½
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_4_assignmovingavg_1_49386/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/49386*
_output_shapes
 *
dtype02=
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_4/batchnorm/add/yÚ
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/add¥
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/Rsqrtà
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/mulÛ
%batch_normalization_4/batchnorm/mul_1Mul$average_pooling1d_4/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2'
%batch_normalization_4/batchnorm/mul_1Ó
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/mul_2Ô
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_4/batchnorm/ReadVariableOpÙ
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/subâ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2'
%batch_normalization_4/batchnorm/add_1½
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_5/moments/mean/reduction_indicesó
"batch_normalization_5/moments/meanMean$average_pooling1d_5/Squeeze:output:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_5/moments/meanÂ
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_5/moments/StopGradient
/batch_normalization_5/moments/SquaredDifferenceSquaredDifference$average_pooling1d_5/Squeeze:output:03batch_normalization_5/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 21
/batch_normalization_5/moments/SquaredDifferenceÅ
8batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_5/moments/variance/reduction_indices
&batch_normalization_5/moments/varianceMean3batch_normalization_5/moments/SquaredDifference:z:0Abatch_normalization_5/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2(
&batch_normalization_5/moments/varianceÃ
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_5/moments/SqueezeË
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_5/moments/Squeeze_1
+batch_normalization_5/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/49412*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_5/AssignMovingAvg/decayÔ
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_5_assignmovingavg_49412*
_output_shapes
: *
dtype026
4batch_normalization_5/AssignMovingAvg/ReadVariableOpÞ
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/49412*
_output_shapes
: 2+
)batch_normalization_5/AssignMovingAvg/subÕ
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/49412*
_output_shapes
: 2+
)batch_normalization_5/AssignMovingAvg/mul±
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_5_assignmovingavg_49412-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/49412*
_output_shapes
 *
dtype02;
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_5/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/49418*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_5/AssignMovingAvg_1/decayÚ
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_5_assignmovingavg_1_49418*
_output_shapes
: *
dtype028
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpè
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/49418*
_output_shapes
: 2-
+batch_normalization_5/AssignMovingAvg_1/subß
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/49418*
_output_shapes
: 2-
+batch_normalization_5/AssignMovingAvg_1/mul½
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_5_assignmovingavg_1_49418/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/49418*
_output_shapes
 *
dtype02=
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_5/batchnorm/add/yÚ
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_5/batchnorm/add¥
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_5/batchnorm/Rsqrtà
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_5/batchnorm/mulÛ
%batch_normalization_5/batchnorm/mul_1Mul$average_pooling1d_5/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2'
%batch_normalization_5/batchnorm/mul_1Ó
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_5/batchnorm/mul_2Ô
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_5/batchnorm/ReadVariableOpÙ
#batch_normalization_5/batchnorm/subSub6batch_normalization_5/batchnorm/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_5/batchnorm/subâ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2'
%batch_normalization_5/batchnorm/add_1¬
	add_2/addAddV2)batch_normalization_4/batchnorm/add_1:z:0)batch_normalization_5/batchnorm/add_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
	add_2/add¹
Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_5_multi_head_attention_5_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpÑ
>transformer_block_5/multi_head_attention_5/query/einsum/EinsumEinsumadd_2/add:z:0Utransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2@
>transformer_block_5/multi_head_attention_5/query/einsum/Einsum
Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOpReadVariableOpLtransformer_block_5_multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOpÆ
4transformer_block_5/multi_head_attention_5/query/addAddV2Gtransformer_block_5/multi_head_attention_5/query/einsum/Einsum:output:0Ktransformer_block_5/multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 26
4transformer_block_5/multi_head_attention_5/query/add³
Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_5_multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpË
<transformer_block_5/multi_head_attention_5/key/einsum/EinsumEinsumadd_2/add:z:0Stransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2>
<transformer_block_5/multi_head_attention_5/key/einsum/Einsum
Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOpReadVariableOpJtransformer_block_5_multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp¾
2transformer_block_5/multi_head_attention_5/key/addAddV2Etransformer_block_5/multi_head_attention_5/key/einsum/Einsum:output:0Itransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 24
2transformer_block_5/multi_head_attention_5/key/add¹
Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_5_multi_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpÑ
>transformer_block_5/multi_head_attention_5/value/einsum/EinsumEinsumadd_2/add:z:0Utransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2@
>transformer_block_5/multi_head_attention_5/value/einsum/Einsum
Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOpReadVariableOpLtransformer_block_5_multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOpÆ
4transformer_block_5/multi_head_attention_5/value/addAddV2Gtransformer_block_5/multi_head_attention_5/value/einsum/Einsum:output:0Ktransformer_block_5/multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 26
4transformer_block_5/multi_head_attention_5/value/add©
0transformer_block_5/multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>22
0transformer_block_5/multi_head_attention_5/Mul/y
.transformer_block_5/multi_head_attention_5/MulMul8transformer_block_5/multi_head_attention_5/query/add:z:09transformer_block_5/multi_head_attention_5/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 20
.transformer_block_5/multi_head_attention_5/MulÎ
8transformer_block_5/multi_head_attention_5/einsum/EinsumEinsum6transformer_block_5/multi_head_attention_5/key/add:z:02transformer_block_5/multi_head_attention_5/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ*
equationaecd,abcd->acbe2:
8transformer_block_5/multi_head_attention_5/einsum/Einsum
:transformer_block_5/multi_head_attention_5/softmax/SoftmaxSoftmaxAtransformer_block_5/multi_head_attention_5/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2<
:transformer_block_5/multi_head_attention_5/softmax/SoftmaxÉ
@transformer_block_5/multi_head_attention_5/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2B
@transformer_block_5/multi_head_attention_5/dropout/dropout/ConstÔ
>transformer_block_5/multi_head_attention_5/dropout/dropout/MulMulDtransformer_block_5/multi_head_attention_5/softmax/Softmax:softmax:0Itransformer_block_5/multi_head_attention_5/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2@
>transformer_block_5/multi_head_attention_5/dropout/dropout/Mulø
@transformer_block_5/multi_head_attention_5/dropout/dropout/ShapeShapeDtransformer_block_5/multi_head_attention_5/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2B
@transformer_block_5/multi_head_attention_5/dropout/dropout/Shapeã
Wtransformer_block_5/multi_head_attention_5/dropout/dropout/random_uniform/RandomUniformRandomUniformItransformer_block_5/multi_head_attention_5/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ*
dtype0*

seed*2Y
Wtransformer_block_5/multi_head_attention_5/dropout/dropout/random_uniform/RandomUniformÛ
Itransformer_block_5/multi_head_attention_5/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2K
Itransformer_block_5/multi_head_attention_5/dropout/dropout/GreaterEqual/y
Gtransformer_block_5/multi_head_attention_5/dropout/dropout/GreaterEqualGreaterEqual`transformer_block_5/multi_head_attention_5/dropout/dropout/random_uniform/RandomUniform:output:0Rtransformer_block_5/multi_head_attention_5/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2I
Gtransformer_block_5/multi_head_attention_5/dropout/dropout/GreaterEqual¢
?transformer_block_5/multi_head_attention_5/dropout/dropout/CastCastKtransformer_block_5/multi_head_attention_5/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2A
?transformer_block_5/multi_head_attention_5/dropout/dropout/CastÐ
@transformer_block_5/multi_head_attention_5/dropout/dropout/Mul_1MulBtransformer_block_5/multi_head_attention_5/dropout/dropout/Mul:z:0Ctransformer_block_5/multi_head_attention_5/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2B
@transformer_block_5/multi_head_attention_5/dropout/dropout/Mul_1å
:transformer_block_5/multi_head_attention_5/einsum_1/EinsumEinsumDtransformer_block_5/multi_head_attention_5/dropout/dropout/Mul_1:z:08transformer_block_5/multi_head_attention_5/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationacbe,aecd->abcd2<
:transformer_block_5/multi_head_attention_5/einsum_1/EinsumÚ
Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_5_multi_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp¤
Itransformer_block_5/multi_head_attention_5/attention_output/einsum/EinsumEinsumCtransformer_block_5/multi_head_attention_5/einsum_1/Einsum:output:0`transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabcd,cde->abe2K
Itransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum´
Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_5_multi_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpî
?transformer_block_5/multi_head_attention_5/attention_output/addAddV2Rtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum:output:0Vtransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2A
?transformer_block_5/multi_head_attention_5/attention_output/add¡
,transformer_block_5/dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2.
,transformer_block_5/dropout_14/dropout/Const
*transformer_block_5/dropout_14/dropout/MulMulCtransformer_block_5/multi_head_attention_5/attention_output/add:z:05transformer_block_5/dropout_14/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2,
*transformer_block_5/dropout_14/dropout/MulÏ
,transformer_block_5/dropout_14/dropout/ShapeShapeCtransformer_block_5/multi_head_attention_5/attention_output/add:z:0*
T0*
_output_shapes
:2.
,transformer_block_5/dropout_14/dropout/Shape¯
Ctransformer_block_5/dropout_14/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_5/dropout_14/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
dtype0*

seed**
seed22E
Ctransformer_block_5/dropout_14/dropout/random_uniform/RandomUniform³
5transformer_block_5/dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=27
5transformer_block_5/dropout_14/dropout/GreaterEqual/y¿
3transformer_block_5/dropout_14/dropout/GreaterEqualGreaterEqualLtransformer_block_5/dropout_14/dropout/random_uniform/RandomUniform:output:0>transformer_block_5/dropout_14/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 25
3transformer_block_5/dropout_14/dropout/GreaterEqualá
+transformer_block_5/dropout_14/dropout/CastCast7transformer_block_5/dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2-
+transformer_block_5/dropout_14/dropout/Castû
,transformer_block_5/dropout_14/dropout/Mul_1Mul.transformer_block_5/dropout_14/dropout/Mul:z:0/transformer_block_5/dropout_14/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2.
,transformer_block_5/dropout_14/dropout/Mul_1³
transformer_block_5/addAddV2add_2/add:z:00transformer_block_5/dropout_14/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
transformer_block_5/addà
Itransformer_block_5/layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_5/layer_normalization_10/moments/mean/reduction_indices³
7transformer_block_5/layer_normalization_10/moments/meanMeantransformer_block_5/add:z:0Rtransformer_block_5/layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(29
7transformer_block_5/layer_normalization_10/moments/mean
?transformer_block_5/layer_normalization_10/moments/StopGradientStopGradient@transformer_block_5/layer_normalization_10/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2A
?transformer_block_5/layer_normalization_10/moments/StopGradient¿
Dtransformer_block_5/layer_normalization_10/moments/SquaredDifferenceSquaredDifferencetransformer_block_5/add:z:0Htransformer_block_5/layer_normalization_10/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2F
Dtransformer_block_5/layer_normalization_10/moments/SquaredDifferenceè
Mtransformer_block_5/layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_5/layer_normalization_10/moments/variance/reduction_indicesì
;transformer_block_5/layer_normalization_10/moments/varianceMeanHtransformer_block_5/layer_normalization_10/moments/SquaredDifference:z:0Vtransformer_block_5/layer_normalization_10/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2=
;transformer_block_5/layer_normalization_10/moments/variance½
:transformer_block_5/layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_5/layer_normalization_10/batchnorm/add/y¿
8transformer_block_5/layer_normalization_10/batchnorm/addAddV2Dtransformer_block_5/layer_normalization_10/moments/variance:output:0Ctransformer_block_5/layer_normalization_10/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2:
8transformer_block_5/layer_normalization_10/batchnorm/addö
:transformer_block_5/layer_normalization_10/batchnorm/RsqrtRsqrt<transformer_block_5/layer_normalization_10/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2<
:transformer_block_5/layer_normalization_10/batchnorm/Rsqrt
Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_5_layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpÃ
8transformer_block_5/layer_normalization_10/batchnorm/mulMul>transformer_block_5/layer_normalization_10/batchnorm/Rsqrt:y:0Otransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2:
8transformer_block_5/layer_normalization_10/batchnorm/mul
:transformer_block_5/layer_normalization_10/batchnorm/mul_1Multransformer_block_5/add:z:0<transformer_block_5/layer_normalization_10/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2<
:transformer_block_5/layer_normalization_10/batchnorm/mul_1¶
:transformer_block_5/layer_normalization_10/batchnorm/mul_2Mul@transformer_block_5/layer_normalization_10/moments/mean:output:0<transformer_block_5/layer_normalization_10/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2<
:transformer_block_5/layer_normalization_10/batchnorm/mul_2
Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_5_layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp¿
8transformer_block_5/layer_normalization_10/batchnorm/subSubKtransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp:value:0>transformer_block_5/layer_normalization_10/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2:
8transformer_block_5/layer_normalization_10/batchnorm/sub¶
:transformer_block_5/layer_normalization_10/batchnorm/add_1AddV2>transformer_block_5/layer_normalization_10/batchnorm/mul_1:z:0<transformer_block_5/layer_normalization_10/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2<
:transformer_block_5/layer_normalization_10/batchnorm/add_1
Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_5_sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

:  *
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
9transformer_block_5/sequential_5/dense_16/Tensordot/stackÇ
=transformer_block_5/sequential_5/dense_16/Tensordot/transpose	Transpose>transformer_block_5/layer_normalization_10/batchnorm/add_1:z:0Ctransformer_block_5/sequential_5/dense_16/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2?
=transformer_block_5/sequential_5/dense_16/Tensordot/transposeÇ
;transformer_block_5/sequential_5/dense_16/Tensordot/ReshapeReshapeAtransformer_block_5/sequential_5/dense_16/Tensordot/transpose:y:0Btransformer_block_5/sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2=
;transformer_block_5/sequential_5/dense_16/Tensordot/ReshapeÆ
:transformer_block_5/sequential_5/dense_16/Tensordot/MatMulMatMulDtransformer_block_5/sequential_5/dense_16/Tensordot/Reshape:output:0Jtransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2<
:transformer_block_5/sequential_5/dense_16/Tensordot/MatMulÄ
;transformer_block_5/sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
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
<transformer_block_5/sequential_5/dense_16/Tensordot/concat_1¹
3transformer_block_5/sequential_5/dense_16/TensordotReshapeDtransformer_block_5/sequential_5/dense_16/Tensordot/MatMul:product:0Etransformer_block_5/sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 25
3transformer_block_5/sequential_5/dense_16/Tensordot
@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_5_sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp°
1transformer_block_5/sequential_5/dense_16/BiasAddBiasAdd<transformer_block_5/sequential_5/dense_16/Tensordot:output:0Htransformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 23
1transformer_block_5/sequential_5/dense_16/BiasAddÛ
.transformer_block_5/sequential_5/dense_16/ReluRelu:transformer_block_5/sequential_5/dense_16/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 20
.transformer_block_5/sequential_5/dense_16/Relu
Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_5_sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:  *
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
9transformer_block_5/sequential_5/dense_17/Tensordot/stackÅ
=transformer_block_5/sequential_5/dense_17/Tensordot/transpose	Transpose<transformer_block_5/sequential_5/dense_16/Relu:activations:0Ctransformer_block_5/sequential_5/dense_17/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2?
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
<transformer_block_5/sequential_5/dense_17/Tensordot/concat_1¹
3transformer_block_5/sequential_5/dense_17/TensordotReshapeDtransformer_block_5/sequential_5/dense_17/Tensordot/MatMul:product:0Etransformer_block_5/sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 25
3transformer_block_5/sequential_5/dense_17/Tensordot
@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_5_sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp°
1transformer_block_5/sequential_5/dense_17/BiasAddBiasAdd<transformer_block_5/sequential_5/dense_17/Tensordot:output:0Htransformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 23
1transformer_block_5/sequential_5/dense_17/BiasAdd¡
,transformer_block_5/dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2.
,transformer_block_5/dropout_15/dropout/Const
*transformer_block_5/dropout_15/dropout/MulMul:transformer_block_5/sequential_5/dense_17/BiasAdd:output:05transformer_block_5/dropout_15/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2,
*transformer_block_5/dropout_15/dropout/MulÆ
,transformer_block_5/dropout_15/dropout/ShapeShape:transformer_block_5/sequential_5/dense_17/BiasAdd:output:0*
T0*
_output_shapes
:2.
,transformer_block_5/dropout_15/dropout/Shape¯
Ctransformer_block_5/dropout_15/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_5/dropout_15/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
dtype0*

seed**
seed22E
Ctransformer_block_5/dropout_15/dropout/random_uniform/RandomUniform³
5transformer_block_5/dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=27
5transformer_block_5/dropout_15/dropout/GreaterEqual/y¿
3transformer_block_5/dropout_15/dropout/GreaterEqualGreaterEqualLtransformer_block_5/dropout_15/dropout/random_uniform/RandomUniform:output:0>transformer_block_5/dropout_15/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 25
3transformer_block_5/dropout_15/dropout/GreaterEqualá
+transformer_block_5/dropout_15/dropout/CastCast7transformer_block_5/dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2-
+transformer_block_5/dropout_15/dropout/Castû
,transformer_block_5/dropout_15/dropout/Mul_1Mul.transformer_block_5/dropout_15/dropout/Mul:z:0/transformer_block_5/dropout_15/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2.
,transformer_block_5/dropout_15/dropout/Mul_1è
transformer_block_5/add_1AddV2>transformer_block_5/layer_normalization_10/batchnorm/add_1:z:00transformer_block_5/dropout_15/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
transformer_block_5/add_1à
Itransformer_block_5/layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_5/layer_normalization_11/moments/mean/reduction_indicesµ
7transformer_block_5/layer_normalization_11/moments/meanMeantransformer_block_5/add_1:z:0Rtransformer_block_5/layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(29
7transformer_block_5/layer_normalization_11/moments/mean
?transformer_block_5/layer_normalization_11/moments/StopGradientStopGradient@transformer_block_5/layer_normalization_11/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2A
?transformer_block_5/layer_normalization_11/moments/StopGradientÁ
Dtransformer_block_5/layer_normalization_11/moments/SquaredDifferenceSquaredDifferencetransformer_block_5/add_1:z:0Htransformer_block_5/layer_normalization_11/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2F
Dtransformer_block_5/layer_normalization_11/moments/SquaredDifferenceè
Mtransformer_block_5/layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_5/layer_normalization_11/moments/variance/reduction_indicesì
;transformer_block_5/layer_normalization_11/moments/varianceMeanHtransformer_block_5/layer_normalization_11/moments/SquaredDifference:z:0Vtransformer_block_5/layer_normalization_11/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2=
;transformer_block_5/layer_normalization_11/moments/variance½
:transformer_block_5/layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752<
:transformer_block_5/layer_normalization_11/batchnorm/add/y¿
8transformer_block_5/layer_normalization_11/batchnorm/addAddV2Dtransformer_block_5/layer_normalization_11/moments/variance:output:0Ctransformer_block_5/layer_normalization_11/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2:
8transformer_block_5/layer_normalization_11/batchnorm/addö
:transformer_block_5/layer_normalization_11/batchnorm/RsqrtRsqrt<transformer_block_5/layer_normalization_11/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2<
:transformer_block_5/layer_normalization_11/batchnorm/Rsqrt
Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_5_layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpÃ
8transformer_block_5/layer_normalization_11/batchnorm/mulMul>transformer_block_5/layer_normalization_11/batchnorm/Rsqrt:y:0Otransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2:
8transformer_block_5/layer_normalization_11/batchnorm/mul
:transformer_block_5/layer_normalization_11/batchnorm/mul_1Multransformer_block_5/add_1:z:0<transformer_block_5/layer_normalization_11/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2<
:transformer_block_5/layer_normalization_11/batchnorm/mul_1¶
:transformer_block_5/layer_normalization_11/batchnorm/mul_2Mul@transformer_block_5/layer_normalization_11/moments/mean:output:0<transformer_block_5/layer_normalization_11/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2<
:transformer_block_5/layer_normalization_11/batchnorm/mul_2
Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_5_layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp¿
8transformer_block_5/layer_normalization_11/batchnorm/subSubKtransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp:value:0>transformer_block_5/layer_normalization_11/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2:
8transformer_block_5/layer_normalization_11/batchnorm/sub¶
:transformer_block_5/layer_normalization_11/batchnorm/add_1AddV2>transformer_block_5/layer_normalization_11/batchnorm/mul_1:z:0<transformer_block_5/layer_normalization_11/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2<
:transformer_block_5/layer_normalization_11/batchnorm/add_1¨
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indicesø
global_average_pooling1d_2/MeanMean>transformer_block_5/layer_normalization_11/batchnorm/add_1:z:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
global_average_pooling1d_2/Meanx
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axisË
concatenate_2/concatConcatV2(global_average_pooling1d_2/Mean:output:0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
concatenate_2/concat¨
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:( *
dtype02 
dense_18/MatMul/ReadVariableOp¥
dense_18/MatMulMatMulconcatenate_2/concat:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_18/MatMul§
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_18/BiasAdd/ReadVariableOp¥
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_18/BiasAdds
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
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
:ÿÿÿÿÿÿÿÿÿ 2
dropout_16/dropout/Mul
dropout_16/dropout/ShapeShapedense_18/Relu:activations:0*
T0*
_output_shapes
:2
dropout_16/dropout/Shapeî
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seed**
seed221
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
:ÿÿÿÿÿÿÿÿÿ 2!
dropout_16/dropout/GreaterEqual 
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_16/dropout/Cast¦
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_16/dropout/Mul_1¨
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_19/MatMul/ReadVariableOp¤
dense_19/MatMulMatMuldropout_16/dropout/Mul_1:z:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_19/MatMul§
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_19/BiasAdd/ReadVariableOp¥
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_19/BiasAdds
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
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
:ÿÿÿÿÿÿÿÿÿ 2
dropout_17/dropout/Mul
dropout_17/dropout/ShapeShapedense_19/Relu:activations:0*
T0*
_output_shapes
:2
dropout_17/dropout/Shapeî
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0*

seed**
seed221
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
:ÿÿÿÿÿÿÿÿÿ 2!
dropout_17/dropout/GreaterEqual 
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_17/dropout/Cast¦
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout_17/dropout/Mul_1¨
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

: *
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
dense_20/BiasAdd¼
IdentityIdentitydense_20/BiasAdd:output:0:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_4/AssignMovingAvg/ReadVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp3^batch_normalization_4/batchnorm/mul/ReadVariableOp:^batch_normalization_5/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_5/AssignMovingAvg/ReadVariableOp<^batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp3^batch_normalization_5/batchnorm/mul/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp<^token_and_position_embedding_2/embedding_4/embedding_lookup<^token_and_position_embedding_2/embedding_5/embedding_lookupD^transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpH^transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpD^transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpH^transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpO^transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpY^transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_5/multi_head_attention_5/key/add/ReadVariableOpL^transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpD^transformer_block_5/multi_head_attention_5/query/add/ReadVariableOpN^transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpD^transformer_block_5/multi_head_attention_5/value/add/ReadVariableOpN^transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpA^transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpC^transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpA^transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpC^transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ¼i:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::2v
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2v
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_5/AssignMovingAvg/ReadVariableOp4batch_normalization_5/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_5/batchnorm/ReadVariableOp.batch_normalization_5/batchnorm/ReadVariableOp2h
2batch_normalization_5/batchnorm/mul/ReadVariableOp2batch_normalization_5/batchnorm/mul/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2B
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
:ÿÿÿÿÿÿÿÿÿ¼i
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ì
¨
5__inference_batch_normalization_4_layer_call_fn_50219

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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_476652
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

q
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_47999

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
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
E__inference_dropout_16_layer_call_and_return_conditional_losses_50811

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
:ÿÿÿÿÿÿÿÿÿ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÀ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
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
:ÿÿÿÿÿÿÿÿÿ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

ö
C__inference_conv1d_2_layer_call_and_return_conditional_losses_48063

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
:ÿÿÿÿÿÿÿÿÿ¼i 2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
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
:  2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *
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
:ÿÿÿÿÿÿÿÿÿ¼i 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ¼i ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 
 
_user_specified_nameinputs
ô
j
N__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_47530

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
ä0
Å
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_47772

inputs
assignmovingavg_47747
assignmovingavg_1_47753)
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
moments/Squeeze_1Ë
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/47747*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_47747*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpð
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/47747*
_output_shapes
: 2
AssignMovingAvg/subç
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/47747*
_output_shapes
: 2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_47747AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/47747*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÑ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/47753*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_47753*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpú
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/47753*
_output_shapes
: 2
AssignMovingAvg_1/subñ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/47753*
_output_shapes
: 2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_47753AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/47753*
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
ë

Y__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_50021
x&
"embedding_5_embedding_lookup_50008&
"embedding_4_embedding_lookup_50014
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
range­
embedding_5/embedding_lookupResourceGather"embedding_5_embedding_lookup_50008range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_5/embedding_lookup/50008*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02
embedding_5/embedding_lookup
%embedding_5/embedding_lookup/IdentityIdentity%embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_5/embedding_lookup/50008*'
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
:ÿÿÿÿÿÿÿÿÿ¼i2
embedding_4/Cast¸
embedding_4/embedding_lookupResourceGather"embedding_4_embedding_lookup_50014embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_4/embedding_lookup/50014*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *
dtype02
embedding_4/embedding_lookup
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_4/embedding_lookup/50014*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2'
%embedding_4/embedding_lookup/IdentityÅ
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2)
'embedding_4/embedding_lookup/Identity_1®
addAddV20embedding_4/embedding_lookup/Identity_1:output:00embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2
add
IdentityIdentityadd:z:0^embedding_4/embedding_lookup^embedding_5/embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¼i::2<
embedding_4/embedding_lookupembedding_4/embedding_lookup2<
embedding_5/embedding_lookupembedding_5/embedding_lookup:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i

_user_specified_namex
¶S
ø
B__inference_model_2_layer_call_and_return_conditional_losses_49011

inputs
inputs_1(
$token_and_position_embedding_2_48927(
$token_and_position_embedding_2_48929
conv1d_2_48932
conv1d_2_48934
batch_normalization_4_48939
batch_normalization_4_48941
batch_normalization_4_48943
batch_normalization_4_48945
batch_normalization_5_48948
batch_normalization_5_48950
batch_normalization_5_48952
batch_normalization_5_48954
transformer_block_5_48958
transformer_block_5_48960
transformer_block_5_48962
transformer_block_5_48964
transformer_block_5_48966
transformer_block_5_48968
transformer_block_5_48970
transformer_block_5_48972
transformer_block_5_48974
transformer_block_5_48976
transformer_block_5_48978
transformer_block_5_48980
transformer_block_5_48982
transformer_block_5_48984
transformer_block_5_48986
transformer_block_5_48988
dense_18_48993
dense_18_48995
dense_19_48999
dense_19_49001
dense_20_49005
dense_20_49007
identity¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢ dense_20/StatefulPartitionedCall¢"dropout_16/StatefulPartitionedCall¢"dropout_17/StatefulPartitionedCall¢6token_and_position_embedding_2/StatefulPartitionedCall¢+transformer_block_5/StatefulPartitionedCall
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs$token_and_position_embedding_2_48927$token_and_position_embedding_2_48929*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *b
f]R[
Y__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_4803128
6token_and_position_embedding_2/StatefulPartitionedCallÒ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0conv1d_2_48932conv1d_2_48934*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_480632"
 conv1d_2/StatefulPartitionedCallµ
#average_pooling1d_5/PartitionedCallPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_475302%
#average_pooling1d_5/PartitionedCall
#average_pooling1d_4/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_475152%
#average_pooling1d_4/PartitionedCall¼
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_4/PartitionedCall:output:0batch_normalization_4_48939batch_normalization_4_48941batch_normalization_4_48943batch_normalization_4_48945*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_481162/
-batch_normalization_4/StatefulPartitionedCall¼
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_5/PartitionedCall:output:0batch_normalization_5_48948batch_normalization_5_48950batch_normalization_5_48952batch_normalization_5_48954*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_482072/
-batch_normalization_5/StatefulPartitionedCall»
add_2/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:06batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_482692
add_2/PartitionedCallþ
+transformer_block_5/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0transformer_block_5_48958transformer_block_5_48960transformer_block_5_48962transformer_block_5_48964transformer_block_5_48966transformer_block_5_48968transformer_block_5_48970transformer_block_5_48972transformer_block_5_48974transformer_block_5_48976transformer_block_5_48978transformer_block_5_48980transformer_block_5_48982transformer_block_5_48984transformer_block_5_48986transformer_block_5_48988*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_transformer_block_5_layer_call_and_return_conditional_losses_484262-
+transformer_block_5/StatefulPartitionedCallº
*global_average_pooling1d_2/PartitionedCallPartitionedCall4transformer_block_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_486672,
*global_average_pooling1d_2/PartitionedCall
concatenate_2/PartitionedCallPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_486812
concatenate_2/PartitionedCall´
 dense_18/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_18_48993dense_18_48995*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_487012"
 dense_18/StatefulPartitionedCall
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_487292$
"dropout_16/StatefulPartitionedCall¹
 dense_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0dense_19_48999dense_19_49001*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_19_layer_call_and_return_conditional_losses_487582"
 dense_19/StatefulPartitionedCall¼
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_487862$
"dropout_17/StatefulPartitionedCall¹
 dense_20/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_20_49005dense_20_49007*
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
GPU2*0J 8 *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_488142"
 dense_20/StatefulPartitionedCall
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_5/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ¼i:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_5/StatefulPartitionedCall+transformer_block_5/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

F
*__inference_dropout_16_layer_call_fn_50826

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_487342
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¾
òC
!__inference__traced_restore_51751
file_prefix$
 assignvariableop_conv1d_2_kernel$
 assignvariableop_1_conv1d_2_bias2
.assignvariableop_2_batch_normalization_4_gamma1
-assignvariableop_3_batch_normalization_4_beta8
4assignvariableop_4_batch_normalization_4_moving_mean<
8assignvariableop_5_batch_normalization_4_moving_variance2
.assignvariableop_6_batch_normalization_5_gamma1
-assignvariableop_7_batch_normalization_5_beta8
4assignvariableop_8_batch_normalization_5_moving_mean<
8assignvariableop_9_batch_normalization_5_moving_variance'
#assignvariableop_10_dense_18_kernel%
!assignvariableop_11_dense_18_bias'
#assignvariableop_12_dense_19_kernel%
!assignvariableop_13_dense_19_bias'
#assignvariableop_14_dense_20_kernel%
!assignvariableop_15_dense_20_bias
assignvariableop_16_beta_1
assignvariableop_17_beta_2
assignvariableop_18_decay%
!assignvariableop_19_learning_rate#
assignvariableop_20_adamax_iterM
Iassignvariableop_21_token_and_position_embedding_2_embedding_4_embeddingsM
Iassignvariableop_22_token_and_position_embedding_2_embedding_5_embeddingsO
Kassignvariableop_23_transformer_block_5_multi_head_attention_5_query_kernelM
Iassignvariableop_24_transformer_block_5_multi_head_attention_5_query_biasM
Iassignvariableop_25_transformer_block_5_multi_head_attention_5_key_kernelK
Gassignvariableop_26_transformer_block_5_multi_head_attention_5_key_biasO
Kassignvariableop_27_transformer_block_5_multi_head_attention_5_value_kernelM
Iassignvariableop_28_transformer_block_5_multi_head_attention_5_value_biasZ
Vassignvariableop_29_transformer_block_5_multi_head_attention_5_attention_output_kernelX
Tassignvariableop_30_transformer_block_5_multi_head_attention_5_attention_output_bias'
#assignvariableop_31_dense_16_kernel%
!assignvariableop_32_dense_16_bias'
#assignvariableop_33_dense_17_kernel%
!assignvariableop_34_dense_17_biasH
Dassignvariableop_35_transformer_block_5_layer_normalization_10_gammaG
Cassignvariableop_36_transformer_block_5_layer_normalization_10_betaH
Dassignvariableop_37_transformer_block_5_layer_normalization_11_gammaG
Cassignvariableop_38_transformer_block_5_layer_normalization_11_beta
assignvariableop_39_total
assignvariableop_40_count0
,assignvariableop_41_adamax_conv1d_2_kernel_m.
*assignvariableop_42_adamax_conv1d_2_bias_m<
8assignvariableop_43_adamax_batch_normalization_4_gamma_m;
7assignvariableop_44_adamax_batch_normalization_4_beta_m<
8assignvariableop_45_adamax_batch_normalization_5_gamma_m;
7assignvariableop_46_adamax_batch_normalization_5_beta_m0
,assignvariableop_47_adamax_dense_18_kernel_m.
*assignvariableop_48_adamax_dense_18_bias_m0
,assignvariableop_49_adamax_dense_19_kernel_m.
*assignvariableop_50_adamax_dense_19_bias_m0
,assignvariableop_51_adamax_dense_20_kernel_m.
*assignvariableop_52_adamax_dense_20_bias_mV
Rassignvariableop_53_adamax_token_and_position_embedding_2_embedding_4_embeddings_mV
Rassignvariableop_54_adamax_token_and_position_embedding_2_embedding_5_embeddings_mX
Tassignvariableop_55_adamax_transformer_block_5_multi_head_attention_5_query_kernel_mV
Rassignvariableop_56_adamax_transformer_block_5_multi_head_attention_5_query_bias_mV
Rassignvariableop_57_adamax_transformer_block_5_multi_head_attention_5_key_kernel_mT
Passignvariableop_58_adamax_transformer_block_5_multi_head_attention_5_key_bias_mX
Tassignvariableop_59_adamax_transformer_block_5_multi_head_attention_5_value_kernel_mV
Rassignvariableop_60_adamax_transformer_block_5_multi_head_attention_5_value_bias_mc
_assignvariableop_61_adamax_transformer_block_5_multi_head_attention_5_attention_output_kernel_ma
]assignvariableop_62_adamax_transformer_block_5_multi_head_attention_5_attention_output_bias_m0
,assignvariableop_63_adamax_dense_16_kernel_m.
*assignvariableop_64_adamax_dense_16_bias_m0
,assignvariableop_65_adamax_dense_17_kernel_m.
*assignvariableop_66_adamax_dense_17_bias_mQ
Massignvariableop_67_adamax_transformer_block_5_layer_normalization_10_gamma_mP
Lassignvariableop_68_adamax_transformer_block_5_layer_normalization_10_beta_mQ
Massignvariableop_69_adamax_transformer_block_5_layer_normalization_11_gamma_mP
Lassignvariableop_70_adamax_transformer_block_5_layer_normalization_11_beta_m0
,assignvariableop_71_adamax_conv1d_2_kernel_v.
*assignvariableop_72_adamax_conv1d_2_bias_v<
8assignvariableop_73_adamax_batch_normalization_4_gamma_v;
7assignvariableop_74_adamax_batch_normalization_4_beta_v<
8assignvariableop_75_adamax_batch_normalization_5_gamma_v;
7assignvariableop_76_adamax_batch_normalization_5_beta_v0
,assignvariableop_77_adamax_dense_18_kernel_v.
*assignvariableop_78_adamax_dense_18_bias_v0
,assignvariableop_79_adamax_dense_19_kernel_v.
*assignvariableop_80_adamax_dense_19_bias_v0
,assignvariableop_81_adamax_dense_20_kernel_v.
*assignvariableop_82_adamax_dense_20_bias_vV
Rassignvariableop_83_adamax_token_and_position_embedding_2_embedding_4_embeddings_vV
Rassignvariableop_84_adamax_token_and_position_embedding_2_embedding_5_embeddings_vX
Tassignvariableop_85_adamax_transformer_block_5_multi_head_attention_5_query_kernel_vV
Rassignvariableop_86_adamax_transformer_block_5_multi_head_attention_5_query_bias_vV
Rassignvariableop_87_adamax_transformer_block_5_multi_head_attention_5_key_kernel_vT
Passignvariableop_88_adamax_transformer_block_5_multi_head_attention_5_key_bias_vX
Tassignvariableop_89_adamax_transformer_block_5_multi_head_attention_5_value_kernel_vV
Rassignvariableop_90_adamax_transformer_block_5_multi_head_attention_5_value_bias_vc
_assignvariableop_91_adamax_transformer_block_5_multi_head_attention_5_attention_output_kernel_va
]assignvariableop_92_adamax_transformer_block_5_multi_head_attention_5_attention_output_bias_v0
,assignvariableop_93_adamax_dense_16_kernel_v.
*assignvariableop_94_adamax_dense_16_bias_v0
,assignvariableop_95_adamax_dense_17_kernel_v.
*assignvariableop_96_adamax_dense_17_bias_vQ
Massignvariableop_97_adamax_transformer_block_5_layer_normalization_10_gamma_vP
Lassignvariableop_98_adamax_transformer_block_5_layer_normalization_10_beta_vQ
Massignvariableop_99_adamax_transformer_block_5_layer_normalization_11_gamma_vQ
Massignvariableop_100_adamax_transformer_block_5_layer_normalization_11_beta_v
identity_102¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_100¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98¢AssignVariableOp_99Ú3
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*æ2
valueÜ2BÙ2fB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÝ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*á
value×BÔfB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices¬
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*®
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*t
dtypesj
h2f	2
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

Identity_2³
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_4_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3²
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_4_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¹
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_4_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5½
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_4_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6³
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_5_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7²
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_5_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¹
AssignVariableOp_8AssignVariableOp4assignvariableop_8_batch_normalization_5_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9½
AssignVariableOp_9AssignVariableOp8assignvariableop_9_batch_normalization_5_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10«
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_18_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11©
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_18_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_19_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_19_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_20_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_20_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¢
AssignVariableOp_16AssignVariableOpassignvariableop_16_beta_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¢
AssignVariableOp_17AssignVariableOpassignvariableop_17_beta_2Identity_17:output:0"/device:CPU:0*
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
T0	*
_output_shapes
:2
Identity_20§
AssignVariableOp_20AssignVariableOpassignvariableop_20_adamax_iterIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ñ
AssignVariableOp_21AssignVariableOpIassignvariableop_21_token_and_position_embedding_2_embedding_4_embeddingsIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ñ
AssignVariableOp_22AssignVariableOpIassignvariableop_22_token_and_position_embedding_2_embedding_5_embeddingsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ó
AssignVariableOp_23AssignVariableOpKassignvariableop_23_transformer_block_5_multi_head_attention_5_query_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ñ
AssignVariableOp_24AssignVariableOpIassignvariableop_24_transformer_block_5_multi_head_attention_5_query_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ñ
AssignVariableOp_25AssignVariableOpIassignvariableop_25_transformer_block_5_multi_head_attention_5_key_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ï
AssignVariableOp_26AssignVariableOpGassignvariableop_26_transformer_block_5_multi_head_attention_5_key_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ó
AssignVariableOp_27AssignVariableOpKassignvariableop_27_transformer_block_5_multi_head_attention_5_value_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ñ
AssignVariableOp_28AssignVariableOpIassignvariableop_28_transformer_block_5_multi_head_attention_5_value_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Þ
AssignVariableOp_29AssignVariableOpVassignvariableop_29_transformer_block_5_multi_head_attention_5_attention_output_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ü
AssignVariableOp_30AssignVariableOpTassignvariableop_30_transformer_block_5_multi_head_attention_5_attention_output_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31«
AssignVariableOp_31AssignVariableOp#assignvariableop_31_dense_16_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32©
AssignVariableOp_32AssignVariableOp!assignvariableop_32_dense_16_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33«
AssignVariableOp_33AssignVariableOp#assignvariableop_33_dense_17_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34©
AssignVariableOp_34AssignVariableOp!assignvariableop_34_dense_17_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Ì
AssignVariableOp_35AssignVariableOpDassignvariableop_35_transformer_block_5_layer_normalization_10_gammaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ë
AssignVariableOp_36AssignVariableOpCassignvariableop_36_transformer_block_5_layer_normalization_10_betaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ì
AssignVariableOp_37AssignVariableOpDassignvariableop_37_transformer_block_5_layer_normalization_11_gammaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ë
AssignVariableOp_38AssignVariableOpCassignvariableop_38_transformer_block_5_layer_normalization_11_betaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¡
AssignVariableOp_39AssignVariableOpassignvariableop_39_totalIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40¡
AssignVariableOp_40AssignVariableOpassignvariableop_40_countIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41´
AssignVariableOp_41AssignVariableOp,assignvariableop_41_adamax_conv1d_2_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42²
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adamax_conv1d_2_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43À
AssignVariableOp_43AssignVariableOp8assignvariableop_43_adamax_batch_normalization_4_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¿
AssignVariableOp_44AssignVariableOp7assignvariableop_44_adamax_batch_normalization_4_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45À
AssignVariableOp_45AssignVariableOp8assignvariableop_45_adamax_batch_normalization_5_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46¿
AssignVariableOp_46AssignVariableOp7assignvariableop_46_adamax_batch_normalization_5_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47´
AssignVariableOp_47AssignVariableOp,assignvariableop_47_adamax_dense_18_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48²
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adamax_dense_18_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49´
AssignVariableOp_49AssignVariableOp,assignvariableop_49_adamax_dense_19_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50²
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adamax_dense_19_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51´
AssignVariableOp_51AssignVariableOp,assignvariableop_51_adamax_dense_20_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52²
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adamax_dense_20_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Ú
AssignVariableOp_53AssignVariableOpRassignvariableop_53_adamax_token_and_position_embedding_2_embedding_4_embeddings_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54Ú
AssignVariableOp_54AssignVariableOpRassignvariableop_54_adamax_token_and_position_embedding_2_embedding_5_embeddings_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55Ü
AssignVariableOp_55AssignVariableOpTassignvariableop_55_adamax_transformer_block_5_multi_head_attention_5_query_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Ú
AssignVariableOp_56AssignVariableOpRassignvariableop_56_adamax_transformer_block_5_multi_head_attention_5_query_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57Ú
AssignVariableOp_57AssignVariableOpRassignvariableop_57_adamax_transformer_block_5_multi_head_attention_5_key_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58Ø
AssignVariableOp_58AssignVariableOpPassignvariableop_58_adamax_transformer_block_5_multi_head_attention_5_key_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Ü
AssignVariableOp_59AssignVariableOpTassignvariableop_59_adamax_transformer_block_5_multi_head_attention_5_value_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Ú
AssignVariableOp_60AssignVariableOpRassignvariableop_60_adamax_transformer_block_5_multi_head_attention_5_value_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61ç
AssignVariableOp_61AssignVariableOp_assignvariableop_61_adamax_transformer_block_5_multi_head_attention_5_attention_output_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62å
AssignVariableOp_62AssignVariableOp]assignvariableop_62_adamax_transformer_block_5_multi_head_attention_5_attention_output_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63´
AssignVariableOp_63AssignVariableOp,assignvariableop_63_adamax_dense_16_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64²
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adamax_dense_16_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65´
AssignVariableOp_65AssignVariableOp,assignvariableop_65_adamax_dense_17_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66²
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adamax_dense_17_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67Õ
AssignVariableOp_67AssignVariableOpMassignvariableop_67_adamax_transformer_block_5_layer_normalization_10_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68Ô
AssignVariableOp_68AssignVariableOpLassignvariableop_68_adamax_transformer_block_5_layer_normalization_10_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69Õ
AssignVariableOp_69AssignVariableOpMassignvariableop_69_adamax_transformer_block_5_layer_normalization_11_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Ô
AssignVariableOp_70AssignVariableOpLassignvariableop_70_adamax_transformer_block_5_layer_normalization_11_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71´
AssignVariableOp_71AssignVariableOp,assignvariableop_71_adamax_conv1d_2_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72²
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adamax_conv1d_2_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73À
AssignVariableOp_73AssignVariableOp8assignvariableop_73_adamax_batch_normalization_4_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74¿
AssignVariableOp_74AssignVariableOp7assignvariableop_74_adamax_batch_normalization_4_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75À
AssignVariableOp_75AssignVariableOp8assignvariableop_75_adamax_batch_normalization_5_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76¿
AssignVariableOp_76AssignVariableOp7assignvariableop_76_adamax_batch_normalization_5_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77´
AssignVariableOp_77AssignVariableOp,assignvariableop_77_adamax_dense_18_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78²
AssignVariableOp_78AssignVariableOp*assignvariableop_78_adamax_dense_18_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79´
AssignVariableOp_79AssignVariableOp,assignvariableop_79_adamax_dense_19_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80²
AssignVariableOp_80AssignVariableOp*assignvariableop_80_adamax_dense_19_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81´
AssignVariableOp_81AssignVariableOp,assignvariableop_81_adamax_dense_20_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82²
AssignVariableOp_82AssignVariableOp*assignvariableop_82_adamax_dense_20_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83Ú
AssignVariableOp_83AssignVariableOpRassignvariableop_83_adamax_token_and_position_embedding_2_embedding_4_embeddings_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84Ú
AssignVariableOp_84AssignVariableOpRassignvariableop_84_adamax_token_and_position_embedding_2_embedding_5_embeddings_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85Ü
AssignVariableOp_85AssignVariableOpTassignvariableop_85_adamax_transformer_block_5_multi_head_attention_5_query_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86Ú
AssignVariableOp_86AssignVariableOpRassignvariableop_86_adamax_transformer_block_5_multi_head_attention_5_query_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87Ú
AssignVariableOp_87AssignVariableOpRassignvariableop_87_adamax_transformer_block_5_multi_head_attention_5_key_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88Ø
AssignVariableOp_88AssignVariableOpPassignvariableop_88_adamax_transformer_block_5_multi_head_attention_5_key_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89Ü
AssignVariableOp_89AssignVariableOpTassignvariableop_89_adamax_transformer_block_5_multi_head_attention_5_value_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90Ú
AssignVariableOp_90AssignVariableOpRassignvariableop_90_adamax_transformer_block_5_multi_head_attention_5_value_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91ç
AssignVariableOp_91AssignVariableOp_assignvariableop_91_adamax_transformer_block_5_multi_head_attention_5_attention_output_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92å
AssignVariableOp_92AssignVariableOp]assignvariableop_92_adamax_transformer_block_5_multi_head_attention_5_attention_output_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93´
AssignVariableOp_93AssignVariableOp,assignvariableop_93_adamax_dense_16_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94²
AssignVariableOp_94AssignVariableOp*assignvariableop_94_adamax_dense_16_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95´
AssignVariableOp_95AssignVariableOp,assignvariableop_95_adamax_dense_17_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96²
AssignVariableOp_96AssignVariableOp*assignvariableop_96_adamax_dense_17_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97Õ
AssignVariableOp_97AssignVariableOpMassignvariableop_97_adamax_transformer_block_5_layer_normalization_10_gamma_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98Ô
AssignVariableOp_98AssignVariableOpLassignvariableop_98_adamax_transformer_block_5_layer_normalization_10_beta_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99Õ
AssignVariableOp_99AssignVariableOpMassignvariableop_99_adamax_transformer_block_5_layer_normalization_11_gamma_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100Ø
AssignVariableOp_100AssignVariableOpMassignvariableop_100_adamax_transformer_block_5_layer_normalization_11_beta_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1009
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_101Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_101
Identity_102IdentityIdentity_101:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_102"%
identity_102Identity_102:output:0*«
_input_shapes
: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002*
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

á
N__inference_transformer_block_5_layer_call_and_return_conditional_losses_48426

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
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_5/query/einsum/EinsumEinsuminputsAmulti_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2,
*multi_head_attention_5/query/einsum/EinsumÛ
/multi_head_attention_5/query/add/ReadVariableOpReadVariableOp8multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/query/add/ReadVariableOpö
 multi_head_attention_5/query/addAddV23multi_head_attention_5/query/einsum/Einsum:output:07multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2"
 multi_head_attention_5/query/add÷
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_5/key/einsum/EinsumEinsuminputs?multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2*
(multi_head_attention_5/key/einsum/EinsumÕ
-multi_head_attention_5/key/add/ReadVariableOpReadVariableOp6multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_5/key/add/ReadVariableOpî
multi_head_attention_5/key/addAddV21multi_head_attention_5/key/einsum/Einsum:output:05multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2 
multi_head_attention_5/key/addý
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_5/value/einsum/EinsumEinsuminputsAmulti_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2,
*multi_head_attention_5/value/einsum/EinsumÛ
/multi_head_attention_5/value/add/ReadVariableOpReadVariableOp8multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/value/add/ReadVariableOpö
 multi_head_attention_5/value/addAddV23multi_head_attention_5/value/einsum/Einsum:output:07multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2"
 multi_head_attention_5/value/add
multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_5/Mul/yÇ
multi_head_attention_5/MulMul$multi_head_attention_5/query/add:z:0%multi_head_attention_5/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
multi_head_attention_5/Mulþ
$multi_head_attention_5/einsum/EinsumEinsum"multi_head_attention_5/key/add:z:0multi_head_attention_5/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ*
equationaecd,abcd->acbe2&
$multi_head_attention_5/einsum/EinsumÆ
&multi_head_attention_5/softmax/SoftmaxSoftmax-multi_head_attention_5/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2(
&multi_head_attention_5/softmax/Softmax¡
,multi_head_attention_5/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,multi_head_attention_5/dropout/dropout/Const
*multi_head_attention_5/dropout/dropout/MulMul0multi_head_attention_5/softmax/Softmax:softmax:05multi_head_attention_5/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2,
*multi_head_attention_5/dropout/dropout/Mul¼
,multi_head_attention_5/dropout/dropout/ShapeShape0multi_head_attention_5/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_5/dropout/dropout/Shape§
Cmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_5/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ*
dtype0*

seed*2E
Cmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniform³
5multi_head_attention_5/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_5/dropout/dropout/GreaterEqual/yÄ
3multi_head_attention_5/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_5/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ25
3multi_head_attention_5/dropout/dropout/GreaterEqualæ
+multi_head_attention_5/dropout/dropout/CastCast7multi_head_attention_5/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2-
+multi_head_attention_5/dropout/dropout/Cast
,multi_head_attention_5/dropout/dropout/Mul_1Mul.multi_head_attention_5/dropout/dropout/Mul:z:0/multi_head_attention_5/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2.
,multi_head_attention_5/dropout/dropout/Mul_1
&multi_head_attention_5/einsum_1/EinsumEinsum0multi_head_attention_5/dropout/dropout/Mul_1:z:0$multi_head_attention_5/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationacbe,aecd->abcd2(
&multi_head_attention_5/einsum_1/Einsum
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpÔ
5multi_head_attention_5/attention_output/einsum/EinsumEinsum/multi_head_attention_5/einsum_1/Einsum:output:0Lmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabcd,cde->abe27
5multi_head_attention_5/attention_output/einsum/Einsumø
:multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_5/attention_output/add/ReadVariableOp
+multi_head_attention_5/attention_output/addAddV2>multi_head_attention_5/attention_output/einsum/Einsum:output:0Bmulti_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2-
+multi_head_attention_5/attention_output/addy
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_14/dropout/ConstÂ
dropout_14/dropout/MulMul/multi_head_attention_5/attention_output/add:z:0!dropout_14/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dropout_14/dropout/Mul
dropout_14/dropout/ShapeShape/multi_head_attention_5/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_14/dropout/Shapeó
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
dtype0*

seed**
seed221
/dropout_14/dropout/random_uniform/RandomUniform
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_14/dropout/GreaterEqual/yï
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2!
dropout_14/dropout/GreaterEqual¥
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dropout_14/dropout/Cast«
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dropout_14/dropout/Mul_1p
addAddV2inputsdropout_14/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
add¸
5layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_10/moments/mean/reduction_indicesã
#layer_normalization_10/moments/meanMeanadd:z:0>layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2%
#layer_normalization_10/moments/meanÏ
+layer_normalization_10/moments/StopGradientStopGradient,layer_normalization_10/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2-
+layer_normalization_10/moments/StopGradientï
0layer_normalization_10/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_10/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 22
0layer_normalization_10/moments/SquaredDifferenceÀ
9layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_10/moments/variance/reduction_indices
'layer_normalization_10/moments/varianceMean4layer_normalization_10/moments/SquaredDifference:z:0Blayer_normalization_10/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2)
'layer_normalization_10/moments/variance
&layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_10/batchnorm/add/yï
$layer_normalization_10/batchnorm/addAddV20layer_normalization_10/moments/variance:output:0/layer_normalization_10/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2&
$layer_normalization_10/batchnorm/addº
&layer_normalization_10/batchnorm/RsqrtRsqrt(layer_normalization_10/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2(
&layer_normalization_10/batchnorm/Rsqrtã
3layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_10/batchnorm/mul/ReadVariableOpó
$layer_normalization_10/batchnorm/mulMul*layer_normalization_10/batchnorm/Rsqrt:y:0;layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2&
$layer_normalization_10/batchnorm/mulÁ
&layer_normalization_10/batchnorm/mul_1Muladd:z:0(layer_normalization_10/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_10/batchnorm/mul_1æ
&layer_normalization_10/batchnorm/mul_2Mul,layer_normalization_10/moments/mean:output:0(layer_normalization_10/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_10/batchnorm/mul_2×
/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_10/batchnorm/ReadVariableOpï
$layer_normalization_10/batchnorm/subSub7layer_normalization_10/batchnorm/ReadVariableOp:value:0*layer_normalization_10/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2&
$layer_normalization_10/batchnorm/subæ
&layer_normalization_10/batchnorm/add_1AddV2*layer_normalization_10/batchnorm/mul_1:z:0(layer_normalization_10/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_10/batchnorm/add_1Ø
.sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

:  *
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
%sequential_5/dense_16/Tensordot/stack÷
)sequential_5/dense_16/Tensordot/transpose	Transpose*layer_normalization_10/batchnorm/add_1:z:0/sequential_5/dense_16/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2+
)sequential_5/dense_16/Tensordot/transpose÷
'sequential_5/dense_16/Tensordot/ReshapeReshape-sequential_5/dense_16/Tensordot/transpose:y:0.sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_5/dense_16/Tensordot/Reshapeö
&sequential_5/dense_16/Tensordot/MatMulMatMul0sequential_5/dense_16/Tensordot/Reshape:output:06sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_5/dense_16/Tensordot/MatMul
'sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
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
(sequential_5/dense_16/Tensordot/concat_1é
sequential_5/dense_16/TensordotReshape0sequential_5/dense_16/Tensordot/MatMul:product:01sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2!
sequential_5/dense_16/TensordotÎ
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOpà
sequential_5/dense_16/BiasAddBiasAdd(sequential_5/dense_16/Tensordot:output:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
sequential_5/dense_16/BiasAdd
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
sequential_5/dense_16/ReluØ
.sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:  *
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
%sequential_5/dense_17/Tensordot/stackõ
)sequential_5/dense_17/Tensordot/transpose	Transpose(sequential_5/dense_16/Relu:activations:0/sequential_5/dense_17/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2+
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
(sequential_5/dense_17/Tensordot/concat_1é
sequential_5/dense_17/TensordotReshape0sequential_5/dense_17/Tensordot/MatMul:product:01sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2!
sequential_5/dense_17/TensordotÎ
,sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_17/BiasAdd/ReadVariableOpà
sequential_5/dense_17/BiasAddBiasAdd(sequential_5/dense_17/Tensordot:output:04sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
sequential_5/dense_17/BiasAddy
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_15/dropout/Const¹
dropout_15/dropout/MulMul&sequential_5/dense_17/BiasAdd:output:0!dropout_15/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dropout_15/dropout/Mul
dropout_15/dropout/ShapeShape&sequential_5/dense_17/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_15/dropout/Shapeó
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
dtype0*

seed**
seed221
/dropout_15/dropout/random_uniform/RandomUniform
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_15/dropout/GreaterEqual/yï
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2!
dropout_15/dropout/GreaterEqual¥
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dropout_15/dropout/Cast«
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dropout_15/dropout/Mul_1
add_1AddV2*layer_normalization_10/batchnorm/add_1:z:0dropout_15/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
add_1¸
5layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_11/moments/mean/reduction_indiceså
#layer_normalization_11/moments/meanMean	add_1:z:0>layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2%
#layer_normalization_11/moments/meanÏ
+layer_normalization_11/moments/StopGradientStopGradient,layer_normalization_11/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2-
+layer_normalization_11/moments/StopGradientñ
0layer_normalization_11/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_11/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 22
0layer_normalization_11/moments/SquaredDifferenceÀ
9layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_11/moments/variance/reduction_indices
'layer_normalization_11/moments/varianceMean4layer_normalization_11/moments/SquaredDifference:z:0Blayer_normalization_11/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2)
'layer_normalization_11/moments/variance
&layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_11/batchnorm/add/yï
$layer_normalization_11/batchnorm/addAddV20layer_normalization_11/moments/variance:output:0/layer_normalization_11/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2&
$layer_normalization_11/batchnorm/addº
&layer_normalization_11/batchnorm/RsqrtRsqrt(layer_normalization_11/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2(
&layer_normalization_11/batchnorm/Rsqrtã
3layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_11/batchnorm/mul/ReadVariableOpó
$layer_normalization_11/batchnorm/mulMul*layer_normalization_11/batchnorm/Rsqrt:y:0;layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2&
$layer_normalization_11/batchnorm/mulÃ
&layer_normalization_11/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_11/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_11/batchnorm/mul_1æ
&layer_normalization_11/batchnorm/mul_2Mul,layer_normalization_11/moments/mean:output:0(layer_normalization_11/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_11/batchnorm/mul_2×
/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_11/batchnorm/ReadVariableOpï
$layer_normalization_11/batchnorm/subSub7layer_normalization_11/batchnorm/ReadVariableOp:value:0*layer_normalization_11/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2&
$layer_normalization_11/batchnorm/subæ
&layer_normalization_11/batchnorm/add_1AddV2*layer_normalization_11/batchnorm/mul_1:z:0(layer_normalization_11/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_11/batchnorm/add_1Ý
IdentityIdentity*layer_normalization_11/batchnorm/add_1:z:00^layer_normalization_10/batchnorm/ReadVariableOp4^layer_normalization_10/batchnorm/mul/ReadVariableOp0^layer_normalization_11/batchnorm/ReadVariableOp4^layer_normalization_11/batchnorm/mul/ReadVariableOp;^multi_head_attention_5/attention_output/add/ReadVariableOpE^multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_5/key/add/ReadVariableOp8^multi_head_attention_5/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/query/add/ReadVariableOp:^multi_head_attention_5/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/value/add/ReadVariableOp:^multi_head_attention_5/value/einsum/Einsum/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp/^sequential_5/dense_16/Tensordot/ReadVariableOp-^sequential_5/dense_17/BiasAdd/ReadVariableOp/^sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:ÿÿÿÿÿÿÿÿÿÂ ::::::::::::::::2b
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
.sequential_5/dense_17/Tensordot/ReadVariableOp.sequential_5/dense_17/Tensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
½
t
H__inference_concatenate_2_layer_call_and_return_conditional_losses_50773
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
:ÿÿÿÿÿÿÿÿÿ(2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ã
ü
G__inference_sequential_5_layer_call_and_return_conditional_losses_47945

inputs
dense_16_47934
dense_16_47936
dense_17_47939
dense_17_47941
identity¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall
 dense_16/StatefulPartitionedCallStatefulPartitionedCallinputsdense_16_47934dense_16_47936*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_478512"
 dense_16/StatefulPartitionedCall¼
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_47939dense_17_47941*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_478972"
 dense_17/StatefulPartitionedCallÈ
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
í	
Ü
C__inference_dense_18_layer_call_and_return_conditional_losses_50790

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:( *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

O
3__inference_average_pooling1d_5_layer_call_fn_47536

inputs
identityå
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
GPU2*0J 8 *W
fRRP
N__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_475302
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
Ò

ß
3__inference_transformer_block_5_layer_call_fn_50744

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
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_transformer_block_5_layer_call_and_return_conditional_losses_485532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:ÿÿÿÿÿÿÿÿÿÂ ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
²÷
'
 __inference__wrapped_model_47506
input_5
input_6M
Imodel_2_token_and_position_embedding_2_embedding_5_embedding_lookup_47291M
Imodel_2_token_and_position_embedding_2_embedding_4_embedding_lookup_47297@
<model_2_conv1d_2_conv1d_expanddims_1_readvariableop_resource4
0model_2_conv1d_2_biasadd_readvariableop_resourceC
?model_2_batch_normalization_4_batchnorm_readvariableop_resourceG
Cmodel_2_batch_normalization_4_batchnorm_mul_readvariableop_resourceE
Amodel_2_batch_normalization_4_batchnorm_readvariableop_1_resourceE
Amodel_2_batch_normalization_4_batchnorm_readvariableop_2_resourceC
?model_2_batch_normalization_5_batchnorm_readvariableop_resourceG
Cmodel_2_batch_normalization_5_batchnorm_mul_readvariableop_resourceE
Amodel_2_batch_normalization_5_batchnorm_readvariableop_1_resourceE
Amodel_2_batch_normalization_5_batchnorm_readvariableop_2_resourceb
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
Tmodel_2_transformer_block_5_layer_normalization_11_batchnorm_readvariableop_resource3
/model_2_dense_18_matmul_readvariableop_resource4
0model_2_dense_18_biasadd_readvariableop_resource3
/model_2_dense_19_matmul_readvariableop_resource4
0model_2_dense_19_biasadd_readvariableop_resource3
/model_2_dense_20_matmul_readvariableop_resource4
0model_2_dense_20_biasadd_readvariableop_resource
identity¢6model_2/batch_normalization_4/batchnorm/ReadVariableOp¢8model_2/batch_normalization_4/batchnorm/ReadVariableOp_1¢8model_2/batch_normalization_4/batchnorm/ReadVariableOp_2¢:model_2/batch_normalization_4/batchnorm/mul/ReadVariableOp¢6model_2/batch_normalization_5/batchnorm/ReadVariableOp¢8model_2/batch_normalization_5/batchnorm/ReadVariableOp_1¢8model_2/batch_normalization_5/batchnorm/ReadVariableOp_2¢:model_2/batch_normalization_5/batchnorm/mul/ReadVariableOp¢'model_2/conv1d_2/BiasAdd/ReadVariableOp¢3model_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp¢'model_2/dense_18/BiasAdd/ReadVariableOp¢&model_2/dense_18/MatMul/ReadVariableOp¢'model_2/dense_19/BiasAdd/ReadVariableOp¢&model_2/dense_19/MatMul/ReadVariableOp¢'model_2/dense_20/BiasAdd/ReadVariableOp¢&model_2/dense_20/MatMul/ReadVariableOp¢Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup¢Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup¢Kmodel_2/transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp¢Omodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp¢Kmodel_2/transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp¢Omodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp¢Vmodel_2/transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp¢`model_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp¢Imodel_2/transformer_block_5/multi_head_attention_5/key/add/ReadVariableOp¢Smodel_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp¢Kmodel_2/transformer_block_5/multi_head_attention_5/query/add/ReadVariableOp¢Umodel_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp¢Kmodel_2/transformer_block_5/multi_head_attention_5/value/add/ReadVariableOp¢Umodel_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp¢Hmodel_2/transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp¢Jmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp¢Hmodel_2/transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp¢Jmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp
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
,model_2/token_and_position_embedding_2/rangeð
Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookupResourceGatherImodel_2_token_and_position_embedding_2_embedding_5_embedding_lookup_472915model_2/token_and_position_embedding_2/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*\
_classR
PNloc:@model_2/token_and_position_embedding_2/embedding_5/embedding_lookup/47291*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02E
Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup´
Lmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/IdentityIdentityLmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*\
_classR
PNloc:@model_2/token_and_position_embedding_2/embedding_5/embedding_lookup/47291*'
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
:ÿÿÿÿÿÿÿÿÿ¼i29
7model_2/token_and_position_embedding_2/embedding_4/Castû
Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookupResourceGatherImodel_2_token_and_position_embedding_2_embedding_4_embedding_lookup_47297;model_2/token_and_position_embedding_2/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*\
_classR
PNloc:@model_2/token_and_position_embedding_2/embedding_4/embedding_lookup/47297*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *
dtype02E
Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup¹
Lmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/IdentityIdentityLmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*\
_classR
PNloc:@model_2/token_and_position_embedding_2/embedding_4/embedding_lookup/47297*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2N
Lmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identityº
Nmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1IdentityUmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2P
Nmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1Ê
*model_2/token_and_position_embedding_2/addAddV2Wmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1:output:0Wmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2,
*model_2/token_and_position_embedding_2/add
&model_2/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2(
&model_2/conv1d_2/conv1d/ExpandDims/dimò
"model_2/conv1d_2/conv1d/ExpandDims
ExpandDims.model_2/token_and_position_embedding_2/add:z:0/model_2/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2$
"model_2/conv1d_2/conv1d/ExpandDimsë
3model_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<model_2_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype025
3model_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp
(model_2/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_2/conv1d_2/conv1d/ExpandDims_1/dimû
$model_2/conv1d_2/conv1d/ExpandDims_1
ExpandDims;model_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:01model_2/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2&
$model_2/conv1d_2/conv1d/ExpandDims_1û
model_2/conv1d_2/conv1dConv2D+model_2/conv1d_2/conv1d/ExpandDims:output:0-model_2/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *
paddingSAME*
strides
2
model_2/conv1d_2/conv1dÆ
model_2/conv1d_2/conv1d/SqueezeSqueeze model_2/conv1d_2/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2!
model_2/conv1d_2/conv1d/Squeeze¿
'model_2/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_2/conv1d_2/BiasAdd/ReadVariableOpÑ
model_2/conv1d_2/BiasAddBiasAdd(model_2/conv1d_2/conv1d/Squeeze:output:0/model_2/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2
model_2/conv1d_2/BiasAdd
model_2/conv1d_2/ReluRelu!model_2/conv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2
model_2/conv1d_2/Relu
*model_2/average_pooling1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_2/average_pooling1d_5/ExpandDims/dimþ
&model_2/average_pooling1d_5/ExpandDims
ExpandDims.model_2/token_and_position_embedding_2/add:z:03model_2/average_pooling1d_5/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2(
&model_2/average_pooling1d_5/ExpandDimsý
#model_2/average_pooling1d_5/AvgPoolAvgPool/model_2/average_pooling1d_5/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
ksize
*
paddingVALID*
strides
2%
#model_2/average_pooling1d_5/AvgPoolÑ
#model_2/average_pooling1d_5/SqueezeSqueeze,model_2/average_pooling1d_5/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
squeeze_dims
2%
#model_2/average_pooling1d_5/Squeeze
*model_2/average_pooling1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_2/average_pooling1d_4/ExpandDims/dimó
&model_2/average_pooling1d_4/ExpandDims
ExpandDims#model_2/conv1d_2/Relu:activations:03model_2/average_pooling1d_4/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼i 2(
&model_2/average_pooling1d_4/ExpandDimsý
#model_2/average_pooling1d_4/AvgPoolAvgPool/model_2/average_pooling1d_4/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
ksize
*
paddingVALID*
strides
2%
#model_2/average_pooling1d_4/AvgPoolÑ
#model_2/average_pooling1d_4/SqueezeSqueeze,model_2/average_pooling1d_4/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
squeeze_dims
2%
#model_2/average_pooling1d_4/Squeezeì
6model_2/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp?model_2_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype028
6model_2/batch_normalization_4/batchnorm/ReadVariableOp£
-model_2/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-model_2/batch_normalization_4/batchnorm/add/y
+model_2/batch_normalization_4/batchnorm/addAddV2>model_2/batch_normalization_4/batchnorm/ReadVariableOp:value:06model_2/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2-
+model_2/batch_normalization_4/batchnorm/add½
-model_2/batch_normalization_4/batchnorm/RsqrtRsqrt/model_2/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-model_2/batch_normalization_4/batchnorm/Rsqrtø
:model_2/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_2_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:model_2/batch_normalization_4/batchnorm/mul/ReadVariableOpý
+model_2/batch_normalization_4/batchnorm/mulMul1model_2/batch_normalization_4/batchnorm/Rsqrt:y:0Bmodel_2/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+model_2/batch_normalization_4/batchnorm/mulû
-model_2/batch_normalization_4/batchnorm/mul_1Mul,model_2/average_pooling1d_4/Squeeze:output:0/model_2/batch_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2/
-model_2/batch_normalization_4/batchnorm/mul_1ò
8model_2/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_2_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model_2/batch_normalization_4/batchnorm/ReadVariableOp_1ý
-model_2/batch_normalization_4/batchnorm/mul_2Mul@model_2/batch_normalization_4/batchnorm/ReadVariableOp_1:value:0/model_2/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-model_2/batch_normalization_4/batchnorm/mul_2ò
8model_2/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_2_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8model_2/batch_normalization_4/batchnorm/ReadVariableOp_2û
+model_2/batch_normalization_4/batchnorm/subSub@model_2/batch_normalization_4/batchnorm/ReadVariableOp_2:value:01model_2/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+model_2/batch_normalization_4/batchnorm/sub
-model_2/batch_normalization_4/batchnorm/add_1AddV21model_2/batch_normalization_4/batchnorm/mul_1:z:0/model_2/batch_normalization_4/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2/
-model_2/batch_normalization_4/batchnorm/add_1ì
6model_2/batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp?model_2_batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype028
6model_2/batch_normalization_5/batchnorm/ReadVariableOp£
-model_2/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-model_2/batch_normalization_5/batchnorm/add/y
+model_2/batch_normalization_5/batchnorm/addAddV2>model_2/batch_normalization_5/batchnorm/ReadVariableOp:value:06model_2/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2-
+model_2/batch_normalization_5/batchnorm/add½
-model_2/batch_normalization_5/batchnorm/RsqrtRsqrt/model_2/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-model_2/batch_normalization_5/batchnorm/Rsqrtø
:model_2/batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_2_batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:model_2/batch_normalization_5/batchnorm/mul/ReadVariableOpý
+model_2/batch_normalization_5/batchnorm/mulMul1model_2/batch_normalization_5/batchnorm/Rsqrt:y:0Bmodel_2/batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+model_2/batch_normalization_5/batchnorm/mulû
-model_2/batch_normalization_5/batchnorm/mul_1Mul,model_2/average_pooling1d_5/Squeeze:output:0/model_2/batch_normalization_5/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2/
-model_2/batch_normalization_5/batchnorm/mul_1ò
8model_2/batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_2_batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model_2/batch_normalization_5/batchnorm/ReadVariableOp_1ý
-model_2/batch_normalization_5/batchnorm/mul_2Mul@model_2/batch_normalization_5/batchnorm/ReadVariableOp_1:value:0/model_2/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-model_2/batch_normalization_5/batchnorm/mul_2ò
8model_2/batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_2_batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8model_2/batch_normalization_5/batchnorm/ReadVariableOp_2û
+model_2/batch_normalization_5/batchnorm/subSub@model_2/batch_normalization_5/batchnorm/ReadVariableOp_2:value:01model_2/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+model_2/batch_normalization_5/batchnorm/sub
-model_2/batch_normalization_5/batchnorm/add_1AddV21model_2/batch_normalization_5/batchnorm/mul_1:z:0/model_2/batch_normalization_5/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2/
-model_2/batch_normalization_5/batchnorm/add_1Ì
model_2/add_2/addAddV21model_2/batch_normalization_4/batchnorm/add_1:z:01model_2/batch_normalization_5/batchnorm/add_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
model_2/add_2/addÑ
Umodel_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOp^model_2_transformer_block_5_multi_head_attention_5_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpñ
Fmodel_2/transformer_block_5/multi_head_attention_5/query/einsum/EinsumEinsummodel_2/add_2/add:z:0]model_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2H
Fmodel_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum¯
Kmodel_2/transformer_block_5/multi_head_attention_5/query/add/ReadVariableOpReadVariableOpTmodel_2_transformer_block_5_multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_2/transformer_block_5/multi_head_attention_5/query/add/ReadVariableOpæ
<model_2/transformer_block_5/multi_head_attention_5/query/addAddV2Omodel_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum:output:0Smodel_2/transformer_block_5/multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2>
<model_2/transformer_block_5/multi_head_attention_5/query/addË
Smodel_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOp\model_2_transformer_block_5_multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02U
Smodel_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpë
Dmodel_2/transformer_block_5/multi_head_attention_5/key/einsum/EinsumEinsummodel_2/add_2/add:z:0[model_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2F
Dmodel_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum©
Imodel_2/transformer_block_5/multi_head_attention_5/key/add/ReadVariableOpReadVariableOpRmodel_2_transformer_block_5_multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02K
Imodel_2/transformer_block_5/multi_head_attention_5/key/add/ReadVariableOpÞ
:model_2/transformer_block_5/multi_head_attention_5/key/addAddV2Mmodel_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum:output:0Qmodel_2/transformer_block_5/multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2<
:model_2/transformer_block_5/multi_head_attention_5/key/addÑ
Umodel_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOp^model_2_transformer_block_5_multi_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpñ
Fmodel_2/transformer_block_5/multi_head_attention_5/value/einsum/EinsumEinsummodel_2/add_2/add:z:0]model_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2H
Fmodel_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum¯
Kmodel_2/transformer_block_5/multi_head_attention_5/value/add/ReadVariableOpReadVariableOpTmodel_2_transformer_block_5_multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_2/transformer_block_5/multi_head_attention_5/value/add/ReadVariableOpæ
<model_2/transformer_block_5/multi_head_attention_5/value/addAddV2Omodel_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum:output:0Smodel_2/transformer_block_5/multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2>
<model_2/transformer_block_5/multi_head_attention_5/value/add¹
8model_2/transformer_block_5/multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2:
8model_2/transformer_block_5/multi_head_attention_5/Mul/y·
6model_2/transformer_block_5/multi_head_attention_5/MulMul@model_2/transformer_block_5/multi_head_attention_5/query/add:z:0Amodel_2/transformer_block_5/multi_head_attention_5/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 28
6model_2/transformer_block_5/multi_head_attention_5/Mulî
@model_2/transformer_block_5/multi_head_attention_5/einsum/EinsumEinsum>model_2/transformer_block_5/multi_head_attention_5/key/add:z:0:model_2/transformer_block_5/multi_head_attention_5/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ*
equationaecd,abcd->acbe2B
@model_2/transformer_block_5/multi_head_attention_5/einsum/Einsum
Bmodel_2/transformer_block_5/multi_head_attention_5/softmax/SoftmaxSoftmaxImodel_2/transformer_block_5/multi_head_attention_5/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2D
Bmodel_2/transformer_block_5/multi_head_attention_5/softmax/Softmax 
Cmodel_2/transformer_block_5/multi_head_attention_5/dropout/IdentityIdentityLmodel_2/transformer_block_5/multi_head_attention_5/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2E
Cmodel_2/transformer_block_5/multi_head_attention_5/dropout/Identity
Bmodel_2/transformer_block_5/multi_head_attention_5/einsum_1/EinsumEinsumLmodel_2/transformer_block_5/multi_head_attention_5/dropout/Identity:output:0@model_2/transformer_block_5/multi_head_attention_5/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationacbe,aecd->abcd2D
Bmodel_2/transformer_block_5/multi_head_attention_5/einsum_1/Einsumò
`model_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpimodel_2_transformer_block_5_multi_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02b
`model_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpÄ
Qmodel_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/EinsumEinsumKmodel_2/transformer_block_5/multi_head_attention_5/einsum_1/Einsum:output:0hmodel_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabcd,cde->abe2S
Qmodel_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/EinsumÌ
Vmodel_2/transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOp_model_2_transformer_block_5_multi_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02X
Vmodel_2/transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp
Gmodel_2/transformer_block_5/multi_head_attention_5/attention_output/addAddV2Zmodel_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum:output:0^model_2/transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2I
Gmodel_2/transformer_block_5/multi_head_attention_5/attention_output/addò
/model_2/transformer_block_5/dropout_14/IdentityIdentityKmodel_2/transformer_block_5/multi_head_attention_5/attention_output/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 21
/model_2/transformer_block_5/dropout_14/IdentityÓ
model_2/transformer_block_5/addAddV2model_2/add_2/add:z:08model_2/transformer_block_5/dropout_14/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2!
model_2/transformer_block_5/addð
Qmodel_2/transformer_block_5/layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_2/transformer_block_5/layer_normalization_10/moments/mean/reduction_indicesÓ
?model_2/transformer_block_5/layer_normalization_10/moments/meanMean#model_2/transformer_block_5/add:z:0Zmodel_2/transformer_block_5/layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2A
?model_2/transformer_block_5/layer_normalization_10/moments/mean£
Gmodel_2/transformer_block_5/layer_normalization_10/moments/StopGradientStopGradientHmodel_2/transformer_block_5/layer_normalization_10/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2I
Gmodel_2/transformer_block_5/layer_normalization_10/moments/StopGradientß
Lmodel_2/transformer_block_5/layer_normalization_10/moments/SquaredDifferenceSquaredDifference#model_2/transformer_block_5/add:z:0Pmodel_2/transformer_block_5/layer_normalization_10/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2N
Lmodel_2/transformer_block_5/layer_normalization_10/moments/SquaredDifferenceø
Umodel_2/transformer_block_5/layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_2/transformer_block_5/layer_normalization_10/moments/variance/reduction_indices
Cmodel_2/transformer_block_5/layer_normalization_10/moments/varianceMeanPmodel_2/transformer_block_5/layer_normalization_10/moments/SquaredDifference:z:0^model_2/transformer_block_5/layer_normalization_10/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2E
Cmodel_2/transformer_block_5/layer_normalization_10/moments/varianceÍ
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752D
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add/yß
@model_2/transformer_block_5/layer_normalization_10/batchnorm/addAddV2Lmodel_2/transformer_block_5/layer_normalization_10/moments/variance:output:0Kmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2B
@model_2/transformer_block_5/layer_normalization_10/batchnorm/add
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/RsqrtRsqrtDmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2D
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/Rsqrt·
Omodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_2_transformer_block_5_layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02Q
Omodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpã
@model_2/transformer_block_5/layer_normalization_10/batchnorm/mulMulFmodel_2/transformer_block_5/layer_normalization_10/batchnorm/Rsqrt:y:0Wmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2B
@model_2/transformer_block_5/layer_normalization_10/batchnorm/mul±
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul_1Mul#model_2/transformer_block_5/add:z:0Dmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2D
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul_1Ö
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul_2MulHmodel_2/transformer_block_5/layer_normalization_10/moments/mean:output:0Dmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2D
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul_2«
Kmodel_2/transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOpTmodel_2_transformer_block_5_layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02M
Kmodel_2/transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpß
@model_2/transformer_block_5/layer_normalization_10/batchnorm/subSubSmodel_2/transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp:value:0Fmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2B
@model_2/transformer_block_5/layer_normalization_10/batchnorm/subÖ
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add_1AddV2Fmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul_1:z:0Dmodel_2/transformer_block_5/layer_normalization_10/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2D
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add_1¬
Jmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOpSmodel_2_transformer_block_5_sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

:  *
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
Amodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/stackç
Emodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/transpose	TransposeFmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add_1:z:0Kmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2G
Emodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/transposeç
Cmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReshapeReshapeImodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/transpose:y:0Jmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2E
Cmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Reshapeæ
Bmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/MatMulMatMulLmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Reshape:output:0Rmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2D
Bmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/MatMulÔ
Cmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2E
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
Dmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat_1Ù
;model_2/transformer_block_5/sequential_5/dense_16/TensordotReshapeLmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/MatMul:product:0Mmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2=
;model_2/transformer_block_5/sequential_5/dense_16/Tensordot¢
Hmodel_2/transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOpQmodel_2_transformer_block_5_sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel_2/transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpÐ
9model_2/transformer_block_5/sequential_5/dense_16/BiasAddBiasAddDmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot:output:0Pmodel_2/transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2;
9model_2/transformer_block_5/sequential_5/dense_16/BiasAddó
6model_2/transformer_block_5/sequential_5/dense_16/ReluReluBmodel_2/transformer_block_5/sequential_5/dense_16/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 28
6model_2/transformer_block_5/sequential_5/dense_16/Relu¬
Jmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOpSmodel_2_transformer_block_5_sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:  *
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
Amodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/stackå
Emodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/transpose	TransposeDmodel_2/transformer_block_5/sequential_5/dense_16/Relu:activations:0Kmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2G
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
Dmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat_1Ù
;model_2/transformer_block_5/sequential_5/dense_17/TensordotReshapeLmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/MatMul:product:0Mmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2=
;model_2/transformer_block_5/sequential_5/dense_17/Tensordot¢
Hmodel_2/transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOpQmodel_2_transformer_block_5_sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel_2/transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpÐ
9model_2/transformer_block_5/sequential_5/dense_17/BiasAddBiasAddDmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot:output:0Pmodel_2/transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2;
9model_2/transformer_block_5/sequential_5/dense_17/BiasAddé
/model_2/transformer_block_5/dropout_15/IdentityIdentityBmodel_2/transformer_block_5/sequential_5/dense_17/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 21
/model_2/transformer_block_5/dropout_15/Identity
!model_2/transformer_block_5/add_1AddV2Fmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add_1:z:08model_2/transformer_block_5/dropout_15/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2#
!model_2/transformer_block_5/add_1ð
Qmodel_2/transformer_block_5/layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_2/transformer_block_5/layer_normalization_11/moments/mean/reduction_indicesÕ
?model_2/transformer_block_5/layer_normalization_11/moments/meanMean%model_2/transformer_block_5/add_1:z:0Zmodel_2/transformer_block_5/layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2A
?model_2/transformer_block_5/layer_normalization_11/moments/mean£
Gmodel_2/transformer_block_5/layer_normalization_11/moments/StopGradientStopGradientHmodel_2/transformer_block_5/layer_normalization_11/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2I
Gmodel_2/transformer_block_5/layer_normalization_11/moments/StopGradientá
Lmodel_2/transformer_block_5/layer_normalization_11/moments/SquaredDifferenceSquaredDifference%model_2/transformer_block_5/add_1:z:0Pmodel_2/transformer_block_5/layer_normalization_11/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2N
Lmodel_2/transformer_block_5/layer_normalization_11/moments/SquaredDifferenceø
Umodel_2/transformer_block_5/layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_2/transformer_block_5/layer_normalization_11/moments/variance/reduction_indices
Cmodel_2/transformer_block_5/layer_normalization_11/moments/varianceMeanPmodel_2/transformer_block_5/layer_normalization_11/moments/SquaredDifference:z:0^model_2/transformer_block_5/layer_normalization_11/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2E
Cmodel_2/transformer_block_5/layer_normalization_11/moments/varianceÍ
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752D
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add/yß
@model_2/transformer_block_5/layer_normalization_11/batchnorm/addAddV2Lmodel_2/transformer_block_5/layer_normalization_11/moments/variance:output:0Kmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2B
@model_2/transformer_block_5/layer_normalization_11/batchnorm/add
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/RsqrtRsqrtDmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2D
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/Rsqrt·
Omodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_2_transformer_block_5_layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02Q
Omodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpã
@model_2/transformer_block_5/layer_normalization_11/batchnorm/mulMulFmodel_2/transformer_block_5/layer_normalization_11/batchnorm/Rsqrt:y:0Wmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2B
@model_2/transformer_block_5/layer_normalization_11/batchnorm/mul³
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul_1Mul%model_2/transformer_block_5/add_1:z:0Dmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2D
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul_1Ö
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul_2MulHmodel_2/transformer_block_5/layer_normalization_11/moments/mean:output:0Dmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2D
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul_2«
Kmodel_2/transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOpTmodel_2_transformer_block_5_layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02M
Kmodel_2/transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpß
@model_2/transformer_block_5/layer_normalization_11/batchnorm/subSubSmodel_2/transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp:value:0Fmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2B
@model_2/transformer_block_5/layer_normalization_11/batchnorm/subÖ
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add_1AddV2Fmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul_1:z:0Dmodel_2/transformer_block_5/layer_normalization_11/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2D
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add_1¸
9model_2/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9model_2/global_average_pooling1d_2/Mean/reduction_indices
'model_2/global_average_pooling1d_2/MeanMeanFmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add_1:z:0Bmodel_2/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'model_2/global_average_pooling1d_2/Mean
!model_2/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_2/concatenate_2/concat/axisê
model_2/concatenate_2/concatConcatV20model_2/global_average_pooling1d_2/Mean:output:0input_6*model_2/concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model_2/concatenate_2/concatÀ
&model_2/dense_18/MatMul/ReadVariableOpReadVariableOp/model_2_dense_18_matmul_readvariableop_resource*
_output_shapes

:( *
dtype02(
&model_2/dense_18/MatMul/ReadVariableOpÅ
model_2/dense_18/MatMulMatMul%model_2/concatenate_2/concat:output:0.model_2/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_2/dense_18/MatMul¿
'model_2/dense_18/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_2/dense_18/BiasAdd/ReadVariableOpÅ
model_2/dense_18/BiasAddBiasAdd!model_2/dense_18/MatMul:product:0/model_2/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_2/dense_18/BiasAdd
model_2/dense_18/ReluRelu!model_2/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_2/dense_18/Relu
model_2/dropout_16/IdentityIdentity#model_2/dense_18/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_2/dropout_16/IdentityÀ
&model_2/dense_19/MatMul/ReadVariableOpReadVariableOp/model_2_dense_19_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02(
&model_2/dense_19/MatMul/ReadVariableOpÄ
model_2/dense_19/MatMulMatMul$model_2/dropout_16/Identity:output:0.model_2/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_2/dense_19/MatMul¿
'model_2/dense_19/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_2/dense_19/BiasAdd/ReadVariableOpÅ
model_2/dense_19/BiasAddBiasAdd!model_2/dense_19/MatMul:product:0/model_2/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_2/dense_19/BiasAdd
model_2/dense_19/ReluRelu!model_2/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_2/dense_19/Relu
model_2/dropout_17/IdentityIdentity#model_2/dense_19/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_2/dropout_17/IdentityÀ
&model_2/dense_20/MatMul/ReadVariableOpReadVariableOp/model_2_dense_20_matmul_readvariableop_resource*
_output_shapes

: *
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
model_2/dense_20/BiasAddÌ
IdentityIdentity!model_2/dense_20/BiasAdd:output:07^model_2/batch_normalization_4/batchnorm/ReadVariableOp9^model_2/batch_normalization_4/batchnorm/ReadVariableOp_19^model_2/batch_normalization_4/batchnorm/ReadVariableOp_2;^model_2/batch_normalization_4/batchnorm/mul/ReadVariableOp7^model_2/batch_normalization_5/batchnorm/ReadVariableOp9^model_2/batch_normalization_5/batchnorm/ReadVariableOp_19^model_2/batch_normalization_5/batchnorm/ReadVariableOp_2;^model_2/batch_normalization_5/batchnorm/mul/ReadVariableOp(^model_2/conv1d_2/BiasAdd/ReadVariableOp4^model_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp(^model_2/dense_18/BiasAdd/ReadVariableOp'^model_2/dense_18/MatMul/ReadVariableOp(^model_2/dense_19/BiasAdd/ReadVariableOp'^model_2/dense_19/MatMul/ReadVariableOp(^model_2/dense_20/BiasAdd/ReadVariableOp'^model_2/dense_20/MatMul/ReadVariableOpD^model_2/token_and_position_embedding_2/embedding_4/embedding_lookupD^model_2/token_and_position_embedding_2/embedding_5/embedding_lookupL^model_2/transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpP^model_2/transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpL^model_2/transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpP^model_2/transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpW^model_2/transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpa^model_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpJ^model_2/transformer_block_5/multi_head_attention_5/key/add/ReadVariableOpT^model_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpL^model_2/transformer_block_5/multi_head_attention_5/query/add/ReadVariableOpV^model_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpL^model_2/transformer_block_5/multi_head_attention_5/value/add/ReadVariableOpV^model_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpI^model_2/transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpK^model_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpI^model_2/transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpK^model_2/transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ä
_input_shapes²
¯:ÿÿÿÿÿÿÿÿÿ¼i:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::2p
6model_2/batch_normalization_4/batchnorm/ReadVariableOp6model_2/batch_normalization_4/batchnorm/ReadVariableOp2t
8model_2/batch_normalization_4/batchnorm/ReadVariableOp_18model_2/batch_normalization_4/batchnorm/ReadVariableOp_12t
8model_2/batch_normalization_4/batchnorm/ReadVariableOp_28model_2/batch_normalization_4/batchnorm/ReadVariableOp_22x
:model_2/batch_normalization_4/batchnorm/mul/ReadVariableOp:model_2/batch_normalization_4/batchnorm/mul/ReadVariableOp2p
6model_2/batch_normalization_5/batchnorm/ReadVariableOp6model_2/batch_normalization_5/batchnorm/ReadVariableOp2t
8model_2/batch_normalization_5/batchnorm/ReadVariableOp_18model_2/batch_normalization_5/batchnorm/ReadVariableOp_12t
8model_2/batch_normalization_5/batchnorm/ReadVariableOp_28model_2/batch_normalization_5/batchnorm/ReadVariableOp_22x
:model_2/batch_normalization_5/batchnorm/mul/ReadVariableOp:model_2/batch_normalization_5/batchnorm/mul/ReadVariableOp2R
'model_2/conv1d_2/BiasAdd/ReadVariableOp'model_2/conv1d_2/BiasAdd/ReadVariableOp2j
3model_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp3model_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2R
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
:ÿÿÿÿÿÿÿÿÿ¼i
!
_user_specified_name	input_5:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6
í	
Ü
C__inference_dense_19_layer_call_and_return_conditional_losses_48758

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ñ
}
(__inference_dense_17_layer_call_fn_51111

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
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_478972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÂ ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
ìÞ
á
N__inference_transformer_block_5_layer_call_and_return_conditional_losses_48553

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
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_5/query/einsum/EinsumEinsuminputsAmulti_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2,
*multi_head_attention_5/query/einsum/EinsumÛ
/multi_head_attention_5/query/add/ReadVariableOpReadVariableOp8multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/query/add/ReadVariableOpö
 multi_head_attention_5/query/addAddV23multi_head_attention_5/query/einsum/Einsum:output:07multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2"
 multi_head_attention_5/query/add÷
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_5/key/einsum/EinsumEinsuminputs?multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2*
(multi_head_attention_5/key/einsum/EinsumÕ
-multi_head_attention_5/key/add/ReadVariableOpReadVariableOp6multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_5/key/add/ReadVariableOpî
multi_head_attention_5/key/addAddV21multi_head_attention_5/key/einsum/Einsum:output:05multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2 
multi_head_attention_5/key/addý
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_5/value/einsum/EinsumEinsuminputsAmulti_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabc,cde->abde2,
*multi_head_attention_5/value/einsum/EinsumÛ
/multi_head_attention_5/value/add/ReadVariableOpReadVariableOp8multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/value/add/ReadVariableOpö
 multi_head_attention_5/value/addAddV23multi_head_attention_5/value/einsum/Einsum:output:07multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2"
 multi_head_attention_5/value/add
multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_5/Mul/yÇ
multi_head_attention_5/MulMul$multi_head_attention_5/query/add:z:0%multi_head_attention_5/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
multi_head_attention_5/Mulþ
$multi_head_attention_5/einsum/EinsumEinsum"multi_head_attention_5/key/add:z:0multi_head_attention_5/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ*
equationaecd,abcd->acbe2&
$multi_head_attention_5/einsum/EinsumÆ
&multi_head_attention_5/softmax/SoftmaxSoftmax-multi_head_attention_5/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2(
&multi_head_attention_5/softmax/SoftmaxÌ
'multi_head_attention_5/dropout/IdentityIdentity0multi_head_attention_5/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂÂ2)
'multi_head_attention_5/dropout/Identity
&multi_head_attention_5/einsum_1/EinsumEinsum0multi_head_attention_5/dropout/Identity:output:0$multi_head_attention_5/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationacbe,aecd->abcd2(
&multi_head_attention_5/einsum_1/Einsum
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpÔ
5multi_head_attention_5/attention_output/einsum/EinsumEinsum/multi_head_attention_5/einsum_1/Einsum:output:0Lmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ *
equationabcd,cde->abe27
5multi_head_attention_5/attention_output/einsum/Einsumø
:multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_5/attention_output/add/ReadVariableOp
+multi_head_attention_5/attention_output/addAddV2>multi_head_attention_5/attention_output/einsum/Einsum:output:0Bmulti_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2-
+multi_head_attention_5/attention_output/add
dropout_14/IdentityIdentity/multi_head_attention_5/attention_output/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dropout_14/Identityp
addAddV2inputsdropout_14/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
add¸
5layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_10/moments/mean/reduction_indicesã
#layer_normalization_10/moments/meanMeanadd:z:0>layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2%
#layer_normalization_10/moments/meanÏ
+layer_normalization_10/moments/StopGradientStopGradient,layer_normalization_10/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2-
+layer_normalization_10/moments/StopGradientï
0layer_normalization_10/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_10/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 22
0layer_normalization_10/moments/SquaredDifferenceÀ
9layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_10/moments/variance/reduction_indices
'layer_normalization_10/moments/varianceMean4layer_normalization_10/moments/SquaredDifference:z:0Blayer_normalization_10/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2)
'layer_normalization_10/moments/variance
&layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_10/batchnorm/add/yï
$layer_normalization_10/batchnorm/addAddV20layer_normalization_10/moments/variance:output:0/layer_normalization_10/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2&
$layer_normalization_10/batchnorm/addº
&layer_normalization_10/batchnorm/RsqrtRsqrt(layer_normalization_10/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2(
&layer_normalization_10/batchnorm/Rsqrtã
3layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_10/batchnorm/mul/ReadVariableOpó
$layer_normalization_10/batchnorm/mulMul*layer_normalization_10/batchnorm/Rsqrt:y:0;layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2&
$layer_normalization_10/batchnorm/mulÁ
&layer_normalization_10/batchnorm/mul_1Muladd:z:0(layer_normalization_10/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_10/batchnorm/mul_1æ
&layer_normalization_10/batchnorm/mul_2Mul,layer_normalization_10/moments/mean:output:0(layer_normalization_10/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_10/batchnorm/mul_2×
/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_10/batchnorm/ReadVariableOpï
$layer_normalization_10/batchnorm/subSub7layer_normalization_10/batchnorm/ReadVariableOp:value:0*layer_normalization_10/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2&
$layer_normalization_10/batchnorm/subæ
&layer_normalization_10/batchnorm/add_1AddV2*layer_normalization_10/batchnorm/mul_1:z:0(layer_normalization_10/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_10/batchnorm/add_1Ø
.sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

:  *
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
%sequential_5/dense_16/Tensordot/stack÷
)sequential_5/dense_16/Tensordot/transpose	Transpose*layer_normalization_10/batchnorm/add_1:z:0/sequential_5/dense_16/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2+
)sequential_5/dense_16/Tensordot/transpose÷
'sequential_5/dense_16/Tensordot/ReshapeReshape-sequential_5/dense_16/Tensordot/transpose:y:0.sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'sequential_5/dense_16/Tensordot/Reshapeö
&sequential_5/dense_16/Tensordot/MatMulMatMul0sequential_5/dense_16/Tensordot/Reshape:output:06sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential_5/dense_16/Tensordot/MatMul
'sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
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
(sequential_5/dense_16/Tensordot/concat_1é
sequential_5/dense_16/TensordotReshape0sequential_5/dense_16/Tensordot/MatMul:product:01sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2!
sequential_5/dense_16/TensordotÎ
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOpà
sequential_5/dense_16/BiasAddBiasAdd(sequential_5/dense_16/Tensordot:output:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
sequential_5/dense_16/BiasAdd
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
sequential_5/dense_16/ReluØ
.sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:  *
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
%sequential_5/dense_17/Tensordot/stackõ
)sequential_5/dense_17/Tensordot/transpose	Transpose(sequential_5/dense_16/Relu:activations:0/sequential_5/dense_17/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2+
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
(sequential_5/dense_17/Tensordot/concat_1é
sequential_5/dense_17/TensordotReshape0sequential_5/dense_17/Tensordot/MatMul:product:01sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2!
sequential_5/dense_17/TensordotÎ
,sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_17/BiasAdd/ReadVariableOpà
sequential_5/dense_17/BiasAddBiasAdd(sequential_5/dense_17/Tensordot:output:04sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
sequential_5/dense_17/BiasAdd
dropout_15/IdentityIdentity&sequential_5/dense_17/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
dropout_15/Identity
add_1AddV2*layer_normalization_10/batchnorm/add_1:z:0dropout_15/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
add_1¸
5layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_11/moments/mean/reduction_indiceså
#layer_normalization_11/moments/meanMean	add_1:z:0>layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2%
#layer_normalization_11/moments/meanÏ
+layer_normalization_11/moments/StopGradientStopGradient,layer_normalization_11/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2-
+layer_normalization_11/moments/StopGradientñ
0layer_normalization_11/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_11/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 22
0layer_normalization_11/moments/SquaredDifferenceÀ
9layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_11/moments/variance/reduction_indices
'layer_normalization_11/moments/varianceMean4layer_normalization_11/moments/SquaredDifference:z:0Blayer_normalization_11/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ*
	keep_dims(2)
'layer_normalization_11/moments/variance
&layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_11/batchnorm/add/yï
$layer_normalization_11/batchnorm/addAddV20layer_normalization_11/moments/variance:output:0/layer_normalization_11/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2&
$layer_normalization_11/batchnorm/addº
&layer_normalization_11/batchnorm/RsqrtRsqrt(layer_normalization_11/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ2(
&layer_normalization_11/batchnorm/Rsqrtã
3layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_11/batchnorm/mul/ReadVariableOpó
$layer_normalization_11/batchnorm/mulMul*layer_normalization_11/batchnorm/Rsqrt:y:0;layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2&
$layer_normalization_11/batchnorm/mulÃ
&layer_normalization_11/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_11/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_11/batchnorm/mul_1æ
&layer_normalization_11/batchnorm/mul_2Mul,layer_normalization_11/moments/mean:output:0(layer_normalization_11/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_11/batchnorm/mul_2×
/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_11/batchnorm/ReadVariableOpï
$layer_normalization_11/batchnorm/subSub7layer_normalization_11/batchnorm/ReadVariableOp:value:0*layer_normalization_11/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2&
$layer_normalization_11/batchnorm/subæ
&layer_normalization_11/batchnorm/add_1AddV2*layer_normalization_11/batchnorm/mul_1:z:0(layer_normalization_11/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2(
&layer_normalization_11/batchnorm/add_1Ý
IdentityIdentity*layer_normalization_11/batchnorm/add_1:z:00^layer_normalization_10/batchnorm/ReadVariableOp4^layer_normalization_10/batchnorm/mul/ReadVariableOp0^layer_normalization_11/batchnorm/ReadVariableOp4^layer_normalization_11/batchnorm/mul/ReadVariableOp;^multi_head_attention_5/attention_output/add/ReadVariableOpE^multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_5/key/add/ReadVariableOp8^multi_head_attention_5/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/query/add/ReadVariableOp:^multi_head_attention_5/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/value/add/ReadVariableOp:^multi_head_attention_5/value/einsum/Einsum/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp/^sequential_5/dense_16/Tensordot/ReadVariableOp-^sequential_5/dense_17/BiasAdd/ReadVariableOp/^sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:ÿÿÿÿÿÿÿÿÿÂ ::::::::::::::::2b
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
.sequential_5/dense_17/Tensordot/ReadVariableOp.sequential_5/dense_17/Tensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
_user_specified_nameinputs
	
Ü
C__inference_dense_20_layer_call_and_return_conditional_losses_48814

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
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
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

d
E__inference_dropout_17_layer_call_and_return_conditional_losses_50858

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
:ÿÿÿÿÿÿÿÿÿ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÀ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
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
:ÿÿÿÿÿÿÿÿÿ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
³0
Å
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_48207

inputs
assignmovingavg_48182
assignmovingavg_1_48188)
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
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
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
moments/Squeeze_1Ë
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/48182*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_48182*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpð
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/48182*
_output_shapes
: 2
AssignMovingAvg/subç
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/48182*
_output_shapes
: 2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_48182AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/48182*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÑ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/48188*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_48188*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpú
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/48188*
_output_shapes
: 2
AssignMovingAvg_1/subñ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/48188*
_output_shapes
: 2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_48188AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/48188*
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2
batchnorm/add_1¸
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÂ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ 
 
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
serving_default_input_5:0ÿÿÿÿÿÿÿÿÿ¼i
;
input_60
serving_default_input_6:0ÿÿÿÿÿÿÿÿÿ<
dense_200
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:áÛ
®?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
¸_default_save_signature
¹__call__
+º&call_and_return_all_conditional_losses"¬:
_tf_keras_network:{"class_name": "Functional", "name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 13500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}, "name": "token_and_position_embedding_2", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["token_and_position_embedding_2", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_4", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_4", "inbound_nodes": [[["conv1d_2", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_5", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_5", "inbound_nodes": [[["token_and_position_embedding_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["average_pooling1d_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["average_pooling1d_5", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}], ["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block_5", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling1d_2", "inbound_nodes": [[["transformer_block_5", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_2", "inbound_nodes": [[["global_average_pooling1d_2", 0, 0, {}], ["input_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["concatenate_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_16", "inbound_nodes": [[["dense_18", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["dropout_16", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_17", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_17", "inbound_nodes": [[["dense_19", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_20", "inbound_nodes": [[["dropout_17", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0], ["input_6", 0, 0]], "output_layers": [["dense_20", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 13500]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 13500]}, {"class_name": "TensorShape", "items": [null, 8]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adamax", "config": {"name": "Adamax", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
ñ"î
_tf_keras_input_layerÎ{"class_name": "InputLayer", "name": "input_5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 13500]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 13500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}
ç
	token_emb
pos_emb
	variables
trainable_variables
regularization_losses
	keras_api
»__call__
+¼&call_and_return_all_conditional_losses"º
_tf_keras_layer {"class_name": "TokenAndPositionEmbedding", "name": "token_and_position_embedding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
é	

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
½__call__
+¾&call_and_return_all_conditional_losses"Â
_tf_keras_layer¨{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13500, 32]}}

$	variables
%trainable_variables
&regularization_losses
'	keras_api
¿__call__
+À&call_and_return_all_conditional_losses"ø
_tf_keras_layerÞ{"class_name": "AveragePooling1D", "name": "average_pooling1d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_4", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}

(	variables
)trainable_variables
*regularization_losses
+	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses"ø
_tf_keras_layerÞ{"class_name": "AveragePooling1D", "name": "average_pooling1d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_5", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
¹	
,axis
	-gamma
.beta
/moving_mean
0moving_variance
1	variables
2trainable_variables
3regularization_losses
4	keras_api
Ã__call__
+Ä&call_and_return_all_conditional_losses"ã
_tf_keras_layerÉ{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 450, 32]}}
¹	
5axis
	6gamma
7beta
8moving_mean
9moving_variance
:	variables
;trainable_variables
<regularization_losses
=	keras_api
Å__call__
+Æ&call_and_return_all_conditional_losses"ã
_tf_keras_layerÉ{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 450, 32]}}
µ
>	variables
?trainable_variables
@regularization_losses
A	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses"¤
_tf_keras_layer{"class_name": "Add", "name": "add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 450, 32]}, {"class_name": "TensorShape", "items": [null, 450, 32]}]}

Batt
Cffn
D
layernorm1
E
layernorm2
Fdropout1
Gdropout2
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses"¥
_tf_keras_layer{"class_name": "TransformerBlock", "name": "transformer_block_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}

L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
Ë__call__
+Ì&call_and_return_all_conditional_losses"
_tf_keras_layerî{"class_name": "GlobalAveragePooling1D", "name": "global_average_pooling1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling1d_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
é"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_6", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}}
Î
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses"½
_tf_keras_layer£{"class_name": "Concatenate", "name": "concatenate_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32]}, {"class_name": "TensorShape", "items": [null, 8]}]}
ô

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
é
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
Ñ__call__
+Ò&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
ô

^kernel
_bias
`	variables
atrainable_variables
bregularization_losses
c	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
é
d	variables
etrainable_variables
fregularization_losses
g	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_17", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
õ

hkernel
ibias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
µ

nbeta_1

obeta_2
	pdecay
qlearning_rate
ritermümý-mþ.mÿ6m7mTmUm^m_mhmimsmtmumvmwmxmymzm{m|m}m~mm	m	m	m	m	mvv-v.v6v7vTv Uv¡^v¢_v£hv¤iv¥sv¦tv§uv¨vv©wvªxv«yv¬zv­{v®|v¯}v°~v±v²	v³	v´	vµ	v¶	v·"
	optimizer
«
s0
t1
2
3
-4
.5
/6
07
68
79
810
911
u12
v13
w14
x15
y16
z17
{18
|19
}20
~21
22
23
24
25
26
27
T28
U29
^30
_31
h32
i33"
trackable_list_wrapper

s0
t1
2
3
-4
.5
66
77
u8
v9
w10
x11
y12
z13
{14
|15
}16
~17
18
19
20
21
22
23
T24
U25
^26
_27
h28
i29"
trackable_list_wrapper
 "
trackable_list_wrapper
Ó
	variables
metrics
trainable_variables
layer_metrics
regularization_losses
non_trainable_variables
layers
 layer_regularization_losses
¹__call__
¸_default_save_signature
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
-
Ùserving_default"
signature_map
´
s
embeddings
	variables
trainable_variables
regularization_losses
	keras_api
Ú__call__
+Û&call_and_return_all_conditional_losses"
_tf_keras_layerõ{"class_name": "Embedding", "name": "embedding_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 5, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13500]}}
±
t
embeddings
	variables
trainable_variables
regularization_losses
	keras_api
Ü__call__
+Ý&call_and_return_all_conditional_losses"
_tf_keras_layerò{"class_name": "Embedding", "name": "embedding_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13500, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
	variables
metrics
trainable_variables
layer_metrics
regularization_losses
non_trainable_variables
layers
 layer_regularization_losses
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
_generic_user_object
%:#  2conv1d_2/kernel
: 2conv1d_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 	variables
metrics
!trainable_variables
layer_metrics
"regularization_losses
non_trainable_variables
layers
 layer_regularization_losses
½__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
$	variables
metrics
%trainable_variables
layer_metrics
&regularization_losses
non_trainable_variables
layers
  layer_regularization_losses
¿__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
(	variables
¡metrics
)trainable_variables
¢layer_metrics
*regularization_losses
£non_trainable_variables
¤layers
 ¥layer_regularization_losses
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_4/gamma
(:& 2batch_normalization_4/beta
1:/  (2!batch_normalization_4/moving_mean
5:3  (2%batch_normalization_4/moving_variance
<
-0
.1
/2
03"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
1	variables
¦metrics
2trainable_variables
§layer_metrics
3regularization_losses
¨non_trainable_variables
©layers
 ªlayer_regularization_losses
Ã__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_5/gamma
(:& 2batch_normalization_5/beta
1:/  (2!batch_normalization_5/moving_mean
5:3  (2%batch_normalization_5/moving_variance
<
60
71
82
93"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
:	variables
«metrics
;trainable_variables
¬layer_metrics
<regularization_losses
­non_trainable_variables
®layers
 ¯layer_regularization_losses
Å__call__
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
>	variables
°metrics
?trainable_variables
±layer_metrics
@regularization_losses
²non_trainable_variables
³layers
 ´layer_regularization_losses
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object

µ_query_dense
¶
_key_dense
·_value_dense
¸_softmax
¹_dropout_layer
º_output_dense
»	variables
¼trainable_variables
½regularization_losses
¾	keras_api
Þ__call__
+ß&call_and_return_all_conditional_losses"
_tf_keras_layerê{"class_name": "MultiHeadAttention", "name": "multi_head_attention_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multi_head_attention_5", "trainable": true, "dtype": "float32", "num_heads": 1, "key_dim": 32, "value_dim": 32, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}
²
¿layer_with_weights-0
¿layer-0
Àlayer_with_weights-1
Àlayer-1
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
à__call__
+á&call_and_return_all_conditional_losses"Ë
_tf_keras_sequential¬{"class_name": "Sequential", "name": "sequential_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 450, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_16_input"}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 450, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 450, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_16_input"}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
í
	Åaxis

gamma
	beta
Æ	variables
Çtrainable_variables
Èregularization_losses
É	keras_api
â__call__
+ã&call_and_return_all_conditional_losses"¶
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_10", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 450, 32]}}
í
	Êaxis

gamma
	beta
Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
ä__call__
+å&call_and_return_all_conditional_losses"¶
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_11", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 450, 32]}}
í
Ï	variables
Ðtrainable_variables
Ñregularization_losses
Ò	keras_api
æ__call__
+ç&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_14", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
í
Ó	variables
Ôtrainable_variables
Õregularization_losses
Ö	keras_api
è__call__
+é&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_15", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}

u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
11
12
13
14
15"
trackable_list_wrapper

u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
H	variables
×metrics
Itrainable_variables
Ølayer_metrics
Jregularization_losses
Ùnon_trainable_variables
Úlayers
 Ûlayer_regularization_losses
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
L	variables
Ümetrics
Mtrainable_variables
Ýlayer_metrics
Nregularization_losses
Þnon_trainable_variables
ßlayers
 àlayer_regularization_losses
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
P	variables
ámetrics
Qtrainable_variables
âlayer_metrics
Rregularization_losses
ãnon_trainable_variables
älayers
 ålayer_regularization_losses
Í__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
!:( 2dense_18/kernel
: 2dense_18/bias
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
V	variables
æmetrics
Wtrainable_variables
çlayer_metrics
Xregularization_losses
ènon_trainable_variables
élayers
 êlayer_regularization_losses
Ï__call__
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
Z	variables
ëmetrics
[trainable_variables
ìlayer_metrics
\regularization_losses
ínon_trainable_variables
îlayers
 ïlayer_regularization_losses
Ñ__call__
+Ò&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses"
_generic_user_object
!:  2dense_19/kernel
: 2dense_19/bias
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
`	variables
ðmetrics
atrainable_variables
ñlayer_metrics
bregularization_losses
ònon_trainable_variables
ólayers
 ôlayer_regularization_losses
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
d	variables
õmetrics
etrainable_variables
ölayer_metrics
fregularization_losses
÷non_trainable_variables
ølayers
 ùlayer_regularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_20/kernel
:2dense_20/bias
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
j	variables
úmetrics
ktrainable_variables
ûlayer_metrics
lregularization_losses
ünon_trainable_variables
ýlayers
 þlayer_regularization_losses
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2Adamax/iter
G:E 25token_and_position_embedding_2/embedding_4/embeddings
H:F	¼i 25token_and_position_embedding_2/embedding_5/embeddings
M:K  27transformer_block_5/multi_head_attention_5/query/kernel
G:E 25transformer_block_5/multi_head_attention_5/query/bias
K:I  25transformer_block_5/multi_head_attention_5/key/kernel
E:C 23transformer_block_5/multi_head_attention_5/key/bias
M:K  27transformer_block_5/multi_head_attention_5/value/kernel
G:E 25transformer_block_5/multi_head_attention_5/value/bias
X:V  2Btransformer_block_5/multi_head_attention_5/attention_output/kernel
N:L 2@transformer_block_5/multi_head_attention_5/attention_output/bias
!:  2dense_16/kernel
: 2dense_16/bias
!:  2dense_17/kernel
: 2dense_17/bias
>:< 20transformer_block_5/layer_normalization_10/gamma
=:; 2/transformer_block_5/layer_normalization_10/beta
>:< 20transformer_block_5/layer_normalization_11/gamma
=:; 2/transformer_block_5/layer_normalization_11/beta
(
ÿ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
<
/0
01
82
93"
trackable_list_wrapper
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
16"
trackable_list_wrapper
 "
trackable_list_wrapper
'
s0"
trackable_list_wrapper
'
s0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
metrics
trainable_variables
layer_metrics
regularization_losses
non_trainable_variables
layers
 layer_regularization_losses
Ú__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
'
t0"
trackable_list_wrapper
'
t0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
metrics
trainable_variables
layer_metrics
regularization_losses
non_trainable_variables
layers
 layer_regularization_losses
Ü__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
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
/0
01"
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
80
91"
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
Ê
partial_output_shape
full_output_shape

ukernel
vbias
	variables
trainable_variables
regularization_losses
	keras_api
ê__call__
+ë&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "EinsumDense", "name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 450, 32]}}
Æ
partial_output_shape
full_output_shape

wkernel
xbias
	variables
trainable_variables
regularization_losses
	keras_api
ì__call__
+í&call_and_return_all_conditional_losses"è
_tf_keras_layerÎ{"class_name": "EinsumDense", "name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 450, 32]}}
Ê
partial_output_shape
full_output_shape

ykernel
zbias
	variables
trainable_variables
regularization_losses
	keras_api
î__call__
+ï&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "EinsumDense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 450, 32]}}
ë
	variables
trainable_variables
regularization_losses
	keras_api
ð__call__
+ñ&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}}
ç
 	variables
¡trainable_variables
¢regularization_losses
£	keras_api
ò__call__
+ó&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
ß
¤partial_output_shape
¥full_output_shape

{kernel
|bias
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
ô__call__
+õ&call_and_return_all_conditional_losses"
_tf_keras_layerç{"class_name": "EinsumDense", "name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 32], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 450, 1, 32]}}
X
u0
v1
w2
x3
y4
z5
{6
|7"
trackable_list_wrapper
X
u0
v1
w2
x3
y4
z5
{6
|7"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
»	variables
ªmetrics
¼trainable_variables
«layer_metrics
½regularization_losses
¬non_trainable_variables
­layers
 ®layer_regularization_losses
Þ__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses"
_generic_user_object
ý

}kernel
~bias
¯	variables
°trainable_variables
±regularization_losses
²	keras_api
ö__call__
+÷&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 450, 32]}}


kernel
	bias
³	variables
´trainable_variables
µregularization_losses
¶	keras_api
ø__call__
+ù&call_and_return_all_conditional_losses"Ô
_tf_keras_layerº{"class_name": "Dense", "name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 450, 32]}}
=
}0
~1
2
3"
trackable_list_wrapper
=
}0
~1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Á	variables
·metrics
Âtrainable_variables
¸layer_metrics
Ãregularization_losses
¹non_trainable_variables
ºlayers
 »layer_regularization_losses
à__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses"
_generic_user_object
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
Æ	variables
¼metrics
Çtrainable_variables
½layer_metrics
Èregularization_losses
¾non_trainable_variables
¿layers
 Àlayer_regularization_losses
â__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses"
_generic_user_object
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
Ë	variables
Ámetrics
Ìtrainable_variables
Âlayer_metrics
Íregularization_losses
Ãnon_trainable_variables
Älayers
 Ålayer_regularization_losses
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
¸
Ï	variables
Æmetrics
Ðtrainable_variables
Çlayer_metrics
Ñregularization_losses
Ènon_trainable_variables
Élayers
 Êlayer_regularization_losses
æ__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ó	variables
Ëmetrics
Ôtrainable_variables
Ìlayer_metrics
Õregularization_losses
Ínon_trainable_variables
Îlayers
 Ïlayer_regularization_losses
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
B0
C1
D2
E3
F4
G5"
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

Ðtotal

Ñcount
Ò	variables
Ó	keras_api"
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
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
Ômetrics
trainable_variables
Õlayer_metrics
regularization_losses
Önon_trainable_variables
×layers
 Ølayer_regularization_losses
ê__call__
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
Ùmetrics
trainable_variables
Úlayer_metrics
regularization_losses
Ûnon_trainable_variables
Ülayers
 Ýlayer_regularization_losses
ì__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
Þmetrics
trainable_variables
ßlayer_metrics
regularization_losses
ànon_trainable_variables
álayers
 âlayer_regularization_losses
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
ãmetrics
trainable_variables
älayer_metrics
regularization_losses
ånon_trainable_variables
ælayers
 çlayer_regularization_losses
ð__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 	variables
èmetrics
¡trainable_variables
élayer_metrics
¢regularization_losses
ênon_trainable_variables
ëlayers
 ìlayer_regularization_losses
ò__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¦	variables
ímetrics
§trainable_variables
îlayer_metrics
¨regularization_losses
ïnon_trainable_variables
ðlayers
 ñlayer_regularization_losses
ô__call__
+õ&call_and_return_all_conditional_losses
'õ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
P
µ0
¶1
·2
¸3
¹4
º5"
trackable_list_wrapper
 "
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¯	variables
òmetrics
°trainable_variables
ólayer_metrics
±regularization_losses
ônon_trainable_variables
õlayers
 ölayer_regularization_losses
ö__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
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
¸
³	variables
÷metrics
´trainable_variables
ølayer_metrics
µregularization_losses
ùnon_trainable_variables
úlayers
 ûlayer_regularization_losses
ø__call__
+ù&call_and_return_all_conditional_losses
'ù"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
¿0
À1"
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
Ð0
Ñ1"
trackable_list_wrapper
.
Ò	variables"
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
,:*  2Adamax/conv1d_2/kernel/m
":  2Adamax/conv1d_2/bias/m
0:. 2$Adamax/batch_normalization_4/gamma/m
/:- 2#Adamax/batch_normalization_4/beta/m
0:. 2$Adamax/batch_normalization_5/gamma/m
/:- 2#Adamax/batch_normalization_5/beta/m
(:&( 2Adamax/dense_18/kernel/m
":  2Adamax/dense_18/bias/m
(:&  2Adamax/dense_19/kernel/m
":  2Adamax/dense_19/bias/m
(:& 2Adamax/dense_20/kernel/m
": 2Adamax/dense_20/bias/m
N:L 2>Adamax/token_and_position_embedding_2/embedding_4/embeddings/m
O:M	¼i 2>Adamax/token_and_position_embedding_2/embedding_5/embeddings/m
T:R  2@Adamax/transformer_block_5/multi_head_attention_5/query/kernel/m
N:L 2>Adamax/transformer_block_5/multi_head_attention_5/query/bias/m
R:P  2>Adamax/transformer_block_5/multi_head_attention_5/key/kernel/m
L:J 2<Adamax/transformer_block_5/multi_head_attention_5/key/bias/m
T:R  2@Adamax/transformer_block_5/multi_head_attention_5/value/kernel/m
N:L 2>Adamax/transformer_block_5/multi_head_attention_5/value/bias/m
_:]  2KAdamax/transformer_block_5/multi_head_attention_5/attention_output/kernel/m
U:S 2IAdamax/transformer_block_5/multi_head_attention_5/attention_output/bias/m
(:&  2Adamax/dense_16/kernel/m
":  2Adamax/dense_16/bias/m
(:&  2Adamax/dense_17/kernel/m
":  2Adamax/dense_17/bias/m
E:C 29Adamax/transformer_block_5/layer_normalization_10/gamma/m
D:B 28Adamax/transformer_block_5/layer_normalization_10/beta/m
E:C 29Adamax/transformer_block_5/layer_normalization_11/gamma/m
D:B 28Adamax/transformer_block_5/layer_normalization_11/beta/m
,:*  2Adamax/conv1d_2/kernel/v
":  2Adamax/conv1d_2/bias/v
0:. 2$Adamax/batch_normalization_4/gamma/v
/:- 2#Adamax/batch_normalization_4/beta/v
0:. 2$Adamax/batch_normalization_5/gamma/v
/:- 2#Adamax/batch_normalization_5/beta/v
(:&( 2Adamax/dense_18/kernel/v
":  2Adamax/dense_18/bias/v
(:&  2Adamax/dense_19/kernel/v
":  2Adamax/dense_19/bias/v
(:& 2Adamax/dense_20/kernel/v
": 2Adamax/dense_20/bias/v
N:L 2>Adamax/token_and_position_embedding_2/embedding_4/embeddings/v
O:M	¼i 2>Adamax/token_and_position_embedding_2/embedding_5/embeddings/v
T:R  2@Adamax/transformer_block_5/multi_head_attention_5/query/kernel/v
N:L 2>Adamax/transformer_block_5/multi_head_attention_5/query/bias/v
R:P  2>Adamax/transformer_block_5/multi_head_attention_5/key/kernel/v
L:J 2<Adamax/transformer_block_5/multi_head_attention_5/key/bias/v
T:R  2@Adamax/transformer_block_5/multi_head_attention_5/value/kernel/v
N:L 2>Adamax/transformer_block_5/multi_head_attention_5/value/bias/v
_:]  2KAdamax/transformer_block_5/multi_head_attention_5/attention_output/kernel/v
U:S 2IAdamax/transformer_block_5/multi_head_attention_5/attention_output/bias/v
(:&  2Adamax/dense_16/kernel/v
":  2Adamax/dense_16/bias/v
(:&  2Adamax/dense_17/kernel/v
":  2Adamax/dense_17/bias/v
E:C 29Adamax/transformer_block_5/layer_normalization_10/gamma/v
D:B 28Adamax/transformer_block_5/layer_normalization_10/beta/v
E:C 29Adamax/transformer_block_5/layer_normalization_11/gamma/v
D:B 28Adamax/transformer_block_5/layer_normalization_11/beta/v
2
 __inference__wrapped_model_47506ß
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
input_5ÿÿÿÿÿÿÿÿÿ¼i
!
input_6ÿÿÿÿÿÿÿÿÿ
ê2ç
'__inference_model_2_layer_call_fn_49082
'__inference_model_2_layer_call_fn_49923
'__inference_model_2_layer_call_fn_49997
'__inference_model_2_layer_call_fn_49244À
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
Ö2Ó
B__inference_model_2_layer_call_and_return_conditional_losses_49849
B__inference_model_2_layer_call_and_return_conditional_losses_48831
B__inference_model_2_layer_call_and_return_conditional_losses_48919
B__inference_model_2_layer_call_and_return_conditional_losses_49622À
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
ã2à
>__inference_token_and_position_embedding_2_layer_call_fn_50030
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
þ2û
Y__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_50021
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
Ò2Ï
(__inference_conv1d_2_layer_call_fn_50055¢
²
FullArgSpec
args
jself
jinputs
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
í2ê
C__inference_conv1d_2_layer_call_and_return_conditional_losses_50046¢
²
FullArgSpec
args
jself
jinputs
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
2
3__inference_average_pooling1d_4_layer_call_fn_47521Ó
²
FullArgSpec
args
jself
jinputs
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
©2¦
N__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_47515Ó
²
FullArgSpec
args
jself
jinputs
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
2
3__inference_average_pooling1d_5_layer_call_fn_47536Ó
²
FullArgSpec
args
jself
jinputs
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
©2¦
N__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_47530Ó
²
FullArgSpec
args
jself
jinputs
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
2
5__inference_batch_normalization_4_layer_call_fn_50206
5__inference_batch_normalization_4_layer_call_fn_50124
5__inference_batch_normalization_4_layer_call_fn_50219
5__inference_batch_normalization_4_layer_call_fn_50137´
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
2ÿ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_50091
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_50173
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_50111
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_50193´
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
2
5__inference_batch_normalization_5_layer_call_fn_50301
5__inference_batch_normalization_5_layer_call_fn_50288
5__inference_batch_normalization_5_layer_call_fn_50370
5__inference_batch_normalization_5_layer_call_fn_50383´
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
2ÿ
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_50255
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_50275
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_50337
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_50357´
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
Ï2Ì
%__inference_add_2_layer_call_fn_50395¢
²
FullArgSpec
args
jself
jinputs
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
ê2ç
@__inference_add_2_layer_call_and_return_conditional_losses_50389¢
²
FullArgSpec
args
jself
jinputs
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
 2
3__inference_transformer_block_5_layer_call_fn_50744
3__inference_transformer_block_5_layer_call_fn_50707°
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
Ö2Ó
N__inference_transformer_block_5_layer_call_and_return_conditional_losses_50543
N__inference_transformer_block_5_layer_call_and_return_conditional_losses_50670°
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
­2ª
:__inference_global_average_pooling1d_2_layer_call_fn_50755
:__inference_global_average_pooling1d_2_layer_call_fn_50766¯
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
ã2à
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_50761
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_50750¯
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
×2Ô
-__inference_concatenate_2_layer_call_fn_50779¢
²
FullArgSpec
args
jself
jinputs
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
ò2ï
H__inference_concatenate_2_layer_call_and_return_conditional_losses_50773¢
²
FullArgSpec
args
jself
jinputs
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
Ò2Ï
(__inference_dense_18_layer_call_fn_50799¢
²
FullArgSpec
args
jself
jinputs
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
í2ê
C__inference_dense_18_layer_call_and_return_conditional_losses_50790¢
²
FullArgSpec
args
jself
jinputs
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
2
*__inference_dropout_16_layer_call_fn_50826
*__inference_dropout_16_layer_call_fn_50821´
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
È2Å
E__inference_dropout_16_layer_call_and_return_conditional_losses_50811
E__inference_dropout_16_layer_call_and_return_conditional_losses_50816´
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
Ò2Ï
(__inference_dense_19_layer_call_fn_50846¢
²
FullArgSpec
args
jself
jinputs
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
í2ê
C__inference_dense_19_layer_call_and_return_conditional_losses_50837¢
²
FullArgSpec
args
jself
jinputs
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
2
*__inference_dropout_17_layer_call_fn_50868
*__inference_dropout_17_layer_call_fn_50873´
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
È2Å
E__inference_dropout_17_layer_call_and_return_conditional_losses_50863
E__inference_dropout_17_layer_call_and_return_conditional_losses_50858´
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
Ò2Ï
(__inference_dense_20_layer_call_fn_50892¢
²
FullArgSpec
args
jself
jinputs
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
í2ê
C__inference_dense_20_layer_call_and_return_conditional_losses_50883¢
²
FullArgSpec
args
jself
jinputs
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
ÑBÎ
#__inference_signature_wrapper_49328input_5input_6"
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
þ2û
,__inference_sequential_5_layer_call_fn_51032
,__inference_sequential_5_layer_call_fn_47983
,__inference_sequential_5_layer_call_fn_51019
,__inference_sequential_5_layer_call_fn_47956À
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
ê2ç
G__inference_sequential_5_layer_call_and_return_conditional_losses_51006
G__inference_sequential_5_layer_call_and_return_conditional_losses_50949
G__inference_sequential_5_layer_call_and_return_conditional_losses_47914
G__inference_sequential_5_layer_call_and_return_conditional_losses_47928À
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
Ò2Ï
(__inference_dense_16_layer_call_fn_51072¢
²
FullArgSpec
args
jself
jinputs
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
í2ê
C__inference_dense_16_layer_call_and_return_conditional_losses_51063¢
²
FullArgSpec
args
jself
jinputs
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
Ò2Ï
(__inference_dense_17_layer_call_fn_51111¢
²
FullArgSpec
args
jself
jinputs
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
í2ê
C__inference_dense_17_layer_call_and_return_conditional_losses_51102¢
²
FullArgSpec
args
jself
jinputs
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
 Þ
 __inference__wrapped_model_47506¹'ts0-/.9687uvwxyz{|}~TU^_hiY¢V
O¢L
JG
"
input_5ÿÿÿÿÿÿÿÿÿ¼i
!
input_6ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_20"
dense_20ÿÿÿÿÿÿÿÿÿ×
@__inference_add_2_layer_call_and_return_conditional_losses_50389d¢a
Z¢W
UR
'$
inputs/0ÿÿÿÿÿÿÿÿÿÂ 
'$
inputs/1ÿÿÿÿÿÿÿÿÿÂ 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÂ 
 ¯
%__inference_add_2_layer_call_fn_50395d¢a
Z¢W
UR
'$
inputs/0ÿÿÿÿÿÿÿÿÿÂ 
'$
inputs/1ÿÿÿÿÿÿÿÿÿÂ 
ª "ÿÿÿÿÿÿÿÿÿÂ ×
N__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_47515E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ®
3__inference_average_pooling1d_4_layer_call_fn_47521wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×
N__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_47530E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ®
3__inference_average_pooling1d_5_layer_call_fn_47536wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_50091l/0-.8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÂ 
 À
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_50111l0-/.8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÂ 
 Ð
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_50173|/0-.@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ð
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_50193|0-/.@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
5__inference_batch_normalization_4_layer_call_fn_50124_/0-.8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
p
ª "ÿÿÿÿÿÿÿÿÿÂ 
5__inference_batch_normalization_4_layer_call_fn_50137_0-/.8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
p 
ª "ÿÿÿÿÿÿÿÿÿÂ ¨
5__inference_batch_normalization_4_layer_call_fn_50206o/0-.@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¨
5__inference_batch_normalization_4_layer_call_fn_50219o0-/.@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ð
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_50255|8967@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ð
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_50275|9687@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 À
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_50337l89678¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÂ 
 À
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_50357l96878¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÂ 
 ¨
5__inference_batch_normalization_5_layer_call_fn_50288o8967@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¨
5__inference_batch_normalization_5_layer_call_fn_50301o9687@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
5__inference_batch_normalization_5_layer_call_fn_50370_89678¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
p
ª "ÿÿÿÿÿÿÿÿÿÂ 
5__inference_batch_normalization_5_layer_call_fn_50383_96878¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
p 
ª "ÿÿÿÿÿÿÿÿÿÂ Ð
H__inference_concatenate_2_layer_call_and_return_conditional_losses_50773Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ 
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 §
-__inference_concatenate_2_layer_call_fn_50779vZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ 
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ(­
C__inference_conv1d_2_layer_call_and_return_conditional_losses_50046f4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¼i 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¼i 
 
(__inference_conv1d_2_layer_call_fn_50055Y4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¼i 
ª "ÿÿÿÿÿÿÿÿÿ¼i ­
C__inference_dense_16_layer_call_and_return_conditional_losses_51063f}~4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÂ 
 
(__inference_dense_16_layer_call_fn_51072Y}~4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
ª "ÿÿÿÿÿÿÿÿÿÂ ®
C__inference_dense_17_layer_call_and_return_conditional_losses_51102g4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÂ 
 
(__inference_dense_17_layer_call_fn_51111Z4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
ª "ÿÿÿÿÿÿÿÿÿÂ £
C__inference_dense_18_layer_call_and_return_conditional_losses_50790\TU/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ(
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 {
(__inference_dense_18_layer_call_fn_50799OTU/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ(
ª "ÿÿÿÿÿÿÿÿÿ £
C__inference_dense_19_layer_call_and_return_conditional_losses_50837\^_/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 {
(__inference_dense_19_layer_call_fn_50846O^_/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ £
C__inference_dense_20_layer_call_and_return_conditional_losses_50883\hi/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense_20_layer_call_fn_50892Ohi/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dropout_16_layer_call_and_return_conditional_losses_50811\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ¥
E__inference_dropout_16_layer_call_and_return_conditional_losses_50816\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 }
*__inference_dropout_16_layer_call_fn_50821O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "ÿÿÿÿÿÿÿÿÿ }
*__inference_dropout_16_layer_call_fn_50826O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "ÿÿÿÿÿÿÿÿÿ ¥
E__inference_dropout_17_layer_call_and_return_conditional_losses_50858\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ¥
E__inference_dropout_17_layer_call_and_return_conditional_losses_50863\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 }
*__inference_dropout_17_layer_call_fn_50868O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "ÿÿÿÿÿÿÿÿÿ }
*__inference_dropout_17_layer_call_fn_50873O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "ÿÿÿÿÿÿÿÿÿ º
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_50750a8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÂ 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 Ô
U__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_50761{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
:__inference_global_average_pooling1d_2_layer_call_fn_50755T8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÂ 

 
ª "ÿÿÿÿÿÿÿÿÿ ¬
:__inference_global_average_pooling1d_2_layer_call_fn_50766nI¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿú
B__inference_model_2_layer_call_and_return_conditional_losses_48831³'ts/0-.8967uvwxyz{|}~TU^_hia¢^
W¢T
JG
"
input_5ÿÿÿÿÿÿÿÿÿ¼i
!
input_6ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ú
B__inference_model_2_layer_call_and_return_conditional_losses_48919³'ts0-/.9687uvwxyz{|}~TU^_hia¢^
W¢T
JG
"
input_5ÿÿÿÿÿÿÿÿÿ¼i
!
input_6ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ü
B__inference_model_2_layer_call_and_return_conditional_losses_49622µ'ts/0-.8967uvwxyz{|}~TU^_hic¢`
Y¢V
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿ¼i
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ü
B__inference_model_2_layer_call_and_return_conditional_losses_49849µ'ts0-/.9687uvwxyz{|}~TU^_hic¢`
Y¢V
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿ¼i
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ò
'__inference_model_2_layer_call_fn_49082¦'ts/0-.8967uvwxyz{|}~TU^_hia¢^
W¢T
JG
"
input_5ÿÿÿÿÿÿÿÿÿ¼i
!
input_6ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÒ
'__inference_model_2_layer_call_fn_49244¦'ts0-/.9687uvwxyz{|}~TU^_hia¢^
W¢T
JG
"
input_5ÿÿÿÿÿÿÿÿÿ¼i
!
input_6ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÔ
'__inference_model_2_layer_call_fn_49923¨'ts/0-.8967uvwxyz{|}~TU^_hic¢`
Y¢V
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿ¼i
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÔ
'__inference_model_2_layer_call_fn_49997¨'ts0-/.9687uvwxyz{|}~TU^_hic¢`
Y¢V
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿ¼i
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÄ
G__inference_sequential_5_layer_call_and_return_conditional_losses_47914y}~D¢A
:¢7
-*
dense_16_inputÿÿÿÿÿÿÿÿÿÂ 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÂ 
 Ä
G__inference_sequential_5_layer_call_and_return_conditional_losses_47928y}~D¢A
:¢7
-*
dense_16_inputÿÿÿÿÿÿÿÿÿÂ 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÂ 
 ¼
G__inference_sequential_5_layer_call_and_return_conditional_losses_50949q}~<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÂ 
 ¼
G__inference_sequential_5_layer_call_and_return_conditional_losses_51006q}~<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÂ 
 
,__inference_sequential_5_layer_call_fn_47956l}~D¢A
:¢7
-*
dense_16_inputÿÿÿÿÿÿÿÿÿÂ 
p

 
ª "ÿÿÿÿÿÿÿÿÿÂ 
,__inference_sequential_5_layer_call_fn_47983l}~D¢A
:¢7
-*
dense_16_inputÿÿÿÿÿÿÿÿÿÂ 
p 

 
ª "ÿÿÿÿÿÿÿÿÿÂ 
,__inference_sequential_5_layer_call_fn_51019d}~<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
p

 
ª "ÿÿÿÿÿÿÿÿÿÂ 
,__inference_sequential_5_layer_call_fn_51032d}~<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
p 

 
ª "ÿÿÿÿÿÿÿÿÿÂ ò
#__inference_signature_wrapper_49328Ê'ts0-/.9687uvwxyz{|}~TU^_hij¢g
¢ 
`ª]
-
input_5"
input_5ÿÿÿÿÿÿÿÿÿ¼i
,
input_6!
input_6ÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_20"
dense_20ÿÿÿÿÿÿÿÿÿº
Y__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_50021]ts+¢(
!¢

xÿÿÿÿÿÿÿÿÿ¼i
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¼i 
 
>__inference_token_and_position_embedding_2_layer_call_fn_50030Pts+¢(
!¢

xÿÿÿÿÿÿÿÿÿ¼i
ª "ÿÿÿÿÿÿÿÿÿ¼i Ï
N__inference_transformer_block_5_layer_call_and_return_conditional_losses_50543}uvwxyz{|}~8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÂ 
 Ï
N__inference_transformer_block_5_layer_call_and_return_conditional_losses_50670}uvwxyz{|}~8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÂ 
 §
3__inference_transformer_block_5_layer_call_fn_50707puvwxyz{|}~8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
p
ª "ÿÿÿÿÿÿÿÿÿÂ §
3__inference_transformer_block_5_layer_call_fn_50744puvwxyz{|}~8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÂ 
p 
ª "ÿÿÿÿÿÿÿÿÿÂ 